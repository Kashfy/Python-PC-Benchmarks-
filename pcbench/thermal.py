"""Hardware temperature and fan readings, in degrees Celsius.

Temperature is the single most useful diagnostic a benchmark can report
alongside throughput: it explains *why* a machine throttled, distinguishes a
slow chip from an overheating one, and turns a sustained-load run into a real
cooling assessment.

Every platform hides it somewhere different:

* **macOS** publishes no public API. ``pmset`` gives only a throttle
  percentage and ``powermetrics`` needs root, so readings come from the
  ``sensors_engine`` helper, which uses the unprivileged IOHID thermal usage
  page.
* **Linux** exposes ``/sys/class/thermal`` and ``/sys/class/hwmon`` — no
  privileges required, and hwmon also carries fan tachometers.
* **Windows** offers WMI ``MSAcpi_ThermalZoneTemperature``, which many vendors
  simply do not implement; absence is reported rather than guessed at.

Every reading is Celsius. Nothing here fabricates a value: when no sensor is
readable the result is ``None`` and the report says temperature is unavailable.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess

SOURCE_NAME = "sensors_engine.m"
BINARY_NAME = "sensors_engine"

# Above this a consumer CPU is at or near its thermal limit.
HOT_CELSIUS = 90.0
WARM_CELSIUS = 75.0


def _run(cmd: list[str], timeout: int = 6) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #
def _build_sensors_engine(script_dir: str) -> str | None:
    """Compile the IOHID helper if missing or stale. Returns its path."""
    src = os.path.join(script_dir, SOURCE_NAME)
    if not os.path.isfile(src):
        return None
    exe = os.path.join(script_dir, BINARY_NAME)
    if (os.path.isfile(exe)
            and os.path.getmtime(exe) >= os.path.getmtime(src)):
        return exe
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return None
    proc = subprocess.run(
        [cc, "-O2", "-fobjc-arc", src, "-o", exe,
         "-framework", "Foundation", "-framework", "IOKit"],
        capture_output=True, text=True)
    return exe if proc.returncode == 0 else None


def _macos(script_dir: str) -> dict:
    exe = _build_sensors_engine(script_dir)
    if not exe:
        return {}
    out = _run([exe, "--json"])
    if not out:
        return {}
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {}
    if "error" in data:
        return {}
    result = {}
    if "cpu_celsius" in data:
        result["cpu_celsius"] = round(float(data["cpu_celsius"]), 1)
        result["cpu_avg_celsius"] = round(float(data["cpu_avg_celsius"]), 1)
        result["sensor_count"] = data.get("sensor_count")
    if "battery_celsius" in data:
        result["battery_celsius"] = round(float(data["battery_celsius"]), 1)
    if result:
        result["source"] = "IOHID thermal sensors"
    return result


# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #
# hwmon chip names that report the CPU package, most specific first.
_CPU_HWMON = ("coretemp", "k10temp", "zenpower", "cpu_thermal",
              "soc_thermal", "cpu-thermal")


def _linux() -> dict:
    temps: list[tuple[str, float]] = []

    # hwmon carries labelled sensors and is the most reliable source.
    for chip_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        chip = _read(os.path.join(chip_dir, "name"))
        for temp_file in sorted(glob.glob(os.path.join(chip_dir,
                                                       "temp*_input"))):
            raw = _read(temp_file)
            if not raw.lstrip("-").isdigit():
                continue
            celsius = int(raw) / 1000.0
            if not (-50 < celsius < 150):
                continue
            label = _read(temp_file.replace("_input", "_label")) or chip
            temps.append((f"{chip}/{label}" if chip else label, celsius))

    # Thermal zones are a coarser fallback, present even in minimal systems.
    if not temps:
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            raw = _read(os.path.join(zone, "temp"))
            if raw.lstrip("-").isdigit():
                celsius = int(raw) / 1000.0
                if -50 < celsius < 150:
                    temps.append((_read(os.path.join(zone, "type")) or "zone",
                                  celsius))
    if not temps:
        return {}

    # Prefer a recognised CPU package sensor over, say, a disk or chipset one.
    cpu = [t for name, t in temps
           if any(k in name.lower() for k in _CPU_HWMON)
           or "package" in name.lower() or "tctl" in name.lower()]
    pool = cpu or [t for _, t in temps]

    result = {
        "cpu_celsius": round(max(pool), 1),
        "cpu_avg_celsius": round(sum(pool) / len(pool), 1),
        "sensor_count": len(pool),
        "source": "sysfs hwmon/thermal",
    }

    fans = []
    for fan_file in sorted(glob.glob("/sys/class/hwmon/hwmon*/fan*_input")):
        raw = _read(fan_file)
        if raw.isdigit() and int(raw) > 0:
            fans.append(int(raw))
    if fans:
        result["fan_rpm"] = fans

    for name, t in temps:
        if "nvme" in name.lower() or "drivetemp" in name.lower():
            result["drive_celsius"] = round(t, 1)
            break
    return result


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _windows() -> dict:
    # Reported in tenths of a Kelvin. Many OEMs never populate this class, so
    # an empty result is normal rather than an error.
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance -Namespace root/WMI "
                "-ClassName MSAcpi_ThermalZoneTemperature "
                "-ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty CurrentTemperature"])
    values = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            celsius = int(line) / 10.0 - 273.15
            if -50 < celsius < 150:
                values.append(celsius)
    if not values:
        return {}
    return {
        "cpu_celsius": round(max(values), 1),
        "cpu_avg_celsius": round(sum(values) / len(values), 1),
        "sensor_count": len(values),
        "source": "WMI MSAcpi_ThermalZoneTemperature",
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def read(script_dir: str = ".") -> dict:
    """Current temperatures in Celsius. Empty dict when nothing is readable."""
    try:
        system = platform.system()
        if system == "Darwin":
            return _macos(script_dir)
        if system == "Linux":
            return _linux()
        if system == "Windows":
            return _windows()
    except Exception:
        pass
    return {}


def cpu_celsius(script_dir: str = ".") -> float | None:
    return read(script_dir).get("cpu_celsius")


def describe(temps: dict) -> str:
    """One-line summary for the report, e.g. ``51.8 C (warm)``."""
    t = temps.get("cpu_celsius")
    if t is None:
        return "unavailable"
    if t >= HOT_CELSIUS:
        state = "hot"
    elif t >= WARM_CELSIUS:
        state = "warm"
    else:
        state = "normal"
    text = f"{t:.1f} °C ({state})"
    if temps.get("sensor_count"):
        text += f", max of {temps['sensor_count']} sensors"
    return text


# --------------------------------------------------------------------------- #
# Battery health — a genuine laptop diagnostic
# --------------------------------------------------------------------------- #
def battery_health() -> dict:
    """Charge cycles and remaining capacity versus the design capacity."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = _run(["ioreg", "-r", "-c", "AppleSmartBattery"])
            def grab(key):
                m = re.search(rf'"{key}"\s*=\s*(-?\d+)', out)
                return int(m.group(1)) if m else None
            design = grab("DesignCapacity")
            current = grab("AppleRawMaxCapacity") or grab("MaxCapacity")
            cycles = grab("CycleCount")
            temp = grab("Temperature")
            info = {}
            if cycles is not None:
                info["cycle_count"] = cycles
            if design and current and design > 0:
                # AppleRawMaxCapacity is in mAh like DesignCapacity; the
                # "MaxCapacity" key is a percentage on some models, so only
                # trust the ratio when it lands in a plausible range.
                pct = current / design * 100.0
                if 1 <= pct <= 120:
                    info["health_percent"] = round(pct, 1)
                    info["design_capacity_mah"] = design
                    info["current_capacity_mah"] = current
            if temp is not None and 0 < temp < 10000:
                info["celsius"] = round(temp / 100.0, 1)
            return info

        if system == "Linux":
            for base in sorted(glob.glob("/sys/class/power_supply/BAT*")):
                info = {}
                cycles = _read(os.path.join(base, "cycle_count"))
                if cycles.isdigit() and int(cycles) > 0:
                    info["cycle_count"] = int(cycles)
                full = _read(os.path.join(base, "energy_full")) or \
                    _read(os.path.join(base, "charge_full"))
                design = _read(os.path.join(base, "energy_full_design")) or \
                    _read(os.path.join(base, "charge_full_design"))
                if full.isdigit() and design.isdigit() and int(design) > 0:
                    info["health_percent"] = round(
                        int(full) / int(design) * 100.0, 1)
                temp = _read(os.path.join(base, "temp"))
                if temp.lstrip("-").isdigit():
                    info["celsius"] = round(int(temp) / 10.0, 1)
                if info:
                    return info
            return {}

        if system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance -Namespace root/WMI "
                        "-ClassName BatteryFullChargedCapacity "
                        "-ErrorAction SilentlyContinue)"
                        ".FullChargedCapacity"])
            design = _run(["powershell", "-NoProfile", "-Command",
                           "(Get-CimInstance -Namespace root/WMI "
                           "-ClassName BatteryStaticData "
                           "-ErrorAction SilentlyContinue).DesignedCapacity"])
            if out.strip().isdigit() and design.strip().isdigit() \
                    and int(design) > 0:
                return {"health_percent": round(
                    int(out) / int(design) * 100.0, 1)}
            return {}
    except Exception:
        pass
    return {}
