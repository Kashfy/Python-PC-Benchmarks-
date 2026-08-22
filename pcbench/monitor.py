"""Live telemetry: watch the machine instead of benchmarking it.

Half of what people want from a "benchmark tool" is not a score at all. It is
an answer to "what is my computer doing right now, and why is it slow?" — and
that question is unanswerable from a benchmark, because by the time the run
finishes the moment has passed.

Monitor mode samples the things that explain performance over a window of
time and prints them as they happen:

* **Clock speed** collapsing under load is throttling; collapsing at idle is a
  power-management setting.
* **Temperature** rising toward the limit before the clocks drop identifies
  *thermal* throttling specifically, as opposed to a power or current limit,
  which drops clocks with the chip still cool.
* **Load average** far above the core count means the slowness is contention
  from other software, not the hardware.
* **Power draw** pinned at the package limit while temperature is moderate is
  the signature of a power cap — common on laptops in a low-power mode, and
  fixable in settings rather than with a heatsink.

Samples are retained so the session ends with a summary and, on request, a
JSON or CSV trace to attach to a support ticket.
"""

from __future__ import annotations

import os
import platform
import time

from . import power as power_mod
from . import system as system_mod
from . import thermal as thermal_mod
from .sustained import sparkline


def sample(script_dir: str = ".", cpu_model: str = "") -> dict:
    """One telemetry snapshot. Every field is optional by design.

    Nothing here raises: a machine with no readable sensors must still produce
    a usable monitor session showing the fields it does have.
    """
    snap: dict = {"t": time.time()}
    try:
        snap["cpu_mhz"] = system_mod.cpu_frequency_mhz()
    except Exception:
        snap["cpu_mhz"] = None
    try:
        temps = thermal_mod.read(script_dir)
        snap["cpu_celsius"] = temps.get("cpu_celsius")
        snap["gpu_celsius"] = temps.get("gpu_celsius")
        snap["fan_rpm"] = temps.get("fan_rpm")
    except Exception:
        pass
    try:
        load = system_mod.load_average()
        snap["load1"] = load[0] if load else None
    except Exception:
        snap["load1"] = None
    try:
        snap["thermal_pressure"] = system_mod.thermal_pressure()
    except Exception:
        pass
    try:
        snap["on_ac"] = system_mod.on_ac_power()
    except Exception:
        pass
    snap.update(_utilisation())
    return snap


def _utilisation() -> dict:
    """CPU and memory utilisation, via psutil when present, else the OS.

    psutil is optional everywhere in this tool, so there is a stdlib fallback:
    ``/proc/stat`` deltas on Linux and ``os.getloadavg`` elsewhere. A monitor
    that only works with a third-party package installed is not much of a
    monitor.
    """
    out: dict = {}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["mem_pct"] = vm.percent
        out["mem_available_bytes"] = vm.available
        swap = psutil.swap_memory()
        out["swap_pct"] = swap.percent
        # Swap *activity* is what hurts, not swap existence: a machine with a
        # full-but-idle swap file is fine, one paging continuously is not.
        out["swap_in_bytes"] = swap.sin
        out["swap_out_bytes"] = swap.sout
        return out
    except Exception:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                fields = {}
                for line in f:
                    key, _, rest = line.partition(":")
                    fields[key] = int(rest.split()[0]) * 1024
            total = fields.get("MemTotal", 0)
            available = fields.get("MemAvailable", 0)
            if total:
                out["mem_pct"] = round(100.0 * (total - available) / total, 1)
                out["mem_available_bytes"] = available
        except (OSError, ValueError, IndexError):
            pass
    return out


def _fmt(value, spec: str = ".0f", dash: str = "  --") -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return dash
    return format(value, spec)


def run(duration: float, interval: float = 1.0, script_dir: str = ".",
        cpu_model: str = "", quiet: bool = False,
        with_power: bool = False) -> dict:
    """Sample for ``duration`` seconds, printing a line per sample.

    Power sampling is opt-in because reading it costs a privileged subprocess
    on macOS every interval, which is itself a load on the machine being
    observed.
    """
    interval = max(0.2, float(interval))
    samples: list[dict] = []
    # psutil's first cpu_percent() call has no previous reading to diff
    # against and always returns 0.0. Priming it here keeps a spurious zero
    # out of the first row and out of the session minimum.
    try:
        import psutil
        psutil.cpu_percent(interval=None)
    except Exception:
        pass
    end = time.time() + max(interval, float(duration))

    if not quiet:
        header = (f"  {'time':>6}  {'MHz':>6}  {'°C':>5}  {'CPU%':>5}  "
                  f"{'mem%':>5}  {'load':>5}")
        if with_power:
            header += f"  {'watts':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))

    start = time.time()
    while time.time() < end:
        snap = sample(script_dir, cpu_model)
        if with_power:
            try:
                p = power_mod.measure(cpu_model)
                snap["watts"] = p.get("watts")
                snap["power_source"] = p.get("source")
            except Exception:
                snap["watts"] = None
        samples.append(snap)

        if not quiet:
            line = (f"  {snap['t'] - start:6.1f}  "
                    f"{_fmt(snap.get('cpu_mhz')):>6}  "
                    f"{_fmt(snap.get('cpu_celsius'), '.1f'):>5}  "
                    f"{_fmt(snap.get('cpu_pct'), '.0f'):>5}  "
                    f"{_fmt(snap.get('mem_pct'), '.0f'):>5}  "
                    f"{_fmt(snap.get('load1'), '.2f'):>5}")
            if with_power:
                line += f"  {_fmt(snap.get('watts'), '.1f'):>6}"
            pressure = snap.get("thermal_pressure")
            if pressure and pressure.lower() not in ("nominal", "normal"):
                line += f"   thermal pressure: {pressure}"
            print(line, flush=True)

        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    return summarize(samples, os.cpu_count() or 1)


def _series(samples: list[dict], key: str) -> list[float]:
    return [s[key] for s in samples
            if isinstance(s.get(key), (int, float))
            and not isinstance(s.get(key), bool)]


def summarize(samples: list[dict], cores: int = 1) -> dict:
    """Reduce a monitor session to a summary plus interpretation."""
    out: dict = {"samples": len(samples), "series": {}}
    for key in ("cpu_mhz", "cpu_celsius", "gpu_celsius", "cpu_pct", "mem_pct",
                "load1", "watts", "fan_rpm"):
        values = _series(samples, key)
        if not values:
            continue
        out["series"][key] = {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "mean": round(sum(values) / len(values), 2),
            "last": round(values[-1], 2),
            "spark": sparkline(values),
        }
    out["observations"] = observations(out["series"], samples, cores)
    out["raw"] = samples
    return out


def observations(series: dict, samples: list[dict], cores: int) -> list[str]:
    """Plain-language readings of what the numbers mean.

    Each observation names the *cause* it points to, because a number alone
    ("clock dropped 40%") does not tell anyone what to change.
    """
    notes: list[str] = []

    mhz = series.get("cpu_mhz")
    temp = series.get("cpu_celsius")
    if mhz and mhz["max"] > 0:
        drop = 100.0 * (mhz["max"] - mhz["min"]) / mhz["max"]
        if drop >= 25:
            if temp and temp["max"] >= 90:
                notes.append(
                    f"clock speed varied by {drop:.0f}% while peaking at "
                    f"{temp['max']:.0f} °C — thermal throttling; check "
                    f"airflow, dust, and thermal paste")
            elif temp and temp["max"] < 75:
                notes.append(
                    f"clock speed varied by {drop:.0f}% while staying below "
                    f"{temp['max']:.0f} °C — a power or current limit rather "
                    f"than heat; check the OS power profile")
            else:
                notes.append(f"clock speed varied by {drop:.0f}% during the "
                             f"session")

    load = series.get("load1")
    if load and load["max"] > cores * 1.5:
        notes.append(
            f"load average peaked at {load['max']:.1f} on {cores} cores — the "
            f"machine is oversubscribed; other software is competing for CPU")

    mem = series.get("mem_pct")
    if mem and mem["max"] >= 90:
        notes.append(
            f"memory reached {mem['max']:.0f}% — the system is close to "
            f"paging, which slows everything regardless of CPU speed")

    swap_in = [s.get("swap_in_bytes") for s in samples
               if isinstance(s.get("swap_in_bytes"), (int, float))]
    if len(swap_in) >= 2 and swap_in[-1] - swap_in[0] > 64 * 1024 * 1024:
        notes.append(
            f"{(swap_in[-1] - swap_in[0]) / (1024 ** 2):.0f} MB was paged in "
            f"from swap during the session — add RAM or close applications")

    pressures = {s.get("thermal_pressure") for s in samples
                 if s.get("thermal_pressure")}
    hot = {p for p in pressures if p.lower() not in ("nominal", "normal")}
    if hot:
        notes.append(f"OS reported thermal pressure: {', '.join(sorted(hot))}")

    if any(s.get("on_ac") is False for s in samples):
        notes.append("machine was on battery for part of the session — most "
                     "laptops cap performance when unplugged")

    if not notes:
        notes.append("nothing anomalous: clocks, temperature, and load stayed "
                     "within normal ranges for the session")
    return notes


def render(result: dict) -> str:
    """Terminal summary block for a completed monitor session."""
    lines = []
    labels = {"cpu_mhz": "CPU clock (MHz)", "cpu_celsius": "CPU temp (°C)",
              "gpu_celsius": "GPU temp (°C)", "cpu_pct": "CPU busy (%)",
              "mem_pct": "Memory used (%)", "load1": "Load average",
              "watts": "Power (W)", "fan_rpm": "Fan (RPM)"}
    for key, stats in result.get("series", {}).items():
        lines.append(f"  {labels.get(key, key):<18} "
                     f"min {stats['min']:>7g}  mean {stats['mean']:>7g}  "
                     f"max {stats['max']:>7g}  {stats['spark']}")
    if result.get("observations"):
        lines.append("")
        for note in result["observations"]:
            lines.append(f"  - {note}")
    return "\n".join(lines)


def save_trace(result: dict, path: str) -> str:
    """Write the raw samples as CSV, for attaching to a bug report."""
    import csv

    rows = result.get("raw") or []
    if not rows:
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
