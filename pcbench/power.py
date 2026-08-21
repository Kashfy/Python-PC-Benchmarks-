"""Power draw and performance-per-watt.

Perf-per-watt is increasingly the metric that matters — two chips can post the
same throughput while one draws three times the power, and for AI and laptop
workloads that efficiency gap is often the whole story.

Measuring real power is privileged on most systems, so this module is honest
about its source and confidence:

* **measured**  — Apple ``powermetrics`` (needs sudo) or Linux RAPL
  (``/sys/class/powercap``).
* **estimated** — a rough package-TDP lookup when no meter is accessible.

The source is always reported, so an estimate is never mistaken for a
measurement.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time

# Rough package power (watts) under load, by chip family. Deliberately
# approximate — only a fallback when nothing can be measured.
_TDP_ESTIMATES = {
    r"apple m1\b": 15, r"apple m1 pro": 30, r"apple m1 max": 40,
    r"apple m1 ultra": 60,
    r"apple m2\b": 15, r"apple m2 pro": 30, r"apple m2 max": 45,
    r"apple m3\b": 15, r"apple m3 pro": 30, r"apple m3 max": 45,
    r"apple m4\b": 18, r"apple m4 pro": 35, r"apple m4 max": 50,
    r"core i9": 65, r"core i7": 45, r"core i5": 35, r"core i3": 25,
    r"ryzen 9": 65, r"ryzen 7": 45, r"ryzen 5": 35,
    r"xeon": 150, r"epyc": 200, r"threadripper": 280,
}


def _run(cmd: list[str], timeout: int = 6) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def estimate_tdp(cpu_model: str) -> int | None:
    model = (cpu_model or "").lower()
    for pattern, watts in _TDP_ESTIMATES.items():
        if re.search(pattern, model):
            return watts
    return None


# --------------------------------------------------------------------------- #
# macOS — powermetrics (root only)
# --------------------------------------------------------------------------- #
def _measure_macos(sample_ms: int = 500) -> dict | None:
    # -n1 one sample; powermetrics refuses to run without root.
    rc, out = _run(["sudo", "-n", "powermetrics",
                    "-n", "1", "-i", str(sample_ms),
                    "--samplers", "cpu_power,gpu_power"], timeout=8)
    if rc != 0 or not out:
        return None
    watts = {}
    for label, key in (("CPU Power", "cpu_w"), ("GPU Power", "gpu_w"),
                       ("Combined Power", "package_w"),
                       ("Package Power", "package_w"), ("ANE Power", "ane_w")):
        m = re.search(rf"{label}:\s*([\d.]+)\s*mW", out)
        if m:
            watts[key] = float(m.group(1)) / 1000.0
    if not watts:
        return None
    if "package_w" not in watts:
        watts["package_w"] = watts.get("cpu_w", 0) + watts.get("gpu_w", 0)
    watts["source"] = "measured (powermetrics)"
    return watts


# --------------------------------------------------------------------------- #
# Linux — RAPL energy counters
# --------------------------------------------------------------------------- #
_RAPL = "/sys/class/powercap"


def _rapl_energy_uj() -> int | None:
    total = 0
    found = False
    try:
        for name in os.listdir(_RAPL):
            if name.startswith("intel-rapl:") and name.count(":") == 1:
                path = os.path.join(_RAPL, name, "energy_uj")
                try:
                    with open(path) as f:
                        total += int(f.read().strip())
                        found = True
                except OSError:
                    pass
    except OSError:
        return None
    return total if found else None


def _measure_linux(window_s: float = 0.5) -> dict | None:
    e0 = _rapl_energy_uj()
    if e0 is None:
        return None
    time.sleep(window_s)
    e1 = _rapl_energy_uj()
    if e1 is None or e1 < e0:            # counter wrapped; skip this sample
        return None
    watts = (e1 - e0) / 1e6 / window_s
    return {"package_w": round(watts, 2), "source": "measured (RAPL)"}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def measure(cpu_model: str = "") -> dict:
    """Return a power reading with its source and confidence.

    Called while a load is running so the number reflects active draw, not
    idle. Falls back to a TDP estimate, clearly labelled.
    """
    sysname = platform.system()
    reading = None
    if sysname == "Darwin":
        reading = _measure_macos()
    elif sysname == "Linux":
        reading = _measure_linux()

    if reading:
        reading["estimated"] = False
        return reading

    tdp = estimate_tdp(cpu_model)
    if tdp:
        return {"package_w": float(tdp), "estimated": True,
                "source": f"estimated from ~{tdp}W TDP for this chip class",
                "hint": _hint(sysname)}
    return {"package_w": None, "estimated": True,
            "source": "unavailable", "hint": _hint(sysname)}


def _hint(sysname: str) -> str:
    if sysname == "Darwin":
        return ("run with sudo for a real reading: "
                "sudo python3 benchmark.py")
    if sysname == "Linux":
        return "real readings need Intel/AMD RAPL (/sys/class/powercap)"
    return "on-die power metering is not exposed on this platform"


def measure_under_load(cpu_model: str = "", load_s: float = 1.5) -> dict:
    """Sample power while the CPU is deliberately busy.

    Idle power is uninteresting; the number that matters is draw under work. A
    background thread burns all cores for the sampling window. If only a TDP
    estimate is available the load is skipped (the estimate is load-independent).
    """
    if platform.system() not in ("Darwin", "Linux"):
        return measure(cpu_model)

    import threading

    stop = threading.Event()

    def burn():
        x = 0
        while not stop.is_set():
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF

    workers = [threading.Thread(target=burn, daemon=True)
               for _ in range(max(1, os.cpu_count() or 1))]
    for w in workers:
        w.start()
    try:
        time.sleep(0.2)                 # let the load ramp
        reading = measure(cpu_model)
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=0.5)
    return reading


def perf_per_watt(composite_score: float, power: dict) -> dict | None:
    """Score per watt — the efficiency figure. None if power is unknown."""
    watts = power.get("package_w")
    if not watts or not composite_score:
        return None
    return {
        "score_per_watt": round(composite_score / watts, 2),
        "watts": watts,
        "estimated": power.get("estimated", True),
    }
