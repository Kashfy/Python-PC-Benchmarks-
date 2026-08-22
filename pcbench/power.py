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


def _burn(stop_flag, ready_flag=None) -> None:
    """Saturate one core until told to stop. Must be importable for spawn().

    Signals ``ready_flag`` once it is actually spinning. ``Process.start()``
    returns as soon as the child is forked or spawned, which on macOS is
    several hundred milliseconds before the child has finished importing
    Python and begun work — so a parent that starts sampling immediately
    measures the ramp rather than the load.
    """
    x = 0
    if ready_flag is not None:
        ready_flag.set()
    while not stop_flag.is_set():
        for _ in range(10_000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def measure_under_load(cpu_model: str = "", load_s: float = 1.5) -> dict:
    """Sample power while every core is deliberately busy.

    Idle power is uninteresting; the number that matters is draw under work.

    The load runs in **processes, not threads**. An earlier version used
    threads, and under CPython's GIL ten of them saturate exactly one core —
    so on a 10-core machine the "all-core" reading was single-core power. On an
    M1 Max that reported 6.9 W where the real all-core figure is several times
    higher, and every perf-per-watt figure derived from it was correspondingly
    inflated. The bug was invisible wherever power was only a TDP estimate,
    because the estimate ignores load entirely.

    If only a TDP estimate is available the load is skipped, since the estimate
    is load-independent and spawning processes would cost time for nothing.
    """
    if platform.system() not in ("Darwin", "Linux"):
        return measure(cpu_model)

    # No point loading the machine if nothing can actually meter it.
    probe = measure(cpu_model)
    if probe.get("estimated"):
        return probe

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    count = max(1, os.cpu_count() or 1)
    ready = [ctx.Event() for _ in range(count)]
    workers = [ctx.Process(target=_burn, args=(stop, ready[i]), daemon=True)
               for i in range(count)]
    reading = probe
    try:
        for w in workers:
            w.start()
        # Wait for the load to actually exist rather than assuming it does.
        # A fixed sleep would sample partway up the ramp on machines where
        # spawn is slow, understating draw in exactly the way the threaded
        # version did.
        deadline = time.monotonic() + 15.0
        for event in ready:
            event.wait(timeout=max(0.0, deadline - time.monotonic()))
        started = sum(1 for e in ready if e.is_set())

        time.sleep(max(0.5, min(load_s, 3.0)))
        reading = measure(cpu_model)
        reading["load"] = f"{started}/{count} process(es) at 100%"
        if started < count:
            reading["load_warning"] = (
                f"only {started} of {count} load processes started in time; "
                f"the reading may understate full-load draw")
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=2.0)
            if w.is_alive():
                w.terminate()
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


# --------------------------------------------------------------------------- #
# Energy to solution
# --------------------------------------------------------------------------- #
# Watts is a rate and answers "how hot will this get?". It does not answer the
# question that decides a datacenter bill or a battery's life: "how much energy
# did finishing this job actually cost?"
#
# Those come apart constantly. A chip that draws twice the power but finishes in
# a third of the time uses less total energy — the "race to idle" effect that
# governs mobile battery life. Comparing two machines on watts alone gets that
# backwards, and comparing them on speed alone ignores the bill.
#
# So this measures joules for a *fixed amount of work*, which is the only
# formulation that lets machines of different speeds be compared honestly.
#
# Measurement method differs by platform, and the difference matters enough to
# report: Linux RAPL exposes a cumulative energy counter, so subtracting two
# readings gives exact joules with no sampling error at all. macOS exposes only
# instantaneous power, so the figure there is an integral of samples and
# inherits their noise. A TDP estimate is a last resort and is labelled as one.
# --------------------------------------------------------------------------- #


def energy_to_solution(work, cpu_model: str = "", label: str = "workload",
                       sample_interval: float = 0.5) -> dict:
    """Run ``work()`` and report the energy it consumed.

    ``work`` must return the number of work units completed (iterations,
    samples, tokens — whatever makes sense), so efficiency can be expressed as
    work per joule. Returning None just omits that ratio.
    """
    sysname = platform.system()

    start_uj = _rapl_energy_uj() if sysname == "Linux" else None
    samples: list[float] = []
    stop = None
    sampler = None

    if start_uj is None:
        # No cumulative counter: sample instantaneous power on a background
        # thread and integrate. Started before the work so the first sample is
        # not paid for out of the measured interval.
        import threading

        stop = threading.Event()

        # measure() falls back to a TDP estimate when no meter is readable, so
        # whether the samples are real has to be carried through. Integrating
        # estimates and calling the total "measured" would be the exact kind of
        # false precision this module exists to avoid.
        sampled_estimates = []

        def poll() -> None:
            while not stop.is_set():
                reading = measure(cpu_model)
                watts = reading.get("package_w")
                if isinstance(watts, (int, float)):
                    samples.append(float(watts))
                    sampled_estimates.append(bool(reading.get("estimated")))
                stop.wait(sample_interval)

        sampler = threading.Thread(target=poll, daemon=True)
        sampler.start()

    start = time.perf_counter()
    units = work()
    elapsed = time.perf_counter() - start

    if sampler is not None:
        stop.set()
        sampler.join(timeout=5.0)

    end_uj = _rapl_energy_uj() if sysname == "Linux" else None

    joules = None
    method = None
    estimated = True
    samples_were_estimated = bool(
        locals().get("sampled_estimates") and all(sampled_estimates))
    if start_uj is not None and end_uj is not None and end_uj >= start_uj:
        joules = (end_uj - start_uj) / 1e6
        method = "RAPL cumulative energy counter (exact)"
        estimated = False
    elif samples:
        mean_w = sum(samples) / len(samples)
        joules = mean_w * elapsed
        estimated = samples_were_estimated
        method = (
            f"a ~{mean_w:.0f}W TDP estimate for this chip class, held over "
            f"{elapsed:.1f}s" if estimated else
            f"integrated from {len(samples)} power sample(s) at "
            f"{sample_interval:g}s intervals")
    else:
        tdp = estimate_tdp(cpu_model)
        if tdp:
            joules = float(tdp) * elapsed
            method = f"estimated from a ~{tdp}W TDP for this chip class"

    result: dict = {
        "label": label,
        "seconds": round(elapsed, 3),
        "joules": round(joules, 2) if joules is not None else None,
        "watt_hours": round(joules / 3600.0, 5) if joules is not None else None,
        "mean_watts": (round(joules / elapsed, 2)
                       if joules is not None and elapsed > 0 else None),
        "method": method or "unavailable",
        "estimated": estimated,
    }
    if isinstance(units, (int, float)) and units and joules:
        result["units"] = units
        result["units_per_joule"] = float(f"{units / joules:.6g}")
        result["joules_per_kilo_unit"] = float(
            f"{1000.0 * joules / units:.6g}")
    if joules is None:
        result["hint"] = _hint(sysname)
    return result


def compare_efficiency(a: dict, b: dict) -> str:
    """One sentence contrasting two energy-to-solution results.

    Written to make the race-to-idle case explicit, because a machine that
    finishes sooner while drawing more power is the outcome people most often
    misread.
    """
    ja, jb = a.get("joules"), b.get("joules")
    ta, tb = a.get("seconds"), b.get("seconds")
    if not (ja and jb and ta and tb):
        return "not enough data to compare energy"

    faster = "A" if ta < tb else "B"
    leaner = "A" if ja < jb else "B"
    energy_gap = 100.0 * abs(ja - jb) / max(ja, jb)
    time_gap = 100.0 * abs(ta - tb) / max(ta, tb)

    if faster == leaner:
        return (f"{leaner} is both {time_gap:.0f}% faster and uses "
                f"{energy_gap:.0f}% less energy")
    return (f"{faster} finishes {time_gap:.0f}% sooner but {leaner} uses "
            f"{energy_gap:.0f}% less energy — the faster machine draws more "
            f"power than it saves in time")


def render_energy(result: dict | None) -> str:
    """Terminal block for an energy measurement."""
    if not result:
        return ""
    if result.get("joules") is None:
        lines = [f"  Energy              : unavailable"]
        if result.get("hint"):
            lines.append(f"                        {result['hint']}")
        return "\n".join(lines)

    lines = [
        f"  Energy to solution  : {result['joules']:,.1f} J "
        f"over {result['seconds']:.1f}s ({result['mean_watts']:,.1f} W mean)",
    ]
    if result.get("units_per_joule"):
        lines.append(f"  Efficiency          : "
                     f"{result['units_per_joule']:,.6g} units/J "
                     f"({result['joules_per_kilo_unit']:.4g} J per 1000 units)")
    lines.append(f"      source: {result['method']}")
    if result.get("estimated"):
        lines.append("      this is a TDP estimate, not a measurement")
    return "\n".join(lines)
