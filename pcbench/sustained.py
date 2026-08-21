"""Sustained-load (thermal throttling) mode.

A three-second benchmark only ever measures *burst* performance. Most machines
— especially thin and fanless laptops — hold peak clocks for a minute or two
and then drop to whatever their cooling can actually dissipate. The difference
between those two numbers is often 30–50% and is invisible to a short run, yet
it is precisely what determines performance on real sustained work like
compiling, rendering, or exporting video.

This module runs a load continuously, samples throughput in fixed windows, and
reports peak vs. sustained plus the droop between them.
"""

from __future__ import annotations

from . import limits
from .core import clock
from .system import machine_state, thermal_pressure
from .workloads import PRIMES_PER_CHUNK, cpu_integer_chunk

# Fraction of the run, taken from the end, that defines "sustained"
# performance — by then thermals have equilibrated.
_TAIL_FRACTION = 0.25


def run_sustained(duration: float, window: float = 5.0,
                  workers: int = 1) -> dict:
    """Load the CPU for ``duration`` seconds, sampling every ``window``.

    ``workers`` > 1 runs the load on multiple processes to generate enough heat
    to provoke throttling on machines with capable cooling.
    """
    aborted = ""
    if workers > 1:
        samples, aborted = _run_parallel(duration, window, workers)
    else:
        samples, aborted = _run_single(duration, window)

    if not samples:
        return {"error": "no samples collected"}

    rates = [s["rate"] for s in samples]
    peak = max(rates)
    tail_start = max(1, int(len(rates) * (1 - _TAIL_FRACTION)))
    tail = rates[tail_start:] or rates[-1:]
    sustained = sum(tail) / len(tail)
    droop = (1 - sustained / peak) * 100 if peak else 0.0

    return {
        "duration_s": round(sum(s["window_s"] for s in samples), 1),
        "window_s": window,
        "workers": workers,
        "unit": "primes/s",
        "samples": samples,
        "peak_rate": peak,
        "sustained_rate": sustained,
        "droop_percent": round(droop, 1),
        "verdict": _verdict(droop),
        "state_after": machine_state(),
        "aborted_early": bool(aborted),
        "abort_reason": aborted,
    }


def _verdict(droop: float) -> str:
    if droop < 5:
        return ("no meaningful throttling — cooling sustains full performance")
    if droop < 15:
        return "mild throttling — typical for a well-cooled laptop"
    if droop < 30:
        return "moderate throttling — sustained work runs noticeably slower"
    return ("heavy throttling — cooling is the limiting factor for sustained "
            "workloads")


def _check_thermal() -> str:
    """Return an abort reason if the machine is in thermal distress.

    Hitting a thermal limit is not damage — the hardware throttles and will
    ultimately shut itself down to stay safe. But once a machine is deep into
    throttling, further load measures nothing new, and on a system whose
    cooling has already failed it only invites an abrupt shutdown.
    """
    should, reason = limits.thermal_should_abort(thermal_pressure())
    return reason if should else ""


def _run_single(duration: float, window: float) -> tuple[list[dict], str]:
    samples = []
    run_start = clock()
    while clock() - run_start < duration:
        abort = _check_thermal()
        if abort:
            return samples, abort
        w_start = clock()
        chunks = 0
        while clock() - w_start < window:
            cpu_integer_chunk()
            chunks += 1
            if clock() - run_start >= duration:
                break
        elapsed = clock() - w_start
        if elapsed <= 0:
            break
        samples.append({
            "t": round(clock() - run_start, 1),
            "window_s": round(elapsed, 2),
            "rate": chunks * PRIMES_PER_CHUNK / elapsed,
        })
    return samples, ""


def _window_worker(window: float) -> int:
    """Run chunks for one window; returns primes tested."""
    start = clock()
    chunks = 0
    while clock() - start < window:
        cpu_integer_chunk()
        chunks += 1
    return chunks * PRIMES_PER_CHUNK


def _run_parallel(duration: float, window: float,
                  workers: int) -> tuple[list[dict], str]:
    """Sample aggregate throughput across ``workers`` processes.

    The pool is created once and reused for every window so process-spawn cost
    is not charged to the measurement, and so the load stays continuous
    (letting heat build) rather than pausing between windows.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    samples = []
    with ctx.Pool(processes=workers) as pool:
        run_start = clock()
        while clock() - run_start < duration:
            abort = _check_thermal()
            if abort:
                return samples, abort
            w_start = clock()
            primes = sum(pool.map(_window_worker, [window] * workers))
            elapsed = clock() - w_start
            if elapsed <= 0:
                break
            samples.append({
                "t": round(clock() - run_start, 1),
                "window_s": round(elapsed, 2),
                "rate": primes / elapsed,
            })
    return samples, ""


def sparkline(values: list[float]) -> str:
    """Render values as a compact unicode bar chart, scaled to their own range."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return blocks[-1] * len(values)
    return "".join(
        blocks[min(len(blocks) - 1,
                   int((v - lo) / (hi - lo) * (len(blocks) - 1)))]
        for v in values)
