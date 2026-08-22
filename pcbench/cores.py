"""Per-core performance mapping and scaling analysis.

Aggregate multi-core throughput hides the shape of a modern CPU. A chip that
reports "10 cores" may be four fast cores plus six slow ones, and a 5x scaling
result is then entirely expected rather than a disappointment. This module
recovers that structure.

Two complementary methods:

* **Scaling curve** (everywhere) — measure aggregate throughput at 1, 2, … N
  workers. The *marginal* gain from each additional worker reveals what kind of
  core it landed on: on an Apple M4 the first four workers each add ~3.9M
  primes/s and the remaining six add ~1.1M, which is precisely its 4
  performance + 6 efficiency layout. This needs no affinity API, which matters
  because macOS does not expose one.

* **Per-core pinning** (Linux and Windows) — pin a worker to each core in turn
  for a direct measurement. More precise where the OS allows it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform

from .core import clock
from .workloads import PRIMES_PER_CHUNK, cpu_integer_chunk

# A worker counts as scaling "near-linearly" while it still contributes at
# least this fraction of what the first worker did.
LINEAR_SCALING_RATIO = 0.70

# The slow group must average below this fraction of the fast group before the
# machine is called hybrid; a gentler difference is ordinary scaling loss.
HYBRID_DROP_RATIO = 0.65

# How close the scaling knee must sit to the physical core count before the
# drop is attributed to SMT. A knee exactly at the number of physical cores is
# the signature of hyperthreads taking over; one somewhere else is not.
SMT_KNEE_TOLERANCE = 1


def _worker(duration: float) -> int:
    """Run prime chunks for ``duration``; returns primes tested."""
    start = clock()
    chunks = 0
    while clock() - start < duration:
        cpu_integer_chunk()
        chunks += 1
    return chunks * PRIMES_PER_CHUNK


def _pinned_worker(args) -> tuple[int, int]:
    """Run the workload pinned to one core. Returns (core, primes)."""
    core, duration = args
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
        elif platform.system() == "Windows":
            import ctypes
            ctypes.windll.kernel32.SetThreadAffinityMask(
                ctypes.windll.kernel32.GetCurrentThread(), 1 << core)
    except Exception:
        pass                    # unpinned result is still a valid sample
    return core, _worker(duration)


def scaling_curve(seconds: float = 0.5,
                  max_workers: int | None = None) -> list[dict]:
    """Aggregate throughput at 1..N workers, with the marginal gain of each."""
    max_workers = max_workers or os.cpu_count() or 1
    ctx = mp.get_context("spawn")
    points, previous = [], 0.0

    for n in range(1, max_workers + 1):
        with ctx.Pool(processes=n) as pool:
            primes = sum(pool.map(_worker, [seconds] * n))

        # Divide by the workers' own measured duration, not wall time. Wall
        # time includes pool creation, which costs more for larger n and would
        # therefore understate exactly the higher worker counts this analysis
        # depends on — the dominant source of noise in the marginal gains.
        aggregate = primes / seconds

        points.append({
            "workers": n,
            "aggregate_rate": aggregate,
            "marginal_rate": aggregate - previous,
            "scaling_vs_one": (aggregate / points[0]["aggregate_rate"]
                               if points else 1.0),
        })
        previous = aggregate
    return points


def classify_cores(points: list[dict], physical_cores: int | None = None,
                   logical_cores: int | None = None) -> dict:
    """Characterise the core layout from the marginal gains.

    **What this reports, and why not more.** An earlier version of this
    function estimated exact performance/efficiency core counts. That estimate
    proved unreliable and was removed: the last performance core shares cache
    and memory bandwidth with its siblings, so its marginal contribution falls
    to roughly the level of the first efficiency core. Measured on an Apple M4
    (truly 4P + 6E) the sorted gains run

        4.42, 4.27, 4.22, 2.56 | 2.23, 1.40, 1.38, 1.26, 0.81 M primes/s

    and the real boundary between 2.56 and 2.23 is indistinguishable from
    ordinary scaling loss — successive runs "detected" 4/6, 3/7 and 8/2.

    What *is* stable across runs, and genuinely useful, is reported instead:
    how far scaling stays near-linear, whether the machine is hybrid at all,
    and the throughput ratio between the fast and slow groups. For exact
    per-core figures, ``per_core_map`` pins work to each core on platforms
    that expose thread affinity.
    """
    gains = [p["marginal_rate"] for p in points]
    if len(gains) < 2 or gains[0] <= 0:
        return {"hybrid": False, "note": "too few cores to classify"}

    logical_cores = logical_cores or (os.cpu_count() or len(gains))

    reference = gains[0]

    # How many workers scale near-linearly: the run of leading workers each
    # still contributing most of a full core. This is the actionable number —
    # it says how many parallel workers a machine profitably sustains.
    linear = 0
    for g in gains:
        if g >= reference * LINEAR_SCALING_RATIO:
            linear += 1
        else:
            break

    positives = sorted((g for g in gains if g > 0), reverse=True)
    fast = [g for g in positives if g >= reference * LINEAR_SCALING_RATIO]
    slow = [g for g in positives if g < reference * LINEAR_SCALING_RATIO]
    fast_each = sum(fast) / len(fast) if fast else 0.0
    slow_each = sum(slow) / len(slow) if slow else 0.0
    hybrid = bool(slow) and fast_each > 0 and \
        slow_each < fast_each * HYBRID_DROP_RATIO

    # A drop in marginal gain has two completely different causes, and the
    # curve alone cannot tell them apart:
    #
    #   * Hybrid layout — physically different core types (Apple silicon,
    #     Intel 12th gen and later). The slow workers landed on smaller cores.
    #   * SMT / Hyper-Threading — the same cores running a second thread each.
    #     Once every physical core is busy, additional logical CPUs share
    #     execution units and contribute roughly a third as much.
    #
    # They call for different actions (pin to P-cores vs. consider disabling
    # SMT), so guessing is worse than declining to guess. The physical core
    # count settles it: a knee sitting at exactly that number is SMT.
    smt_capable = bool(physical_cores and logical_cores
                       and logical_cores > physical_cores)
    knee_at_physical = bool(
        physical_cores
        and abs(linear - physical_cores) <= SMT_KNEE_TOLERANCE)

    cause = None
    if slow and smt_capable and knee_at_physical:
        cause = "smt"
    elif slow and smt_capable:
        cause = "ambiguous"
    elif hybrid:
        cause = "hybrid"

    result = {
        "hybrid": hybrid and cause == "hybrid",
        "cause": cause,
        "linear_up_to_workers": linear,
        "fast_rate_each": fast_each,
        "slow_rate_each": slow_each,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
    }
    relative = (slow_each / fast_each) if fast_each and slow else None
    if relative is not None:
        result["slow_relative"] = round(relative, 2)
    pct = f"{relative * 100:.0f}%" if relative is not None else "less"

    if cause == "smt":
        result["note"] = (
            f"scales near-linearly to {linear} worker(s), matching this "
            f"machine's {physical_cores} physical cores; beyond that each "
            f"added worker contributes about {pct} as much because it is a "
            f"second thread on an already-busy core (SMT), not a slower core")
    elif cause == "ambiguous":
        result["note"] = (
            f"scales near-linearly to {linear} worker(s), then each added "
            f"worker contributes about {pct} as much. With "
            f"{physical_cores} physical and {logical_cores} logical cores the "
            f"curve cannot separate SMT from a hybrid layout, so no cause is "
            f"claimed")
    elif cause == "hybrid":
        result["note"] = (
            f"scales near-linearly to {linear} worker(s); beyond that each "
            f"added worker contributes about {pct} as much. With no SMT on "
            f"this machine that indicates a hybrid design with slower "
            f"efficiency cores")
    else:
        result["note"] = (
            f"scales near-linearly to {linear} worker(s); cores appear "
            f"uniform in performance")
    return result


def per_core_map(seconds: float = 0.4) -> list[dict] | None:
    """Measure each core individually by pinning. None where unsupported.

    macOS provides no thread-affinity API, so this returns None there and the
    scaling curve carries the analysis instead.
    """
    if not hasattr(os, "sched_setaffinity") and platform.system() != "Windows":
        return None
    n = os.cpu_count() or 1
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(processes=1) as pool:
            results = pool.map(_pinned_worker,
                               [(c, seconds) for c in range(n)])
    except Exception:
        return None
    out = []
    for core, primes in results:
        out.append({"core": core, "rate": primes / seconds})
    peak = max((r["rate"] for r in out), default=0) or 1
    for r in out:
        r["relative"] = round(r["rate"] / peak, 3)
    return out


def analyze(seconds: float = 0.5, max_workers: int | None = None,
            physical_cores: int | None = None) -> dict:
    """Full core analysis: scaling curve, inferred layout, per-core map."""
    points = scaling_curve(seconds, max_workers)
    if not points:
        return {"error": "no scaling data"}

    if physical_cores is None:
        from .system import physical_cores as detect_physical
        physical_cores = detect_physical()
    logical = os.cpu_count() or 1
    classes = classify_cores(points, physical_cores, logical)
    peak = points[-1]
    return {
        "logical_cores": logical,
        "physical_cores": physical_cores,
        "points": points,
        "classes": classes,
        "peak_aggregate_rate": peak["aggregate_rate"],
        "scaling_factor": round(peak["scaling_vs_one"], 2),
        "per_core": per_core_map(min(seconds, 0.4)),
        "unit": "primes/s",
    }
