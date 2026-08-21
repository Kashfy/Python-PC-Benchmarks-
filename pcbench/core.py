"""Timing, statistics, warm-up, and result validation primitives.

Everything in here is dependency-free and side-effect free so it can be unit
tested directly.
"""

from __future__ import annotations

import statistics
import time

# Fraction of a test's target duration spent warming up (cold caches, CPU
# frequency ramp, branch predictor training) before any timing is recorded.
WARMUP_FRACTION = 0.15
WARMUP_MIN_SECONDS = 0.05
WARMUP_MAX_SECONDS = 1.0


def clock() -> float:
    """Monotonic high-resolution clock. Never affected by wall-clock changes."""
    return time.perf_counter()


def warmup(chunk_func, seconds: float) -> int:
    """Run ``chunk_func`` untimed to bring the machine to a steady state.

    Returns the number of warm-up iterations performed. Always runs at least
    one iteration so the code path, its caches, and any lazy imports are hot
    before measurement begins.
    """
    budget = min(max(seconds * WARMUP_FRACTION, WARMUP_MIN_SECONDS),
                 WARMUP_MAX_SECONDS)
    start = clock()
    n = 0
    while True:
        chunk_func()
        n += 1
        if clock() - start >= budget:
            return n


def timed_loop(chunk_func, seconds: float) -> tuple[float, int]:
    """Call ``chunk_func`` repeatedly for ~``seconds``.

    Returns ``(elapsed_seconds, iterations)``. The caller multiplies iterations
    by its work-per-chunk constant to derive a rate.
    """
    start = clock()
    count = 0
    while True:
        chunk_func()
        count += 1
        elapsed = clock() - start
        if elapsed >= seconds:
            return elapsed, count


def summarize(samples: list[float]) -> dict:
    """Reduce per-repeat rates to a robust statistical summary.

    ``median`` is the headline figure because it resists one-off outliers
    (thermal spikes, a background process waking up). ``cv`` is the coefficient
    of variation (stdev/mean) — a normalized stability indicator that is
    comparable across tests with wildly different magnitudes.
    """
    if not samples:
        return {"median": 0.0, "mean": 0.0, "stdev": 0.0, "cv": 0.0,
                "min": 0.0, "max": 0.0, "samples": []}
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median": statistics.median(samples),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else 0.0,
        "min": min(samples),
        "max": max(samples),
        "samples": [round(s, 4) for s in samples],
    }


def stability_note(cv: float) -> str:
    """Human label for a coefficient of variation."""
    if cv <= 0.02:
        return "excellent"
    if cv <= 0.05:
        return "good"
    if cv <= 0.10:
        return "fair"
    return "unstable"


class ValidationError(Exception):
    """Raised when a workload produces a numerically incorrect result.

    A benchmark that computes the *wrong answer* quickly is not fast, it is
    broken. Mismatches point at unstable overclocks, failing RAM, inadequate
    cooling, or a miscompiled toolchain — which is exactly the kind of fault
    this tool exists to surface.
    """


def check_exact(name: str, got, expected) -> None:
    if got != expected:
        raise ValidationError(
            f"{name}: expected {expected!r}, got {got!r} — this indicates "
            f"hardware instability or a miscompiled runtime")


def check_close(name: str, got: float, expected: float,
                rel_tol: float = 1e-9) -> None:
    """Compare floats with a relative tolerance.

    A tolerance is required because platform math libraries (`libm`, the MSVC
    CRT, Apple's Accelerate) round transcendental functions differently in the
    last bits. The tolerance is tight enough that a genuine computation fault
    still trips it.
    """
    if expected == 0:
        ok = abs(got) <= rel_tol
    else:
        ok = abs(got - expected) / abs(expected) <= rel_tol
    if not ok:
        raise ValidationError(
            f"{name}: expected ~{expected!r}, got {got!r} — this indicates "
            f"hardware instability or a miscompiled runtime")
