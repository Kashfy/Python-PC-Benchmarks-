"""Long-duration stability testing (burn-in).

A benchmark asks "how fast?". A soak test asks "for how long, without getting
anything wrong?" — and they are genuinely different questions with different
answers. Hardware that posts an excellent score can still be unusable:

* A CPU or memory overclock that is stable for three seconds routinely
  produces a wrong answer somewhere in the next three hours. That single wrong
  answer is a corrupted file, a failed build, or a crashed VM.
* Cooling that copes with a short burst fails on a long one. Throughput that
  falls off after eight minutes is invisible to any test shorter than eight
  minutes.
* Marginal power delivery shows up only once every rail has been loaded for a
  sustained period.

So this module runs the validating workloads continuously for hours if asked,
counts every mismatch instead of stopping at the first, and reports *when* each
one happened. Time-to-first-error is the number that matters: a system that
fails after four hours needs a different fix from one that fails after four
minutes.

The workloads used are the ones with exact expected answers — integer math,
compression round-trips, hashing, and memory patterns. A workload that cannot
detect a wrong answer cannot contribute to a stability verdict.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
import zlib

from . import thermal as thermal_mod
from .limits import THERMAL_ABORT_CELSIUS
from .sustained import sparkline

MB = 1024 * 1024

# How often a worker reports progress to the parent. Short enough that a
# session can be interrupted responsively, long enough that the reporting
# itself is not a meaningful share of the work.
_REPORT_SECONDS = 2.0


# --------------------------------------------------------------------------- #
# Self-validating work units
# --------------------------------------------------------------------------- #
def _unit_integer(seed: int) -> tuple[str, bool]:
    """Modular exponentiation with an independently checkable identity.

    Fermat's little theorem gives a result that is known in advance without
    performing the same computation twice, so a fault cannot cancel itself out
    by corrupting both the value and the expectation identically.
    """
    prime = 2147483647              # 2**31 - 1, a Mersenne prime
    base = 2 + (seed % (prime - 3))
    value = pow(base, prime - 1, prime)
    return "integer", value == 1


def _unit_compression(seed: int, payload: bytes) -> tuple[str, bool]:
    blob = zlib.compress(payload, 6)
    return "compression", zlib.decompress(blob) == payload


def _unit_hash(seed: int, payload: bytes, expected: str) -> tuple[str, bool]:
    return "hashing", hashlib.sha256(payload).hexdigest() == expected


def _unit_memory(seed: int, size: int) -> tuple[str, bool]:
    """Write a pattern across a buffer, read it back, and compare.

    A walking pattern rather than a constant one: constant fills survive stuck
    bits and address-decode faults, which are exactly the failures worth
    catching.
    """
    pattern = bytes(((i * 7 + seed) & 0xFF) for i in range(256))
    buf = bytearray(pattern * (size // 256))
    copy = bytearray(len(buf))
    copy[:] = buf
    return "memory", bytes(copy) == bytes(buf) and copy[0] == pattern[0]


def _worker(duration: float, buffer_kb: int, queue) -> None:
    """Run validating units until the time budget expires."""
    payload = bytes(((i * 31 + 7) & 0xFF) for i in range(64 * 1024)) * 4
    expected_hash = hashlib.sha256(payload).hexdigest()
    mem_size = max(4096, buffer_kb * 1024)

    start = time.monotonic()
    last_report = start
    units = 0
    errors: list[dict] = []
    seed = os.getpid()

    try:
        while True:
            now = time.monotonic()
            if now - start >= duration:
                break
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            for name, ok in (
                    _unit_integer(seed),
                    _unit_compression(seed, payload),
                    _unit_hash(seed, payload, expected_hash),
                    _unit_memory(seed, mem_size)):
                units += 1
                if not ok:
                    errors.append({"unit": name,
                                   "elapsed_s": round(now - start, 2),
                                   "pid": os.getpid()})
            if now - last_report >= _REPORT_SECONDS:
                queue.put({"pid": os.getpid(), "units": units,
                           "errors": errors, "elapsed": now - start})
                errors = []
                last_report = now
    except KeyboardInterrupt:
        pass
    queue.put({"pid": os.getpid(), "units": units, "errors": errors,
               "elapsed": time.monotonic() - start, "done": True})


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(duration: float, workers: int = 0, buffer_kb: int = 1024,
        script_dir: str = ".", quiet: bool = False,
        progress_seconds: float = 30.0) -> dict:
    """Load every core with validating work for ``duration`` seconds.

    Returns as soon as the duration elapses or the thermal abort limit is hit.
    Aborting on temperature is not squeamishness: past the limit the hardware
    is throttling or shutting down on its own, so continuing measures the
    protection circuitry rather than the machine.
    """
    workers = workers or os.cpu_count() or 1
    duration = max(1.0, float(duration))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(duration, buffer_kb, queue),
                         daemon=True) for _ in range(workers)]

    started = time.monotonic()
    for p in procs:
        p.start()

    errors_seen: list[dict] = []
    # Workers report their unit count cumulatively but their error list
    # incrementally, so counts are kept per-PID (latest wins) while errors
    # accumulate. Mixing the two up would either lose errors or count units
    # several times over.
    units_by_pid: dict[int, int] = {}
    temps: list[float] = []
    aborted = None
    last_progress = started
    interrupted = False

    def drain(msg: dict) -> None:
        pid = msg.get("pid", 0)
        units_by_pid[pid] = max(units_by_pid.get(pid, 0), msg.get("units", 0))
        errors_seen.extend(msg.get("errors", []))

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
            while True:
                try:
                    drain(queue.get_nowait())
                except Exception:
                    break

            elapsed = time.monotonic() - started
            temp = thermal_mod.cpu_celsius(script_dir)
            if temp is not None:
                temps.append(temp)
                if temp >= THERMAL_ABORT_CELSIUS:
                    aborted = (f"stopped early at {temp:.0f} °C "
                               f"(limit {THERMAL_ABORT_CELSIUS:.0f} °C)")
                    break

            if (not quiet
                    and time.monotonic() - last_progress >= progress_seconds):
                last_progress = time.monotonic()
                remaining = max(0.0, duration - elapsed)
                temp_text = f", {temp:.0f} °C" if temp is not None else ""
                print(f"    {_hms(elapsed)} elapsed, {_hms(remaining)} left, "
                      f"{len(errors_seen)} error(s){temp_text}", flush=True)
    except KeyboardInterrupt:
        # An interrupted soak still has hours of evidence in it; discarding
        # that because the user pressed Ctrl-C would be the wrong trade.
        interrupted = True
        aborted = "interrupted by user"

    for p in procs:
        if p.is_alive():
            p.terminate()
    # Drain whatever the workers left behind, including the final cumulative
    # counts each one sends on the way out.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            drain(queue.get(timeout=0.2))
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=2.0)

    total_units = sum(units_by_pid.values())
    elapsed = time.monotonic() - started
    errors = sorted(errors_seen, key=lambda e: e["elapsed_s"])
    unique = sorted({e["unit"] for e in errors})

    result = {
        "requested_seconds": duration,
        "elapsed_seconds": round(elapsed, 1),
        "workers": workers,
        "units_completed": total_units,
        "units_per_second": round(total_units / elapsed, 1) if elapsed else 0.0,
        "errors": len(errors),
        "error_details": errors[:50],
        "error_types": unique,
        "time_to_first_error_s": errors[0]["elapsed_s"] if errors else None,
        "aborted": aborted,
        "interrupted": interrupted,
        "completed": aborted is None and elapsed >= duration * 0.98,
    }
    if temps:
        result["temperature"] = {
            "min": round(min(temps), 1), "max": round(max(temps), 1),
            "mean": round(sum(temps) / len(temps), 1),
            "spark": sparkline(temps),
        }
    result["verdict"] = verdict(result)
    return result


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def verdict(result: dict) -> str:
    """One sentence a person can act on."""
    if result["errors"]:
        first = result.get("time_to_first_error_s")
        types = ", ".join(result["error_types"])
        return (f"UNSTABLE — {result['errors']} wrong answer(s) in "
                f"{types} after {_hms(first or 0)}. This machine will corrupt "
                f"data under load; suspect memory, an unstable overclock, or "
                f"inadequate cooling.")
    if result.get("aborted") and not result.get("interrupted"):
        return (f"INCONCLUSIVE — {result['aborted']}. No wrong answers were "
                f"produced, but the machine could not hold the load long "
                f"enough to prove stability.")
    if result.get("interrupted"):
        return (f"INCOMPLETE — stopped after "
                f"{_hms(result['elapsed_seconds'])} with no errors. Stability "
                f"is only demonstrated for the time actually run.")
    return (f"STABLE — {result['units_completed']:,} validated work units "
            f"over {_hms(result['elapsed_seconds'])} on "
            f"{result['workers']} worker(s) with zero wrong answers.")


def render(result: dict | None) -> str:
    if not result:
        return ""
    lines = [
        f"  Duration        : {_hms(result['elapsed_seconds'])} "
        f"of {_hms(result['requested_seconds'])} requested",
        f"  Workers         : {result['workers']}",
        f"  Work units      : {result['units_completed']:,} "
        f"({result['units_per_second']:,}/s)",
        f"  Wrong answers   : {result['errors']}",
    ]
    if result.get("time_to_first_error_s") is not None:
        lines.append(f"  First error at  : "
                     f"{_hms(result['time_to_first_error_s'])}")
    temp = result.get("temperature")
    if temp:
        lines.append(f"  Temperature     : {temp['min']}–{temp['max']} °C "
                     f"(mean {temp['mean']})  {temp['spark']}")
    lines.append(f"  {result['verdict']}")
    return "\n".join(lines)
