"""Detect conditions that changed *during* a run and distorted a result.

Machine state is checked before a run starts, but a run takes minutes and the
machine can change underneath it: a backup kicks off, a browser wakes, the
chassis heats up. A result taken under those conditions is wrong in a way no
amount of statistical repetition inside the test can fix, because every repeat
is equally affected.

This samples load and temperature around each individual test and flags the
ones measured under changed conditions, so a distorted number is labelled
rather than silently ranked alongside clean ones.
"""

from __future__ import annotations

import os

# A test is flagged when load per core rose by more than this during it.
LOAD_DELTA_THRESHOLD = 0.25

# ...or when the CPU warmed by more than this many degrees, which usually
# means clocks fell part-way through.
TEMP_RISE_THRESHOLD = 12.0

# ...or when it was already hot enough to be throttling.
TEMP_HOT_THRESHOLD = 85.0


def sample(script_dir: str = ".") -> dict:
    """A cheap snapshot of the conditions a measurement runs under."""
    snapshot: dict = {}
    try:
        la = os.getloadavg()
        snapshot["load_per_core"] = la[0] / (os.cpu_count() or 1)
    except (OSError, AttributeError):
        pass
    try:
        from .thermal import read as read_temps
        temp = read_temps(script_dir).get("cpu_celsius")
        if temp is not None:
            snapshot["celsius"] = temp
    except Exception:
        pass
    return snapshot


def compare_samples(before: dict, after: dict) -> dict:
    """Judge whether conditions changed enough to distrust the result."""
    findings: list[str] = []
    detail: dict = {}

    b_load, a_load = before.get("load_per_core"), after.get("load_per_core")
    if b_load is not None and a_load is not None:
        delta = a_load - b_load
        detail["load_delta"] = round(delta, 3)
        if delta > LOAD_DELTA_THRESHOLD:
            findings.append(
                f"system load rose {delta:.2f} per core during this test — "
                f"something else started competing for the CPU")

    b_temp, a_temp = before.get("celsius"), after.get("celsius")
    if b_temp is not None and a_temp is not None:
        rise = a_temp - b_temp
        detail["temp_rise_c"] = round(rise, 1)
        detail["temp_end_c"] = round(a_temp, 1)
        if rise > TEMP_RISE_THRESHOLD:
            findings.append(
                f"CPU warmed {rise:.0f} °C during this test, so clocks likely "
                f"fell part-way through")
        elif a_temp >= TEMP_HOT_THRESHOLD:
            findings.append(
                f"CPU was at {a_temp:.0f} °C — near or at its thermal limit")

    return {
        "disturbed": bool(findings),
        "notes": findings,
        **detail,
    }


def summarize(results: dict) -> dict:
    """Roll per-test findings into a run-level verdict."""
    disturbed = [name for name, r in results.items()
                 if isinstance(r, dict)
                 and (r.get("interference") or {}).get("disturbed")]
    return {
        "disturbed_tests": disturbed,
        "clean": not disturbed,
        "note": ("all tests ran under stable conditions" if not disturbed else
                 f"{len(disturbed)} test(s) ran under changing conditions and "
                 f"may understate this machine: {', '.join(disturbed)}"),
    }
