"""Run-over-run regression detection.

A benchmark you run once is a number; a benchmark you run repeatedly is a
monitor. This compares the current run against this machine's own history and
flags metrics that moved beyond normal noise — catching a failing SSD, a
dust-clogged cooler, a driver regression, or a background hog that a single run
would never reveal.

Comparison is against the same *hostname* only: cross-machine differences are
expected and belong in ``compare``, not here.
"""

from __future__ import annotations

# Metrics where higher is better, with the CSV column each lives in.
_HIGHER_BETTER = {
    "cpu_int_primes_s": "CPU integer",
    "cpu_multi_primes_s": "CPU multi-core",
    "hashing_mb_s": "SHA-256",
    "mem_mb_s": "Memory bandwidth",
    "disk_write_mb_s": "Disk write",
    "disk_read_mb_s": "Disk read",
    "disk_iops": "Disk random IOPS",
    "gpu_fp32_gflops": "GPU FP32",
    "npu_gflops": "Neural Engine",
    "composite_score": "Composite score",
}

# Default: flag a change larger than 10%. Benchmarks are noisy, so a smaller
# threshold would cry wolf.
DEFAULT_THRESHOLD = 0.10

# Metrics whose result depends on a run setting. Comparing a 64 MB disk test
# against a 256 MB one shows a large "regression" that is purely the settings
# changing — bigger files exhaust an SSD's SLC cache — so each of these is only
# compared against baseline runs that used the same value.
_CONFIG_DEPS = {
    "disk_write_mb_s": "cfg_disk_mb",
    "disk_read_mb_s": "cfg_disk_mb",
    "disk_iops": "cfg_disk_mb",
    "mem_mb_s": "cfg_mem_mb",
}


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _baseline(history: list[dict], current: dict) -> dict | None:
    """Median of each metric across this host's comparable prior runs.

    The median (rather than the single last run) resists one noisy previous
    result becoming a false baseline. Metrics listed in ``_CONFIG_DEPS`` are
    additionally restricted to runs that used the same setting, so a changed
    ``--disk-mb`` never masquerades as failing hardware.
    """
    rows = [r for r in history
            if r.get("hostname") == current.get("hostname")
            and r.get("timestamp_utc") != current.get("timestamp_utc")]
    if not rows:
        return None

    import statistics
    base: dict = {"_runs": len(rows), "_skipped": []}
    for col in _HIGHER_BETTER:
        candidates = rows
        dep = _CONFIG_DEPS.get(col)
        if dep:
            want = str(current.get(dep, ""))
            candidates = [r for r in rows if str(r.get(dep, "")) == want]
            if not candidates:
                base["_skipped"].append(col)
                continue
        vals = [f for f in (_to_float(r.get(col)) for r in candidates)
                if f is not None]
        if vals:
            base[col] = statistics.median(vals)
    return base


def analyze(current_row: dict, history: list[dict],
            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compare the current run to this host's baseline.

    ``current_row`` is the flattened CSV row for this run; ``history`` is every
    row read back from the CSV (this run may or may not be included).
    """
    base = _baseline(history, current_row)
    if not base:
        return {"status": "no_baseline",
                "note": "first run on this machine; nothing to compare against"}

    findings = []
    for col, label in _HIGHER_BETTER.items():
        now = _to_float(current_row.get(col))
        was = base.get(col)
        if now is None or not was:
            continue
        delta = (now - was) / was
        if abs(delta) >= threshold:
            findings.append({
                "metric": label,
                "column": col,
                "current": round(now, 1),
                "baseline": round(was, 1),
                "change_pct": round(delta * 100, 1),
                "direction": "faster" if delta > 0 else "slower",
            })

    findings.sort(key=lambda f: f["change_pct"])   # worst regressions first
    regressions = [f for f in findings if f["direction"] == "slower"]
    return {
        "status": "regression" if regressions else "ok",
        "baseline_runs": base["_runs"],
        "skipped_metrics": base.get("_skipped", []),
        "threshold_pct": round(threshold * 100, 1),
        "findings": findings,
        "regression_count": len(regressions),
    }


def render(result: dict) -> str:
    """Human-readable regression summary for the console."""
    status = result.get("status")
    if status == "no_baseline":
        return "  Regression check: " + result["note"]

    lines = [f"  Compared against {result['baseline_runs']} prior run(s) on "
             f"this machine (threshold ±{result['threshold_pct']}%):"]
    skipped = result.get("skipped_metrics") or []
    if skipped:
        lines.append(f"    ({len(skipped)} metric(s) skipped — no prior run "
                     f"used the same settings)")
    if not result["findings"]:
        lines.append("  ✓ No significant change — performance is stable.")
        return "\n".join(lines)

    for f in result["findings"]:
        mark = "▼" if f["direction"] == "slower" else "▲"
        lines.append(f"    {mark} {f['metric']:<20} {f['change_pct']:+.1f}%  "
                     f"({f['baseline']:,.0f} → {f['current']:,.0f})")
    if result["regression_count"]:
        lines.append(f"  ⚠ {result['regression_count']} metric(s) regressed. "
                     f"Check cooling, background load, or hardware health.")
    return "\n".join(lines)
