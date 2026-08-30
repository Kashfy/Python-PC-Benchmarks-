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
# compared against baseline runs that used the same value. A metric may depend
# on more than one, in which case every one of them has to match.
_CONFIG_DEPS = {
    # `method_version` covers the case a settings column cannot: the tool
    # itself changing how a number is produced. When the random-read test
    # started bypassing the page cache the figure fell by more than an order
    # of magnitude on every machine — a correction, not a failing drive, and
    # without this it would be reported as the most severe regression the tool
    # can detect.
    "disk_write_mb_s": ("cfg_disk_mb", "method_version"),
    "disk_read_mb_s": ("cfg_disk_mb", "method_version"),
    "disk_iops": ("cfg_disk_mb", "method_version"),
    "composite_score": "method_version",
    "mem_mb_s": "cfg_mem_mb",
    # Several workloads are pure-Python loops, so their result depends on the
    # interpreter as much as the hardware — CPython releases differ by tens of
    # percent on this kind of code. Upgrading Python would otherwise appear as
    # a hardware change.
    "cpu_int_primes_s": "python_version",
    "cpu_multi_primes_s": "python_version",
    "nn_train_steps_s": "python_version",
    "kmeans_dist_s": "python_version",
    "knn_cmp_s": "python_version",
}


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


#: Prior runs needed before a metric's own variability can be estimated.
#: Below this, a change is reported but labelled provisional -- variance
#: cannot be inferred from one or two samples.
MIN_RUNS_FOR_SPREAD = 3

#: How many robust standard deviations a change must exceed to count as
#: outside a metric's normal run-to-run variation.
SPREAD_MULTIPLIER = 3.0


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
            deps = (dep,) if isinstance(dep, str) else dep
            candidates = [
                r for r in rows
                if all(str(r.get(d, "")) == str(current.get(d, ""))
                       for d in deps)]
            if not candidates:
                base["_skipped"].append(col)
                continue
        vals = [f for f in (_to_float(r.get(col)) for r in candidates)
                if f is not None]
        if vals:
            base[col] = statistics.median(vals)
            # Keep the spread, not just the centre. A fixed percentage
            # threshold treats every metric as equally repeatable, and they
            # are not: sequential disk throughput routinely swings 30% between
            # runs on the same machine while integer CPU work varies under 1%.
            # Without this, the noisy metrics generate most of the "findings".
            base.setdefault("_spread", {})[col] = _mad(vals)
            base.setdefault("_samples", {})[col] = len(vals)
    return base


def _mad(values: list[float]) -> float:
    """Median absolute deviation — a spread measure that ignores outliers.

    Chosen over standard deviation because the histories being summarised are
    short and frequently contain exactly the one-off bad run this check exists
    to avoid being fooled by. Scaled by 1.4826 so it estimates the standard
    deviation of a normal distribution, which makes the multiplier below
    interpretable in the usual way.
    """
    import statistics
    if len(values) < 2:
        return 0.0
    centre = statistics.median(values)
    return 1.4826 * statistics.median([abs(v - centre) for v in values])


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
        if abs(delta) < threshold:
            continue

        samples = (base.get("_samples") or {}).get(col, 0)
        spread = (base.get("_spread") or {}).get(col, 0.0)
        finding = {
            "metric": label,
            "column": col,
            "current": round(now, 1),
            "baseline": round(was, 1),
            "change_pct": round(delta * 100, 1),
            "direction": "faster" if delta > 0 else "slower",
            "prior_runs": samples,
        }

        # With enough history the metric's own variability decides whether a
        # change is real. Below that there is nothing to estimate it from, and
        # the finding is reported as provisional rather than suppressed.
        if samples >= MIN_RUNS_FOR_SPREAD and spread > 0:
            deviations = abs(now - was) / spread
            finding["deviations_from_normal"] = round(deviations, 1)
            finding["typical_spread_pct"] = round(100.0 * spread / was, 1)
            finding["confidence"] = ("outside normal variation"
                                     if deviations >= SPREAD_MULTIPLIER
                                     else "within normal variation")
        elif samples >= MIN_RUNS_FOR_SPREAD:
            finding["confidence"] = "outside normal variation"
        else:
            finding["confidence"] = "provisional"
        findings.append(finding)

    findings.sort(key=lambda f: f["change_pct"])   # worst regressions first
    # Only a change the metric's own history cannot explain counts as a
    # regression. A metric that swings this much routinely is reported, but it
    # does not raise the alarm.
    regressions = [f for f in findings
                   if f["direction"] == "slower"
                   and f.get("confidence") != "within normal variation"]
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
        note = ""
        if f.get("confidence") == "within normal variation":
            note = (f"   (within this metric's normal ±"
                    f"{f.get('typical_spread_pct', 0):.0f}% spread)")
        elif f.get("confidence") == "provisional":
            note = f"   (provisional — only {f.get('prior_runs', 0)} prior run)"
        lines.append(f"    {mark} {f['metric']:<20} {f['change_pct']:+.1f}%  "
                     f"({f['baseline']:,.0f} → {f['current']:,.0f}){note}")
    if result["regression_count"]:
        lines.append(f"  ⚠ {result['regression_count']} metric(s) regressed "
                     f"beyond their normal variation. Most likely causes, in "
                     f"order: background load during the run, thermal "
                     f"throttling, a changed setting, then hardware health.")
    elif result["findings"]:
        # "Provisional" and "within normal variation" are different states and
        # must not share a summary: one means the change is explained by known
        # noise, the other means there is not yet enough history to judge.
        provisional = [f for f in result["findings"]
                       if f.get("confidence") == "provisional"]
        if provisional:
            lines.append(
                f"  ? {len(provisional)} change(s) seen, but with "
                f"{result['baseline_runs']} prior run(s) there is not enough "
                f"history to tell a real change from normal variation. A few "
                f"more runs will settle it.")
        else:
            lines.append("  ✓ Changes seen are within each metric's normal "
                         "run-to-run variation.")
    return "\n".join(lines)
