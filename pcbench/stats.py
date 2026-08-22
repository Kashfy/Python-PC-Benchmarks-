"""Confidence intervals and significance testing.

A median and a coefficient of variation describe *one* run. They cannot answer
the question people actually ask when they change something:

    "Build A scores 312 and build B scores 321. Is B faster?"

Usually not. With three repeats and 4% run-to-run variance, a 3% difference is
indistinguishable from noise, and reporting it as an improvement is how
performance work goes wrong — teams ship changes that did nothing and chase
regressions that never happened.

So this module provides the two things needed to answer it honestly:

* **A confidence interval** on each measurement, so a single number carries its
  own uncertainty rather than implying precision it does not have.
* **A significance test** between two sets of samples, reporting whether the
  difference is larger than the noise, and how much data would be needed if it
  is not yet conclusive.

The test is the **Mann-Whitney U** rank-sum test, chosen deliberately over the
more familiar t-test. Benchmark samples are not normally distributed: they are
bounded below by the hardware's best case and have a long upper tail of
interference (a background process, a thermal event, a page fault storm). A
t-test assumes symmetry it does not have, and the median that the rest of this
tool reports is a rank statistic anyway, so testing ranks keeps the headline
figure and the test consistent.

Everything is pure standard library — no SciPy — because the tool must keep
running on a bare interpreter, and these are a few dozen lines of arithmetic.
"""

from __future__ import annotations

import math
import statistics

#: Below this many samples per side, no significance claim is made at all.
#: Two samples cannot distinguish a real effect from a coin flip, and a test
#: that returns a confident answer from two points is worse than no test.
MIN_SAMPLES = 3

#: Effect sizes below this are treated as practically irrelevant even when
#: statistically detectable, because at some point a difference stops
#: mattering regardless of how confident one is that it exists.
NEGLIGIBLE_PCT = 1.0


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Two-sided critical value of Student's t.

    A short table rather than an inverse-CDF implementation: benchmark repeat
    counts are small integers, and the table is exact where it matters and
    converges to the normal value beyond it.
    """
    if confidence >= 0.99:
        table = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
                 6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
                 12: 3.055, 15: 2.947, 20: 2.845, 30: 2.750, 60: 2.660}
        fallback = 2.576
    elif confidence >= 0.95:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}
        fallback = 1.960
    else:
        table = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
                 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
                 12: 1.782, 15: 1.753, 20: 1.725, 30: 1.697, 60: 1.671}
        fallback = 1.645

    if df <= 0:
        return fallback
    if df in table:
        return table[df]
    usable = [k for k in sorted(table) if k <= df]
    return table[usable[-1]] if usable else fallback


def confidence_interval(samples: list[float],
                        confidence: float = 0.95) -> dict:
    """Mean with a confidence interval, plus the relative half-width.

    ``relative_margin_pct`` is the number to look at: it says how large a
    difference this measurement is even capable of resolving. A 6% margin means
    a 3% "improvement" is unmeasurable with the data in hand.
    """
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": 0.0, "low": 0.0, "high": 0.0,
                "margin": 0.0, "relative_margin_pct": None,
                "confidence": confidence}
    mean = statistics.fmean(samples)
    if n == 1:
        return {"n": 1, "mean": mean, "low": mean, "high": mean,
                "margin": 0.0, "relative_margin_pct": None,
                "confidence": confidence,
                "note": "a single sample carries no uncertainty estimate"}

    stdev = statistics.stdev(samples)
    margin = _t_critical(n - 1, confidence) * stdev / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "low": mean - margin,
        "high": mean + margin,
        "margin": margin,
        "relative_margin_pct": (100.0 * margin / mean) if mean else None,
        "confidence": confidence,
    }


def required_samples(samples: list[float], detect_pct: float,
                     confidence: float = 0.95) -> int | None:
    """How many repeats are needed to resolve a ``detect_pct`` difference.

    Answers the practical follow-up to an inconclusive comparison: "how much
    longer do I have to run this?" Returns None when the data cannot support
    the estimate.
    """
    if len(samples) < 2 or detect_pct <= 0:
        return None
    mean = statistics.fmean(samples)
    if not mean:
        return None
    cv = statistics.stdev(samples) / mean
    if cv == 0:
        return MIN_SAMPLES
    # n >= (z * cv / relative_effect)^2, using the normal approximation and
    # rounding up; exact enough for a "run it this many times" answer.
    z = _t_critical(60, confidence)
    n = (z * cv / (detect_pct / 100.0)) ** 2
    return max(MIN_SAMPLES, int(math.ceil(n)))


# --------------------------------------------------------------------------- #
# Mann-Whitney U
# --------------------------------------------------------------------------- #
def _rank(values: list[float]) -> list[float]:
    """Ranks with ties averaged, as the rank-sum test requires."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while (j + 1 < len(indexed)
               and values[indexed[j + 1]] == values[indexed[i]]):
            j += 1
        average = (i + j + 2) / 2.0        # ranks are 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = average
        i = j + 1
    return ranks


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney(a: list[float], b: list[float]) -> dict:
    """Two-sided rank-sum test, returning U and an approximate p-value.

    Uses the normal approximation with a tie correction and a continuity
    correction. The approximation is poor for very small samples, which is
    precisely why :func:`compare` refuses to draw conclusions below
    ``MIN_SAMPLES`` rather than reporting a p-value that looks authoritative.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"u": None, "p": None,
                "note": "one side has no samples"}

    combined = list(a) + list(b)
    ranks = _rank(combined)
    rank_sum_a = sum(ranks[:na])

    u_a = rank_sum_a - na * (na + 1) / 2.0
    u_b = na * nb - u_a
    u = min(u_a, u_b)

    mean_u = na * nb / 2.0

    # Tie correction: without it, p-values are overstated when many samples
    # share a value, which happens with coarse timer resolution.
    counts: dict[float, int] = {}
    for value in combined:
        counts[value] = counts.get(value, 0) + 1
    n = na + nb
    tie_term = sum(c ** 3 - c for c in counts.values())
    variance = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) \
        if n > 1 else 0.0

    if variance <= 0:
        return {"u": u, "p": 1.0,
                "note": "all samples are identical; no difference to detect"}

    z = (abs(u - mean_u) - 0.5) / math.sqrt(variance)   # continuity correction
    p = 2.0 * (1.0 - _normal_cdf(max(0.0, z)))
    return {"u": u, "z": z, "p": min(1.0, max(0.0, p)),
            "n_a": na, "n_b": nb}


def cliffs_delta(a: list[float], b: list[float]) -> float | None:
    """Effect size: the probability b exceeds a, minus the reverse.

    Reported alongside p because significance and magnitude are different
    questions. With enough repeats a 0.3% difference becomes statistically
    significant while remaining completely irrelevant, and delta is what makes
    that visible.
    """
    if not a or not b:
        return None
    greater = sum(1 for x in a for y in b if y > x)
    less = sum(1 for x in a for y in b if y < x)
    return (greater - less) / float(len(a) * len(b))


def _delta_label(delta: float) -> str:
    """Conventional thresholds for Cliff's delta."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(baseline: list[float], candidate: list[float],
            alpha: float = 0.05, label: str = "metric",
            higher_is_better: bool = True) -> dict:
    """Decide whether ``candidate`` genuinely differs from ``baseline``.

    Returns a verdict that says which of the three real outcomes applies:
    a difference was demonstrated, no difference was demonstrated, or there is
    not enough data to say. The third is a distinct answer and is reported as
    such — collapsing it into "no difference" is how underpowered comparisons
    get mistaken for evidence of no effect.
    """
    result: dict = {
        "metric": label,
        "baseline": summary(baseline),
        "candidate": summary(candidate),
        "alpha": alpha,
    }

    b_med = result["baseline"]["median"]
    c_med = result["candidate"]["median"]
    change_pct = (100.0 * (c_med - b_med) / b_med) if b_med else None
    result["change_pct"] = round(change_pct, 2) if change_pct is not None else None

    if len(baseline) < MIN_SAMPLES or len(candidate) < MIN_SAMPLES:
        needed = required_samples(baseline or candidate, abs(change_pct or 5.0))
        result.update({
            "conclusive": False, "significant": None, "p": None,
            "verdict": (
                f"INCONCLUSIVE — {len(baseline)} vs {len(candidate)} samples "
                f"is too few to distinguish a real difference from noise "
                f"(need at least {MIN_SAMPLES} each"
                + (f", ideally {needed}" if needed else "") + ")"),
        })
        return result

    test = mann_whitney(baseline, candidate)
    delta = cliffs_delta(baseline, candidate)
    result.update({
        "p": round(test["p"], 5) if test.get("p") is not None else None,
        "u": test.get("u"),
        "cliffs_delta": round(delta, 3) if delta is not None else None,
        "effect_size": _delta_label(delta) if delta is not None else None,
        "conclusive": True,
    })

    significant = (test.get("p") is not None and test["p"] < alpha)
    result["significant"] = significant

    direction = "faster" if (change_pct or 0) > 0 else "slower"
    if not higher_is_better:
        direction = "slower" if (change_pct or 0) > 0 else "faster"

    if not significant:
        needed = required_samples(baseline, max(abs(change_pct or 0.0), 1.0))
        result["verdict"] = (
            f"NO SIGNIFICANT DIFFERENCE — the {abs(change_pct or 0):.1f}% gap "
            f"is within run-to-run noise (p={test['p']:.3f}). "
            + (f"Resolving a difference this small would take about {needed} "
               f"repeats per side." if needed else ""))
    elif abs(change_pct or 0) < NEGLIGIBLE_PCT:
        result["verdict"] = (
            f"SIGNIFICANT BUT NEGLIGIBLE — the difference is real "
            f"(p={test['p']:.3f}) but only {abs(change_pct or 0):.2f}%, which "
            f"is unlikely to matter in practice")
    else:
        result["verdict"] = (
            f"SIGNIFICANT — candidate is {abs(change_pct or 0):.1f}% "
            f"{direction} (p={test['p']:.3f}, {result['effect_size']} effect)")
    return result


def summary(samples: list[float]) -> dict:
    """Median, spread, and confidence interval for one set of samples."""
    if not samples:
        return {"n": 0, "median": 0.0, "mean": 0.0, "cv": 0.0,
                "ci_low": 0.0, "ci_high": 0.0}
    ci = confidence_interval(samples)
    mean = ci["mean"]
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else 0.0,
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "relative_margin_pct": (round(ci["relative_margin_pct"], 2)
                                if ci["relative_margin_pct"] is not None
                                else None),
    }


def render_comparison(results: list[dict]) -> str:
    """Terminal table for an A/B run."""
    if not results:
        return "  nothing to compare"
    lines = [f"  {'METRIC':<20} {'BASELINE':>12} {'CANDIDATE':>12} "
             f"{'CHANGE':>9}  VERDICT"]
    lines.append("  " + "-" * 86)
    for r in results:
        change = (f"{r['change_pct']:+.1f}%" if r.get("change_pct") is not None
                  else "n/a")
        mark = ("=" if not r.get("conclusive")
                else ("*" if r.get("significant") else " "))
        lines.append(
            f"  {r['metric'][:20]:<20} {r['baseline']['median']:>12,.1f} "
            f"{r['candidate']['median']:>12,.1f} {change:>9} {mark} "
            f"{r['verdict'].split(' — ')[0]}")
    lines.append("")
    lines.append("  * = statistically significant   = = too few samples to say")
    for r in results:
        lines.append(f"    {r['metric']}: {r['verdict']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Payload-level A/B
# --------------------------------------------------------------------------- #
#: Metrics where a *lower* number is better, so a "significant improvement"
#: verdict points the right way.
LOWER_IS_BETTER = {
    "latency", "syscall_ns", "context_switch_us", "process_spawn_ms",
    "fsync_median_us", "droop_pct", "sustained_droop_pct",
}


def _samples_from(entry: dict) -> list[float]:
    """Per-repeat samples if the benchmark recorded them, else the headline."""
    if not isinstance(entry, dict):
        return []
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        return [float(s) for s in samples
                if isinstance(s, (int, float))]
    for field in ("rate", "read_rate", "write_rate"):
        value = entry.get(field)
        if isinstance(value, (int, float)) and value > 0:
            return [float(value)]
    return []


def collect_samples(payload: dict) -> dict:
    """Map every benchmark in a payload to its per-repeat samples."""
    out: dict[str, list[float]] = {}
    for name, entry in (payload.get("results") or {}).items():
        if not isinstance(entry, dict) or entry.get("skipped"):
            continue
        if entry.get("error"):
            continue
        samples = _samples_from(entry)
        if samples:
            out[name] = samples
    return out


def compare_payloads(baseline: dict, candidate: dict,
                     alpha: float = 0.05) -> dict:
    """Statistically compare two saved runs, metric by metric.

    Comparability is checked first and reported as a warning rather than an
    error. Two runs on different machines, or with different durations, *can*
    be compared — sometimes that is exactly the intent — but the difference
    will not mean what a like-for-like comparison means, and saying so is the
    difference between a useful tool and a misleading one.
    """
    warnings: list[str] = []
    a_sys = baseline.get("system") or {}
    b_sys = candidate.get("system") or {}
    if a_sys.get("cpu_model") != b_sys.get("cpu_model"):
        warnings.append(
            f"different CPUs ({a_sys.get('cpu_model')} vs "
            f"{b_sys.get('cpu_model')}) — this is a hardware comparison, not "
            f"a change comparison")
    if a_sys.get("hostname") != b_sys.get("hostname"):
        warnings.append(f"different machines ({a_sys.get('hostname')} vs "
                        f"{b_sys.get('hostname')})")

    a_cfg = baseline.get("config") or {}
    b_cfg = candidate.get("config") or {}
    for key in ("seconds", "repeats", "disk_mb", "mem_mb"):
        if a_cfg.get(key) != b_cfg.get(key):
            warnings.append(
                f"different --{key.replace('_', '-')} "
                f"({a_cfg.get(key)} vs {b_cfg.get(key)}) — results are not "
                f"directly comparable")

    for side, payload in (("baseline", baseline), ("candidate", candidate)):
        confinement = payload.get("confinement") or {}
        if confinement.get("constrained"):
            warnings.append(f"the {side} run was resource-constrained "
                            f"({confinement.get('container') or 'cgroup'})")

    a_samples = collect_samples(baseline)
    b_samples = collect_samples(candidate)
    shared = sorted(set(a_samples) & set(b_samples))

    comparisons = [
        compare(a_samples[name], b_samples[name], alpha=alpha, label=name,
                higher_is_better=name not in LOWER_IS_BETTER)
        for name in shared
    ]

    significant = [c for c in comparisons if c.get("significant")]
    regressions = [c for c in significant
                   if (c.get("change_pct") or 0) < 0
                   and c["metric"] not in LOWER_IS_BETTER]
    improvements = [c for c in significant if c not in regressions]

    only_a = sorted(set(a_samples) - set(b_samples))
    only_b = sorted(set(b_samples) - set(a_samples))

    return {
        "comparisons": comparisons,
        "compared": len(shared),
        "significant": len(significant),
        "regressions": [c["metric"] for c in regressions],
        "improvements": [c["metric"] for c in improvements],
        "only_in_baseline": only_a,
        "only_in_candidate": only_b,
        "warnings": warnings,
        "composite": _composite_delta(baseline, candidate),
        "verdict": _overall_verdict(comparisons, regressions, improvements),
    }


def _composite_delta(baseline: dict, candidate: dict) -> dict | None:
    a = ((baseline.get("scores") or {}).get("composite"))
    b = ((candidate.get("scores") or {}).get("composite"))
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and a):
        return None
    return {"baseline": a, "candidate": b,
            "change_pct": round(100.0 * (b - a) / a, 2),
            "note": ("the composite is a single number per run, so it has no "
                     "samples to test; treat it as descriptive and read the "
                     "per-metric verdicts for evidence")}


def _overall_verdict(comparisons: list[dict], regressions: list[dict],
                     improvements: list[dict]) -> str:
    if not comparisons:
        return ("no metric appears in both runs — check that the two runs "
                "used the same tests")
    inconclusive = [c for c in comparisons if not c.get("conclusive")]
    if regressions:
        names = ", ".join(c["metric"] for c in regressions[:4])
        return (f"REGRESSION — {len(regressions)} metric(s) got significantly "
                f"worse: {names}"
                + (" and others" if len(regressions) > 4 else ""))
    if improvements:
        names = ", ".join(c["metric"] for c in improvements[:4])
        return (f"IMPROVEMENT — {len(improvements)} metric(s) got "
                f"significantly better: {names}"
                + (" and others" if len(improvements) > 4 else ""))
    if len(inconclusive) == len(comparisons):
        return ("INCONCLUSIVE — every metric had too few repeats to test. "
                "Re-run both sides with --repeats 5 or more.")
    return (f"NO SIGNIFICANT DIFFERENCE across {len(comparisons)} metric(s) — "
            f"any gaps are within run-to-run noise")


def render_payload_comparison(result: dict) -> str:
    """Full terminal report for an A/B comparison."""
    lines: list[str] = []
    for warning in result.get("warnings", []):
        lines.append(f"  !  {warning}")
    if result.get("warnings"):
        lines.append("")

    lines.append(render_comparison(result.get("comparisons", [])))

    composite = result.get("composite")
    if composite:
        lines.append("")
        lines.append(f"  Composite: {composite['baseline']} -> "
                     f"{composite['candidate']} "
                     f"({composite['change_pct']:+.1f}%)")
        lines.append(f"      {composite['note']}")

    if result.get("only_in_baseline") or result.get("only_in_candidate"):
        lines.append("")
        if result.get("only_in_baseline"):
            lines.append(f"  Only in baseline : "
                         f"{', '.join(result['only_in_baseline'])}")
        if result.get("only_in_candidate"):
            lines.append(f"  Only in candidate: "
                         f"{', '.join(result['only_in_candidate'])}")

    lines.append("")
    lines.append(f"  {result.get('verdict', '')}")
    return "\n".join(lines)
