"""Turn measurements into conclusions: bottleneck analysis and a spec sheet.

The tool collects a great many numbers and, until now, left the reader to
decide what they meant. Most people running a benchmark want one of two
answers: *what is holding this machine back?* and *what are its specs?* Both
are derivable from data already gathered.

Bottleneck analysis works on the normalised subscores rather than raw rates,
because those are the only figures comparable across subsystems — 4 million
primes per second and 500 MB/s are otherwise not on the same axis.
"""

from __future__ import annotations

from .scoring import CATEGORY_GROUPS, category_scores

# A category this far below the machine's own median is called a bottleneck.
# The comparison is internal on purpose: it answers "what is weak *for this
# machine*", which is what determines whether an upgrade would help.
BOTTLENECK_RATIO = 0.6
STRENGTH_RATIO = 1.5

# "ai" is a roll-up of gpu, npu and ml rather than an independent subsystem.
# Including it would double-count those and distort the median every other
# category is judged against.
DERIVED_CATEGORIES = {"ai"}

# Categories whose workloads run in pure Python, so what they measure is the
# interpreter running on the CPU rather than an independent subsystem. Their
# scores track single-core integer throughput almost exactly — measured at
# within 2-4% of `cpu_int` on both an M1 Max and an M4 — so naming one as a
# bottleneck restates the CPU result while implying a separate, fixable
# weakness. They stay in the report and in the composite (pure-Python ML speed
# is a real thing to care about if that is what you run); they are just not
# allowed to masquerade as an independent finding.
INTERPRETER_BOUND = {"ml": "cpu_int"}

# How close an interpreter-bound category must sit to its driving subscore
# before its weakness is attributed to the CPU rather than to itself.
TRACKS_TOLERANCE = 0.25

# What a weak category usually means in practice.
_ADVICE = {
    "cpu": "CPU-bound work (compiling, simulation, encoding) will be the "
           "limiting factor",
    "memory": "memory bandwidth limits large working sets — video editing, "
              "large datasets, many VMs",
    "disk": "storage is the limit: application launches, file operations and "
            "builds will feel slow regardless of CPU",
    "gpu": "graphics and GPU compute are the weak point — gaming, rendering, "
           "GPU-accelerated AI",
    "npu": "on-device AI inference is comparatively slow",
    "ml": "CPU-side machine-learning workloads are the weak point",
    "numeric": "scientific and numeric computing is limited by this machine's "
               "floating-point throughput",
    "crypto": "encryption and compression throughput is comparatively low",
    "system": "process creation and compilation are slow, which dominates "
              "build systems and shell-heavy work",
}

# A category is the geometric mean of its members, so a single weak measurement
# drags it down while every other member is fine. Naming the category alone
# then describes the wrong problem: a machine that compiles at four times the
# baseline and has slow syscalls was being told its compiler was slow. When one
# member is clearly the culprit, its own advice is reported instead.
_SUBSCORE_ADVICE = {
    "compile": "compilation is slow, which dominates build systems",
    "syscall": "system calls are expensive, which is felt in I/O-heavy and "
               "shell-heavy work — speculative-execution mitigations are the "
               "usual cause and are a security trade-off, not a fault",
    "disk_write": "sequential writes are slow, which shows up in copying, "
                  "extracting archives, and container image pulls",
    "disk_read": "sequential reads are slow, which shows up when loading "
                 "large files",
    "disk_iops": "small random reads are slow at queue depth 1, which is what "
                 "application launches and package managers issue",
    "disk_iops_peak": "the drive stops scaling with queued requests, which "
                      "limits databases and anything with concurrent I/O",
    "memory": "memory copy bandwidth is low for this machine",
    "mem_scaling": "memory bandwidth does not scale across cores, so "
                   "multi-process work will contend for it",
    "cpu_int": "single-core integer throughput is the limit",
    "cpu_multi": "multi-core throughput is the limit",
    "blas_matmul": "dense linear algebra is slow — check which BLAS is "
                   "linked, since the reference build is several times slower "
                   "than OpenBLAS or MKL",
    "aes": "bulk encryption is slow, which suggests AES is not being done in "
           "hardware",
    "stream_triad": "sustained memory bandwidth is the limit",
}

#: How much better the best member of a category must be than the worst before
#: the finding is attributed to that one member rather than to the category.
_SPLIT_RATIO = 2.0


def _weakest_member(category: str, subscores: dict) -> tuple[str, float] | None:
    """The member measurement responsible for a weak category, if there is one.

    Returns ``None`` when the members agree with each other, because then the
    category really is uniformly weak and its own advice is the right answer.
    """
    members = [(k, subscores[k]) for k in CATEGORY_GROUPS.get(category, ())
               if isinstance(subscores.get(k), (int, float)) and subscores[k] > 0]
    if len(members) < 2:
        return None
    worst = min(members, key=lambda kv: kv[1])
    best = max(members, key=lambda kv: kv[1])
    if best[1] < worst[1] * _SPLIT_RATIO:
        return None
    return worst


def tracks_cpu(category: str, score: float, subscores: dict) -> bool:
    """True when an interpreter-bound category is just restating the CPU.

    If pure-Python ML scores what single-core integer work scores, the finding
    is "this CPU core is modest", not "machine learning is slow here" — and the
    two lead to completely different actions.
    """
    driver = INTERPRETER_BOUND.get(category)
    if not driver:
        return False
    reference = subscores.get(driver)
    if not isinstance(reference, (int, float)) or reference <= 0:
        return False
    return abs(score - reference) / reference <= TRACKS_TOLERANCE


def analyse(scores: dict) -> dict:
    """Identify the weakest and strongest subsystems relative to each other."""
    subscores = scores.get("subscores") or {}
    cats = {k: v for k, v in category_scores(subscores).items()
            if k not in DERIVED_CATEGORIES}
    # A single category cannot be a bottleneck relative to itself.
    if len(cats) < 2:
        return {"available": False,
                "note": "not enough categories measured to identify a "
                        "bottleneck — run more tests"}

    values = sorted(cats.values())
    mid = len(values) // 2
    median = (values[mid] if len(values) % 2
              else (values[mid - 1] + values[mid]) / 2)
    if not median:
        return {"available": False, "note": "no usable scores"}

    weak = sorted((c for c in cats.items() if c[1] < median * BOTTLENECK_RATIO),
                  key=lambda kv: kv[1])
    strong = sorted((c for c in cats.items()
                     if c[1] > median * STRENGTH_RATIO),
                    key=lambda kv: kv[1], reverse=True)

    slowest = min(cats.items(), key=lambda kv: kv[1])
    fastest = max(cats.items(), key=lambda kv: kv[1])

    return {
        "available": True,
        "categories": cats,
        "median": round(median, 1),
        "balance_ratio": round(fastest[1] / slowest[1], 2) if slowest[1] else None,
        "bottlenecks": [{"category": k, "score": v,
                         "relative": round(v / median, 2),
                         "impact": _impact(k, v, subscores),
                         "restates_cpu": tracks_cpu(k, v, subscores)}
                        for k, v in weak],
        "strengths": [{"category": k, "score": v,
                       "relative": round(v / median, 2)} for k, v in strong],
        "weakest": {"category": slowest[0], "score": slowest[1]},
        "strongest": {"category": fastest[0], "score": fastest[1]},
        "verdict": _verdict(cats, weak, median, fastest, slowest,
                            subscores),
    }


def _impact(category: str, score: float, subscores: dict) -> str:
    """What a weak category means, corrected for interpreter-bound ones."""
    if tracks_cpu(category, score, subscores):
        driver = INTERPRETER_BOUND[category]
        return (f"these workloads run in pure Python, and they score what "
                f"{driver} scores — so this reflects single-core CPU "
                f"throughput, not a separate weakness. Installing NumPy or "
                f"PyTorch bypasses the interpreter and changes the picture "
                f"entirely")

    culprit = _weakest_member(category, subscores)
    if culprit and culprit[0] in _SUBSCORE_ADVICE:
        key, value = culprit
        others = [(k, v) for k, v in
                  ((k, subscores.get(k)) for k in CATEGORY_GROUPS[category])
                  if isinstance(v, (int, float)) and v > 0 and k != key]
        detail = ""
        if others:
            best = max(others, key=lambda kv: kv[1])
            detail = (f" — {key} scores {value:.0f} while {best[0]} scores "
                      f"{best[1]:.0f}, so the category average understates "
                      f"everything except {key}")
        return _SUBSCORE_ADVICE[key] + detail
    return _ADVICE.get(category, "")


def _verdict(cats: dict, weak: list, median: float,
             fastest: tuple, slowest: tuple, subscores: dict) -> str:
    if not weak:
        ratio = fastest[1] / slowest[1] if slowest[1] else 1
        if ratio < 2:
            return ("well balanced — no subsystem is holding the others back")
        return (f"reasonably balanced; {slowest[0]} is the weakest area but "
                f"not severely so")

    # A category that merely restates the CPU result is not an independent
    # finding, so it must not be the headline when something else is.
    independent = [w for w in weak
                   if not tracks_cpu(w[0], w[1], subscores)]
    if not independent:
        names = ", ".join(w[0] for w in weak)
        return (f"{names} scores lowest, but these are pure-Python workloads "
                f"that track single-core CPU throughput — the finding is that "
                f"this machine's cores are modest, not that a separate "
                f"subsystem is weak")

    names = ", ".join(w[0] for w in independent)
    verdict = (f"{names} {'is' if len(independent) == 1 else 'are'} well below "
               f"this machine's own average and will limit overall performance")
    dependent = [w[0] for w in weak if w not in independent]
    if dependent:
        verdict += (f" ({', '.join(dependent)} also scores low, but only "
                    f"because it re-measures the CPU through the interpreter)")
    return verdict


def render(result: dict) -> str:
    """Human-readable bottleneck summary."""
    if not result.get("available"):
        return "  " + result.get("note", "unavailable")

    lines = []
    cats = result["categories"]
    peak = max(cats.values()) or 1
    for name, score in sorted(cats.items(), key=lambda kv: kv[1],
                              reverse=True):
        bar = "█" * max(1, int(score / peak * 30))
        mark = ""
        if any(b["category"] == name for b in result["bottlenecks"]):
            mark = "  ← bottleneck"
        elif name == result["strongest"]["category"]:
            mark = "  ← strongest"
        lines.append(f"    {name.upper():<9} {bar:<30} {score:>7.1f}{mark}")

    lines.append("")
    lines.append(f"  Verdict: {result['verdict']}")
    for b in result["bottlenecks"]:
        if b["impact"]:
            lines.append(f"    • {b['category']}: {b['impact']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Spec sheet
# --------------------------------------------------------------------------- #
def spec_sheet(payload: dict) -> str:
    """A one-page Markdown summary, for a support ticket or a listing."""
    info = payload.get("system", {})
    state = payload.get("state", {})
    scores = payload.get("scores", {})
    results = payload.get("results", {})
    accel = payload.get("accelerators", {}) or {}

    def rate(key, field="rate"):
        entry = results.get(key)
        if isinstance(entry, dict) and isinstance(entry.get(field),
                                                  (int, float)):
            return entry[field]
        return None

    lines = [
        f"# {info.get('cpu_model', 'Unknown machine')}",
        "",
        f"*Generated by pcbench {payload.get('version', '')} on "
        f"{payload.get('timestamp_utc', '')}*",
        "",
        "## Hardware",
        "",
        "| | |",
        "|---|---|",
        f"| CPU | {info.get('cpu_model', '?')} |",
        f"| Architecture | {info.get('arch_family', '?')} "
        f"({info.get('architecture', '?')}, {info.get('arch_bits', '?')}-bit) |",
        f"| Cores | {info.get('cpu_cores_physical') or '?'} physical / "
        f"{info.get('cpu_cores_logical', '?')} logical |",
        f"| Memory | {info.get('ram_total_gb', '?')} GB |",
        f"| OS | {info.get('os', '?')} {info.get('os_release', '')} |",
    ]
    if info.get("cpu_features"):
        lines.append(f"| CPU features | {', '.join(info['cpu_features'])} |")
    for gpu in accel.get("gpus", [])[:2]:
        lines.append(f"| GPU | {gpu.get('name', '?')} |")
    for npu in accel.get("npus", [])[:2]:
        lines.append(f"| NPU | {npu.get('name', '?')} |")
    batt = state.get("battery") or {}
    if batt.get("health_percent") is not None:
        lines.append(f"| Battery health | {batt['health_percent']:.1f}% of "
                     f"design, {batt.get('cycle_count', '?')} cycles |")
    if state.get("cpu_celsius") is not None:
        lines.append(f"| Temperature at test | "
                     f"{state['cpu_celsius']:.1f} °C |")

    lines += ["", "## Performance", "",
              f"**Composite score: {scores.get('composite', 0):.0f}** "
              f"(baseline machine = 100)", "", "| Measurement | Result |",
              "|---|---|"]

    for key, label, unit, field in [
            ("cpu_int", "CPU integer", "primes/s", "rate"),
            ("cpu_multi", "CPU multi-core", "primes/s", "rate"),
            ("hashing", "SHA-256", "MB/s", "rate"),
            ("blas_matmul", "BLAS matmul FP64", "GFLOPS", "rate"),
            ("memory", "Memory bandwidth", "MB/s", "rate"),
            ("disk", "Disk write", "MB/s", "write_rate"),
            ("disk", "Disk random read", "IOPS", "peak_iops"),
            ("compile", "C compile", "per minute", "rate")]:
        value = rate(key, field)
        if value:
            lines.append(f"| {label} | {value:,.0f} {unit} |")

    cats = category_scores(scores.get("subscores") or {})
    if cats:
        lines += ["", "## Subsystem scores", "", "| Subsystem | Score |",
                  "|---|---|"]
        for name, score in sorted(cats.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name.upper()} | {score:.0f} |")

    bottleneck = analyse(scores)
    if bottleneck.get("available"):
        lines += ["", "## Assessment", "", bottleneck["verdict"] + "."]

    return "\n".join(lines) + "\n"
