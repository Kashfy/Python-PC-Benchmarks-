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

from .scoring import category_scores

# A category this far below the machine's own median is called a bottleneck.
# The comparison is internal on purpose: it answers "what is weak *for this
# machine*", which is what determines whether an upgrade would help.
BOTTLENECK_RATIO = 0.6
STRENGTH_RATIO = 1.5

# "ai" is a roll-up of gpu, npu and ml rather than an independent subsystem.
# Including it would double-count those and distort the median every other
# category is judged against.
DERIVED_CATEGORIES = {"ai"}

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
                         "impact": _ADVICE.get(k, "")}
                        for k, v in weak],
        "strengths": [{"category": k, "score": v,
                       "relative": round(v / median, 2)} for k, v in strong],
        "weakest": {"category": slowest[0], "score": slowest[1]},
        "strongest": {"category": fastest[0], "score": fastest[1]},
        "verdict": _verdict(cats, weak, median, fastest, slowest),
    }


def _verdict(cats: dict, weak: list, median: float,
             fastest: tuple, slowest: tuple) -> str:
    if not weak:
        ratio = fastest[1] / slowest[1] if slowest[1] else 1
        if ratio < 2:
            return ("well balanced — no subsystem is holding the others back")
        return (f"reasonably balanced; {slowest[0]} is the weakest area but "
                f"not severely so")
    names = ", ".join(w[0] for w in weak)
    return (f"{names} {'is' if len(weak) == 1 else 'are'} well below this "
            f"machine's own average and will limit overall performance")


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
