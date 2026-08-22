"""Reference classes: interpreting one run with no history to compare against.

The composite score is a number relative to a fixed baseline, which makes runs
comparable to each other but tells a first-time user nothing. "412" answers
neither of the questions people actually have:

1. **Is this normal for hardware like mine?** A 16-core workstation scoring
   what a thin-and-light laptop scores is not a good result — it is a broken
   one, and the score alone will never say so.
2. **What class of machine is this?** Useful when specs are unknown or lied
   about: refurbished hardware, a cloud instance type, or a "gaming laptop"
   that turns out to be throttled to ultrabook performance.

So this module does two things. It places a measured score into a named class,
and — more usefully — it *predicts* what the machine should score from its own
inventory (core count, memory, storage type, architecture) and flags a large
shortfall as a fault to investigate.

The expectation model is deliberately coarse. It exists to separate "roughly as
expected" from "dramatically wrong", which is a distinction that survives crude
modelling; it is not a spec-sheet lookup and does not pretend to be.
"""

from __future__ import annotations

#: Named performance classes by composite score. Boundaries are approximate
#: and chosen where the *use case* changes, not at round numbers.
CLASSES = [
    (0, 40, "embedded / SBC",
     "Raspberry Pi class. Fine for sensors, automation, and light services; "
     "compiling or running an IDE on it will be painful."),
    (40, 90, "entry / thin client",
     "Low-power laptop or older desktop. Comfortable for browsing and "
     "documents; slow for builds, media work, or virtualisation."),
    (90, 180, "mainstream laptop",
     "Typical current-generation notebook. Handles development, office work, "
     "and light content creation without complaint."),
    (180, 350, "performance laptop / mainstream desktop",
     "Fast enough that the machine is rarely the bottleneck for everyday "
     "professional work, including compiling and containers."),
    (350, 700, "workstation",
     "Serious multi-core throughput. Suited to large builds, simulation, "
     "video work, and running several VMs at once."),
    (700, 10 ** 9, "high-end workstation / server",
     "Server-class or top-tier workstation throughput. Expect this from many "
     "cores, fast storage, and generous memory bandwidth."),
]


def classify(composite: float) -> dict:
    """Name the performance class a composite score falls into."""
    for low, high, name, description in CLASSES:
        if low <= composite < high:
            return {"class": name, "description": description,
                    "range": [low, None if high >= 10 ** 9 else high],
                    "composite": composite}
    return {"class": "unknown", "description": "", "range": None,
            "composite": composite}


# --------------------------------------------------------------------------- #
# Balance model
# --------------------------------------------------------------------------- #
# An earlier version of this module tried to predict the composite from the
# architecture and core count. That does not work, and cannot: "ARM64" covers
# both an Apple M-series chip and a Raspberry Pi, whose per-core throughput
# differs by more than 10x. Any prediction built on the ISA name alone flags
# every single-board computer as broken.
#
# So the expectation is anchored on something the run actually measured: the
# machine's own single-threaded performance. That is a real property of this
# silicon, and it makes the question answerable — not "how fast should this
# chip be?" (unknowable without a hardware database) but "given how fast one
# core of this chip is, do the rest of the subsystems keep up?" A machine whose
# composite sits far below its own single-core score has a specific weak
# subsystem; one far above is being carried by an accelerator.

#: Subscores produced by one thread on one core, with no storage or
#: accelerator involvement. Their geometric mean is the per-core anchor.
_SINGLE_THREAD_KEYS = ("cpu_int", "cpu_float", "compression", "hashing",
                       "json", "raytrace", "image", "logparse")

#: How far the composite may sit from the single-thread anchor before the
#: imbalance is worth reporting. Wide, because a machine legitimately gains
#: from many cores and fast storage, or legitimately lacks both.
_BALANCE_LOW = 0.45
_BALANCE_HIGH = 2.5


def single_thread_anchor(subscores: dict) -> float | None:
    """Geometric mean of the purely single-threaded subscores."""
    import math
    import statistics

    values = [subscores[k] for k in _SINGLE_THREAD_KEYS
              if isinstance(subscores.get(k), (int, float)) and subscores[k] > 0]
    if len(values) < 2:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in values))


def expected(info: dict, subscores: dict | None = None) -> dict:
    """Plausible composite band for this machine, anchored on its own cores.

    Returns an empty band when there is nothing to anchor on, rather than
    guessing. A missing assessment is far better than a confident wrong one.
    """
    anchor = single_thread_anchor(subscores or {})
    cores = info.get("cpu_cores_logical") or 1
    physical = info.get("cpu_cores_physical") or cores
    ram_gb = info.get("ram_total_gb") or 0

    if anchor is None:
        return {"expected_composite": None, "low": None, "high": None,
                "basis": "not enough single-threaded results to anchor on",
                "anchor": None}

    # Most subscores in the composite are per-core, so core count lifts it only
    # mildly; the exponent is small on purpose.
    estimate = anchor * (max(1.0, physical) ** 0.12)
    if ram_gb and ram_gb < 4:
        estimate *= 0.75          # paging suppresses everything below ~4 GB

    return {
        "expected_composite": round(estimate, 1),
        "low": round(estimate * _BALANCE_LOW, 1),
        "high": round(estimate * _BALANCE_HIGH, 1),
        "anchor": round(anchor, 1),
        "basis": (f"a single-core score of {anchor:.0f} across "
                  f"{physical} physical / {cores} logical cores"),
    }


def assess(payload: dict) -> dict:
    """Place a run in a class and check it against its own single-core anchor.

    The balance check is the diagnostic half. A composite far below what this
    machine's own cores can do means a specific subsystem is dragging — and
    since the anchor comes from the same silicon, the comparison holds equally
    on a Raspberry Pi and on a 96-core server.
    """
    info = payload.get("system") or {}
    scores = payload.get("scores") or {}
    subscores = scores.get("subscores") or {}
    composite = scores.get("composite") or 0.0

    placement = classify(composite)
    model = expected(info, subscores)
    result = {**placement, **model}

    if not composite:
        result["verdict"] = "no composite score was produced"
        return result
    if model["expected_composite"] is None:
        result["verdict"] = (
            f"performance class: {placement['class']}. Too few single-threaded "
            f"tests ran to assess whether the machine is balanced.")
        return result

    ratio = composite / model["expected_composite"]
    result["ratio_to_expected"] = round(ratio, 2)

    if composite < model["low"]:
        result["flag"] = "unbalanced (subsystem drag)"
        result["verdict"] = (
            f"composite {composite:.0f} is well below what this machine's own "
            f"cores suggest ({model['expected_composite']:.0f} from "
            f"{model['basis']}). One subsystem is dragging the rest down — "
            f"see the bottleneck section, and check storage, memory "
            f"configuration, and thermal throttling.")
    elif composite > model["high"]:
        result["flag"] = "accelerator-led"
        result["verdict"] = (
            f"composite {composite:.0f} is well above this machine's "
            f"single-core anchor ({model['expected_composite']:.0f}) — a GPU, "
            f"NPU, or very fast storage is carrying the composite rather than "
            f"the CPU cores.")
    else:
        result["flag"] = "balanced"
        result["verdict"] = (
            f"composite {composite:.0f}, balanced against this machine's own "
            f"cores. Performance class: {placement['class']}.")
    return result


# --------------------------------------------------------------------------- #
# Per-category sanity checks
# --------------------------------------------------------------------------- #
#: Absolute floors below which a subsystem is suspect regardless of class.
#: These are set where the hardware itself is implausible, not merely slow, so
#: that firing one always means "go look at this".
FLOORS = {
    "disk_read": (80.0, "MB/s",
                  "sequential read this low means a hard disk, a failing SSD, "
                  "a USB 2.0 enclosure, or a filesystem in a degraded state"),
    "disk_write": (40.0, "MB/s",
                   "sequential write this low suggests a full or failing "
                   "drive, SMR media, or an exhausted SSD write cache"),
    "disk_iops": (200.0, "IOPS",
                  "random read IOPS in this range is rotational-media "
                  "territory; on an SSD it indicates a real fault"),
    "memory": (800.0, "MB/s",
               "memory bandwidth this low points at single-channel operation, "
               "a downclocked module, or heavy contention"),
}


def subsystem_checks(results: dict) -> list[dict]:
    """Flag subsystems whose absolute figures are implausible for the class."""
    out = []
    sources = {
        "disk_read": ("disk", "read_rate"),
        "disk_write": ("disk", "write_rate"),
        "disk_iops": ("disk", "random_read_iops"),
        "memory": ("memory", "rate"),
    }
    for key, (result_key, field) in sources.items():
        entry = results.get(result_key)
        if not isinstance(entry, dict) or entry.get("skipped"):
            continue
        value = entry.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        floor, unit, explanation = FLOORS[key]
        if value < floor:
            out.append({"metric": key, "value": round(value, 1), "unit": unit,
                        "floor": floor, "note": explanation})
    return out


def render(assessment: dict, checks: list[dict] | None = None) -> str:
    """Terminal block placing the machine and listing any suspect subsystem."""
    lines = [f"  Class      : {assessment.get('class', 'unknown')}"]
    if assessment.get("description"):
        lines.append(f"               {assessment['description']}")
    if assessment.get("expected_composite"):
        lines.append(f"  Balance    : ~{assessment['expected_composite']} "
                     f"expected (range {assessment['low']}–"
                     f"{assessment['high']}) from {assessment['basis']}")
    if assessment.get("verdict"):
        lines.append(f"  Assessment : {assessment['verdict']}")
    for check in checks or []:
        lines.append(f"  !  {check['metric']} = {check['value']} "
                     f"{check['unit']} (below {check['floor']}): "
                     f"{check['note']}")
    return "\n".join(lines)
