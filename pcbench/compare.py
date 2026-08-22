"""Cross-device comparison from the accumulated CSV history.

Every run appends a row to ``benchmarks.csv``; this module reads that history
back so a fleet of machines can be ranked side by side, which is the point of
collecting the data in the first place.
"""

from __future__ import annotations

import csv
import os

# Columns rendered in the comparison table: (csv field, header, format)
_COLUMNS = [
    ("composite_score", "Score", "{:.0f}"),
    ("cpu_int_primes_s", "CPU int", "{:,.0f}"),
    ("cpu_multi_primes_s", "CPU multi", "{:,.0f}"),
    ("hashing_mb_s", "SHA256", "{:,.0f}"),
    ("mem_mb_s", "Mem MB/s", "{:,.0f}"),
    ("disk_write_mb_s", "Disk W", "{:,.0f}"),
    ("disk_iops", "IOPS", "{:,.0f}"),
    ("gpu_matmul_fp16_tflops", "matmul TF", "{:,.1f}"),
    ("npu_gflops", "NPU GF", "{:,.0f}"),
    ("ml_train_samples_s", "train s/s", "{:,.0f}"),
    ("score_per_watt", "score/W", "{:,.1f}"),
    ("cpu_celsius", "CPU °C", "{:,.0f}"),
]


def load_history(csv_path: str) -> list[dict]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def latest_per_host(rows: list[dict]) -> list[dict]:
    """Keep only the most recent run for each hostname.

    Rows are timestamped in UTC ISO-8601, which sorts correctly as text.
    """
    best: dict[str, dict] = {}
    for row in rows:
        host = row.get("hostname", "?")
        if host not in best or row.get("timestamp_utc", "") > best[host].get(
                "timestamp_utc", ""):
            best[host] = row
    return sorted(best.values(),
                  key=lambda r: _to_float(r.get("composite_score")),
                  reverse=True)


# Metrics that are pure-Python loops, and so reflect the interpreter as well
# as the hardware. Comparing these across CPython versions is not valid.
_INTERPRETER_BOUND = {
    "cpu_int_primes_s", "cpu_multi_primes_s", "nn_train_steps_s",
    "kmeans_dist_s", "knn_cmp_s", "composite_score",
}

# Two machines whose composite scores differ by less than this are treated as
# indistinguishable. Benchmarks carry a few percent of run-to-run noise, so a
# smaller gap is not evidence of a real difference.
SIGNIFICANCE_THRESHOLD = 0.05


def _interpreter_warning(entries: list[dict], shown: list[str]) -> str | None:
    """Warn when compared runs used different Python versions.

    Names the specific columns affected, since the hardware-bound ones
    (SHA-256, disk, BLAS, GPU) remain perfectly valid across versions — only
    the pure-Python loops do not.
    """
    versions = {r.get("python_version", "") for r in entries
                if r.get("python_version")}
    if len(versions) <= 1:
        return None
    affected = [c for c in shown if c in _INTERPRETER_BOUND]
    listed = ", ".join(sorted(versions))
    text = (f"  !  Runs used different Python versions ({listed}). "
            f"Pure-Python benchmarks\n     depend on the interpreter, which "
            f"differs by tens of percent between\n     CPython releases.")
    if affected:
        text += (f"\n     Affected here: {', '.join(affected)}."
                 f"\n     Hardware-bound columns are unaffected.")
    return text


def _spread_by_host(rows: list[dict]) -> dict:
    """Observed composite-score spread per host, from repeated runs.

    A machine's own run-to-run variation is the fairest yardstick for deciding
    whether a gap to another machine is meaningful.
    """
    import statistics
    from collections import defaultdict
    scores = defaultdict(list)
    for r in rows:
        v = _to_float(r.get("composite_score"))
        if v:
            scores[r.get("hostname", "?")].append(v)
    out = {}
    for host, values in scores.items():
        if len(values) >= 2:
            mean = statistics.fmean(values)
            if mean:
                out[host] = statistics.stdev(values) / mean
    return out


def render_table(rows: list[dict], all_runs: bool = False) -> str:
    """Render a ranked comparison table as text."""
    if not rows:
        return ("No history found. Run the benchmark at least once (results "
                "are appended to results/benchmarks.csv).")

    entries = rows if all_runs else latest_per_host(rows)
    if all_runs:
        entries = sorted(entries,
                         key=lambda r: _to_float(r.get("composite_score")),
                         reverse=True)

    present = [c for c in _COLUMNS
               if any(_to_float(r.get(c[0])) for r in entries)]

    label_w = max(len("Machine"),
                  max(len(_label(r)) for r in entries))
    widths = [max(len(h), 9) for _, h, _ in present]

    head = "  " + "Machine".ljust(label_w)
    head += "".join("  " + h.rjust(w) for (_, h, _), w in zip(present, widths))
    lines = [head, "  " + "-" * (len(head) - 2)]

    top = _to_float(entries[0].get("composite_score")) if entries else 0.0
    for rank, row in enumerate(entries, 1):
        line = "  " + _label(row).ljust(label_w)
        for (field, _, fmt), w in zip(present, widths):
            val = _to_float(row.get(field))
            line += "  " + (fmt.format(val) if val else "-").rjust(w)
        if rank > 1 and top:
            rel = _to_float(row.get("composite_score")) / top * 100
            line += f"   ({rel:.0f}% of best)"
        elif rank == 1:
            line += "   (best)"
        lines.append(line)

    lines.append("")
    lines.append(f"  {len(entries)} machine(s). "
                 f"{'All runs' if all_runs else 'Latest run per host'}.")

    # Say plainly when a ranking gap is too small to mean anything.
    spread = _spread_by_host(rows)
    if len(entries) >= 2 and top:
        second = _to_float(entries[1].get("composite_score"))
        if second:
            gap = abs(top - second) / top
            noise = max(spread.get(_host(entries[0]), 0.0),
                        spread.get(_host(entries[1]), 0.0),
                        SIGNIFICANCE_THRESHOLD)
            if gap < noise:
                lines.append(
                    f"  Note: the top two differ by {gap * 100:.1f}%, within "
                    f"the {noise * 100:.0f}% run-to-run noise — "
                    f"treat them as equivalent.")

    warning = _interpreter_warning(entries, [c[0] for c in present])
    if warning:
        lines.append("")
        lines.append(warning)
    return "\n".join(lines)


def _host(row: dict) -> str:
    return row.get("hostname", "?")


def _label(row: dict) -> str:
    host = row.get("hostname") or "?"
    arch = row.get("arch_family") or row.get("arch") or ""
    cpu = (row.get("cpu_model") or "").strip()
    if len(cpu) > 28:
        cpu = cpu[:27] + "…"
    bits = [host]
    if cpu:
        bits.append(f"({cpu})")
    elif arch:
        bits.append(f"({arch})")
    return " ".join(bits)
