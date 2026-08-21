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
    ("gpu_fp32_gflops", "GPU GF", "{:,.0f}"),
    ("npu_gflops", "NPU GF", "{:,.0f}"),
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
    return "\n".join(lines)


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
