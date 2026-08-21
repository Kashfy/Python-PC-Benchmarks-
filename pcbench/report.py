"""Rendering and persistence of results: console, JSON, CSV, and HTML."""

from __future__ import annotations

import csv
import html
import json
import os
import re

from .core import stability_note
from .regression import render as regression_render
from .scoring import category_scores
from .sustained import sparkline

WIDTH = 74


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def hr(title: str = "") -> None:
    line = "=" * WIDTH
    print(f"\n{line}\n{title}\n{line}" if title else line)


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def _row(label: str, value: str, suffix: str = "") -> None:
    print(f"  {label:<26}: {value:>14}{suffix}")


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def print_system(info: dict, state: dict, warnings: list[str]) -> None:
    hr("System Information")
    _kv("Hostname", info["hostname"])
    _kv("OS", f"{info['os']} {info['os_release']}")
    _kv("Architecture", f"{info['arch_family']} ({info['architecture']}, "
                        f"{info['arch_bits']}-bit, {info['byte_order']}-endian)")
    _kv("CPU", info["cpu_model"])
    cores = f"{info['cpu_cores_physical'] or '?'} physical / " \
            f"{info['cpu_cores_logical']} logical"
    if info.get("cpu_base_mhz"):
        cores += f"  @ {info['cpu_base_mhz']:.0f} MHz"
    _kv("Cores", cores)
    _kv("RAM", f"{info['ram_total_gb']} GB" if info.get("ram_total_gb")
        else "unknown")
    py = f"{info['python_implementation']} {info['python_version']}"
    if info.get("free_threaded_build"):
        py += "  [free-threaded, GIL " + \
              ("enabled" if info.get("gil_enabled") else "disabled") + "]"
    _kv("Python", py)

    power = state.get("on_ac_power")
    _kv("Power", "AC" if power else ("battery" if power is False else "unknown"))
    if state.get("load_average"):
        _kv("Load average", ", ".join(f"{x:.2f}"
                                      for x in state["load_average"]))
    if state.get("thermal"):
        _kv("Thermal", state["thermal"])

    if warnings:
        print()
        for w in warnings:
            print(f"  !!  {w}")


def _kv(label: str, value) -> None:
    print(f"  {label:<14}: {value}")


def print_results(results: dict) -> None:
    hr("Benchmark Results")
    labels = [
        ("cpu_int", "CPU Integer (primes)", "rate", "primes/s"),
        ("cpu_float", "CPU Float (math ops)", "rate", "iters/s"),
        ("compression", "Compression (zlib)", "rate", "MB/s"),
        ("hashing", "Hashing (SHA-256)", "rate", "MB/s"),
        ("json", "JSON parse", "rate", "MB/s"),
        ("memory", "Memory copy bandwidth", "rate", "MB/s"),
    ]
    for key, label, field, unit in labels:
        r = results.get(key)
        if not isinstance(r, dict):
            continue
        if r.get("error"):
            _row(label, "FAILED", f"  {r['error'][:40]}")
            continue
        note = ""
        if "cv" in r:
            note = f"  ({stability_note(r['cv'])}, ±{r['cv'] * 100:.1f}%)"
        _row(label, fmt(r.get(field)), f" {unit}{note}")

    r = results.get("cpu_multi")
    if isinstance(r, dict) and not r.get("error"):
        single = (results.get("cpu_int") or {}).get("rate")
        scale = f"  →  {r['rate'] / single:.1f}x vs 1 core" if single else ""
        _row(f"CPU Multi-core ({r['workers']}w)", fmt(r["rate"]),
             f" primes/s{scale}")

    r = results.get("disk")
    if isinstance(r, dict):
        if r.get("skipped") or r.get("error"):
            _row("Disk I/O", "skipped", f"  {r.get('error', '')[:40]}")
        else:
            _row("Disk sequential write", fmt(r["write_rate"]), " MB/s")
            _row("Disk sequential read", fmt(r["read_rate"]), " MB/s")
            _row("Disk random read (4K)", fmt(r["random_read_iops"]), " IOPS")
            if not r.get("cache_bypassed"):
                print(f"      note: {r['note']}")

    sweep = results.get("cache_sweep")
    if isinstance(sweep, dict) and sweep.get("points"):
        print("\n  Memory bandwidth by working-set size (cache hierarchy):")
        rates = [p["mb_per_s"] for p in sweep["points"]]
        peak = max(rates) or 1
        for p in sweep["points"]:
            bar = "█" * max(1, int(p["mb_per_s"] / peak * 34))
            print(f"    {p['label']:>7}  {bar:<34} {p['mb_per_s']:>9,.0f} MB/s")
        if sweep.get("cache_to_dram_ratio"):
            print(f"    cache-to-DRAM bandwidth ratio: "
                  f"{sweep['cache_to_dram_ratio']}x")


def print_native(native: dict | None) -> None:
    if not native:
        return
    if "error" in native:
        print(f"\n  (native engine: {native['error']})")
        if native.get("detail"):
            print(f"   {native['detail'][:200]}")
        return
    hr("Native (C) Engine — compiler-optimized")
    for item in native.get("results", []):
        _row(item["name"], fmt(item.get("rate")), f" {item.get('unit', '')}")
    lat = native.get("latency")
    if lat:
        print("\n  Memory latency by working-set size (pointer chase):")
        for p in lat:
            print(f"    {p['label']:>7}  {p['ns']:>8.2f} ns")


def print_accelerators(inv: dict | None, accel: dict | None) -> None:
    """Render GPU/NPU inventory and, where available, their benchmarks."""
    if not inv and not accel:
        return
    hr("Accelerators — GPU / NPU")

    for gpu in (inv or {}).get("gpus", []):
        bits = [gpu.get("name", "unknown")]
        detail = []
        if gpu.get("cores"):
            detail.append(f"{gpu['cores']} cores")
        if gpu.get("vram_mb"):
            detail.append(f"{gpu['vram_mb'] / 1024:.1f} GB")
        if gpu.get("metal"):
            detail.append(str(gpu["metal"]))
        if gpu.get("driver"):
            detail.append(f"driver {gpu['driver']}")
        if detail:
            bits.append("(" + ", ".join(detail) + ")")
        _kv("GPU", " ".join(bits))

    for npu in (inv or {}).get("npus", []):
        label = npu.get("name", "unknown")
        if npu.get("api"):
            label += f"  (via {npu['api']})"
        _kv("NPU", label)

    if not (inv or {}).get("gpus") and not (inv or {}).get("npus"):
        _kv("Detected", "none")

    if accel and "error" in accel:
        print(f"\n  (accelerator engine: {accel['error']})")
        if accel.get("detail"):
            print(f"   {accel['detail'][:200]}")
        return
    if not accel:
        if inv and not inv.get("benchmark_supported"):
            print("\n  Compute benchmarking is Apple-only for now "
                  "(Metal / Core ML); inventory shown above.")
        return

    print()
    for item in accel.get("results", []):
        _row(item.get("name", "?"), fmt(item.get("value")),
             f" {item.get('unit', '')}")

    ane = accel.get("ane")
    if ane:
        # Core ML never reports placement, so the CPU-relative speedup is the
        # only evidence that work actually reached the Neural Engine.
        speedup = ane.get("speedup_vs_cpu") or 0
        verdict = ("Neural Engine ENGAGED" if ane.get("engaged")
                   else "Neural Engine did NOT engage")
        print(f"\n  {verdict} — {speedup:.2f}x vs CPU-only Core ML")

    for note in accel.get("notes", []):
        print(f"  note: {note}")


def print_ai(ml: dict | None) -> None:
    if not ml:
        return
    hr("AI Framework — training & inference")
    if not ml.get("available"):
        print(f"  {ml.get('note', 'no ML framework installed')}")
        return
    if ml.get("error"):
        print(f"  {ml['framework']}: {ml['error']}")
        return
    _kv("Framework", f"{ml['framework']} {ml.get('framework_version', '')}")
    _kv("Device", f"{ml.get('device_name', ml.get('device', '?'))}")
    if ml.get("model_params"):
        _kv("Model params", f"{ml['model_params']:,}")
    print()
    if ml.get("train_samples_per_s"):
        _row("Training throughput", fmt(ml["train_samples_per_s"]),
             " samples/s")
    if ml.get("infer_samples_per_s"):
        _row("Inference throughput", fmt(ml["infer_samples_per_s"]),
             " samples/s")
    if ml.get("note"):
        print(f"  note: {ml['note']}")


def print_network(net: dict | None) -> None:
    if not net or net.get("error"):
        if net and net.get("error"):
            print(f"\n  (network: {net['error']})")
        return
    hr("Network (loopback stack)")
    _row("Loopback throughput", fmt(net.get("loopback_mb_s")), " MB/s")
    lat = net.get("latency") or {}
    if lat:
        _row("Loopback latency p50", fmt(lat.get("p50_us")), " us")
        _row("Loopback latency p99", fmt(lat.get("p99_us")), " us")


def print_power(power: dict | None, ppw: dict | None) -> None:
    if not power:
        return
    hr("Power & Efficiency")
    watts = power.get("package_w")
    if watts:
        tag = " (estimated)" if power.get("estimated") else " (measured)"
        _row("Package power", fmt(watts), f" W{tag}")
        for key, label in (("cpu_w", "CPU power"), ("gpu_w", "GPU power"),
                           ("ane_w", "ANE power")):
            if power.get(key):
                _row(label, fmt(power[key]), " W")
    else:
        print(f"  Power: {power.get('source', 'unavailable')}")
    if ppw:
        _row("Perf-per-watt", fmt(ppw["score_per_watt"]),
             " score/W" + (" (est.)" if ppw.get("estimated") else ""))
    if power.get("hint") and power.get("estimated"):
        print(f"  hint: {power['hint']}")


def print_regression(reg: dict | None) -> None:
    if not reg:
        return
    hr("Regression Check")
    print(regression_render(reg))


def print_sustained(s: dict | None) -> None:
    if not s or s.get("error"):
        return
    hr("Sustained Load — thermal behavior")
    rates = [x["rate"] for x in s["samples"]]
    _row("Duration", f"{s['duration_s']:.0f}", f" s ({s['workers']} worker(s))")
    _row("Peak throughput", fmt(s["peak_rate"]), f" {s['unit']}")
    _row("Sustained (final 25%)", fmt(s["sustained_rate"]), f" {s['unit']}")
    _row("Droop", f"{s['droop_percent']:.1f}", " %")
    if len(rates) > 1:
        print(f"\n  Throughput over time: {sparkline(rates)}")
        print(f"    (each mark = {s['window_s']:.0f}s; left = start, "
              f"right = end)")
    print(f"\n  Verdict: {s['verdict']}")


def print_scores(scores: dict) -> None:
    hr("Scores (baseline machine = 100, higher is better)")
    cats = category_scores(scores["subscores"])
    for k, v in scores["subscores"].items():
        print(f"  {k:<16}: {v:>9.1f}")
    if cats:
        print(f"  {'-' * 27}")
        for k, v in cats.items():
            print(f"  {k.upper():<16}: {v:>9.1f}")
    print(f"  {'-' * 27}")
    print(f"  {'COMPOSITE':<16}: {scores['composite']:>9.1f}")


def print_report(payload: dict) -> None:
    print_system(payload["system"], payload["state"], payload["warnings"])
    print_results(payload["results"])
    print_native(payload.get("native"))
    print_accelerators(payload.get("accelerators"), payload.get("accel"))
    print_ai(payload.get("ml_framework"))
    print_network(payload.get("network"))
    print_power(payload.get("power"), payload.get("perf_per_watt"))
    print_sustained(payload.get("sustained"))
    print_scores(payload["scores"])
    print_regression(payload.get("regression"))
    _print_validation(payload["results"])


def _print_validation(results: dict) -> None:
    failures = [(k, v["error"]) for k, v in results.items()
                if isinstance(v, dict) and v.get("validation_failed")]
    if failures:
        hr("!! VALIDATION FAILURES — HARDWARE MAY BE UNSTABLE")
        for name, err in failures:
            print(f"  {name}: {err}")
        print("\n  A workload returned an incorrect result. This commonly "
              "indicates\n  unstable overclocking, failing RAM, or inadequate "
              "cooling.")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_json(payload: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = payload["timestamp_utc"].replace(":", "").replace("-", "")
    host = re.sub(r"[^A-Za-z0-9_-]", "_", payload["system"]["hostname"] or "host")
    path = os.path.join(out_dir, f"benchmark_{host}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


CSV_FIELDS = [
    "timestamp_utc", "tool_version", "hostname", "os", "arch", "arch_family",
    "cpu_model", "cores_physical", "cores_logical", "ram_gb", "on_ac_power",
    "cpu_int_primes_s", "cpu_float_iters_s", "cpu_multi_primes_s",
    "compression_mb_s", "hashing_mb_s", "json_mb_s", "mem_mb_s",
    "disk_write_mb_s", "disk_read_mb_s", "disk_iops", "disk_cache_bypassed",
    "gpu_name", "gpu_fp32_gflops", "gpu_fp16_gflops", "gpu_bandwidth_mb_s",
    "gpu_matmul_fp32_tflops", "gpu_matmul_fp16_tflops",
    "npu_name", "npu_gflops", "npu_speedup_vs_cpu",
    "ml_train_samples_s", "ml_infer_samples_s",
    "net_loopback_mb_s", "power_watts", "power_estimated", "score_per_watt",
    "sustained_droop_pct", "composite_score",
]


def _first_name(payload: dict, kind: str) -> str:
    """Name of the first detected GPU/NPU, for the CSV row."""
    items = (payload.get("accelerators") or {}).get(kind) or []
    return items[0].get("name", "") if items else ""


def _rate(results: dict, key: str, field: str = "rate") -> float:
    entry = results.get(key)
    if isinstance(entry, dict):
        v = entry.get(field)
        if isinstance(v, (int, float)):
            return round(float(v), 2)
    return 0.0


def flatten_row(payload: dict) -> dict:
    """Project the nested payload into one flat CSV/regression row."""
    info, results = payload["system"], payload["results"]
    sus = payload.get("sustained") or {}
    ppw = payload.get("perf_per_watt") or {}
    power = payload.get("power") or {}

    return {
        "timestamp_utc": payload["timestamp_utc"],
        "tool_version": payload["version"],
        "hostname": info["hostname"],
        "os": info["os"],
        "arch": info["architecture"],
        "arch_family": info["arch_family"],
        "cpu_model": info["cpu_model"],
        "cores_physical": info.get("cpu_cores_physical") or "",
        "cores_logical": info["cpu_cores_logical"],
        "ram_gb": info.get("ram_total_gb") or "",
        "on_ac_power": payload["state"].get("on_ac_power"),
        "cpu_int_primes_s": _rate(results, "cpu_int"),
        "cpu_float_iters_s": _rate(results, "cpu_float"),
        "cpu_multi_primes_s": _rate(results, "cpu_multi"),
        "compression_mb_s": _rate(results, "compression"),
        "hashing_mb_s": _rate(results, "hashing"),
        "json_mb_s": _rate(results, "json"),
        "mem_mb_s": _rate(results, "memory"),
        "disk_write_mb_s": _rate(results, "disk", "write_rate"),
        "disk_read_mb_s": _rate(results, "disk", "read_rate"),
        "disk_iops": _rate(results, "disk", "random_read_iops"),
        "disk_cache_bypassed": (results.get("disk") or {}).get("cache_bypassed"),
        "gpu_name": _first_name(payload, "gpus"),
        "gpu_fp32_gflops": _rate(results, "gpu_fp32"),
        "gpu_fp16_gflops": _rate(results, "gpu_fp16"),
        "gpu_bandwidth_mb_s": _rate(results, "gpu_bandwidth"),
        "npu_name": _first_name(payload, "npus"),
        "npu_gflops": _rate(results, "npu"),
        "npu_speedup_vs_cpu": round(
            ((payload.get("accel") or {}).get("ane") or {})
            .get("speedup_vs_cpu", 0) or 0, 2),
        "gpu_matmul_fp32_tflops": _rate(results, "gpu_matmul_fp32"),
        "gpu_matmul_fp16_tflops": _rate(results, "gpu_matmul_fp16"),
        "ml_train_samples_s": _rate(results, "ml_train"),
        "ml_infer_samples_s": _rate(results, "ml_infer"),
        "net_loopback_mb_s": round(
            (payload.get("network") or {}).get("loopback_mb_s", 0) or 0, 1),
        "power_watts": power.get("package_w") or "",
        "power_estimated": power.get("estimated", ""),
        "score_per_watt": ppw.get("score_per_watt", "") if ppw else "",
        "sustained_droop_pct": sus.get("droop_percent", ""),
        "composite_score": payload["scores"]["composite"],
    }


def append_csv(payload: dict, out_dir: str) -> str:
    """Append one flattened row, rotating the file if the schema changed.

    Appending new columns to a file written by an older version would silently
    misalign every row, so a header mismatch archives the old file instead.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmarks.csv")
    row = flatten_row(payload)

    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing and existing != CSV_FIELDS:
            os.replace(path, path + ".v2.bak")

    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--card:#fff;--line:#e5e4e1;
--accent:#3b6ea5;--good:#2e7d52;--warn:#b4690e;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ecebe8;
--muted:#9b9b97;--card:#1e1e23;--line:#2e2e35;--accent:#7aa9dd;
--good:#5cbd8a;--warn:#e0a54f;--bad:#ef6b60}}
*{box-sizing:border-box}body{margin:0;padding:2rem 1rem;background:var(--bg);
color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
.sub{color:var(--muted);margin:0 0 2rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1.25rem 1.5rem;margin-bottom:1.25rem}
h2{font-size:1rem;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);margin:0 0 1rem}
table{width:100%;border-collapse:collapse}
td,th{padding:.4rem 0;text-align:left;border-bottom:1px solid var(--line)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.score{font-size:2.6rem;font-weight:600;color:var(--accent)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
.warn{color:var(--warn)}.bad{color:var(--bad)}.good{color:var(--good)}
.note{color:var(--muted);font-size:.85rem}
"""


def save_html(payload: dict, out_dir: str) -> str:
    """Write a self-contained HTML report (no external assets)."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = payload["timestamp_utc"].replace(":", "").replace("-", "")
    host = re.sub(r"[^A-Za-z0-9_-]", "_",
                  payload["system"]["hostname"] or "host")
    path = os.path.join(out_dir, f"report_{host}_{stamp}.html")

    e = html.escape
    info, results = payload["system"], payload["results"]
    scores = payload["scores"]

    # Every field is read defensively: a report must still render when a probe
    # could not identify some part of the machine.
    def g(key, default="unknown"):
        val = info.get(key)
        return e(str(val)) if val not in (None, "") else default

    subtitle = " · ".join(x for x in [g("hostname"),
                                      f"{g('os')} {g('os_release', '')}".strip(),
                                      g("arch_family"),
                                      e(payload.get("timestamp_utc", ""))] if x)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Benchmark — {g('hostname')}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{g('cpu_model')}</h1>",
        f"<p class='sub'>{subtitle}</p>",
        "<div class='card'><h2>Composite score</h2>",
        f"<div class='score'>{scores['composite']:.0f}</div>",
        "<p class='note'>Baseline machine = 100. Geometric mean of all "
        "subscores.</p></div>",
    ]

    if payload.get("warnings"):
        parts.append("<div class='card'><h2>Warnings</h2><ul class='warn'>")
        parts += [f"<li>{e(w)}</li>" for w in payload["warnings"]]
        parts.append("</ul></div>")

    parts.append("<div class='card'><h2>System</h2><table>")
    for label, key in [("CPU", "cpu_model"), ("Architecture", "arch_family"),
                       ("Logical cores", "cpu_cores_logical"),
                       ("Physical cores", "cpu_cores_physical"),
                       ("RAM (GB)", "ram_total_gb"),
                       ("OS", "platform"), ("Python", "python_version")]:
        parts.append(f"<tr><td>{label}</td><td class='num'>"
                     f"{e(str(info.get(key, '-')))}</td></tr>")
    parts.append("</table></div>")

    parts.append("<div class='card'><h2>Results</h2><table>")
    for key, label, field, unit in [
            ("cpu_int", "CPU integer", "rate", "primes/s"),
            ("cpu_float", "CPU float", "rate", "iters/s"),
            ("cpu_multi", "CPU multi-core", "rate", "primes/s"),
            ("compression", "Compression", "rate", "MB/s"),
            ("hashing", "SHA-256", "rate", "MB/s"),
            ("json", "JSON parse", "rate", "MB/s"),
            ("memory", "Memory copy", "rate", "MB/s"),
            ("disk", "Disk write", "write_rate", "MB/s"),
            ("disk", "Disk read", "read_rate", "MB/s"),
            ("disk", "Random read", "random_read_iops", "IOPS")]:
        r = results.get(key)
        if isinstance(r, dict) and isinstance(r.get(field), (int, float)):
            parts.append(f"<tr><td>{label}</td><td class='num'>"
                         f"{r[field]:,.0f} {unit}</td></tr>")
    parts.append("</table></div>")

    if scores["subscores"]:
        top = max(scores["subscores"].values())
        parts.append("<div class='card'><h2>Subscores</h2><table>")
        for k, v in scores["subscores"].items():
            pct = v / top * 100 if top else 0
            parts.append(
                f"<tr><td>{e(k)}</td><td style='width:60%'>"
                f"<div class='bar'><i style='width:{pct:.0f}%'></i></div></td>"
                f"<td class='num'>{v:.0f}</td></tr>")
        parts.append("</table></div>")

    inv = payload.get("accelerators") or {}
    accel = payload.get("accel") or {}
    if inv.get("gpus") or inv.get("npus"):
        parts.append("<div class='card'><h2>Accelerators</h2><table>")
        for gpu in inv.get("gpus", []):
            extra = f" · {gpu['cores']} cores" if gpu.get("cores") else ""
            parts.append(f"<tr><td>GPU</td><td class='num'>"
                         f"{e(str(gpu.get('name', '?')))}{e(extra)}</td></tr>")
        for npu in inv.get("npus", []):
            parts.append(f"<tr><td>NPU</td><td class='num'>"
                         f"{e(str(npu.get('name', '?')))}</td></tr>")
        for item in accel.get("results", []):
            if isinstance(item.get("value"), (int, float)):
                parts.append(
                    f"<tr><td>{e(str(item['name']))}</td><td class='num'>"
                    f"{item['value']:,.1f} {e(str(item.get('unit', '')))}"
                    f"</td></tr>")
        parts.append("</table>")
        ane = accel.get("ane")
        if ane:
            cls = "good" if ane.get("engaged") else "warn"
            state = "engaged" if ane.get("engaged") else "did not engage"
            parts.append(
                f"<p class='note {cls}'>Neural Engine {state} — "
                f"{ane.get('speedup_vs_cpu', 0):.2f}x vs CPU-only Core ML.</p>")
        parts.append("</div>")

    ml = payload.get("ml_framework")
    if ml and ml.get("available") and not ml.get("error"):
        parts.append("<div class='card'><h2>AI framework</h2><table>")
        parts.append(f"<tr><td>Framework</td><td class='num'>"
                     f"{e(str(ml.get('framework', '')))} "
                     f"{e(str(ml.get('framework_version', '')))}</td></tr>")
        parts.append(f"<tr><td>Device</td><td class='num'>"
                     f"{e(str(ml.get('device_name', ml.get('device', '?'))))}"
                     f"</td></tr>")
        if ml.get("train_samples_per_s"):
            parts.append(f"<tr><td>Training</td><td class='num'>"
                         f"{ml['train_samples_per_s']:,.0f} samples/s</td></tr>")
        if ml.get("infer_samples_per_s"):
            parts.append(f"<tr><td>Inference</td><td class='num'>"
                         f"{ml['infer_samples_per_s']:,.0f} samples/s</td></tr>")
        parts.append("</table></div>")

    power = payload.get("power") or {}
    ppw = payload.get("perf_per_watt") or {}
    if power.get("package_w"):
        tag = "estimated" if power.get("estimated") else "measured"
        parts.append("<div class='card'><h2>Power & efficiency</h2><table>")
        parts.append(f"<tr><td>Package power</td><td class='num'>"
                     f"{power['package_w']:,.1f} W ({tag})</td></tr>")
        if ppw:
            parts.append(f"<tr><td>Perf-per-watt</td><td class='num'>"
                         f"{ppw['score_per_watt']:,.1f} score/W</td></tr>")
        parts.append("</table></div>")

    reg = payload.get("regression")
    if reg and reg.get("findings"):
        parts.append("<div class='card'><h2>Regression vs. history</h2><table>")
        for f in reg["findings"]:
            cls = "bad" if f["direction"] == "slower" else "good"
            parts.append(
                f"<tr><td>{e(str(f['metric']))}</td><td class='num {cls}'>"
                f"{f['change_pct']:+.1f}%</td></tr>")
        parts.append("</table></div>")

    sus = payload.get("sustained")
    if sus and not sus.get("error"):
        cls = ("good" if sus["droop_percent"] < 15
               else "warn" if sus["droop_percent"] < 30 else "bad")
        parts.append(
            "<div class='card'><h2>Sustained load</h2>"
            f"<table><tr><td>Peak</td><td class='num'>"
            f"{sus['peak_rate']:,.0f}</td></tr>"
            f"<tr><td>Sustained</td><td class='num'>"
            f"{sus['sustained_rate']:,.0f}</td></tr>"
            f"<tr><td>Droop</td><td class='num {cls}'>"
            f"{sus['droop_percent']:.1f}%</td></tr></table>"
            f"<p class='note'>{e(sus['verdict'])}</p></div>")

    parts.append("</div></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return path
