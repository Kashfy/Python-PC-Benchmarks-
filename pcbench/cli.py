"""Command-line interface and run orchestration."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from datetime import datetime, timezone

from . import __version__
from . import accel as accel_mod
from . import native as native_mod
from . import report as report_mod
from . import workloads as wl
from .compare import load_history, render_table
from .core import ValidationError
from .scoring import compute_scores
from .sustained import run_sustained
from .system import inventory, machine_state, state_warnings

TESTS = ["cpu_int", "cpu_float", "cpu_multi", "compression", "hashing",
         "json", "memory", "cache_sweep", "disk"]
DEFAULT_TESTS = list(TESTS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pcbench",
        description="Cross-platform PC benchmark & diagnostics tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"pcbench {__version__}")

    g = p.add_argument_group("workload")
    g.add_argument("--seconds", type=float, default=3.0,
                   help="Target duration per test, per repeat")
    g.add_argument("--repeats", type=int, default=3,
                   help="Repeats per test (median is reported)")
    g.add_argument("--only", default="",
                   help="Comma-separated subset of: " + ",".join(TESTS))
    g.add_argument("--skip", default="",
                   help="Comma-separated tests to exclude")
    g.add_argument("--quick", action="store_true",
                   help="Fast pass (1s x 2 repeats, small disk test)")
    g.add_argument("--disk-mb", type=int, default=256,
                   help="Disk test file size in MB")
    g.add_argument("--mem-mb", type=int, default=64,
                   help="Memory test buffer size in MB")

    g = p.add_argument_group("sustained load")
    g.add_argument("--sustained", metavar="DURATION", default=None,
                   help="Run a thermal/throttling test, e.g. 5m, 300s, 90")
    g.add_argument("--sustained-window", type=float, default=5.0,
                   help="Sampling window for the sustained test, seconds")
    g.add_argument("--sustained-workers", type=int, default=0,
                   help="Load processes for the sustained test (0 = all cores)")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="results",
                   help="Directory for JSON/CSV/HTML output")
    g.add_argument("--html", action="store_true",
                   help="Also write a self-contained HTML report")
    g.add_argument("--json-stdout", action="store_true",
                   help="Print the full payload as JSON to stdout")
    g.add_argument("--no-save", action="store_true",
                   help="Do not write any result files")
    g.add_argument("--compare", action="store_true",
                   help="Show a ranked table of past runs and exit")
    g.add_argument("--all-runs", action="store_true",
                   help="With --compare, list every run, not just the latest "
                        "per machine")

    g = p.add_argument_group("accelerators")
    g.add_argument("--no-gpu", action="store_true",
                   help="Skip GPU compute benchmarks")
    g.add_argument("--no-npu", action="store_true",
                   help="Skip NPU / Apple Neural Engine benchmarks")
    g.add_argument("--no-accel", action="store_true",
                   help="Skip all accelerator benchmarks (inventory still "
                        "reported)")

    g = p.add_argument_group("other")
    g.add_argument("--no-native", action="store_true",
                   help="Skip the optional native C engine")
    g.add_argument("--force", action="store_true",
                   help="Run even when machine state would distort results")
    return p


def parse_duration(text: str) -> float:
    """Parse '90', '300s', '5m', or '1h' into seconds."""
    t = text.strip().lower()
    mult = 1.0
    if t.endswith("h"):
        mult, t = 3600.0, t[:-1]
    elif t.endswith("m"):
        mult, t = 60.0, t[:-1]
    elif t.endswith("s"):
        t = t[:-1]
    try:
        value = float(t)
    except ValueError:
        raise ValueError(f"invalid duration: {text!r} (try 30s, 5m, 1h)")
    if value <= 0:
        raise ValueError(f"duration must be positive: {text!r}")
    return value * mult


def select_tests(only: str, skip: str) -> list[str]:
    chosen = ([t.strip() for t in only.split(",") if t.strip()]
              if only else list(DEFAULT_TESTS))
    unknown = [t for t in chosen if t not in TESTS]
    if unknown:
        raise ValueError(f"unknown test(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(TESTS)}")
    excluded = [t.strip() for t in skip.split(",") if t.strip()]
    unknown = [t for t in excluded if t not in TESTS]
    if unknown:
        raise ValueError(f"unknown test(s) in --skip: {', '.join(unknown)}")
    return [t for t in chosen if t not in excluded]


def _runners(args, info, disk_dir) -> dict:
    s, r = args.seconds, args.repeats
    return {
        "cpu_int": lambda: wl.bench_cpu_integer(s, r),
        "cpu_float": lambda: wl.bench_cpu_float(s, r),
        "cpu_multi": lambda: wl.bench_cpu_multicore(s, info["cpu_cores_logical"]),
        "compression": lambda: wl.bench_compression(s, r),
        "hashing": lambda: wl.bench_hashing(s, r),
        "json": lambda: wl.bench_json(s, r),
        "memory": lambda: wl.bench_memory(s, r, args.mem_mb),
        "cache_sweep": lambda: wl.bench_cache_sweep(
            s, info.get("ram_total_bytes", 0)),
        "disk": lambda: wl.bench_disk(s, r, args.disk_mb, disk_dir),
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        path = os.path.join(args.output_dir, "benchmarks.csv")
        print(render_table(load_history(path), all_runs=args.all_runs))
        return 0

    if args.quick:
        args.seconds, args.repeats, args.disk_mb = 1.0, 2, 64

    try:
        selected = select_tests(args.only, args.skip)
        sustained_seconds = (parse_duration(args.sustained)
                             if args.sustained else None)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    quiet = args.json_stdout
    disk_dir = tempfile.gettempdir() if args.no_save else args.output_dir
    try:
        os.makedirs(disk_dir, exist_ok=True)
    except OSError:
        disk_dir = tempfile.gettempdir()

    info = inventory()
    state = machine_state()
    warnings = state_warnings(state)

    if not quiet:
        report_mod.hr(f"PC Benchmark & Diagnostics v{__version__}")
        print(f"  seconds/test={args.seconds}  repeats={args.repeats}  "
              f"tests={','.join(selected) or 'none'}")

    # Distorting conditions are worth stopping for: a result taken on battery
    # or under load looks like a hardware difference but is not.
    if warnings and not args.force and not quiet:
        print()
        for w in warnings:
            print(f"  !!  {w}")
        print("\n  Re-run with --force to benchmark anyway.")
        return 3

    results: dict = {}
    runners = _runners(args, info, disk_dir)
    for name in selected:
        if not quiet:
            print(f"  running {name} ...", flush=True)
        try:
            results[name] = runners[name]()
        except ValidationError as e:
            # A wrong answer is a hardware finding, not a crash.
            results[name] = {"error": str(e), "validation_failed": True}
            if not quiet:
                print(f"    !! {name}: VALIDATION FAILED — {e}",
                      file=sys.stderr)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            if not quiet:
                print(f"    ! {name} failed: {e}", file=sys.stderr)

    native = None
    if not args.no_native:
        if not quiet:
            print("  running native engine ...", flush=True)
        native = native_mod.run(args.seconds, args.repeats, _repo_root(),
                                threads=info["cpu_cores_logical"])

    # Accelerator inventory is cheap and always collected; benchmarking it is
    # opt-out and only implemented on Apple platforms.
    accel_inv = accel_mod.inventory(info.get("cpu_model", ""))
    accel = None
    if not args.no_accel and not (args.no_gpu and args.no_npu):
        if accel_inv["benchmark_supported"]:
            if not quiet:
                print("  running accelerator engine (GPU/NPU) ...", flush=True)
            accel = accel_mod.run(args.seconds, _repo_root(), disk_dir,
                                  gpu=not args.no_gpu, ane=not args.no_npu)
            # Fold the headline accelerator rates in so they score like any
            # other metric.
            for key, value in accel_mod.extract_rates(accel).items():
                results[key] = {"rate": value}

    sustained = None
    if sustained_seconds:
        workers = args.sustained_workers or info["cpu_cores_logical"]
        if not quiet:
            print(f"  running sustained load for {sustained_seconds:.0f}s "
                  f"on {workers} worker(s) ...", flush=True)
        sustained = run_sustained(sustained_seconds, args.sustained_window,
                                  workers)

    scores = compute_scores(results)
    payload = {
        "tool": "pcbench",
        "version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "config": {"seconds": args.seconds, "repeats": args.repeats,
                   "tests": selected, "disk_mb": args.disk_mb,
                   "mem_mb": args.mem_mb,
                   "sustained_seconds": sustained_seconds},
        "system": info,
        "state": state,
        "warnings": warnings,
        "results": results,
        "native": native,
        "accelerators": accel_inv,
        "accel": accel,
        "sustained": sustained,
        "scores": scores,
    }

    if args.json_stdout:
        print(json.dumps(payload, indent=2, default=str))
    else:
        report_mod.print_report(payload)

    if not args.no_save:
        try:
            jp = report_mod.save_json(payload, args.output_dir)
            cp = report_mod.append_csv(payload, args.output_dir)
            hp = (report_mod.save_html(payload, args.output_dir)
                  if args.html else None)
            if not quiet:
                report_mod.hr("Saved")
                print(f"  JSON: {jp}")
                print(f"  CSV : {cp}")
                if hp:
                    print(f"  HTML: {hp}")
        except OSError as e:
            print(f"  ! could not save results: {e}", file=sys.stderr)

    if any(isinstance(v, dict) and v.get("validation_failed")
           for v in results.values()):
        return 4
    return 0


def _repo_root() -> str:
    """Directory holding native_engine.c (the package's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def entry() -> None:
    mp.freeze_support()
    sys.exit(main())
