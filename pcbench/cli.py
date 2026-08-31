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
from . import apps as apps_mod
from . import config as config_mod
from . import container as container_mod
from . import cores as cores_mod
from . import diagnose
from . import checkup as checkup_mod
from . import counters as counters_mod
from . import datascience as ds_mod
from . import drivelife
from . import export as export_mod
from . import gates as gates_mod
from . import health
from . import hwinfo
from . import iobench
from . import monitor as monitor_mod
from . import numa as numa_mod
from . import provenance
from . import reference
from . import soak as soak_mod
from . import standards as standards_mod
from . import stats as stats_mod
from . import storage as storage_mod
from . import plugins as plugins_mod
from . import interference
from . import cryptobench
from . import gpucompute
from . import numeric
from . import optional
from . import mlbench
from . import mlframework
from . import npu as npu_mod
from . import native as native_mod
from . import network
from . import power
from . import regression
from . import sysbench
from . import report as report_mod
from . import wizard
from . import workloads as wl
from .compare import load_history, render_table
from .core import ValidationError
from .scoring import compute_scores
from .sustained import run_sustained
from . import system as system_mod
from .system import inventory, machine_state, state_warnings

#: Synthetic tests: each isolates one subsystem, which is what makes them
#: useful for diagnosis.
SYNTHETIC_TESTS = ["cpu_int", "cpu_float", "cpu_multi", "cores", "compression",
                   "hashing", "json", "memory", "mem_scaling", "cache_sweep",
                   "disk", "nn_training", "kmeans", "knn", "compile",
                   "latency"]

#: Application-shaped tests: each mixes subsystems the way real software does,
#: which is what makes them useful for deciding whether a machine suits a job.
APP_TESTS = ["sqlite", "fsync", "raytrace", "image", "logparse", "video"]

TESTS = SYNTHETIC_TESTS + APP_TESTS

#: One-line descriptions for ``--list-tests``.
DESCRIPTIONS = {
    "cpu_int": "Integer math, single core (primes/s)",
    "cpu_float": "Floating-point math, single core (iters/s)",
    "cpu_multi": "Integer math across every core (primes/s)",
    "cores": "Per-core-count scaling curve and efficiency-core detection",
    "compression": "zlib compress+decompress round-trip (MB/s)",
    "hashing": "SHA-256, reaching hardware crypto instructions (MB/s)",
    "json": "JSON parse throughput (MB/s)",
    "memory": "Large-buffer memory copy bandwidth (MB/s)",
    "mem_scaling": "Memory bandwidth against concurrent readers (MB/s)",
    "cache_sweep": "Latency across working-set sizes; reveals cache levels",
    "disk": "Sequential read/write, random IOPS, queue-depth sweep",
    "nn_training": "Pure-Python MLP training steps/s",
    "kmeans": "k-means distance computations/s",
    "knn": "k-nearest-neighbour comparisons/s",
    "compile": "C compiles per minute (needs a compiler on PATH)",
    "latency": "Syscall, context-switch, and process-spawn latency",
    **apps_mod.DESCRIPTIONS,
}

# Curated subsets for common situations, so a user does not have to know
# which of twenty-two tests matter for their machine.
PROFILES = {
    "quick": ["cpu_int", "cpu_multi", "memory", "disk"],
    "tiny": ["cpu_int", "memory"],
    "cpu": ["cpu_int", "cpu_float", "cpu_multi", "cores", "compression",
            "hashing", "json"],
    "ai": ["cpu_multi", "nn_training", "kmeans", "knn", "memory"],
    "dev": ["cpu_int", "cpu_multi", "compile", "disk", "latency", "json",
            "sqlite", "logparse"],
    "storage": ["disk", "fsync"],
    "laptop": ["cpu_int", "cpu_multi", "cores", "memory", "disk", "latency"],
    "server": ["cpu_multi", "cores", "memory", "mem_scaling", "disk",
               "latency", "compression", "sqlite", "fsync"],
    "apps": ["sqlite", "fsync", "raytrace", "image", "logparse", "video"],
    "database": ["sqlite", "fsync", "disk", "memory", "cpu_multi"],
    "media": ["video", "image", "raytrace", "cpu_multi", "memory", "disk"],
    "workstation": ["cpu_int", "cpu_multi", "cores", "memory", "mem_scaling",
                    "disk", "compile", "raytrace", "video"],
    "ci": ["cpu_int", "cpu_multi", "memory", "disk", "compile", "sqlite"],
}
# ``video`` is excluded from the default set: it is a fixed multi-second
# encode rather than a time-boxed loop, and it needs ffmpeg installed. Ask for
# it with ``--only video`` or ``--profile media``.
DEFAULT_TESTS = SYNTHETIC_TESTS + [t for t in APP_TESTS if t != "video"]


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
    g.add_argument("--profile", default="", metavar="NAME",
                   help="Preset test selection: " + ", ".join(PROFILES))
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
    g.add_argument("--spec-sheet", action="store_true",
                   help="Also write a one-page Markdown spec sheet")
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

    g = p.add_argument_group("AI / efficiency / monitoring")
    g.add_argument("--ai", action="store_true",
                   help="Force the ML-framework training/inference benchmark "
                        "(auto-runs when PyTorch/ONNX is installed)")
    g.add_argument("--no-ai", action="store_true",
                   help="Skip the ML-framework benchmark even if installed")
    g.add_argument("--ai-batch", type=int, default=64,
                   help="Batch size for the ML-framework benchmark")
    g.add_argument("--no-power", action="store_true",
                   help="Skip power / perf-per-watt measurement")
    g.add_argument("--health", action="store_true",
                   help="Run RAM integrity and drive SMART health checks")
    g.add_argument("--no-drive-life", action="store_true",
                   help="Skip the SSD lifetime/wear report (terabytes written, "
                        "power-on hours, remaining endurance)")
    g.add_argument("--health-mb", type=int, default=256,
                   help="Memory to cover in the RAM integrity test")
    g.add_argument("--network-host", default="", metavar="HOST",
                   help="Measure real latency to this host (sends external "
                        "traffic; off by default)")
    g.add_argument("--network-url", default="", metavar="URL",
                   help="Measure download throughput from this URL (sends "
                        "external traffic; off by default)")
    g.add_argument("--no-plugins", action="store_true",
                   help="Skip benchmarks found in the plugins/ directory")
    g.add_argument("--no-network", action="store_true",
                   help="Skip the loopback network benchmark")
    g.add_argument("--no-regression", action="store_true",
                   help="Skip run-over-run regression detection")
    g.add_argument("--regression-threshold", type=float, default=10.0,
                   help="Percent change that counts as a regression")

    g.add_argument("--no-optional", action="store_true",
                   help="Skip all benchmarks that need optional packages")

    g = p.add_argument_group("storage devices")
    g.add_argument("--disk-path", default="", metavar="PATH",
                   help="Comma-separated mount points to benchmark storage on "
                        "(in addition to the main run)")
    g.add_argument("--disk-all", action="store_true",
                   help="Benchmark every writable local filesystem found")
    g.add_argument("--list-devices", action="store_true",
                   help="List mounted storage and whether each can be "
                        "benchmarked, then exit")
    g.add_argument("--drive-speed", nargs="?", const="all", default=None,
                   metavar="MOUNT[,MOUNT]",
                   help="Measure sequential read/write and random-read IOPS "
                        "per drive, then exit. Omit the list for every "
                        "benchmarkable filesystem")

    g = p.add_argument_group("stability / monitoring")
    g.add_argument("--soak", metavar="DURATION", default=None,
                   help="Burn-in: run validating work for this long and count "
                        "wrong answers, e.g. 30m, 4h")
    g.add_argument("--soak-workers", type=int, default=0,
                   help="Load processes for the soak test (0 = all cores)")
    g.add_argument("--monitor", metavar="DURATION", default=None,
                   help="Watch clocks, temperature, load, and memory for this "
                        "long instead of benchmarking, e.g. 60s, 10m")
    g.add_argument("--monitor-interval", type=float, default=1.0,
                   help="Seconds between monitor samples")
    g.add_argument("--monitor-power", action="store_true",
                   help="Also sample power draw while monitoring (costs a "
                        "privileged subprocess per sample on macOS)")
    g.add_argument("--monitor-trace", default="", metavar="PATH",
                   help="Write raw monitor samples to this CSV file")

    g = p.add_argument_group("integration / CI")
    g.add_argument("--prometheus", default="", metavar="PATH",
                   help="Write Prometheus exposition text (for a node_exporter "
                        "textfile collector)")
    g.add_argument("--junit", default="", metavar="PATH",
                   help="Write a JUnit XML report, so CI renders results and "
                        "regressions as tests")
    g.add_argument("--sqlite", default="", metavar="PATH",
                   help="Append this run to a SQLite history database")
    g.add_argument("--markdown", default="", metavar="PATH",
                   help="Write a Markdown summary for an issue or PR comment")
    g.add_argument("--fail-under", type=float, default=None, metavar="SCORE",
                   help="Exit non-zero when the composite score is below this")
    g.add_argument("--assert", dest="assert_", action="append", default=[],
                   metavar="EXPR",
                   help="Threshold that must hold, e.g. 'disk.read_rate>=500'. "
                        "Repeatable; a failure exits non-zero")

    g = p.add_argument_group("configuration")
    g.add_argument("--config", default="", metavar="PATH",
                   help="Read settings from this TOML/JSON file instead of "
                        "searching for one")
    g.add_argument("--no-config", action="store_true",
                   help="Ignore any pcbench.toml and PCBENCH_* variables")
    g.add_argument("--init-config", nargs="?", const="pcbench.toml",
                   default=None, metavar="PATH",
                   help="Write a commented starter config file and exit")
    g.add_argument("--list-tests", action="store_true",
                   help="List every test and profile, then exit")

    g = p.add_argument_group("hardware stats (no benchmarking)")
    g.add_argument("--stats", nargs="?", const="all", default=None,
                   metavar="SECTIONS",
                   help="Report hardware facts without running any benchmark: "
                        "battery wear, SSD endurance, temperatures, GPU "
                        "inventory, OS settings. Comma-separated sections, or "
                        "omit for all")
    g.add_argument("--list-stats", action="store_true",
                   help="List the available --stats sections, then exit")
    g.add_argument("--menu", action="store_true",
                   help="Guided setup, driven by the arrow keys: choose a "
                        "benchmark, a stats section, a monitor or a "
                        "comparison one screen at a time, then confirm the "
                        "command it builds")

    g = p.add_argument_group("diagnosis")
    g.add_argument("--checkup", action="store_true",
                   help="Diagnose why this machine is slow: thermal, power, "
                        "contention, memory, disk headroom, drive health and "
                        "a short measurement, ranked by likely impact")
    g.add_argument("--no-measure", action="store_true",
                   help="With --checkup, read settings and live state only; "
                        "skip the short benchmark and the throttling check")

    g = p.add_argument_group("analysis depth")
    g.add_argument("--counters", action="store_true",
                   help="Collect hardware performance counters (IPC, cache "
                        "and branch misses) around the CPU tests; needs perf "
                        "on Linux")
    g.add_argument("--no-provenance", action="store_true",
                   help="Skip capture of governor, mitigations, hugepages and "
                        "microcode")
    g.add_argument("--no-standards", action="store_true",
                   help="Skip STREAM, LINPACK, and the CoreMark-style suite")
    g.add_argument("--no-linpack", action="store_true",
                   help="Skip LINPACK only (it needs NumPy and a few seconds)")
    g.add_argument("--numa", action="store_true",
                   help="Report NUMA topology")
    g.add_argument("--numa-bandwidth", action="store_true",
                   help="Also measure the local/remote bandwidth matrix "
                        "(needs numactl; implies --numa)")
    g.add_argument("--energy", action="store_true",
                   help="Measure energy-to-solution in joules for a fixed "
                        "workload")

    g = p.add_argument_group("data science / ML")
    g.add_argument("--datascience", action="store_true",
                   help="LLM prefill/decode tokens per second, input-pipeline "
                        "throughput, batch scaling, and dataframe operations")
    g.add_argument("--ds-prefill-tokens", type=int, default=256,
                   help="Prompt length for the LLM prefill measurement")
    g.add_argument("--ds-decode-tokens", type=int, default=32,
                   help="Tokens to generate for the LLM decode measurement")
    g.add_argument("--no-dataframes", action="store_true",
                   help="Skip the dataframe benchmarks")

    g = p.add_argument_group("configurable I/O")
    g.add_argument("--io", action="store_true",
                   help="Run the storage job suite (database, sequential, log "
                        "write, mixed VM)")
    g.add_argument("--io-job", action="append", default=[], metavar="SPEC",
                   help="Custom I/O job, e.g. "
                        "'oltp:bs=8k,pattern=randread,qd=32'. Repeatable; "
                        "replaces the default suite")

    g = p.add_argument_group("two-node network")
    g.add_argument("--net-server", action="store_true",
                   help="Run the receiving half and wait for a peer (opens a "
                        "listening port)")
    g.add_argument("--net-client", default="", metavar="HOST",
                   help="Measure throughput, latency, and jitter against a "
                        "peer running --net-server")
    g.add_argument("--net-port", type=int, default=network.DEFAULT_PORT,
                   help="Port for the two-node network test")
    g.add_argument("--net-streams", type=int, default=4,
                   help="Parallel streams for the two-node throughput test")

    g = p.add_argument_group("internet speed test")
    g.add_argument("--internet", action="store_true",
                   help="Measure download, upload, latency and jitter against "
                        "a public endpoint, then exit. Sends traffic off this "
                        "machine and spends bandwidth, so it is never part of "
                        "a normal run")
    g.add_argument("--internet-seconds", type=float, default=5.0,
                   metavar="N",
                   help="Time budget per direction for --internet")
    g.add_argument("--internet-max-mb", type=int, default=200, metavar="MB",
                   help="Byte budget for --internet; upload uses a quarter of "
                        "it, so a metered link cannot be drained")
    g.add_argument("--internet-server", default=network.DEFAULT_SPEED_SERVER,
                   metavar="URL",
                   help="Endpoint for --internet; must serve /__down and "
                        "/__up")

    g = p.add_argument_group("A/B comparison")
    g.add_argument("--compare-runs", nargs=2, default=None,
                   metavar=("BASELINE.json", "CANDIDATE.json"),
                   help="Statistically compare two saved runs and exit")
    g.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold for --compare-runs")

    g = p.add_argument_group("other")
    g.add_argument("--no-native", action="store_true",
                   help="Skip the optional native C engine")
    g.add_argument("--no-autoscale", action="store_true",
                   help="Do not shrink test sizes on small or CPU-limited "
                        "machines")
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
        "memory": lambda: wl.bench_memory(s, r, args.mem_mb,
                                          info.get("ram_total_bytes", 0)),
        "cache_sweep": lambda: wl.bench_cache_sweep(
            s, info.get("ram_total_bytes", 0)),
        "disk": lambda: wl.bench_disk(s, r, args.disk_mb, disk_dir),
        "nn_training": lambda: mlbench.bench_nn_training(s, r),
        "kmeans": lambda: mlbench.bench_kmeans(s, r),
        "knn": lambda: mlbench.bench_knn(s, r),
        "cores": lambda: cores_mod.analyze(min(s, 0.6),
                                           info["cpu_cores_logical"]),
        "mem_scaling": lambda: wl.bench_memory_scaling(
            min(s, 0.5), 64, info.get("ram_total_bytes", 0)),
        "compile": lambda: sysbench.bench_compile(max(1, min(r, 3))),
        "latency": lambda: sysbench.bench_latency_suite(),
        # Application-shaped workloads.
        "sqlite": lambda: apps_mod.bench_sqlite(s, r),
        "fsync": lambda: apps_mod.bench_fsync(s, disk_dir),
        "raytrace": lambda: apps_mod.bench_raytrace(s, r),
        "image": lambda: apps_mod.bench_image(s, r),
        "logparse": lambda: apps_mod.bench_logparse(s, r),
        "video": lambda: apps_mod.bench_video(s, disk_dir),
    }


def accel_enabled(args, device: str) -> bool:
    """Whether the ``device`` accelerator benchmark should run.

    ``--no-accel`` is the blanket opt-out, so it has to gate *every*
    accelerator section; the per-device flags only narrow it further. The
    OpenCL block tested ``--no-gpu`` alone, which left ``--no-accel`` skipping
    the NPU and the Apple engine while still running GPU compute -- the
    opposite of what its help text promises.
    """
    return not args.no_accel and not getattr(args, f"no_{device}")


def _scratch_dir() -> str:
    """A writable directory for test files, on real storage where possible.

    ``--no-save`` used to send the disk test to the system temp directory,
    which on most current Linux distributions is a tmpfs — so the "disk"
    figures were memory bandwidth. The working directory is on real storage
    far more often, so it is tried first and the temp directory is the
    fallback. The files are deleted either way.
    """
    temp = tempfile.gettempdir()
    if not wl.memory_filesystem(temp):
        return temp
    for candidate in (os.getcwd(), os.path.expanduser("~")):
        try:
            if (candidate and os.path.isdir(candidate)
                    and os.access(candidate, os.W_OK)
                    and not wl.memory_filesystem(candidate)):
                return candidate
        except OSError:
            continue
    return temp                      # disclosed in the result, not hidden


def _check_output_writable(out_dir: str) -> str | None:
    """Verify results can be saved *before* spending minutes benchmarking.

    Running once with sudo (for real power readings) leaves root-owned files
    behind, and every later unprivileged run then fails to save. Discovering
    that after a ten-minute run is needlessly painful, so it is checked up
    front and the fix is spelled out.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        return f"cannot create {out_dir}: {e}"

    probe = os.path.join(out_dir, ".pcbench_write_test")
    try:
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
    except OSError:
        owner = ""
        try:
            import pwd
            st = os.stat(out_dir)
            existing = [f for f in os.listdir(out_dir) if not f.startswith(".")]
            if existing:
                st = os.stat(os.path.join(out_dir, existing[0]))
            owner = pwd.getpwuid(st.st_uid).pw_name
        except Exception:
            pass
        hint = (f" (files are owned by '{owner}')" if owner else "")
        return (f"results in '{out_dir}' are not writable{hint}. "
                f"If you ran with sudo before, reclaim them:\n"
                f"      sudo chown -R $(whoami) {out_dir}\n"
                f"    or write elsewhere with --output-dir DIR, "
                f"or skip saving with --no-save")
    return None


def list_tests() -> str:
    """Human-readable catalogue of tests and profiles."""
    lines = ["Synthetic tests (one subsystem each):"]
    for name in SYNTHETIC_TESTS:
        default = " " if name in DEFAULT_TESTS else "-"
        lines.append(f"  {default} {name:<14} {DESCRIPTIONS.get(name, '')}")
    lines.append("")
    lines.append("Application tests (subsystems mixed as real software does):")
    for name in APP_TESTS:
        default = " " if name in DEFAULT_TESTS else "-"
        lines.append(f"  {default} {name:<14} {DESCRIPTIONS.get(name, '')}")
    lines.append("")
    lines.append("  ('-' marks a test excluded from the default run)")
    lines.append("")
    lines.append("Profiles (--profile NAME):")
    for name, tests in PROFILES.items():
        lines.append(f"    {name:<14} {', '.join(tests)}")
    return "\n".join(lines)


def _autoscale(args, info: dict, confinement: dict, quiet: bool) -> list[str]:
    """Shrink workload sizes to fit a small or confined machine.

    Defaults sized for a laptop are actively harmful on the machines this tool
    most needs to work on. A 256 MB disk test on a Raspberry Pi with a 16 GB
    card is a meaningful fraction of the card; a 64 MB memory test on a 512 MB
    board pushes it into swap; sixteen workers inside a half-core container all
    contend for the same slice. Each adjustment is reported, because a silently
    different workload is a silently incomparable result.
    """
    notes: list[str] = []
    ram = confinement.get("effective_ram_bytes") or info.get("ram_total_bytes") or 0
    cores = confinement.get("effective_cores") or info.get("cpu_cores_logical") or 1
    gb = ram / (1024 ** 3) if ram else 0

    if gb and gb < 2:
        # Sub-2 GB machines: SBCs, minimal cloud instances, tight containers.
        if args.mem_mb > 16:
            args.mem_mb = 16
            notes.append(f"memory test reduced to 16 MB for a "
                         f"{gb:.1f} GB machine")
        if args.disk_mb > 64:
            args.disk_mb = 64
            notes.append(f"disk test reduced to 64 MB for a "
                         f"{gb:.1f} GB machine")
    elif gb and gb < 4:
        if args.mem_mb > 32:
            args.mem_mb = 32
            notes.append(f"memory test reduced to 32 MB for a "
                         f"{gb:.1f} GB machine")
        if args.disk_mb > 128:
            args.disk_mb = 128
            notes.append(f"disk test reduced to 128 MB for a "
                         f"{gb:.1f} GB machine")

    if confinement.get("cpu_quota_cores") and cores < (info.get(
            "cpu_cores_logical") or 1):
        notes.append(f"multicore tests will use {cores} worker(s) to match "
                     f"the CPU quota rather than "
                     f"{info.get('cpu_cores_logical')} host cores")

    if notes and not quiet:
        for note in notes:
            print(f"  ~ {note}")
    return notes


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tests:
        print(list_tests())
        return 0

    if args.init_config is not None:
        try:
            path = config_mod.write_sample(args.init_config)
        except config_mod.ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    config_info: dict = {}
    if not args.no_config:
        try:
            config_info = config_mod.apply(args, parser,
                                           args.config or None)
        except config_mod.ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.list_stats:
        print("Hardware stats sections (--stats a,b,c; omit for all):\n")
        for name, (label, _) in hwinfo.SECTIONS.items():
            print(f"  {name:<14} {label}")
        print("\nNone of these run a benchmark or load the machine.")
        return 0

    if args.stats is not None:
        try:
            sections = hwinfo.parse_sections(args.stats)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        data = hwinfo.collect(sections, _repo_root())
        if args.json_stdout:
            print(json.dumps({"tool": "pcbench", "mode": "stats",
                              "version": __version__, "stats": data},
                             indent=2, default=str))
        else:
            print(hwinfo.render(data, _repo_root()))
        return 0

    if args.menu:
        # The menu assembles a fresh command line, so anything typed beside
        # --menu is discarded. Say so rather than appearing to ignore it.
        typed = [a for a in (argv if argv is not None else sys.argv[1:])
                 if a != "--menu"]
        if typed:
            print(f"  note: --menu builds its own command line, so "
                  f"{' '.join(typed)} will not be used.")
        choice = wizard.run()
        if choice is None:
            return 0
        return main(choice)

    if args.list_devices:
        inv = storage_mod.inventory(args.disk_mb)
        print(_render_devices(inv))
        return 0

    if args.compare:
        path = os.path.join(args.output_dir, "benchmarks.csv")
        print(render_table(load_history(path), all_runs=args.all_runs))
        return 0

    # A/B: compare two saved runs statistically. Exits 6 on a regression so a
    # CI job can gate on "this change made something slower" with evidence
    # rather than on a raw percentage that may be noise.
    if args.compare_runs:
        try:
            payloads = []
            for path in args.compare_runs:
                with open(path, encoding="utf-8") as f:
                    payloads.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        verdict = stats_mod.compare_payloads(payloads[0], payloads[1],
                                             alpha=args.alpha)
        if args.json_stdout:
            print(json.dumps(verdict, indent=2, default=str))
        else:
            report_mod.hr("A/B comparison")
            print(f"  baseline : {args.compare_runs[0]}")
            print(f"  candidate: {args.compare_runs[1]}\n")
            print(stats_mod.render_payload_comparison(verdict))
        return 6 if verdict["regressions"] else 0

    # Diagnosis is its own job: the question is not "how fast is this"
    # but "what is holding it back", and the evidence for that is mostly
    # not a benchmark result.
    if args.checkup:
        if not args.json_stdout:
            report_mod.hr("Checkup — what is holding this machine back")
            if not args.no_measure:
                print("  reading system state, then measuring briefly ...\n")
        result = checkup_mod.run(_repo_root(), measure=not args.no_measure,
                                 disk_dir=None if args.no_save
                                 else args.output_dir)
        if args.json_stdout:
            print(json.dumps({"tool": "pcbench", "mode": "checkup",
                              "version": __version__, "checkup": result},
                             indent=2, default=str))
        else:
            print(checkup_mod.render(result))
        # A caller can gate on this: 1 for something actively hurting, 0 for
        # a clean bill or advisory findings only.
        return 1 if result["counts"].get("critical") else 0

    # A drive-speed check is the whole job: it answers "how fast is this
    # disk?" without the twenty other tests that surround the same numbers
    # in a full run.
    if args.drive_speed is not None:
        requested = [m.strip() for m in args.drive_speed.split(",")
                     if m.strip() and m.strip().lower() != "all"]
        inv = storage_mod.inventory(args.disk_mb)
        chosen = storage_mod.targets(inv, requested or None,
                                     all_devices=not requested)
        if not chosen:
            print("error: no benchmarkable filesystem found. "
                  "Use --list-devices to see what was detected, or name a "
                  "mount with --drive-speed PATH", file=sys.stderr)
            return 2
        seconds, repeats = (1.0, 2) if args.quick else (args.seconds,
                                                        max(1, args.repeats))
        if not args.json_stdout:
            report_mod.hr(f"Drive read/write speed — {len(chosen)} device(s)")
        result = storage_mod.run(chosen, seconds, repeats, args.disk_mb)
        if args.json_stdout:
            print(json.dumps({"tool": "pcbench", "mode": "drive_speed",
                              "version": __version__, "storage": result},
                             indent=2, default=str))
        else:
            print(storage_mod.render_speeds(result))
        return 0

    # An internet measurement is opt-in and stands alone: it spends the
    # user's bandwidth on a third party, which no benchmark run should do as
    # a side effect.
    if args.internet:
        if not args.json_stdout:
            report_mod.hr("Internet speed")
            print(f"  measuring against {args.internet_server} ...\n")
        result = network.internet_speed(args.internet_server,
                                        args.internet_seconds,
                                        args.internet_max_mb)
        if args.json_stdout:
            print(json.dumps({"tool": "pcbench", "mode": "internet",
                              "version": __version__, "internet": result},
                             indent=2, default=str))
        else:
            print(network.render_internet(result))
        down = (result.get("download") or {}).get("error")
        return 2 if (result.get("error") or down) else 0

    # The receiving half of the two-node network test. Blocks until stopped,
    # so it cannot be combined with a benchmark run.
    if args.net_server:
        report_mod.hr(f"Network server on port {args.net_port}")
        result = network.serve(args.net_port)
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
            return 2
        print(f"\n  {result['sessions']} session(s), "
              f"{result['bytes_received'] / (1024 ** 2):,.0f} MB received")
        return 0

    # The measuring half. A network measurement is the whole job, so it does
    # not run alongside the benchmark suite.
    if args.net_client:
        report_mod.hr(f"Network peer test — {args.net_client}")
        result = network.run_peer(args.net_client, args.net_port,
                                  min(args.seconds * 2, 10.0),
                                  args.net_streams)
        if args.json_stdout:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(network.render_peer(result))
        failed = (result.get("latency") or {}).get("error")
        return 2 if failed else 0

    # Monitor mode replaces the benchmark entirely: it answers "what is this
    # machine doing right now?", which no benchmark result can.
    if args.monitor:
        try:
            seconds = parse_duration(args.monitor)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        info = inventory()
        watching = args.json_stdout
        if not watching:
            report_mod.hr(f"Live monitor — {seconds:.0f}s "
                          f"at {args.monitor_interval:g}s intervals")
        result = monitor_mod.run(seconds, args.monitor_interval, _repo_root(),
                                 info.get("cpu_model", ""), quiet=watching,
                                 with_power=args.monitor_power)
        if args.monitor_trace:
            try:
                monitor_mod.save_trace(result, args.monitor_trace)
            except OSError as e:
                print(f"  ! could not write trace: {e}", file=sys.stderr)
        if watching:
            print(json.dumps({"tool": "pcbench", "mode": "monitor",
                              "version": __version__, "system": info,
                              "monitor": result}, indent=2, default=str))
        else:
            report_mod.hr("Summary")
            print(monitor_mod.render(result))
            if args.monitor_trace:
                print(f"\n  trace: {args.monitor_trace}")
        return 0

    if args.quick:
        args.seconds, args.repeats, args.disk_mb = 1.0, 2, 64

    try:
        if args.profile:
            if args.profile not in PROFILES:
                raise ValueError(
                    f"unknown profile: {args.profile!r}. "
                    f"Valid: {', '.join(PROFILES)}")
            if args.only:
                raise ValueError("--profile and --only are mutually exclusive")
            args.only = ",".join(PROFILES[args.profile])
        selected = select_tests(args.only, args.skip)
        sustained_seconds = (parse_duration(args.sustained)
                             if args.sustained else None)
        soak_seconds = parse_duration(args.soak) if args.soak else None
        # Malformed assertions are caught here rather than after the run: a
        # typo in a CI gate must not cost ten minutes to discover.
        for expression in args.assert_:
            gates_mod.parse(expression)
    except (ValueError, gates_mod.GateError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    quiet = args.json_stdout

    # Fail fast on an unwritable output directory rather than after the run.
    if not args.no_save:
        problem = _check_output_writable(args.output_dir)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 5

    disk_dir = _scratch_dir() if args.no_save else args.output_dir
    try:
        os.makedirs(disk_dir, exist_ok=True)
    except OSError:
        disk_dir = _scratch_dir()

    info = inventory()
    state = machine_state(_repo_root())
    from . import thermal as thermal_mod
    state["battery"] = thermal_mod.battery_health() or None
    warnings = state_warnings(state)

    # What this process is actually allowed to use, which in a container, a
    # cgroup, or a CI runner is not what the hardware has.
    confinement = container_mod.detect(info.get("cpu_cores_logical"),
                                       info.get("ram_total_bytes", 0))
    confinement_warnings = container_mod.warnings(confinement)

    if not quiet:
        report_mod.hr(f"PC Benchmark & Diagnostics v{__version__}")
        print(f"  seconds/test={args.seconds}  repeats={args.repeats}  "
              f"tests={','.join(selected) or 'none'}")
        if config_info.get("path"):
            print(f"  config: {config_info['path']} "
                  f"({len(config_info.get('applied') or {})} setting(s))")
        for note in confinement_warnings:
            print(f"  i  {note}")

    autoscale_notes = ([] if args.no_autoscale
                       else _autoscale(args, info, confinement, quiet))

    # Multicore workloads size themselves to the *effective* core count so a
    # quota-limited container does not spawn workers that only contend.
    effective_cores = confinement.get("effective_cores") or info[
        "cpu_cores_logical"]

    # Distorting conditions are worth stopping for: a result taken on battery
    # or under load looks like a hardware difference but is not.
    if warnings and not args.force and not quiet:
        print()
        for w in warnings:
            print(f"  !!  {w}")
        print("\n  Re-run with --force to benchmark anyway.")
        return 3

    # Resource counters bracket the whole benchmark phase: they are free, and
    # page faults or involuntary context switches during it invalidate
    # everything else that was measured.
    counters_before = counters_mod.resource_snapshot()

    results: dict = {}
    run_info = dict(info, cpu_cores_logical=effective_cores)
    if confinement.get("effective_ram_bytes"):
        run_info["ram_total_bytes"] = confinement["effective_ram_bytes"]
    runners = _runners(args, run_info, disk_dir)
    for name in selected:
        if not quiet:
            print(f"  running {name} ...", flush=True)
        before = interference.sample(_repo_root())
        try:
            results[name] = runners[name]()
            # Conditions are checked around each test, not just once at the
            # start: a run takes minutes and the machine can change underneath
            # it, which no amount of repetition inside the test would reveal.
            if isinstance(results[name], dict):
                verdict = interference.compare_samples(
                    before, interference.sample(_repo_root()), name)
                if verdict["disturbed"]:
                    results[name]["interference"] = verdict
                    if not quiet:
                        for note in verdict["notes"]:
                            print(f"    ! {name}: {note}", file=sys.stderr)
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
        cache_bytes, cache_source = system_mod.last_level_cache_bytes()
        native = native_mod.run(
            args.seconds, args.repeats, _repo_root(),
            threads=info["cpu_cores_logical"],
            stream_mb=native_mod.stream_array_mb(
                cache_bytes, info.get("ram_total_bytes", 0)),
            disk_dir=disk_dir)
        if isinstance(native, dict):
            native["last_level_cache_bytes"] = cache_bytes
            native["last_level_cache_source"] = cache_source

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

    # AI framework tier — only touches a third-party dependency, and only when
    # the user has one installed (or forces it).
    ml = None
    ml_detected = mlframework.detect()
    if not args.no_ai and (args.ai or ml_detected["available"]):
        if ml_detected["available"]:
            if not quiet:
                print("  running AI framework benchmark ...", flush=True)
            ml = mlframework.run(args.seconds, args.ai_batch)
            for key, value in mlframework.extract_rates(ml).items():
                results[key] = {"rate": value}
        elif args.ai:
            ml = mlframework.run(args.seconds)  # returns the "not found" note

    # Local network stack.
    net = None
    if not args.no_network:
        if not quiet:
            print("  running network benchmark ...", flush=True)
        net = network.run(min(args.seconds, 2.0))

    # User-supplied benchmarks from plugins/, treated like built-ins.
    plugin_results = None
    if not args.no_plugins:
        found = plugins_mod.discover(_repo_root())
        if found:
            if not quiet:
                print(f"  running {len(found)} plugin(s) ...", flush=True)
            plugin_results = plugins_mod.run_all(
                found, args.seconds, args.repeats)

    # Hardware health checks — opt-in, since the RAM test takes a while.
    health_result = None
    if args.health:
        if not quiet:
            print("  running health checks ...", flush=True)
        health_result = health.run(args.health_mb,
                                   info.get("ram_total_bytes", 0))

    # External network tests only run when a target was named.
    if net is not None and (args.network_host or args.network_url):
        if not quiet:
            print("  running external network tests ...", flush=True)
        net["external"] = network.run_external(args.network_host or None,
                                               args.network_url or None)

    # Power / perf-per-watt, sampled under load.
    power_info = None
    if not args.no_power:
        if not quiet:
            print("  measuring power ...", flush=True)
        power_info = power.measure_under_load(info.get("cpu_model", ""))

    # Optional-package tiers: BLAS numerics, hardware crypto, OpenCL GPU.
    # Each is skipped silently when its packages are absent.
    numeric_result = crypto_result = opencl_result = None
    if not args.no_optional:
        if numeric.available()["numpy"]:
            if not quiet:
                print("  running BLAS/LAPACK numerics ...", flush=True)
            numeric_result = numeric.run(min(args.seconds, 1.5),
                                         max(1, min(args.repeats, 2)))
            for key, value in numeric.extract_rates(numeric_result).items():
                results[key] = {"rate": value}

        if any(cryptobench.available().values()):
            if not quiet:
                print("  running crypto / compression ...", flush=True)
            crypto_result = cryptobench.run(min(args.seconds, 1.0),
                                            max(1, min(args.repeats, 2)))
            for key, value in cryptobench.extract_rates(crypto_result).items():
                results[key] = {"rate": value}

        if accel_enabled(args, "gpu") and gpucompute.available()["pyopencl"]:
            if not quiet:
                print("  running OpenCL GPU compute ...", flush=True)
            opencl_result = gpucompute.run(min(args.seconds, 1.5))
            for key, value in gpucompute.extract_rates(opencl_result).items():
                results[key] = {"rate": value}

    # Cross-vendor NPU via ONNX Runtime (Intel / AMD / Qualcomm / DirectML).
    npu_result = None
    if accel_enabled(args, "npu"):
        if npu_mod.detect().get("available"):
            if not quiet:
                print("  running ONNX Runtime NPU benchmark ...", flush=True)
            npu_result = npu_mod.run(min(args.seconds, 3.0), disk_dir)
            for key, value in npu_mod.extract_rates(npu_result).items():
                results[key] = {"rate": value}
        else:
            npu_result = npu_mod.detect()

    # rusage costs nothing and is always collected; the PMU tier only when
    # asked, since it costs an extra subprocess run of a CPU workload.
    counters_result = {
        "resources": counters_mod.resource_delta(
            counters_before, counters_mod.resource_snapshot()),
    }
    if args.counters:
        if not quiet:
            print("  reading hardware performance counters ...", flush=True)
        counters_result["pmu"] = counters_mod.measure_command(
            [sys.executable, "-c",
             "import pcbench.workloads as w; w.bench_cpu_integer(1.0, 1)"])
    counters_result["notes"] = counters_mod.interpret(
        counters_result.get("pmu") or {}, counters_result["resources"])

    # Industry reference workloads. STREAM and the CoreMark-style suite come
    # from the native engine at no extra cost; LINPACK needs NumPy.
    standards_result = None
    if not args.no_standards:
        if not quiet:
            print("  collecting reference workloads (STREAM / LINPACK / "
                  "CoreMark-style) ...", flush=True)
        standards_result = standards_mod.run(
            native, info.get("ram_total_bytes", 0),
            with_linpack=not args.no_linpack)
        for key, value in standards_mod.extract_rates(standards_result).items():
            results[key] = {"rate": value}

    numa_result = None
    if args.numa or args.numa_bandwidth:
        if not quiet:
            print("  inspecting NUMA topology ...", flush=True)
        numa_result = numa_mod.run(measure=args.numa_bandwidth)
        numa_result["notes"] = numa_mod.notes(numa_result)

    datascience_result = None
    if args.datascience:
        if not quiet:
            print("  running the data-science tier ...", flush=True)
        datascience_result = ds_mod.run(
            memory_bytes=(confinement.get("effective_ram_bytes")
                          or info.get("ram_total_bytes", 0)),
            seconds=args.seconds,
            skip_dataframes=args.no_dataframes)
        for key, value in ds_mod.extract_rates(datascience_result).items():
            results[key] = {"rate": value}

    io_result = None
    if args.io or args.io_job:
        try:
            jobs = ([iobench.parse_job(spec, min(args.seconds, 3.0),
                                       args.disk_mb) for spec in args.io_job]
                    if args.io_job
                    else iobench.default_suite(min(args.seconds, 3.0),
                                               args.disk_mb))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if not quiet:
            print(f"  running {len(jobs)} storage I/O job(s) ...", flush=True)
        io_result = iobench.run(jobs, disk_dir, quiet=quiet)

    # Drive lifetime is read-only, costs a few milliseconds, and answers a
    # question no benchmark can: how much life the storage has left.
    drive_life = None
    if not args.no_drive_life:
        drive_life = drivelife.run(_repo_root())

    energy_result = None
    if args.energy:
        if not quiet:
            print("  measuring energy to solution ...", flush=True)

        def fixed_work() -> int:
            # A fixed amount of work, so joules are comparable between
            # machines of different speeds -- which is the entire point.
            iterations = 2_000_000
            total = 0
            for i in range(iterations):
                total += i * i
            return iterations

        energy_result = power.energy_to_solution(
            fixed_work, info.get("cpu_model", ""), "2M integer operations")

    # Extra storage devices, benchmarked with the same workload as the main
    # disk test so the numbers sit side by side.
    storage_result = None
    if args.disk_path or args.disk_all:
        requested = [p.strip() for p in args.disk_path.split(",") if p.strip()]
        inv = storage_mod.inventory(args.disk_mb)
        chosen = storage_mod.targets(inv, requested or None, args.disk_all)
        if chosen:
            if not quiet:
                print(f"  benchmarking {len(chosen)} storage device(s) ...",
                      flush=True)
            storage_result = storage_mod.run(chosen, min(args.seconds, 3.0),
                                             max(1, min(args.repeats, 2)),
                                             args.disk_mb)
            storage_result["inventory"] = inv
        else:
            storage_result = {"devices": [], "inventory": inv,
                              "note": "no writable local filesystem qualified"}

    sustained = None
    if sustained_seconds:
        workers = args.sustained_workers or effective_cores
        if not quiet:
            print(f"  running sustained load for {sustained_seconds:.0f}s "
                  f"on {workers} worker(s) ...", flush=True)
        sustained = run_sustained(sustained_seconds, args.sustained_window,
                                  workers, _repo_root())

    # Burn-in. Runs last because it is the longest phase by far and everything
    # before it should already be recorded if the machine falls over.
    soak_result = None
    if soak_seconds:
        workers = args.soak_workers or effective_cores
        if not quiet:
            print(f"  soaking for {soak_seconds:.0f}s on {workers} worker(s) "
                  f"— press Ctrl-C to stop early and keep the findings ...",
                  flush=True)
        soak_result = soak_mod.run(soak_seconds, workers,
                                   script_dir=_repo_root(), quiet=quiet)

    scores = compute_scores(results)
    if plugin_results:
        # Plugins score against their own declared baselines and join the
        # composite like any built-in metric.
        scores["subscores"].update(plugins_mod.scores(plugin_results))
        if scores["subscores"]:
            import math as _math
            import statistics as _stats
            scores["composite"] = round(_math.exp(_stats.fmean(
                _math.log(v) for v in scores["subscores"].values() if v > 0)), 1)
    ppw = power.perf_per_watt(scores["composite"], power_info) if power_info \
        else None

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
        "interference": interference.summarize(results),
        "results": results,
        "native": native,
        "accelerators": accel_inv,
        "accel": accel,
        "ml_framework": ml,
        "npu_onnx": npu_result,
        "numeric": numeric_result,
        "crypto": crypto_result,
        "opencl": opencl_result,
        "optional_packages": optional.status(),
        "network": net,
        "power": power_info,
        "perf_per_watt": ppw,
        "sustained": sustained,
        "soak": soak_result,
        "storage": storage_result,
        "standards": standards_result,
        "counters": counters_result,
        "numa": numa_result,
        "datascience": datascience_result,
        "io": io_result,
        "energy": energy_result,
        "drive_life": drive_life,
        "provenance": (None if args.no_provenance else provenance.collect()),
        "confinement": confinement,
        "confinement_warnings": confinement_warnings,
        "autoscale": autoscale_notes,
        "config_file": config_info or None,
        "scores": scores,
        "bottleneck": diagnose.analyse(scores),
        "plugins": plugin_results,
        "health": health_result,
    }
    # Placing the machine and checking it against its own single-core anchor
    # needs the finished scores, so it happens after the payload is assembled.
    payload["reference"] = reference.assess(payload)
    payload["subsystem_checks"] = reference.subsystem_checks(
        results, scores.get("subscores"))
    if payload.get("provenance"):
        payload["provenance_notes"] = provenance.notes(payload["provenance"])

    # Regression check against this machine's own history.
    if not args.no_regression:
        history = load_history(os.path.join(args.output_dir, "benchmarks.csv"))
        current_row = report_mod.flatten_row(payload)
        payload["regression"] = regression.analyze(
            current_row, history, args.regression_threshold / 100.0)

    # Threshold gates. Evaluated before output so their verdicts can be
    # embedded in the payload, the JUnit report, and the terminal summary.
    gate_results = gates_mod.evaluate(payload, list(args.assert_),
                                      args.fail_under)
    if gate_results:
        payload["gates"] = gate_results

    if args.json_stdout:
        print(json.dumps(payload, indent=2, default=str))
    else:
        report_mod.print_report(payload)
        if gate_results:
            report_mod.hr("Thresholds")
            print(gates_mod.render(gate_results))

    if not args.no_save:
        try:
            jp = report_mod.save_json(payload, args.output_dir)
            cp = report_mod.append_csv(payload, args.output_dir)
            hp = (report_mod.save_html(payload, args.output_dir)
                  if args.html else None)
            sp = (report_mod.save_spec_sheet(payload, args.output_dir)
                  if args.spec_sheet else None)
            if not quiet:
                report_mod.hr("Saved")
                print(f"  JSON: {jp}")
                print(f"  CSV : {cp}")
                if hp:
                    print(f"  HTML: {hp}")
                if sp:
                    print(f"  SPEC: {sp}")
        except OSError as e:
            print(f"  ! could not save results: {e}", file=sys.stderr)

    # Integration exports are written even with --no-save: their whole purpose
    # is to feed another system, and "do not litter the results directory" is
    # not a reason to withhold a file the user named explicitly.
    for label, path, writer in (
            ("PROM", args.prometheus, export_mod.save_prometheus),
            ("SQLITE", args.sqlite, export_mod.save_sqlite),
            ("MD", args.markdown, export_mod.save_markdown)):
        if not path:
            continue
        try:
            writer(payload, path)
            if not quiet:
                print(f"  {label}: {path}")
        except Exception as e:
            # An export failing must never discard a finished benchmark: the
            # results are already printed and saved by this point.
            print(f"  ! could not write {path}: {e}", file=sys.stderr)
    if args.junit:
        try:
            export_mod.save_junit(payload, args.junit, gate_results)
            if not quiet:
                print(f"  JUNIT: {args.junit}")
        except OSError as e:
            print(f"  ! could not write {args.junit}: {e}", file=sys.stderr)

    # Exit codes, most severe first, so a caller checking `$?` gets the worst
    # thing that happened rather than the last.
    if any(isinstance(v, dict) and v.get("validation_failed")
           for v in results.values()):
        return 4
    if soak_result and soak_result.get("errors"):
        return 7
    if gates_mod.failed(gate_results):
        return 6
    return 0


def _render_devices(inv: dict) -> str:
    """Table of mounted storage and whether each can be benchmarked."""
    devices = inv.get("devices", [])
    if not devices:
        return "  no mounted filesystems could be enumerated"
    lines = [f"  {'MOUNT':<28} {'KIND':<22} {'FREE':>9}  STATUS"]
    lines.append("  " + "-" * 76)
    for d in devices:
        free = d.get("free_bytes")
        free_text = f"{free / (1024 ** 3):.1f} GB" if free else "?"
        status = ("benchmarkable" if d["benchmarkable"]
                  else f"skip: {d.get('skip_reason', 'unavailable')}")
        lines.append(f"  {d['mount'][:28]:<28} {(d.get('kind') or '?')[:22]:<22} "
                     f"{free_text:>9}  {status}")
    lines.append("")
    lines.append(f"  {inv.get('benchmarkable_count', 0)} device(s) can be "
                 f"benchmarked. Use --disk-all, or --disk-path MOUNT[,MOUNT].")
    return "\n".join(lines)


def _repo_root() -> str:
    """Directory holding native_engine.c (the package's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def entry() -> None:
    mp.freeze_support()
    sys.exit(main())
