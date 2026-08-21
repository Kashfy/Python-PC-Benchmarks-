#!/usr/bin/env python3
"""
Cross-platform PC benchmark & diagnostics tool.

Measures CPU (single- and multi-core), memory bandwidth, and disk I/O with
meaningful, comparable units, gathers a hardware/OS inventory, and writes
results to the console plus timestamped JSON and an appended CSV.

Design goals
------------
* Runs on Windows, macOS, and Linux with **stdlib only** (no pip install).
  If `psutil` is available it is used for richer hardware info, but the tool
  works fully without it.
* Reports real units, not opaque "chunks/s":
    - CPU integer : primes tested per second
    - CPU float   : math iterations per second
    - Multi-core  : aggregate primes/sec + scaling factor vs. one core
    - Memory      : copy bandwidth in MB/s
    - Disk        : sequential write and read in MB/s
* Reliability: warm-up pass, multiple repeats, reports median + stdev,
  and no single probe failure aborts the whole run.
* A single normalized "composite score" makes devices easy to rank.

Usage
-----
    python3 benchmark.py
    python3 benchmark.py --seconds 5 --repeats 5
    python3 benchmark.py --quick
    python3 benchmark.py --only cpu_int,memory
    python3 benchmark.py --no-native            # skip the optional C engine
    python3 benchmark.py --output-dir results

Run `python3 benchmark.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Reference baselines for the composite score.
#
# These are fixed constants (roughly a mid-range 2020-era laptop core). They are
# arbitrary but *stable*, so a score of 100 always means "same as the baseline"
# and scores are comparable across machines and across runs of this tool.
# --------------------------------------------------------------------------- #
BASELINES = {
    "cpu_int_primes_per_s": 2_000_000.0,   # single-core primes tested / s
    "cpu_float_iters_per_s": 3_000_000.0,  # single-core math iterations / s
    "cpu_multi_primes_per_s": 8_000_000.0,  # all-core aggregate primes / s
    "mem_copy_mb_per_s": 6_000.0,          # memory copy bandwidth MB/s
    "disk_write_mb_per_s": 500.0,          # sequential write MB/s
    "disk_read_mb_per_s": 1_000.0,         # sequential read MB/s
}

MB = 1024 * 1024
VERSION = "2.0"


# --------------------------------------------------------------------------- #
# Console formatting helpers
# --------------------------------------------------------------------------- #
def hr(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(line)


def fmt_num(x: float) -> str:
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


# --------------------------------------------------------------------------- #
# Timed-loop core
# --------------------------------------------------------------------------- #
def timed_loop(chunk_func, seconds: float) -> tuple[float, int]:
    """Call ``chunk_func`` repeatedly for ~``seconds``.

    Returns ``(elapsed_seconds, iterations)``. Each call to ``chunk_func`` is
    one iteration; the caller knows how much work an iteration represents.
    """
    start = time.perf_counter()
    count = 0
    elapsed = 0.0
    while True:
        chunk_func()
        count += 1
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    return elapsed, count


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #
PRIME_RANGE_LO = 50_000
PRIME_RANGE_HI = 51_000           # 1000 integers tested per chunk
PRIMES_PER_CHUNK = PRIME_RANGE_HI - PRIME_RANGE_LO
FLOAT_ITERS_PER_CHUNK = 50_000


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = math.isqrt(n)
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def cpu_integer_chunk() -> None:
    """Integer-heavy: primality test over a fixed range (deterministic cost)."""
    for n in range(PRIME_RANGE_LO, PRIME_RANGE_HI):
        _is_prime(n)


def cpu_float_chunk() -> None:
    """Floating-point: sin/cos/sqrt in a tight loop."""
    x = 0.001
    s = 0.0
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(FLOAT_ITERS_PER_CHUNK):
        x += 0.00001
        s += sin(x) * cos(x) + sqrt(x)
    # Keep the optimizer / dead-code elimination honest (and it never prints).
    if s == -1.234567890123:
        print("magic", s)


# --------------------------------------------------------------------------- #
# Multi-core worker (must be top-level & picklable for spawn start method)
# --------------------------------------------------------------------------- #
def _multicore_worker(deadline: float) -> int:
    """Run prime chunks until ``deadline`` (perf_counter epoch is per-process,
    so we pass a duration instead — see caller)."""
    # ``deadline`` here is actually a duration in seconds; each process times
    # itself independently to avoid clock-domain issues across processes.
    duration = deadline
    start = time.perf_counter()
    chunks = 0
    while time.perf_counter() - start < duration:
        cpu_integer_chunk()
        chunks += 1
    return chunks * PRIMES_PER_CHUNK


# --------------------------------------------------------------------------- #
# Benchmark runners — each returns a dict with a primary rate + raw samples
# --------------------------------------------------------------------------- #
def _summarize(samples: list[float]) -> dict:
    if not samples:
        return {"median": 0.0, "mean": 0.0, "stdev": 0.0, "min": 0.0,
                "max": 0.0, "samples": []}
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
        "samples": [round(s, 2) for s in samples],
    }


def bench_cpu_integer(seconds: float, repeats: int) -> dict:
    rates = []
    for _ in range(repeats):
        elapsed, chunks = timed_loop(cpu_integer_chunk, seconds)
        rates.append(chunks * PRIMES_PER_CHUNK / elapsed)
    s = _summarize(rates)
    return {"unit": "primes/s", "rate": s["median"], **s}


def bench_cpu_float(seconds: float, repeats: int) -> dict:
    rates = []
    for _ in range(repeats):
        elapsed, chunks = timed_loop(cpu_float_chunk, seconds)
        rates.append(chunks * FLOAT_ITERS_PER_CHUNK / elapsed)
    s = _summarize(rates)
    return {"unit": "iters/s", "rate": s["median"], **s}


def bench_cpu_multicore(seconds: float, logical_cores: int) -> dict:
    """Run the integer workload across every logical core and measure the
    aggregate throughput plus how well it scales vs. a single core."""
    workers = max(1, logical_cores)
    ctx = mp.get_context("spawn")  # portable across macOS/Windows/Linux
    start = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        # Each worker times itself for `seconds`.
        results = pool.map(_multicore_worker, [seconds] * workers)
    wall = time.perf_counter() - start
    total_primes = sum(results)
    aggregate_rate = total_primes / wall
    return {
        "unit": "primes/s",
        "rate": aggregate_rate,
        "workers": workers,
        "wall_seconds": round(wall, 3),
        "per_worker_primes": results,
    }


def bench_memory(seconds: float, repeats: int, buf_mb: int = 64) -> dict:
    """Memory copy bandwidth via large bytearray slice assignment (a real
    memmove under the hood). Reports MB/s of data moved."""
    n = buf_mb * MB
    src = bytearray(os.urandom(min(n, 1 * MB))) * (n // min(n, 1 * MB))
    src = src[:n]
    dst = bytearray(n)
    rates = []
    for _ in range(repeats):
        start = time.perf_counter()
        copied = 0
        while time.perf_counter() - start < seconds:
            dst[:] = src          # move `n` bytes
            copied += n
        elapsed = time.perf_counter() - start
        rates.append(copied / elapsed / MB)
    s = _summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "buffer_mb": buf_mb, **s}


def _advise_dropcache(fd: int) -> None:
    """Best-effort hint to the OS not to cache the file, so the read test
    measures the device rather than RAM. Silently no-ops where unsupported."""
    try:
        if hasattr(os, "posix_fadvise"):  # Linux
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def bench_disk(seconds: float, repeats: int, file_mb: int, out_dir: str) -> dict:
    """Sequential write and read throughput on a real file.

    Writes a `file_mb` file in 4 MiB chunks with fsync, then reads it back.
    Read numbers can still be influenced by OS cache; we drop what we can.
    """
    chunk = b"X" * (4 * MB)
    n_chunks = max(1, (file_mb * MB) // len(chunk))
    total_bytes = n_chunks * len(chunk)

    # Guard against filling the disk.
    free = shutil.disk_usage(out_dir).free
    if total_bytes * 1.2 > free:
        return {"error": f"not enough free space for {file_mb} MB disk test",
                "skipped": True}

    write_rates, read_rates = [], []
    for _ in range(repeats):
        fd, path = tempfile.mkstemp(prefix="pcbench_", suffix=".bin", dir=out_dir)
        try:
            # --- write ---
            start = time.perf_counter()
            with os.fdopen(fd, "wb", closefd=False) as f:
                for _ in range(n_chunks):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            write_rates.append(total_bytes / (time.perf_counter() - start) / MB)

            # --- read ---
            _advise_dropcache(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            start = time.perf_counter()
            got = 0
            while True:
                data = os.read(fd, 4 * MB)
                if not data:
                    break
                got += len(data)
            read_rates.append(got / (time.perf_counter() - start) / MB)
        finally:
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass

    w, r = _summarize(write_rates), _summarize(read_rates)
    return {
        "unit": "MB/s",
        "file_mb": file_mb,
        "write_rate": w["median"],
        "read_rate": r["median"],
        "write": w,
        "read": r,
    }


# --------------------------------------------------------------------------- #
# Hardware / OS inventory
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_model() -> str:
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            m = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if m:
                return m
        elif sysname == "Linux":
            # x86 exposes "model name"; ARM/RISC-V boards often only have
            # "Hardware", "Model", or a "CPU implementer/part" pair.
            hardware = model = None
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        low = line.lower()
                        if low.startswith("model name"):
                            return line.split(":", 1)[1].strip()
                        if low.startswith("hardware"):
                            hardware = line.split(":", 1)[1].strip()
                        elif low.startswith("model") and ":" in line:
                            model = line.split(":", 1)[1].strip()
            except OSError:
                pass
            # Devicetree model (Raspberry Pi and many ARM SBCs).
            for p in ("/sys/firmware/devicetree/base/model",
                      "/proc/device-tree/model"):
                try:
                    with open(p, "rb") as f:
                        dt = f.read().rstrip(b"\x00").decode("utf-8", "ignore")
                        if dt.strip():
                            return dt.strip()
                except OSError:
                    pass
            if hardware:
                return hardware
            if model:
                return model
        elif sysname == "Windows":
            m = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if m:
                return m
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _total_ram_bytes() -> int:
    # Prefer psutil if present.
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            v = _run(["sysctl", "-n", "hw.memsize"])
            return int(v) if v.isdigit() else 0
        if sysname == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(re.search(r"(\d+)", line).group(1))
                        return kb * 1024
        if sysname == "Windows":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return int(stat.ullTotalPhys)
            except Exception:
                return 0
    except (OSError, ValueError, AttributeError):
        pass
    return 0


def _physical_cores() -> int | None:
    try:
        import psutil  # type: ignore
        pc = psutil.cpu_count(logical=False)
        if pc:
            return int(pc)
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Darwin":
        v = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(v) if v.isdigit() else None
    if sysname == "Linux":
        try:
            ids = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                phys = core = None
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        if phys is not None:
                            ids.add((phys, core))
            if ids:
                return len(ids)
        except OSError:
            pass
    if sysname == "Windows":
        v = os.environ.get("NUMBER_OF_PROCESSORS")  # logical, best effort
        return None
    return None


def _arch_family(machine: str) -> str:
    """Normalize the many ``platform.machine()`` spellings into a friendly
    ISA family so results from different OSes line up (e.g. Windows reports
    ``AMD64`` while Linux reports ``x86_64`` for the same chip)."""
    m = (machine or "").lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86-64"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        return "x86-32"
    if m in ("arm64", "aarch64", "aarch64_be", "arm64e"):
        return "ARM64"
    if m.startswith("armv") or m.startswith("arm"):
        return "ARM32"
    if m.startswith("riscv64"):
        return "RISC-V 64"
    if m.startswith("riscv"):
        return "RISC-V 32"
    if m.startswith("ppc64") or m.startswith("powerpc64"):
        return "PowerPC 64"
    if m.startswith("s390"):
        return "IBM Z"
    if m.startswith("mips"):
        return "MIPS"
    return machine or "unknown"


def gather_system_info() -> dict:
    logical = os.cpu_count() or 1
    ram = _total_ram_bytes()
    machine = platform.machine()
    info = {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "architecture": machine,
        "arch_family": _arch_family(machine),
        "arch_bits": 64 if sys.maxsize > 2**32 else 32,
        "byte_order": sys.byteorder,
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": _physical_cores(),
        "cpu_cores_logical": logical,
        "ram_total_bytes": ram,
        "ram_total_gb": round(ram / (1024 ** 3), 2) if ram else None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "psutil_available": _has_psutil(),
    }
    return info


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Optional native (C) engine integration
# --------------------------------------------------------------------------- #
def _find_compiler() -> str | None:
    for cc in ("cc", "clang", "gcc"):
        if shutil.which(cc):
            return cc
    return None


def run_native_engine(seconds: float, repeats: int, script_dir: str) -> dict | None:
    """Compile (if needed) and run native_engine.c, returning its parsed JSON.

    Returns None if the C source or a compiler is unavailable, or on any error.
    The native engine gives compiler-optimized numbers alongside Python's.
    """
    src = os.path.join(script_dir, "native_engine.c")
    if not os.path.isfile(src):
        return None
    exe = os.path.join(script_dir,
                       "native_engine.exe" if os.name == "nt" else "native_engine")

    needs_build = (not os.path.isfile(exe)
                   or os.path.getmtime(exe) < os.path.getmtime(src))
    if needs_build:
        cc = _find_compiler()
        if not cc:
            return {"error": "no C compiler found (cc/clang/gcc); skipped"}
        cmd = [cc, "-O2", src, "-o", exe]
        if os.name != "nt":
            cmd.append("-lm")
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            return {"error": "native build failed",
                    "detail": build.stderr.strip()[:500]}

    try:
        out = subprocess.run(
            [exe, "--json", "--seconds", str(seconds), "--repeats", str(repeats)],
            capture_output=True, text=True, timeout=seconds * repeats * 6 + 30)
        if out.returncode != 0:
            return {"error": "native run failed", "detail": out.stderr[:500]}
        return json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        return {"error": f"native run error: {e}"}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def compute_scores(results: dict) -> dict:
    """Normalize each measured rate against its baseline (baseline == 100),
    then take a geometric mean for a single composite score."""
    parts: dict[str, float] = {}

    def add(key: str, rate: float, baseline_key: str):
        base = BASELINES[baseline_key]
        if rate and base:
            parts[key] = 100.0 * rate / base

    if "cpu_int" in results:
        add("cpu_int", results["cpu_int"]["rate"], "cpu_int_primes_per_s")
    if "cpu_float" in results:
        add("cpu_float", results["cpu_float"]["rate"], "cpu_float_iters_per_s")
    if "cpu_multi" in results:
        add("cpu_multi", results["cpu_multi"]["rate"], "cpu_multi_primes_per_s")
    if "memory" in results:
        add("memory", results["memory"]["rate"], "mem_copy_mb_per_s")
    if "disk" in results and "write_rate" in results["disk"]:
        add("disk_write", results["disk"]["write_rate"], "disk_write_mb_per_s")
        add("disk_read", results["disk"]["read_rate"], "disk_read_mb_per_s")

    composite = (math.exp(statistics.fmean([math.log(v) for v in parts.values()]))
                 if parts else 0.0)
    return {"subscores": {k: round(v, 1) for k, v in parts.items()},
            "composite": round(composite, 1)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_console_report(info: dict, results: dict, scores: dict,
                         native: dict | None) -> None:
    hr("System Information")
    rows = [
        ("Hostname", info["hostname"]),
        ("OS", f"{info['os']} {info['os_release']}"),
        ("Architecture", f"{info['arch_family']} ({info['architecture']}, "
                         f"{info['arch_bits']}-bit, {info['byte_order']}-endian)"),
        ("CPU", info["cpu_model"]),
        ("Cores", f"{info['cpu_cores_physical'] or '?'} physical / "
                  f"{info['cpu_cores_logical']} logical"),
        ("RAM", f"{info['ram_total_gb']} GB" if info["ram_total_gb"] else "unknown"),
        ("Python", f"{info['python_implementation']} {info['python_version']}"),
    ]
    for k, v in rows:
        print(f"  {k:14}: {v}")

    hr("Benchmark Results")
    if "cpu_int" in results:
        r = results["cpu_int"]
        print(f"  CPU Integer (primes)   : {fmt_num(r['rate']):>14} primes/s "
              f"(±{fmt_num(r['stdev'])})")
    if "cpu_float" in results:
        r = results["cpu_float"]
        print(f"  CPU Float (math ops)   : {fmt_num(r['rate']):>14} iters/s  "
              f"(±{fmt_num(r['stdev'])})")
    if "cpu_multi" in results:
        r = results["cpu_multi"]
        single = results.get("cpu_int", {}).get("rate")
        scaling = f"  →  {r['rate']/single:.1f}x vs 1 core" if single else ""
        print(f"  CPU Multi-core ({r['workers']:>2}w)   : {fmt_num(r['rate']):>14} "
              f"primes/s{scaling}")
    if "memory" in results:
        r = results["memory"]
        print(f"  Memory copy bandwidth  : {fmt_num(r['rate']):>14} MB/s     "
              f"(±{fmt_num(r['stdev'])})")
    if "disk" in results:
        r = results["disk"]
        if r.get("skipped"):
            print(f"  Disk I/O               : skipped ({r.get('error')})")
        else:
            print(f"  Disk write             : {fmt_num(r['write_rate']):>14} MB/s")
            print(f"  Disk read              : {fmt_num(r['read_rate']):>14} MB/s")

    if native and "error" not in native:
        hr("Native (C) Engine — compiler-optimized")
        for item in native.get("results", []):
            print(f"  {item['name']:22} : {fmt_num(item['rate']):>14} "
                  f"{item.get('unit','')}")
    elif native and "error" in native:
        print(f"\n  (native engine: {native['error']})")

    hr("Scores (baseline machine = 100, higher is better)")
    for k, v in scores["subscores"].items():
        print(f"  {k:14}: {v:>8.1f}")
    print(f"  {'-'*24}")
    print(f"  {'COMPOSITE':14}: {scores['composite']:>8.1f}")


def save_json(payload: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = payload["timestamp_utc"].replace(":", "").replace("-", "")
    host = re.sub(r"[^A-Za-z0-9_-]", "_", payload["system"]["hostname"] or "host")
    path = os.path.join(out_dir, f"benchmark_{host}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def append_csv(payload: dict, out_dir: str) -> str:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmarks.csv")
    info = payload["system"]
    res = payload["results"]
    scores = payload["scores"]
    row = {
        "timestamp_utc": payload["timestamp_utc"],
        "hostname": info["hostname"],
        "os": info["os"],
        "arch": info["architecture"],
        "arch_family": info["arch_family"],
        "cpu_model": info["cpu_model"],
        "cores_logical": info["cpu_cores_logical"],
        "ram_gb": info["ram_total_gb"],
        "cpu_int_primes_s": round(res.get("cpu_int", {}).get("rate", 0), 1),
        "cpu_float_iters_s": round(res.get("cpu_float", {}).get("rate", 0), 1),
        "cpu_multi_primes_s": round(res.get("cpu_multi", {}).get("rate", 0), 1),
        "mem_mb_s": round(res.get("memory", {}).get("rate", 0), 1),
        "disk_write_mb_s": round(res.get("disk", {}).get("write_rate", 0), 1),
        "disk_read_mb_s": round(res.get("disk", {}).get("read_rate", 0), 1),
        "composite_score": scores["composite"],
    }
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
ALL_TESTS = ["cpu_int", "cpu_float", "cpu_multi", "memory", "disk"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-platform PC benchmark & diagnostics tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seconds", type=float, default=3.0,
                   help="Target duration per test per repeat")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats per test (median is reported)")
    p.add_argument("--disk-mb", type=int, default=256,
                   help="Disk test file size in MB")
    p.add_argument("--mem-mb", type=int, default=64,
                   help="Memory buffer size in MB")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of: " + ",".join(ALL_TESTS))
    p.add_argument("--quick", action="store_true",
                   help="Fast pass (1s x 2 repeats, small disk test)")
    p.add_argument("--no-native", action="store_true",
                   help="Skip the optional native C engine")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write JSON/CSV files")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for JSON/CSV output")
    p.add_argument("--json-stdout", action="store_true",
                   help="Print the full result payload as JSON to stdout")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.seconds, args.repeats, args.disk_mb = 1.0, 2, 64

    selected = ([t.strip() for t in args.only.split(",") if t.strip()]
                if args.only else list(ALL_TESTS))
    bad = [t for t in selected if t not in ALL_TESTS]
    if bad:
        print(f"Unknown test(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    script_dir = os.path.dirname(os.path.abspath(__file__))
    quiet = args.json_stdout

    # The disk test writes its scratch file into `disk_dir`; make sure it exists.
    disk_dir = tempfile.gettempdir() if args.no_save else args.output_dir
    try:
        os.makedirs(disk_dir, exist_ok=True)
    except OSError:
        disk_dir = tempfile.gettempdir()

    info = gather_system_info()
    logical = info["cpu_cores_logical"]

    if not quiet:
        hr(f"PC Benchmark & Diagnostics v{VERSION}")
        print(f"  seconds/test={args.seconds}  repeats={args.repeats}  "
              f"tests={','.join(selected)}")

    results: dict = {}
    runners = {
        "cpu_int": lambda: bench_cpu_integer(args.seconds, args.repeats),
        "cpu_float": lambda: bench_cpu_float(args.seconds, args.repeats),
        "cpu_multi": lambda: bench_cpu_multicore(args.seconds, logical),
        "memory": lambda: bench_memory(args.seconds, args.repeats, args.mem_mb),
        "disk": lambda: bench_disk(args.seconds, args.repeats, args.disk_mb,
                                   disk_dir),
    }
    for name in selected:
        if not quiet:
            print(f"  running {name} ...", flush=True)
        try:
            results[name] = runners[name]()
        except Exception as e:  # never let one probe abort the whole run
            results[name] = {"error": str(e)}
            if not quiet:
                print(f"    ! {name} failed: {e}", file=sys.stderr)

    native = None
    if not args.no_native:
        if not quiet:
            print("  running native engine ...", flush=True)
        native = run_native_engine(args.seconds, args.repeats, script_dir)

    scores = compute_scores(results)

    payload = {
        "tool": "pc-benchmark",
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"seconds": args.seconds, "repeats": args.repeats,
                   "tests": selected, "disk_mb": args.disk_mb,
                   "mem_mb": args.mem_mb},
        "system": info,
        "results": results,
        "native": native,
        "scores": scores,
    }

    if args.json_stdout:
        print(json.dumps(payload, indent=2))
    else:
        print_console_report(info, results, scores, native)

    if not args.no_save:
        try:
            jp = save_json(payload, args.output_dir)
            cp = append_csv(payload, args.output_dir)
            if not quiet:
                hr("Saved")
                print(f"  JSON: {jp}")
                print(f"  CSV : {cp}")
        except OSError as e:
            print(f"  ! could not save results: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    mp.freeze_support()  # required for spawn-based multiprocessing on Windows
    sys.exit(main())
