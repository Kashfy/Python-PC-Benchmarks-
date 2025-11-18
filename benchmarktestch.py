#!/usr/bin/env python3
"""
Simple cross-platform benchmark script.

Run this on each machine like:
    python3 benchmark.py

Optionally:
    python3 benchmark.py --seconds 5
"""

import time
import math
import platform
import tempfile
import os
import argparse
from statistics import mean, stdev

# ---------- Helpers ----------

def run_benchmark_chunked(chunk_func, seconds=3.0):
    """
    Run chunk_func repeatedly for ~`seconds` and return:
      elapsed_time, num_chunks, chunks_per_second
    """
    start = time.perf_counter()
    count = 0
    while True:
        chunk_func()
        count += 1
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    cps = count / elapsed if elapsed > 0 else float('inf')
    return elapsed, count, cps


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------- Workload chunks ----------

def cpu_integer_chunk():
    """
    Integer-heavy workload: check primality for a range of numbers.
    Uses a simple (intentionally not-optimized) primality test.
    """
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0 and n != 2:
            return False
        r = int(math.isqrt(n))
        for i in range(3, r + 1, 2):
            if n % i == 0:
                return False
        return True

    # Check primes in a small range; cost is consistent across machines.
    for n in range(50_000, 51_000):
        _ = is_prime(n)


def cpu_float_chunk():
    """
    Floating-point workload using sin/cos/sqrt in a tight loop.
    """
    x = 0.001
    s = 0.0
    for i in range(50_000):
        x += 0.00001
        s += math.sin(x) * math.cos(x) + math.sqrt(x)
    # Prevent optimization away
    if s == -1.23456789:
        print("magic", s)


def memory_chunk():
    """
    Memory/alloc workload: allocate a list of floats and sum it.
    Size is moderate to avoid crashes but still stress memory and cache.
    """
    size = 500_000  # ~4 MB for float-like values in CPython (approx)
    data = [float(i) for i in range(size)]
    total = sum(data)
    if total == -1.23456789:
        print("magic", total)


def disk_chunk():
    """
    Disk I/O workload: write and read a temporary file (~5 MB).
    """
    data = b"X" * (5 * 1024 * 1024)  # 5 MB
    fd, path = tempfile.mkstemp(prefix="py_bench_", suffix=".bin")
    os.close(fd)
    try:
        # Write
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Read
        with open(path, "rb") as f:
            _ = f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------- Main ----------

def gather_system_info():
    info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
    }
    return info


def run_all_benchmarks(seconds_per_test=3.0, repeats=3):
    """
    Run each benchmark multiple times and summarize.
    """
    tests = [
        ("CPU Integer (primes)", cpu_integer_chunk),
        ("CPU Float (math ops)", cpu_float_chunk),
        ("Memory (alloc & sum)", memory_chunk),
        ("Disk I/O (5 MB R/W)", disk_chunk),
    ]

    results = []

    for name, func in tests:
        print_header(f"Running: {name}")
        cps_values = []
        for r in range(1, repeats + 1):
            elapsed, count, cps = run_benchmark_chunked(func, seconds=seconds_per_test)
            cps_values.append(cps)
            print(f"  Run {r}: {elapsed:.2f} s, {count} chunks -> {cps:.2f} chunks/s")
        avg = mean(cps_values)
        sd = stdev(cps_values) if len(cps_values) > 1 else 0.0
        results.append((name, avg, sd))

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple cross-platform benchmark.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Target duration per test (per repeat) in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per test (default: 3)",
    )
    args = parser.parse_args()

    # System info
    print_header("System Information")
    info = gather_system_info()
    for k, v in info.items():
        print(f"{k:16}: {v}")

    # Benchmarks
    results = run_all_benchmarks(
        seconds_per_test=args.seconds,
        repeats=args.repeats,
    )

    # Summary
    print_header("Summary (higher is better)")
    for name, avg, sd in results:
        print(f"{name:24}: {avg:10.2f} chunks/s  (std dev: {sd:.2f})")

    # Optional JSON-like line for easy copy-paste into a notes file
    print_header("Machine Score Snapshot")
    snapshot = {
        "system": info["system"],
        "machine": info["machine"],
        "processor": info["processor"],
        "python": info["python_version"],
        "results": {name: avg for name, avg, _ in results},
    }
    print(snapshot)


if __name__ == "__main__":
    main()

