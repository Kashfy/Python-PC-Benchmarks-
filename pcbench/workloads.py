"""Benchmark workloads.

Every workload follows the same contract:

* a ``*_chunk`` function performs one fixed unit of work,
* a ``bench_*`` function warms up, repeats the measurement, validates the
  computed result, and returns a dict containing at least ``unit`` and
  ``rate``.

Validation matters as much as speed here: a machine that computes the wrong
answer quickly is faulty, not fast. See :mod:`pcbench.core`.
"""

from __future__ import annotations

import hashlib
import json as _json
import math
import multiprocessing as mp
import os
import platform
import random
import shutil
import tempfile
import zlib

from .core import (ValidationError, check_close, check_exact, clock,
                   summarize, timed_loop, warmup)

MB = 1024 * 1024
KB = 1024

# --------------------------------------------------------------------------- #
# Fixed workload constants and their known-good results.
# --------------------------------------------------------------------------- #
PRIME_LO, PRIME_HI = 50_000, 51_000
PRIMES_PER_CHUNK = PRIME_HI - PRIME_LO          # integers tested per chunk
EXPECTED_PRIME_COUNT = 89                        # primes in [50000, 51000)

FLOAT_ITERS_PER_CHUNK = 50_000
EXPECTED_FLOAT_SUM = 35173.9049856305            # deterministic to ~1e-12


# --------------------------------------------------------------------------- #
# CPU: integer
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def cpu_integer_chunk() -> int:
    """Primality-test a fixed range. Returns the prime count for validation."""
    count = 0
    for n in range(PRIME_LO, PRIME_HI):
        if _is_prime(n):
            count += 1
    return count


def bench_cpu_integer(seconds: float, repeats: int) -> dict:
    check_exact("cpu_int", cpu_integer_chunk(), EXPECTED_PRIME_COUNT)
    warmup(cpu_integer_chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, chunks = timed_loop(cpu_integer_chunk, seconds)
        rates.append(chunks * PRIMES_PER_CHUNK / elapsed)
    s = summarize(rates)
    return {"unit": "primes/s", "rate": s["median"], "validated": True, **s}


# --------------------------------------------------------------------------- #
# CPU: floating point
# --------------------------------------------------------------------------- #
def cpu_float_chunk() -> float:
    x, s = 0.001, 0.0
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(FLOAT_ITERS_PER_CHUNK):
        x += 0.00001
        s += sin(x) * cos(x) + sqrt(x)
    return s


def bench_cpu_float(seconds: float, repeats: int) -> dict:
    # Platform libm implementations differ in the last bits, so compare with a
    # relative tolerance rather than exactly.
    check_close("cpu_float", cpu_float_chunk(), EXPECTED_FLOAT_SUM, 1e-9)
    warmup(cpu_float_chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, chunks = timed_loop(cpu_float_chunk, seconds)
        rates.append(chunks * FLOAT_ITERS_PER_CHUNK / elapsed)
    s = summarize(rates)
    return {"unit": "iters/s", "rate": s["median"], "validated": True, **s}


# --------------------------------------------------------------------------- #
# CPU: multi-core
#
# The worker must be a module-level function so it survives pickling under the
# "spawn" start method used on macOS and Windows.
# --------------------------------------------------------------------------- #
def _multicore_worker(duration: float) -> tuple[int, int]:
    """Run integer chunks for ``duration``. Returns (primes_tested, checksum).

    Each worker times itself; ``perf_counter`` epochs are per-process, so a
    shared deadline would be meaningless across processes.
    """
    start = clock()
    chunks = 0
    checksum = 0
    while clock() - start < duration:
        checksum = cpu_integer_chunk()
        chunks += 1
    return chunks * PRIMES_PER_CHUNK, checksum


def bench_cpu_multicore(seconds: float, workers: int | None = None) -> dict:
    workers = max(1, workers or os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    start = clock()
    with ctx.Pool(processes=workers) as pool:
        pairs = pool.map(_multicore_worker, [seconds] * workers)
    wall = clock() - start

    total = sum(p for p, _ in pairs)
    for _, checksum in pairs:
        check_exact("cpu_multi", checksum, EXPECTED_PRIME_COUNT)
    return {
        "unit": "primes/s",
        "rate": total / wall if wall else 0.0,
        "workers": workers,
        "wall_seconds": round(wall, 3),
        "per_worker_primes": [p for p, _ in pairs],
        "validated": True,
    }


# --------------------------------------------------------------------------- #
# CPU: real-world mixed workloads
#
# Synthetic loops measure one execution unit; these measure the paths real
# software actually exercises. SHA-256 in particular reaches hardware crypto
# instructions (ARMv8 crypto extensions, x86 SHA-NI), so it exposes real
# architectural differences the synthetic tests miss.
# --------------------------------------------------------------------------- #
def _corpus(size: int = 4 * MB) -> bytes:
    """Semi-repetitive, compressible data resembling real text/logs.

    Random bytes would be incompressible and make the zlib result meaningless.
    """
    rnd = random.Random(1234)  # fixed seed: identical corpus everywhere
    words = [b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot",
             b"golf", b"hotel", b"india", b"juliet", b"kilo", b"lima"]
    out = bytearray()
    while len(out) < size:
        out += b" ".join(rnd.choice(words) for _ in range(64)) + b"\n"
    return bytes(out[:size])


def bench_compression(seconds: float, repeats: int) -> dict:
    data = _corpus(2 * MB)
    expected = zlib.crc32(data)

    def chunk():
        blob = zlib.compress(data, 6)
        if zlib.crc32(zlib.decompress(blob)) != expected:
            raise ValidationError("compression: round-trip mismatch — "
                                  "possible memory or CPU fault")

    chunk()
    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n * len(data) / elapsed / MB)
    s = summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "validated": True,
            "note": "zlib level 6 compress+decompress round-trip", **s}


def bench_hashing(seconds: float, repeats: int) -> dict:
    data = _corpus(4 * MB)
    expected = hashlib.sha256(data).hexdigest()

    def chunk():
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValidationError("sha256: digest mismatch — possible memory "
                                  "or CPU fault")

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n * len(data) / elapsed / MB)
    s = summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "validated": True,
            "note": "SHA-256; uses hardware crypto where available", **s}


def bench_json(seconds: float, repeats: int) -> dict:
    rnd = random.Random(99)
    doc = [{"id": i, "name": f"item-{i}", "score": rnd.random(),
            "tags": ["a", "b", "c"], "ok": bool(i % 2)} for i in range(5000)]
    text = _json.dumps(doc)
    nbytes = len(text.encode())

    def chunk():
        parsed = _json.loads(text)
        if len(parsed) != 5000:
            raise ValidationError("json: parse produced wrong element count")

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n * nbytes / elapsed / MB)
    s = summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "validated": True,
            "note": "JSON parse throughput", **s}


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def bench_memory(seconds: float, repeats: int, buf_mb: int = 64) -> dict:
    """Sustained copy bandwidth on a buffer far larger than cache."""
    n = buf_mb * MB
    src = bytearray(b"\xa5" * n)
    dst = bytearray(n)

    def chunk():
        dst[:] = src

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        start = clock()
        moved = 0
        while clock() - start < seconds:
            chunk()
            moved += n
        rates.append(moved / (clock() - start) / MB)

    if dst[:8] != src[:8] or dst[-8:] != src[-8:]:
        raise ValidationError("memory: copied data does not match source — "
                              "possible failing RAM")
    s = summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "buffer_mb": buf_mb,
            "validated": True, **s}


# Working-set sizes for the cache sweep. The sweep starts at 128 KB because
# below that a single copy completes faster than the interpreter overhead of
# issuing it, which flattens and then inverts the curve — measuring Python
# rather than the cache. Resolving L1 requires the native engine.
_SWEEP_SIZES = [128 * KB, 512 * KB, 2 * MB, 8 * MB, 32 * MB, 128 * MB]


def bench_cache_sweep(total_seconds: float = 3.0,
                      ram_bytes: int = 0) -> dict:
    """Copy bandwidth across growing working sets, revealing the cache tiers.

    Bandwidth stays high while the working set fits in a cache level and drops
    at each boundary, so the steps in this curve locate the L2/L3/DRAM
    transitions without needing privileged access to CPU topology.
    """
    sizes = [s for s in _SWEEP_SIZES
             if not ram_bytes or s * 4 < ram_bytes]
    if not sizes:
        return {"skipped": True, "error": "insufficient RAM for sweep"}

    per_size = max(0.05, total_seconds / len(sizes))
    points = []
    for size in sizes:
        src = bytearray(b"\x5a" * size)
        dst = bytearray(size)
        # One untimed copy per size to fault in pages and prime the caches.
        dst[:] = src
        start = clock()
        moved = 0
        while clock() - start < per_size:
            dst[:] = src
            moved += size
        elapsed = clock() - start
        points.append({
            "bytes": size,
            "label": (f"{size // KB} KB" if size < MB else f"{size // MB} MB"),
            "mb_per_s": round(moved / elapsed / MB, 1),
        })

    best = max(p["mb_per_s"] for p in points)
    worst = min(p["mb_per_s"] for p in points)
    return {
        "points": points,
        "peak_mb_per_s": best,
        "dram_mb_per_s": points[-1]["mb_per_s"],
        "cache_to_dram_ratio": round(best / worst, 2) if worst else None,
    }


# --------------------------------------------------------------------------- #
# Disk
#
# Read benchmarks are worthless if the OS serves them from the page cache, so
# we try hard to bypass it and report whether we succeeded.
# --------------------------------------------------------------------------- #
_F_NOCACHE = 48  # macOS fcntl command number


def _set_nocache(fd: int) -> bool:
    """Disable page-cache buffering for this descriptor (macOS).

    This must be set *before* the file is written. ``F_NOCACHE`` stops new I/O
    from populating the cache but does not evict pages that are already there,
    so enabling it after a write would leave the whole file cached and make the
    subsequent read benchmark measure RAM instead of the device.
    """
    if platform.system() != "Darwin":
        return False
    try:
        import fcntl
        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", _F_NOCACHE), 1)
        return True
    except (ImportError, OSError):
        return False


def _drop_cache(fd: int) -> bool:
    """Evict this file's pages from the page cache (Linux)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def bench_disk(seconds: float, repeats: int, file_mb: int,
               out_dir: str) -> dict:
    """Sequential write/read plus 4 KiB random-read IOPS.

    Sequential throughput characterizes bulk transfers; random IOPS is what
    actually determines how responsive a machine feels, since real workloads
    are dominated by small scattered reads.
    """
    block = 4 * MB
    n_blocks = max(1, (file_mb * MB) // block)
    total = n_blocks * block

    try:
        free = shutil.disk_usage(out_dir).free
    except OSError as e:
        return {"skipped": True, "error": f"cannot stat {out_dir}: {e}"}
    if total * 1.2 > free:
        return {"skipped": True,
                "error": f"needs ~{total * 1.2 / MB:.0f} MB free, "
                         f"has {free / MB:.0f} MB"}

    buf = b"\xc3" * block
    writes, reads, iops = [], [], []
    bypassed = False

    for _ in range(repeats):
        fd, path = tempfile.mkstemp(prefix="pcbench_", suffix=".bin",
                                    dir=out_dir)
        try:
            # Must precede the write so the data never enters the cache.
            nocache = _set_nocache(fd)

            # ---- sequential write (timed through fsync) ----
            start = clock()
            for _ in range(n_blocks):
                os.write(fd, buf)
            os.fsync(fd)
            writes.append(total / (clock() - start) / MB)

            # ---- sequential read ----
            # On Linux nothing can be excluded up front, so evict instead.
            bypassed = nocache or _drop_cache(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            start = clock()
            got = 0
            while True:
                data = os.read(fd, block)
                if not data:
                    break
                got += len(data)
            reads.append(got / (clock() - start) / MB)

            # ---- 4 KiB random reads ----
            iops.append(_random_read_iops(fd, total, min(seconds, 2.0)))
        except OSError as e:
            return {"skipped": True, "error": f"disk I/O failed: {e}"}
        finally:
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass

    w, r, io = summarize(writes), summarize(reads), summarize(iops)
    return {
        "unit": "MB/s",
        "file_mb": file_mb,
        "write_rate": w["median"],
        "read_rate": r["median"],
        "random_read_iops": io["median"],
        "cache_bypassed": bypassed,
        "write": w,
        "read": r,
        "random": io,
        "note": ("reads bypassed the page cache"
                 if bypassed else
                 "page cache NOT bypassed on this platform; read numbers are "
                 "an optimistic upper bound"),
    }


def _random_read_iops(fd: int, size: int, budget: float) -> float:
    """4 KiB reads at random offsets; returns operations per second."""
    page = 4 * KB
    max_off = max(0, size - page)
    rnd = random.Random(4242)
    start = clock()
    ops = 0
    while clock() - start < budget:
        # Batch to keep clock() overhead off the measured path.
        for _ in range(64):
            os.pread(fd, page, rnd.randrange(0, max_off + 1, page)
                     if max_off else 0)
        ops += 64
    return ops / (clock() - start)
