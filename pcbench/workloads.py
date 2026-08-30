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
import mmap
import multiprocessing as mp
import os
import platform
import random
import shutil
import tempfile
import zlib

from . import limits
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
def bench_memory(seconds: float, repeats: int, buf_mb: int = 64,
                 ram_bytes: int = 0) -> dict:
    """Sustained copy bandwidth on a buffer far larger than cache.

    The buffer size is clamped against physical RAM. Over-allocating would push
    the machine into swap, which stalls the whole system and writes heavily to
    the SSD — a benchmark must never do that to the machine it is measuring.
    """
    if not ram_bytes:
        from .system import total_ram_bytes
        ram_bytes = total_ram_bytes()
    buf_mb, notice = limits.safe_mem_mb(buf_mb, ram_bytes)

    n = buf_mb * MB
    # Fill without materializing a temporary bytes object of the same size:
    # `bytearray(b"\xa5" * n)` would briefly hold 2n bytes for one buffer.
    src = bytearray(n)
    for i in range(0, n, MB):
        src[i:i + MB] = b"\xa5" * min(MB, n - i)
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
    out = {"unit": "MB/s", "rate": s["median"], "buffer_mb": buf_mb,
           "validated": True, **s}
    if notice:
        out["safety_notice"] = notice
    return out


def _memcopy_worker(args) -> int:
    """Copy a private buffer pair for `seconds`; returns bytes moved.

    Must be module-level to survive pickling under the "spawn" start method.
    """
    buf_bytes, seconds = args
    src = bytearray(b"\xa5") * buf_bytes
    dst = bytearray(buf_bytes)
    moved = 0
    start = clock()
    while clock() - start < seconds:
        dst[:] = src
        moved += buf_bytes
    return moved


def bench_memory_scaling(seconds: float = 0.4, buf_mb: int = 64,
                         ram_bytes: int = 0) -> dict:
    """Copy bandwidth as concurrency rises, revealing the memory ceiling.

    A single core rarely saturates a memory controller, so single-threaded
    bandwidth can understate the machine. Sweeping concurrency shows where the
    controller — rather than the core — becomes the limit.

    **Processes, not threads.** CPython does not release the GIL during
    ``bytearray`` slice assignment, so a threaded version serializes: measured
    here it reported a flat 40 GB/s at every thread count, which is the GIL
    speaking rather than the memory subsystem. Separate processes contend for
    real bandwidth.
    """
    cores = os.cpu_count() or 1
    counts = sorted({1, 2, max(1, cores // 2), cores})

    # The buffer must exceed last-level cache or this measures cache, not
    # memory. Footprint is 2 buffers per process at peak concurrency, so the
    # per-process size is derived from the total safe budget divided by that.
    budget_mb, _ = limits.safe_mem_mb(1 << 20, ram_bytes or 0)   # 1/8 of RAM
    per_proc_mb = max(1, budget_mb // (2 * max(1, cores)))
    buf_mb = max(32, min(buf_mb, per_proc_mb))
    n = buf_mb * MB

    ctx = mp.get_context("spawn")
    points = []
    for procs in counts:
        with ctx.Pool(processes=procs) as pool:
            moved = sum(pool.map(_memcopy_worker, [(n, seconds)] * procs))
        # Each process ran for `seconds` concurrently, so divide by that
        # rather than wall time, which would include pool spawn cost.
        points.append({"processes": procs,
                       "mb_per_s": round(moved / seconds / MB, 1)})

    peak = max(points, key=lambda p: p["mb_per_s"])
    single = points[0]["mb_per_s"] or 1
    return {
        "unit": "MB/s",
        "rate": peak["mb_per_s"],
        "points": points,
        "peak_processes": peak["processes"],
        "single_mb_per_s": points[0]["mb_per_s"],
        "scaling": round(peak["mb_per_s"] / single, 2),
        "buffer_mb": buf_mb,
    }


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
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError as e:
        return {"skipped": True, "error": f"cannot stat {out_dir}: {e}"}

    # Clamp for free-space headroom and cumulative flash wear before sizing.
    file_mb, notice = limits.safe_disk_mb(file_mb, free, repeats)

    block = 4 * MB
    n_blocks = max(1, (file_mb * MB) // block)
    total = n_blocks * block

    if total * limits.DISK_FREE_HEADROOM > free:
        return {"skipped": True,
                "error": f"needs ~{total * limits.DISK_FREE_HEADROOM / MB:.0f} "
                         f"MB free, has {free / MB:.0f} MB"}

    _clean_stale_files(out_dir)
    # Incompressible, and a different block each time. A file of one repeated
    # byte is free to store on anything that compresses -- btrfs with
    # ``compress=``, ZFS, NTFS compression, and the inline compression in many
    # SSD controllers -- so the write never reaches the medium at the rate
    # being reported. It also breaks the read path: btrfs serves O_DIRECT
    # reads of a compressed extent through the page cache, which is what made
    # the random-read figures on this machine impossible. Rotating a
    # memoryview over one random pool costs nothing per write and additionally
    # defeats block-level deduplication.
    pool = os.urandom(block + 4 * KB)
    view = memoryview(pool)
    offsets = [i * 512 for i in range(8)]
    writes, reads, iops = [], [], []
    seq_bypassed = False
    direct_method = "buffered"
    latency: dict | None = None
    qd_sweep: dict | None = None

    for _ in range(repeats):
        fd, path = tempfile.mkstemp(prefix="pcbench_", suffix=".bin",
                                    dir=out_dir)
        try:
            # Must precede the write so the data never enters the cache.
            nocache = _set_nocache(fd)

            # ---- sequential write (timed through fsync) ----
            start = clock()
            for i in range(n_blocks):
                off = offsets[i % len(offsets)]
                os.write(fd, view[off:off + block])
            os.fsync(fd)
            writes.append(total / (clock() - start) / MB)

            # ---- sequential read ----
            # On Linux nothing can be excluded up front, so evict instead.
            seq_bypassed = nocache or _drop_cache(fd)
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
            # The sequential read above pulled the whole file back into the
            # page cache, and the file is far smaller than RAM, so these must
            # go through a descriptor that bypasses the cache or they measure
            # memory. See :class:`DirectReader`.
            with DirectReader(path, fd) as handle:
                direct_method = handle.method
                iops.append(_random_read_iops(handle, total,
                                              min(seconds, 2.0)))
                if latency is None:
                    latency = _random_read_latency(handle, total)
            if qd_sweep is None:
                qd_sweep = _queue_depth_sweep(fd, total, min(seconds, 1.0),
                                              path)
        except OSError as e:
            return {"skipped": True, "error": f"disk I/O failed: {e}"}
        finally:
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass

    w, r, io = summarize(writes), summarize(reads), summarize(iops)
    ramfs = memory_filesystem(out_dir)
    bypassed, cache_note = direct_read_note(direct_method, latency, ramfs)
    result = {
        "unit": "MB/s",
        "file_mb": file_mb,
        # Disclosed so flash wear over many runs is never invisible.
        "total_written_mb": limits.total_write_mb(file_mb, repeats),
        "write_rate": w["median"],
        "read_rate": r["median"],
        "random_read_iops": io["median"],
        "random_read_latency": latency,
        "queue_depth_sweep": qd_sweep,
        "peak_iops": (qd_sweep or {}).get("peak_iops", 0.0),
        # Reported separately because they are bypassed by different means and
        # can disagree: the sequential read is evicted before it runs, the
        # random reads are issued unbuffered.
        "cache_bypassed": bypassed,
        "sequential_cache_bypassed": seq_bypassed and not ramfs,
        "direct_method": direct_method,
        "memory_filesystem": ramfs,
        "write": w,
        "read": r,
        "random": io,
        "note": cache_note,
    }
    if notice:
        result["safety_notice"] = notice
    return result


def _clean_stale_files(out_dir: str) -> int:
    """Remove scratch files orphaned by a previous interrupted run.

    Only files matching this tool's own ``pcbench_*.bin`` prefix are touched,
    and only ones older than a minute, so a concurrent run is never disturbed.
    """
    import glob
    removed = 0
    cutoff = clock_wall() - 60
    for path in glob.glob(os.path.join(out_dir, "pcbench_*.bin")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def clock_wall() -> float:
    """Wall-clock seconds (file mtimes are wall-clock, not perf_counter)."""
    import time as _time
    return _time.time()


# --------------------------------------------------------------------------- #
# Portable positional I/O
# --------------------------------------------------------------------------- #
# `os.pread` and `os.pwrite` are POSIX-only; on Windows they do not exist and
# the whole storage section died with "module 'os' has no attribute 'pread'".
#
# The obvious fallback -- lseek then read -- is *not* equivalent, because it is
# two operations against a shared file pointer. Every queue-depth test here
# runs several threads against one descriptor, so another thread can move the
# pointer between the seek and the read, and the result would be silently wrong
# rather than slow. Windows therefore gets one descriptor per thread, which is
# what makes seek+read safe again, and is closer to what fio does anyway.
HAS_PREAD = hasattr(os, "pread")


def pread(fd: int, count: int, offset: int) -> bytes:
    """Read at an absolute offset without disturbing other readers.

    On a platform without `os.pread`, the caller MUST own ``fd`` exclusively.
    Use :func:`open_reader` to obtain a private descriptor for each thread.
    """
    if HAS_PREAD:
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, count)


def pwrite(fd: int, data: bytes, offset: int) -> int:
    """Write at an absolute offset. Same exclusivity rule as :func:`pread`."""
    if HAS_PREAD:
        return os.pwrite(fd, data, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.write(fd, data)


def open_reader(path: str, shared_fd: int) -> tuple[int, bool]:
    """Get a descriptor safe for one thread to read through.

    Returns ``(fd, owned)``. Where `os.pread` exists the shared descriptor is
    returned unchanged and ``owned`` is False, because positional reads on it
    are already independent. Otherwise a private descriptor is opened and the
    caller must close it.
    """
    if HAS_PREAD or not path:
        return shared_fd, False
    try:
        return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)), True
    except OSError:
        return shared_fd, False



# --------------------------------------------------------------------------- #
# Unbuffered reads
# --------------------------------------------------------------------------- #
# Dropping the cache once before the sequential read is not enough for the
# random-read phase that follows: the sequential read pulls the whole file
# straight back in, and a test file small enough to be safe for flash wear is
# far smaller than RAM, so every random read after it is served from memory.
# The symptom is unmistakable once you know it -- sub-microsecond p50 latency
# and queue depth 1 beating queue depth 32, which no real device does -- and
# the numbers it produces are memory bandwidth wearing a storage label.
#
# The only reliable cure is to stop the reads entering the cache at all, which
# every platform spells differently:
#
# * **Linux** -- ``O_DIRECT`` at open time. Requires the buffer, the offset and
#   the length to be aligned to the logical block size; 4 KiB satisfies every
#   device this tool targets. Some filesystems (tmpfs, a few network mounts)
#   reject the flag outright, and a few accept it and quietly fall back to
#   buffered I/O, which is why the result is plausibility-checked afterwards.
# * **macOS** -- ``F_NOCACHE`` on the descriptor. No alignment requirement.
# * **Windows** -- ``FILE_FLAG_NO_BUFFERING``, which ``os.open`` does not
#   expose, so the handle comes from ``CreateFileW`` through ctypes. Same
#   alignment rules as Linux.
#
# Where none of them works the reader says so instead of pretending, and the
# caller discloses it in the result.

#: Alignment unit for unbuffered I/O. Every device this tool targets reports a
#: logical block size of 512 or 4096 bytes, so 4 KiB is aligned for all of them
#: and is also the block size the random-read test uses.
DIRECT_BLOCK = 4 * KB

#: Latency below which a "device" read cannot be genuine. PCIe round trip plus
#: NAND access is tens of microseconds on a good NVMe drive and about 6 us on
#: the fastest Optane part ever sold, so anything under this came from RAM.
IMPLAUSIBLE_DEVICE_US = 3.0

# CreateFileW constants (winnt.h), used only on Windows.
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ_WRITE = 0x00000003
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000


class DirectReader:
    """One thread's reader, bypassing the page cache where the OS allows it.

    ``method`` records how it got there -- ``o_direct``, ``f_nocache``,
    ``no_buffering`` or ``buffered`` when the platform or the filesystem
    refused. A reader is owned by exactly one thread: the Windows and
    fallback paths carry a file position, so sharing one would race.
    """

    def __init__(self, path: str, shared_fd: int,
                 block: int = DIRECT_BLOCK, direct: bool = True) -> None:
        self.block = block
        self.method = "buffered"
        self._fd = shared_fd
        self._owned = False
        self._handle = None          # Windows HANDLE, when NO_BUFFERING works
        self._buf = None             # aligned destination for direct reads
        self._win = None             # (kernel32, address) once bound

        if path and direct:
            if os.name == "nt":
                self._open_no_buffering(path)
            elif hasattr(os, "O_DIRECT") and hasattr(os, "preadv"):
                self._open_o_direct(path)
            elif platform.system() == "Darwin":
                self._open_nocache(path)
        if self.method == "buffered" and not HAS_PREAD:
            # No positional read, so seek-then-read is the only option and a
            # shared descriptor would race on the file pointer between
            # threads — silently reading the wrong offset rather than failing.
            self._fd, self._owned = open_reader(path, shared_fd)

    # -- platform openers ---------------------------------------------------
    def _open_o_direct(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError:
            return                                  # filesystem refused it
        buf = mmap.mmap(-1, self.block)             # page-aligned by mmap
        try:
            os.preadv(fd, [buf], 0)                 # probe before committing
        except OSError:
            os.close(fd)
            buf.close()
            return
        self._fd, self._owned, self._buf = fd, True, buf
        self.method = "o_direct"

    def _open_nocache(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        if not _set_nocache(fd):
            os.close(fd)
            return
        self._fd, self._owned = fd, True
        self.method = "f_nocache"

    def _open_no_buffering(self, path: str) -> None:
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.restype = wintypes.HANDLE
            kernel32.CreateFileW.argtypes = [
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                wintypes.HANDLE]
            kernel32.ReadFile.restype = wintypes.BOOL
            kernel32.ReadFile.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            kernel32.SetFilePointerEx.restype = wintypes.BOOL
            kernel32.SetFilePointerEx.argtypes = [
                wintypes.HANDLE, ctypes.c_longlong,
                ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            handle = kernel32.CreateFileW(
                path, _GENERIC_READ, _FILE_SHARE_READ_WRITE, None,
                _OPEN_EXISTING, _FILE_FLAG_NO_BUFFERING, None)
            # CreateFileW returns INVALID_HANDLE_VALUE, which through a
            # HANDLE restype arrives as the unsigned form of -1 rather than
            # as -1 itself.
            if not handle or handle == ctypes.c_void_p(-1).value:
                return
            # A ctypes buffer rather than an mmap: taking the address of an
            # mmap through `from_buffer` exports a pointer into it, and
            # `mmap.close()` then raises BufferError. Over-allocate and round
            # up instead, since NO_BUFFERING requires sector alignment.
            align = DIRECT_BLOCK
            raw = ctypes.create_string_buffer(self.block + align)
            address = (ctypes.addressof(raw) + align - 1) & ~(align - 1)
            self._handle, self._buf = handle, raw
            self._win = (kernel32, address)
            if self._read_windows(0) <= 0:          # probe before committing
                self.close()
                self.method = "buffered"
                return
            self.method = "no_buffering"
        except Exception:
            # ctypes binding failures must never take the disk test with them.
            self.close()
            self.method = "buffered"

    # -- reading ------------------------------------------------------------
    def _read_windows(self, offset: int) -> int:
        import ctypes
        from ctypes import wintypes
        kernel32, address = self._win
        moved = ctypes.c_longlong(0)
        if not kernel32.SetFilePointerEx(self._handle,
                                         ctypes.c_longlong(offset),
                                         ctypes.byref(moved), 0):
            return 0
        got = wintypes.DWORD(0)
        if not kernel32.ReadFile(self._handle, ctypes.c_void_p(address),
                                 wintypes.DWORD(self.block),
                                 ctypes.byref(got), None):
            return 0
        return got.value

    def read(self, offset: int) -> int:
        """Read one aligned block at ``offset``; returns the bytes read."""
        if self._handle is not None:
            return self._read_windows(offset)
        if self._buf is not None:
            return os.preadv(self._fd, [self._buf], offset)
        return len(pread(self._fd, self.block, offset))

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._win[0].CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None
            self._win = None
        if self._buf is not None:
            # An mmap on the O_DIRECT path, a ctypes buffer on the Windows
            # one; only the former needs closing.
            try:
                self._buf.close()
            except (AttributeError, BufferError):
                pass
            self._buf = None
        if self._owned:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._owned = False

    def __enter__(self) -> "DirectReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


#: Filesystems that live in RAM. A "disk" test on one of these measures memory
#: with extra steps, and it is easy to hit by accident: /tmp is tmpfs on most
#: current Linux distributions, so any run that fell back to the system temp
#: directory reported memory bandwidth as storage throughput.
MEMORY_FILESYSTEMS = {"tmpfs", "ramfs", "devtmpfs", "rootfs"}


def memory_filesystem(path: str) -> str | None:
    """Name the RAM-backed filesystem ``path`` sits on, or None.

    Linux only, because it is the only platform where the default temp
    directory is commonly a RAM filesystem — macOS puts /tmp on the boot
    volume and Windows puts %TEMP% in the user profile.
    """
    try:
        target = os.path.realpath(path)
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as f:
            entries = [line.split()[:3] for line in f if len(line.split()) >= 3]
    except (OSError, IndexError):
        return None
    best, best_type = "", None
    for _device, point, fstype in entries:
        point = point.replace("\\040", " ")
        if (target == point or target.startswith(point.rstrip("/") + "/")) \
                and len(point) >= len(best):
            best, best_type = point, fstype
    return best_type if best_type in MEMORY_FILESYSTEMS else None


def direct_read_note(method: str, latency: dict | None,
                     ramfs: str | None = None) -> tuple[bool, str]:
    """Judge whether the random-read figures really came from the device.

    Returns ``(bypassed, note)``. Three ways they might not have: the platform
    had no bypass mechanism; the filesystem accepted one and served from cache
    anyway (btrfs with compression and most network filesystems do exactly
    this, which is otherwise indistinguishable from a very fast drive, so the
    latency is checked against physics as well as the flag); or there is no
    device involved at all because the directory is a RAM filesystem.
    """
    if ramfs:
        return False, (f"the test directory is on {ramfs}, which is RAM — "
                       f"these are memory figures, not storage ones. Point "
                       f"the run at a directory on real storage "
                       f"(--output-dir, or --disk-path for the I/O suite)")
    if method == "buffered":
        return False, ("the page cache could not be bypassed on this "
                       "platform, so the random-read figures include cache "
                       "hits and are an optimistic upper bound")
    p50 = (latency or {}).get("p50_us")
    if p50 is not None and p50 < IMPLAUSIBLE_DEVICE_US:
        return False, (f"reads were issued with the cache bypassed "
                       f"({method}), but a {p50:.2f} us median is faster than "
                       f"any storage device, so the filesystem served them "
                       f"from memory anyway — treat the random-read figures "
                       f"as an upper bound, not as device performance")
    return True, f"random reads bypassed the page cache ({method})"


def _random_read_iops(reader: "DirectReader", size: int,
                      budget: float) -> float:
    """4 KiB reads at random offsets, one at a time (queue depth 1)."""
    page = reader.block
    max_off = max(0, size - page)
    rnd = random.Random(4242)
    start = clock()
    ops = 0
    while clock() - start < budget:
        # Batch to keep clock() overhead off the measured path.
        for _ in range(64):
            reader.read(rnd.randrange(0, max_off + 1, page) if max_off else 0)
        ops += 64
    return ops / (clock() - start)


def _random_read_latency(reader: "DirectReader", size: int,
                         samples: int = 3000) -> dict:
    """Per-operation latency for 4 KiB random reads, in microseconds.

    Tail latency is what users feel: a drive with a good median but a poor p99
    produces visible stalls. Throughput alone cannot show this.
    """
    page = reader.block
    max_off = max(0, size - page)
    rnd = random.Random(99)
    times = []
    for _ in range(samples):
        offset = rnd.randrange(0, max_off + 1, page) if max_off else 0
        t0 = clock()
        reader.read(offset)
        times.append((clock() - t0) * 1e6)
    times.sort()
    n = len(times)
    return {
        "p50_us": round(times[n // 2], 2),
        "p99_us": round(times[min(n - 1, int(n * 0.99))], 2),
        "max_us": round(times[-1], 2),
    }


def _random_read_iops_qd(fd: int, size: int, budget: float,
                         queue_depth: int, path: str = "") -> float:
    """4 KiB random reads with ``queue_depth`` requests outstanding.

    Queue depth is the difference between a drive looking mediocre and looking
    fast. At depth 1 each read waits for the previous one to complete, so the
    result is bounded by latency rather than the device: an NVMe SSD that
    sustains hundreds of thousands of IOPS measures only tens of thousands.
    Real workloads keep many requests in flight, so the depth sweep shows the
    drive's actual ceiling.

    Threads are used rather than async I/O because the read syscall releases
    the GIL for its duration, so they genuinely overlap.

    ``path`` gives each thread its own :class:`DirectReader`, which is what
    keeps the reads off the page cache and, on platforms without `os.pread`,
    what stops the threads racing on a shared file pointer.
    """
    import threading

    page = DIRECT_BLOCK
    max_off = max(0, size - page)
    stop = threading.Event()
    counts = [0] * queue_depth

    def reader(slot: int) -> None:
        rnd = random.Random(1000 + slot)
        local = 0
        handle = DirectReader(path, fd, page)
        try:
            while not stop.is_set():
                for _ in range(16):
                    handle.read(
                        rnd.randrange(0, max_off + 1, page) if max_off else 0)
                local += 16
        except OSError:
            pass
        finally:
            handle.close()
        counts[slot] = local

    threads = [threading.Thread(target=reader, args=(i,), daemon=True)
               for i in range(queue_depth)]
    start = clock()
    for t in threads:
        t.start()
    # Wait on the event rather than spinning: a busy-wait loop here would hold
    # the GIL and starve the very reader threads being measured, collapsing
    # the result to a fraction of the single-threaded figure.
    stop.wait(budget)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
    elapsed = clock() - start
    return sum(counts) / elapsed if elapsed > 0 else 0.0


# Queue depths swept to find where a drive stops scaling.
QUEUE_DEPTHS = (1, 4, 16, 32)


def _queue_depth_sweep(fd: int, size: int, budget: float,
                       path: str = "") -> dict:
    """Random-read IOPS across increasing queue depths."""
    points = []
    for qd in QUEUE_DEPTHS:
        if qd == 1:
            with DirectReader(path, fd) as handle:
                iops = _random_read_iops(handle, size, budget)
        else:
            iops = _random_read_iops_qd(fd, size, budget, qd, path)
        points.append({"queue_depth": qd, "iops": round(iops, 1),
                       "mb_per_s": round(iops * 4 / 1024, 1)})
    best = max(points, key=lambda p: p["iops"])
    result = {
        "points": points,
        "peak_iops": best["iops"],
        "peak_queue_depth": best["queue_depth"],
        "qd1_iops": points[0]["iops"],
        "scaling": (round(best["iops"] / points[0]["iops"], 1)
                    if points[0]["iops"] else None),
    }
    if best["queue_depth"] == 1:
        # Every real device goes faster with requests queued. When the peak
        # lands at depth 1 the reads were not reaching the device at all, or
        # the thread overhead of deeper queues exceeded the device latency --
        # which is itself a sign the reads were served from memory.
        result["note"] = ("queue depth 1 was the fastest point, which no "
                          "device does — the deeper queues were limited by "
                          "something other than the drive")
    return result
