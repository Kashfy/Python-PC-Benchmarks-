"""Configurable storage I/O, in the shape ``fio`` job files describe.

The built-in disk test measures one fixed pattern, which is the right default
and the wrong tool for evaluating storage. Real systems have a specific access
pattern, and a device that is excellent at one is routinely poor at another:

* A **database** issues 8-16 KiB random reads at high queue depth. Sequential
  bandwidth predicts its performance hardly at all.
* A **log or backup target** writes large sequential blocks and cares about
  nothing else.
* A **VM host** issues a 70/30 read/write mix across many concurrent guests.
* **SMR drives and cheap SSDs** post excellent numbers until their cache fills,
  then collapse — which only a sustained mixed-write job reveals.

So a job here is described the way ``fio`` describes one: block size, pattern,
read/write mix, queue depth, duration, and whether the OS cache is bypassed.
The defaults are a small suite covering the four profiles above, and any of
them can be overridden or replaced from the command line or a config file.

**On the queue depth.** Real high-queue-depth I/O uses asynchronous submission
(``io_uring``, ``libaio``, overlapped I/O). Python has none of those portably,
so concurrency here is a thread pool issuing blocking ``pread``/``pwrite``
calls. That reaches high queue depths on real devices — the kernel sees many
outstanding requests either way — but carries more per-request CPU overhead
than ``fio`` does, so figures at very high depths on very fast NVMe devices
will read low. The limitation is stated in the results rather than hidden.
"""

from __future__ import annotations

import os
import random
import shutil
import threading
import time

from . import limits
from . import workloads as wl
from .core import clock

MB = 1024 * 1024
KB = 1024


class JobSpec:
    """One I/O job, in the vocabulary ``fio`` uses."""

    def __init__(self, name: str, block_size: int = 4 * KB,
                 pattern: str = "randread", read_pct: int = 100,
                 queue_depth: int = 1, seconds: float = 3.0,
                 direct: bool = True, file_mb: int = 256):
        if pattern not in ("read", "write", "randread", "randwrite", "randrw"):
            raise ValueError(
                f"unknown pattern {pattern!r}; valid: read, write, randread, "
                f"randwrite, randrw")
        if not 0 <= read_pct <= 100:
            raise ValueError("read_pct must be between 0 and 100")
        self.name = name
        self.block_size = max(512, int(block_size))
        self.pattern = pattern
        self.read_pct = int(read_pct)
        self.queue_depth = max(1, int(queue_depth))
        self.seconds = max(0.2, float(seconds))
        self.direct = bool(direct)
        self.file_mb = max(4, int(file_mb))

    @property
    def is_random(self) -> bool:
        return self.pattern.startswith("rand")

    @property
    def writes(self) -> bool:
        return self.pattern in ("write", "randwrite") or (
            self.pattern == "randrw" and self.read_pct < 100)

    def describe(self) -> str:
        mix = (f", {self.read_pct}/{100 - self.read_pct} r/w"
               if self.pattern == "randrw" else "")
        return (f"{self.pattern} bs={_bs_label(self.block_size)} "
                f"qd={self.queue_depth}{mix}"
                f"{'' if self.direct else ', cached'}")

    def to_dict(self) -> dict:
        return {"name": self.name, "block_size": self.block_size,
                "pattern": self.pattern, "read_pct": self.read_pct,
                "queue_depth": self.queue_depth, "seconds": self.seconds,
                "direct": self.direct, "file_mb": self.file_mb}


def _bs_label(size: int) -> str:
    if size >= MB:
        return f"{size // MB}M"
    if size >= KB:
        return f"{size // KB}K"
    return str(size)


#: The default suite. Four jobs matching the four profiles in the module
#: docstring, deliberately short so the whole suite fits in a normal run.
def default_suite(seconds: float = 2.0, file_mb: int = 256) -> list[JobSpec]:
    return [
        JobSpec("database", block_size=8 * KB, pattern="randread",
                queue_depth=16, seconds=seconds, file_mb=file_mb),
        JobSpec("sequential", block_size=1 * MB, pattern="read",
                queue_depth=1, seconds=seconds, file_mb=file_mb),
        JobSpec("log_write", block_size=64 * KB, pattern="write",
                queue_depth=4, seconds=seconds, file_mb=file_mb),
        JobSpec("vm_mixed", block_size=16 * KB, pattern="randrw",
                read_pct=70, queue_depth=8, seconds=seconds, file_mb=file_mb),
    ]


def parse_job(text: str, seconds: float = 2.0,
              file_mb: int = 256) -> JobSpec:
    """Parse ``name:bs=4k,pattern=randread,qd=32,rw=70`` into a JobSpec.

    A compact one-line syntax rather than fio's INI files: the point is to make
    an ad-hoc job expressible as a command-line argument, not to reimplement
    fio's configuration language.
    """
    name, _, rest = text.partition(":")
    name = name.strip() or "custom"
    spec: dict = {"seconds": seconds, "file_mb": file_mb}
    for item in rest.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key, value = key.strip().lower(), value.strip()
        if key in ("bs", "block_size"):
            spec["block_size"] = _parse_size(value)
        elif key in ("pattern", "rw_pattern", "mode"):
            spec["pattern"] = value
        elif key in ("qd", "queue_depth", "iodepth"):
            spec["queue_depth"] = int(value)
        elif key in ("rw", "read_pct", "rwmixread"):
            spec["read_pct"] = int(value)
        elif key in ("time", "seconds", "runtime"):
            spec["seconds"] = float(value)
        elif key in ("size", "file_mb"):
            spec["file_mb"] = int(_parse_size(value) // MB) or 4
        elif key == "direct":
            spec["direct"] = value not in ("0", "false", "no")
        else:
            raise ValueError(
                f"unknown job option {key!r} in {text!r}. Valid: bs, pattern, "
                f"qd, rw, time, size, direct")
    return JobSpec(name, **spec)


def _parse_size(text: str) -> int:
    t = text.strip().lower()
    mult = 1
    if t.endswith("k"):
        mult, t = KB, t[:-1]
    elif t.endswith("m"):
        mult, t = MB, t[:-1]
    elif t.endswith("g"):
        mult, t = 1024 * MB, t[:-1]
    elif t.endswith("b"):
        t = t[:-1]
    return int(float(t) * mult)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _open_file(path: str, size: int, direct: bool) -> tuple[int, bool]:
    """Create and open the test file, returning ``(fd, cache_bypassed)``.

    Ordering is the whole point of this function. ``F_NOCACHE`` stops *new* I/O
    from populating the page cache but never evicts what is already there, so
    it has to be set before the file is written — setting it afterwards leaves
    the entire file resident and every subsequent read measures RAM. An earlier
    version of this module did exactly that and reported 16 GB/s sequential
    reads from a device incapable of a tenth of that.

    On Linux there is nothing to set up front, so the file is written normally
    and then evicted with ``POSIX_FADV_DONTNEED``.

    This descriptor carries the *writes*. Reads go through a per-thread
    :class:`~pcbench.workloads.DirectReader` instead, because one eviction
    cannot hold for the length of a job: the file is smaller than RAM and
    becomes resident again on the first pass over it.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)

    # Must precede the write (macOS).
    bypassed = wl._set_nocache(fd) if direct else False

    current = os.fstat(fd).st_size
    if current < size:
        # Real data, not a sparse hole: reading a hole never reaches the
        # device and would report memory speed just as surely.
        chunk = os.urandom(MB)
        os.lseek(fd, 0, os.SEEK_SET)
        written = 0
        while written < size:
            n = min(len(chunk), size - written)
            os.write(fd, chunk[:n])
            written += n
    os.fsync(fd)

    if direct and not bypassed:
        # Linux: evict after the fact.
        bypassed = wl._drop_cache(fd)
    return fd, bypassed


def run_job(job: JobSpec, directory: str) -> dict:
    """Execute one job and return latency and throughput statistics."""
    size = job.file_mb * MB
    try:
        free_bytes = shutil.disk_usage(directory).free
    except OSError:
        free_bytes = 0
    # repeats=1: each job writes its file once, unlike the main disk test which
    # rewrites it per repeat and must budget flash wear across all of them.
    allowed, notice = limits.safe_disk_mb(job.file_mb, free_bytes, 1)
    if allowed < job.file_mb:
        size = allowed * MB
    if size < job.block_size * 16:
        return {"skipped": True, "name": job.name,
                "reason": f"not enough space for a {job.file_mb} MB test file"}

    path = os.path.join(directory, f".pcbench_io_{os.getpid()}")
    fd = None
    try:
        fd, cache_bypassed = _open_file(path, size, job.direct)
        blocks = max(1, size // job.block_size)

        stop = threading.Event()
        latencies: list[list[float]] = [[] for _ in range(job.queue_depth)]
        counts = [0] * job.queue_depth
        bytes_done = [0] * job.queue_depth
        read_ops = [0] * job.queue_depth
        write_ops = [0] * job.queue_depth
        errors: list[str] = []
        payload = os.urandom(job.block_size)

        read_methods: list[str] = []

        def worker(slot: int) -> None:
            rnd = random.Random(1000 + slot)
            position = slot % blocks
            # Each thread gets its own reader: it bypasses the page cache
            # where the platform allows, and on a platform without positional
            # reads it is also what keeps the threads off a shared file
            # pointer, a race that reads the wrong offset silently rather
            # than raising.
            reader = wl.DirectReader(path, fd, job.block_size,
                                     direct=job.direct)
            read_methods.append(reader.method)
            try:
                while not stop.is_set():
                    if job.is_random:
                        offset = rnd.randrange(blocks) * job.block_size
                    else:
                        offset = (position % blocks) * job.block_size
                        position += job.queue_depth
                    is_read = _choose_read(job, rnd)
                    t0 = clock()
                    if is_read:
                        n = reader.read(offset)
                        read_ops[slot] += 1
                    else:
                        # The reader is read-only, so writes go through the
                        # shared descriptor.
                        n = wl.pwrite(fd, payload, offset)
                        write_ops[slot] += 1
                    latency = (clock() - t0) * 1e6
                    if len(latencies[slot]) < 200_000:
                        latencies[slot].append(latency)
                    counts[slot] += 1
                    bytes_done[slot] += n
            except OSError as e:
                errors.append(str(e))
            finally:
                reader.close()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(job.queue_depth)]
        start = clock()
        for t in threads:
            t.start()
        time.sleep(job.seconds)
        stop.set()
        for t in threads:
            t.join(timeout=10.0)
        elapsed = clock() - start

        if job.writes:
            os.fsync(fd)

        total_ops = sum(counts)
        total_bytes = sum(bytes_done)
        if not total_ops or elapsed <= 0:
            return {"skipped": True, "name": job.name,
                    "reason": "no I/O completed"
                              + (f": {errors[0]}" if errors else "")}

        merged = sorted(v for lst in latencies for v in lst)
        result = {
            "name": job.name,
            "spec": job.to_dict(),
            "describe": job.describe(),
            "iops": total_ops / elapsed,
            "throughput_mb_s": total_bytes / elapsed / MB,
            "operations": total_ops,
            "read_ops": sum(read_ops),
            "write_ops": sum(write_ops),
            "elapsed_s": round(elapsed, 3),
            "file_mb": size // MB,
        }
        if notice:
            result["safety_notice"] = notice
        if merged:
            result["latency_us"] = {
                "min": round(merged[0], 1),
                "p50": round(_pct(merged, 50), 1),
                "p95": round(_pct(merged, 95), 1),
                "p99": round(_pct(merged, 99), 1),
                "p999": round(_pct(merged, 99.9), 1),
                "max": round(merged[-1], 1),
            }
        # Judge the reads on what they measured, not on which flag was set.
        # A filesystem can accept the bypass and serve from cache anyway --
        # btrfs with compression and most network mounts do -- and the only
        # way to tell is that the latency is faster than storage can be.
        method = read_methods[0] if read_methods else "buffered"
        ramfs = wl.memory_filesystem(directory)
        if not job.direct and not ramfs:
            reads_bypassed = False
        elif sum(read_ops) or ramfs:
            reads_bypassed, caution = wl.direct_read_note(
                method, {"p50_us": result.get("latency_us", {}).get("p50")},
                ramfs)
            if not reads_bypassed:
                result["caution"] = caution
        else:
            reads_bypassed = cache_bypassed        # write-only job
        result["cache_bypassed"] = reads_bypassed
        result["direct_method"] = method
        result["memory_filesystem"] = ramfs
        result["sequential_cache_bypassed"] = cache_bypassed and not ramfs
        if errors:
            result["errors"] = errors[:3]
        return result
    except OSError as e:
        return {"skipped": True, "name": job.name,
                "reason": f"{type(e).__name__}: {e}"}
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(path)
        except OSError:
            pass


def _choose_read(job: JobSpec, rnd: random.Random) -> bool:
    if job.pattern in ("read", "randread"):
        return True
    if job.pattern in ("write", "randwrite"):
        return False
    return rnd.randrange(100) < job.read_pct


def _pct(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(len(sorted_values) * percentile / 100.0)
    return sorted_values[min(index, len(sorted_values) - 1)]


def run(jobs: list[JobSpec], directory: str, quiet: bool = True) -> dict:
    """Run a suite of jobs in sequence."""
    results = []
    for job in jobs:
        if not quiet:
            print(f"    io: {job.name} ({job.describe()}) ...", flush=True)
        results.append(run_job(job, directory))
    return {"jobs": results,
            "note": ("queue depth is reached with blocking calls on threads "
                     "rather than async submission; very high depths on very "
                     "fast NVMe carry more CPU overhead than fio would")}


def render(result: dict | None) -> str:
    """Terminal table for the I/O suite."""
    if not result or not result.get("jobs"):
        return ""
    lines = [f"  {'JOB':<12} {'PATTERN':<28} {'IOPS':>10} {'MB/s':>9} "
             f"{'p50 us':>8} {'p99 us':>9}"]
    lines.append("  " + "-" * 80)
    for job in result["jobs"]:
        if job.get("skipped"):
            lines.append(f"  {job['name'][:12]:<12} skipped — "
                         f"{job.get('reason', '')}")
            continue
        lat = job.get("latency_us") or {}
        lines.append(
            f"  {job['name'][:12]:<12} {job['describe'][:28]:<28} "
            f"{job['iops']:>10,.0f} {job['throughput_mb_s']:>9,.1f} "
            f"{lat.get('p50', 0):>8,.0f} {lat.get('p99', 0):>9,.0f}")
        if job.get("caution"):
            lines.append(f"      ! {job['caution']}")
    lines.append("")
    lines.append(f"      {result.get('note', '')}")
    return "\n".join(lines)
