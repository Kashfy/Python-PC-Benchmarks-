"""Application-shaped workloads: what the machine does, not what it can do.

The synthetic tests in :mod:`pcbench.workloads` isolate one subsystem each,
which is exactly right for diagnosis and exactly wrong for answering "will this
machine be good at my job?". Real software mixes subsystems in ratios that no
single synthetic test reproduces:

* A **database** is small random reads, an fsync-bound write path, and B-tree
  pointer chasing — it stresses storage latency and cache, and is nearly
  indifferent to peak sequential bandwidth.
* A **renderer** is branchy float math over a working set that fits in L2, with
  almost no memory traffic — the opposite balance.
* **Log/text processing** is a byte-at-a-time scan bounded by memory bandwidth
  and branch prediction.
* **Image processing** is a strided 2-D access pattern that punishes small
  caches specifically.
* **Video encoding** is the only common desktop workload that saturates every
  core *and* the vector units for minutes at a time, which is why it is the
  workload that finds inadequate cooling first.

Each benchmark here validates its own output. A renderer that produces the
wrong pixel or a database that returns the wrong row count is reporting a
hardware fault, not a fast result — the same contract the synthetic tests use.

Everything runs on the standard library. ``video`` is the sole exception: it
shells out to an ``ffmpeg`` that is already installed, and reports itself as
skipped when there is none.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import tempfile

from .core import (ValidationError, check_close, check_exact, clock,
                   summarize, timed_loop, warmup)

MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# Database (SQLite OLTP)
# --------------------------------------------------------------------------- #
# SQLite is the most-deployed database in the world and ships with Python, so
# this measures a real storage engine rather than a model of one. The table is
# built once and reused: the metric of interest is steady-state transaction
# rate, not the cost of creating a schema.
# --------------------------------------------------------------------------- #
_ROWS = 20_000
# Chosen to divide _ROWS exactly, so the expected row count per bucket is an
# integer and a mismatch is unambiguously a fault rather than rounding.
_BUCKETS = 500


def _seed_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = MEMORY;
        PRAGMA synchronous = OFF;
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            bucket INTEGER NOT NULL,
            score REAL NOT NULL
        );
        CREATE INDEX idx_bucket ON items(bucket);
    """)
    conn.executemany(
        "INSERT INTO items (id, name, bucket, score) VALUES (?, ?, ?, ?)",
        [(i, f"item-{i:06d}", i % _BUCKETS, (i * 2654435761 % 100000) / 1000.0)
         for i in range(_ROWS)])
    conn.commit()


def bench_sqlite(seconds: float, repeats: int) -> dict:
    """Mixed read/write OLTP transactions per second.

    Runs against an in-memory database on purpose: with the file cached this
    would measure the page cache anyway, and isolating the *engine* from the
    disk keeps this test orthogonal to the storage tests. Disk-bound database
    behaviour is what ``disk`` random-read IOPS and fsync latency measure.
    """
    conn = sqlite3.connect(":memory:")
    _seed_db(conn)
    cur = conn.cursor()
    state = {"n": 0}

    def chunk():
        # One "transaction batch": indexed lookups, a range scan with an
        # aggregate, and an update — the shape of a typical request handler.
        i = state["n"] % _BUCKETS
        state["n"] += 1
        rows = cur.execute(
            "SELECT COUNT(*), AVG(score) FROM items WHERE bucket = ?",
            (i,)).fetchone()
        if rows[0] != _ROWS // _BUCKETS:
            raise ValidationError(
                f"sqlite: index scan returned {rows[0]} rows, "
                f"expected {_ROWS // _BUCKETS}")
        cur.execute("SELECT name FROM items WHERE id = ?", (i * 7 % _ROWS,))
        cur.fetchone()
        cur.execute("UPDATE items SET score = score WHERE id = ?", (i,))

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n / elapsed)
    conn.close()
    s = summarize(rates)
    return {"unit": "txn/s", "rate": s["median"], "validated": True,
            "note": f"SQLite OLTP mix over {_ROWS} rows "
                    f"(indexed lookup + aggregate + update)", **s}


def _full_flush(fd: int) -> str:
    """Force data to the storage medium, returning the mechanism used.

    On macOS plain ``fsync`` only pushes data out of the OS cache and into the
    drive's own volatile buffer; ``F_FULLFSYNC`` is what actually makes it
    durable, and the two differ by an order of magnitude. Measuring the wrong
    one would report a laptop as having enterprise-class commit latency.
    """
    if platform.system() == "Darwin":
        try:
            import fcntl
            F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)
            fcntl.fcntl(fd, F_FULLFSYNC)
            return "F_FULLFSYNC"
        except (ImportError, OSError):
            pass
    os.fsync(fd)
    return "fsync"


def bench_fsync(seconds: float, path: str) -> dict:
    """Durable-commit latency: how long one flush-to-storage actually takes.

    This is the single number that decides real database write throughput, and
    it is invisible to sequential-bandwidth tests. A consumer SSD that writes
    3 GB/s may still commit only a few hundred transactions per second, while
    an enterprise drive with power-loss protection commits tens of thousands.
    A suspiciously large figure (>100k/s) usually means the drive is lying
    about flushes rather than that it is fast.
    """
    fd = None
    tmp = os.path.join(path, f".pcbench_fsync_{os.getpid()}")
    mechanism = "fsync"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        payload = b"x" * 4096
        budget = max(0.3, min(seconds, 3.0))
        latencies: list[float] = []
        start = clock()
        while clock() - start < budget:
            t0 = clock()
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            mechanism = _full_flush(fd)
            latencies.append((clock() - t0) * 1e6)
            if len(latencies) >= 20_000:
                break
        if not latencies:
            return {"skipped": True, "reason": "no fsync samples collected"}
        latencies.sort()
        mid = latencies[len(latencies) // 2]
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
        rate = 1e6 / mid if mid else 0.0
        result = {
            "unit": "commits/s", "rate": rate,
            "median_us": round(mid, 2), "p99_us": round(p99, 2),
            "samples": len(latencies),
            "mechanism": mechanism,
            "note": f"durable 4 KiB write + {mechanism}; the ceiling on "
                    f"database commit rate",
        }
        if rate > 100_000:
            result["caution"] = (
                "over 100k commits/s implies the device or filesystem is "
                "acknowledging flushes without persisting them — data is at "
                "risk on power loss")
        return result
    except OSError as e:
        return {"skipped": True, "reason": f"fsync test unavailable: {e}"}
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Rendering (ray tracer)
# --------------------------------------------------------------------------- #
# Branchy scalar float math with a tiny working set. This is the profile of
# rendering, physics, and simulation code, and it is the workload where
# single-core IPC and boost-clock behaviour show up most clearly.
# --------------------------------------------------------------------------- #
_SPHERES = [
    # (cx, cy, cz, radius, albedo)
    (0.0, 0.0, -3.0, 1.0, 0.8),
    (1.6, 0.3, -4.0, 0.9, 0.55),
    (-1.7, -0.2, -3.5, 0.7, 0.35),
    (0.0, -101.0, -4.0, 100.0, 0.65),
]


def _trace(width: int, height: int) -> float:
    """Render one frame, returning summed luminance (the validation value)."""
    total = 0.0
    inv_w, inv_h = 1.0 / width, 1.0 / height
    for py in range(height):
        y = (1.0 - 2.0 * (py + 0.5) * inv_h) * 0.75
        for px in range(width):
            x = (2.0 * (px + 0.5) * inv_w - 1.0)
            # Normalise the primary ray.
            dz = -1.0
            inv_len = 1.0 / math.sqrt(x * x + y * y + 1.0)
            dx, dy, dz = x * inv_len, y * inv_len, dz * inv_len
            best_t, best_albedo, nx, ny, nz = 1e30, 0.0, 0.0, 0.0, 0.0
            for cx, cy, cz, r, albedo in _SPHERES:
                ox, oy, oz = -cx, -cy, -cz
                b = ox * dx + oy * dy + oz * dz
                c = ox * ox + oy * oy + oz * oz - r * r
                disc = b * b - c
                if disc <= 0.0:
                    continue
                t = -b - math.sqrt(disc)
                if 0.001 < t < best_t:
                    best_t = t
                    best_albedo = albedo
                    hx, hy, hz = dx * t, dy * t, dz * t
                    nx, ny, nz = (hx - cx) / r, (hy - cy) / r, (hz - cz) / r
            if best_t < 1e29:
                # Single directional light with a hard shadow term folded in.
                lam = nx * 0.577 + ny * 0.577 + nz * 0.577
                total += best_albedo * max(0.0, lam)
            else:
                total += 0.15 + 0.35 * (1.0 - y)
    return total


def bench_raytrace(seconds: float, repeats: int) -> dict:
    """Frames per second for a fixed 64x48 ray-traced scene."""
    w, h = 64, 48
    reference = _trace(w, h)

    def chunk():
        # Rendering is deterministic; a differing image means the FPU or memory
        # produced a wrong answer.
        check_close("raytrace", _trace(w, h), reference, rel_tol=1e-12)

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n / elapsed)
    s = summarize(rates)
    return {"unit": "frames/s", "rate": s["median"], "validated": True,
            "resolution": f"{w}x{h}", "rays_per_frame": w * h,
            "note": "scalar ray tracer: branchy float math, cache-resident",
            **s}


# --------------------------------------------------------------------------- #
# Image processing (separable convolution)
# --------------------------------------------------------------------------- #
# A strided 2-D pass over a buffer larger than L1. The column pass in
# particular walks memory with a stride equal to the row width, which is the
# access pattern that separates a large L2 from a small one.
# --------------------------------------------------------------------------- #
def _blur(buf: bytearray, w: int, h: int) -> bytearray:
    """3-tap separable box blur, rows then columns."""
    tmp = bytearray(len(buf))
    out = bytearray(len(buf))
    for y in range(h):
        row = y * w
        for x in range(1, w - 1):
            i = row + x
            tmp[i] = (buf[i - 1] + buf[i] + buf[i + 1]) // 3
    for y in range(1, h - 1):
        row = y * w
        for x in range(w):
            i = row + x
            out[i] = (tmp[i - w] + tmp[i] + tmp[i + w]) // 3
    return out


def bench_image(seconds: float, repeats: int) -> dict:
    """Megapixels per second through a separable blur."""
    w, h = 320, 240
    src = bytearray(((x * 37 + y * 91) & 0xFF) for y in range(h)
                    for x in range(w))
    reference = sum(_blur(src, w, h))
    megapixels = (w * h) / 1e6

    def chunk():
        check_exact("image", sum(_blur(src, w, h)), reference)

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n * megapixels / elapsed)
    s = summarize(rates)
    return {"unit": "MP/s", "rate": s["median"], "validated": True,
            "resolution": f"{w}x{h}",
            "note": "separable 3x3 blur; strided 2-D access stresses L2",
            **s}


# --------------------------------------------------------------------------- #
# Text / log processing
# --------------------------------------------------------------------------- #
_LOG_LINE = ('127.0.0.{ip} - - [10/Oct/2024:13:55:{sec:02d} +0000] '
             '"GET /api/v{v}/items/{id} HTTP/1.1" {status} {bytes}\n')
_LOG_RE = re.compile(
    r'^(\d+\.\d+\.\d+\.\d+) \S+ \S+ \[([^\]]+)\] '
    r'"(\w+) (\S+) [^"]*" (\d{3}) (\d+)$')


def _log_corpus(lines: int = 20_000) -> str:
    return "".join(
        _LOG_LINE.format(ip=i % 250 + 1, sec=i % 60, v=i % 3 + 1,
                         id=i * 17 % 99999,
                         status=(200, 404, 500, 301)[i % 4],
                         bytes=i * 13 % 50000)
        for i in range(lines))


def bench_logparse(seconds: float, repeats: int) -> dict:
    """Regex log parsing throughput in MB/s.

    Log ingestion, ETL, and build-system output scanning all reduce to this:
    a linear scan with a backtracking matcher per line.
    """
    text = _log_corpus()
    lines = text.splitlines()
    nbytes = len(text.encode())
    expected_errors = sum(1 for i in range(len(lines)) if i % 4 in (2,))

    def chunk():
        errors = 0
        for line in lines:
            m = _LOG_RE.match(line)
            if m is None:
                raise ValidationError("logparse: line failed to match")
            if m.group(5) == "500":
                errors += 1
        if errors != expected_errors:
            raise ValidationError(
                f"logparse: counted {errors} errors, expected "
                f"{expected_errors}")

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, n = timed_loop(chunk, seconds)
        rates.append(n * nbytes / elapsed / MB)
    s = summarize(rates)
    return {"unit": "MB/s", "rate": s["median"], "validated": True,
            "lines": len(lines),
            "note": "regex-parse Apache-style access logs", **s}


# --------------------------------------------------------------------------- #
# Video encoding (optional; uses an installed ffmpeg)
# --------------------------------------------------------------------------- #
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


#: Source clip parameters. ``testsrc2`` is deliberately busier than ``testsrc``
#: — moving gradients and noise rather than flat colour bars — because a
#: trivially compressible source turns the benchmark into a measurement of how
#: fast x264 can skip macroblocks.
_VIDEO_SIZE = "1920x1080"
_VIDEO_FPS = 30
_VIDEO_PRESET = "medium"

#: Bounds on the measured encode. Below the floor the result is dominated by
#: process startup; above the ceiling a slow machine waits far too long.
_VIDEO_MIN_FRAMES = 60
_VIDEO_MAX_FRAMES = 3000
_VIDEO_MAX_SECONDS = 120.0


def _encode(exe: str, frames: int, out_path: str,
            timeout: float) -> tuple[float, subprocess.CompletedProcess]:
    """Run one encode, returning ``(elapsed_seconds, completed_process)``."""
    duration = max(1, math.ceil(frames / _VIDEO_FPS))
    cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi",
           "-i", f"testsrc2=size={_VIDEO_SIZE}:rate={_VIDEO_FPS}:d={duration}",
           "-c:v", "libx264", "-preset", _VIDEO_PRESET, "-crf", "23",
           "-frames:v", str(frames), out_path]
    start = clock()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return clock() - start, proc


def bench_video(seconds: float = 5.0, workdir: str | None = None) -> dict:
    """H.264 encode speed in frames per second, via an installed ffmpeg.

    Video encoding is the archetypal "all cores, all vector units, for
    minutes" desktop workload, and it is usually the first thing to expose a
    cooling system that cannot hold boost clocks. Source frames are generated
    by ffmpeg itself, so nothing is downloaded and the input is identical on
    every machine.

    The frame count is calibrated rather than fixed. A fixed count cannot serve
    both ends of the hardware range this tool targets: 300 frames is under a
    second on a modern desktop — short enough that process startup dominates
    the result — and several minutes on a single-board computer. So a short
    probe encode measures the machine first, and the real encode is sized from
    that to fill the requested time budget.
    """
    exe = ffmpeg_path()
    if not exe:
        return {"skipped": True,
                "reason": "ffmpeg not found on PATH",
                "hint": "install ffmpeg to enable the video-encode benchmark"}

    workdir = workdir or tempfile.gettempdir()
    out_path = os.path.join(workdir, f".pcbench_video_{os.getpid()}.mp4")
    budget = max(2.0, min(float(seconds) * 2.0, _VIDEO_MAX_SECONDS))

    try:
        # Probe: also warms the page cache and pays libx264's one-off setup,
        # neither of which should land in the measured result.
        probe_elapsed, proc = _encode(exe, _VIDEO_MIN_FRAMES, out_path,
                                      timeout=_VIDEO_MAX_SECONDS)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            return {"skipped": True,
                    "reason": "ffmpeg failed"
                              + (f": {detail[-1]}" if detail else ""),
                    "hint": "the installed ffmpeg may lack libx264"}

        probe_fps = (_VIDEO_MIN_FRAMES / probe_elapsed
                     if probe_elapsed > 0 else _VIDEO_MIN_FRAMES)
        frames = int(max(_VIDEO_MIN_FRAMES,
                         min(_VIDEO_MAX_FRAMES, probe_fps * budget)))

        elapsed, proc = _encode(exe, frames, out_path,
                                timeout=_VIDEO_MAX_SECONDS + 60.0)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            return {"skipped": True,
                    "reason": "ffmpeg failed"
                              + (f": {detail[-1]}" if detail else "")}
    except subprocess.TimeoutExpired:
        return {"skipped": True,
                "reason": f"ffmpeg encode exceeded "
                          f"{_VIDEO_MAX_SECONDS:.0f}s; this machine is too "
                          f"slow for the video benchmark"}
    except OSError as e:
        return {"skipped": True, "reason": f"could not run ffmpeg: {e}"}
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    if elapsed <= 0:
        return {"skipped": True, "reason": "encode completed too fast to time"}

    return {"unit": "fps", "rate": frames / elapsed,
            "frames": frames, "elapsed_s": round(elapsed, 2),
            "probe_fps": round(probe_fps, 1),
            "codec": f"libx264 {_VIDEO_SIZE} preset={_VIDEO_PRESET} crf=23",
            "note": "software H.264 encode; sustained all-core vector load"}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: Tests exposed to the CLI, mapped to a short description for ``--list-tests``.
DESCRIPTIONS = {
    "sqlite": "SQLite OLTP transactions (database engine)",
    "fsync": "Durable commit latency (database write ceiling)",
    "raytrace": "Ray-traced frames/s (rendering, simulation)",
    "image": "Separable blur megapixels/s (image processing)",
    "logparse": "Regex log parsing MB/s (ETL, log ingestion)",
    "video": "H.264 encode fps via ffmpeg (media production)",
}


def extract_rates(results: dict) -> dict:
    """Pull scoreable rates out of the app results already in ``results``."""
    out = {}
    for name in DESCRIPTIONS:
        entry = results.get(name)
        if isinstance(entry, dict) and not entry.get("skipped"):
            rate = entry.get("rate")
            if isinstance(rate, (int, float)) and rate > 0:
                out[name] = rate
    return out
