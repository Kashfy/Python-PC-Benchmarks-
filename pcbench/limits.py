"""Hardware-safety limits.

A benchmark is supposed to load hardware hard — that is the point, and modern
CPUs, GPUs, and SSDs are designed to run at 100% indefinitely with their own
thermal and wear protection. What a benchmark must *never* do is push a system
into a state the hardware cannot protect itself from:

* **Memory exhaustion.** Allocating more than physical RAM does not just fail
  cleanly — it drives the OS into swap thrashing, which freezes the machine and
  writes tens of gigabytes to the SSD as a side effect. Worse on Linux, where
  the OOM killer may terminate unrelated processes.
* **Filling the disk.** A disk with no free space can corrupt in-flight writes
  from *other* applications and, on some filesystems, prevent recovery.
* **Excessive flash wear.** SSD endurance is finite (TBW). A single run should
  never consume a meaningful fraction of it.
* **RAM-backed temp directories.** On many Linux systems ``/tmp`` is tmpfs,
  which is RAM. A "disk" test writing there consumes memory, not storage.

These caps are deliberately conservative. They clamp rather than abort, and the
clamp is always reported so the user knows the request was reduced.
"""

from __future__ import annotations

MB = 1024 * 1024
GB = 1024 * MB

# The memory test allocates TWO buffers (source and destination), so the real
# footprint is 2x the requested size. Capping the buffer at 1/8 of RAM keeps
# the total under 1/4 of RAM — comfortably clear of swap on any machine.
MEM_FRACTION_OF_RAM = 0.125
MEM_MIN_MB = 8
MEM_DEFAULT_CAP_MB = 4096          # used when RAM cannot be detected

# Cumulative bytes the disk test may write in one run (file size x repeats).
# 16 GB is roughly 0.01% of a typical consumer SSD's rated endurance, so even
# hundreds of runs stay negligible.
DISK_MAX_TOTAL_WRITE_MB = 16 * 1024
DISK_MIN_MB = 4
# Require meaningfully more free space than the file itself: filesystems slow
# down and misbehave near full, and other processes need room too.
DISK_FREE_HEADROOM = 1.5

# Temperature / throttle thresholds at which a sustained run stops early.
# Reaching these is not itself damage — the hardware throttles and ultimately
# shuts down to protect itself — but continuing past them tells us nothing new
# and risks an abrupt shutdown on a machine whose cooling has already failed.
THERMAL_ABORT_CELSIUS = 100.0
THERMAL_ABORT_SPEED_LIMIT_PCT = 40


def safe_mem_mb(requested_mb: int, total_ram_bytes: int) -> tuple[int, str | None]:
    """Clamp the memory-test buffer to a size that cannot induce swapping.

    Returns ``(allowed_mb, notice)`` where ``notice`` is None if unchanged.
    """
    requested_mb = max(MEM_MIN_MB, int(requested_mb))
    if total_ram_bytes and total_ram_bytes > 0:
        cap = int(total_ram_bytes * MEM_FRACTION_OF_RAM / MB)
    else:
        cap = MEM_DEFAULT_CAP_MB
    cap = max(MEM_MIN_MB, cap)
    if requested_mb <= cap:
        return requested_mb, None
    return cap, (
        f"memory test reduced from {requested_mb} MB to {cap} MB: the test "
        f"allocates two buffers, and a larger size risks swap thrashing "
        f"(which stalls the machine and writes heavily to the SSD)")


def safe_disk_mb(requested_mb: int, free_bytes: int,
                 repeats: int) -> tuple[int, str | None]:
    """Clamp the disk-test file size for free space and cumulative flash wear.

    Guards two independent things: that a single file fits with headroom, and
    that ``file_size x repeats`` stays a negligible fraction of SSD endurance.
    """
    requested_mb = max(DISK_MIN_MB, int(requested_mb))
    repeats = max(1, int(repeats))
    notices = []

    allowed = requested_mb

    # 1. Cumulative write volume across all repeats.
    per_run_cap = max(DISK_MIN_MB, DISK_MAX_TOTAL_WRITE_MB // repeats)
    if allowed > per_run_cap:
        notices.append(
            f"limited to {per_run_cap} MB so total writes stay under "
            f"{DISK_MAX_TOTAL_WRITE_MB // 1024} GB this run "
            f"({repeats} repeats), preserving SSD endurance")
        allowed = per_run_cap

    # 2. Free space, with headroom so the filesystem never runs to full.
    if free_bytes and free_bytes > 0:
        space_cap = int(free_bytes / DISK_FREE_HEADROOM / MB)
        if allowed > space_cap:
            notices.append(
                f"limited to {space_cap} MB to leave "
                f"{DISK_FREE_HEADROOM:g}x headroom on a filesystem with "
                f"{free_bytes / GB:.1f} GB free")
            allowed = space_cap

    allowed = max(DISK_MIN_MB, allowed)
    if allowed >= requested_mb:
        return requested_mb, None
    return allowed, ("disk test " + "; ".join(notices))


def total_write_mb(file_mb: int, repeats: int) -> int:
    """Bytes this run will write to storage, so it can be disclosed."""
    return max(1, int(file_mb)) * max(1, int(repeats))


def thermal_should_abort(thermal: str | None) -> tuple[bool, str]:
    """Decide whether a sustained run should stop early on thermal grounds.

    ``thermal`` is the free-form string from ``system.thermal_pressure()``:
    ``"nominal"``, ``"throttled (N%)"``, or ``"max NNC"``.
    """
    if not thermal:
        return False, ""
    import re

    m = re.search(r"throttled \((\d+)%\)", thermal)
    if m and int(m.group(1)) < THERMAL_ABORT_SPEED_LIMIT_PCT:
        return True, (f"CPU throttled to {m.group(1)}% of nominal speed — "
                      f"stopping early; cooling is the limiting factor")
    m = re.search(r"max (-?\d+(?:\.\d+)?)C", thermal)
    if m and float(m.group(1)) >= THERMAL_ABORT_CELSIUS:
        return True, (f"package temperature reached {m.group(1)}C — stopping "
                      f"early to avoid an abrupt thermal shutdown")
    return False, ""
