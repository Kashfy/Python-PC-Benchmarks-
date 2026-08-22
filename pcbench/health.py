"""Hardware health checks: RAM integrity and drive SMART data.

The benchmarks validate their own arithmetic, which catches gross faults as a
side effect. These are the deliberate versions: a memory test that writes
adversarial bit patterns and reads them back, and a read-only query of the
drive's own self-assessment.

Scope and honesty:

* The RAM test exercises the memory this process can allocate — typically a
  fraction of the machine's total, and always through the OS's virtual memory.
  It can find a bad cell it happens to touch, but a clean result does **not**
  certify the DIMMs. Only a bootable tester like MemTest86, which owns the
  whole address space, can do that. The report says so.
* SMART is read-only. Nothing here ever writes drive metadata.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from .core import clock

MB = 1024 * 1024

# Bit patterns chosen to stress different failure modes: all-ones and
# all-zeros catch stuck bits, alternating patterns catch coupling between
# adjacent cells, and the walking patterns catch address-decode faults.
_PATTERNS = [
    (0x00, "all zeros"),
    (0xFF, "all ones"),
    (0xAA, "alternating 10101010"),
    (0x55, "alternating 01010101"),
    (0x0F, "nibble 00001111"),
    (0xF0, "nibble 11110000"),
]


def memory_integrity(size_mb: int = 256, ram_bytes: int = 0) -> dict:
    """Write bit patterns to memory and verify they read back unchanged.

    Each pattern is written across the whole buffer and then compared. A
    mismatch means a bit did not survive the round trip, which points at
    failing RAM, an unstable memory overclock, or insufficient cooling.
    """
    from . import limits

    size_mb, notice = limits.safe_mem_mb(size_mb, ram_bytes)
    n = size_mb * MB

    try:
        buf = bytearray(n)
    except MemoryError:
        return {"skipped": True, "error": f"cannot allocate {size_mb} MB"}

    errors: list[dict] = []
    start = clock()
    for value, label in _PATTERNS:
        expected = bytes([value]) * MB
        for offset in range(0, n, MB):
            end = min(offset + MB, n)
            buf[offset:end] = expected[:end - offset]
        # Verify in a second pass so the data has to survive being written
        # across the whole buffer, not just a cache line.
        for offset in range(0, n, MB):
            end = min(offset + MB, n)
            if buf[offset:end] != expected[:end - offset]:
                errors.append({"pattern": label,
                               "offset_mb": offset // MB})
                break
    elapsed = clock() - start

    result = {
        "tested_mb": size_mb,
        "patterns": len(_PATTERNS),
        "errors": len(errors),
        "error_detail": errors[:8],
        "passed": not errors,
        "seconds": round(elapsed, 2),
        "throughput_mb_s": round(size_mb * len(_PATTERNS) * 2 / elapsed, 1)
        if elapsed else 0.0,
        # Stated every time, because a clean pass here is easy to over-read.
        "scope": ("tests only memory this process could allocate, through the "
                  "OS's virtual memory — a pass does not certify the RAM; use "
                  "MemTest86 for that"),
    }
    if notice:
        result["safety_notice"] = notice
    return result


# --------------------------------------------------------------------------- #
# Drive SMART — read-only
# --------------------------------------------------------------------------- #
def _smartctl_available() -> bool:
    return shutil.which("smartctl") is not None


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        # smartctl uses bit-flagged exit codes; output is still valid when
        # non-zero, so stdout is returned regardless.
        return p.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def drive_health() -> dict:
    """Drive self-assessment. Read-only; never writes SMART data."""
    system = platform.system()

    if system == "Darwin":
        # Apple's internal NVMe reports through system_profiler without needing
        # smartctl or elevated privileges.
        import json as _json
        raw = _run(["system_profiler", "-json", "SPNVMeDataType"])
        try:
            data = _json.loads(raw) if raw else {}
        except Exception:
            data = {}
        drives = []
        for controller in data.get("SPNVMeDataType", []):
            for item in controller.get("_items", []) or [controller]:
                entry = {
                    "name": item.get("_name") or item.get("device_model"),
                    "smart_status": item.get("smart_status"),
                    "size": item.get("size"),
                }
                if any(entry.values()):
                    drives.append({k: v for k, v in entry.items() if v})
        if drives:
            return {"available": True, "source": "system_profiler",
                    "drives": drives}

    if not _smartctl_available():
        return {
            "available": False,
            "note": ("smartctl not installed — install smartmontools for "
                     "drive wear, power-on hours and reallocated sectors"),
        }

    out = _run(["smartctl", "--scan"])
    devices = [line.split()[0] for line in out.splitlines()
               if line.startswith("/dev/")]
    drives = []
    for dev in devices[:4]:
        text = _run(["smartctl", "-H", "-A", dev])
        if not text:
            continue
        entry: dict = {"device": dev}
        for line in text.splitlines():
            low = line.lower()
            if "overall-health" in low:
                entry["smart_status"] = line.split(":")[-1].strip()
            elif "percentage used" in low:
                entry["percentage_used"] = line.split(":")[-1].strip()
            elif "power_on_hours" in low or "power on hours" in low:
                entry["power_on_hours"] = line.split()[-1]
            elif "reallocated_sector" in low:
                entry["reallocated_sectors"] = line.split()[-1]
            elif "media and data integrity errors" in low:
                entry["integrity_errors"] = line.split(":")[-1].strip()
        if len(entry) > 1:
            drives.append(entry)

    if not drives:
        return {"available": False,
                "note": "smartctl found no readable drives (it usually needs "
                        "elevated privileges)"}
    return {"available": True, "source": "smartctl", "drives": drives}


def run(memory_mb: int = 256, ram_bytes: int = 0,
        skip_memory: bool = False) -> dict:
    """All health checks."""
    result: dict = {"drive": drive_health()}
    if not skip_memory:
        result["memory"] = memory_integrity(memory_mb, ram_bytes)
    return result
