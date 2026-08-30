"""Diagnosis: why is this machine slower than it should be?

A benchmark answers "how fast is this?". It does not answer the question
people actually arrive with, which is "this used to be fine — what happened?"
Those are different jobs. A score cannot tell someone that their disk is 98%
full, that Spotlight has been reindexing for an hour, that the battery
adapter is unplugged, or that the machine has been thermally throttled since
it was put on a duvet.

So this module gathers evidence rather than scores it, and reports findings
ranked by how much each is likely to be costing. Every finding carries what
was measured, why it matters, and what to do — because "your disk is slow" is
not actionable and "your boot volume has 2 GB free, which stops the
filesystem finding contiguous space to write into" is.

Two rules, inherited from the rest of the tool:

* **An observation is not a verdict.** A spinning disk is not a fault; it is a
  specification. Long uptime is not a problem; it is a thing worth knowing.
  Findings say which they are, and the severity reflects confidence as much as
  impact.
* **Absence of evidence is reported as absence.** A check that could not run
  on this platform says so rather than silently passing, because a clean
  report that skipped half its checks is worse than no report.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time

#: Ranking. Critical means "this is measurably hurting you now"; warning
#: means "this commonly causes slowness and is present"; info is context that
#: shapes what the other findings mean, not a problem in itself.
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

#: A volume below this is where filesystems start struggling to find
#: contiguous free space, which shows up as general sluggishness long before
#: anything reports "disk full".
DISK_LOW_PERCENT = 10.0
DISK_CRITICAL_PERCENT = 5.0

#: Available memory below this means the machine is buying every new
#: allocation by evicting something else.
RAM_LOW_PERCENT = 15.0
RAM_CRITICAL_PERCENT = 8.0

#: Load per core. Above 0.7 the machine is busy; above 1.0 work is queuing.
LOAD_BUSY = 0.70
LOAD_SATURATED = 1.0

#: A single process holding this much of one core is worth naming.
PROCESS_CPU_PERCENT = 40.0

#: Uptime past which an unexplained slowdown is worth a reboot before
#: anything else is investigated.
LONG_UPTIME_DAYS = 21

#: Throughput lost between the first and last window of a sustained load.
DROOP_WARN_PERCENT = 15.0
DROOP_CRITICAL_PERCENT = 30.0


def _finding(ident: str, severity: str, area: str, title: str,
             evidence: str, impact: str, fix: str) -> dict:
    return {"id": ident, "severity": severity, "area": area, "title": title,
            "evidence": evidence, "impact": impact, "fix": fix}


def _run(cmd: list[str], timeout: int = 8) -> str:
    """Best-effort subprocess. Returns '' rather than raising, ever."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        return proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _powershell(script: str) -> str:
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 script], timeout=20)


# --------------------------------------------------------------------------- #
# Evidence: what is running
# --------------------------------------------------------------------------- #
def top_processes(limit: int = 6) -> dict:
    """The heaviest CPU consumers right now.

    The single most common answer to "why is my computer slow" is "because
    something is using it", and no amount of benchmarking will say which
    something.

    Each platform needs a different source for an honest answer. On Linux
    ``ps`` reports %CPU averaged over the whole life of the process, so
    something that saturated a core for an hour and then went idle still
    reads high — useless for "right now" — and /proc is sampled twice
    instead. BSD and macOS ``ps`` already report recent usage. Windows has a
    performance counter that means what it says.
    """
    if os.name == "nt":
        return _top_processes_windows(limit)
    if sys.platform.startswith("linux"):
        return _top_processes_linux(limit)
    out = _run(["ps", "-Ao", "pid,pcpu,pmem,comm"])
    if not out:
        return {"error": "could not list processes"}
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rows.append({"pid": int(parts[0]), "cpu_percent": float(parts[1]),
                         "mem_percent": float(parts[2]),
                         "name": os.path.basename(parts[3].strip())})
        except ValueError:
            continue
    if not rows:
        return {"error": "no process could be parsed"}
    rows.sort(key=lambda r: r["cpu_percent"], reverse=True)
    return {"processes": rows[:limit], "counted": len(rows),
            "total_cpu_percent": round(sum(r["cpu_percent"] for r in rows), 1)}


#: Ticks per second for /proc CPU accounting. Effectively always 100, but
#: the kernel is entitled to differ and does on some embedded builds.
def _clock_ticks() -> float:
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (AttributeError, ValueError, OSError):
        return 100.0


def _proc_cpu_times() -> dict:
    """``{pid: (ticks, name)}`` for every process /proc will show us."""
    out = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8",
                      errors="replace") as handle:
                data = handle.read()
        except OSError:
            continue                    # the process exited; entirely normal
        parsed = parse_proc_stat(data)
        if parsed:
            out[int(entry)] = parsed
    return out


def parse_proc_stat(data: str) -> tuple[int, str] | None:
    """``(utime + stime in ticks, name)`` from one ``/proc/PID/stat``.

    The comm field is field 2, is wrapped in parentheses, and may itself
    contain spaces and parentheses — ``(Web Content)``, ``(foo (bar))``. A
    plain split on whitespace therefore misaligns every field after it, so
    the split has to start after the *last* ``)``.
    """
    close = data.rfind(")")
    if close <= 0:
        return None
    name = data[data.find("(") + 1:close]
    fields = data[close + 2:].split()
    if len(fields) < 13:
        return None
    try:
        return int(fields[11]) + int(fields[12]), name
    except ValueError:
        return None


def _top_processes_linux(limit: int, interval: float = 0.3) -> dict:
    first = _proc_cpu_times()
    if not first:
        return {"error": "could not read /proc"}
    time.sleep(interval)
    second = _proc_cpu_times()
    ticks = _clock_ticks()
    total_ram = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total_ram = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        total_ram = 0

    rows = []
    for pid, (after, name) in second.items():
        before = first.get(pid)
        if before is None:
            continue                    # started during the sample
        used = (after - before[0]) / ticks / interval * 100.0
        rows.append({"pid": pid, "cpu_percent": round(max(0.0, used), 1),
                     "mem_percent": _rss_percent(pid, total_ram),
                     "name": name})
    if not rows:
        return {"error": "no process could be sampled"}
    rows.sort(key=lambda r: r["cpu_percent"], reverse=True)
    return {"processes": rows[:limit], "counted": len(rows),
            "total_cpu_percent": round(sum(r["cpu_percent"] for r in rows), 1)}


def _rss_percent(pid: int, total_ram: int) -> float | None:
    if not total_ram:
        return None
    try:
        with open(f"/proc/{pid}/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return round(pages * os.sysconf("SC_PAGE_SIZE") / total_ram * 100, 1)


def _top_processes_windows(limit: int) -> dict:
    # Pipe-delimited rather than CSV: a process name containing a comma is
    # rare but would silently shift every column if it happened.
    script = (
        "Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | "
        "Where-Object {$_.Name -ne '_Total' -and $_.Name -ne 'Idle'} | "
        "Sort-Object PercentProcessorTime -Descending | "
        f"Select-Object -First {limit} | "
        "ForEach-Object { '{0}|{1}|{2}' -f $_.Name,"
        "$_.PercentProcessorTime,$_.IDProcess }")
    return _parse_windows_processes(_powershell(script), limit)


def _parse_windows_processes(out: str, limit: int) -> dict:
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[2]), "cpu_percent": float(parts[1]),
                         "mem_percent": None, "name": parts[0]})
        except ValueError:
            continue
    if not rows:
        return {"error": "could not list processes"}
    return {"processes": rows[:limit], "counted": len(rows),
            "total_cpu_percent": round(sum(r["cpu_percent"] for r in rows), 1)}


# --------------------------------------------------------------------------- #
# Evidence: memory and swap
# --------------------------------------------------------------------------- #
def memory_pressure() -> dict:
    """Available memory and swap use.

    Swap *existing* is normal and means nothing. Swap being actively written
    while free memory is scarce is the difference between a machine that is
    using its RAM and one that is out of it, and only the second is slow.
    """
    if sys.platform == "darwin":
        return _memory_macos()
    if os.name == "nt":
        return _memory_windows()
    return _memory_linux()


def parse_meminfo(text: str) -> dict:
    """``/proc/meminfo`` into bytes. Values there are kB unless stated."""
    fields = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if len(parts) >= 1 and parts[0].isdigit():
            scale = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
            fields[key] = int(parts[0]) * scale
    return fields


def _memory_linux() -> dict:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            fields = parse_meminfo(handle.read())
    except OSError as e:
        return {"error": f"cannot read /proc/meminfo: {e}"}
    total = fields.get("MemTotal", 0)
    available = fields.get("MemAvailable", fields.get("MemFree", 0))
    swap_total = fields.get("SwapTotal", 0)
    swap_used = swap_total - fields.get("SwapFree", 0)
    return _memory_result(total, available, swap_total, swap_used)


def parse_vm_stat(out: str) -> tuple[dict, int]:
    """``(page counts, page size)`` from ``vm_stat`` output."""
    page = 4096
    match = re.search(r"page size of (\d+) bytes", out)
    if match:
        page = int(match.group(1))
    counts = {k: int(v) for k, v in
              re.findall(r"^(.+?):\s+(\d+)\.", out, re.MULTILINE)}
    return counts, page


def parse_swapusage(text: str) -> tuple[int, int]:
    """``(total, used)`` bytes from ``sysctl vm.swapusage``."""
    total = used = 0
    for kind, value in re.findall(r"(total|used) = ([\d.]+)M", text):
        if kind == "total":
            total = int(float(value) * 1024 * 1024)
        elif kind == "used":
            used = int(float(value) * 1024 * 1024)
    return total, used


def _memory_macos() -> dict:
    counts, page = parse_vm_stat(_run(["vm_stat"]))
    if not counts:
        return {"error": "vm_stat returned nothing usable"}
    free = counts.get("Pages free", 0) + counts.get("Pages inactive", 0)
    total_pages = sum(counts.get(k, 0) for k in (
        "Pages free", "Pages active", "Pages inactive", "Pages wired down",
        "Pages occupied by compressor"))
    swap_total, swap_used = parse_swapusage(
        _run(["sysctl", "-n", "vm.swapusage"]))
    result = _memory_result(total_pages * page, free * page,
                            swap_total, swap_used)
    # Compressed pages are macOS keeping more in RAM rather than swapping;
    # counting them as pressure would flag every healthy Mac.
    result["compressed_bytes"] = counts.get(
        "Pages occupied by compressor", 0) * page
    return result


def _memory_windows() -> dict:
    out = _powershell(
        "$o = Get-CimInstance Win32_OperatingSystem; "
        "'{0},{1}' -f $o.TotalVisibleMemorySize,$o.FreePhysicalMemory")
    parts = out.strip().split(",")
    if len(parts) != 2 or not parts[0].strip().isdigit():
        return {"error": "could not query memory"}
    total = int(parts[0]) * 1024
    available = int(parts[1]) * 1024
    page = _powershell(
        "$p = Get-CimInstance Win32_PageFileUsage; "
        "'{0},{1}' -f ($p.AllocatedBaseSize | Measure-Object -Sum).Sum,"
        "($p.CurrentUsage | Measure-Object -Sum).Sum").strip().split(",")
    swap_total = swap_used = 0
    if len(page) == 2 and page[0].strip().isdigit():
        swap_total = int(page[0]) * 1024 * 1024
        swap_used = int(page[1]) * 1024 * 1024 if page[1].strip().isdigit() \
            else 0
    return _memory_result(total, available, swap_total, swap_used)


def _memory_result(total: int, available: int, swap_total: int,
                   swap_used: int) -> dict:
    return {
        "total_bytes": total,
        "available_bytes": available,
        "available_percent": (round(available / total * 100, 1)
                              if total else None),
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_used,
        "swap_used_percent": (round(swap_used / swap_total * 100, 1)
                              if swap_total else 0.0),
    }


# --------------------------------------------------------------------------- #
# Evidence: uptime, disk headroom, power mode
# --------------------------------------------------------------------------- #
def uptime_seconds() -> float | None:
    """How long since boot. Long uptime is context, not a fault."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/uptime", encoding="utf-8") as handle:
                return float(handle.read().split()[0])
        if sys.platform == "darwin":
            out = _run(["sysctl", "-n", "kern.boottime"])
            match = re.search(r"sec\s*=\s*(\d+)", out)
            if match:
                return max(0.0, time.time() - int(match.group(1)))
        if os.name == "nt":
            out = _powershell(
                "[int]((Get-Date) - (Get-CimInstance Win32_OperatingSystem)"
                ".LastBootUpTime).TotalSeconds").strip()
            if out.lstrip("-").isdigit():
                return float(out)
    except (OSError, ValueError, IndexError):
        return None
    return None


#: Filesystems that are not storage: they are kernel interfaces, and they
#: report themselves full as a matter of course.
_PSEUDO_FS = {"devfs", "devtmpfs", "tmpfs", "proc", "procfs", "sysfs",
              "cgroup", "cgroup2", "overlay", "squashfs", "autofs", "fdescfs",
              "map", "nullfs", "efivarfs", "ramfs", "debugfs", "tracefs"}

#: Below this a mount is a kernel interface or a boot stub, not a volume
#: whose fullness anyone experiences as slowness.
_REAL_VOLUME_BYTES = 1024 ** 3


def disk_headroom() -> dict:
    """Free space on every mount that can be measured.

    A nearly-full filesystem is one of the most common causes of a machine
    that "got slow" with nothing else changed, and one of the least often
    suspected, because nothing fails — writes just get slower as the
    allocator works harder to find contiguous space.
    """
    from . import storage

    volumes = []
    try:
        mounts = storage.mounts()
    except Exception as e:                      # pragma: no cover - defensive
        return {"error": f"could not enumerate mounts: {e}"}
    seen: set = set()
    for entry in mounts:
        mount = entry.get("mount")
        if not mount:
            continue
        if (entry.get("fstype") or "").lower() in _PSEUDO_FS:
            continue
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        # devfs and friends report themselves permanently 100% full, and a
        # ramdisk running out is not what anyone means by "my disk is full".
        if usage.total < _REAL_VOLUME_BYTES:
            continue
        # APFS volumes and Btrfs subvolumes share one pool, so the same free
        # space arrives once per mount. Report the pool, not the mount count.
        fingerprint = (usage.total, usage.free)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        volumes.append({
            "mount": mount,
            "kind": entry.get("kind"),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "free_percent": round(usage.free / usage.total * 100, 1),
        })
    if not volumes:
        return {"error": "no filesystem could be measured"}
    volumes.sort(key=lambda v: v["free_percent"])
    return {"volumes": volumes}


def power_mode() -> dict:
    """Whether the machine has been *asked* to be slow.

    A power profile is the one cause of slowness that is entirely
    intentional, which is why it is worth checking before anything is blamed
    on hardware. The platform detail lives in :mod:`pcbench.provenance`,
    which already covers all three; this only decides what counts as "set to
    go slow".
    """
    from . import provenance

    policy = provenance.frequency_policy()
    if not policy.get("available"):
        return {"error": policy.get("note") or "power policy is not exposed",
                "low_power_mode": None}

    platform_name = policy.get("platform")
    if platform_name == "Darwin":
        return {"platform": platform_name,
                "low_power_mode": policy.get("low_power_mode") == "on",
                "label": f"low power mode {policy.get('low_power_mode')}",
                "source": "pmset -g"}
    if platform_name == "Windows":
        scheme = (policy.get("power_scheme") or "").strip()
        lowered = scheme.lower()
        # "Power saver" is the throttling one; "Balanced" and "High
        # performance" are not, and neither is a vendor-named profile.
        return {"platform": platform_name,
                "low_power_mode": "power saver" in lowered
                or "energiesparmodus" in lowered,
                "label": scheme or None,
                "source": "powercfg /getactivescheme"}
    governor = policy.get("governor")
    return {"platform": platform_name or "Linux",
            "governor": governor,
            "low_power_mode": governor in ("powersave", "conservative"),
            "turbo": policy.get("turbo"),
            "min_mhz": policy.get("min_mhz"),
            "max_mhz": policy.get("max_mhz"),
            "label": f"governor {governor}" if governor else None,
            "source": "cpufreq"}


# --------------------------------------------------------------------------- #
# Gathering
# --------------------------------------------------------------------------- #
def _accelerators() -> dict:
    """GPU/NPU inventory plus whatever telemetry is reachable without a run."""
    from . import accel
    from . import gpucompute
    from . import system as system_mod

    out = dict(accel.inventory(system_mod.cpu_model()))
    out["packages"] = gpucompute.available()
    try:
        out["opencl_devices"] = gpucompute.devices()
    except Exception:                           # pragma: no cover - defensive
        out["opencl_devices"] = []
    try:
        out["nvidia"] = gpucompute.nvidia_telemetry()
    except Exception:                           # pragma: no cover - defensive
        out["nvidia"] = []
    return out


def gather(script_dir: str = ".") -> dict:
    """Every cheap source of evidence. Never raises; failures are recorded."""
    from . import container as container_mod
    from . import drivelife
    from . import provenance
    from . import system as system_mod

    evidence: dict = {}
    for name, source in (
            ("system", system_mod.inventory),
            ("state", lambda: system_mod.machine_state(script_dir)),
            ("processes", top_processes),
            ("memory", memory_pressure),
            ("disks", disk_headroom),
            ("power_mode", power_mode),
            ("provenance", provenance.collect),
            ("drives", lambda: drivelife.run(script_dir)),
            ("accelerators", _accelerators),
    ):
        try:
            evidence[name] = source()
        except Exception as e:                  # pragma: no cover - defensive
            evidence[name] = {"error": f"{type(e).__name__}: {e}"}
    evidence["uptime_seconds"] = uptime_seconds()
    info = evidence.get("system") or {}
    try:
        evidence["confinement"] = container_mod.detect(
            info.get("cpu_cores_logical"), info.get("ram_total_bytes", 0))
    except Exception:                           # pragma: no cover - defensive
        evidence["confinement"] = {}
    return evidence


def probe(seconds: float = 1.0, droop_seconds: float = 8.0,
          disk_dir: str | None = None) -> dict:
    """A short measurement, so findings can cite numbers and not just settings.

    Deliberately small. The point is not to score the machine — the benchmark
    does that — but to catch the two faults that only appear under load: a
    subsystem far below any reasonable floor, and throughput that collapses
    once the machine heats up.
    """
    import tempfile

    from . import scoring
    from . import sustained
    from . import workloads as wl

    out: dict = {}
    target = disk_dir or tempfile.gettempdir()
    try:
        out["cpu_int"] = wl.bench_cpu_integer(seconds, 2)
        out["memory"] = wl.bench_memory(seconds, 2, 32, 0)
        out["disk"] = wl.bench_disk(seconds, 1, 64, target)
    except Exception as e:                      # pragma: no cover - defensive
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        out["scores"] = scoring.compute_scores(
            {k: v for k, v in out.items() if isinstance(v, dict)})
    except Exception:                           # pragma: no cover - defensive
        out["scores"] = {}
    if droop_seconds > 0:
        try:
            out["sustained"] = sustained.run_sustained(droop_seconds,
                                                       window=2.0)
        except Exception as e:                  # pragma: no cover - defensive
            out["sustained"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
#: Words that mean the platform has started managing heat. macOS reports
#: nominal/fair/serious/critical; Linux and Windows phrase it differently,
#: so this matches on meaning rather than on an enumeration.
_THERMAL_STRAIN = ("serious", "critical", "heavy", "trapping", "sleeping")


def _check_thermal(ev: dict) -> list[dict]:
    state = ev.get("state") or {}
    found = []
    # The state string is a composed description ("nominal, max 52C"), not a
    # bare level, so this looks for the words that mean trouble rather than
    # for the absence of the word that means fine.
    thermal = (state.get("thermal") or "").lower()
    temp = state.get("cpu_celsius")
    if "throttl" in thermal:
        found.append(_finding(
            "thermal_throttled", "critical", "thermal",
            "The CPU is being thermally throttled right now",
            f"thermal pressure reports {state.get('thermal')!r}"
            + (f", CPU at {temp:.0f} °C" if temp else ""),
            "Clock speed is being cut to keep the chip within its limit, so "
            "everything is slower — often by a third or more — until it "
            "cools.",
            "Check that vents and fans are clear and the machine is on a hard "
            "surface. On a laptop, a dust-blocked fan or dried thermal paste "
            "is the usual cause; on a desktop, a stopped case fan."))
    elif any(word in thermal for word in _THERMAL_STRAIN):
        found.append(_finding(
            "thermal_strained", "warning", "thermal",
            "The platform reports thermal strain",
            f"thermal pressure reports {state.get('thermal')!r}",
            "Not throttling yet, but the system has started managing heat, "
            "and sustained work from here will be cut short.",
            "Clear the vents and check fan operation, then re-run."))
    elif isinstance(temp, (int, float)) and temp >= 90:
        found.append(_finding(
            "thermal_hot", "warning", "thermal",
            f"The CPU is running hot at {temp:.0f} °C",
            f"{temp:.0f} °C while idle or lightly loaded",
            "Not throttling yet, but there is no headroom left: any sustained "
            "work will start throttling almost immediately.",
            "Clear the vents and check fan operation. Run "
            "`pcbench --sustained 5m` to see how far throughput falls under "
            "load."))
    return found


def _check_power(ev: dict) -> list[dict]:
    state = ev.get("state") or {}
    mode = ev.get("power_mode") or {}
    found = []
    if state.get("on_ac_power") is False:
        found.append(_finding(
            "on_battery", "warning", "power",
            "Running on battery",
            "no AC power detected",
            "Most machines cut clock speed substantially on battery — a "
            "20-50% drop is normal and intended, not a fault.",
            "Plug in and re-check before investigating anything else."))
    if mode.get("low_power_mode"):
        label = mode.get("label") or mode.get("governor") or "a saving mode"
        found.append(_finding(
            "low_power_mode", "warning", "power",
            "The machine is set to a power-saving mode",
            f"{mode.get('source', 'power policy')} reports {label!r}",
            "This is the machine doing exactly what it was told: trading "
            "speed for battery life or heat. It is the one cause of "
            "slowness that is entirely intentional.",
            "Switch to a balanced or high-performance profile and re-check."
            + (" On Linux: `sudo cpupower frequency-set -g performance`."
               if mode.get("governor") else "")))
    if mode.get("turbo") is False:
        found.append(_finding(
            "turbo_disabled", "warning", "power",
            "Turbo / boost clocks are disabled",
            "cpufreq reports boost turned off",
            "The CPU is pinned to its base frequency, which on a modern chip "
            "is far below what it can reach for short bursts — exactly the "
            "bursts that make a machine feel responsive.",
            "Re-enable boost in firmware, or via "
            "`/sys/devices/system/cpu/cpufreq/boost`."))
    return found


def _check_contention(ev: dict) -> list[dict]:
    state = ev.get("state") or {}
    procs = ev.get("processes") or {}
    found = []
    load = state.get("load_per_core")
    if isinstance(load, (int, float)) and load >= LOAD_BUSY:
        severity = "critical" if load >= LOAD_SATURATED else "warning"
        found.append(_finding(
            "high_load", severity, "contention",
            f"The machine is already busy (load {load:.2f} per core)",
            f"load average per core = {load:.2f}",
            "Work is queuing for CPU time. Everything you start has to wait "
            "behind it, which is felt as general slowness rather than as any "
            "one slow application."
            if load >= LOAD_SATURATED else
            "A substantial share of the CPU is already committed before you "
            "ask it to do anything.",
            "See which processes below are responsible. If none look "
            "familiar, a runaway background task, an indexer, or an update "
            "service is the usual explanation."))
    for entry in (procs.get("processes") or [])[:3]:
        if entry.get("cpu_percent", 0) >= PROCESS_CPU_PERCENT:
            found.append(_finding(
                f"busy_process_{entry['pid']}", "warning", "contention",
                f"{entry['name']} is using {entry['cpu_percent']:.0f}% CPU",
                f"pid {entry['pid']}, {entry['cpu_percent']:.0f}% of a core"
                + (f", {entry['mem_percent']:.0f}% of RAM"
                   if entry.get("mem_percent") else ""),
                "One process holding this much CPU while you are not asking "
                "it to is the single most common cause of a machine that "
                "feels slow.",
                f"Look up what {entry['name']} is before killing it — "
                f"indexers, backup agents and update services do this "
                f"legitimately, and finish on their own."))
    return found


def _check_memory(ev: dict) -> list[dict]:
    mem = ev.get("memory") or {}
    found = []
    available = mem.get("available_percent")
    swap_percent = mem.get("swap_used_percent") or 0.0
    if isinstance(available, (int, float)):
        gigabytes = (mem.get("available_bytes") or 0) / (1024 ** 3)
        if available < RAM_CRITICAL_PERCENT:
            found.append(_finding(
                "memory_exhausted", "critical", "memory",
                f"Almost no memory left ({available:.0f}% free, "
                f"{gigabytes:.1f} GB)",
                f"{available:.0f}% of {mem['total_bytes'] / (1024 ** 3):.0f} "
                f"GB available, swap {swap_percent:.0f}% used",
                "Every new allocation now costs an eviction, and the machine "
                "spends its time moving pages instead of doing work. This "
                "feels far worse than a slow CPU ever does.",
                "Close what you are not using — browser tabs first, "
                "they are usually the largest consumer. If this is "
                "normal for your workload, the machine needs more RAM."))
        elif available < RAM_LOW_PERCENT:
            found.append(_finding(
                "memory_low", "warning", "memory",
                f"Memory is tight ({available:.0f}% free, "
                f"{gigabytes:.1f} GB)",
                f"{available:.0f}% available, swap {swap_percent:.0f}% used",
                "There is little room left for the filesystem cache, so reads "
                "that used to come from memory now go to disk.",
                "Close unused applications and re-check. Sustained pressure "
                "here is the clearest signal that more RAM would help."))
    return found


def _check_storage(ev: dict) -> list[dict]:
    found = []
    for volume in (ev.get("disks") or {}).get("volumes", []):
        free_pct = volume["free_percent"]
        free_gb = volume["free_bytes"] / (1024 ** 3)
        if free_pct >= DISK_LOW_PERCENT:
            continue
        severity = ("critical" if free_pct < DISK_CRITICAL_PERCENT
                    else "warning")
        found.append(_finding(
            f"disk_full_{volume['mount']}", severity, "storage",
            f"{volume['mount']} is {100 - free_pct:.0f}% full "
            f"({free_gb:.1f} GB free)",
            f"{free_gb:.1f} GB of "
            f"{volume['total_bytes'] / (1024 ** 3):.0f} GB free "
            f"({free_pct:.1f}%)",
            "A nearly-full filesystem has to work harder to find contiguous "
            "space for every write, and on an SSD it also loses the spare "
            "area the controller uses to spread wear. Nothing fails — "
            "writes just get slower, which is why this is so rarely "
            "suspected.",
            "Free space until at least 15% is available. This is one of the "
            "few causes where the fix is immediate and obvious."))

    drives = ev.get("drives") or {}
    try:
        from . import drivelife

        for note in drivelife.warnings(drives):
            lowered = note.lower()
            critical = any(word in lowered for word in
                           ("critical warning", "failing", "media/data",
                            "end of rated life"))
            found.append(_finding(
                "drive_health", "critical" if critical else "warning",
                "storage", "The drive reports a health problem", note,
                "SMART is the drive telling you about itself. A drive that "
                "has started returning errors or exhausting spare blocks will "
                "get slower as it retries, before it fails outright.",
                "Back up now. Replacement is the only fix for wear; there is "
                "no software remedy."))
    except Exception:                           # pragma: no cover - defensive
        pass
    return found


def _check_configuration(ev: dict) -> list[dict]:
    found = []
    info = ev.get("system") or {}
    confinement = ev.get("confinement") or {}
    uptime = ev.get("uptime_seconds")

    if isinstance(uptime, (int, float)) and uptime > LONG_UPTIME_DAYS * 86400:
        found.append(_finding(
            "long_uptime", "info", "software",
            f"Up for {uptime / 86400:.0f} days without a restart",
            f"uptime {uptime / 86400:.0f} days",
            "Not a fault. But leaked memory, stuck background tasks and "
            "pending updates all accumulate, and a restart is the cheapest "
            "thing to rule out before investigating anything harder.",
            "Restart, then re-run this check. If it fixes the problem you "
            "have saved yourself an investigation."))

    quota = confinement.get("cpu_quota_cores")
    if quota and info.get("cpu_cores_logical") and \
            quota < info["cpu_cores_logical"]:
        found.append(_finding(
            "cpu_quota", "info", "config",
            f"This process may use {quota:g} of "
            f"{info['cpu_cores_logical']} cores",
            f"cgroup quota limits it to {quota:g} cores",
            "The machine is not slow — this process is capped. Anything "
            "measured here reflects the limit, not the hardware.",
            "Raise the container's CPU limit, or measure on the host."))

    if info.get("virtualization"):
        found.append(_finding(
            "virtualized", "info", "config",
            f"Running inside {info['virtualization']}",
            f"virtualization detected: {info['virtualization']}",
            "Timing inside a VM includes the host's scheduling of it, so "
            "results here reflect the guest's share rather than the "
            "hardware.",
            "Compare against another guest on the same host, not against "
            "bare metal."))
    return found


def _check_measured(ev: dict) -> list[dict]:
    """Findings that need the probe: floors, and throughput droop."""
    measured = ev.get("probe") or {}
    if not measured:
        return []
    found = []
    from . import reference

    subscores = (measured.get("scores") or {}).get("subscores") or {}
    for check in reference.subsystem_checks(measured, subscores):
        if check.get("severity") == "expected":
            continue
        found.append(_finding(
            f"floor_{check['metric']}", "warning", "measured",
            f"{check['metric'].replace('_', ' ')} measured "
            f"{check['value']} {check['unit']}",
            f"below the {check['floor']} {check['unit']} floor",
            check["note"],
            "Re-run with `pcbench --drive-speed` or `--only memory` to "
            "confirm, and check the findings above first — a busy or hot "
            "machine produces exactly this."))

    droop = measured.get("sustained") or {}
    percent = droop.get("droop_percent")
    if isinstance(percent, (int, float)) and percent >= DROOP_WARN_PERCENT:
        severity = ("critical" if percent >= DROOP_CRITICAL_PERCENT
                    else "warning")
        found.append(_finding(
            "sustained_droop", severity, "thermal",
            f"Throughput fell {percent:.0f}% while under load",
            f"first window to last: {percent:.0f}% lost over "
            f"{droop.get('duration_s', 0):.0f}s",
            "The machine is fast when idle and slow when working, which is "
            "the signature of thermal or power limiting rather than of slow "
            "hardware. It is also why a short benchmark can look fine while "
            "real work feels slow.",
            "Check cooling first. If temperatures are fine, the power limit "
            "may be set low in firmware — common on thin laptops and "
            "small-form-factor desktops."))
    return found


#: A GPU this hot is throttling or about to; NVIDIA cards clock down from
#: the low 80s and hard-limit in the high 80s.
GPU_HOT_CELSIUS = 85.0

#: Below this, the machine is short of RAM for a modern desktop whatever
#: else is true, and every other finding is downstream of it.
LOW_RAM_GB = 6.0


def _check_cpu(ev: dict) -> list[dict]:
    """Settings that leave CPU performance on the table."""
    found = []
    prov = ev.get("provenance") or {}
    info = ev.get("system") or {}

    smt = prov.get("smt") or {}
    if (smt.get("available") and smt.get("supported")
            and not smt.get("enabled")):
        found.append(_finding(
            "smt_disabled", "info", "cpu",
            "Simultaneous multithreading is turned off",
            "the CPU supports SMT/Hyper-Threading but it is disabled",
            "Roughly a third of multithreaded throughput is unavailable. "
            "This is a deliberate choice on some servers, for security or "
            "for latency consistency, so it is only a problem if it was not "
            "intended.",
            "Re-enable it in firmware, or via "
            "`/sys/devices/system/cpu/smt/control`, if the machine is used "
            "for throughput rather than latency."))

    mitigations = prov.get("mitigations") or {}
    vulnerable = mitigations.get("vulnerable") or []
    mitigated = mitigations.get("mitigated") or []
    if mitigations.get("available") and len(mitigated) >= 4:
        found.append(_finding(
            "mitigations_heavy", "info", "cpu",
            f"{len(mitigated)} CPU vulnerability mitigations are active",
            f"mitigated: {', '.join(sorted(mitigated)[:5])}"
            + (" and others" if len(mitigated) > 5 else ""),
            "Mitigations cost real performance — most on syscall-heavy and "
            "context-switch-heavy work, which is where a machine feels slow. "
            "This is the correct default and turning them off trades "
            "security for speed.",
            "Nothing to do on a general-purpose machine. Only consider "
            "`mitigations=off` on an isolated, single-tenant host where the "
            "risk is understood."))
    elif vulnerable:
        found.append(_finding(
            "mitigations_missing", "info", "cpu",
            f"{len(vulnerable)} CPU vulnerability is unmitigated",
            f"vulnerable: {', '.join(sorted(vulnerable)[:5])}",
            "Not a performance problem — the opposite. Noted because it "
            "means this machine's numbers are not comparable with a "
            "mitigated one.",
            "No action for speed. Update the kernel and microcode if the "
            "exposure matters."))

    confinement = ev.get("confinement") or {}
    affinity = confinement.get("affinity_cores")
    total = info.get("cpu_cores_logical")
    if affinity and total and affinity < total:
        found.append(_finding(
            "affinity_restricted", "warning", "cpu",
            f"This process may only run on {affinity} of {total} cores",
            f"CPU affinity mask allows {affinity} core(s)",
            "Anything measured here uses a fraction of the machine. If the "
            "restriction is system-wide rather than inherited from a "
            "launcher, everything on it is similarly capped.",
            "Check for a taskset/start-affinity wrapper, or a scheduler "
            "policy applying it."))
    return found


def _check_gpu(ev: dict) -> list[dict]:
    """Accelerators present but idle, unusable, or too hot to hold clocks."""
    found = []
    accelerators = ev.get("accelerators") or {}
    if accelerators.get("error"):
        return found

    for card in accelerators.get("nvidia") or []:
        name = card.get("name", "the GPU")
        celsius = card.get("celsius")
        if isinstance(celsius, (int, float)) and celsius >= GPU_HOT_CELSIUS:
            found.append(_finding(
                f"gpu_hot_{card.get('index', 0)}", "warning", "gpu",
                f"{name} is at {celsius:.0f} °C",
                f"{celsius:.0f} °C"
                + (f", fan {card['fan_percent']:.0f}%"
                   if card.get("fan_percent") is not None else ""),
                "NVIDIA cards begin dropping clocks in the low 80s and hard-"
                "limit above that, so sustained GPU work is being cut short "
                "regardless of what the card is rated for.",
                "Clear intake dust and check case airflow. A card at this "
                "temperature while idle points at a stopped fan."))
        total = card.get("vram_total_mb")
        used = card.get("vram_used_mb")
        if (isinstance(total, (int, float)) and isinstance(used, (int, float))
                and total > 0 and used / total >= 0.92):
            found.append(_finding(
                f"vram_full_{card.get('index', 0)}", "warning", "gpu",
                f"{name} has {total - used:.0f} MB of VRAM free "
                f"of {total:.0f} MB",
                f"{used / total * 100:.0f}% of VRAM in use",
                "Once VRAM is exhausted the driver spills to system memory "
                "over the bus, which is an order of magnitude slower — the "
                "usual explanation for a model or a game that suddenly runs "
                "at a fraction of its normal speed.",
                "Close other GPU consumers, or reduce batch size, texture "
                "quality, or resolution."))

    gpus = accelerators.get("gpus") or []
    packages = accelerators.get("packages") or {}
    if (len(gpus) > 1 and not accelerators.get("opencl_devices")
            and not packages.get("pyopencl")):
        found.append(_finding(
            "gpu_unverified", "info", "gpu",
            f"{len(gpus)} GPUs are present but none could be tested",
            "no OpenCL runtime is reachable (pyopencl is not installed)",
            "Whether the discrete GPU is actually being used cannot be "
            "confirmed from here. On a laptop with switchable graphics, work "
            "silently running on the integrated GPU is a common and "
            "invisible cause of poor performance.",
            "Install the gpu tier (`pip install -e '.[gpu]'`) and re-run, or "
            "check the vendor control panel for which GPU applications are "
            "assigned to."))
    return found


def _check_ram_config(ev: dict) -> list[dict]:
    """How much RAM there is, as opposed to how much is free right now."""
    found = []
    info = ev.get("system") or {}
    confinement = ev.get("confinement") or {}
    total = info.get("ram_total_bytes") or 0
    gigabytes = total / (1024 ** 3)
    if 0 < gigabytes < LOW_RAM_GB:
        found.append(_finding(
            "ram_small", "warning", "memory",
            f"The machine has {gigabytes:.1f} GB of RAM",
            f"{gigabytes:.1f} GB total",
            "Below roughly 6 GB a modern desktop and a browser will page "
            "constantly no matter what else is fixed, and every other "
            "finding here is downstream of that.",
            "More RAM is the only real fix. Until then, fewer browser tabs "
            "and a lighter desktop help more than anything else."))

    limit = confinement.get("effective_ram_bytes")
    if limit and total and limit < total * 0.9:
        found.append(_finding(
            "ram_capped", "info", "memory",
            f"This process may use {limit / (1024 ** 3):.1f} GB of "
            f"{gigabytes:.1f} GB",
            "a cgroup or container memory limit is in force",
            "The machine is not short of memory — this process is capped, "
            "and will be killed rather than slowed if it exceeds the limit.",
            "Raise the container's memory limit if the workload needs it."))
    return found


#: How much slower than this machine's own history counts as a change worth
#: reporting. Wide, because the probe is short and a single past run is not a
#: distribution: below this, run-to-run noise dominates.
REGRESSION_PERCENT = 25.0

#: Metrics worth comparing against history, as (probe path, CSV column,
#: label).
#:
#: Only CPU. The checkup's probe is deliberately small — a 64 MB disk file
#: and a 32 MB memory buffer against the 256/64 MB a real run uses — and
#: disk and memory throughput both depend heavily on those sizes. Comparing
#: them against recorded full runs reported a 74% disk regression on a
#: perfectly healthy machine, which is exactly the kind of confident
#: nonsense this module must not produce. A time-boxed rate like primes per
#: second does not depend on how long it ran, so it compares honestly.
_HISTORY_METRICS = (
    (("cpu_int", "rate"), "cpu_int_primes_s", "single-core CPU"),
)


def history_rows(output_dir: str = "results") -> list[dict]:
    """Past runs recorded for *this* machine, oldest first."""
    from . import compare

    try:
        import socket

        host = socket.gethostname()
    except Exception:                           # pragma: no cover - defensive
        return []
    rows = compare.load_history(os.path.join(output_dir, "benchmarks.csv"))
    mine = [r for r in rows if r.get("hostname") == host]
    return sorted(mine, key=lambda r: r.get("timestamp_utc", ""))


def _check_regression(ev: dict) -> list[dict]:
    """Is this machine slower than it used to be?

    "It used to be fast" is the actual complaint most of the time, and it is
    the one question a single snapshot cannot answer. Where this tool has run
    before, it can: the same probe against the same machine's own record is a
    far stronger signal than any absolute threshold, because it controls for
    the hardware entirely.
    """
    measured = ev.get("probe") or {}
    rows = ev.get("history") or []
    if not measured or len(rows) < 1:
        return []
    found = []
    for (section, field), column, label in _HISTORY_METRICS:
        entry = measured.get(section)
        if not isinstance(entry, dict):
            continue
        now = entry.get(field)
        past = [float(r[column]) for r in rows
                if r.get(column) and _is_number(r[column])
                and float(r[column]) > 0]
        if not isinstance(now, (int, float)) or now <= 0 or not past:
            continue
        best = max(past)
        drop = (best - now) / best * 100
        if drop < REGRESSION_PERCENT:
            continue
        found.append(_finding(
            f"regressed_{column}", "warning", "measured",
            f"{label} is {drop:.0f}% slower than this machine's best",
            f"{now:,.0f} now against {best:,.0f} recorded on "
            f"{_when(rows, column, best)} ({len(past)} past run(s))",
            "The hardware has not changed, so something else has. This is "
            "the strongest signal available that a machine genuinely got "
            "slower rather than always having been this speed.",
            "Check the findings above first — heat, power mode and "
            "background load all produce exactly this. If none apply, "
            "compare the full runs with "
            "`pcbench --compare-runs old.json new.json`."))
    return found


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _when(rows: list[dict], column: str, value: float) -> str:
    for row in rows:
        if row.get(column) and _is_number(row[column]) \
                and abs(float(row[column]) - value) < 1e-6:
            return (row.get("timestamp_utc") or "an earlier run")[:10]
    return "an earlier run"


_CHECKS = (_check_thermal, _check_power, _check_cpu, _check_gpu,
           _check_contention, _check_memory, _check_ram_config,
           _check_storage, _check_configuration, _check_measured,
           _check_regression)


def analyse(evidence: dict) -> list[dict]:
    """Every finding the evidence supports, most serious first."""
    found: list[dict] = []
    for check in _CHECKS:
        try:
            found.extend(check(evidence))
        except Exception as e:                  # pragma: no cover - defensive
            found.append(_finding(
                f"check_failed_{check.__name__}", "info", "config",
                "A check could not complete", f"{type(e).__name__}: {e}",
                "This check produced no answer, so treat its area as "
                "unexamined rather than clean.", "Report this as a bug."))
    found.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))
    return found


# --------------------------------------------------------------------------- #
# Running and rendering
# --------------------------------------------------------------------------- #
def run(script_dir: str = ".", measure: bool = True,
        disk_dir: str | None = None, output_dir: str = "results") -> dict:
    """Gather evidence, optionally measure, and rank what was found."""
    started = time.perf_counter()
    evidence = gather(script_dir)
    evidence["history"] = history_rows(output_dir)
    if measure:
        evidence["probe"] = probe(disk_dir=disk_dir)
    findings = analyse(evidence)
    return {
        "findings": findings,
        "evidence": evidence,
        "measured": bool(measure),
        "counts": {
            level: sum(1 for f in findings if f["severity"] == level)
            for level in ("critical", "warning", "info")
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def verdict(result: dict) -> str:
    """One sentence: the most likely explanation, or that there isn't one."""
    counts = result.get("counts") or {}
    findings = result.get("findings") or []
    actionable = [f for f in findings if f["severity"] != "info"]
    if not actionable:
        if not result.get("measured"):
            return ("Nothing found in the settings and live state that would "
                    "explain a slowdown. Re-run without --no-measure to also "
                    "check throughput and throttling.")
        return ("Nothing found that would explain a slowdown: no throttling, "
                "no contention, memory and disk headroom are fine, and the "
                "short measurement came out where it should.")
    first = actionable[0]
    tail = ""
    if len(actionable) > 1:
        tail = (f" {len(actionable) - 1} other finding(s) may be "
                f"contributing.")
    lead = ("The most likely cause is" if counts.get("critical")
            else "The most likely contributor is")
    return f"{lead}: {first['title'].lower()}.{tail}"


def _context_block(evidence: dict) -> list[str]:
    """The numbers behind the findings, shown whether or not any fired."""
    lines = []
    state = evidence.get("state") or {}
    memory = evidence.get("memory") or {}
    uptime = evidence.get("uptime_seconds")

    temp = state.get("cpu_celsius")
    bits = []
    if isinstance(temp, (int, float)):
        bits.append(f"{temp:.0f} °C")
    if state.get("thermal"):
        bits.append(str(state["thermal"]))
    if state.get("on_ac_power") is not None:
        bits.append("on AC" if state["on_ac_power"] else "on battery")
    if isinstance(state.get("load_per_core"), (int, float)):
        bits.append(f"load {state['load_per_core']:.2f}/core")
    if bits:
        lines.append(f"  State      : {', '.join(bits)}")

    if memory.get("available_percent") is not None:
        lines.append(
            f"  Memory     : {memory['available_bytes'] / (1024 ** 3):.1f} GB "
            f"of {memory['total_bytes'] / (1024 ** 3):.0f} GB available "
            f"({memory['available_percent']:.0f}%), swap "
            f"{memory.get('swap_used_percent', 0):.0f}% used")

    for volume in (evidence.get("disks") or {}).get("volumes", [])[:3]:
        lines.append(
            f"  Disk       : {volume['mount']} — "
            f"{volume['free_bytes'] / (1024 ** 3):.1f} GB free of "
            f"{volume['total_bytes'] / (1024 ** 3):.0f} GB "
            f"({volume['free_percent']:.0f}%)")

    if isinstance(uptime, (int, float)):
        days, hours = divmod(uptime / 3600, 24)
        lines.append(f"  Uptime     : {days:.0f}d {hours:.0f}h")

    processes = (evidence.get("processes") or {}).get("processes") or []
    if processes:
        named = ", ".join(f"{p['name']} {p['cpu_percent']:.0f}%"
                          for p in processes[:3])
        lines.append(f"  Busiest    : {named}")

    measured = evidence.get("probe") or {}
    disk = measured.get("disk") if isinstance(measured.get("disk"),
                                              dict) else {}
    if disk.get("read_rate"):
        lines.append(f"  Measured   : disk {disk['read_rate']:.0f} MB/s read, "
                     f"{disk.get('write_rate', 0):.0f} MB/s write")
    droop = measured.get("sustained") or {}
    if isinstance(droop.get("droop_percent"), (int, float)):
        lines.append(f"  Under load : throughput fell "
                     f"{droop['droop_percent']:.0f}% over "
                     f"{droop.get('duration_s', 0):.0f}s")
    return lines


def render(result: dict) -> str:
    """The report: a verdict, then each finding with evidence and a remedy."""
    findings = result.get("findings") or []
    counts = result.get("counts") or {}
    lines = [f"  {chunk}" for chunk in _wrap(verdict(result), 70)]
    lines.append("")

    if findings:
        summary = ", ".join(
            f"{counts[level]} {level}" for level in
            ("critical", "warning", "info") if counts.get(level))
        lines.append(f"  {len(findings)} finding(s): {summary}")
        lines.append("")
        for number, finding in enumerate(findings, 1):
            lines.append(f"  [{number:>2}] {finding['severity'].upper():<8} "
                         f"{finding['title']}")
            lines.append(f"      evidence : {finding['evidence']}")
            for label, key in (("impact", "impact"), ("fix", "fix")):
                for index, chunk in enumerate(_wrap(finding[key], 62)):
                    prefix = f"      {label:<9}: " if index == 0 else " " * 17
                    lines.append(prefix + chunk)
            lines.append("")

    context = _context_block(result.get("evidence") or {})
    if context:
        lines.append("  ── what this was based on " + "─" * 46)
        lines += context
    if not result.get("measured"):
        lines.append("")
        lines.append("  Throughput and throttling were not measured "
                     "(--no-measure).")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]
