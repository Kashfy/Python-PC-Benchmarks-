"""Hardware performance counters — the "why" behind a throughput number.

Every other measurement in this tool reports *how fast*. None of them report
*what limited it*, and without that a slow result is a dead end. Counters turn
the same run into a diagnosis, because the three common causes leave completely
different fingerprints:

===================  ==============================================
Symptom              Cause
===================  ==============================================
Low score, high IPC  The cores are executing efficiently and there
                     simply are not enough cycles — a clock or
                     power limit, not a code or memory problem.
Low IPC, high cache   The working set does not fit. Adding cores
miss rate            will not help; the memory system is the wall.
Low IPC, high branch  The workload is unpredictable to the branch
miss rate            predictor — a code-shape problem.
===================  ==============================================

**Two tiers, because privilege differs enormously.**

*Resource counters* (page faults, context switches, block I/O) come from
``getrusage`` and cost nothing, need no privileges, and work on every Unix. They
already answer a lot: involuntary context switches mean CPU contention from
other processes, and major page faults mean the machine is paging, which no
amount of CPU analysis would have revealed.

*PMU counters* (cycles, instructions, cache misses, branch misses) are real
hardware registers and need kernel cooperation. On Linux they come from
``perf``. On macOS the equivalent interface is a private framework requiring
root and an Apple entitlement, and on Windows most counters require a driver —
so on those platforms this module reports honestly that they are unavailable
rather than substituting something weaker and calling it the same thing.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time

# `resource` is POSIX-only and does not exist on Windows. Importing it at
# module scope made `import pcbench.cli` raise ModuleNotFoundError there, which
# stopped the whole tool from starting -- not merely this section. Anything
# platform-specific has to be imported where it is used, or guarded like this.
try:
    import resource
except ImportError:                                     # Windows
    resource = None

#: perf events requested. Names rather than raw codes so perf maps them to
#: whatever the local microarchitecture actually calls them.
PERF_EVENTS = [
    "cycles", "instructions", "cache-references", "cache-misses",
    "branches", "branch-misses",
]


# --------------------------------------------------------------------------- #
# Tier 1: resource counters (always available)
# --------------------------------------------------------------------------- #
def resource_snapshot(children: bool = False) -> dict:
    """Cumulative resource counters for this process (or its children).

    The wall clock is captured alongside them because every counter here is
    cumulative: a raw count is meaningless without the interval it accumulated
    over, and comparing counts across runs of different lengths is how a
    perfectly healthy long run gets flagged as contended.
    """
    if resource is None:
        return _windows_snapshot(children)

    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    try:
        r = resource.getrusage(who)
    except (OSError, ValueError):
        return {}
    return {
        "wall_clock": time.monotonic(),
        "user_s": r.ru_utime,
        "system_s": r.ru_stime,
        "max_rss_bytes": _rss_bytes(r.ru_maxrss),
        "minor_faults": r.ru_minflt,
        "major_faults": r.ru_majflt,
        "block_input": r.ru_inblock,
        "block_output": r.ru_oublock,
        "voluntary_switches": r.ru_nvcsw,
        "involuntary_switches": r.ru_nivcsw,
    }


def _windows_snapshot(children: bool = False) -> dict:
    """The same counters on Windows, via psutil.

    Windows has no getrusage. psutil exposes the equivalents through
    GetProcessMemoryInfo and GetProcessTimes, so the section works wherever
    psutil is installed and reports itself unavailable where it is not --
    rather than taking the whole tool down, which is what the unguarded import
    used to do.

    Child totals are not aggregated: Windows does not maintain them the way
    RUSAGE_CHILDREN does, and summing live children would silently miss any
    that had already exited.
    """
    if children:
        return {"wall_clock": time.monotonic(),
                "note": "Windows does not accumulate counters for exited "
                        "children"}
    try:
        import psutil
    except ImportError:
        return {"wall_clock": time.monotonic(),
                "note": "resource counters need psutil on Windows "
                        "(pip install psutil)"}
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        switches = proc.num_ctx_switches()
        times = proc.cpu_times()
        return {
            "wall_clock": time.monotonic(),
            "user_s": times.user,
            "system_s": times.system,
            "max_rss_bytes": getattr(mem, "peak_wset", mem.rss),
            # Windows reports one page-fault total rather than splitting minor
            # from major, so it is recorded as minor and major is left absent
            # instead of being invented.
            "minor_faults": getattr(mem, "num_page_faults", 0),
            "block_input": 0,
            "block_output": 0,
            "voluntary_switches": switches.voluntary,
            "involuntary_switches": switches.involuntary,
        }
    except Exception as e:
        return {"wall_clock": time.monotonic(),
                "note": f"could not read Windows counters: {e}"}


def _rss_bytes(maxrss: int) -> int:
    """``ru_maxrss`` is kilobytes on Linux and bytes on macOS/BSD."""
    return maxrss if platform.system() == "Darwin" else maxrss * 1024


def resource_delta(before: dict, after: dict) -> dict:
    """Difference between two snapshots, with peak RSS taken as a maximum."""
    if not before or not after:
        return {}
    out = {}
    for key, value in after.items():
        if key == "wall_clock":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            # Carries a note rather than a counter (Windows without psutil).
            out[key] = value
            continue
        if key == "max_rss_bytes":
            out[key] = max(value, before.get(key, 0) or 0)
        else:
            out[key] = value - (before.get(key, 0) or 0)

    elapsed = after.get("wall_clock", 0) - before.get("wall_clock", 0)
    if elapsed > 0:
        out["elapsed_s"] = round(elapsed, 2)
        if isinstance(out.get("involuntary_switches"), (int, float)):
            out["involuntary_switches_per_s"] = round(
                out["involuntary_switches"] / elapsed, 1)
        # Absent on Windows, where the OS reports a single page-fault total.
        if isinstance(out.get("major_faults"), (int, float)):
            out["major_faults_per_s"] = round(out["major_faults"] / elapsed, 2)
    return out


# --------------------------------------------------------------------------- #
# Tier 2: PMU counters via perf
# --------------------------------------------------------------------------- #
def perf_available() -> dict:
    """Whether PMU counters can be read here, and precisely why not if they cannot.

    The "why not" is the useful part: every failure mode has a specific,
    one-line fix, and a bare "unavailable" would send the user looking for a
    hardware problem that does not exist.
    """
    system = platform.system()
    if system != "Linux":
        return {"available": False, "reason": (
            f"PMU counters are not accessible on {system}. macOS exposes them "
            f"only through a private framework requiring root and an Apple "
            f"entitlement; Windows requires a kernel driver. Resource counters "
            f"are still collected.")}

    if not shutil.which("perf"):
        return {"available": False, "reason": (
            "the 'perf' tool is not installed"), "fix": (
            "install it: 'apt install linux-tools-common linux-tools-$(uname "
            "-r)' on Debian/Ubuntu, 'dnf install perf' on Fedora/RHEL")}

    paranoid = _paranoid_level()
    if paranoid is not None and paranoid > 2:
        return {"available": False,
                "reason": (f"kernel.perf_event_paranoid is {paranoid}, which "
                           f"blocks user-space measurement"),
                "fix": ("sudo sysctl -w kernel.perf_event_paranoid=2 "
                        "(2 allows a process to measure itself)")}

    # A trial run is the only reliable test: containers frequently drop
    # CAP_PERFMON, and virtual machines commonly expose no PMU at all, neither
    # of which shows up in perf_event_paranoid.
    probe = _run_perf(["true"], ["instructions"], timeout=10)
    if probe.get("error"):
        return {"available": False,
                "reason": probe["error"],
                "fix": ("PMU access is commonly unavailable inside containers "
                        "and VMs; run on the host, or grant CAP_PERFMON")}
    return {"available": True, "paranoid": paranoid}


def _paranoid_level() -> int | None:
    try:
        with open("/proc/sys/kernel/perf_event_paranoid", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


_PERF_LINE = re.compile(
    r"^\s*([\d,.]+|<not supported>|<not counted>)\s+([A-Za-z0-9_\-.:]+)")


def parse_perf_stat(text: str) -> dict:
    """Parse ``perf stat`` stderr into ``{event: count}``.

    Unsupported and uncounted events are recorded as None rather than dropped:
    "this core has no cache-miss counter" is a fact worth reporting, and is
    different from "the event was zero".
    """
    counts: dict[str, int | None] = {}
    for line in text.splitlines():
        m = _PERF_LINE.match(line)
        if not m:
            continue
        raw, event = m.group(1), m.group(2)
        if raw.startswith("<"):
            counts[event] = None
            continue
        try:
            counts[event] = int(raw.replace(",", "").replace(".", ""))
        except ValueError:
            counts[event] = None
    return counts


def _run_perf(command: list[str], events: list[str],
              timeout: float = 120.0) -> dict:
    cmd = ["perf", "stat", "-x", " ", "-e", ",".join(events), "--"] + command
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "perf stat exceeded its timeout"}
    except OSError as e:
        return {"error": f"could not run perf: {e}"}

    # -x uses "count<sep>unit<sep>event"; both formats are parsed so a perf
    # that ignores -x still yields data.
    counts: dict[str, int | None] = {}
    for line in (proc.stderr or "").splitlines():
        parts = line.split(" ")
        if len(parts) >= 3 and parts[0] and not parts[0].startswith("#"):
            raw, event = parts[0], parts[2]
            if raw.startswith("<"):
                counts[event] = None
            else:
                try:
                    counts[event] = int(float(raw))
                except ValueError:
                    pass
    if not counts:
        counts = parse_perf_stat(proc.stderr or "")
    if not counts:
        detail = (proc.stderr or "").strip().splitlines()
        return {"error": ("perf produced no counter data"
                          + (f": {detail[-1]}" if detail else ""))}
    return {"counts": counts}


def measure_command(command: list[str], events: list[str] | None = None,
                    timeout: float = 300.0) -> dict:
    """Run a command under perf and return derived rates."""
    status = perf_available()
    if not status.get("available"):
        return {"skipped": True, **status}
    result = _run_perf(command, events or PERF_EVENTS, timeout)
    if result.get("error"):
        return {"skipped": True, "reason": result["error"]}
    return derive(result["counts"])


def derive(counts: dict) -> dict:
    """Turn raw counter values into the ratios that actually mean something.

    Absolute counts are close to useless for comparison — a longer run has more
    of everything. The ratios (IPC, misses per thousand instructions) are
    scale-free and comparable across machines and run lengths.
    """
    def get(name: str) -> int | None:
        value = counts.get(name)
        return value if isinstance(value, int) else None

    cycles = get("cycles")
    instructions = get("instructions")
    cache_refs = get("cache-references")
    cache_misses = get("cache-misses")
    branches = get("branches")
    branch_misses = get("branch-misses")

    out: dict = {"raw": counts}
    if cycles and instructions:
        out["ipc"] = round(instructions / cycles, 3)
    if cache_misses is not None and instructions:
        out["cache_misses_per_kilo_instruction"] = round(
            1000.0 * cache_misses / instructions, 2)
    if cache_misses is not None and cache_refs:
        out["cache_miss_rate_pct"] = round(100.0 * cache_misses / cache_refs, 2)
    if branch_misses is not None and branches:
        out["branch_miss_rate_pct"] = round(
            100.0 * branch_misses / branches, 2)
    if branch_misses is not None and instructions:
        out["branch_misses_per_kilo_instruction"] = round(
            1000.0 * branch_misses / instructions, 2)
    return out


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
# Involuntary context switches are reported as data but are deliberately NOT
# used to infer CPU contention, because for this workload they do not measure
# it. Measured per test on one machine in a single sitting, so ambient load
# applies equally to every row:
#
#     disk           132,152 /s   <- 94% of the switches in that sample
#     latency         83,027 /s
#     cache_sweep         571 /s
#     sqlite              445 /s
#     nn_training         422 /s
#     memory              356 /s
#     cpu_int             121 /s
#     cpu_multi            71 /s
#
# Two tests produce the overwhelming majority, and both do so by construction.
# Skipping them takes a full run from 8,812/s to 985/s -- an 89% drop, which is
# the falsification test to repeat on any machine. The `disk`
# test issues hundreds of thousands of blocking pread() calls to measure random
# read IOPS, and every one that blocks is a preemption. The `latency` suite
# *is* a context-switch benchmark -- producing context switches is its
# measurement. A whole-run figure is therefore a restatement of how many I/O
# operations the disk test completed, which is already reported as IOPS.
#
# Note what is not on that list: spawning. `cpu_multi` spawns a worker per core
# and sits near the bottom, because the parent process being measured spends
# the test waiting. An earlier version of this comment blamed worker spawning;
# that was wrong, and the per-test numbers above are what disproved it.
#
# Real contention detection already exists and is grounded properly: load
# average against core count in `system.state_warnings`, and per-test condition
# sampling in `interference`.

#: Major page faults per second. Unlike context switches this is unambiguous —
#: the machine went to backing store for memory, and whatever caused it, every
#: figure measured during it is affected.
HIGH_MAJOR_FAULTS_PER_S = 5.0

#: IPC below this suggests the core is stalling rather than executing. Modern
#: out-of-order cores retire 4-8 instructions per cycle at peak, and
#: well-behaved compute code reaches 2-3.
LOW_IPC = 1.0
GOOD_IPC = 2.0

#: Cache misses per thousand instructions. Above ~20 the workload is missing
#: often enough that memory latency dominates its runtime.
HIGH_CACHE_MPKI = 20.0

#: Branch mispredictions as a share of branches. Above ~5% the predictor is
#: failing often enough to cost meaningful frontend throughput.
HIGH_BRANCH_MISS_PCT = 5.0


def interpret(derived: dict, resources: dict | None = None) -> list[str]:
    """Plain-language conclusions, naming the limiting factor where possible."""
    notes: list[str] = []
    ipc = derived.get("ipc")
    mpki = derived.get("cache_misses_per_kilo_instruction")
    branch_pct = derived.get("branch_miss_rate_pct")

    if ipc is not None:
        if ipc < LOW_IPC:
            causes = []
            if mpki is not None and mpki > HIGH_CACHE_MPKI:
                causes.append(f"cache misses ({mpki:.0f} per 1000 "
                              f"instructions)")
            if branch_pct is not None and branch_pct > HIGH_BRANCH_MISS_PCT:
                causes.append(f"branch mispredictions ({branch_pct:.1f}%)")
            if causes:
                notes.append(
                    f"IPC is {ipc:.2f} — the cores are stalling, and the "
                    f"counters point at {' and '.join(causes)}. More cores or "
                    f"a higher clock will not help; the working set or the "
                    f"code shape is the limit.")
            else:
                notes.append(
                    f"IPC is {ipc:.2f} — the cores are stalling for a reason "
                    f"the collected counters do not explain. Common causes "
                    f"are memory latency, lock contention, and dependency "
                    f"chains.")
        elif ipc >= GOOD_IPC:
            notes.append(
                f"IPC is {ipc:.2f} — the cores are executing efficiently. A "
                f"low score with efficient execution points at clock speed or "
                f"a power limit rather than at memory or code.")
    if mpki is not None and mpki > HIGH_CACHE_MPKI and (ipc or 0) >= LOW_IPC:
        notes.append(
            f"{mpki:.0f} cache misses per 1000 instructions is high but IPC is "
            f"holding up — the machine is absorbing the misses, likely through "
            f"memory-level parallelism and prefetching.")

    res = resources or {}
    fault_rate = res.get("major_faults_per_s")
    if isinstance(fault_rate, (int, float)) and fault_rate > HIGH_MAJOR_FAULTS_PER_S:
        notes.append(
            f"{res.get('major_faults', 0):,} major page faults "
            f"({fault_rate:,.1f}/s) — memory was fetched from backing store "
            f"during the run, which dominates any CPU effect. On a machine "
            f"with little free RAM this is the benchmark's own footprint; "
            f"either way the figures measured during it are affected.")
    return notes


def render(result: dict | None) -> str:
    """Terminal block for the counters section."""
    if not result:
        return ""
    lines: list[str] = []
    pmu = result.get("pmu") or {}
    if pmu.get("skipped"):
        lines.append(f"  PMU counters   : unavailable — {pmu.get('reason', '')}")
        if pmu.get("fix"):
            lines.append(f"                   fix: {pmu['fix']}")
    else:
        if pmu.get("ipc") is not None:
            lines.append(f"  IPC                       : {pmu['ipc']:.3f} "
                         f"instructions per cycle")
        for key, label, unit in (
                ("cache_miss_rate_pct", "Cache miss rate", "%"),
                ("cache_misses_per_kilo_instruction", "Cache misses / 1k instr", ""),
                ("branch_miss_rate_pct", "Branch miss rate", "%"),
                ("branch_misses_per_kilo_instruction", "Branch misses / 1k instr", "")):
            if pmu.get(key) is not None:
                lines.append(f"  {label:<26}: {pmu[key]}{unit}")

    res = result.get("resources") or {}
    if res.get("note") and "involuntary_switches" not in res:
        lines.append(f"  Resource counters         : {res['note']}")
        res = {}
    if res:
        rate = res.get("involuntary_switches_per_s")
        suffix = (f"  ({rate:,.0f}/s over {res.get('elapsed_s', 0):,.0f}s)"
                  if rate is not None else "")
        lines.append(f"  Involuntary switches      : "
                     f"{res.get('involuntary_switches', 0):,}{suffix}")
        lines.append("      (mostly from the disk test's blocking reads and "
                     "the latency test, which measures context switches by "
                     "making them; not a contention signal)")
        if isinstance(res.get("major_faults"), (int, float)):
            lines.append(f"  Page faults (minor/major) : "
                         f"{res.get('minor_faults', 0):,} / "
                         f"{res['major_faults']:,}")
        elif isinstance(res.get("minor_faults"), (int, float)):
            lines.append(f"  Page faults               : "
                         f"{res['minor_faults']:,}  (Windows reports one "
                         f"total, not minor/major)")
        if res.get("max_rss_bytes"):
            lines.append(f"  Peak resident memory      : "
                         f"{res['max_rss_bytes'] / (1024 ** 2):,.0f} MB")

    for note in result.get("notes", []):
        lines.append(f"      i {note}")
    return "\n".join(lines)
