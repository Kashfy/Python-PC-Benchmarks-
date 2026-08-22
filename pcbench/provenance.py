"""System configuration that silently changes every number.

Two machines with identical hardware routinely benchmark 5-30% apart, and the
reason is almost never the hardware. It is configuration that nothing in a
normal report reveals:

* **Speculative-execution mitigations.** Spectre, Meltdown, MDS, Retbleed and
  their relatives are fixed in software at a real cost — 5% on ordinary work,
  30% or more on syscall-heavy and context-switch-heavy workloads. A machine
  with mitigations off looks like faster hardware and is not.
* **The CPU frequency governor.** ``powersave`` versus ``performance`` is
  frequently the entire explanation for a laptop that scores badly, and it is
  a one-line fix rather than a hardware fault.
* **Transparent hugepages.** ``always`` versus ``never`` moves memory-bound and
  database workloads by double digits in either direction depending on the
  access pattern.
* **SMT / Hyper-Threading, turbo state, and microcode revision.** Each changes
  results substantially, and microcode updates have historically *reduced*
  performance to fix errata.

None of this is measurable — it has to be read. Every field here is a plain
file or command read, and the whole module is advisory: it never changes what
runs, only what the report can explain. That distinction matters, because a
benchmark that silently "optimised" the machine first would be measuring a
configuration the user does not actually run.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess

_TIMEOUT = 5


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=_TIMEOUT)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------- #
# Speculative-execution mitigations
# --------------------------------------------------------------------------- #
def mitigations() -> dict:
    """Per-vulnerability mitigation status, and whether any are disabled.

    Linux exposes one file per vulnerability under
    ``/sys/devices/system/cpu/vulnerabilities/``. The text is free-form, but
    the three states that matter are distinguishable: "Not affected",
    "Vulnerable" (mitigation off), and anything else (mitigated, at a cost).
    """
    base = "/sys/devices/system/cpu/vulnerabilities"
    if platform.system() != "Linux" or not os.path.isdir(base):
        return {"available": False}

    entries: dict[str, str] = {}
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return {"available": False}

    for name in names:
        text = _read(os.path.join(base, name))
        if text:
            entries[name] = text

    vulnerable = [k for k, v in entries.items()
                  if v.lower().startswith("vulnerable")]
    mitigated = [k for k, v in entries.items()
                 if not v.lower().startswith(("vulnerable", "not affected"))]

    return {
        "available": True,
        "status": entries,
        "vulnerable": vulnerable,
        "mitigated": mitigated,
        "not_affected": [k for k, v in entries.items()
                         if v.lower().startswith("not affected")],
        # The kernel command line is the usual place mitigations get turned
        # off, and it explains a machine that is inexplicably fast.
        "cmdline_override": _mitigation_cmdline(),
    }


def _mitigation_cmdline() -> str | None:
    cmdline = _read("/proc/cmdline")
    hits = [token for token in cmdline.split()
            if token.startswith(("mitigations=", "nospectre", "nopti",
                                 "spectre_v2=", "spec_store_bypass_disable=",
                                 "kpti=", "no_stf_barrier", "tsx="))]
    return " ".join(hits) if hits else None


# --------------------------------------------------------------------------- #
# CPU frequency policy
# --------------------------------------------------------------------------- #
def frequency_policy() -> dict:
    """Governor, driver, frequency limits, and boost state."""
    system = platform.system()
    if system == "Linux":
        return _frequency_linux()
    if system == "Darwin":
        # Apple silicon has no user-selectable governor; the OS manages
        # residency itself. Low-power mode is the one user-facing control.
        low_power = _run(["pmset", "-g"])
        state = None
        m = re.search(r"lowpowermode\s+(\d)", low_power, re.I)
        if m:
            state = "on" if m.group(1) == "1" else "off"
        return {"available": state is not None, "platform": "Darwin",
                "low_power_mode": state,
                "note": "macOS manages CPU frequency itself; there is no "
                        "user-selectable governor"}
    if system == "Windows":
        scheme = _run(["powercfg", "/getactivescheme"])
        return {"available": bool(scheme), "platform": "Windows",
                "power_scheme": scheme or None}
    return {"available": False}


def _frequency_linux() -> dict:
    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    if not os.path.isdir(base):
        return {"available": False,
                "note": "cpufreq is not exposed (common in VMs and containers)"}

    governor = _read(os.path.join(base, "scaling_governor")) or None
    driver = _read(os.path.join(base, "scaling_driver")) or None

    # Governors may differ per core on hybrid parts, which is itself worth
    # knowing — a report that shows only cpu0 would hide it.
    governors = set()
    try:
        for name in os.listdir("/sys/devices/system/cpu"):
            if re.fullmatch(r"cpu\d+", name):
                g = _read(f"/sys/devices/system/cpu/{name}/cpufreq/"
                          f"scaling_governor")
                if g:
                    governors.add(g)
    except OSError:
        pass

    boost = None
    raw = _read("/sys/devices/system/cpu/cpufreq/boost")
    if raw in ("0", "1"):
        boost = raw == "1"
    else:
        # intel_pstate spells it inversely, as "no_turbo".
        no_turbo = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        if no_turbo in ("0", "1"):
            boost = no_turbo == "0"

    def khz(name: str) -> int | None:
        value = _read(os.path.join(base, name))
        try:
            return int(value) // 1000
        except (TypeError, ValueError):
            return None

    return {
        "available": True,
        "platform": "Linux",
        "governor": governor,
        "governors_in_use": sorted(governors) if len(governors) > 1 else None,
        "driver": driver,
        "min_mhz": khz("scaling_min_freq"),
        "max_mhz": khz("scaling_max_freq"),
        "boost_enabled": boost,
        "energy_performance_preference":
            _read(os.path.join(base, "energy_performance_preference")) or None,
    }


# --------------------------------------------------------------------------- #
# Memory policy
# --------------------------------------------------------------------------- #
def memory_policy() -> dict:
    """Transparent hugepages, swappiness, overcommit, and NUMA balancing."""
    if platform.system() != "Linux":
        return {"available": False}

    def bracketed(path: str) -> str | None:
        # These files list every option with the active one in brackets:
        # "always [madvise] never".
        text = _read(path)
        m = re.search(r"\[([^\]]+)\]", text)
        return m.group(1) if m else (text or None)

    thp = bracketed("/sys/kernel/mm/transparent_hugepage/enabled")
    defrag = bracketed("/sys/kernel/mm/transparent_hugepage/defrag")

    def as_int(path: str) -> int | None:
        try:
            return int(_read(path))
        except (TypeError, ValueError):
            return None

    return {
        "available": thp is not None or os.path.exists("/proc/sys/vm/swappiness"),
        "transparent_hugepages": thp,
        "thp_defrag": defrag,
        "swappiness": as_int("/proc/sys/vm/swappiness"),
        "overcommit_memory": as_int("/proc/sys/vm/overcommit_memory"),
        "numa_balancing": as_int("/proc/sys/kernel/numa_balancing"),
        "nr_hugepages": as_int("/proc/sys/vm/nr_hugepages"),
    }


# --------------------------------------------------------------------------- #
# SMT and microcode
# --------------------------------------------------------------------------- #
def smt_state() -> dict:
    """Simultaneous multithreading / Hyper-Threading state.

    "Not implemented by this chip" and "implemented but switched off" are
    different facts and must not be conflated: Apple silicon and many Intel
    i5/i3 parts have no SMT at all, and reporting those as "SMT DISABLED"
    would send the user hunting for a BIOS setting that does not exist. So
    ``supported`` is reported separately from ``enabled``, and is left None
    whenever the platform cannot tell the two apart.
    """
    if platform.system() == "Linux":
        # /sys/.../smt/control is authoritative: it distinguishes "off" from
        # "notsupported" explicitly.
        control = _read("/sys/devices/system/cpu/smt/control")
        active = _read("/sys/devices/system/cpu/smt/active")
        if control or active:
            supported = None
            if control in ("notsupported", "notimplemented"):
                supported = False
            elif control in ("on", "off", "forceoff"):
                supported = True
            return {"available": True,
                    "control": control or None,
                    "supported": supported,
                    "enabled": (active == "1" if active in ("0", "1")
                                else None)}
        return {"available": False}

    if platform.system() == "Darwin":
        try:
            logical = int(_run(["sysctl", "-n", "hw.logicalcpu"]) or 0)
            physical = int(_run(["sysctl", "-n", "hw.physicalcpu"]) or 0)
        except ValueError:
            return {"available": False}
        if not (logical and physical):
            return {"available": False}
        if logical > physical:
            return {"available": True, "control": None,
                    "supported": True, "enabled": True}
        # Equal counts. On ARM there is no SMT to enable; on x86 this could be
        # a chip without it or one with it turned off, and macOS does not say.
        if platform.machine().startswith("arm"):
            return {"available": True, "control": None, "supported": False,
                    "enabled": False,
                    "note": "Apple silicon does not implement SMT"}
        return {"available": True, "control": None, "supported": None,
                "enabled": False,
                "note": "one thread per core; macOS does not distinguish a "
                        "chip without Hyper-Threading from one with it off"}
    return {"available": False}


def microcode() -> dict:
    """Microcode / firmware revision.

    Worth recording because microcode updates have repeatedly *reduced*
    performance in order to fix errata, so two machines on different revisions
    are not directly comparable even with identical silicon.
    """
    if platform.system() == "Linux":
        for line in _read("/proc/cpuinfo").splitlines():
            if line.lower().startswith("microcode"):
                return {"available": True,
                        "revision": line.split(":", 1)[1].strip()}
        return {"available": False}
    if platform.system() == "Darwin":
        version = _run(["sysctl", "-n", "machdep.cpu.microcode_version"])
        if version:
            return {"available": True, "revision": version}
        # Apple silicon reports firmware rather than x86-style microcode.
        build = _run(["sysctl", "-n", "kern.osversion"])
        return {"available": bool(build), "revision": build or None,
                "note": "OS build; Apple silicon exposes no microcode revision"}
    return {"available": False}


def kernel_info() -> dict:
    """Kernel version, build flags of interest, and the boot command line."""
    info = {"release": platform.release(), "version": platform.version()}
    if platform.system() == "Linux":
        cmdline = _read("/proc/cmdline")
        info["cmdline"] = cmdline or None
        # Preemption model and tick configuration change latency behaviour
        # markedly, which is exactly what the latency suite measures.
        for token in cmdline.split():
            if token.startswith(("isolcpus=", "nohz", "preempt=",
                                 "processor.max_cstate=", "intel_idle")):
                info.setdefault("scheduling_flags", []).append(token)
    return info


# --------------------------------------------------------------------------- #
# Aggregate and interpretation
# --------------------------------------------------------------------------- #
def collect() -> dict:
    """Everything above, in one call. Never raises."""
    out: dict = {}
    for name, func in (("mitigations", mitigations),
                       ("frequency", frequency_policy),
                       ("memory", memory_policy),
                       ("smt", smt_state),
                       ("microcode", microcode),
                       ("kernel", kernel_info)):
        try:
            out[name] = func()
        except Exception as e:
            out[name] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    return out


def notes(info: dict) -> list[str]:
    """Configuration facts that materially affect the numbers just measured.

    Each note names the effect and its direction, because "THP is madvise" is
    useless to anyone who does not already know what that costs.
    """
    out: list[str] = []

    mit = info.get("mitigations") or {}
    if mit.get("available"):
        if mit.get("cmdline_override"):
            out.append(
                f"speculative-execution mitigations are altered on the kernel "
                f"command line ({mit['cmdline_override']}) — this machine may "
                f"score 5-30% above an otherwise identical one, and is not "
                f"comparable to a default configuration")
        elif mit.get("vulnerable"):
            out.append(
                f"mitigations are disabled for: {', '.join(mit['vulnerable'])} "
                f"— faster than a mitigated machine, and not comparable to one")
        elif mit.get("mitigated"):
            out.append(
                f"{len(mit['mitigated'])} speculative-execution mitigation(s) "
                f"active — these cost most on syscall- and context-switch-heavy "
                f"work, so the latency suite is affected more than the CPU tests")

    freq = info.get("frequency") or {}
    governor = freq.get("governor")
    if governor and governor not in ("performance",):
        out.append(
            f"CPU governor is '{governor}' rather than 'performance' — clocks "
            f"ramp on demand, which suppresses short tests most. This is a "
            f"settings change, not a hardware limit")
    if freq.get("governors_in_use"):
        out.append(f"cores are running different governors "
                   f"({', '.join(freq['governors_in_use'])}) — multicore "
                   f"results will be uneven")
    if freq.get("boost_enabled") is False:
        out.append("CPU boost/turbo is disabled — sustained results are "
                   "unaffected but peak single-core results are capped")
    if freq.get("low_power_mode") == "on":
        out.append("macOS Low Power Mode is on — the OS is deliberately "
                   "capping performance")

    mem = info.get("memory") or {}
    thp = mem.get("transparent_hugepages")
    if thp == "never":
        out.append("transparent hugepages are off — memory-bound and database "
                   "workloads lose measurably to TLB pressure here")
    elif thp == "always":
        out.append("transparent hugepages are 'always' — helps large "
                   "sequential working sets, and can hurt latency-sensitive "
                   "workloads through allocation stalls")
    if isinstance(mem.get("swappiness"), int) and mem["swappiness"] >= 60:
        out.append(f"vm.swappiness is {mem['swappiness']} — under memory "
                   f"pressure the kernel will page out rather than drop cache")

    smt = info.get("smt") or {}
    if (smt.get("available") and smt.get("enabled") is False
            and smt.get("supported") is True):
        out.append("SMT / Hyper-Threading is supported but switched off — "
                   "fewer logical cores than the hardware provides, so "
                   "multicore totals are lower by configuration, not by fault")
    return out


def render(info: dict, note_list: list[str] | None = None) -> str:
    """Terminal block. Only prints what the platform actually exposes."""
    lines: list[str] = []

    def row(label: str, value) -> None:
        if value not in (None, "", [], {}):
            lines.append(f"  {label:<26}: {value}")

    freq = info.get("frequency") or {}
    if freq.get("available"):
        row("CPU governor", freq.get("governor"))
        row("Scaling driver", freq.get("driver"))
        if freq.get("min_mhz") or freq.get("max_mhz"):
            row("Frequency range",
                f"{freq.get('min_mhz', '?')}–{freq.get('max_mhz', '?')} MHz")
        if freq.get("boost_enabled") is not None:
            row("Boost / turbo",
                "enabled" if freq["boost_enabled"] else "DISABLED")
        row("Energy pref", freq.get("energy_performance_preference"))
        row("Low Power Mode", freq.get("low_power_mode"))
        row("Power scheme", freq.get("power_scheme"))

    mem = info.get("memory") or {}
    if mem.get("available"):
        row("Transparent hugepages", mem.get("transparent_hugepages"))
        row("Swappiness", mem.get("swappiness"))
        if mem.get("numa_balancing") is not None:
            row("NUMA balancing",
                "on" if mem["numa_balancing"] else "off")

    smt = info.get("smt") or {}
    if smt.get("available"):
        if smt.get("supported") is False:
            row("SMT / Hyper-Threading", "not implemented by this CPU")
        elif smt.get("enabled") is True:
            row("SMT / Hyper-Threading", "enabled")
        elif smt.get("enabled") is False:
            row("SMT / Hyper-Threading",
                "DISABLED" if smt.get("supported") else
                "one thread per core (support unknown)")

    mit = info.get("mitigations") or {}
    if mit.get("available"):
        if mit.get("vulnerable"):
            row("Mitigations", f"DISABLED for {', '.join(mit['vulnerable'])}")
        else:
            row("Mitigations", f"{len(mit.get('mitigated', []))} active, "
                               f"{len(mit.get('not_affected', []))} not "
                               f"applicable")
        row("Mitigation cmdline", mit.get("cmdline_override"))

    micro = info.get("microcode") or {}
    if micro.get("available"):
        row("Microcode", micro.get("revision"))

    for note in note_list or []:
        lines.append(f"      i {note}")
    return "\n".join(lines)
