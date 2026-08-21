"""Hardware/OS inventory and live machine-state capture.

Two distinct concerns live here:

* **Inventory** — static facts about the machine (CPU, RAM, architecture).
* **State** — volatile conditions at the moment of the run (AC vs. battery,
  system load, thermal pressure). Benchmark numbers are close to
  uninterpretable without state, because a throttled or busy machine produces
  results that look like hardware differences but are not.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

_TIMEOUT = 5


def _run(cmd: list[str]) -> str:
    """Run a command, returning trimmed stdout or '' on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=_TIMEOUT)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def has_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
_ARCH_MAP = [
    (("x86_64", "amd64", "x64"), "x86-64"),
    (("i386", "i486", "i586", "i686", "x86"), "x86-32"),
    (("arm64", "aarch64", "aarch64_be", "arm64e"), "ARM64"),
    (("riscv64",), "RISC-V 64"),
    (("riscv32", "riscv"), "RISC-V 32"),
    (("s390x", "s390"), "IBM Z"),
]


def arch_family(machine: str) -> str:
    """Normalize ``platform.machine()`` into a canonical ISA family.

    Different operating systems spell the same chip differently (Windows says
    ``AMD64`` where Linux says ``x86_64``); normalizing lets results from mixed
    fleets line up.
    """
    m = (machine or "").lower()
    for names, family in _ARCH_MAP:
        if m in names:
            return family
    if m.startswith("armv") or m.startswith("arm"):
        return "ARM32"
    if m.startswith("riscv"):
        return "RISC-V 64" if "64" in m else "RISC-V 32"
    if m.startswith(("ppc64", "powerpc64")):
        return "PowerPC 64"
    if m.startswith(("ppc", "powerpc")):
        return "PowerPC 32"
    if m.startswith("mips"):
        return "MIPS"
    if m.startswith("loongarch"):
        return "LoongArch"
    return machine or "unknown"


# --------------------------------------------------------------------------- #
# CPU / RAM / cores
# --------------------------------------------------------------------------- #
def cpu_model() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        m = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if m:
            return m
    elif sysname == "Linux":
        # x86 exposes "model name"; ARM/RISC-V boards often do not.
        hardware = model = None
        for line in _read("/proc/cpuinfo").splitlines():
            low = line.lower()
            if low.startswith("model name"):
                return line.split(":", 1)[1].strip()
            if low.startswith("hardware"):
                hardware = line.split(":", 1)[1].strip()
            elif low.startswith("model") and ":" in line:
                model = line.split(":", 1)[1].strip()
        # Device tree covers most ARM single-board computers.
        for p in ("/sys/firmware/devicetree/base/model",
                  "/proc/device-tree/model"):
            try:
                with open(p, "rb") as f:
                    dt = f.read().rstrip(b"\x00").decode("utf-8", "ignore")
                if dt.strip():
                    return dt.strip()
            except OSError:
                pass
        if hardware:
            return hardware
        if model:
            return model
    elif sysname == "Windows":
        # Registry is authoritative and needs no subprocess.
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if name:
                return str(name).strip()
        except Exception:
            pass
        env = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if env:
            return env
    return platform.processor() or platform.machine()


def total_ram_bytes() -> int:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Darwin":
        v = _run(["sysctl", "-n", "hw.memsize"])
        return int(v) if v.isdigit() else 0
    if sysname == "Linux":
        m = re.search(r"MemTotal:\s+(\d+)\s*kB", _read("/proc/meminfo"))
        return int(m.group(1)) * 1024 if m else 0
    if sysname == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        except Exception:
            return 0
    return 0


def physical_cores() -> int | None:
    """Physical (not logical) core count, or None if undetectable."""
    try:
        import psutil  # type: ignore
        pc = psutil.cpu_count(logical=False)
        if pc:
            return int(pc)
    except Exception:
        pass

    sysname = platform.system()
    if sysname == "Darwin":
        v = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(v) if v.isdigit() else None

    if sysname == "Linux":
        # Count distinct (physical id, core id) pairs.
        pairs, phys = set(), None
        for line in _read("/proc/cpuinfo").splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id") and phys is not None:
                pairs.add((phys, line.split(":")[1].strip()))
        if pairs:
            return len(pairs)
        return None

    if sysname == "Windows":
        # PowerShell first (wmic is deprecated/removed on Windows 11+).
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | "
                    "Measure-Object -Property NumberOfCores -Sum).Sum"])
        if out.strip().isdigit():
            return int(out.strip())
        out = _run(["wmic", "cpu", "get", "NumberOfCores"])
        nums = [int(t) for t in re.findall(r"\d+", out)]
        if nums:
            return sum(nums)
        return None

    return None


def cpu_frequency_mhz() -> float | None:
    """Nominal/base CPU frequency in MHz, where the OS exposes it."""
    sysname = platform.system()
    if sysname == "Darwin":
        v = _run(["sysctl", "-n", "hw.tbfrequency"])  # not the CPU clock
        v = _run(["sysctl", "-n", "hw.cpufrequency"])
        if v.isdigit() and int(v) > 0:
            return int(v) / 1e6
        return None  # Apple Silicon does not publish a nominal clock
    if sysname == "Linux":
        m = re.search(r"cpu MHz\s*:\s*([\d.]+)", _read("/proc/cpuinfo"))
        if m:
            return float(m.group(1))
        khz = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if khz.strip().isdigit():
            return int(khz.strip()) / 1000.0
        return None
    if sysname == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).MaxClockSpeed"])
        m = re.search(r"\d+", out)
        return float(m.group()) if m else None
    return None


def gil_status() -> dict:
    """Report whether this interpreter runs with the GIL.

    Python 3.13+ can be built free-threaded, which changes multi-threaded
    scaling dramatically — worth recording alongside any CPU result.
    """
    info = {"free_threaded_build": False, "gil_enabled": True}
    try:
        import sysconfig
        info["free_threaded_build"] = bool(
            sysconfig.get_config_var("Py_GIL_DISABLED"))
    except Exception:
        pass
    getter = getattr(sys, "_is_gil_enabled", None)
    if callable(getter):
        try:
            info["gil_enabled"] = bool(getter())
        except Exception:
            pass
    elif info["free_threaded_build"]:
        info["gil_enabled"] = False
    return info


def inventory() -> dict:
    """Static hardware/OS/runtime facts."""
    machine = platform.machine()
    ram = total_ram_bytes()
    gil = gil_status()
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "architecture": machine,
        "arch_family": arch_family(machine),
        "arch_bits": 64 if sys.maxsize > 2 ** 32 else 32,
        "byte_order": sys.byteorder,
        "cpu_model": cpu_model(),
        "cpu_cores_physical": physical_cores(),
        "cpu_cores_logical": os.cpu_count() or 1,
        "cpu_base_mhz": cpu_frequency_mhz(),
        "ram_total_bytes": ram,
        "ram_total_gb": round(ram / (1024 ** 3), 2) if ram else None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "free_threaded_build": gil["free_threaded_build"],
        "gil_enabled": gil["gil_enabled"],
        "cpu_features": cpu_features(),
        "virtualization": virtualization(),
        "psutil_available": has_psutil(),
    }


# --------------------------------------------------------------------------- #
# CPU instruction-set features
#
# These explain benchmark results rather than merely decorating them: hardware
# SHA or AES instructions make the hashing benchmark several times faster, and
# wide SIMD (AVX-512, SVE) shows up directly in floating-point throughput.
# --------------------------------------------------------------------------- #
_FEATURE_LABELS = {
    # x86
    "aes": "AES-NI", "sha_ni": "SHA-NI", "sha-ni": "SHA-NI",
    "avx": "AVX", "avx2": "AVX2", "avx512f": "AVX-512",
    "sse4_2": "SSE4.2", "fma": "FMA",
    # ARM
    "aes_arm": "AES", "sha256": "SHA-256", "sha512": "SHA-512",
    "sha3": "SHA-3", "asimd": "NEON", "neon": "NEON", "sve": "SVE",
    "sve2": "SVE2", "i8mm": "Int8 matmul", "bf16": "BFloat16",
    "dotprod": "DotProd", "amx": "AMX",
}


def cpu_features() -> list[str]:
    """Human-readable list of performance-relevant CPU instruction sets."""
    system = platform.system()
    found: list[str] = []
    try:
        if system == "Darwin":
            if platform.machine().startswith("arm"):
                out = _run(["sysctl", "-a"])
                checks = [
                    (r"hw\.optional\.arm\.FEAT_AES:\s*1", "AES"),
                    (r"hw\.optional\.arm\.FEAT_SHA256:\s*1", "SHA-256"),
                    (r"hw\.optional\.arm\.FEAT_SHA512:\s*1", "SHA-512"),
                    (r"hw\.optional\.arm\.FEAT_SHA3:\s*1", "SHA-3"),
                    (r"hw\.optional\.arm\.FEAT_DotProd:\s*1", "DotProd"),
                    (r"hw\.optional\.arm\.FEAT_I8MM:\s*1", "Int8 matmul"),
                    (r"hw\.optional\.arm\.FEAT_BF16:\s*1", "BFloat16"),
                    (r"hw\.optional\.AdvSIMD:\s*1", "NEON"),
                    (r"hw\.optional\.arm\.FEAT_SVE:\s*1", "SVE"),
                    (r"hw\.optional\.amx_version:\s*[1-9]", "AMX"),
                ]
                for pattern, label in checks:
                    if re.search(pattern, out):
                        found.append(label)
            else:
                flags = _run(["sysctl", "-n", "machdep.cpu.features",
                              "machdep.cpu.leaf7_features"]).lower()
                for key, label in (("aes", "AES-NI"), ("sha", "SHA-NI"),
                                   ("avx512f", "AVX-512"), ("avx2", "AVX2"),
                                   ("avx1.0", "AVX"), ("fma", "FMA")):
                    if key in flags:
                        found.append(label)

        elif system == "Linux":
            flags = set()
            for line in _read("/proc/cpuinfo").splitlines():
                low = line.lower()
                if low.startswith("flags") or low.startswith("features"):
                    flags.update(low.split(":", 1)[1].split())
            for key in ("aes", "sha_ni", "avx512f", "avx2", "avx", "fma",
                        "sse4_2", "sha256", "sha512", "sha3", "asimd",
                        "sve", "sve2", "i8mm", "bf16", "dotprod"):
                if key in flags:
                    label = _FEATURE_LABELS.get(key, key.upper())
                    if label not in found:
                        found.append(label)

        elif system == "Windows":
            # PowerShell exposes IsProcessorFeaturePresent indirectly; the
            # identifier string is a reliable coarse fallback.
            ident = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor).Name"]).lower()
            blob = ident + " " + out
            if "intel" in blob or "amd" in blob:
                found.append("x86-64")
    except Exception:
        pass
    return found


def virtualization() -> str | None:
    """Detect a hypervisor. Benchmarks in a VM are not comparable to bare metal."""
    system = platform.system()
    try:
        if system == "Darwin":
            if _run(["sysctl", "-n", "kern.hv_vmm_present"]).strip() == "1":
                return "virtual machine"
            return None
        if system == "Linux":
            for path in ("/sys/class/dmi/id/product_name",
                         "/sys/class/dmi/id/sys_vendor"):
                v = _read(path).lower()
                for needle, name in (("kvm", "KVM"), ("vmware", "VMware"),
                                     ("virtualbox", "VirtualBox"),
                                     ("qemu", "QEMU"), ("xen", "Xen"),
                                     ("hyper-v", "Hyper-V"),
                                     ("parallels", "Parallels")):
                    if needle in v:
                        return name
            if "hypervisor" in _read("/proc/cpuinfo").lower():
                return "hypervisor present"
            return None
        if system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).Model"])
            low = out.lower()
            for needle, name in (("virtual", "virtual machine"),
                                 ("vmware", "VMware"), ("kvm", "KVM"),
                                 ("parallels", "Parallels")):
                if needle in low:
                    return name
            return None
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Volatile machine state
# --------------------------------------------------------------------------- #
def on_ac_power() -> bool | None:
    """True on AC, False on battery, None if there is no battery / unknown.

    Laptops aggressively down-clock on battery, so a run on battery can look
    like a slower machine when it is only a slower power profile.
    """
    sysname = platform.system()
    if sysname == "Darwin":
        out = _run(["pmset", "-g", "batt"])
        if "AC Power" in out:
            return True
        if "Battery Power" in out:
            return False
        return None
    if sysname == "Linux":
        import glob
        for p in glob.glob("/sys/class/power_supply/A*/online"):
            v = _read(p).strip()
            if v in ("0", "1"):
                return v == "1"
        # No AC adapter node at all usually means a desktop/server.
        if not glob.glob("/sys/class/power_supply/BAT*"):
            return True
        return None
    if sysname == "Windows":
        try:
            import ctypes

            class SPS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_byte),
                            ("BatteryFlag", ctypes.c_byte),
                            ("BatteryLifePercent", ctypes.c_byte),
                            ("SystemStatusFlag", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong),
                            ("BatteryFullLifeTime", ctypes.c_ulong)]

            st = SPS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
                if st.ACLineStatus == 1:
                    return True
                if st.ACLineStatus == 0:
                    return False
        except Exception:
            pass
        return None
    return None


def load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()  # not available on Windows
    except (OSError, AttributeError):
        return None


def thermal_pressure() -> str | None:
    """Best-effort thermal/throttling indicator."""
    sysname = platform.system()
    if sysname == "Darwin":
        out = _run(["pmset", "-g", "therm"])
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        if m:
            limit = int(m.group(1))
            return "nominal" if limit >= 100 else f"throttled ({limit}%)"
        if "No thermal warning level" in out:
            return "nominal"
        return None
    if sysname == "Linux":
        import glob
        temps = []
        for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            v = _read(p).strip()
            if v.lstrip("-").isdigit():
                temps.append(int(v) / 1000.0)
        if temps:
            return f"max {max(temps):.0f}C"
        return None
    return None


def machine_state(script_dir: str = ".") -> dict:
    """Volatile conditions captured at the start of a run."""
    la = load_average()
    logical = os.cpu_count() or 1

    # Real degrees Celsius where the platform exposes a sensor. This is both a
    # reported figure and the input to the sustained-load thermal cutoff.
    from . import thermal as _thermal
    temps = _thermal.read(script_dir)

    state = {
        "on_ac_power": on_ac_power(),
        "load_average": [round(x, 2) for x in la] if la else None,
        "load_per_core": round(la[0] / logical, 3) if la else None,
        "thermal": thermal_pressure(),
        "temperatures": temps or None,
        "cpu_celsius": temps.get("cpu_celsius") if temps else None,
    }
    # Fold a measured temperature into the free-form thermal string so the
    # existing throttle/abort checks see it too.
    if state["cpu_celsius"] is not None:
        base = state["thermal"] or ""
        state["thermal"] = (f"{base}, max {state['cpu_celsius']:.0f}C".lstrip(", ")
                            if base else f"max {state['cpu_celsius']:.0f}C")
    return state


def state_warnings(state: dict) -> list[str]:
    """Conditions that will visibly distort results, phrased for the user."""
    warns = []
    if state.get("on_ac_power") is False:
        warns.append("Running on BATTERY — CPU is likely down-clocked. "
                     "Plug in for comparable numbers.")
    lpc = state.get("load_per_core")
    if lpc is not None and lpc > 0.30:
        warns.append(f"System already busy (load/core = {lpc:.2f}). "
                     "Close other applications before benchmarking.")
    temp = state.get("cpu_celsius")
    if temp is not None and temp >= 95:
        warns.append(f"CPU is already at {temp:.0f} °C before starting. "
                     f"Let the machine cool down first.")
    therm = state.get("thermal") or ""
    if "throttled" in therm:
        warns.append(f"CPU is thermally throttled ({therm}). "
                     "Let the machine cool down first.")
    return warns
