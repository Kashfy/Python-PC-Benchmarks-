"""GPU and NPU detection, plus orchestration of the native accelerator engine.

Scope, stated plainly:

* **Inventory** works on Windows, macOS, and Linux — GPU model, memory, driver,
  and NPU presence.
* **Benchmarking** currently covers Apple platforms only, via Metal (GPU) and
  Core ML (Neural Engine). Compute benchmarks on other vendors need their
  SDKs — CUDA, ROCm, oneAPI, or an OpenCL runtime — and shipping code for
  hardware that cannot be tested would be worse than reporting the gap
  honestly. Non-Apple systems get full inventory and a note.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess

from . import coreml_model

SOURCE_NAME = "accel_engine.m"
BINARY_NAME = "accel_engine"
_TIMEOUT = 15


def _run(cmd: list[str], timeout: int = _TIMEOUT) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------- #
# GPU inventory
# --------------------------------------------------------------------------- #
def _gpus_macos() -> list[dict]:
    raw = _run(["system_profiler", "-json", "SPDisplaysDataType"])
    gpus = []
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return gpus
    for item in data.get("SPDisplaysDataType", []):
        gpu = {
            "name": item.get("sppci_model") or item.get("_name") or "unknown",
            "vendor": item.get("spdisplays_vendor"),
            "cores": _int_or_none(item.get("sppci_cores")),
            "vram_mb": _vram_mb(item),
            "metal": item.get("spdisplays_metalfamily")
                     or item.get("spdisplays_metal"),
            "bus": item.get("sppci_bus"),
        }
        gpus.append({k: v for k, v in gpu.items() if v is not None})
    return gpus


def _vram_mb(item: dict) -> float | None:
    for key in ("spdisplays_vram_shared", "spdisplays_vram", "sppci_vram"):
        val = item.get(key)
        if not val:
            continue
        m = re.search(r"([\d.]+)\s*(GB|MB)", str(val), re.I)
        if m:
            n = float(m.group(1))
            return n * 1024 if m.group(2).upper() == "GB" else n
    return None


def _int_or_none(v) -> int | None:
    try:
        return int(re.search(r"\d+", str(v)).group())
    except (AttributeError, TypeError, ValueError):
        return None


def _gpus_linux() -> list[dict]:
    gpus = []

    # NVIDIA exposes everything through nvidia-smi when the driver is loaded.
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0],
                             "vram_mb": _float_or_none(parts[1]),
                             "driver": parts[2], "vendor": "NVIDIA"})
        if gpus:
            return gpus

    # Otherwise fall back to PCI enumeration, which needs no driver tooling.
    out = _run(["lspci", "-mm"])
    for line in out.splitlines():
        if re.search(r'"(VGA compatible controller|3D controller|Display '
                     r'controller)"', line):
            fields = re.findall(r'"([^"]*)"', line)
            if len(fields) >= 3:
                gpus.append({"name": f"{fields[1]} {fields[2]}".strip(),
                             "vendor": fields[1]})
    if gpus:
        return gpus

    # Last resort: the DRM subsystem is present even in minimal containers.
    try:
        for card in sorted(os.listdir("/sys/class/drm")):
            if re.fullmatch(r"card\d+", card):
                vendor = _read(f"/sys/class/drm/{card}/device/vendor")
                device = _read(f"/sys/class/drm/{card}/device/device")
                if vendor:
                    gpus.append({"name": f"DRM {card} "
                                         f"[{vendor}:{device or '?'}]",
                                 "vendor": _pci_vendor(vendor)})
    except OSError:
        pass
    return gpus


_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel",
                "0x1a03": "ASPEED", "0x15ad": "VMware"}


def _pci_vendor(vid: str) -> str:
    return _PCI_VENDORS.get((vid or "").strip().lower(), vid)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def _float_or_none(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


#: Win32_VideoController.AdapterRAM is a 32-bit DWORD. Any card with 4 GB or
#: more reports a value pinned just below 4 GiB -- an RTX 5070 Ti with 16 GB
#: reported 4293918720 bytes, which the tool printed as "4.0 GB". Values at or
#: above this are treated as capped rather than published as fact.
_ADAPTER_RAM_CAP = 4_000_000_000


def _gpu_vram_from_registry() -> dict:
    """Accurate VRAM per adapter from the display-class registry key.

    ``HardwareInformation.qwMemorySize`` is a 64-bit value written by the
    driver, so unlike AdapterRAM it is correct above 4 GB. Keyed by adapter
    description so it can be matched back to the WMI listing.
    """
    script = (
        "$p='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
        "{4d36e968-e325-11ce-bfc1-08002be10318}'; "
        "Get-ChildItem $p -ErrorAction SilentlyContinue | ForEach-Object { "
        "  $k = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; "
        "  if ($k.'HardwareInformation.qwMemorySize') { "
        "    [pscustomobject]@{ name=$k.DriverDesc; "
        "      bytes=[uint64]$k.'HardwareInformation.qwMemorySize' } } "
        "} | ConvertTo-Json -Compress"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", script])
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        data = [data]
    out = {}
    for item in data:
        name, size = item.get("name"), item.get("bytes")
        if name and isinstance(size, (int, float)) and size > 0:
            out[str(name).strip().lower()] = int(size)
    return out


def _gpus_windows() -> list[dict]:
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor | "
                "ConvertTo-Json -Compress"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    registry = _gpu_vram_from_registry()
    gpus = []
    for item in data:
        name = item.get("Name") or "unknown"
        ram = item.get("AdapterRAM")

        # Prefer the 64-bit registry value; fall back to AdapterRAM only when
        # it is small enough to be trustworthy.
        vram_bytes = registry.get(str(name).strip().lower())
        source = "registry qwMemorySize" if vram_bytes else None
        if not vram_bytes and isinstance(ram, (int, float)) \
                and 0 < ram < _ADAPTER_RAM_CAP:
            vram_bytes, source = int(ram), "WMI AdapterRAM"

        entry = {
            "name": name,
            "vram_mb": vram_bytes / (1024 * 1024) if vram_bytes else None,
            "vram_source": source,
            "driver": item.get("DriverVersion"),
        }
        if not vram_bytes and isinstance(ram, (int, float)) \
                and ram >= _ADAPTER_RAM_CAP:
            # Saying nothing beats saying 4 GB about a 16 GB card.
            entry["vram_note"] = (
                "WMI reports a 32-bit value capped near 4 GB and the driver "
                "registry entry was unreadable, so VRAM is unknown")
        gpus.append(entry)
    return [{k: v for k, v in g.items() if v is not None} for g in gpus]


def detect_gpus() -> list[dict]:
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            return _gpus_macos()
        if sysname == "Linux":
            return _gpus_linux()
        if sysname == "Windows":
            return _gpus_windows()
    except Exception:
        pass
    return []


# --------------------------------------------------------------------------- #
# NPU inventory
# --------------------------------------------------------------------------- #
# Apple shipped the Neural Engine in every Apple-silicon Mac, so presence can
# be inferred from the chip rather than probed.
_APPLE_SILICON = re.compile(r"\bApple\s+M\d", re.I)

# Windows exposes NPUs as PnP devices; vendors name them inconsistently.
_WINDOWS_NPU_HINTS = re.compile(
    r"neural processor|neural engine|ai boost|\bnpu\b|hexagon|xdna|"
    r"\bipu\b|compute accelerator", re.I)

# PCI IDs for the discrete NPU blocks on current laptop silicon. Vendor ID
# alone is not enough — Intel and AMD both ship GPUs under the same vendor —
# so the device ID identifies the NPU specifically.
_NPU_PCI_IDS = {
    # Intel "AI Boost" NPU (VPU), driver: intel_vpu
    ("0x8086", "0x7d1d"): ("Intel AI Boost NPU (Meteor Lake)", "Intel"),
    ("0x8086", "0xad1d"): ("Intel AI Boost NPU (Arrow Lake)", "Intel"),
    ("0x8086", "0x643e"): ("Intel AI Boost NPU (Lunar Lake)", "Intel"),
    ("0x8086", "0xb03e"): ("Intel AI Boost NPU (Panther Lake)", "Intel"),
    # AMD XDNA / Ryzen AI, driver: amdxdna
    ("0x1022", "0x1502"): ("AMD Ryzen AI NPU (XDNA, Phoenix)", "AMD"),
    ("0x1022", "0x17f0"): ("AMD Ryzen AI NPU (XDNA2, Strix)", "AMD"),
}

# Linux kernel drivers that bind an NPU, when the PCI ID is unrecognised.
_NPU_DRIVERS = {
    "intel_vpu": ("Intel AI Boost NPU", "Intel"),
    "ivpu": ("Intel AI Boost NPU", "Intel"),
    "amdxdna": ("AMD Ryzen AI NPU (XDNA)", "AMD"),
    "qaic": ("Qualcomm Cloud AI accelerator", "Qualcomm"),
}

# Which software stack can actually reach each vendor's NPU.
_NPU_APIS = {
    "Intel": "OpenVINO / DirectML / ONNX Runtime",
    "AMD": "Vitis AI / DirectML / ONNX Runtime",
    "Qualcomm": "QNN / DirectML / ONNX Runtime",
    "Apple": "Core ML",
}


def _npus_linux() -> list[dict]:
    """Detect NPUs through the accel subsystem and PCI IDs.

    Linux exposes NPUs as /dev/accel/accelN (the accel subsystem added for
    compute accelerators). Reading the backing PCI device identifies the
    vendor precisely, which matters because a bare "accel0" node says nothing
    about whose silicon it is.
    """
    found: list[dict] = []
    seen: set[str] = set()

    for node in sorted(glob.glob("/sys/class/accel/accel*")):
        name = os.path.basename(node)
        vendor_id = _read(f"{node}/device/vendor").strip().lower()
        device_id = _read(f"{node}/device/device").strip().lower()

        label = vendor = None
        if (vendor_id, device_id) in _NPU_PCI_IDS:
            label, vendor = _NPU_PCI_IDS[(vendor_id, device_id)]
        else:
            # Fall back to whichever driver claimed the device.
            driver = os.path.basename(
                os.path.realpath(f"{node}/device/driver")) \
                if os.path.exists(f"{node}/device/driver") else ""
            if driver in _NPU_DRIVERS:
                label, vendor = _NPU_DRIVERS[driver]
            elif vendor_id:
                label = f"Accelerator {name} [{vendor_id}:{device_id}]"
                vendor = _pci_vendor(vendor_id)

        if label and label not in seen:
            seen.add(label)
            found.append({"name": label, "vendor": vendor,
                          "device": f"/dev/accel/{name}",
                          "api": _NPU_APIS.get(vendor,
                                               "Level Zero / ONNX Runtime"),
                          "benchmarkable": False})

    # A device node with no sysfs class entry still indicates an NPU.
    if not found:
        for dev, (label, vendor) in (
                ("/dev/accel/accel0", ("Compute accelerator", None)),
                ("/dev/amdxdna", _NPU_DRIVERS["amdxdna"])):
            if os.path.exists(dev):
                found.append({"name": label, "vendor": vendor,
                              "device": dev,
                              "api": _NPU_APIS.get(vendor,
                                                   "vendor runtime"),
                              "benchmarkable": False})
    return found


def _npus_windows() -> list[dict]:
    """Detect NPUs via PnP enumeration.

    Windows 11 gives NPUs their own "ComputeAccelerator" device class, but
    older drivers register under System, so the query matches on name too.
    """
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.PNPClass -eq 'ComputeAccelerator' -or "
                "$_.Name -match 'Neural|NPU|AI Boost|Hexagon|XDNA|IPU' } | "
                "Select-Object Name,Manufacturer,PNPClass | "
                "ConvertTo-Json -Compress"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    found = []
    for item in data:
        name = (item.get("Name") or "").strip()
        klass = item.get("PNPClass") or ""
        if not name:
            continue
        if klass != "ComputeAccelerator" and not _WINDOWS_NPU_HINTS.search(name):
            continue
        maker = item.get("Manufacturer") or ""
        vendor = ("Intel" if re.search(r"intel", name + maker, re.I) else
                  "AMD" if re.search(r"amd|xdna", name + maker, re.I) else
                  "Qualcomm" if re.search(r"qualcomm|hexagon", name + maker,
                                          re.I) else maker or None)
        found.append({"name": name, "vendor": vendor,
                      "api": _NPU_APIS.get(vendor, "DirectML / ONNX Runtime"),
                      "benchmarkable": False})
    return found


def detect_npus(cpu_model: str = "") -> list[dict]:
    """Enumerate neural accelerators across Apple, Intel, AMD, and Qualcomm."""
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            if _APPLE_SILICON.search(cpu_model or ""):
                return [{
                    "name": "Apple Neural Engine",
                    "vendor": "Apple",
                    "api": "Core ML",
                    "benchmarkable": True,
                    "note": "not directly programmable; reachable only "
                            "through Core ML",
                }]
            return []
        if sysname == "Linux":
            return _npus_linux()
        if sysname == "Windows":
            return _npus_windows()
    except Exception:
        pass
    return []


def inventory(cpu_model: str = "") -> dict:
    gpus = detect_gpus()
    npus = detect_npus(cpu_model)
    return {
        "gpus": gpus,
        "npus": npus,
        "gpu_count": len(gpus),
        "npu_count": len(npus),
        "benchmark_supported": platform.system() == "Darwin",
    }


# --------------------------------------------------------------------------- #
# Native accelerator engine
# --------------------------------------------------------------------------- #
def _frameworks() -> list[str]:
    return ["-framework", "Foundation", "-framework", "Metal",
            "-framework", "MetalPerformanceShaders", "-framework", "CoreML"]


def build(src: str, exe: str) -> tuple[bool, str]:
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return False, "clang not found"
    cmd = [cc, "-O2", "-fobjc-arc", src, "-o", exe] + _frameworks()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:600]
    return True, ""


def run(seconds: float, script_dir: str, out_dir: str,
        gpu: bool = True, ane: bool = True) -> dict | None:
    """Build and run the Metal/Core ML engine, returning its parsed JSON.

    Returns None on non-Apple platforms (where the engine does not apply), or
    an ``{"error": ...}`` dict on any failure. Never raises.
    """
    if platform.system() != "Darwin":
        return None
    src = os.path.join(script_dir, SOURCE_NAME)
    if not os.path.isfile(src):
        return None

    exe = os.path.join(script_dir, BINARY_NAME)
    if (not os.path.isfile(exe)
            or os.path.getmtime(exe) < os.path.getmtime(src)):
        ok, err = build(src, exe)
        if not ok:
            return {"error": "accelerator engine build failed", "detail": err}

    cmd = [exe, "--json", "--seconds", str(seconds)]
    if not gpu:
        cmd.append("--no-gpu")

    if ane:
        # The model is generated rather than shipped, so its size can be tuned
        # to whatever reliably engages the ANE.
        try:
            model_path = coreml_model.write_model(
                os.path.join(out_dir, "ane_model.mlmodel"))
            c, s = coreml_model.DEFAULT_CHANNELS, coreml_model.DEFAULT_SPATIAL
            cmd += ["--model", model_path,
                    "--flops", str(coreml_model.flops_per_inference()),
                    "--shape", f"1,{c},{s},{s}"]
        except OSError as e:
            cmd.append("--no-ane")
            ane = False
            _ = e
    else:
        cmd.append("--no-ane")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=seconds * 12 + 120)
    except subprocess.SubprocessError as e:
        return {"error": f"accelerator engine run error: {e}"}
    if proc.returncode != 0:
        return {"error": "accelerator engine run failed",
                "detail": (proc.stderr or "").strip()[:600]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"accelerator output not valid JSON: {e}"}


def extract_rates(payload: dict | None) -> dict:
    """Pull the headline accelerator numbers out for scoring."""
    rates: dict[str, float] = {}
    if not payload or "error" in payload:
        return rates
    for item in payload.get("results", []):
        name, value = item.get("name", ""), item.get("value")
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        if name == "GPU FP32 FMA":
            rates["gpu_fp32"] = float(value)
        elif name == "GPU FP16 FMA":
            rates["gpu_fp16"] = float(value)
        elif name == "GPU memory bandwidth":
            rates["gpu_bandwidth"] = float(value)
        elif name == "GPU matmul FP32 (GEMM)":
            rates["gpu_matmul_fp32"] = float(value)  # TFLOPS
        elif name == "GPU matmul FP16 (GEMM)":
            rates["gpu_matmul_fp16"] = float(value)  # TFLOPS
        elif name == "Neural Engine throughput":
            rates["npu"] = float(value)
    return rates
