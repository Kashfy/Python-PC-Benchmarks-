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
    gpus = []
    for item in data:
        ram = item.get("AdapterRAM")
        gpus.append({
            "name": item.get("Name") or "unknown",
            # AdapterRAM is a signed 32-bit field, so it misreports anything
            # at or above 4 GB; drop it rather than publish a wrong number.
            "vram_mb": (ram / (1024 * 1024)
                        if isinstance(ram, (int, float)) and 0 < ram < 2 ** 32
                        else None),
            "driver": item.get("DriverVersion"),
        })
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

_WINDOWS_NPU_HINTS = re.compile(
    r"neural processor|neural engine|ai boost|npu|hexagon|xdna", re.I)


def detect_npus(cpu_model: str = "") -> list[dict]:
    sysname = platform.system()
    npus: list[dict] = []
    try:
        if sysname == "Darwin":
            if _APPLE_SILICON.search(cpu_model or ""):
                npus.append({
                    "name": "Apple Neural Engine",
                    "vendor": "Apple",
                    "api": "Core ML",
                    "benchmarkable": True,
                    "note": "not directly programmable; reachable only "
                            "through Core ML",
                })
        elif sysname == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_PnPEntity | "
                        "Where-Object { $_.Name -match "
                        "'Neural|NPU|AI Boost|Hexagon' } | "
                        "Select-Object Name,Manufacturer | "
                        "ConvertTo-Json -Compress"])
            if out:
                try:
                    data = json.loads(out)
                except json.JSONDecodeError:
                    data = []
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    name = item.get("Name", "")
                    if _WINDOWS_NPU_HINTS.search(name):
                        npus.append({"name": name,
                                     "vendor": item.get("Manufacturer"),
                                     "api": "DirectML / OpenVINO",
                                     "benchmarkable": False})
        elif sysname == "Linux":
            # Intel's NPU driver exposes /dev/accel/accelN; AMD XDNA uses
            # an amdxdna node.
            for node in ("/dev/accel", "/sys/class/accel"):
                if os.path.exists(node):
                    try:
                        entries = os.listdir(node)
                    except OSError:
                        entries = []
                    for e in entries:
                        npus.append({"name": f"Accelerator ({e})",
                                     "api": "Level Zero / OpenVINO",
                                     "benchmarkable": False})
            if os.path.exists("/dev/amdxdna"):
                npus.append({"name": "AMD XDNA NPU", "vendor": "AMD",
                             "api": "XRT", "benchmarkable": False})
    except Exception:
        pass
    return npus


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
            "-framework", "CoreML"]


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
        elif name == "Neural Engine throughput":
            rates["npu"] = float(value)
    return rates
