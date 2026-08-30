"""Registry of optional third-party packages, grouped into installable tiers.

The tool's core runs on the standard library alone and always will — that is
what lets it be copied onto an unfamiliar machine and run immediately. But the
standard library imposes real ceilings:

* Pure-Python arithmetic measures **CPython**, not the CPU. Measured on an
  Apple M4 the Python neural-net benchmark reaches ~113 MFLOPS while the C
  engine reaches ~320,000 — a thousandfold gap that no amount of tuning in
  Python closes.
* There is no AES primitive, so hardware crypto cannot be benchmarked.
* There is no portable GPU compute API, which is why GPU benchmarking has been
  Apple-only.

Optional packages lift each of those ceilings. Every one is genuinely
optional: nothing here is imported at module load, absence is always reported
rather than raised, and the stdlib results remain untouched either way.

``install.py`` turns this registry into an interactive installer.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import re
import site
from typing import NamedTuple


class Package(NamedTuple):
    """One optional dependency and what it unlocks."""
    import_name: str        # what `import X` uses
    pip_name: str           # what pip installs
    purpose: str            # why the tool wants it
    approx_mb: int          # rough download size, for the installer prompt
    critical: bool = False  # the tier is largely pointless without it
    needs_gpu: str = ""     # only applies with this GPU vendor present


TIERS: dict[str, dict] = {
    "compute": {
        "summary": "Real BLAS/LAPACK numerics — CPU GFLOPS that reflect the "
                   "hardware rather than the interpreter",
        "packages": [
            Package("numpy", "numpy",
                    "BLAS matrix multiply, FFT, true vectorised bandwidth",
                    20, critical=True),
            Package("scipy", "scipy",
                    "LAPACK decompositions: SVD, eigenvalues, Cholesky", 35),
            Package("numba", "numba",
                    "JIT-compiled Python, showing the optimised-code ceiling",
                    30),
        ],
    },
    "gpu": {
        "summary": "Cross-platform GPU compute and vendor telemetry",
        "packages": [
            Package("pyopencl", "pyopencl",
                    "GPU compute on NVIDIA, AMD, and Intel via OpenCL",
                    15, critical=True),
            Package("pynvml", "nvidia-ml-py",
                    "NVIDIA temperature, power draw, VRAM and utilisation", 1),
        ],
    },
    "crypto": {
        "summary": "Hardware crypto and modern compression codecs",
        "packages": [
            Package("cryptography", "cryptography",
                    "AES-GCM throughput using AES-NI / ARM crypto extensions",
                    5, critical=True),
            Package("zstandard", "zstandard",
                    "Zstandard compression — the modern zlib replacement", 5),
            Package("lz4", "lz4", "LZ4 — speed-oriented compression", 2),
            Package("blake3", "blake3", "BLAKE3 hashing alongside SHA-256", 2),
        ],
    },
    "data": {
        "summary": "Dataframe engines for the analytics benchmarks, and "
                   "LINPACK's optimised BLAS",
        "packages": [
            Package("numpy", "numpy",
                    "LINPACK/HPL GFLOPS and the CPU LLM backend", 20,
                    critical=True),
            Package("polars", "polars",
                    "Columnar dataframe engine — filter, group-by, join, sort",
                    35),
            Package("pandas", "pandas",
                    "The reference dataframe API most analysis is written in",
                    45),
            Package("duckdb", "duckdb",
                    "Embedded analytical SQL over the same data", 40),
        ],
    },
    "ai": {
        "summary": "Cross-vendor NPU and GPU inference through ONNX Runtime",
        "packages": [
            Package("onnxruntime", "onnxruntime",
                    "NPU and GPU inference through execution providers — "
                    "CUDA, ROCm, DirectML, Core ML, OpenVINO, QNN",
                    40, critical=True),
            # After onnxruntime, always: the version to install is read out of
            # the provider library ONNX Runtime ships.
            Package("tensorrt_libs", "tensorrt",
                    "NVIDIA's optimising inference runtime — the provider "
                    "ONNX Runtime tries first on NVIDIA, and typically 30-50% "
                    "faster than the plain CUDA one",
                    1200, needs_gpu="nvidia"),
        ],
    },
    "system": {
        "summary": "Better sensors, richer output, and reference ML",
        "packages": [
            Package("psutil", "psutil",
                    "Per-core utilisation, sensors and counters on Windows "
                    "and Linux", 1, critical=True),
            Package("cpuinfo", "py-cpuinfo",
                    "Full instruction-set detection via CPUID — the only way "
                    "to see AES-NI and SHA-NI on Windows", 1),
            Package("rich", "rich",
                    "Formatted tables and colour in the terminal report", 2),
            Package("matplotlib", "matplotlib",
                    "Real charts and trend graphs in the HTML report", 30),
            Package("sklearn", "scikit-learn",
                    "Reference k-means/k-NN implementations at real scale",
                    40),
        ],
    },
}

# PyTorch is large enough, and its build matrix hardware-specific enough, that
# it is named rather than offered: the wheel that reaches a GPU differs by
# vendor and CUDA version, and picking wrong silently installs a CPU-only build.
HEAVY = [
    Package("torch", "torch",
            "Dense GEMM in TFLOPS on CUDA / ROCm / MPS — the AI-compute "
            "figure in the GPU section", 800),
]


# --------------------------------------------------------------------------- #
# Distributions that depend on the hardware
# --------------------------------------------------------------------------- #
def _site_dirs() -> list[str]:
    """Every site-packages directory this interpreter reads from."""
    paths = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user = site.getusersitepackages()
        if isinstance(user, str):
            paths.append(user)
    except Exception:
        pass
    return paths


def _gpu_vendors() -> set[str]:
    """Lower-cased vendor names of the GPUs in this machine.

    Imported lazily: this module is imported by everything, and GPU detection
    shells out on some platforms.
    """
    try:
        from . import accel
        return {(gpu.get("vendor") or "").lower()
                for gpu in accel.detect_gpus()}
    except Exception:
        return set()


def onnxruntime_distribution() -> str:
    """Which ONNX Runtime wheel actually reaches this machine's accelerators.

    One import name, several distributions, and the difference is the whole
    point of the tier. The plain ``onnxruntime`` wheel on PyPI carries only
    ``CPUExecutionProvider`` on Windows and Linux, so installing it on a
    machine with a discrete GPU produces an NPU section that runs, engages
    nothing, and reports the CPU. macOS is the exception: Core ML is in the
    default wheel.

    The CUDA build additionally needs a CUDA runtime and cuDNN, which come as
    extras — without them ONNX Runtime advertises the provider and then fails
    to load it, which reads as "the accelerator did not engage".
    """
    import platform as _platform
    system = _platform.system()
    if system == "Darwin":
        return "onnxruntime"
    vendors = _gpu_vendors()
    if "nvidia" in vendors:
        return "onnxruntime-gpu[cuda,cudnn]"
    if "amd" in vendors and system == "Linux":
        return "onnxruntime-gpu"              # ROCm execution provider
    if system == "Windows":
        return "onnxruntime-directml"         # reaches any DX12 GPU or NPU
    return "onnxruntime"


#: Where ONNX Runtime keeps the provider library that dlopens TensorRT, and
#: the soname it was built against. Reading the requirement out of the binary
#: beats hardcoding a version: ONNX Runtime pins a TensorRT major, and pip's
#: newest is routinely ahead of it — today `pip install tensorrt` fetches 11.x
#: while ONNX Runtime 1.29 wants libnvinfer.so.10, which fails at load with
#: nothing in the report but "the accelerator did not engage".
_NVINFER_SONAME = re.compile(rb"libnvinfer\.so\.(\d+)")


def _tensorrt_major() -> str | None:
    """The TensorRT major ONNX Runtime's TensorRT provider was built against."""
    try:
        import onnxruntime
        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        for name in os.listdir(capi):
            if "providers_tensorrt" not in name:
                continue
            with open(os.path.join(capi, name), "rb") as fh:
                found = _NVINFER_SONAME.findall(fh.read())
            if found:
                return max(found, key=int).decode()
    except Exception:
        pass
    return None


def _cuda_major() -> str:
    """The CUDA major the installed NVIDIA wheels are built for."""
    for path in _site_dirs():
        for entry in sorted(glob.glob(os.path.join(path, "nvidia", "cu*")),
                            reverse=True):
            digits = os.path.basename(entry)[2:]
            if digits.isdigit():
                return digits
    return "13"


def tensorrt_distribution() -> str | None:
    """The TensorRT distribution ONNX Runtime here can actually load.

    ``None`` when it cannot be determined, which means ONNX Runtime is not
    installed yet or was built without the TensorRT provider — installing a
    guess would be 1.2 GB of libraries nothing loads.

    Only the ``-libs`` package, never the metapackage. ONNX Runtime dlopens
    the C++ library and never touches the Python bindings, and the bindings
    are the part that lags: at the time of writing there is no TensorRT 10
    binding wheel for Python 3.14 at all, so asking for `tensorrt` on a 3.14
    interpreter resolves to 11.x, which ONNX Runtime cannot load.
    """
    major = _tensorrt_major()
    if not major:
        return None
    ceiling = int(major) + 1
    return (f"tensorrt-cu{_cuda_major()}-libs>={major},<{ceiling}")


def pip_target(pkg: Package) -> str | None:
    """The distribution to install for ``pkg`` on this machine.

    ``None`` when the package does not apply here and must be skipped.
    Resolved one package at a time, immediately before installing it, because
    some answers depend on what an earlier package in the same tier installed.
    """
    if pkg.needs_gpu and pkg.needs_gpu not in _gpu_vendors():
        return None
    if pkg.import_name == "onnxruntime":
        return onnxruntime_distribution()
    if pkg.import_name == "tensorrt_libs":
        return tensorrt_distribution()
    return pkg.pip_name


def applies(pkg: Package) -> bool:
    """Whether ``pkg`` is relevant to this machine's hardware at all."""
    return not pkg.needs_gpu or pkg.needs_gpu in _gpu_vendors()


def tier_packages(name: str) -> list[Package]:
    """A tier's packages, less the ones this machine's hardware rules out.

    A TensorRT that will never load is not "not installed yet", it is not
    applicable — listing it as missing on an AMD machine would be an
    instruction to download 1.2 GB for nothing.
    """
    tier = TIERS.get(name)
    if not tier:
        return []
    return [pkg for pkg in tier["packages"] if applies(pkg)]


# --------------------------------------------------------------------------- #
# OpenCL, which pip cannot install
# --------------------------------------------------------------------------- #
# `pyopencl` is only the Python binding. The driver it talks to is the vendor's
# *ICD*, a system package, and a machine can have a working GPU, a working
# driver and pyopencl installed and still enumerate nothing —
# `PLATFORM_NOT_FOUND_KHR`, which reads like a hardware fault. Only Linux
# normally needs anything: Windows ships the ICD with the display driver and
# macOS ships it with the OS.
_ICD_PACKAGES = {
    "pacman": {"nvidia": "opencl-nvidia", "amd": "rocm-opencl-runtime",
               "intel": "intel-compute-runtime", None: "ocl-icd"},
    "apt": {"nvidia": "nvidia-opencl-icd", "amd": "mesa-opencl-icd",
            "intel": "intel-opencl-icd", None: "ocl-icd-libopencl1"},
    "dnf": {"nvidia": "xorg-x11-drv-nvidia-cuda",
            "amd": "mesa-libOpenCL", "intel": "intel-compute-runtime",
            None: "ocl-icd"},
    "zypper": {"nvidia": "nvidia-computeG06", "amd": "Mesa-libOpenCL",
               "intel": "intel-opencl", None: "ocl-icd-devel"},
}

_PACKAGE_MANAGERS = (("pacman", "/usr/bin/pacman", "sudo pacman -S"),
                     ("apt", "/usr/bin/apt", "sudo apt install"),
                     ("dnf", "/usr/bin/dnf", "sudo dnf install"),
                     ("zypper", "/usr/bin/zypper", "sudo zypper install"))


def opencl_icd_hint() -> str | None:
    """The command that registers an OpenCL driver here, if one is needed.

    Returns None on platforms where the ICD arrives with the driver or the OS,
    and where no package manager is recognised — a wrong command is worse than
    no command.
    """
    import platform as _platform
    if _platform.system() != "Linux":
        return None
    for name, probe, command in _PACKAGE_MANAGERS:
        if not os.path.exists(probe):
            continue
        table = _ICD_PACKAGES[name]
        vendors = _gpu_vendors()
        wanted = [table[key] for key in ("nvidia", "amd", "intel")
                  if key in vendors and table.get(key)]
        if not wanted:
            wanted = [table[None]]
        return f"{command} {' '.join(dict.fromkeys(wanted))}"
    return None


def _importable(module: str) -> bool:
    """True if the module can be imported, without actually importing it.

    ``find_spec`` avoids the cost and side effects of importing heavyweight
    packages merely to discover whether they exist.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def version_of(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
        return str(getattr(mod, "__version__", "") or "") or "installed"
    except Exception:
        return None


def all_packages() -> list[Package]:
    out: list[Package] = []
    for tier in TIERS.values():
        out.extend(tier["packages"])
    return out + HEAVY


def status() -> dict:
    """Which optional packages are present, grouped by tier."""
    result: dict = {"tiers": {}, "heavy": {}}
    for name, tier in TIERS.items():
        packages = tier_packages(name)
        entries = {pkg.pip_name: _importable(pkg.import_name)
                   for pkg in packages}
        result["tiers"][name] = {
            "summary": tier["summary"],
            "packages": entries,
            "complete": all(entries.values()),
            "usable": any(
                entries[p.pip_name] for p in packages if p.critical
            ) if any(p.critical for p in packages) else any(entries.values()),
        }
    for pkg in HEAVY:
        result["heavy"][pkg.pip_name] = _importable(pkg.import_name)
    return result


def missing(tier_names: list[str] | None = None) -> list[Package]:
    """Packages not yet installed, for the tiers requested (default: all)."""
    names = tier_names or list(TIERS)
    out = []
    for name in names:
        for pkg in tier_packages(name):
            if not _importable(pkg.import_name):
                out.append(pkg)
    return out


def have(module: str) -> bool:
    """Convenience check used by benchmark modules before importing."""
    return _importable(module)


def summary_line() -> str:
    """One-line description of which tiers are usable, for the report."""
    st = status()
    ready = [n for n, t in st["tiers"].items() if t["usable"]]
    if not ready:
        return ("no optional tiers installed — run 'python3 install.py' to "
                "unlock BLAS numerics, GPU compute, and hardware crypto")
    absent = [n for n in TIERS if n not in ready]
    text = "tiers active: " + ", ".join(ready)
    if absent:
        text += f"  (not installed: {', '.join(absent)})"
    return text
