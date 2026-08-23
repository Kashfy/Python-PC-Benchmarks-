"""GPU compute on NVIDIA, AMD, Intel and Apple — integrated and discrete.

Two backends, because they measure genuinely different things and a modern GPU
needs both:

* **OpenCL** (via ``pyopencl``) — implemented by every vendor, so one set of
  kernels runs anywhere. Measures raw **FMA throughput** in GFLOPS and
  **memory bandwidth** in MB/s. This reaches the shader cores and nothing else.
* **PyTorch** (via CUDA, ROCm or XPU) — measures **dense matrix multiply** in
  TFLOPS, which is what AI workloads actually run. On hardware with tensor
  cores this is several times the FP32 shader figure and OpenCL cannot reach
  it at all, so a report with only the OpenCL number understates an RTX or
  Instinct card badly for the one workload people buy them for.

**Discrete GPUs are preferred over integrated ones** when a machine has both.
Selecting purely by measured throughput picks the discrete card almost always,
and "almost always" is the problem: a discrete card throttled by a power
setting or falling back to a slow driver path would silently lose to the iGPU
and the report would describe the wrong hardware. Classification comes from
OpenCL's ``host_unified_memory`` where the driver reports it.

``pynvml`` additionally exposes NVIDIA temperature, power draw, VRAM, and
utilisation, which no portable API provides.
"""

from __future__ import annotations

from .core import clock
from .optional import have

# Kernel source. Four independent dependent chains give the scheduler
# instruction-level parallelism while preventing the compiler from collapsing
# the loop; the result is written out so nothing is dead code.
_KERNEL = """
__kernel void fma_bench(__global float *out, const unsigned int iters) {
    int gid = get_global_id(0);
    float a = out[gid], b = 1.0000001f, c = 0.0000001f;
    float a2 = a + 1.0f, a3 = a + 2.0f, a4 = a + 3.0f;
    for (unsigned int k = 0; k < iters; ++k) {
        a  = fma(a,  b, c); a2 = fma(a2, b, c);
        a3 = fma(a3, b, c); a4 = fma(a4, b, c);
    }
    out[gid] = a + a2 + a3 + a4;
}

__kernel void bandwidth(__global const float4 *src, __global float4 *dst) {
    int gid = get_global_id(0);
    dst[gid] = src[gid];
}
"""

THREADS = 1 << 20          # 1 Mi work items
INNER_ITERS = 2048         # FMA loop trips per work item
COPY_MB = 128


def available() -> dict:
    return {"pyopencl": have("pyopencl"), "pynvml": have("pynvml")}


# --------------------------------------------------------------------------- #
# Device classification
# --------------------------------------------------------------------------- #
# A machine with both an integrated and a discrete GPU must be measured on the
# discrete one. Selecting purely by measured throughput almost always picks it,
# but "almost always" is the problem: a discrete card whose driver falls back
# to a slow path, or which errors during one kernel, silently loses to the iGPU
# and the report then describes the wrong hardware without saying so.
#
# OpenCL exposes the fact directly. `host_unified_memory` is true when the
# device shares system RAM, which is what integrated means; vendor names are
# used only to fill in where a driver does not report it.

_INTEGRATED_HINTS = ("uhd graphics", "hd graphics", "iris", "vega ", "radeon(tm) graphics",
                     "radeon graphics", "integrated", "igpu", "apple m")
_DISCRETE_HINTS = ("geforce", "rtx", "gtx", "quadro", "tesla", "titan",
                   "radeon rx", "radeon pro", "instinct", "arc ")

VENDORS = {
    "nvidia": ("nvidia", "geforce", "quadro", "tesla", "titan"),
    "amd": ("amd", "advanced micro", "radeon", "instinct", "gfx"),
    "intel": ("intel", "iris", "uhd graphics", "hd graphics", "arc "),
    "apple": ("apple",),
}


def vendor_of(name: str, platform_name: str = "") -> str:
    blob = f"{name} {platform_name}".lower()
    for vendor, needles in VENDORS.items():
        if any(n in blob for n in needles):
            return vendor
    return "unknown"


def classify(name: str, unified_memory: bool | None,
             device_type: str = "GPU") -> str:
    """Return "discrete", "integrated", or "unknown" for a GPU.

    ``unified_memory`` is authoritative where the driver reports it: a device
    sharing host memory is integrated by definition. Apple silicon is the one
    case where unified memory does not mean weak, so it is classified from the
    name instead and treated as discrete-class for selection purposes.
    """
    lowered = (name or "").lower()
    if "gpu" not in (device_type or "gpu").lower():
        return "unknown"

    if any(h in lowered for h in _DISCRETE_HINTS):
        return "discrete"
    if "apple" in lowered:
        # Unified memory, but it is the only GPU present and is the fast one.
        return "discrete"
    if unified_memory is True:
        return "integrated"
    if any(h in lowered for h in _INTEGRATED_HINTS):
        return "integrated"
    if unified_memory is False:
        return "discrete"
    return "unknown"


def select_best(results: list[dict]) -> dict | None:
    """Choose the device to report and score.

    Discrete beats integrated even when the integrated device measured faster,
    because the discrete card is what the machine is expected to use and a
    reversal is a finding rather than a reason to switch. Within a class the
    fastest wins.
    """
    usable = [r for r in results if r.get("fp32_gflops")]
    if not usable:
        return None
    rank = {"discrete": 0, "unknown": 1, "integrated": 2}
    return min(usable, key=lambda r: (rank.get(r.get("class"), 1),
                                      -r["fp32_gflops"]))


def selection_note(results: list[dict], chosen: dict | None) -> str | None:
    """Explain the choice when more than one GPU could have been picked."""
    usable = [r for r in results if r.get("fp32_gflops")]
    if not chosen or len(usable) < 2:
        return None

    others = [r for r in usable if r is not chosen]
    fastest = max(usable, key=lambda r: r["fp32_gflops"])
    note = (f"{len(usable)} GPUs measured; scoring the "
            f"{chosen.get('class', 'selected')} {chosen['name']}")
    if fastest is not chosen:
        note += (f". Note that the {fastest.get('class', 'other')} "
                 f"{fastest['name']} measured higher "
                 f"({fastest['fp32_gflops']:,.0f} vs "
                 f"{chosen['fp32_gflops']:,.0f} GFLOPS) — on a machine with a "
                 f"discrete card that usually means a driver or power setting "
                 f"is holding it back")
    elif others:
        note += (f" (fastest of {len(usable)}; also measured: "
                 + ", ".join(f"{o['name']} at {o['fp32_gflops']:,.0f} GFLOPS"
                             for o in others[:3]) + ")")
    return note


def devices() -> list[dict]:
    """Enumerate OpenCL devices without benchmarking them."""
    if not have("pyopencl"):
        return []
    try:
        import pyopencl as cl
        found = []
        for platform in cl.get_platforms():
            for dev in platform.get_devices():
                dev_type = cl.device_type.to_string(dev.type)
                unified = getattr(dev, "host_unified_memory", None)
                found.append({
                    "name": dev.name.strip(),
                    "platform": platform.name.strip(),
                    "type": dev_type,
                    "vendor": vendor_of(dev.name, platform.name),
                    "class": classify(dev.name, bool(unified)
                                      if unified is not None else None,
                                      dev_type),
                    "compute_units": dev.max_compute_units,
                    "global_mem_mb": round(dev.global_mem_size / (1024 ** 2)),
                    "max_clock_mhz": dev.max_clock_frequency,
                })
        return found
    except Exception:
        return []


def _bench_device(dev, seconds: float) -> dict:
    """FMA throughput and copy bandwidth for one OpenCL device."""
    import numpy as np
    import pyopencl as cl

    ctx = cl.Context(devices=[dev])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, _KERNEL).build()
    mf = cl.mem_flags

    dev_type = cl.device_type.to_string(dev.type)
    unified = getattr(dev, "host_unified_memory", None)
    result: dict = {
        "name": dev.name.strip(),
        "type": dev_type,
        "vendor": vendor_of(dev.name),
        "class": classify(dev.name,
                          bool(unified) if unified is not None else None,
                          dev_type),
        "global_mem_mb": round(dev.global_mem_size / (1024 ** 2)),
        "compute_units": dev.max_compute_units,
    }

    # ---- FMA throughput ----
    host = np.zeros(THREADS, dtype=np.float32)
    buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host)
    kernel = program.fma_bench
    kernel.set_args(buf, np.uint32(INNER_ITERS))

    cl.enqueue_nd_range_kernel(queue, kernel, (THREADS,), None).wait()
    start, n = clock(), 0
    while clock() - start < seconds:
        cl.enqueue_nd_range_kernel(queue, kernel, (THREADS,), None).wait()
        n += 1
    elapsed = clock() - start
    # 4 chains x 2 FLOPs per fma x inner iterations x work items
    flops = n * THREADS * INNER_ITERS * 4.0 * 2.0
    result["fp32_gflops"] = round(flops / elapsed / 1e9, 1)

    # ---- Memory bandwidth ----
    try:
        n_float4 = (COPY_MB * 1024 * 1024) // 16
        src = cl.Buffer(ctx, mf.READ_ONLY, size=n_float4 * 16)
        dst = cl.Buffer(ctx, mf.WRITE_ONLY, size=n_float4 * 16)
        bw = program.bandwidth
        bw.set_args(src, dst)
        cl.enqueue_nd_range_kernel(queue, bw, (n_float4,), None).wait()
        start, n = clock(), 0
        while clock() - start < seconds:
            cl.enqueue_nd_range_kernel(queue, bw, (n_float4,), None).wait()
            n += 1
        elapsed = clock() - start
        # One read plus one write per element.
        moved = n * n_float4 * 16 * 2.0
        result["bandwidth_mb_s"] = round(moved / elapsed / (1024 ** 2), 1)
    except Exception as e:
        result["bandwidth_error"] = f"{type(e).__name__}: {e}"

    return result


def nvidia_telemetry() -> list[dict]:
    """Per-GPU NVIDIA temperature, power, memory and utilisation."""
    if not have("pynvml"):
        return []
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []
    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            entry: dict = {"index": i}

            def attempt(key, fn, scale=1.0, ndigits=1):
                # Each metric is optional: consumer cards omit several of
                # these, and a missing one must not lose the others.
                try:
                    entry[key] = round(fn() * scale, ndigits)
                except Exception:
                    pass

            try:
                name = pynvml.nvmlDeviceGetName(h)
                entry["name"] = (name.decode() if isinstance(name, bytes)
                                 else str(name))
            except Exception:
                pass
            attempt("celsius", lambda: pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU))
            attempt("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(h),
                    0.001, 1)
            attempt("fan_percent", lambda: pynvml.nvmlDeviceGetFanSpeed(h))
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                entry["vram_total_mb"] = round(mem.total / (1024 ** 2))
                entry["vram_used_mb"] = round(mem.used / (1024 ** 2))
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                entry["gpu_percent"] = util.gpu
                entry["memory_percent"] = util.memory
            except Exception:
                pass
            out.append(entry)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def run(seconds: float = 1.0) -> dict:
    """Benchmark every OpenCL device and collect NVIDIA telemetry."""
    avail = available()
    if not avail["pyopencl"]:
        # PyTorch alone can still measure a CUDA/ROCm/XPU device, so the
        # absence of OpenCL is not the end of GPU benchmarking.
        return {"available": False,
                "note": "pyopencl not installed — run 'python3 install.py "
                        "--tier gpu' for cross-platform GPU benchmarking",
                "matmul": torch_matmul(min(seconds, 2.0)),
                "nvidia": nvidia_telemetry()}
    try:
        import pyopencl as cl
    except Exception as e:
        return {"available": False, "error": f"pyopencl import failed: {e}"}

    results = []
    try:
        for platform in cl.get_platforms():
            for dev in platform.get_devices():
                try:
                    entry = _bench_device(dev, seconds)
                    entry["platform"] = platform.name.strip()
                    results.append(entry)
                except Exception as e:
                    results.append({"name": getattr(dev, "name", "?").strip(),
                                    "error": f"{type(e).__name__}: {e}"})
    except Exception as e:
        return {"available": False, "error": f"OpenCL enumeration failed: {e}"}

    for entry in results:
        entry["vendor"] = entry.get("vendor") or vendor_of(
            entry.get("name", ""), entry.get("platform", ""))
    best = select_best(results)
    return {
        "available": True,
        "devices": results,
        "best_device": best["name"] if best else None,
        "best_gflops": best["fp32_gflops"] if best else None,
        "best_class": best.get("class") if best else None,
        "best_vendor": best.get("vendor") if best else None,
        "selection_note": selection_note(results, best),
        "matmul": torch_matmul(min(seconds, 2.0)),
        "nvidia": nvidia_telemetry(),
    }


# --------------------------------------------------------------------------- #
# PyTorch backend — dense matmul, the AI-compute metric
# --------------------------------------------------------------------------- #
#: Matrix order for the GEMM. Large enough to saturate a modern GPU and small
#: enough to fit comfortably in the memory of a modest one.
_MATMUL_N = 4096


def torch_device() -> tuple[str, str] | None:
    """Best PyTorch GPU device available, as ``(device, vendor)``.

    ROCm builds of PyTorch expose AMD hardware through the ``cuda`` device, so
    the vendor is read from the device name rather than the API used.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            vendor = "amd" if ("radeon" in name or "instinct" in name
                               or getattr(torch.version, "hip", None)) else "nvidia"
            return "cuda", vendor
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu", "intel"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "apple"
    except Exception:
        return None
    return None


def torch_matmul(seconds: float = 2.0) -> dict:
    """Dense GEMM throughput in TFLOPS, at fp32 and fp16.

    A matrix multiply of order N costs 2*N^3 floating-point operations. Timing
    is taken after an explicit device barrier, because GPU work is asynchronous
    and without one this would measure how fast Python can enqueue kernels.
    """
    selected = torch_device()
    if selected is None:
        return {"skipped": True,
                "reason": "no PyTorch GPU device (needs a CUDA, ROCm or XPU "
                          "build of torch)",
                "hint": "pip install torch"}
    device, vendor = selected

    try:
        import torch
    except Exception as e:
        return {"skipped": True, "reason": f"torch import failed: {e}"}

    dev = torch.device(device)

    def sync() -> None:
        getattr(torch, device).synchronize()

    def measure(dtype) -> float | None:
        try:
            a = torch.randn(_MATMUL_N, _MATMUL_N, device=dev, dtype=dtype)
            b = torch.randn(_MATMUL_N, _MATMUL_N, device=dev, dtype=dtype)
            for _ in range(2):            # warm up kernel selection
                a @ b
            sync()
            start, iterations = clock(), 0
            while clock() - start < seconds:
                a @ b
                iterations += 1
            sync()
            elapsed = clock() - start
            del a, b
            if not elapsed or not iterations:
                return None
            flops = 2.0 * _MATMUL_N ** 3 * iterations
            return flops / elapsed / 1e12
        except Exception:
            return None

    result: dict = {"device": device, "vendor": vendor, "n": _MATMUL_N}
    try:
        result["name"] = (torch.cuda.get_device_name(0) if device == "cuda"
                          else device.upper())
    except Exception:
        result["name"] = device.upper()

    with torch.inference_mode():
        fp32 = measure(torch.float32)
        fp16 = measure(torch.float16)
    sync_error = fp32 is None and fp16 is None
    if sync_error:
        return {"skipped": True,
                "reason": f"matmul failed on the {device} device"}

    if fp32:
        result["matmul_fp32_tflops"] = round(fp32, 3)
    if fp16:
        result["matmul_fp16_tflops"] = round(fp16, 3)
    if fp32 and fp16:
        result["fp16_speedup"] = round(fp16 / fp32, 2)
        if fp16 > fp32 * 2.5:
            result["note"] = ("fp16 is more than 2.5x fp32, which means "
                              "dedicated matrix hardware (tensor cores or "
                              "equivalent) is being used")
    return result


def extract_rates(payload: dict | None) -> dict:
    """Scoreable rates from both backends.

    The matmul figures use the same score keys as the Metal engine, so an RTX
    card and an Apple GPU land in the same category and are directly
    comparable.
    """
    if not payload:
        return {}
    out: dict = {}
    if payload.get("available") and payload.get("best_gflops"):
        out["gpu_opencl"] = float(payload["best_gflops"])

    matmul = payload.get("matmul") or {}
    if not matmul.get("skipped"):
        if matmul.get("matmul_fp32_tflops"):
            out["gpu_matmul_fp32"] = float(matmul["matmul_fp32_tflops"])
        if matmul.get("matmul_fp16_tflops"):
            out["gpu_matmul_fp16"] = float(matmul["matmul_fp16_tflops"])
    return out
