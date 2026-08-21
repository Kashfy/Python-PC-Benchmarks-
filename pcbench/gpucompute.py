"""Cross-platform GPU compute via OpenCL, plus NVIDIA telemetry.

GPU benchmarking has been Apple-only, because Metal is the only GPU API the
tool could reach without a vendor SDK. OpenCL closes that: it is implemented by
NVIDIA, AMD, Intel, and Apple, so one set of kernels measures any of them.

Two measurements, mirroring the Metal engine so numbers are comparable:

* **FMA throughput** — long chains of fused multiply-adds, reported in GFLOPS.
* **Memory bandwidth** — a large device-memory copy, reported in MB/s.

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


def devices() -> list[dict]:
    """Enumerate OpenCL devices without benchmarking them."""
    if not have("pyopencl"):
        return []
    try:
        import pyopencl as cl
        found = []
        for platform in cl.get_platforms():
            for dev in platform.get_devices():
                found.append({
                    "name": dev.name.strip(),
                    "platform": platform.name.strip(),
                    "type": cl.device_type.to_string(dev.type),
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

    result: dict = {
        "name": dev.name.strip(),
        "type": cl.device_type.to_string(dev.type),
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
        return {"available": False,
                "note": "pyopencl not installed — run 'python3 install.py "
                        "--tier gpu' for cross-platform GPU benchmarking",
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

    usable = [r for r in results if r.get("fp32_gflops")]
    best = max(usable, key=lambda r: r["fp32_gflops"]) if usable else None
    return {
        "available": True,
        "devices": results,
        "best_device": best["name"] if best else None,
        "best_gflops": best["fp32_gflops"] if best else None,
        "nvidia": nvidia_telemetry(),
    }


def extract_rates(payload: dict | None) -> dict:
    if not payload or not payload.get("available"):
        return {}
    if payload.get("best_gflops"):
        return {"gpu_opencl": float(payload["best_gflops"])}
    return {}
