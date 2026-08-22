"""Normalization of raw rates into comparable scores.

Each metric is divided by a fixed baseline so that **baseline == 100**, and the
subscores are combined with a geometric mean. The geometric mean is used rather
than an arithmetic one so no single category can dominate the composite: a
machine has to be well-rounded to score highly, and the result stays meaningful
even though subscores span very different magnitudes.

The baseline constants are arbitrary but *stable* — roughly a mid-range
2020-era laptop core. Their only job is to be a fixed yardstick, so changing
them invalidates comparisons against previously recorded runs.
"""

from __future__ import annotations

import math
import statistics

BASELINES = {
    "cpu_int": 2_000_000.0,        # primes/s, single core
    "cpu_float": 3_000_000.0,      # iters/s, single core
    "cpu_multi": 8_000_000.0,      # primes/s, all cores
    "compression": 60.0,           # MB/s zlib round-trip
    "hashing": 500.0,              # MB/s SHA-256
    "json": 80.0,                  # MB/s parse
    "memory": 6_000.0,             # MB/s copy
    "disk_write": 500.0,           # MB/s
    "disk_read": 1_000.0,          # MB/s
    "disk_iops": 20_000.0,         # 4 KiB random read ops/s
    # Accelerators. Absent hardware is simply omitted from the composite
    # rather than scored as zero, so a machine without a GPU or NPU is not
    # punished for lacking one.
    "gpu_fp32": 1_000.0,           # GFLOPS
    "gpu_fp16": 1_500.0,           # GFLOPS
    "gpu_bandwidth": 100_000.0,    # MB/s
    "gpu_matmul_fp32": 1.0,        # TFLOPS (dense GEMM — the AI-compute metric)
    "gpu_matmul_fp16": 2.0,        # TFLOPS
    "npu": 2_000.0,                # GFLOPS effective
    # AI framework tier (optional; only when torch/onnxruntime present).
    "ml_train": 500.0,             # training samples/s
    "ml_infer": 2_000.0,           # inference samples/s
    # Cross-vendor NPU via ONNX Runtime execution providers.
    "npu_onnx": 500.0,             # GFLOPS on the fastest engaged accelerator
    # Classic ML workloads, pure Python (no framework needed).
    "nn_training": 400.0,          # MLP training steps/s
    "kmeans": 1_000_000.0,         # point-centroid distances/s
    "knn": 1_000_000.0,            # neighbour comparisons/s
    # System-level and depth-aware measurements.
    "disk_iops_peak": 100_000.0,   # peak random-read IOPS across queue depths
    "mem_scaling": 20_000.0,       # peak multi-process copy bandwidth MB/s
    "compile": 300.0,              # compiles per minute
    "syscall": 20_000_000.0,       # syscalls/s (i.e. 50 ns each)
    # Optional-package tiers. Absent packages are omitted from the composite
    # rather than scored as zero.
    "blas_matmul": 100.0,          # GFLOPS fp64 via BLAS
    "fft": 5.0,                    # GFLOPS
    "lapack": 500.0,               # Cholesky decompositions/s
    "aes": 2_000.0,                # MB/s AES-256-GCM
    "zstd": 400.0,                 # MB/s Zstandard
    "lz4": 800.0,                  # MB/s LZ4
    "blake3": 1_000.0,             # MB/s BLAKE3
    "gpu_opencl": 1_000.0,         # GFLOPS via OpenCL
    # Application-shaped workloads. These are the scores that answer "will
    # this machine be good at my job?" rather than "how fast is this
    # subsystem?", so they carry the same weight as the synthetic ones.
    "sqlite": 50_000.0,            # SQLite OLTP transactions/s
    "raytrace": 150.0,             # ray-traced frames/s
    "image": 2.5,                  # megapixels/s through a separable blur
    "logparse": 80.0,              # MB/s of regex log parsing
    "video": 70.0,                 # H.264 1080p encode fps (needs ffmpeg)
}

# Deliberately *not* scored, though it is measured and reported:
#
# ``fsync`` — durable-commit latency is the most useful storage diagnostic
# there is, but the operation being timed is not the same across platforms.
# macOS needs ``F_FULLFSYNC`` to reach the medium where Linux's ``fsync``
# suffices, and the two differ by two orders of magnitude on identical
# hardware. Folding that into a cross-platform composite would make the score
# a measurement of the operating system's flush semantics.

# Which result key and field each score is derived from.
_SOURCES = [
    ("cpu_int", "cpu_int", "rate"),
    ("cpu_float", "cpu_float", "rate"),
    ("cpu_multi", "cpu_multi", "rate"),
    ("compression", "compression", "rate"),
    ("hashing", "hashing", "rate"),
    ("json", "json", "rate"),
    ("memory", "memory", "rate"),
    ("disk_write", "disk", "write_rate"),
    ("disk_read", "disk", "read_rate"),
    ("disk_iops", "disk", "random_read_iops"),
    ("gpu_fp32", "gpu_fp32", "rate"),
    ("gpu_fp16", "gpu_fp16", "rate"),
    ("gpu_bandwidth", "gpu_bandwidth", "rate"),
    ("gpu_matmul_fp32", "gpu_matmul_fp32", "rate"),
    ("gpu_matmul_fp16", "gpu_matmul_fp16", "rate"),
    ("npu", "npu", "rate"),
    ("ml_train", "ml_train", "rate"),
    ("ml_infer", "ml_infer", "rate"),
    ("npu_onnx", "npu_onnx", "rate"),
    ("nn_training", "nn_training", "rate"),
    ("kmeans", "kmeans", "rate"),
    ("knn", "knn", "rate"),
    ("disk_iops_peak", "disk", "peak_iops"),
    ("mem_scaling", "mem_scaling", "rate"),
    ("compile", "compile", "rate"),
    ("syscall", "latency", "rate"),
    ("blas_matmul", "blas_matmul", "rate"),
    ("fft", "fft", "rate"),
    ("lapack", "lapack", "rate"),
    ("aes", "aes", "rate"),
    ("zstd", "zstd", "rate"),
    ("lz4", "lz4", "rate"),
    ("blake3", "blake3", "rate"),
    ("gpu_opencl", "gpu_opencl", "rate"),
    ("sqlite", "sqlite", "rate"),
    ("raytrace", "raytrace", "rate"),
    ("image", "image", "rate"),
    ("logparse", "logparse", "rate"),
    ("video", "video", "rate"),
]


def compute_scores(results: dict) -> dict:
    """Return ``{"subscores": {...}, "composite": float}`` for a result set."""
    subscores: dict[str, float] = {}
    for score_key, result_key, field in _SOURCES:
        entry = results.get(result_key)
        if not isinstance(entry, dict) or entry.get("skipped"):
            continue
        rate = entry.get(field)
        base = BASELINES[score_key]
        if isinstance(rate, (int, float)) and rate > 0 and base > 0:
            subscores[score_key] = 100.0 * rate / base

    composite = 0.0
    if subscores:
        composite = math.exp(
            statistics.fmean(math.log(v) for v in subscores.values()))
    return {
        "subscores": {k: round(v, 1) for k, v in subscores.items()},
        "composite": round(composite, 1),
    }


def category_scores(subscores: dict) -> dict:
    """Roll subscores up into CPU / memory / disk headline numbers."""
    groups = {
        "cpu": ["cpu_int", "cpu_float", "cpu_multi", "compression",
                "hashing", "json"],
        "numeric": ["blas_matmul", "fft", "lapack"],
        "crypto": ["aes", "zstd", "lz4", "blake3"],
        "memory": ["memory", "mem_scaling"],
        "disk": ["disk_write", "disk_read", "disk_iops", "disk_iops_peak"],
        "system": ["compile", "syscall"],
        "apps": ["sqlite", "raytrace", "image", "logparse", "video"],
        "gpu": ["gpu_fp32", "gpu_fp16", "gpu_bandwidth",
                "gpu_matmul_fp32", "gpu_matmul_fp16", "gpu_opencl"],
        "npu": ["npu", "npu_onnx"],
        "ml": ["nn_training", "kmeans", "knn"],
        "ai": ["gpu_matmul_fp32", "gpu_matmul_fp16", "npu", "npu_onnx",
               "ml_train", "ml_infer", "nn_training", "kmeans", "knn"],
    }
    out = {}
    for name, keys in groups.items():
        vals = [subscores[k] for k in keys if k in subscores and subscores[k] > 0]
        if vals:
            out[name] = round(
                math.exp(statistics.fmean(math.log(v) for v in vals)), 1)
    return out
