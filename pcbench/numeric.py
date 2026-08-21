"""BLAS/LAPACK numeric benchmarks — CPU performance the interpreter cannot show.

The pure-Python workloads in :mod:`pcbench.mlbench` are comparable across
machines but measure CPython, not silicon: on an Apple M4 they reach roughly
113 MFLOPS while the same chip sustains hundreds of GFLOPS through an optimised
BLAS. These benchmarks call into whatever BLAS the platform ships — Accelerate
on macOS, OpenBLAS or MKL elsewhere — so the result reflects the hardware and
is directly comparable to how real numeric software performs.

Requires ``numpy``; LAPACK decompositions additionally require ``scipy``.
Absent either, the corresponding section reports itself unavailable and the
run continues.
"""

from __future__ import annotations

import math

from .core import clock, summarize
from .optional import have

# Matrix sizes chosen to exceed cache so the result measures sustained
# throughput rather than a cache-resident burst.
MATMUL_N = 1536
FFT_N = 1 << 21          # 2 Mi complex samples
LAPACK_N = 768


def available() -> dict:
    return {"numpy": have("numpy"), "scipy": have("scipy"),
            "numba": have("numba")}


def _timed(fn, seconds: float, min_iters: int = 1) -> tuple[float, int]:
    """Run ``fn`` until ``seconds`` elapse; returns (elapsed, iterations)."""
    start = clock()
    n = 0
    while True:
        fn()
        n += 1
        elapsed = clock() - start
        if elapsed >= seconds and n >= min_iters:
            return elapsed, n


def bench_matmul(seconds: float = 1.0, repeats: int = 2) -> dict:
    """Dense matrix multiply through BLAS, in FP64 and FP32.

    A dense N x N multiply is 2*N^3 floating-point operations. This is the
    single most representative CPU number for scientific and AI work, since
    almost every heavy numeric kernel reduces to it.
    """
    if not have("numpy"):
        return {"skipped": True, "error": "numpy not installed"}
    import numpy as np

    out: dict = {"unit": "GFLOPS", "matrix_n": MATMUL_N}
    flops = 2.0 * MATMUL_N ** 3

    for label, dtype in (("fp64", np.float64), ("fp32", np.float32)):
        rng = np.random.default_rng(0)
        a = rng.random((MATMUL_N, MATMUL_N), dtype=np.float32).astype(dtype)
        b = rng.random((MATMUL_N, MATMUL_N), dtype=np.float32).astype(dtype)
        a @ b                                    # warm up / thread spin-up
        rates = []
        for _ in range(max(1, repeats)):
            elapsed, n = _timed(lambda: a @ b, seconds)
            rates.append(n * flops / elapsed / 1e9)
        s = summarize(rates)
        out[label] = round(s["median"], 1)

    # The headline rate is FP64: it is what BLAS benchmarks conventionally
    # report, and it is the harder number.
    out["rate"] = out["fp64"]
    try:
        import numpy as _np
        cfg = _np.__config__.show(mode="dicts")  # numpy >= 2
        blas = (cfg.get("Build Dependencies", {})
                   .get("blas", {}).get("name"))
        if blas:
            out["blas"] = blas
    except Exception:
        pass
    return out


def bench_fft(seconds: float = 1.0, repeats: int = 2) -> dict:
    """Fast Fourier transform throughput.

    An N-point complex FFT costs about 5*N*log2(N) floating-point operations.
    FFTs are memory-access-bound rather than arithmetic-bound, so this probes a
    different limit from matrix multiply.
    """
    if not have("numpy"):
        return {"skipped": True, "error": "numpy not installed"}
    import numpy as np

    rng = np.random.default_rng(1)
    data = rng.random(FFT_N) + 1j * rng.random(FFT_N)
    flops = 5.0 * FFT_N * math.log2(FFT_N)

    np.fft.fft(data)                              # warm up plan caches
    rates = []
    for _ in range(max(1, repeats)):
        elapsed, n = _timed(lambda: np.fft.fft(data), seconds)
        rates.append(n * flops / elapsed / 1e9)
    s = summarize(rates)
    return {"unit": "GFLOPS", "rate": round(s["median"], 2),
            "points": FFT_N, "cv": s["cv"]}


def bench_lapack(seconds: float = 1.0) -> dict:
    """LAPACK decompositions: SVD, Cholesky, and eigenvalues.

    These are the operations behind regression, PCA, and simulation. They stress
    the BLAS differently from a plain multiply — more dependent operations and
    less regular memory access — so a machine can be fast at one and mediocre
    at the other.
    """
    if not have("scipy"):
        return {"skipped": True, "error": "scipy not installed"}
    import numpy as np
    import scipy.linalg as sla

    rng = np.random.default_rng(2)
    a = rng.random((LAPACK_N, LAPACK_N))
    # A symmetric positive-definite matrix is required for Cholesky and makes
    # the eigenvalue problem well-conditioned.
    spd = a @ a.T + LAPACK_N * np.eye(LAPACK_N)

    ops = {
        "cholesky_per_s": lambda: sla.cholesky(spd),
        "svd_per_s": lambda: sla.svd(a, compute_uv=False),
        "eigenvalues_per_s": lambda: sla.eigh(spd, eigvals_only=True),
    }
    out: dict = {"unit": "ops/s", "matrix_n": LAPACK_N}
    budget = max(0.2, seconds / len(ops))
    for name, fn in ops.items():
        fn()                                      # warm up
        elapsed, n = _timed(fn, budget)
        out[name] = round(n / elapsed, 2)
    out["rate"] = out["cholesky_per_s"]
    return out


def run(seconds: float = 1.0, repeats: int = 2) -> dict:
    """All available numeric benchmarks."""
    avail = available()
    if not avail["numpy"]:
        return {"available": False,
                "note": "numpy not installed — run 'python3 install.py "
                        "--tier compute' for real BLAS numbers"}
    import numpy as np
    result: dict = {
        "available": True,
        "numpy_version": np.__version__,
        "matmul": bench_matmul(seconds, repeats),
        "fft": bench_fft(seconds, repeats),
    }
    if avail["scipy"]:
        result["lapack"] = bench_lapack(seconds)
    return result


def extract_rates(payload: dict | None) -> dict:
    """Headline numeric rates, for scoring."""
    if not payload or not payload.get("available"):
        return {}
    rates = {}
    mm = payload.get("matmul") or {}
    if isinstance(mm.get("rate"), (int, float)) and mm["rate"] > 0:
        rates["blas_matmul"] = float(mm["rate"])
    fft = payload.get("fft") or {}
    if isinstance(fft.get("rate"), (int, float)) and fft["rate"] > 0:
        rates["fft"] = float(fft["rate"])
    lp = payload.get("lapack") or {}
    if isinstance(lp.get("rate"), (int, float)) and lp["rate"] > 0:
        rates["lapack"] = float(lp["rate"])
    return rates
