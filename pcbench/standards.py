"""Reference workloads whose numbers mean something outside this tool.

Every benchmark in this project so far is internally consistent and externally
meaningless: nobody has published a figure for ``raytrace`` or ``logparse``, so
a result can only be compared against another run of the same tool. That is
enough for regression detection and useless for the question an engineer
actually asks — "is 96 GB/s good for this hardware?"

The three workloads here answer it, because the whole industry already speaks
them:

* **STREAM** (McCalpin) — the memory-bandwidth standard. Vendors publish STREAM
  Triad figures, HPC procurement is written against them, and a number here is
  directly comparable to those. Implemented in the native C engine, where it
  belongs: the reference implementation is C, and a Python version would
  measure the interpreter.
* **LINPACK / HPL** — the dense linear-algebra standard, and the metric that
  ranks the TOP500. Solves ``Ax = b`` by LU decomposition with partial
  pivoting and reports GFLOPS against the standard ``2/3 N^3 + 2 N^2``
  operation count. Needs an optimised BLAS through NumPy; a hand-written LU
  would report a tenth of the machine's real capability and mean nothing.
* **CoreMark-style** — the embedded integer reference. Also in the native
  engine. Reported under a deliberately distinct name because it is a
  reimplementation of the kernel mix rather than the certified benchmark; see
  :func:`coremark_caveat`.

A word on why the honesty matters more here than elsewhere. The value of a
standard is that its numbers are comparable; publishing an approximation under
the standard's name destroys exactly the property that made it worth
implementing. So LINPACK reports its residual (as HPL requires), STREAM
validates its arrays and reports the array size so the cache rule can be
checked, and the CoreMark-style figure is never called a CoreMark score.
"""

from __future__ import annotations

import math

from .core import clock

#: HPL's verification threshold on the scaled residual. A solve that exceeds
#: it has not actually solved the system, whatever GFLOPS it claimed.
RESIDUAL_TOLERANCE = 16.0


def available() -> dict:
    """Which standard workloads can run here, and why not when they cannot."""
    status = {"stream": True, "coremark_style": True, "linpack": False}
    try:
        import numpy  # noqa: F401
        status["linpack"] = True
    except Exception:
        status["linpack_reason"] = (
            "LINPACK needs NumPy with an optimised BLAS; without one the "
            "result would measure Python, not the CPU")
    status["stream_reason"] = ("STREAM and CoreMark-style run in the native C "
                               "engine and need a C compiler")
    return status


# --------------------------------------------------------------------------- #
# LINPACK / HPL
# --------------------------------------------------------------------------- #
def linpack_size(ram_bytes: int = 0, max_bytes: int = 512 * 1024 ** 2) -> int:
    """Choose a matrix order N for the available memory.

    HPL efficiency rises with N because the O(N^3) arithmetic increasingly
    dominates the O(N^2) memory traffic — which is why TOP500 submissions use
    matrices that fill most of a machine's RAM. That is not appropriate for a
    diagnostic that must finish promptly and must not push the machine into
    swap, so N is capped and the resulting figure is honestly below the peak a
    tuned HPL run would reach.
    """
    budget = max_bytes
    if ram_bytes:
        budget = min(budget, int(ram_bytes * 0.06))
    n = int(math.sqrt(budget / 8.0))
    # Round down to a multiple of 64: BLAS kernels are blocked, and a ragged
    # trailing block costs efficiency for no measurement benefit.
    n = max(256, (n // 64) * 64)
    return n


def linpack(ram_bytes: int = 0, repeats: int = 1) -> dict:
    """Solve a dense system and report GFLOPS with HPL's operation count."""
    try:
        import numpy as np
    except Exception:
        return {"skipped": True,
                "reason": "NumPy is not installed",
                "hint": "pip install numpy (or run install.py) to enable "
                        "LINPACK"}

    n = linpack_size(ram_bytes)
    rng = np.random.default_rng(20240101)      # fixed seed: identical work
    try:
        a = rng.random((n, n), dtype=np.float64)
        # Diagonal dominance keeps the system well conditioned, so a large
        # residual indicates a computation fault rather than a hard problem.
        a[np.diag_indices(n)] += n
        x_true = np.ones(n, dtype=np.float64)
        b = a @ x_true
    except MemoryError:
        return {"skipped": True,
                "reason": f"not enough memory for an N={n} matrix"}

    # HPL's operation count for LU with partial pivoting plus the solve.
    flops = (2.0 / 3.0) * n ** 3 + 2.0 * n ** 2

    rates = []
    residual = None
    for _ in range(max(1, repeats)):
        start = clock()
        x = np.linalg.solve(a, b)
        elapsed = clock() - start
        if elapsed <= 0:
            continue
        rates.append(flops / elapsed / 1e9)

        # HPL's scaled residual. Anything above the tolerance means the answer
        # is wrong, and a wrong answer at high speed is not a fast machine.
        if residual is None:
            r = b - a @ x
            denom = (np.linalg.norm(a, 1) * np.linalg.norm(x, 1)
                     * np.finfo(np.float64).eps * n)
            residual = float(np.linalg.norm(r, 1) / denom) if denom else 0.0

    if not rates:
        return {"skipped": True, "reason": "solve completed too fast to time"}

    rates.sort()
    rate = rates[len(rates) // 2]
    passed = residual is not None and residual < RESIDUAL_TOLERANCE

    result = {
        "unit": "GFLOPS",
        "rate": rate,
        "n": n,
        "matrix_mb": round(n * n * 8 / (1024 ** 2), 1),
        "residual": round(residual, 4) if residual is not None else None,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "validated": passed,
        "samples": [round(r, 3) for r in rates],
        "note": (f"LU with partial pivoting on an N={n} system, "
                 f"HPL operation count (2/3 N^3 + 2 N^2)"),
    }
    if not passed:
        result["error"] = (
            f"residual {residual:.2f} exceeds HPL's tolerance of "
            f"{RESIDUAL_TOLERANCE} — the system was not correctly solved, "
            f"which indicates a numerical or hardware fault")
        result["validation_failed"] = True
    result["caveat"] = (
        f"N={n} is capped to keep the run short and clear of swap; a tuned HPL "
        f"run with a matrix filling most of RAM would report a higher figure "
        f"on the same hardware")
    return result


# --------------------------------------------------------------------------- #
# Extraction from the native engine
# --------------------------------------------------------------------------- #
def coremark_caveat() -> str:
    return ("kernel mix matches EEMBC CoreMark (list, matrix, state machine, "
            "CRC) but this is not the certified benchmark: published CoreMark "
            "scores come from EEMBC's exact source under fixed reporting "
            "rules. Use it to compare cores against each other, not against "
            "published CoreMark numbers.")


def from_native(native: dict | None) -> dict:
    """Pull the STREAM and CoreMark-style results out of the native payload."""
    if not isinstance(native, dict) or native.get("error"):
        reason = ((native or {}).get("error")
                  or "the native C engine did not run")
        return {"stream": {"skipped": True, "reason": reason},
                "coremark_style": {"skipped": True, "reason": reason}}

    out: dict = {}

    stream = native.get("stream")
    if isinstance(stream, dict) and stream.get("triad"):
        out["stream"] = dict(stream)
        out["stream"]["cache_rule"] = _stream_cache_note(
            stream.get("array_bytes", 0))
        if not stream.get("validated", True):
            out["stream"]["validation_failed"] = True
            out["stream"]["error"] = (
                "STREAM arrays did not contain the expected values — either "
                "the compiler elided the kernels or the machine computed them "
                "incorrectly")
    else:
        out["stream"] = {"skipped": True,
                         "reason": "the native engine reported no STREAM data"}

    cm = native.get("coremark_style")
    if isinstance(cm, dict) and cm.get("rate"):
        out["coremark_style"] = dict(cm)
        out["coremark_style"]["caveat"] = coremark_caveat()
    else:
        out["coremark_style"] = {
            "skipped": True,
            "reason": "the native engine reported no CoreMark-style data"}
    return out


def _stream_cache_note(array_bytes: int) -> str:
    """STREAM requires each array to be ~4x the last-level cache."""
    if not array_bytes:
        return "array size unknown"
    mb = array_bytes / 1e6
    return (f"{mb:.0f} MB per array; STREAM requires roughly 4x the "
            f"last-level cache, so this is valid for caches up to about "
            f"{mb / 4:.0f} MB. A machine with a larger cache than that needs "
            f"--stream-mb raised, or the figure includes cache bandwidth.")


# --------------------------------------------------------------------------- #
# Scoring and reporting
# --------------------------------------------------------------------------- #
def extract_rates(result: dict | None) -> dict:
    """Scoreable rates from the standards section."""
    if not result:
        return {}
    out = {}
    stream = result.get("stream") or {}
    if not stream.get("skipped") and stream.get("triad"):
        out["stream_triad"] = float(stream["triad"])
    cm = result.get("coremark_style") or {}
    if not cm.get("skipped") and cm.get("rate"):
        out["coremark_style"] = float(cm["rate"])
    lin = result.get("linpack") or {}
    if not lin.get("skipped") and lin.get("rate") and lin.get("validated"):
        out["linpack"] = float(lin["rate"])
    return out


def render(result: dict | None) -> str:
    """Terminal block for the standards section."""
    if not result:
        return ""
    lines: list[str] = []

    stream = result.get("stream") or {}
    if stream.get("skipped"):
        lines.append(f"  STREAM              : skipped — {stream['reason']}")
    else:
        lines.append(f"  STREAM Copy         : {stream.get('copy', 0):>12,.1f} MB/s")
        lines.append(f"  STREAM Scale        : {stream.get('scale', 0):>12,.1f} MB/s")
        lines.append(f"  STREAM Add          : {stream.get('add', 0):>12,.1f} MB/s")
        lines.append(f"  STREAM Triad        : {stream.get('triad', 0):>12,.1f} MB/s"
                     f"   <- the quoted figure")
        lines.append(f"      {stream.get('cache_rule', '')}")
        lines.append(f"      MB/s on STREAM's 1e6-byte convention, directly "
                     f"comparable to published results")
        if stream.get("error"):
            lines.append(f"      !! {stream['error']}")

    lin = result.get("linpack") or {}
    if lin.get("skipped"):
        lines.append(f"  LINPACK             : skipped — {lin['reason']}")
    else:
        lines.append(f"  LINPACK (HPL)       : {lin.get('rate', 0):>12,.2f} GFLOPS"
                     f"   (N={lin.get('n')}, {lin.get('matrix_mb')} MB)")
        if lin.get("residual") is not None:
            verdict = "passed" if lin.get("validated") else "FAILED"
            lines.append(f"      residual {lin['residual']:.3f} "
                         f"(tolerance {lin['residual_tolerance']}) — {verdict}")
        if lin.get("error"):
            lines.append(f"      !! {lin['error']}")
        lines.append(f"      {lin.get('caveat', '')}")

    cm = result.get("coremark_style") or {}
    if cm.get("skipped"):
        lines.append(f"  CoreMark-style      : skipped — {cm['reason']}")
    else:
        lines.append(f"  CoreMark-style      : {cm.get('rate', 0):>12,.1f} "
                     f"iterations/s")
        lines.append(f"      {cm.get('caveat', '')}")
    return "\n".join(lines)


def run(native: dict | None, ram_bytes: int = 0,
        with_linpack: bool = True) -> dict:
    """Assemble the whole standards section."""
    result = from_native(native)
    if with_linpack:
        result["linpack"] = linpack(ram_bytes)
    else:
        result["linpack"] = {"skipped": True, "reason": "disabled by request"}
    return result
