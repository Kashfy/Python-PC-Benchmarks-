"""System-level benchmarks: compilation, OS latency, and CPU frequency.

Throughput numbers miss a whole class of machine behaviour. A workstation can
post excellent FLOPS while feeling sluggish because process creation is slow or
the scheduler is jittery. These measure the operations that determine how a
machine *feels* rather than how fast it computes.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import tempfile

from .core import clock, summarize

# --------------------------------------------------------------------------- #
# Compile benchmark
# --------------------------------------------------------------------------- #
# Deliberately heavy on the preprocessor and optimiser rather than long: this
# is the mix that dominates real build times, and it keeps the source
# self-contained so no project checkout is needed.
_COMPILE_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define REPEAT8(x)  x x x x x x x x
#define REPEAT64(x) REPEAT8(REPEAT8(x))

static double accumulate(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) { REPEAT64(s += sin(v[i]) * cos(v[i]);) }
    return s;
}

typedef struct { double a, b, c; int tag; } record_t;

static int compare_records(const void *x, const void *y) {
    const record_t *p = x, *q = y;
    return (p->a > q->a) - (p->a < q->a);
}

static double process(record_t *rs, int n) {
    qsort(rs, n, sizeof(record_t), compare_records);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        REPEAT8(total += rs[i].a * rs[i].b - rs[i].c;)
    }
    return total;
}

int main(void) {
    enum { N = 256 };
    double v[N];
    record_t rs[N];
    for (int i = 0; i < N; ++i) {
        v[i] = i * 0.001;
        rs[i] = (record_t){ v[i], v[i] * 2, v[i] * 3, i };
    }
    printf("%.6f %.6f\n", accumulate(v, N), process(rs, N));
    return 0;
}
"""


def _find_cc() -> str | None:
    """First available C compiler, in the platform's preferred order.

    Shares the ordering with the native engine so both make the same choice —
    notably preferring MinGW's self-contained gcc over a bare clang on
    Windows, which needs a Visual Studio installation it frequently does not
    have.
    """
    from .native import compiler_candidates
    candidates = compiler_candidates()
    return shutil.which(candidates[0]) if candidates else None


def bench_compile(repeats: int = 3) -> dict:
    """Time a full compile of a fixed C source at -O2.

    Compilation is the most relatable benchmark for a developer: it exercises
    the preprocessor, optimiser, and linker together, mixing heavy
    single-threaded computation with file I/O and process startup.
    """
    cc = _find_cc()
    if not cc:
        return {"skipped": True, "error": "no C compiler found"}

    with tempfile.TemporaryDirectory() as work:
        src = os.path.join(work, "bench.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(_COMPILE_SOURCE)

        out = os.path.join(work, "bench.out")
        cmd = [cc, "-O2", src, "-o", out, "-lm"]

        # One untimed compile so the compiler binary and headers are cached;
        # otherwise the first run measures the filesystem, not the CPU.
        first = subprocess.run(cmd, capture_output=True, text=True)
        if first.returncode != 0:
            return {"skipped": True,
                    "error": f"compile failed: "
                             f"{(first.stderr or '').strip()[:200]}"}

        times = []
        for _ in range(max(1, repeats)):
            if os.path.exists(out):
                os.remove(out)
            start = clock()
            proc = subprocess.run(cmd, capture_output=True)
            elapsed = clock() - start
            if proc.returncode != 0:
                return {"skipped": True, "error": "compile failed on repeat"}
            times.append(elapsed)

    s = summarize(times)
    median = s["median"]
    return {
        "unit": "compiles/min",
        "rate": 60.0 / median if median > 0 else 0.0,
        "seconds_per_compile": round(median, 3),
        "compiler": os.path.basename(cc),
        "samples_s": [round(t, 3) for t in times],
    }


# --------------------------------------------------------------------------- #
# OS latency suite
# --------------------------------------------------------------------------- #
def _loop_overhead_ns(iterations: int) -> float:
    """Nanoseconds per iteration of an empty loop, for subtraction.

    The syscall below is timed from Python, so every measurement carries the
    interpreter's per-iteration cost — a bytecode dispatch plus a call. On a
    fast x86 core that is around 25 ns and on a slow ARM board it can be five
    times that, which would otherwise make an identical kernel look slower on
    the slower interpreter. Measuring it separately and subtracting leaves the
    kernel transition, which is the thing being reported.
    """
    start = clock()
    for _ in range(iterations):
        pass
    return (clock() - start) / iterations * 1e9


def bench_syscall_latency(iterations: int = 200_000) -> float:
    """Nanoseconds per trivial system call, net of interpreter overhead.

    ``os.getpid`` is not cached by CPython or by modern glibc, so each call
    really does enter the kernel. This is the floor for any I/O operation the
    machine performs, and it is where speculative-execution mitigations show
    up most clearly: they are paid on every kernel entry and exit.
    """
    getpid = os.getpid
    start = clock()
    for _ in range(iterations):
        getpid()
    total = (clock() - start) / iterations * 1e9
    # Never report a negative or absurdly small figure if the subtraction goes
    # wrong under a noisy scheduler; the loop can only ever be the smaller part.
    return max(total * 0.5, total - _loop_overhead_ns(iterations))


def bench_context_switch(iterations: int = 20_000) -> float:
    """Nanoseconds per thread context switch, via a ping-pong handoff.

    Two threads alternate on a pair of events, so each iteration forces the
    scheduler to switch between them. High values here show up as poor
    responsiveness under load.
    """
    import threading

    ping, pong = threading.Event(), threading.Event()
    done = threading.Event()

    def responder():
        for _ in range(iterations):
            ping.wait()
            ping.clear()
            pong.set()
        done.set()

    t = threading.Thread(target=responder, daemon=True)
    t.start()
    start = clock()
    for _ in range(iterations):
        ping.set()
        pong.wait()
        pong.clear()
    elapsed = clock() - start
    done.wait(timeout=2.0)
    t.join(timeout=2.0)
    # Two switches per round trip.
    return elapsed / (iterations * 2) * 1e9


def bench_process_spawn(iterations: int = 20) -> float:
    """Milliseconds to create and reap a trivial child process.

    Dominates build systems, shell scripting, and CI. Windows is inherently
    slower here than Unix because it has no cheap fork equivalent.
    """
    import sys as _sys

    cmd = [_sys.executable, "-c", "pass"]
    times = []
    subprocess.run(cmd, capture_output=True)          # warm the interpreter
    for _ in range(iterations):
        start = clock()
        subprocess.run(cmd, capture_output=True)
        times.append((clock() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def bench_latency_suite() -> dict:
    """All OS latency measurements, each reported in its natural unit."""
    result: dict = {}
    try:
        result["syscall_ns"] = round(bench_syscall_latency(), 1)
    except Exception as e:
        result["syscall_error"] = str(e)
    try:
        result["context_switch_ns"] = round(bench_context_switch(), 1)
    except Exception as e:
        result["context_switch_error"] = str(e)
    try:
        result["process_spawn_ms"] = round(bench_process_spawn(), 2)
    except Exception as e:
        result["process_spawn_error"] = str(e)
    # A single headline figure for scoring: syscalls per second.
    if result.get("syscall_ns"):
        result["unit"] = "syscalls/s"
        result["rate"] = 1e9 / result["syscall_ns"]
    return result


# --------------------------------------------------------------------------- #
# CPU frequency
# --------------------------------------------------------------------------- #
def cpu_frequency_mhz() -> float | None:
    """Current CPU clock in MHz, where the OS exposes it.

    Completes the throttling picture: temperature explains *why* a machine
    slowed, and frequency shows the mechanism. Apple silicon publishes no
    per-core clock without root, so this returns None there.
    """
    system = platform.system()
    try:
        if system == "Linux":
            # scaling_cur_freq is in kHz and reflects the live clock.
            freqs = []
            for path in glob.glob(
                    "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
                try:
                    with open(path) as f:
                        raw = f.read().strip()
                    if raw.isdigit():
                        freqs.append(int(raw) / 1000.0)
                except OSError:
                    pass
            if freqs:
                return round(max(freqs), 1)
            with open("/proc/cpuinfo", errors="ignore") as f:
                m = re.findall(r"cpu MHz\s*:\s*([\d.]+)", f.read())
            if m:
                return round(max(float(x) for x in m), 1)
            return None

        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).CurrentClockSpeed"],
                capture_output=True, text=True, timeout=6).stdout
            m = re.search(r"\d+", out)
            return float(m.group()) if m else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None
