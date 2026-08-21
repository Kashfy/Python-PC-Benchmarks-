# Architecture

How the PC Benchmark & Diagnostics tool is put together, and why.

## Overview

The project is deliberately small and dependency-free. It has two executable
components and a documentation set:

```
Python-PC-Benchmarks-/
├── benchmark.py        # primary tool (Python, stdlib only)
├── native_engine.c     # optional native engine (C, compiled on demand)
├── README.md
├── docs/               # this folder
└── results/            # generated output (git-ignored)
```

### Two-tier design

```
                 ┌──────────────────────────────────────────┐
                 │              benchmark.py                  │
                 │  (orchestration, hardware inventory,       │
                 │   scoring, JSON/CSV output, console UI)    │
                 └───────────────┬───────────────┬───────────┘
                                 │               │
              runs Python        │               │  subprocess + JSON
              workloads in-proc  │               │
                                 ▼               ▼
                    ┌──────────────────┐   ┌──────────────────────┐
                    │  Python workloads │   │   native_engine.c     │
                    │  cpu/mem/disk     │   │ compiled with cc -O2  │
                    │  + multiprocessing│   │ (compiler-optimized)  │
                    └──────────────────┘   └──────────────────────┘
```

`benchmark.py` is the **control plane**: it detects hardware, runs the Python
workloads, optionally shells out to the compiled C engine, normalizes
everything into scores, and writes the reports. `native_engine.c` is an
**optional data plane** that produces compiler-optimized numbers for the same
workloads so you can compare interpreter-level throughput against native code
on the same machine.

## Design principles

1. **Zero required dependencies.** `benchmark.py` imports only the standard
   library, so it drops onto any machine with Python 3.8+ and runs. `psutil`
   is used *if present* but never required. A C compiler is *optional* — its
   absence just skips the native section.

2. **Portable by construction.** Every OS-specific probe has a branch for
   Windows, macOS, and Linux, and a graceful fallback if all else fails. The
   C engine has `#if defined(_WIN32)` branches for timing and temp files so it
   builds on MSVC/MinGW as well as POSIX.

3. **Fail soft, never abort.** Each benchmark is wrapped so that one failing
   probe (e.g. disk full) is recorded as an `{"error": ...}` entry while the
   rest of the run continues.

4. **Comparable, stable numbers.** Results are reported in real units and
   normalized against fixed baseline constants, so a score means the same
   thing across machines and across versions of the tool.

## Execution flow

`main()` in `benchmark.py` drives the whole run:

```
parse_args()
   │
   ├─ apply --quick preset if set
   ├─ validate --only test list
   ├─ create the disk scratch directory
   │
gather_system_info()          ← hardware / OS / arch inventory
   │
for each selected test:
   runners[name]()            ← bench_cpu_integer / _float / _multicore /
   │                             bench_memory / bench_disk
   └─ exceptions captured as {"error": ...}
   │
run_native_engine()           ← compile (if stale) + run native_engine.c,
   │                             parse its JSON  (unless --no-native)
compute_scores()              ← normalize rates vs. BASELINES, geometric mean
   │
build `payload` dict
   │
   ├─ --json-stdout → print payload as JSON
   └─ else          → print_console_report()
   │
unless --no-save:
   ├─ save_json()             ← results/benchmark_<host>_<ts>.json
   └─ append_csv()            ← results/benchmarks.csv
```

## Data model

Every run produces a single `payload` dictionary, which is exactly what lands
in the JSON file:

```jsonc
{
  "tool": "pc-benchmark",
  "version": "2.0",
  "timestamp_utc": "2026-08-21T01:11:19Z",
  "config":  { "seconds": 3.0, "repeats": 3, "tests": [...], ... },
  "system":  { "os": "...", "arch_family": "ARM64", "cpu_model": "...", ... },
  "results": { "cpu_int": {...}, "cpu_float": {...}, "memory": {...}, ... },
  "native":  { "engine": "native-c", "results": [ ... ] } | null,
  "scores":  { "subscores": { ... }, "composite": 576.4 }
}
```

The CSV is a **flattened projection** of the same payload: one row per run with
the headline rate for each test plus the composite score, for spreadsheet
comparison across many machines.

## Cross-process concerns

The multi-core CPU test uses `multiprocessing` with the **`spawn`** start
method explicitly (`mp.get_context("spawn")`) so behavior is identical on
macOS, Windows, and Linux. Because `spawn` re-imports the module in each
worker, the worker function (`_multicore_worker`) is defined at **module top
level** and is picklable, and `mp.freeze_support()` is called under
`__main__` for Windows/frozen builds. Each worker times *itself* for the
target duration rather than sharing a clock, avoiding cross-process
`perf_counter` epoch differences.

## Native engine integration contract

`run_native_engine()` and `native_engine.c` agree on a small contract:

- Python invokes: `native_engine[.exe] --json --seconds N --repeats M`.
- The engine prints a single JSON object to **stdout** with an `engine` field
  and a `results` array of `{name, unit, rate, stdev}` objects.
- Non-zero exit or unparseable output is captured as an error and reported,
  never fatal.
- The binary is rebuilt only when missing or older than the `.c` source
  (mtime check), so repeated runs are fast.

See [functions.md](functions.md) for the per-function reference and
[technical.md](technical.md) for measurement methodology.
