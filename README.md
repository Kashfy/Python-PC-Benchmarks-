# PC Benchmark & Diagnostics

A reliable, cross-platform benchmark and hardware-diagnostics tool for
**Windows, macOS, and Linux** on **x86-64, ARM64, and other** CPU
architectures.

It measures CPU (single- and multi-core), memory bandwidth, and disk I/O in
**meaningful, comparable units**, gathers a hardware/OS inventory (including
chip architecture), and writes results to the console plus timestamped JSON
and an appended CSV so you can compare many machines over time.

- **`benchmark.py`** — the primary tool. Pure Python **standard library**, so
  it runs on any machine with Python 3.8+ with **no `pip install`**. If
  [`psutil`](https://pypi.org/project/psutil/) happens to be installed it's
  used for richer hardware info, but it is entirely optional.
- **`native_engine.c`** — an optional native (C) engine that `benchmark.py`
  auto-compiles and runs to give **compiler-optimized** numbers alongside the
  Python results. Builds on Windows (MSVC/MinGW), macOS, and Linux.

## Quick start

```bash
python3 benchmark.py
```

That runs every test at default settings and prints a report like:

```
System Information
  Hostname      : my-laptop
  OS            : Linux 6.8.0
  Architecture  : ARM64 (aarch64, 64-bit, little-endian)
  CPU           : Apple M4 / Snapdragon X / Ryzen 7 7840U ...
  Cores         : 8 physical / 16 logical
  RAM           : 16.0 GB
  Python        : CPython 3.12.3

Benchmark Results
  CPU Integer (primes)   :  4,491,982 primes/s
  CPU Float (math ops)   : 19,706,117 iters/s
  CPU Multi-core (10w)   : 22,597,141 primes/s  ->  5.0x vs 1 core
  Memory copy bandwidth  :     41,909 MB/s
  Disk write             :      5,147 MB/s
  Disk read              :     14,027 MB/s

Scores (baseline machine = 100, higher is better)
  ...
  COMPOSITE     :    589.7
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--seconds N` | `3.0` | Target duration per test, per repeat |
| `--repeats M` | `3` | Repeats per test (the **median** is reported) |
| `--disk-mb K` | `256` | Disk-test file size in MB |
| `--mem-mb K` | `64` | Memory-test buffer size in MB |
| `--only a,b` | all | Run a subset: `cpu_int,cpu_float,cpu_multi,memory,disk` |
| `--quick` | off | Fast pass (1s × 2 repeats, small disk test) |
| `--no-native` | off | Skip the optional C engine |
| `--no-save` | off | Don't write JSON/CSV files |
| `--output-dir D` | `results` | Where JSON/CSV go |
| `--json-stdout` | off | Print the full result payload as JSON to stdout |

Examples:

```bash
python3 benchmark.py --seconds 5 --repeats 5      # more stable numbers
python3 benchmark.py --quick                      # ~15s smoke test
python3 benchmark.py --only cpu_int,cpu_multi     # CPU only
python3 benchmark.py --json-stdout --no-save      # pipe JSON elsewhere
```

## What each test measures

| Test | Unit | What it stresses |
|------|------|------------------|
| **CPU Integer** | primes/s | Integer ALU, branch prediction (primality testing) |
| **CPU Float** | iters/s | FPU / libm (`sin`/`cos`/`sqrt` in a tight loop) |
| **CPU Multi-core** | primes/s | All logical cores in parallel + scaling factor vs. 1 core |
| **Memory** | MB/s | Memory copy bandwidth (large buffer `memmove`) |
| **Disk** | MB/s | Sequential write (with `fsync`) and read on a real file |

### Reliability notes

- Each test runs multiple **repeats** and reports the **median** plus standard
  deviation — the median resists one-off outliers (e.g. thermal spikes,
  background load).
- The disk test **`fsync`s** after writing and best-effort drops the OS page
  cache before reading (`posix_fadvise` on Linux); read numbers can still be
  cache-influenced on some platforms, so treat them as a floor.
- No single failing probe aborts the run — it's recorded as an error in the
  output and the rest continue.

### The composite score

Every measured rate is normalized against a fixed baseline (baseline = 100),
and the composite is their geometric mean. The baselines are **arbitrary but
stable constants**, so a score is comparable across machines and across
versions of this tool. Higher is better. They are **not** an endorsement of any
particular reference machine — just a common yardstick.

## Chip architecture

The tool reports both the raw `platform.machine()` value and a normalized
**ISA family**, so results from different OSes line up (Windows calls a chip
`AMD64` where Linux calls it `x86_64` — both map to `x86-64`). Recognized
families include `x86-64`, `x86-32`, `ARM64`, `ARM32`, `RISC-V 64/32`,
`PowerPC 64`, `IBM Z`, and `MIPS`. On Linux ARM single-board computers it also
reads the device-tree model (e.g. `Raspberry Pi 5`).

## The native (C) engine

By default `benchmark.py` looks for `native_engine.c`, compiles it with any
available `cc`/`clang`/`gcc` (`-O2`), runs it, and shows its numbers under
"Native (C) Engine". This lets you compare interpreter-level throughput against
compiler-optimized native code on the same machine. If no compiler is present,
that section is simply skipped — the Python benchmarks still run.

Build it yourself if you like:

```bash
# POSIX (macOS / Linux)
cc -O2 native_engine.c -o native_engine -lm

# Windows, MinGW
gcc -O2 native_engine.c -o native_engine.exe

# Windows, MSVC (Developer Command Prompt)
cl /O2 native_engine.c

./native_engine --json --seconds 5 --repeats 5
```

## Output files

- `results/benchmark_<host>_<timestamp>.json` — full structured result per run
  (system info, every raw sample, native numbers, scores).
- `results/benchmarks.csv` — one appended row per run; open in any spreadsheet
  to compare machines side by side.

## Documentation

Full reference docs live in [`docs/`](docs/README.md):

- [architecture.md](docs/architecture.md) — design, execution flow, data model
- [technical.md](docs/technical.md) — methodology, units, statistics, scoring
- [functions.md](docs/functions.md) — per-function reference
- [packages.md](docs/packages.md) — dependencies, toolchain, CLI
- [troubleshooting.md](docs/troubleshooting.md) — common problems and fixes

## Requirements

- **Python 3.8+** (standard library only; `psutil` optional).
- A C compiler is **optional** — only needed for the native engine.
```
