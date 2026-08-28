# Packages, Dependencies & Toolchain

## Platform support matrix

What each section needs, and what it does where the requirement is missing.
Everything degrades with a stated reason; nothing fails silently.

| Section | macOS | Linux | Windows | Requirement on Windows |
|---|---|---|---|---|
| CPU, memory, cache, app workloads | yes | yes | yes | — |
| Storage (seq, IOPS, queue depth) | yes | yes | yes | — (uses seek+read where `os.pread` is absent) |
| `compile` benchmark | yes | yes | needs a compiler | MinGW `gcc`, or `cl` from a Developer Command Prompt |
| Native engine, **STREAM**, **CoreMark-style** | yes | yes | needs a compiler | same |
| **LINPACK** | needs NumPy | needs NumPy | needs NumPy | `pip install numpy` |
| Resource counters | yes | yes | needs psutil | `pip install psutil` |
| **PMU counters** (`--counters`) | no | needs `perf` | **no** | Requires a kernel driver; no equivalent exists |
| CPU feature list | full | full | needs py-cpuinfo | `pip install py-cpuinfo` (Win32 API has no AES/SHA codes) |
| GPU/NPU compute | yes (Metal/Core ML) | OpenCL only | OpenCL only | `pip install pyopencl` |
| GPU inventory | yes | yes | yes | VRAM from the driver registry, not WMI |
| Power measurement | `powermetrics` (sudo) | RAPL | **estimate only** | No on-die metering exposed |
| Thermals / fans | yes | hwmon | partial | WMI thermal zones where the vendor publishes them |
| Drive lifetime | IOKit NVMe | `nvme-cli`/`smartctl` | partial | Elevated PowerShell; many consumer drives report nulls |
| NUMA topology | n/a (unified) | yes | socket count | — |
| NUMA bandwidth matrix | n/a | needs `numactl` | **no** | `numactl` is Linux-only |
| Soak, monitor, gates, exports, A/B | yes | yes | yes | — |

### Windows compiler selection

Compilers are tried in order and the first that **actually produces a binary**
wins, because being on PATH and being able to compile are different things:

1. `gcc` — MinGW-w64, self-contained, no other dependency
2. `cl` — MSVC; needs a Developer Command Prompt for its environment, and takes
   a different flag dialect (`/O2`, `/Fe:`) which the tool supplies
3. `clang-cl`, then `clang` — LLVM clang relies on a Visual Studio installation
   for headers and linker, and fails without one
4. `cc`

When all fail, every failure is listed rather than a misleading "no compiler
found".

## Runtime requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | **3.8+** | Needs `math.isqrt` and `statistics.fmean`. |
| OS | Windows, macOS, Linux | All three have dedicated probe paths. |
| C compiler | **Optional** | Only for the native engine. |
| Xcode Command Line Tools | **Optional, macOS** | Only for GPU/NPU benchmarks. Full Xcode is *not* needed. |

**No `pip install` is required.** The tool imports only the standard library.

## Installation

Three ways to run it, all equivalent:

```bash
# 1. Straight from a checkout — no install
python3 benchmark.py

# 2. Installed, with a console command
pip install -e .
pcbench --quick

# 3. As a module
python3 -m pcbench
```

The [`benchmark.py`](../benchmark.py) launcher exists so the tool can be copied
onto an unfamiliar machine and run immediately, with no install step and no
`PYTHONPATH` fiddling.

## Standard-library modules used

| Module | Used for |
|--------|----------|
| `argparse` | CLI parsing |
| `csv` | History file read/write |
| `ctypes` | Windows RAM and power queries, and enabling ANSI output on Windows 10+ consoles (lazy import) |
| `datetime` | UTC timestamps |
| `fcntl` | macOS `F_NOCACHE` page-cache bypass (lazy import) |
| `hashlib` | SHA-256 workload |
| `html` | Escaping in the HTML report |
| `json` | JSON output, parsing native engine output |
| `math` | `isqrt`, trig, `exp`/`log` for scoring |
| `multiprocessing` | Multi-core and sustained tests (`spawn`) |
| `os` | File descriptors, `cpu_count`, `pread`, `posix_fadvise` |
| `platform` | OS / arch / Python identification |
| `random` | Deterministic corpus and random-read offsets |
| `re` | Parsing `/proc` files, sanitizing filenames |
| `shutil` | `which`, `disk_usage`, terminal size for `--menu` |
| `statistics` | `median`, `fmean`, `stdev` |
| `subprocess` | `sysctl`, `pmset`, PowerShell, the native engine |
| `sys` | Bitness, byte order, exit codes |
| `sysconfig` | Free-threaded build detection |
| `tempfile` | Scratch files |
| `time` | `perf_counter` |
| `unittest` | Test suite |
| `winreg` | Windows CPU model (lazy import) |
| `zlib` | Compression workload |
| `socket` / `threading` | Loopback network benchmark |
| `struct` | Packing weights into the Core ML and ONNX protobufs |
| `glob` | Sensor, power-supply, thermal and hwmon sysfs enumeration |
| `termios` / `tty` | Raw keyboard mode for `--menu` on Unix (lazy import) |
| `msvcrt` | Raw keyboard mode for `--menu` on Windows (lazy import) |

## Optional package tiers

The core is standard-library only. Optional tiers lift ceilings the standard
library genuinely cannot:

```bash
python3 install.py --list          # what exists and what is installed
python3 install.py                 # all tiers, into ./.venv, after confirming
python3 install.py --tier compute  # one tier only
python3 install.py --here          # current interpreter instead of a venv
```

Or with pip extras: `pip install -e ".[compute]"` … `".[all]"`.

| Tier | Packages | Unlocks |
|------|----------|---------|
| `compute` | numpy, scipy, numba | BLAS matmul (FP64/FP32), FFT, LAPACK SVD/Cholesky/eigenvalues |
| `gpu` | pyopencl, nvidia-ml-py | GPU compute on NVIDIA/AMD/Intel/Apple; NVIDIA temp, power, VRAM |
| `crypto` | cryptography, zstandard, lz4, blake3 | AES-256-GCM via AES-NI, modern compression and hashing |
| `system` | psutil, py-cpuinfo, rich, matplotlib, scikit-learn | Sensors on Windows/Linux, charts, tables, reference ML |

**Why a venv by default.** Installing into a system interpreter risks
permission problems and leaves packages that are awkward to remove. The
installer creates `./.venv`, prints exactly what it will install and the
approximate size, and does nothing until you confirm. Packages are installed
one at a time so a single unavailable wheel — `pyopencl` has none on some
platforms — does not abort the batch.

## Optional dependency: an ML framework (AI tier only)

The AI training/inference benchmark uses **PyTorch** if installed, or **ONNX
Runtime** as an inference-only fallback. Neither is required — the section is
skipped otherwise. This is the only place the tool uses a heavy dependency, and
only when you opt in by installing it:

```bash
pip install torch          # real training + inference, CUDA/ROCm/MPS/CPU
pip install onnxruntime    # inference + cross-vendor NPU benchmarking
```

`onnxruntime` additionally unlocks **cross-vendor NPU benchmarking** (Intel,
AMD, Qualcomm, DirectML). Vendor-specific builds target their own NPU:

```bash
pip install onnxruntime-openvino    # Intel AI Boost NPU
pip install onnxruntime-directml    # any DirectML device on Windows
pip install onnxruntime-qnn         # Qualcomm Hexagon NPU
```

The `onnx` package is **not** required — pcbench writes the ONNX protobuf
itself.

## Optional dependency: `psutil`

Not required. When importable it improves exactly two fields — total RAM and
physical core count. Otherwise the tool falls back to native OS probes. The
`system.psutil_available` field records which path was taken.

```bash
python3 -m pip install psutil       # or: pip install -e ".[extras]"
```

## Native engine toolchain

Standard C11, no third-party libraries. Links the C runtime plus, on POSIX, the
math library and pthreads.

| Platform | Build command |
|----------|---------------|
| macOS / Linux | `cc -O2 native_engine.c -o native_engine -lm -lpthread` |
| Windows (MinGW) | `gcc -O2 native_engine.c -o native_engine.exe` |
| Windows (MSVC) | `cl /O2 native_engine.c` |

`pcbench` builds it automatically, rebuilding only when the binary is missing
or older than the source. Verified warning-free under
`-Wall -Wextra -std=c11` on both clang and gcc.

## Sensors engine toolchain (macOS)

`sensors_engine.m` reads temperatures through the unprivileged IOHID thermal
usage page. It links only Foundation and IOKit and builds in well under a
second:

```bash
clang -O2 -fobjc-arc sensors_engine.m -o sensors_engine \
      -framework Foundation -framework IOKit
```

It is deliberately separate from `accel_engine.m` so temperatures are still
available when accelerator benchmarking is skipped.

## Accelerator engine toolchain (macOS)

`accel_engine.m` is Objective-C linking Foundation, Metal, and CoreML — all
present in the Command Line Tools SDK:

```bash
clang -O2 -fobjc-arc accel_engine.m -o accel_engine \
      -framework Foundation -framework Metal -framework CoreML
```

Metal shaders are compiled **at runtime** via `newLibraryWithSource:`, so the
offline `metal` compiler (full Xcode only) is not required.

The Core ML model used for the Neural Engine test is generated at runtime by
`pcbench/coreml_model.py`, which writes the `.mlmodel` protobuf directly — no
`coremltools` dependency.

## Command-line interface

```
pcbench [options]          #  or:  python3 benchmark.py [options]
```

### Workload

| Flag | Default | Description |
|------|---------|-------------|
| `--seconds N` | `3.0` | Duration per test, per repeat |
| `--repeats M` | `3` | Repeats per test (median reported) |
| `--only a,b` | all | Subset of tests |
| `--skip a,b` | none | Exclude tests |
| `--quick` | off | 1s × 2 repeats, 64 MB disk |
| `--disk-mb K` | `256` | Disk test file size |
| `--mem-mb K` | `64` | Memory buffer size |

Test names: `cpu_int`, `cpu_float`, `cpu_multi`, `compression`, `hashing`,
`json`, `memory`, `mem_scaling`, `cache_sweep`, `disk`, `nn_training`,
`kmeans`, `knn`, `cores`, `compile`, `latency`.

Profiles select curated subsets: `--profile quick|cpu|ai|dev|storage|
laptop|server`.

The three ML workloads (`nn_training`, `kmeans`, `knn`) need no dependencies —
they are pure standard library.

### Sustained load

| Flag | Default | Description |
|------|---------|-------------|
| `--sustained D` | off | Thermal test duration: `30s`, `5m`, `1h` |
| `--sustained-window N` | `5.0` | Sampling window, seconds |
| `--sustained-workers N` | all cores | Load processes |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir D` | `results` | Output location |
| `--html` | off | Also write a self-contained HTML report |
| `--json-stdout` | off | Print full payload as JSON |
| `--no-save` | off | Write no files |
| `--compare` | — | Ranked table of past runs, then exit |
| `--all-runs` | off | With `--compare`, every run rather than latest per host |

### Accelerators

| Flag | Description |
|------|-------------|
| `--no-gpu` | Skip GPU compute benchmarks |
| `--no-npu` | Skip NPU / Apple Neural Engine benchmarks |
| `--no-accel` | Skip all accelerator benchmarks (inventory still reported) |

GPU/NPU **inventory** works on all platforms. Compute **benchmarking** is
Apple-only (Metal + Core ML); see
[technical.md](technical.md#gpu-and-npu-benchmarking) for why.

### AI / efficiency / monitoring

| Flag | Default | Description |
|------|---------|-------------|
| `--ai` | off | Force the AI framework benchmark (auto-runs if installed) |
| `--no-ai` | off | Skip it even if a framework is installed |
| `--ai-batch N` | `64` | Batch size for the AI framework benchmark |
| `--no-power` | off | Skip power / perf-per-watt |
| `--no-network` | off | Skip the loopback network benchmark |
| `--no-regression` | off | Skip run-over-run regression detection |
| `--health` | off | RAM integrity + drive SMART checks |
| `--health-mb N` | `256` | Memory covered by the RAM integrity test |
| `--spec-sheet` | off | Write a one-page Markdown spec sheet |
| `--no-plugins` | off | Skip benchmarks in `plugins/` |
| `--network-host H` | none | Measure real latency to a host (**sends external traffic**) |
| `--network-url U` | none | Measure download throughput (**sends external traffic**) |
| `--no-optional` | off | Skip every benchmark that needs an optional package |
| `--regression-threshold P` | `10` | Percent change counting as a regression |

For real measured power on macOS, run the whole tool with `sudo`.

### Storage devices

| Flag | Description |
|------|-------------|
| `--list-devices` | List mounts and whether each can be benchmarked, then exit |
| `--disk-all` | Benchmark every writable local filesystem |
| `--disk-path P[,P]` | Benchmark named mount points (overrides the safety heuristics) |

### Stability and monitoring

| Flag | Description |
|------|-------------|
| `--soak D` | Burn-in for `D`; counts wrong answers instead of stopping at the first |
| `--soak-workers N` | Load processes for the soak (0 = all cores) |
| `--monitor D` | Live telemetry for `D` instead of benchmarking |
| `--monitor-interval N` | Seconds between monitor samples |
| `--monitor-power` | Also sample power draw (a privileged subprocess per sample on macOS) |
| `--monitor-trace P` | Write raw monitor samples to CSV |

### Integration and CI

| Flag | Description |
|------|-------------|
| `--prometheus P` | Prometheus exposition text, written atomically |
| `--junit P` | JUnit XML: benchmarks, regressions, and gates as test cases |
| `--sqlite P` | Append the run to a SQLite history database |
| `--markdown P` | Markdown summary for an issue or PR comment |
| `--fail-under N` | Exit 6 when the composite is below `N` |
| `--assert EXPR` | Threshold that must hold; repeatable. Exit 6 on failure |

Assertion syntax is `NAME OP VALUE` with `>=`, `<=`, `>`, `<`, `==`, `!=`.
A bare `NAME` resolves to the **score** (baseline = 100); `NAME.field` resolves
to the raw payload value (`disk.read_rate`, `sqlite.rate`,
`sustained.droop_pct`). `category.cpu` addresses a rollup. Every verdict states
which source it used. A metric that was not measured **fails** rather than
passing silently.

### Configuration

| Flag | Description |
|------|-------------|
| `--config P` | Read settings from this TOML/JSON file |
| `--no-config` | Ignore `pcbench.toml` and `PCBENCH_*` variables |
| `--init-config [P]` | Write a commented starter config and exit |
| `--list-tests` | List every test and profile, then exit |

Config files are found by walking up from the working directory
(`pcbench.toml`, `.pcbench.toml`, `pcbench.json`, `.pcbench.json`). TOML needs
Python 3.11+; JSON works everywhere. Precedence is command line > `PCBENCH_*`
environment > config file > defaults.

### Hardware stats (no benchmarking)

| Flag | Description |
|------|-------------|
| `--stats [SECTIONS]` | Report hardware facts without load. Comma-separated, or omit for all |
| `--list-stats` | List the available sections |
| `--menu` | Guided setup, driven by the arrow keys: pick a benchmark, stats section, monitor or comparison one screen at a time, then confirm the command it builds. `PCBENCH_NO_TUI=1` forces typed answers |

Sections: `cpu`, `memory`, `storage`, `drives`, `battery`, `gpu`, `thermal`,
`power`, `os`, `environment`, `numa`, `packages`.

The whole set gathers in roughly two seconds and never loads the machine, so it
is safe to run on a busy server or a laptop on battery. `--json-stdout` applies,
which makes any section usable as a monitoring check:

```bash
pcbench --stats battery --json-stdout | jq '.stats.battery.health_percent'
pcbench --stats drives  --json-stdout | jq '.stats.drives.drives[0].health_pct'
```

A failing section is isolated and reported in place; it never takes the rest of
the report down.

### Analysis depth

| Flag | Description |
|------|-------------|
| `--counters` | PMU counters (IPC, cache/branch misses) plus rusage; needs `perf` on Linux |
| `--no-provenance` | Skip governor/mitigations/hugepages/microcode capture |
| `--no-standards` | Skip STREAM, LINPACK, CoreMark-style |
| `--no-linpack` | Skip LINPACK only |
| `--numa` / `--numa-bandwidth` | NUMA topology; the matrix needs `numactl` |
| `--energy` | Joules for a fixed workload |

### Data science / ML

| Flag | Description |
|------|-------------|
| `--datascience` | LLM prefill/decode, input pipeline, batch scaling, dataframes |
| `--ds-prefill-tokens N` / `--ds-decode-tokens N` | LLM measurement lengths |
| `--no-dataframes` | Skip the dataframe benchmarks |

### Configurable I/O

| Flag | Description |
|------|-------------|
| `--io` | Four-job suite: database, sequential, log write, mixed VM |
| `--io-job SPEC` | `name:bs=8k,pattern=randread,qd=32,rw=70,time=5,size=1g,direct=1` |

Patterns: `read`, `write`, `randread`, `randwrite`, `randrw`.

### Two-node network

| Flag | Description |
|------|-------------|
| `--net-server` | Receiving half; opens a listening port until stopped |
| `--net-client HOST` | Measuring half: RTT, jitter, single and parallel throughput |
| `--net-port N` / `--net-streams N` | Port and stream count |

### A/B comparison

| Flag | Description |
|------|-------------|
| `--compare-runs A.json B.json` | Mann-Whitney U per metric; exits 6 on a significant regression |
| `--alpha P` | Significance threshold (default 0.05) |

### Other

| Flag | Description |
|------|-------------|
| `--no-native` | Skip the C engine |
| `--no-drive-life` | Skip the SSD lifetime/wear report |
| `--no-autoscale` | Do not shrink test sizes on small or CPU-limited machines |
| `--force` | Run despite distorting machine state |
| `--version` | Print version |

The native engine separately accepts `--json`, `--seconds`, `--repeats`,
`--threads`, `--mem-mb`, `--disk-mb`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Invalid arguments |
| `3` | Refused — machine state would distort results |
| `4` | Validation failure — hardware may be unstable |
| `5` | Output directory not writable (often root-owned after a `sudo` run) |
| `6` | A `--fail-under` / `--assert` threshold was not met, or `--compare-runs` found a significant regression |
| `7` | Soak test produced wrong answers — hardware is unstable |

Codes 3, 4, 6, and 7 are designed for scripting: a CI job or fleet sweep can
treat each distinctly from a normal failure. When several apply, the most
severe is returned — 4 (a wrong answer during the benchmark) outranks 7
(a wrong answer during the soak), which outranks 6 (merely too slow).

## Output artifacts

Written to `--output-dir` (default `results/`, git-ignored):

| File | Contents |
|------|----------|
| `benchmark_<host>_<timestamp>.json` | Full payload for one run |
| `benchmarks.csv` | One row per run, for comparison |
| `report_<host>_<timestamp>.html` | Self-contained report (`--html`) |
| `spec_<host>_<timestamp>.md` | One-page spec sheet (`--spec-sheet`) |
| `benchmarks.csv.v2.bak` | Auto-archived history from an older schema |
| `ane_model.mlmodel` | Generated Neural Engine benchmark model (regenerated as needed) |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

564 cases, standard library only — no pytest, no test dependencies.

## Version notes

- `math.isqrt` and `statistics.fmean` set the Python 3.8 floor.
- `os.posix_fadvise` is Linux-only; `fcntl.F_NOCACHE` is macOS-only. Both are
  guarded.
- `os.getloadavg()` is unavailable on Windows and returns `None` there.
- `multiprocessing` uses `spawn` explicitly on all platforms for consistency.
