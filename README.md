# PC Benchmark & Diagnostics

A reliable, cross-platform benchmark and hardware-diagnostics tool for
**Windows, macOS, and Linux** on **x86-64, ARM64, and other** CPU
architectures.

It measures CPU, memory, disk, **GPU, NPU** (Apple Neural Engine, **Intel AI
Boost, AMD Ryzen AI, Qualcomm Hexagon**), and **AI/ML workloads** — including
real neural-network training, clustering, and vector search — in meaningful,
comparable units;
measures **power and perf-per-watt**; detects **thermal throttling** and
**run-over-run regressions**; **validates** that the hardware computes correct
results; benchmarks the **network stack**; and records everything to
JSON/CSV/HTML so you can compare machines over time.

- **Pure Python standard library** — runs on any machine with Python 3.8+,
  **no `pip install` required**.
- **Optional native C engine** — auto-compiled for compiler-optimized numbers
  and real memory-latency measurement.

## Is it safe for my hardware?

**Yes.** It never writes to raw devices, never formats anything, and never
touches firmware, voltages, or clock multipliers — the only software routes to
real physical damage. It writes one temporary file it deletes afterwards.

Resource use is capped so it cannot harm a machine indirectly:

| Risk | Guard |
|------|-------|
| RAM exhaustion / swap thrash | Buffers capped to 1/8 of physical RAM |
| Filling the disk | 1.5× free-space headroom required |
| SSD wear | ≤16 GB written per run (768 MB by default), always disclosed |
| Overheating | Sustained test stops early on severe throttle or >100 °C |

Full audit in [docs/safety.md](docs/safety.md).

## Two ways to run it

**Zero install** — the standard-library core works immediately, everywhere:

```bash
python3 benchmark.py
```

**With optional packages** — unlocks benchmarks the standard library cannot
provide. One command, into a project-local venv, nothing installed until you
confirm:

```bash
python3 install.py            # interactive; --list to preview, --tier to pick
.venv/bin/python benchmark.py
```

| Tier | Unlocks | Why the stdlib can't |
|------|---------|----------------------|
| `compute` | BLAS matmul, FFT, LAPACK (numpy, scipy, numba) | Pure Python measures **CPython, not your CPU** — 113 MFLOPS vs **450 GFLOPS** on the same M4 |
| `gpu` | GPU compute on NVIDIA/AMD/Intel (pyopencl), NVIDIA telemetry (pynvml) | No portable GPU API — GPU benchmarking was Apple-only |
| `crypto` | AES-NI throughput, Zstandard, LZ4, BLAKE3 | **No AES primitive exists** in the stdlib |
| `system` | Better sensors, charts, rich tables, reference ML (psutil, matplotlib, rich, scikit-learn) | Limited sensor access on Windows/Linux |

Everything degrades gracefully: a missing package removes its section and
nothing else. Scores omit absent capabilities rather than penalising them.

## Quick start

```bash
python3 benchmark.py
```

Or install it and use the command:

```bash
pip install -e .
pcbench --quick
```

## What makes it reliable

| Feature | Why it matters |
|---------|----------------|
| **Temperature in °C** | Real sensor readings on macOS (IOHID, no root), Linux (hwmon/thermal), Windows (WMI) — plus temperature tracked through sustained load. |
| **Machine-state guard** | Refuses to run on battery or under load, because those produce numbers that look like hardware differences but aren't. Override with `--force`. |
| **Warm-up pass** | Discards the cold-cache, low-clock first iterations before timing. |
| **Median of repeats** | Resists one-off outliers; reports a stability rating (`excellent`…`unstable`). |
| **Result validation** | Every workload verifies its own output. A wrong answer means unstable RAM/overclock/cooling — and exits with code 4. |
| **Cache-bypassed disk I/O** | Uses `F_NOCACHE`/`posix_fadvise` *before* writing, so reads measure the drive, not RAM. Reports whether it succeeded. |
| **Fail-soft** | One failing probe never aborts the run. |

## Example

```
System Information
  Hostname      : Kashfys-Air.lan
  OS            : Darwin 25.6.0
  Architecture  : ARM64 (arm64, 64-bit, little-endian)
  CPU           : Apple M4
  Cores         : 10 physical / 10 logical
  RAM           : 16.0 GB
  Power         : AC
  Thermal       : nominal

Benchmark Results
  CPU Integer (primes)      :      4,437,140 primes/s  (excellent, ±0.8%)
  CPU Float (math ops)      :     19,847,875 iters/s   (excellent, ±0.3%)
  Compression (zlib)        :           48.0 MB/s
  Hashing (SHA-256)         :          3,207 MB/s
  JSON parse                :          177.2 MB/s
  Memory copy bandwidth     :         42,677 MB/s
  CPU Multi-core (10w)      :     22,337,201 primes/s  →  5.0x vs 1 core
  Disk sequential write     :          5,260 MB/s
  Disk sequential read      :          3,763 MB/s
  Disk random read (4K)     :         37,994 IOPS

  Memory bandwidth by working-set size (cache hierarchy):
     128 KB  ████████████████████████████████      69,754 MB/s
       2 MB  ██████████████████████████████████    73,200 MB/s
      32 MB  ████████████████████                  44,888 MB/s
     128 MB  ███████████████████                   41,914 MB/s
```

## Depth-aware storage testing

Measuring random reads one at a time (queue depth 1) understates a modern SSD
by roughly **5×** — real workloads keep many requests in flight. The tool
sweeps queue depth and reports latency percentiles:

```
  Random read (4K)          :         37,917 IOPS
  Random read latency       :           1.79 us p50   (p99 97.9 us)

    Random-read IOPS by queue depth (how many requests are in flight):
      QD 1  █████                            49,405 IOPS
      QD 4  ██████████████████              152,413 IOPS
      QD16  ███████████████████████████     225,912 IOPS
      QD32  ████████████████████████████    231,746 IOPS
      peak 231,746 IOPS at QD32 — 4.7x the queue-depth-1 figure
```

## Core scaling and hybrid CPUs

Modern chips mix fast and slow cores, which is why "10 cores" rarely means 10×.
The tool measures the marginal gain from each added worker:

```
  Core scaling              :          5.43x on 10 cores
      scales near-linearly to 3 worker(s); beyond that each added worker
      contributes about 38% as much, indicating a hybrid design
       1w  ████████████████████████    4,490,000/s
       4w  ███████████████            15,888,000/s
      10w  ████                       24,398,000/s
```

It deliberately reports *how far scaling stays linear* rather than exact P/E
core counts — see [technical.md](docs/technical.md#core-scaling-analysis) for
why that estimate proved unreliable.

## Thermal / sustained-load testing

A three-second benchmark only measures *burst* speed. Thin and fanless laptops
hold peak clocks for a minute, then drop to whatever their cooling sustains —
often 15–40% lower. That gap is what determines real performance on long work
like compiling or video export.

```bash
python3 benchmark.py --sustained 5m
```

```
Sustained Load — thermal behavior
  Peak throughput           :     24,119,411 primes/s
  Sustained (final 25%)     :     20,894,599 primes/s
  Droop                     :           13.4 %

  Throughput over time: █▆▄▃▂▂▁▂
  Verdict: mild throttling — typical for a well-cooled laptop
```

## Comparing machines

Every run appends to `results/benchmarks.csv`. Rank your fleet with:

```bash
python3 benchmark.py --compare
```

```
  Machine                              Score    CPU int   CPU multi    SHA256    Disk W     IOPS
  ----------------------------------------------------------------------------------------------
  linux-desktop (AMD Ryzen 9 7950X)      612  5,200,000  98,000,000     2,100     4,100   52,000   (best)
  Kashfys-Air.lan (Apple M4)             349  4,481,993  21,357,440     3,207     4,307   46,960   (57% of best)
  rpi5 (Raspberry Pi 5)                   88    900,000   3,400,000       310        95    6,100   (14% of best)
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--seconds N` | `3.0` | Duration per test, per repeat |
| `--repeats M` | `3` | Repeats per test (median reported) |
| `--only a,b` | all | Subset of tests (see below) |
| `--profile NAME` | — | Preset selection: `quick`, `cpu`, `ai`, `dev`, `storage`, `laptop`, `server` |
| `--skip a,b` | none | Exclude tests |
| `--quick` | off | Fast pass (1s × 2 repeats) |
| `--disk-mb K` | `256` | Disk test file size |
| `--mem-mb K` | `64` | Memory buffer size |
| `--sustained D` | off | Thermal test, e.g. `30s`, `5m`, `1h` |
| `--sustained-window N` | `5.0` | Sampling window for the thermal test |
| `--sustained-workers N` | all cores | Load processes for the thermal test |
| `--compare` | — | Show ranked table of past runs and exit |
| `--all-runs` | off | With `--compare`, show every run |
| `--html` | off | Also write a self-contained HTML report |
| `--json-stdout` | off | Print full payload as JSON |
| `--output-dir D` | `results` | Output location |
| `--no-save` | off | Write no files |
| `--no-native` | off | Skip the C engine |
| `--no-gpu` | off | Skip GPU compute benchmarks |
| `--no-npu` | off | Skip NPU / Neural Engine benchmarks |
| `--no-accel` | off | Skip all accelerator benchmarks (inventory still shown) |
| `--ai` | off | Force the AI framework benchmark (auto-runs if torch/onnx installed) |
| `--no-ai` | off | Skip the AI framework benchmark |
| `--ai-batch N` | `64` | Batch size for the AI framework benchmark |
| `--no-power` | off | Skip power / perf-per-watt |
| `--no-network` | off | Skip the loopback network benchmark |
| `--no-regression` | off | Skip run-over-run regression detection |
| `--regression-threshold P` | `10` | Percent change that counts as a regression |
| `--force` | off | Run despite distorting machine state |

Tests: `cpu_int`, `cpu_float`, `cpu_multi`, `compression`, `hashing`, `json`,
`memory`, `mem_scaling`, `cache_sweep`, `disk`, `nn_training`, `kmeans`,
`knn`, `cores`, `compile`, `latency`.

## What each test measures

| Test | Unit | Stresses |
|------|------|----------|
| CPU Integer | primes/s | Integer ALU, branch prediction |
| CPU Float | iters/s | FPU / libm |
| CPU Multi-core | primes/s | All cores + scaling factor |
| Compression | MB/s | zlib round-trip (mixed real-world load) |
| Hashing | MB/s | SHA-256 — reaches hardware crypto (ARM crypto ext., x86 SHA-NI) |
| JSON parse | MB/s | Parser/allocator throughput |
| Memory | MB/s | Sustained copy bandwidth |
| Cache sweep | MB/s | Bandwidth vs. working-set size → cache tiers |
| Disk | MB/s + IOPS | Sequential write/read **and** 4 KiB random reads |
| Core scaling | curve | Marginal gain per added worker — exposes hybrid P/E designs |
| Memory scaling | MB/s | Bandwidth vs. process count — finds the memory-controller ceiling |
| Compile | s / compiles-per-min | Real C compilation at `-O2` |
| OS latency | ns / ms | Syscall, context-switch, and process-spawn cost |
| Neural net training | steps/s | **Real** MLP forward + backprop + SGD, pure Python |
| K-means clustering | distances/s | Lloyd's algorithm — the canonical unsupervised workload |
| K-NN search | comparisons/s | Brute-force similarity search (vector-DB style) |

### Machine-learning workloads, with no framework required

Three classic ML workloads run on **every** machine using only the standard
library — no PyTorch, no NumPy, no install:

```
  Neural net training       :          888.3 steps/s  (21,320 samples/s, 110 MFLOPS)
  K-means clustering        :      2,348,271 distances/s
  K-NN search               :      2,375,082 comparisons/s
```

The neural network is **genuine training** — forward pass, backpropagation,
and SGD weight updates — not a synthetic stand-in. Each workload validates
itself: the network must actually reduce its loss, k-means must converge to a
known inertia, and every point must be its own nearest neighbour. A wrong
answer means faulty hardware, not a slow machine.

## NPU support across vendors

NPU **detection** works on every platform and identifies the vendor precisely,
using PCI device IDs and kernel drivers on Linux rather than guessing:

| Vendor | Device | Detected via |
|--------|--------|--------------|
| Apple | Neural Engine | Chip model (every Apple-silicon Mac) |
| Intel | AI Boost NPU (Meteor/Arrow/Lunar/Panther Lake) | PCI ID, `intel_vpu` driver, PnP |
| AMD | Ryzen AI NPU (XDNA / XDNA2) | PCI ID, `amdxdna` driver, PnP |
| Qualcomm | Hexagon NPU | PnP / `qaic` |

NPU **benchmarking** goes through ONNX Runtime execution providers, so one
model measures whichever accelerator a machine has — OpenVINO (Intel), Vitis AI
(AMD), QNN (Qualcomm), DirectML, CUDA, ROCm, or Core ML:

```bash
pip install onnxruntime      # or onnxruntime-openvino / -directml / -qnn
```

```
NPU — cross-vendor (ONNX Runtime)
  Runtime       : onnxruntime 1.29.0
  Model         : 10x MatMul+Relu, 1024x1024, batch 32

  CPU provider (baseline)   :          2,055 inf/s  (1,379 GFLOPS)
  Apple Neural Engine / GPU :          2,189 inf/s  (1,469 GFLOPS, 1.07x)

  No accelerator beat the CPU by 1.5x — none is reported as engaged.
```

As with the Apple path, **a speedup below 1.5× is reported as "not engaged"**
rather than passing a CPU result off as NPU performance. The ONNX model is
generated by writing the protobuf directly, so only `onnxruntime` is needed —
not the `onnx` package.

## GPU and Neural Engine

Accelerator **inventory** (GPU model, cores, VRAM, driver; NPU presence) works
on all three platforms. Compute **benchmarking** is currently Apple-only, via
Metal and Core ML:

```
Accelerators — GPU / NPU
  GPU           : Apple M4 (10 cores)
  NPU           : Apple Neural Engine  (via Core ML)

  GPU FP32 FMA              :          2,322 GFLOPS
  GPU FP16 FMA              :          2,505 GFLOPS
  GPU memory bandwidth      :         77,946 MB/s
  GPU kernel launch latency :          133.9 us
  Core ML CPU-only          :          416.7 inferences/s
  Neural Engine             :          2,535 inferences/s
  Neural Engine throughput  :          9,186 GFLOPS

  Neural Engine ENGAGED — 6.08x vs CPU-only Core ML
```

The Neural Engine **cannot be programmed directly** — no public API accepts
arbitrary work. Core ML alone decides placement, so the tool runs the same
model under `cpuOnly` and `cpuAndNeuralEngine` and reports the speedup, which
is the only honest evidence the ANE was actually used. Below 1.5x it says so
rather than presenting a CPU result as an NPU one.

The Core ML model is **generated at runtime** by writing the `.mlmodel`
protobuf directly, so no `coremltools` install is needed. It is deliberately
large (64 channels, 64x64, 12 conv layers) because Core ML keeps small models
on the CPU — a 16-channel model measured here never left the CPU at all.

Requires only the Command Line Tools: the Metal shaders are compiled at
**runtime**, avoiding the offline `metal` compiler that ships only with full
Xcode.

### Matrix-multiply TFLOPS (the AI-compute metric)

The GPU section reports dense **GEMM throughput** via MetalPerformanceShaders —
the single operation that dominates neural-network compute. On an M4: **2.8
TFLOPS FP32 / 3.2 TFLOPS FP16**. It also reports Neural Engine **tail latency**
(p50/p99), which is what governs interactive/real-time inference.

## AI training & inference (optional framework tier)

Real *training* needs backprop and an optimizer, which needs a real ML
framework. This is the **only** part of the tool that will use a third-party
dependency — and only if you already have one installed:

```bash
pip install torch          # or: pip install onnxruntime
python3 benchmark.py --ai
```

When PyTorch is present it trains a small CNN and reports **training
samples/sec** and **inference samples/sec**, automatically using CUDA (NVIDIA),
ROCm (AMD), MPS (Apple), or CPU — so this path also covers non-Apple GPUs that
the Metal engine cannot. ONNX Runtime is used as an inference-only fallback.
Without any framework the section is skipped with a one-line install hint; the
rest of the run is unaffected.

## Power & perf-per-watt

Two chips can post the same throughput while one draws triple the power. The
tool samples package power **under load** and reports **score-per-watt**:

- **macOS**: real watts via `powermetrics` (run with `sudo`), else a labelled
  TDP estimate.
- **Linux**: real watts via RAPL (`/sys/class/powercap`).
- Otherwise: a clearly-labelled TDP-class estimate — never presented as a
  measurement.

## Regression detection

Run it more than once and it becomes a monitor. Each run is compared against
**this machine's own history** (median of prior runs) and flags any metric that
moved beyond a threshold:

```
Regression Check
  Compared against 6 prior run(s) on this machine (threshold ±10.0%):
    ▼ Disk write          -34.2%  (5,120 → 3,370)
  ⚠ 1 metric(s) regressed. Check cooling, background load, or hardware health.
```

Catches a failing SSD, a clogged cooler, or a driver regression that a single
run would never show. Metrics that depend on run settings (disk size, memory
buffer) are compared **only against runs that used the same settings**, so
changing a flag never masquerades as failing hardware.

## Network stack

A TCP **loopback** throughput + latency (p50/p99) benchmark — characterizes the
OS network stack with **no external traffic**. A slow number points at CPU
saturation or an intercepting security agent.

## Native engine extras

The native C engine adds **multi-threaded CPU** and **pointer-chase memory
latency**, which maps the cache hierarchy precisely:

```
   16 KB :    0.92 ns     ← L1
  256 KB :    3.46 ns     ← L2
   16 MB :   12.56 ns
   64 MB :   76.23 ns     ← DRAM
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Invalid arguments |
| `3` | Refused: machine state would distort results (use `--force`) |
| `4` | **Validation failure — hardware may be unstable** |

## Native engine

Auto-compiled on first run. Build manually if you prefer:

```bash
cc -O2 native_engine.c -o native_engine -lm -lpthread    # macOS / Linux
gcc -O2 native_engine.c -o native_engine.exe             # Windows, MinGW
cl /O2 native_engine.c                                   # Windows, MSVC
```

No compiler? That section is skipped; everything else still runs.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

204 tests, standard library only (they run with or without the optional tiers).

## Documentation

Full reference docs in [`docs/`](docs/README.md):

- [architecture.md](docs/architecture.md) — design, module layout, data flow
- [technical.md](docs/technical.md) — methodology, units, statistics, scoring
- [functions.md](docs/functions.md) — per-function reference
- [packages.md](docs/packages.md) — dependencies, toolchain, CLI
- [glossary.md](docs/glossary.md) — every acronym and unit explained
- [ml-algorithms.md](docs/ml-algorithms.md) — every ML algorithm, with equations
- [safety.md](docs/safety.md) — hardware-safety audit and resource caps
- [troubleshooting.md](docs/troubleshooting.md) — common problems and fixes

## Requirements

- **Python 3.8+** (standard library only; `psutil` optional for richer detection)
- A C compiler is **optional** — only for the native engine
