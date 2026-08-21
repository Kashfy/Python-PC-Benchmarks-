# PC Benchmark & Diagnostics

A reliable, cross-platform benchmark and hardware-diagnostics tool for
**Windows, macOS, and Linux** on **x86-64, ARM64, and other** CPU
architectures.

It measures CPU, memory, disk, **GPU, and NPU** (including the **Apple Neural
Engine**) in meaningful, comparable units, detects **thermal throttling** under
sustained load, **validates** that the hardware computes correct results,
gathers a full hardware inventory, and records everything to JSON/CSV/HTML so
you can compare machines over time.

- **Pure Python standard library** — runs on any machine with Python 3.8+,
  **no `pip install` required**.
- **Optional native C engine** — auto-compiled for compiler-optimized numbers
  and real memory-latency measurement.

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
| `--force` | off | Run despite distorting machine state |

Tests: `cpu_int`, `cpu_float`, `cpu_multi`, `compression`, `hashing`, `json`,
`memory`, `cache_sweep`, `disk`.

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

84 tests, standard library only.

## Documentation

Full reference docs in [`docs/`](docs/README.md):

- [architecture.md](docs/architecture.md) — design, module layout, data flow
- [technical.md](docs/technical.md) — methodology, units, statistics, scoring
- [functions.md](docs/functions.md) — per-function reference
- [packages.md](docs/packages.md) — dependencies, toolchain, CLI
- [troubleshooting.md](docs/troubleshooting.md) — common problems and fixes

## Requirements

- **Python 3.8+** (standard library only; `psutil` optional for richer detection)
- A C compiler is **optional** — only for the native engine
