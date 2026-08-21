# Technical Reference

Measurement methodology, units, and scoring — what the numbers actually mean.

## Timing

Python uses `time.perf_counter()`; the C engine uses
`clock_gettime(CLOCK_MONOTONIC)` on POSIX and `QueryPerformanceCounter` on
Windows. All are monotonic and high-resolution, so NTP corrections and DST
changes cannot corrupt a measurement.

## The timed-loop model

Each CPU/memory test runs a fixed unit of work (a "chunk") repeatedly until a
target duration elapses:

```
rate = (chunks × work_units_per_chunk) / elapsed_seconds
```

Running for a fixed *time* rather than a fixed *count* means fast and slow
machines both produce a meaningful sample without the run length varying
wildly.

### Warm-up

Before any timing, each test runs untimed for ~15% of its target duration
(clamped to 0.05–1.0 s). This absorbs cold caches, CPU frequency ramp-up,
branch-predictor training, and lazy imports. Without it the first repeat is
systematically slower, which both depresses the result and inflates the
variance.

## What each benchmark measures

| Test | Unit | Stresses |
|------|------|----------|
| **CPU Integer** | primes/s | Integer ALU, division, branch prediction |
| **CPU Float** | iters/s | FPU and the platform math library |
| **CPU Multi-core** | primes/s | All logical cores + scaling factor |
| **Compression** | MB/s | zlib level 6 compress+decompress round-trip |
| **Hashing** | MB/s | SHA-256 — reaches hardware crypto instructions |
| **JSON parse** | MB/s | Parser and allocator throughput |
| **Memory** | MB/s | Sustained copy bandwidth (`memmove`) |
| **Cache sweep** | MB/s | Bandwidth vs. working-set size |
| **Disk** | MB/s + IOPS | Sequential write/read and 4 KiB random reads |

### Why the real-world workloads matter

The synthetic loops measure one execution unit each. Compression, hashing, and
JSON parsing exercise the paths real software actually takes — mixed integer
and branch work, allocator pressure, and memory traffic.

SHA-256 is the most architecturally revealing of the three: it dispatches to
hardware crypto instructions (ARMv8 crypto extensions, x86 SHA-NI) where they
exist. A chip with hardware SHA can be an order of magnitude faster than one
without, which no amount of prime-testing would show.

The compression corpus is generated from a **fixed seed**, so every machine
compresses byte-identical input. It is deliberately semi-repetitive text rather
than random bytes — random data is incompressible, which would make the zlib
result meaningless.

### CPU Multi-core and the scaling factor

One process per logical core, each timed independently, summed:

```
aggregate_rate = total_primes / wall_seconds
scaling_factor = aggregate_rate / single_core_rate
```

Scaling is normally well below the core count, for real reasons:

- **Hybrid cores** — Apple P+E and Intel P/E clusters mix fast and slow cores,
  so N cores never give N× the single-P-core rate. An Apple M4 (4P + 6E)
  lands near 5× on 10 cores.
- **SMT / Hyper-Threading** — two logical threads on one physical core do not
  double throughput.
- **Process spawn overhead** — counted in wall time; dominant on short runs.
  Use `--seconds 5` or more for a clean scaling figure.

The C engine's threaded equivalent avoids the GIL and spawn cost entirely, so
comparing the two isolates interpreter overhead from hardware limits.

### Memory: bandwidth vs. latency

These are different properties and both matter.

**Bandwidth** (Python) — repeatedly copying a large buffer. The cache sweep
repeats this across growing working sets; bandwidth stays high while the set
fits in cache and steps down at each boundary.

The sweep starts at **128 KB**. Below that a single copy finishes faster than
the interpreter overhead of issuing it, which flattens and then *inverts* the
curve — measuring Python rather than the cache. Resolving L1 requires the
native engine.

**Latency** (C engine only) — a pointer chase over a randomly permuted cycle
built with **Sattolo's algorithm**, guaranteeing a single cycle that visits
every slot before repeating so the prefetcher cannot predict the next address.
Each load depends on the previous one, so the result is true access latency:

```
   16 KB :  0.92 ns    ← L1
   64 KB :  0.92 ns
  256 KB :  3.46 ns    ← L2
    1 MB :  4.73 ns
   16 MB : 12.56 ns
   64 MB : 76.23 ns    ← DRAM
```

This cannot be done meaningfully in Python: one CPython bytecode costs tens of
nanoseconds, an order of magnitude more than the ~1 ns L1 hit it would be
trying to measure.

### Disk, and defeating the page cache

Sequential throughput characterizes bulk transfer; **4 KiB random-read IOPS** is
what actually determines how responsive a machine feels, since real workloads
are dominated by small scattered reads.

The hard part is preventing the OS from serving reads out of RAM:

| Platform | Mechanism | Effective? |
|----------|-----------|-----------|
| macOS | `fcntl(fd, F_NOCACHE, 1)` **before writing** | yes |
| Linux | `posix_fadvise(DONTNEED)` after `fsync` | yes |
| Windows | none applied | no — reads are an optimistic upper bound |

The ordering on macOS is essential. `F_NOCACHE` stops *new* I/O from populating
the cache but does not evict pages already there, so enabling it after the
write would leave the whole file cached. Setting it at creation is the
difference between measuring the SSD and measuring RAM — concretely, on an
Apple M4 this moved random reads from an impossible **1,063,411 IOPS** down to
a realistic **37,994**, and sequential reads from 13,836 MB/s to 3,763 MB/s.

The result carries `cache_bypassed`, and the console prints a warning when the
cache could not be bypassed, so an inflated number is never presented as clean.

Writes are timed through `fsync()`, so they reflect data pushed toward the
device rather than buffered in RAM.

`1 MB = 1024 × 1024 bytes` throughout.

## GPU and NPU benchmarking

### What is and isn't covered

| Capability | Windows | macOS | Linux |
|------------|---------|-------|-------|
| GPU inventory (model, VRAM, driver) | yes | yes | yes |
| NPU presence detection | yes | yes | yes |
| GPU compute benchmark | no | **yes** (Metal) | no |
| NPU compute benchmark | no | **yes** (Core ML) | no |

Compute benchmarks need vendor frameworks — CUDA, ROCm, oneAPI, or an OpenCL
runtime. Writing those paths without the hardware to test them would produce
code that looks supported and silently misreports, which is worse than an
honest gap. Non-Apple systems get full inventory and a stated limitation.

### GPU (Metal)

Four measurements, all from compute kernels compiled **at runtime** via
`newLibraryWithSource:` so only the Command Line Tools are needed — the offline
`metal` compiler ships solely with full Xcode.

| Measurement | Method |
|-------------|--------|
| **FP32 / FP16 FMA** | Four independent dependent FMA chains per thread over 1M threads. Independent chains give the scheduler instruction-level parallelism; dependent *within* a chain stops the compiler collapsing the loop. Results are written out so nothing is dead code. |
| **Memory bandwidth** | 256 MB `float4` copy; counts one read plus one write per element. |
| **Kernel launch latency** | Minimal kernel with a full commit/`waitUntilCompleted` round trip — this is dispatch overhead, not GPU work. |

On Apple GPUs FP16 is only modestly faster than FP32 (measured 2,505 vs 2,322
GFLOPS on an M4) because the ALUs run both at similar rates; unlike some
discrete GPUs there is no 2x half-precision path.

Note that launch latency is a **lower-is-better** figure and is deliberately
excluded from scoring, where every other metric is higher-is-better.

### Neural Engine (Core ML)

The ANE is **not directly programmable**. No public API accepts arbitrary
work — Core ML alone decides whether a model runs on CPU, GPU, or ANE. Two
consequences shape the design:

**1. The benchmark must be a model.** `pcbench/coreml_model.py` writes the
`.mlmodel` protobuf byte by byte rather than depending on `coremltools`,
preserving the zero-dependency guarantee. It emits a convolution stack, since
convolution is what the ANE is built for and what Core ML most reliably
offloads.

**2. Placement must be inferred.** Core ML never reports where a model ran, so
the same model is timed under `MLComputeUnitsCPUOnly` and
`MLComputeUnitsCPUAndNeuralEngine`, and the ratio is the evidence:

```
Core ML CPU-only          :   416.7 inferences/s
Neural Engine             : 2,535.0 inferences/s
Neural Engine ENGAGED — 6.08x vs CPU-only Core ML
```

Below **1.5x** the tool reports that the ANE did *not* engage rather than
presenting a CPU number as an NPU result.

**Model size is load-bearing.** Core ML keeps small models on the CPU because
dispatch overhead exceeds the work. Measured on an M4: a 16-channel 32x32
model ran at **0.92x** of CPU-only speed — it never left the CPU — while the
64-channel 64x64 12-layer default reaches **6.08x**. The defaults are large
deliberately.

Effective throughput is derived from the model's known arithmetic cost:

```
FLOPs per layer = out_ch x in_ch x K x K x H x W x 2      (MAC = 2 FLOPs)
effective GFLOPS = inferences/s x total FLOPs / 1e9
```

At 2,535 inferences/s over 3.62 GFLOP that is ~9,190 GFLOPS effective.

### Matrix multiply (GEMM) — the AI-compute number

Dense matrix multiply is what every fully-connected and convolution layer
reduces to, so its sustained TFLOPS is the most meaningful single "AI
performance" figure for a GPU. The engine uses `MPSMatrixMultiplication` (a
vendor-tuned kernel), so it measures the hardware, not our shader-writing.
A dense N×N×N multiply is 2·N³ FLOPs; several are batched per command buffer so
submission overhead doesn't dominate. Reported for FP32 and FP16 (measured 2.8
/ 3.2 TFLOPS on an M4).

### Neural Engine tail latency

Alongside throughput the ANE path records **p50 and p99 per-inference
latency**. Tail latency is what governs real-time inference: a good average
with a bad p99 still produces visible stutter in interactive use.

## Machine-learning workloads (pure Python)

Three classic ML algorithms run with only the standard library, so every
machine produces real ML numbers without installing anything.

| Workload | Unit | What it exercises |
|----------|------|-------------------|
| **Neural net training** | steps/s | Forward pass, backpropagation, SGD updates |
| **K-means clustering** | distances/s | Distance computation, cache locality |
| **K-NN search** | comparisons/s | Brute-force similarity search |

### Neural network training

A genuine multi-layer perceptron (32-24-4, batch 24) with tanh hidden units and
a softmax/cross-entropy output. Each step runs a full forward pass, computes
gradients by backpropagation, and applies an SGD update — the weights really do
change and the loss really does fall.

Reported as training **steps/s**, plus derived **samples/s** and **MFLOPS**
(forward + backward ≈ 3× the forward cost).

*Validation:* before timing, the network trains for 60 steps and the loss must
drop to below 90% of its starting value. On healthy hardware it always does, so
a failure indicates a floating-point or memory fault.

### K-means clustering

Lloyd's algorithm over 1,200 points in 8 dimensions with k=6.

Seeding matters more than it looks. Random seeding frequently draws two
centroids from the same blob, leaving the algorithm in a poor local optimum —
the clustering then fails to converge and the validation check cannot tell a
bad seed from a hardware fault. Using **farthest-point (maximin) init** makes
seeding deterministic and lands one centroid per well-separated cluster:
measured inertia drops from 20.5 to **2.867 per point against a theoretical
ideal of 2.88**. Only then is a high inertia meaningful evidence of a fault.

### K-nearest neighbours

Brute-force k-NN: 40 queries against 900 reference points in 12 dimensions,
k=5 — the operation underneath vector databases and retrieval.

*Validation:* every reference point must be returned as its own nearest
neighbour.

All three use fixed seeds, so the datasets are byte-identical on every machine
and results are directly comparable. Being pure Python they measure the
interpreter as well as the silicon — which is the point: they compare machines
running the same Python, while the native and framework tiers cover compiled
speed separately.

## Cross-vendor NPU via ONNX Runtime

The Apple Neural Engine has a dedicated Core ML path. Every other vendor is
reached through **ONNX Runtime execution providers**, which is the only runtime
that spans them all:

| Provider | Hardware |
|----------|----------|
| `OpenVINOExecutionProvider` | Intel AI Boost NPU (`device_type: NPU`) |
| `VitisAIExecutionProvider` | AMD Ryzen AI NPU (XDNA) |
| `QNNExecutionProvider` | Qualcomm Hexagon NPU |
| `DmlExecutionProvider` | Any DirectML/DX12 device |
| `CoreMLExecutionProvider` | Apple ANE/GPU |
| `CUDA` / `TensorRT` / `ROCm` / `MIGraphX` | Discrete GPUs |

### Methodology

Identical to the ANE approach, and for the same reason — no runtime reliably
reports where an operator executed:

1. Time the model on `CPUExecutionProvider` to establish a baseline.
2. Time it on each accelerator provider.
3. Report the speedup; mark the device **engaged** only above **1.5×**.

Two failure modes are handled explicitly. ONNX Runtime **silently falls back to
CPU** when a provider cannot load, so the code checks `sess.get_providers()`
and rejects the result if the requested provider is not actually active —
otherwise a CPU number would be published as an NPU result. And OpenVINO
defaults to the CPU device unless `device_type: NPU` is passed, so that option
is set explicitly.

### The model

A stack of 10 `MatMul` + `Relu` layers, 1024×1024, batch 32 (0.67 GFLOP per
inference). Matrix multiply is what NPUs are built for, and neither operator
takes attributes, which keeps the encoder small.

Two details make it work:

- **One shared weight tensor.** ONNX permits several nodes to consume the same
  initializer, so the file stays at 4 MB instead of 4 MB × layer count.
- **Weights scaled by 1/dim.** Without it each layer multiplies activation
  magnitude by roughly `dim`, and a deep stack overflows to infinity — the
  model would return NaN rather than a timing.

The model is generated by writing the ONNX protobuf wire format directly
(:mod:`pcbench.onnx_model`), so the benchmark needs only `onnxruntime`, not the
`onnx` package. **This was verified**: ONNX Runtime 1.29 loads the hand-written
model and runs it correctly.

## AI training & inference (optional framework tier)

Raw compute (matmul, Core ML inference) is measured with zero dependencies. But
real **training** — forward pass, backpropagation, optimizer step — cannot be
expressed without an ML framework. Rather than fake it, the tool runs a real
one *if the user has it installed*:

- **PyTorch** (preferred): trains a small but representative CNN and reports
  **training samples/s** and **inference samples/s**. It auto-selects the best
  backend — CUDA (NVIDIA), ROCm (AMD), MPS (Apple), or CPU — so this is also
  the only path that benchmarks non-Apple GPUs.
- **ONNX Runtime** (fallback): inference only (it does not train), reported
  honestly as such.
- **Neither installed**: the section is skipped with a one-line `pip install`
  hint; nothing else is affected.

GPU frameworks are asynchronous, so each measured region ends with a device
`synchronize()` — otherwise the timer would measure how fast work is *queued*,
not how fast it *runs*. A warm-up pass covers cuDNN/Metal autotuning.

This is the **only** third-party dependency the tool will ever use, and never
without the user opting in by installing a framework.

## Power & perf-per-watt

Two chips at equal throughput can differ threefold in power, so efficiency is
often the real story. Power is sampled **under load** (a background CPU burn
runs while the reading is taken) because idle draw is uninteresting.

| Source | Platform | Confidence |
|--------|----------|-----------|
| `powermetrics` (CPU/GPU/ANE watts) | macOS, **needs sudo** | measured |
| RAPL (`/sys/class/powercap`) | Linux (Intel/AMD) | measured |
| Package-TDP lookup | any | **estimated**, always labelled |

`score_per_watt` is the composite score divided by watts. An estimate is never
presented as a measurement — the `source` field and an `(estimated)`/
`(measured)` tag always say which.

## Regression detection

Run once and you have a number; run repeatedly and you have a monitor. Each run
is compared against the **median of this machine's prior runs** (same hostname
only — cross-machine differences belong in `compare`). Any metric that moves
more than the threshold (default ±10%, `--regression-threshold`) is flagged,
regressions first. The median baseline resists a single noisy prior run
becoming a false reference, and the current run is excluded from its own
baseline. This is what turns the tool into a health check: a failing SSD,
clogged cooler, or driver regression shows up as a slowdown against the
machine's own past.

### Comparing only comparable runs

Some metrics depend on the settings used, so a changed flag would otherwise
look like failing hardware. Observed in practice: a `--quick` run (64 MB disk
test) followed by a default run (256 MB) produced a **-40.9% "disk regression"**
that was entirely an artifact — larger files exhaust an SSD's SLC write cache,
so throughput legitimately falls.

The run settings are therefore recorded in the CSV (`cfg_disk_mb`,
`cfg_mem_mb`), and metrics that depend on them are compared **only against
prior runs that used the same value**:

| Metric | Governed by |
|--------|-------------|
| Disk write / read / IOPS | `--disk-mb` |
| Memory bandwidth | `--mem-mb` |

Everything else (CPU, ML, hashing, composite) is setting-independent and always
compared. When no prior run used the same settings, the affected metrics are
skipped and the report says how many — rather than reporting a false
regression.

## Network stack

A TCP **loopback** (127.0.0.1) benchmark: bulk throughput plus ping/pong
round-trip latency (p50/p99) with `TCP_NODELAY` set so Nagle's algorithm
doesn't hide per-message latency. It deliberately sends nothing off-box — it
characterizes the OS network stack, socket buffers, and scheduler, which is a
real reproducible machine property, without depending on an internet connection
or contacting third parties. A slow loopback number is itself diagnostic (CPU
saturation, or a security agent intercepting local traffic).

### Accelerator scoring

GPU, NPU, matmul, and (when present) AI-framework results feed the same
baseline-relative scoring as everything else and roll up into `GPU`, `NPU`, and
an `AI` category (matmul + ANE + framework). Machines **without** a given piece
of hardware simply omit those subscores rather than scoring zero, so absent
hardware never drags the composite down. Power and network are reported but
kept **out** of the composite — power is an efficiency axis, and loopback
network is a diagnostic rather than a comparative performance metric.

## Core scaling analysis

Aggregate multi-core throughput hides a chip's shape. The tool measures
aggregate throughput at 1, 2, … N workers and examines the *marginal* gain of
each added worker.

**What is reported:** how far scaling stays near-linear, whether the machine is
hybrid, and the throughput ratio between the fast and slow groups. All three
are stable across runs.

**What is deliberately not reported:** exact performance/efficiency core
counts. An earlier version estimated them and the estimate proved unreliable —
the last performance core shares cache and memory bandwidth with its siblings,
so its marginal contribution falls to roughly the level of the first efficiency
core. On an Apple M4 (truly 4P + 6E) the sorted gains are

```
4.42, 4.27, 4.22, 2.56 | 2.23, 1.40, 1.38, 1.26, 0.81  M primes/s
```

and the real boundary between 2.56 and 2.23 is indistinguishable from ordinary
scaling loss; successive runs "detected" 4/6, 3/7 and 8/2. `per_core_map` pins
work to each core for an exact answer on Linux and Windows, which expose thread
affinity; macOS does not.

Two measurement details mattered. Aggregate rate is computed from the workers'
own timed duration rather than wall time, because wall time includes pool
creation, which grows with worker count and would understate exactly the higher
counts the analysis depends on. And a monotonic envelope was tried and
rejected: clamping dips upward introduced artificial zero marginals that
shifted the analysis further.

## Storage queue depth

Issuing one 4 KiB read at a time bounds the result by *latency*, not by the
device. Measured on an Apple M4 SSD:

| Queue depth | IOPS |
|-------------|------|
| 1 | 49,405 |
| 4 | 152,413 |
| 16 | 225,912 |
| 32 | 231,746 |

The single-request figure understates the drive **4.7×**. Concurrency is
produced with threads rather than async I/O because `os.pread` releases the GIL
for the duration of the syscall, so they genuinely overlap.

One implementation detail was load-bearing: the main thread must *wait on an
event*, not spin. An initial busy-wait loop held the GIL and starved the reader
threads, collapsing QD4 to 624 IOPS — an eightieth of the correct value.

Latency percentiles (p50/p99/max) are reported alongside, because a drive with
a good median and a poor tail produces visible stalls that throughput cannot
show.

## Memory bandwidth scaling

A single core rarely saturates a memory controller. The tool sweeps
concurrency and reports the peak.

**Processes, not threads.** CPython does not release the GIL during
`bytearray` slice assignment, so a threaded version serializes: it reported a
flat 40 GB/s at every thread count, which was the GIL speaking rather than the
memory subsystem. With processes an M4 shows 39.7 GB/s at one process rising
to 56.6 GB/s at five — **1.43×**, confirming one core does not saturate it.

The buffer must also exceed last-level cache, or the test measures cache
bandwidth instead; per-process size is derived from the safe memory budget
divided by peak concurrency, with a 32 MB floor.

## System-level benchmarks

Throughput misses a class of behaviour: a machine can post excellent FLOPS and
still feel sluggish.

| Measurement | Unit | Why it matters |
|-------------|------|----------------|
| **Compile** (C at `-O2`) | seconds | Exercises preprocessor, optimiser, and linker together — the most relatable number for a developer |
| **Syscall latency** | ns | The floor for every I/O the machine performs |
| **Context switch** | ns | Shows up as poor responsiveness under load |
| **Process spawn** | ms | Dominates build systems, shell scripting, and CI |

The compile benchmark runs one untimed compile first so the compiler binary and
headers are cached; otherwise the first run measures the filesystem.

**Not included: AES throughput.** Python's standard library has no AES
primitive, and a pure-Python implementation would never reach the AES-NI or ARM
crypto instructions that make such a benchmark meaningful — it would measure
table lookups instead. Reporting nothing is better than reporting the wrong
thing. SHA-256 already covers hardware crypto.

## Statistics

Each test runs `--repeats` times; the headline `rate` is the **median**, which
resists a single bad repeat. The full record carries:

| Field | Meaning |
|-------|---------|
| `median` | Headline value |
| `mean` | Arithmetic average |
| `stdev` | Sample standard deviation (0 if one repeat) |
| `cv` | Coefficient of variation (stdev/mean) |
| `min` / `max` | Range |
| `samples` | Every per-repeat rate |

`cv` is normalized, so it is comparable across tests with very different
magnitudes. The console renders it as a rating:

| cv | Rating |
|----|--------|
| ≤ 2% | excellent |
| ≤ 5% | good |
| ≤ 10% | fair |
| > 10% | unstable |

An `unstable` rating means the measurement should not be trusted — increase
`--seconds`/`--repeats` or quiet the machine.

## Validation — benchmark as diagnostic

Every workload verifies its own output against a known-correct result:

| Workload | Check |
|----------|-------|
| CPU Integer | exactly **89** primes in `[50000, 51000)` |
| CPU Float | sum ≈ **35173.9049856305** (relative tolerance 1e-9) |
| Compression | zlib round-trip CRC matches |
| Hashing | SHA-256 digest matches |
| JSON | parsed element count matches |
| Memory | copied bytes match the source |
| Multi-core | every worker returns the correct prime count |

A machine that computes the *wrong answer* quickly is faulty, not fast.
Mismatches point at unstable overclocking, failing RAM, inadequate cooling, or
a miscompiled toolchain. A failure is reported prominently and sets **exit code
4** — it does not crash the run.

The float check uses a relative tolerance because platform math libraries round
transcendental functions differently in the last bits; 1e-9 is loose enough for
`libm` variation and tight enough to catch a genuine computation fault.

## Machine-state guard

Volatile conditions are captured before every run:

| Field | Source |
|-------|--------|
| `on_ac_power` | `pmset` (macOS), `/sys/class/power_supply` (Linux), `GetSystemPowerStatus` (Windows) |
| `load_average` | `os.getloadavg()` (not available on Windows) |
| `thermal` | `pmset -g therm` CPU speed limit (macOS), `/sys/class/thermal` (Linux) |

A run **stops with exit code 3** when any of these would distort results:

- on battery — laptops down-clock aggressively,
- load per core > 0.30 — something else is already using the CPU,
- already thermally throttled.

`--force` overrides. This is deliberately a hard stop: a benchmark taken on
battery looks like a slower machine, and that mistake silently poisons every
later comparison.

## Sustained load and thermal droop

A short benchmark measures burst performance only. `--sustained 5m` runs
continuous load, samples throughput per window, and reports:

```
peak      = best window
sustained = mean of the final 25% of windows   (thermals have equilibrated)
droop     = (1 - sustained/peak) × 100
```

| Droop | Verdict |
|-------|---------|
| < 5% | no meaningful throttling |
| < 15% | mild — typical well-cooled laptop |
| < 30% | moderate — sustained work noticeably slower |
| ≥ 30% | heavy — cooling is the limiting factor |

## Scoring

Each rate is normalized against a fixed baseline (**baseline = 100**):

```
subscore  = 100 × measured_rate / baseline_rate
composite = exp( mean( ln(subscore_i) ) )        # geometric mean
```

The geometric mean prevents any single category from dominating — a machine
must be well-rounded to score highly — and stays meaningful when subscores span
very different magnitudes. Subscores also roll up into CPU / MEMORY / DISK
category scores.

### Baseline constants

| Metric | Baseline |
|--------|----------|
| `cpu_int` | 2,000,000 primes/s |
| `cpu_float` | 3,000,000 iters/s |
| `cpu_multi` | 8,000,000 primes/s |
| `compression` | 60 MB/s |
| `hashing` | 500 MB/s |
| `json` | 80 MB/s |
| `memory` | 6,000 MB/s |
| `disk_write` | 500 MB/s |
| `disk_read` | 1,000 MB/s |
| `disk_iops` | 20,000 IOPS |

These are arbitrary but **stable** — roughly a mid-range 2020-era laptop core.
Their only job is to be a fixed yardstick, so changing one invalidates
comparisons against previously recorded runs.

## Chip-architecture normalization

`platform.machine()` is OS-specific, so values are mapped to a canonical family
— otherwise the same chip ranks as two different architectures depending on the
OS:

| Reported values | `arch_family` |
|-----------------|---------------|
| `x86_64`, `amd64`, `x64` | `x86-64` |
| `i386`–`i686`, `x86` | `x86-32` |
| `arm64`, `aarch64`, `arm64e` | `ARM64` |
| `armv7l`, other `arm*` | `ARM32` |
| `riscv64` / `riscv32` | `RISC-V 64` / `RISC-V 32` |
| `ppc64*` / `ppc*` | `PowerPC 64` / `PowerPC 32` |
| `s390*` | `IBM Z` |
| `mips*`, `loongarch*` | `MIPS`, `LoongArch` |

`arch_bits` comes from `sys.maxsize`, `byte_order` from `sys.byteorder`.

## Reproducibility checklist

- Plug in; close background applications (the state guard enforces both).
- Let the machine cool between runs.
- Use `--seconds 5 --repeats 5` for publishable numbers.
- Compare like-for-like: same tool version, same `--disk-mb`/`--mem-mb`, same
  storage device.
- Check `cache_bypassed` before trusting disk read figures.
