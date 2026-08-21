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
