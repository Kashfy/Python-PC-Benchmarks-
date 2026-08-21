# Technical Reference

Measurement methodology, units, and scoring — what the numbers actually mean.

## Timing

All Python timing uses `time.perf_counter()`, the highest-resolution monotonic
clock available, which is not affected by wall-clock adjustments (NTP, DST).
The C engine uses `clock_gettime(CLOCK_MONOTONIC)` on POSIX and
`QueryPerformanceCounter` on Windows — both monotonic, high-resolution clocks.

## The timed-loop model

Every CPU/memory test runs a small unit of work (a "chunk") repeatedly until a
target duration elapses, then divides total work by elapsed time:

```
rate = (chunks_run × work_units_per_chunk) / elapsed_seconds
```

Because the loop runs for a fixed *time* rather than a fixed *count*, fast and
slow machines both produce a statistically meaningful sample without the run
taking wildly different lengths.

## What each benchmark measures

### CPU Integer — `primes/s`

Tests primality for every integer in `[50000, 51000)` per chunk (1000
integers) using trial division up to √n. This stresses the integer ALU,
division, branch prediction, and instruction dispatch. Reported as **primes
tested per second**.

- Chunk constant: `PRIMES_PER_CHUNK = 1000`
- Deterministic: the same range every time, so cost is identical across
  machines and runs.

### CPU Float — `iters/s`

Runs 50,000 iterations per chunk of `s += sin(x)*cos(x) + sqrt(x)`. This
exercises the FPU and the platform math library (`libm` / MSVC CRT). Reported
as **iterations per second**.

- Chunk constant: `FLOAT_ITERS_PER_CHUNK = 50000`
- Note: `sin`/`cos`/`sqrt` are library calls, not single hardware FLOPs, so
  this is an honest "math-iterations/s" figure rather than a synthetic MFLOPS
  number.

### CPU Multi-core — `primes/s` + scaling factor

Spawns one worker process per **logical core** (`os.cpu_count()`), each running
the integer workload for the target duration, then sums their throughput:

```
aggregate_rate = total_primes_across_workers / wall_seconds
scaling_factor = aggregate_rate / single_core_rate
```

The scaling factor shows real-world parallel efficiency. It is usually below
the logical-core count because of:

- **Hybrid cores** (e.g. Apple P+E cores, Intel P/E cores) — efficiency cores
  are slower, so N cores rarely give N× the single-P-core rate.
- **SMT/Hyper-Threading** — two logical threads on one physical core don't
  double throughput.
- **Process spawn overhead** — included in wall time; more visible on very
  short runs. Use `--seconds 5+` for a cleaner scaling number.

### Memory — `MB/s`

Allocates two buffers of `--mem-mb` MB (default 64) and repeatedly copies one
into the other via bytearray slice assignment (a real `memmove` in CPython;
`memcpy` in the C engine). Reported as **copy bandwidth in MB/s** — bytes moved
per second. A copy touches both a read and a write stream, so this reflects
sustained memory subsystem bandwidth, not just latency.

### Disk — write `MB/s` and read `MB/s`

Writes a `--disk-mb` MB file (default 256) in 4 MiB chunks, then reads it back:

- **Write**: timed from first write to after `fsync()` / flush, so the number
  reflects data actually pushed toward the device, not just buffered in RAM.
- **Read**: seeks to start and reads the whole file. Before reading, the tool
  best-effort drops the OS page cache (`posix_fadvise(..., DONTNEED)` on
  Linux). On macOS/Windows the read may still be partly served from cache, so
  **treat read numbers as a floor**, not a ceiling.
- A disk-space guard skips the test if free space < 1.2× the file size.

`1 MB = 1024 × 1024 bytes` throughout.

## Statistics

Each test runs `--repeats` times (default 3). The reported headline `rate` is
the **median** of the repeats, which resists one-off outliers (thermal spikes,
a background process). The full result also carries mean, standard deviation,
min, max, and the raw samples, so you can judge stability:

| Field | Meaning |
|-------|---------|
| `median` | Reported headline value (robust to outliers) |
| `mean`   | Arithmetic average |
| `stdev`  | Sample standard deviation (0 if only one repeat) |
| `min` / `max` | Range across repeats |
| `samples` | Every per-repeat rate |

A large `stdev` relative to the median signals an unstable measurement — retry
with more `--seconds`/`--repeats` or on a quieter machine.

## Scoring

Each measured rate is normalized against a fixed baseline where **baseline =
100**:

```
subscore = 100 × measured_rate / baseline_rate
```

The **composite** is the geometric mean of the subscores:

```
composite = exp( mean( ln(subscore_i) ) )
```

The geometric mean is used (rather than arithmetic) so that no single category
dominates and a machine must be well-rounded to score high; it also keeps the
composite meaningful when subscores span very different magnitudes.

### Baseline constants

Defined in `BASELINES` in `benchmark.py`. They are **arbitrary but stable** —
roughly a mid-range 2020-era laptop core — chosen only to be a common yardstick.

| Metric | Baseline |
|--------|----------|
| `cpu_int_primes_per_s` | 2,000,000 |
| `cpu_float_iters_per_s` | 3,000,000 |
| `cpu_multi_primes_per_s` | 8,000,000 |
| `mem_copy_mb_per_s` | 6,000 |
| `disk_write_mb_per_s` | 500 |
| `disk_read_mb_per_s` | 1,000 |

Changing a baseline shifts every score computed with it, so keep them fixed if
you want to compare against historical runs. If you re-baseline, note the tool
version alongside your data.

## Chip-architecture normalization

`platform.machine()` returns OS-specific spellings; the tool maps them to a
canonical ISA family so results line up across operating systems:

| Reported `machine` values | `arch_family` |
|---------------------------|---------------|
| `x86_64`, `amd64`, `x64` | `x86-64` |
| `i386`–`i686`, `x86` | `x86-32` |
| `arm64`, `aarch64`, `arm64e` | `ARM64` |
| `armv7l`, `armv6l`, other `arm*` | `ARM32` |
| `riscv64` / `riscv*` | `RISC-V 64` / `RISC-V 32` |
| `ppc64*`, `powerpc64*` | `PowerPC 64` |
| `s390*` | `IBM Z` |
| `mips*` | `MIPS` |

`arch_bits` (32/64) is derived from `sys.maxsize`, and `byte_order` from
`sys.byteorder`.

## Reproducibility tips

- Close background apps; plug in laptops (power profiles throttle on battery).
- Let the machine cool between runs to avoid thermal-throttling skew.
- Use `--seconds 5 --repeats 5` for publishable numbers.
- Compare like-for-like: same tool version, same `--disk-mb`/`--mem-mb`, same
  storage device for the disk test.
