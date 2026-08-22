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

## Bottleneck analysis

Every number is only useful if it answers "so what?". The tool compares each
subsystem against the machine's **own** median and names the weak one:

```
Bottleneck Analysis
    MEMORY    ██████████████████████████████   660.4  ← strongest
    NPU       ███████████████████              421.6
    DISK      ██████████████                   322.8
    CPU       ██████████                       227.9
    GPU       ███████                          160.0  ← bottleneck

  Verdict: gpu is well below this machine's own average
    • gpu: graphics and GPU compute are the weak point
```

`--spec-sheet` writes the same findings as a one-page Markdown summary.

## Plugins — add your own benchmark

Drop a file in `plugins/` and it is discovered, timed, scored, printed, and
written to the CSV automatically:

```python
NAME = "My benchmark"
UNIT = "ops/s"
BASELINE = 1000.0        # rate corresponding to a score of 100

def run(seconds, repeats):
    return {"rate": measured_ops_per_second}
```

See [plugins/example_pi.py](plugins/example_pi.py).

## Hardware health

```bash
python3 benchmark.py --health
```

RAM integrity (six adversarial bit patterns) and read-only drive SMART. The RAM
test always states its scope: it covers only memory this process can allocate,
so a pass does **not** certify your DIMMs — use MemTest86 for that.

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

## Application workloads — "will this machine be good at my job?"

The synthetic tests isolate one subsystem each, which is right for diagnosis
and wrong for deciding whether a machine suits a task. Real software mixes
subsystems in ratios no single synthetic test reproduces, so six workloads
model the shapes real software actually has:

| Test | Unit | Models | Why it is different from the synthetic tests |
|------|------|--------|---------------------------------------------|
| `sqlite` | txn/s | Databases, API request handlers | B-tree pointer chasing and index scans; cache-latency bound, indifferent to peak bandwidth |
| `fsync` | commits/s | Database durability | Time for one flush that actually reaches the medium — the hard ceiling on write throughput |
| `raytrace` | frames/s | Rendering, physics, simulation | Branchy scalar float math on a cache-resident working set |
| `image` | MP/s | Image and video filters | Strided 2-D access that punishes a small L2 specifically |
| `logparse` | MB/s | ETL, log ingestion, build output | Linear byte scan through a backtracking regex |
| `video` | fps | Media production | Software H.264; the workload that finds inadequate cooling first |

Every one validates its own output — a renderer producing the wrong pixel or a
database returning the wrong row count reports a hardware fault, not a fast
result.

Three notes on how these are measured:

- **`fsync` is measured but never scored.** macOS needs `F_FULLFSYNC` to reach
  the medium where Linux's `fsync` suffices, and the two differ by two orders
  of magnitude on identical hardware. Scoring it would measure the operating
  system's flush semantics rather than the drive. It is reported because it is
  the single most useful storage diagnostic there is — and a result above
  100,000 commits/s is flagged, because that means the drive is acknowledging
  flushes it has not performed.
- **`video` calibrates itself.** A fixed frame count cannot serve both ends of
  the hardware range: 300 frames is under a second on a modern desktop and
  several minutes on a single-board computer. A short probe encode measures the
  machine, then the real encode is sized from it.
- **`video` needs `ffmpeg` on `PATH`** and is excluded from the default run.
  Ask for it with `--only video` or `--profile media`.

## Containers, cloud, and CI

A benchmark inside a container measures the slice the scheduler hands out, not
the hardware underneath — and the two differ by an order of magnitude. A cgroup
`cpu.max` of half a core leaves `os.cpu_count()` still reporting 16, so a naive
multicore test spawns 16 workers that fight over 0.5 cores and produce a number
that looks like catastrophic hardware failure.

Every run detects and reports:

- **Container runtime** — Docker, Podman, Kubernetes, LXC, containerd
- **cgroup limits** — CPU quota (v1 and v2), CPU affinity/cpuset, memory cap
- **Cloud provider** — from local DMI strings only; no metadata endpoint is
  ever contacted, because a diagnostics tool has no business phoning a third
  party on every run
- **CI system** — GitHub Actions, GitLab, CircleCI, Jenkins, Buildkite, and others

Workloads are then sized to the **effective** resources rather than the host's,
and the context is recorded in the CSV so a container run and a bare-metal run
never get compared as though they were the same machine.

The same machinery auto-scales for small hardware: under 4 GB of RAM the memory
and disk tests shrink so a Raspberry Pi or a minimal cloud instance is not
pushed into swap. Every adjustment is printed, because a silently different
workload is a silently incomparable result. Disable with `--no-autoscale`.

## Live monitoring — "why is my computer slow *right now*?"

Half of what people want from a benchmark tool is not a score. Monitor mode
samples the things that explain performance and names the cause:

```bash
pcbench --monitor 60s --monitor-trace slow.csv
```

```
    time     MHz     °C   CPU%   mem%   load
  ------------------------------------------
     0.0    4210   51.8     12     78   2.90
     1.1    4180   52.1     31     79   2.90
     ...

  CPU clock (MHz)    min    1400  mean  3120  max   4210  ██▇▆▄▂▁▁
  CPU temp (°C)      min    51.8  mean  84.2  max   97.6  ▁▃▅▆▇███

  - clock speed varied by 67% while peaking at 98 °C — thermal throttling;
    check airflow, dust, and thermal paste
```

The distinction it draws is the one that decides what you actually do: clocks
collapsing **with the chip hot** is thermal throttling and needs cooling;
clocks collapsing **while it stays cool** is a power or current limit and is
fixable in settings.

## Stability soak (burn-in)

A benchmark asks "how fast?". A soak asks "for how long, without getting
anything wrong?" — different questions with different answers. Hardware that
posts an excellent score can still corrupt data: an overclock stable for three
seconds routinely produces a wrong answer somewhere in the next three hours.

```bash
pcbench --soak 4h
```

Every core runs self-validating work — modular exponentiation checked against
Fermat's little theorem, compression round-trips, SHA-256, and walking memory
patterns — for as long as you ask. Errors are **counted**, not stopped at, and
timestamped, because time-to-first-error is what distinguishes a machine that
fails after four hours from one that fails after four minutes. Ctrl-C keeps
every finding so far. Wrong answers exit `7`.

## Storage: every device, not just the one you are standing on

Most machines have more than one drive, and the interesting one is rarely the
one the tool happens to be running from.

```bash
pcbench --list-devices          # what is mounted and what can be tested
pcbench --disk-all              # benchmark every writable local filesystem
pcbench --disk-path /mnt/data,/mnt/scratch
```

Mounts that would produce meaningless numbers or cause harm are refused with a
reason: tmpfs/ramfs (measures RAM and consumes it), network filesystems
(measures the network), read-only media, and anything without enough free
headroom. An explicitly named path always overrides the heuristics — you may
know something they do not.

Rotational media is detected on Linux, because a random-read figure that is
alarming on an SSD is completely normal on a hard disk, and without that the
diagnosis would be wrong.

## CI, monitoring, and fleets

**Thresholds.** Regression detection compares against this machine's own
history, which cannot answer acceptance ("every machine we deploy must reach
250") or fleet health ("alert when any node's disk drops below 500 MB/s" — no
history needed). Assertions read the way a person would say them:

```bash
pcbench --fail-under 250 \
        --assert 'disk.read_rate>=500' \
        --assert 'sustained.droop_pct<=15'
```

A bare name resolves to the **score** (baseline = 100); `name.rate` resolves to
the raw figure. Every verdict says which it used, so a gate never passes or
fails for a reason you cannot see. A metric that was not measured **fails** —
treating "not measured" as "threshold met" is how acceptance checks quietly
stop checking anything. Failures exit `6`.

**Exports.** JSON and CSV cover a human looking back at a run; these cover the
systems results need to arrive in:

```bash
pcbench --prometheus /var/lib/node_exporter/textfile_collector/pcbench.prom \
        --junit results/junit.xml \
        --sqlite results/history.db \
        --markdown results/summary.md
```

- **Prometheus** — machine identity goes in labels, not metric names, so a
  fleet aggregates cleanly and `min by (host)` works. Written atomically, since
  a collector will happily read a half-finished file.
- **JUnit XML** — benchmarks, detected regressions, and failed gates all become
  test cases, so performance problems appear in the same tab as failing unit
  tests instead of buried in job logs.
- **SQLite** — extracted columns for the fast queries, plus the whole payload,
  so a question nobody anticipated is still answerable years later.
- **Markdown** — for pasting into an issue or PR comment.

## Configuration files

A benchmark is only comparable to another benchmark run the same way, and
nothing in the output reveals that two people typed slightly different flags.
A config file makes the run definition a file that gets committed and copied to
every machine:

```bash
pcbench --init-config          # writes a commented pcbench.toml
```

```toml
[run]
seconds = 5
repeats = 5
profile = "server"

[output]
prometheus = "/var/lib/node_exporter/textfile_collector/pcbench.prom"

[gates]
fail_under = 250
assertions = ["disk.read_rate>=500", "sustained.droop_pct<=20"]
```

Precedence is command line > `PCBENCH_*` environment > config file > defaults —
increasing specificity. The file is found by walking up from the working
directory, so a repository-level config applies to every run inside it.

## Performance class — interpreting a single run

The composite is relative to a fixed baseline, which makes runs comparable to
each other and tells a first-time user nothing. So every run is also placed:

```
  Class      : workstation
               Serious multi-core throughput. Suited to large builds,
               simulation, video work, and running several VMs at once.
  Balance    : ~416 expected (range 187–1041) from a single-core score of 316
               across 10 physical / 10 logical cores
  Assessment : composite 443, balanced against this machine's own cores.
```

The balance check is anchored on the machine's **own measured single-core
performance**, not on its CPU name. An earlier design predicted the score from
architecture and core count; that cannot work, because "ARM64" covers both an
Apple M-series chip and a Raspberry Pi, whose per-core throughput differs by
more than 10x — and any such model flags every single-board computer as broken.
Anchoring on measured silicon makes the question answerable: given how fast one
core of *this* chip is, do the other subsystems keep up? A composite far below
the anchor means a specific subsystem is dragging, and the check holds equally
on a Pi and on a 96-core server.

Absolute floors are checked separately, where a figure is implausible rather
than merely slow — sequential reads under 80 MB/s, random reads under 200 IOPS,
memory bandwidth under 800 MB/s — each with what it usually means.

## Reference workloads — numbers that mean something elsewhere

Every other benchmark here is internally consistent and externally meaningless:
nobody has published a figure for `raytrace`, so a result can only be compared
against another run of this tool. These three are the industry's own, so a
number is comparable to what vendors and papers publish.

| Workload | Unit | What it is |
|---|---|---|
| **STREAM** (Copy/Scale/Add/**Triad**) | MB/s | McCalpin's memory-bandwidth standard. Triad is the quoted figure |
| **LINPACK / HPL** | GFLOPS | Dense `Ax=b` by LU with partial pivoting — the metric that ranks the TOP500 |
| **CoreMark-style** | iterations/s | The embedded integer kernel mix: list, matrix, state machine, CRC |

STREAM and CoreMark-style live in the native C engine, which is where they
belong — both reference implementations are C, and a Python version would
measure the interpreter. They cost nothing extra when the engine runs.

Three honesty rules, because the value of a standard *is* its comparability and
publishing an approximation under its name destroys exactly that:

- **STREAM validates its arrays** and reports the array size. A compiler that
  vectorises the loop away produces a spectacular meaningless number, and the
  4x-last-level-cache rule is checkable only if the size is stated.
- **LINPACK reports its residual** against HPL's own tolerance. A solve that
  did not actually solve the system is not a fast solve.
- **CoreMark-style is never called a CoreMark score.** Published CoreMark
  figures come from EEMBC's exact source under fixed reporting rules. This is
  the same kernel mix, useful for comparing cores to each other, and it says so
  every time it is printed.

```
  STREAM Triad        :     96,007.0 MB/s   <- the quoted figure
      67 MB per array; STREAM requires roughly 4x the last-level cache
  LINPACK (HPL)       :       145.48 GFLOPS   (N=8192, 512.0 MB)
      residual 0.002 (tolerance 16.0) — passed
  CoreMark-style      :     53,707.1 iterations/s
```

## System configuration — the 5–30% nobody records

Two machines with identical hardware routinely benchmark far apart, and the
reason is almost never the hardware. Every run now captures the settings that
silently move the numbers:

- **Speculative-execution mitigations** (Spectre, Meltdown, MDS, Retbleed).
  These cost 5% on ordinary work and 30%+ on syscall-heavy work. A box with
  `mitigations=off` looks like faster hardware and is not — and that is now
  stated rather than left to be discovered.
- **CPU governor** — `powersave` versus `performance` is frequently the whole
  explanation for a laptop scoring badly, and it is a settings fix.
- **Transparent hugepages, swappiness, SMT state, microcode revision, turbo.**

The SMT report distinguishes *not implemented by this CPU* from *implemented
and switched off*. Conflating them sends people hunting for a BIOS setting that
does not exist on their chip.

## Performance counters — why, not just how fast

Every other measurement says *how fast*. Counters say *what limited it*, and
the three common causes have completely different fingerprints:

| Symptom | Cause |
|---|---|
| Low score, **high IPC** | Cores executing efficiently, just not enough cycles — a clock or power limit |
| **Low IPC**, high cache miss rate | The working set does not fit. More cores will not help; memory is the wall |
| **Low IPC**, high branch miss rate | Unpredictable control flow — a code-shape problem |

```bash
pcbench --counters
```

Two tiers, because privilege differs enormously. **Resource counters** (page
faults, context switches, peak RSS) come from `getrusage`, cost nothing, need
no privileges, and run everywhere — and major page faults in particular
invalidate everything else in a report:

```
  Page faults (minor/major) : 93,658 / 8,243
      i 8,243 major page faults (89.8/s) — memory was fetched from backing
        store during the run, which dominates any CPU effect.
```

One counter is deliberately reported **without** a verdict attached.
Involuntary context switches look like a contention signal and are not one for
this workload: a full run on a completely idle machine reached ~8,600/s while a
genuinely busier machine measured ~2,500/s, because the count is dominated by
the tool's own worker processes and blocking I/O. It is printed as data, and
contention is judged from load average and per-test condition sampling instead.

**PMU counters** (cycles, instructions, cache and branch misses) are real
hardware registers needing kernel cooperation, so they come from `perf` on
Linux. On macOS the equivalent is a private framework requiring root and an
Apple entitlement; on Windows most counters need a driver. On those platforms
the tool says so plainly, with the specific fix where one exists, rather than
substituting something weaker and calling it the same thing.

## Statistical rigor — "is this 3% real?"

A median and a coefficient of variation describe one run. They cannot answer
the question people actually ask after changing something, and getting it wrong
is how teams ship changes that did nothing and chase regressions that never
happened.

```bash
pcbench --compare-runs before.json after.json
```

```
  METRIC                   BASELINE    CANDIDATE    CHANGE  VERDICT
  --------------------------------------------------------------------------
  cpu_int               1,974,169.1  2,402,776.8    +21.7% * SIGNIFICANT
  disk                      2,500.0      2,500.0     +0.0% = INCONCLUSIVE
  memory                    5,961.6      6,011.9     +0.8%   NO SIGNIFICANT DIFFERENCE

    memory: NO SIGNIFICANT DIFFERENCE — the 0.8% gap is within run-to-run
            noise (p=0.156). Resolving a difference this small would take
            about 11 repeats per side.
```

Three design choices worth knowing:

- **Mann-Whitney U, not a t-test.** Benchmark samples are not normal — they are
  bounded below by the hardware's best case with a long upper tail of
  interference. The reported median is a rank statistic anyway, so testing
  ranks keeps the headline figure and the test consistent.
- **"Inconclusive" is a distinct verdict.** Collapsing *not enough data* into
  *no difference* is how underpowered comparisons get mistaken for evidence of
  no effect. Below three repeats per side, no claim is made — and the tool says
  how many repeats would be needed.
- **Effect size alongside p.** With enough repeats a 0.3% difference becomes
  statistically significant while remaining completely irrelevant; that is
  reported as `SIGNIFICANT BUT NEGLIGIBLE`.

Exits `6` on a significant regression, so CI can gate on evidence rather than
on a raw percentage that may be noise. Confidence intervals are available on
every metric.

## Data science & ML

The old ML tier measured MLP training steps. That was the right thing to
measure in 2015. `--datascience` measures the four numbers that actually decide
whether a machine is usable now:

```bash
pcbench --datascience
```

```
  LLM model           : 8 layers, d_model=1024, 16 heads, 133M parameters
    Accelerated (mps, float16, 254 MB of weights)
      prefill :      6,625 tok/s   (1.77 TFLOPS, compute-bound)
      decode  :        351 tok/s   (94 GB/s achieved, bandwidth-bound)
    CPU (cpu, float32, 509 MB of weights)
      prefill :      2,068 tok/s   (0.55 TFLOPS, compute-bound)
      decode  :        158 tok/s   (84 GB/s achieved, bandwidth-bound)
  Accelerator memory  : 11.8 GB (mps, unified)
      largest model that fits: 5.1B at fp16, 10.2B at int8, 20.3B at int4
  Input pipeline      : 2,422 samples/s per worker, ~24,223 across 10 cores
  Dataframes (2,000,000 rows — filter/groupby/join/sort)
      polars  : 0.034 0.017 0.014 0.035   total 0.099s
      pandas  : 0.009 0.015 0.022 0.178   total 0.225s
      duckdb  : 0.006 0.006 0.006 0.211   total 0.229s
```

- **Prefill and decode are reported separately** because they are bound by
  different hardware. Generating one token requires reading *every* model
  weight, so decode is memory-bandwidth-bound — a GPU with huge FLOPS and
  modest bandwidth generates text slowly. Prefill is a big GEMM and is
  compute-bound. A single "inference" number hides the thing being asked about.
  (Sanity check on the machine above: decode achieved 94 GB/s against a STREAM
  Triad of 96 GB/s — decode really is sitting at the memory wall.)
- **Input-pipeline throughput** is measured because most training runs are
  limited by the CPU decoding and augmenting samples, not by the accelerator.
  A fast GPU behind a slow pipeline trains at the speed of the pipeline, and
  almost nobody measures it.
- **Everything runs on NumPy**, which is a far lower bar than PyTorch. Torch is
  used when present because it is the only way to reach a GPU. The transformer
  uses random weights rather than a downloaded checkpoint: throughput depends
  on the shape of the computation, not the values, and requiring a
  multi-gigabyte download would exclude exactly the constrained machines that
  most need measuring. Both backends run the **same** model so the figures are
  directly comparable.

## NUMA

On multi-socket servers and several chiplet designs, memory is not uniformly
fast, and run-to-run variance is frequently *not* noise — it is the allocator
landing on a different node.

```bash
pcbench --numa --numa-bandwidth
```

```
  Bandwidth MB/s (rows = CPU node, columns = memory node)
              node 0      node 1
  cpu 0       42,000      24,000
  cpu 1       23,500      41,800
  Remote penalty            : 43.3%
      i remote memory is 43% slower than local — pin latency-sensitive
        services with 'numactl --cpunodebind=N --membind=N'
```

Topology comes from sysfs and is free. The measured matrix needs `numactl`,
because the kernel allocates locally by default and the remote case would
otherwise never occur.

## Configurable storage I/O

The built-in disk test measures one fixed pattern — the right default and the
wrong tool for evaluating storage, since a device excellent at one pattern is
routinely poor at another.

```bash
pcbench --io                                          # the default four-job suite
pcbench --io-job 'oltp:bs=8k,pattern=randread,qd=32'  # your own
```

```
  JOB          PATTERN                            IOPS      MB/s   p50 us    p99 us
  database     randread bs=8K qd=16             49,548     387.1      170     1,803
  sequential   read bs=1M qd=1                   1,170   1,169.9      774     1,916
  log_write    write bs=64K qd=4                90,625   5,664.1       20       348
  vm_mixed     randrw bs=16K qd=8, 70/30 r/     42,286     660.7       71     1,491
```

Jobs are described the way `fio` describes them: block size, pattern, read/write
mix, queue depth, duration, cache bypass. **The queue-depth caveat is stated in
the output**: Python has no portable async submission, so depth is reached with
blocking calls on threads. That reaches real queue depths but carries more CPU
overhead than `fio`, so very high depths on very fast NVMe read low.

## Two-node network

Loopback characterises the OS stack and says nothing about the NIC, the cable,
or the path. That needs a second machine, so both halves are included:

```bash
pcbench --net-server              # on machine A (opens a listening port)
pcbench --net-client 10.0.0.5     # on machine B
```

Reports RTT percentiles, **jitter** (RFC 3550 definition), and throughput on one
stream versus several. The gap between them is the diagnosis: if four streams
greatly exceed one, the single-stream figure is limited by the TCP window and
latency rather than by the link's capacity. Where `iperf3` is installed it
remains the better tool and this says so; this exists for the very common case
where nothing can be installed on either end.

## Energy to solution

Watts is a rate and answers "how hot will this get?". It does not answer what
decides a datacenter bill or a battery's life:

```bash
pcbench --energy
```

Measures **joules for a fixed amount of work**, which is the only formulation
that compares machines of different speeds honestly. A chip drawing twice the
power that finishes in a third of the time uses less total energy — the
race-to-idle effect that governs battery life, and the thing a watts-only
comparison gets backwards.

Linux RAPL exposes a cumulative energy counter, so two readings give exact
joules with no sampling error. macOS exposes only instantaneous power, so the
figure is an integral of samples. A TDP estimate is a last resort and is
labelled as one — integrating estimates and calling the result "measured" is
exactly the false precision worth avoiding.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--seconds N` | `3.0` | Duration per test, per repeat |
| `--repeats M` | `3` | Repeats per test (median reported) |
| `--only a,b` | all | Subset of tests (see below) |
| `--profile NAME` | — | Preset selection: `quick`, `tiny`, `cpu`, `ai`, `dev`, `storage`, `laptop`, `server`, `apps`, `database`, `media`, `workstation`, `ci` |
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
| `--no-autoscale` | off | Do not shrink test sizes on small or CPU-limited machines |
| `--list-tests` | — | List every test and profile, then exit |

**Analysis depth**

| Flag | Default | Meaning |
|------|---------|---------|
| `--counters` | off | Hardware performance counters (IPC, cache/branch misses); needs `perf` on Linux |
| `--no-provenance` | off | Skip capture of governor, mitigations, hugepages, microcode |
| `--no-standards` | off | Skip STREAM, LINPACK, and the CoreMark-style suite |
| `--no-linpack` | off | Skip LINPACK only (needs NumPy and a few seconds) |
| `--numa` | off | Report NUMA topology |
| `--numa-bandwidth` | off | Also measure the local/remote bandwidth matrix (needs `numactl`) |
| `--energy` | off | Measure energy-to-solution in joules |

**Data science / ML**

| Flag | Default | Meaning |
|------|---------|---------|
| `--datascience` | off | LLM prefill/decode, input pipeline, batch scaling, dataframes |
| `--ds-prefill-tokens N` | `256` | Prompt length for the prefill measurement |
| `--ds-decode-tokens N` | `32` | Tokens generated for the decode measurement |
| `--no-dataframes` | off | Skip the dataframe benchmarks |

**Configurable I/O**

| Flag | Default | Meaning |
|------|---------|---------|
| `--io` | off | Run the four-job storage suite |
| `--io-job SPEC` | — | Custom job, e.g. `oltp:bs=8k,pattern=randread,qd=32`. Repeatable |

**Two-node network**

| Flag | Default | Meaning |
|------|---------|---------|
| `--net-server` | off | Run the receiving half (opens a listening port) |
| `--net-client HOST` | — | Measure throughput, latency, jitter against a peer |
| `--net-port N` | `51900` | Port for the two-node test |
| `--net-streams N` | `4` | Parallel streams for the throughput test |

**A/B comparison**

| Flag | Default | Meaning |
|------|---------|---------|
| `--compare-runs A.json B.json` | — | Statistically compare two saved runs and exit |
| `--alpha P` | `0.05` | Significance threshold |

**Storage devices**

| Flag | Default | Meaning |
|------|---------|---------|
| `--list-devices` | — | List mounted storage and whether each can be benchmarked |
| `--disk-all` | off | Benchmark every writable local filesystem |
| `--disk-path P[,P]` | — | Benchmark specific mount points |

**Stability & monitoring**

| Flag | Default | Meaning |
|------|---------|---------|
| `--soak D` | off | Burn-in: run validating work for `D` and count wrong answers |
| `--soak-workers N` | all cores | Load processes for the soak |
| `--monitor D` | off | Watch clocks, temperature, load, memory for `D` instead of benchmarking |
| `--monitor-interval N` | `1.0` | Seconds between monitor samples |
| `--monitor-power` | off | Also sample power draw while monitoring |
| `--monitor-trace P` | — | Write raw monitor samples to a CSV |

**Integration / CI**

| Flag | Default | Meaning |
|------|---------|---------|
| `--prometheus P` | — | Write Prometheus exposition text (node_exporter textfile collector) |
| `--junit P` | — | Write a JUnit XML report |
| `--sqlite P` | — | Append the run to a SQLite history database |
| `--markdown P` | — | Write a Markdown summary for an issue or PR |
| `--fail-under N` | — | Exit non-zero when the composite is below `N` |
| `--assert EXPR` | — | Threshold that must hold, e.g. `disk.read_rate>=500`; repeatable |

**Configuration**

| Flag | Default | Meaning |
|------|---------|---------|
| `--config P` | auto | Read settings from this TOML/JSON file |
| `--no-config` | off | Ignore `pcbench.toml` and `PCBENCH_*` variables |
| `--init-config [P]` | — | Write a commented starter config and exit |

Run `pcbench --list-tests` for the full catalogue. Synthetic tests: `cpu_int`,
`cpu_float`, `cpu_multi`, `compression`, `hashing`, `json`, `memory`,
`mem_scaling`, `cache_sweep`, `disk`, `nn_training`, `kmeans`, `knn`, `cores`,
`compile`, `latency`. Application tests: `sqlite`, `fsync`, `raytrace`,
`image`, `logparse`, `video`.

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
| Database (SQLite) | txn/s | Real storage-engine OLTP: index scans, aggregates, updates |
| Durable commits | commits/s | One flush that reaches the medium — the ceiling on DB writes |
| Ray tracing | frames/s | Branchy scalar float math, cache-resident working set |
| Image blur | MP/s | Strided 2-D access — separates a large L2 from a small one |
| Log parsing | MB/s | Linear byte scan through a backtracking regex |
| Video encode | fps | Software H.264 — sustained all-core vector load (needs ffmpeg) |
| STREAM Triad | MB/s | The memory-bandwidth standard, comparable to published figures |
| LINPACK (HPL) | GFLOPS | Dense LU with partial pivoting — the TOP500 metric |
| CoreMark-style | iter/s | Embedded integer kernel mix (not a certified CoreMark score) |
| LLM prefill | tokens/s | Prompt processing — compute-bound GEMM |
| LLM decode | tokens/s | Token generation — memory-bandwidth-bound weight streaming |
| Input pipeline | samples/s | Decode/crop/flip/normalise — the usual training bottleneck |
| Dataframes | queries/s | filter, group-by, join, sort across pandas/polars/duckdb |

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
| `5` | Results directory is not writable |
| `6` | A `--fail-under` / `--assert` threshold was not met |
| `7` | **Soak test produced wrong answers — hardware is unstable** |

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

420 tests, standard library only (they run with or without the optional tiers).

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
- `ffmpeg` on `PATH` is **optional** — only for the `video` encode benchmark
- `perf` (Linux) is **optional** — only for `--counters`; PMU counters are not
  reachable on macOS or Windows without privileged drivers
- `numactl` (Linux) is **optional** — only for the NUMA bandwidth matrix
- NumPy is **optional** — enables LINPACK and the CPU LLM backend
- PyTorch is **optional** — the only way to reach a GPU for `--datascience`
- TOML config files need Python 3.11+; JSON config works on every version
