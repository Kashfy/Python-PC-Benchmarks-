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

## Contents

**Start here**
[Why is this machine slow?](#why-is-this-machine-slow) ·
[Is it safe for my hardware?](#is-it-safe-for-my-hardware) ·
[Choosing what to run](#choosing-what-to-run) ·
[Hardware stats — no benchmark required](#hardware-stats--no-benchmark-required) ·
[Two ways to run it](#two-ways-to-run-it) ·
[Quick start](#quick-start) ·
[Example output](#example) ·
[What makes it reliable](#what-makes-it-reliable) ·
[Requirements](#requirements)

**Platform notes**
[Windows and x86 users — start here](#windows-and-x86-users--start-here) ·
[Containers, cloud, and CI](#containers-cloud-and-ci) ·
[Native engine](#native-engine)

**What it measures**

| Area | Sections |
|---|---|
| CPU | [What each test measures](#what-each-test-measures) · [Core scaling and hybrid CPUs](#core-scaling-and-hybrid-cpus) · [System configuration](#system-configuration--the-530-nobody-records) |
| Real software | [Application workloads](#application-workloads--will-this-machine-be-good-at-my-job) · [Reference workloads (STREAM, LINPACK, CoreMark)](#reference-workloads--numbers-that-mean-something-elsewhere) |
| Storage | [Read/write speed](#drive-readwrite-speed) · [Depth-aware storage testing](#depth-aware-storage-testing) · [Every device](#storage-every-device-not-just-the-one-you-are-standing-on) · [Configurable I/O jobs](#configurable-storage-io) · [Drive lifetime & wear](#drive-lifetime--wear) |
| Memory | [NUMA](#numa) |
| Accelerators | [GPU compute — NVIDIA, AMD, Intel, Apple](#gpu-compute--nvidia-amd-intel-apple) · [GPU and Neural Engine](#gpu-and-neural-engine) · [NPU across vendors](#npu-support-across-vendors) · [AI training & inference](#ai-training--inference-optional-framework-tier) |
| AI / data | [Data science & ML](#data-science--ml) · [ML without a framework](#machine-learning-workloads-with-no-framework-required) |
| Network | [Internet speed](#internet-speed) · [Network stack](#network-stack) · [Two-node network](#two-node-network) |
| Power | [Power & perf-per-watt](#power--perf-per-watt) · [Energy to solution](#energy-to-solution) |

**Interpreting results**
[How the scores are calculated](#how-the-scores-are-calculated) ·
[Performance class](#performance-class--interpreting-a-single-run) ·
[Bottleneck analysis](#bottleneck-analysis) ·
[Performance counters](#performance-counters--why-not-just-how-fast) ·
[Statistical rigor](#statistical-rigor--is-this-3-real) ·
[Regression detection](#regression-detection) ·
[Comparing machines](#comparing-machines)

**Diagnostics & stability**
[Hardware health](#hardware-health) ·
[Thermal / sustained load](#thermal--sustained-load-testing) ·
[Stability soak (burn-in)](#stability-soak-burn-in) ·
[Live monitoring](#live-monitoring--why-is-my-computer-slow-right-now)

**Automation**
[CI, monitoring, and fleets](#ci-monitoring-and-fleets) ·
[Configuration files](#configuration-files) ·
[Plugins](#plugins--add-your-own-benchmark) ·
[Exit codes](#exit-codes)

**Reference**
[All options](#options) ·
[Native engine extras](#native-engine-extras) ·
[Tests](#tests) ·
[Full documentation](#documentation)

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
.venv/bin/python benchmark.py             # macOS / Linux
.venv\Scripts\python.exe benchmark.py     # Windows
```

`install.py` prints the correct path for your platform when it finishes — a
virtual environment puts its interpreter in `bin/` on macOS and Linux and in
`Scripts\` on Windows, so the two are not interchangeable.

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

Not sure what to ask for? Run `pcbench` with no arguments — it opens the
guided menu, walks you through it with the arrow keys, and prints the command
it built. See [choosing what to run](#choosing-what-to-run).

## What makes it reliable

| Feature | Why it matters |
|---------|----------------|
| **Temperature in °C** | Real sensor readings on macOS (IOHID, no root), Linux (hwmon/thermal), Windows (WMI) — plus temperature tracked through sustained load. |
| **Machine-state guard** | Refuses to run on battery or under load, because those produce numbers that look like hardware differences but aren't. Override with `--force`. |
| **Warm-up pass** | Discards the cold-cache, low-clock first iterations before timing. |
| **Median of repeats** | Resists one-off outliers; reports a stability rating (`excellent`…`unstable`). |
| **Result validation** | Every workload verifies its own output. A wrong answer means unstable RAM/overclock/cooling — and exits with code 4. |
| **Cache-bypassed disk I/O** | Random reads go through `O_DIRECT`/`F_NOCACHE`/`FILE_FLAG_NO_BUFFERING`, so they measure the drive, not RAM. Checks the latency against physics in case the filesystem accepted the flag and ignored it, and reports whether it held. |
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
  CPU Integer (primes)      :      4,437,140 primes/s  (±0.8% run-to-run, excellent stability)
  CPU Float (math ops)      :     19,847,875 iters/s   (±0.3% run-to-run, excellent stability)
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

## How the scores are calculated

Every raw rate is normalised against a fixed baseline so that primes/s, MB/s,
IOPS and TFLOPS can be compared and combined at all:

```
subscore  = 100 × measured_rate / baseline_rate

category  = geometric mean of that category's subscores
composite = geometric mean of ALL subscores
```

**100 is the baseline machine** — roughly a mid-range 2020-era laptop. 200 is
twice as fast, 50 is half.

Note what the composite is *not*: it is not an average of the category scores.
It averages every subscore directly, so a six-metric category (`gpu`) carries
more weight than a two-metric one (`memory`). Averaging categories would
silently make those equal.

**Worked example**, from a real M1 Max run:

| Step | Calculation | Result |
|---|---|---|
| One subscore | `100 × 2,268,000 / 2,000,000` (cpu_int) | **113.4** |
| One category | `exp((ln 761.2 + ln 494.7) / 2)` (memory) | **613.6** |
| Composite | geometric mean of all 38 subscores | **226.2** |

**Why geometric.** On that run the arithmetic mean is **285.0** against a
geometric **226.2** — 26% higher, almost entirely from two outliers
(`disk_write` 1405.6, `gpu_matmul_fp32` 809.4). An arithmetic mean lets one
exceptional subsystem hide several weak ones. A geometric mean multiplies
ratios, so halving *any* subscore moves the composite by the same proportion
regardless of which — a machine has to be well-rounded to score highly. It also
makes the result independent of the units each metric happens to use.

**Absent hardware is omitted, never scored as zero.** No GPU means no `gpu_*`
subscores, and the composite is the geometric mean of what remains. A zero term
would collapse a geometric mean entirely, and a low placeholder would penalise a
machine for lacking hardware it was never asked to have. The consequence worth
remembering: **a composite is only comparable to one built from the same set of
subscores** — which is why the full list is printed above every composite.

`fsync` is measured and reported but deliberately unscored: macOS needs
`F_FULLFSYNC` where Linux's `fsync` suffices, and the two differ by two orders
of magnitude on identical hardware, so scoring it would measure the OS rather
than the drive.

Baselines are arbitrary but **frozen** — changing one invalidates every recorded
comparison. A test asserts that each makes the reference machine score exactly
100, and two more parse the documentation and fail if any baseline or category
membership drifts from the code.

Full reference — every baseline constant with its unit, category membership, and
the rules above in detail — is in
[docs/technical.md](docs/technical.md#scoring).

## Windows and x86 users — start here

Everything runs on Windows, but several sections need a one-time setup that
macOS and Linux get for free. This is the whole list.

### Running it

```powershell
python3 benchmark.py                      # zero install, works immediately
```

With the optional packages, note that a virtual environment puts its
interpreter in `Scripts\`, **not** `bin/`:

```powershell
python3 install.py
.venv\Scripts\python.exe benchmark.py     # Windows
```

`.venv/bin/python` is the macOS and Linux path and will fail with *"The term
'.venv/bin/python' is not recognized"*.

### The three things worth installing

| Install | Restores | Command |
|---|---|---|
| **A C compiler** | Native engine, **STREAM**, **CoreMark-style**, and the `compile` benchmark — four sections | `winget install BrechtSanders.WinLibs.POSIX.UCRT` |
| **psutil** | Resource counters (context switches, page faults, peak working set) | `pip install psutil` |
| **py-cpuinfo** | Full CPU feature list — AES-NI, SHA-NI, AVX2, AVX-512 | `pip install py-cpuinfo` |

A bare `clang` is **not** enough. LLVM's clang on Windows relies on a Visual
Studio installation for its headers and linker, and fails with *"unable to find
a Visual Studio installation"* on its own. The tool tries every compiler it can
find and keeps the first that actually produces a binary, preferring
self-contained MinGW `gcc`, then MSVC `cl` from a Developer Command Prompt,
then clang.

Without py-cpuinfo, Windows falls back to `IsProcessorFeaturePresent`, which
has codes for the vector extensions and **none for AES-NI or SHA-NI** — so
those show as undetected rather than absent, and the report says so.

### What Windows genuinely cannot do

These are platform limits, not missing work, and the tool reports each with its
reason rather than failing quietly:

| Section | Status on Windows | Why |
|---|---|---|
| **PMU counters** (`--counters`) | Unavailable | Hardware counters need a kernel driver; `perf` has no Windows equivalent |
| **GPU/NPU compute** | Inventory only | Compute benchmarking is Metal/Core ML for now; OpenCL via `pyopencl` works cross-platform |
| **Power measurement** | TDP estimate | No on-die power metering is exposed; macOS uses `powermetrics`, Linux uses RAPL |
| **NUMA bandwidth matrix** | Topology only | Needs `numactl`, which is Linux-only |
| **Drive lifetime detail** | Often status only | `Get-StorageReliabilityCounter` returns nulls for many consumer drives; try an **elevated** PowerShell |

### Things that look wrong on x86 and are not

- **"cannot separate SMT from a hybrid layout"** in core scaling. On a CPU with
  Hyper-Threading or SMT, the knee in the scaling curve has two possible causes
  that the curve alone cannot distinguish — hyperthreads sharing execution
  units, or physically smaller cores. The tool declines to guess rather than
  assert the wrong one. See [Core scaling](#core-scaling-and-hybrid-cpus).
- **`ml` as the weakest category** when it sits close to `cpu_int`. Those
  workloads are pure Python, so they measure the interpreter on your CPU rather
  than a separate subsystem. The verdict says so.
- **GPU VRAM reported as unknown.** `Win32_VideoController.AdapterRAM` is a
  32-bit field that pins every card at 4 GB or above to just under 4 GiB. The
  tool reads the 64-bit driver registry value instead and reports *unknown*
  rather than printing "4.0 GB" about a 16 GB card.

### Known-good Windows baseline

A Ryzen 7 7800X3D (8C/16T, 63 GB) with no optional packages installed produced
a composite of 192 with 17 subscores — storage, native engine, STREAM,
CoreMark-style, LINPACK and the compile benchmark all skipped for the reasons
above. Installing a compiler and the `compute` tier brings those back.

## Hardware stats — no benchmark required

Most of what this tool knows takes under two seconds to gather and needs no
load at all. Asking "how worn is my battery?" should not require a benchmark
that runs for minutes and heats the machine up.

```bash
pcbench --stats                    # everything, ~2 seconds
pcbench --stats battery            # just one section
pcbench --stats drives,thermal     # or a few
pcbench --list-stats               # what is available
```

```
==========================================================================
Battery
==========================================================================
  Charge                    : 100%  (charging / on AC)
  Health                    : 97.3% of design capacity
  Capacity                  : 5605 of 5760 mAh
  Charge cycles             : 78
  Temperature               : 30.2 °C
```

| Section | Reports |
|---|---|
| `cpu` | Model, cores, base clock, last-level cache, instruction-set extensions |
| `memory` | Total, available, swap usage |
| `storage` | Every mount: type, free space, whether it can be benchmarked |
| `drives` | **SSD endurance** — TB written, health %, power-on hours, projected life |
| `battery` | **Charge, health vs design capacity, cycles, temperature** |
| `gpu` | GPU and NPU inventory, OpenCL devices, PyTorch device |
| `thermal` | Every readable temperature sensor and fan |
| `power` | Idle draw, AC or battery |
| `os` | Kernel, Python build, governor, mitigations, hugepages, SMT, microcode |
| `environment` | Container, cgroup limits, cloud provider, CI system |
| `numa` | Node topology |
| `packages` | Which optional tiers are installed |

Battery and drive figures come with their thresholds explained, since a number
alone is not actionable:

```
      i capacity is 72% of design — most vendors treat 80% as the service
        threshold, so this battery is at or past the point where replacement
        is normally offered
```

`--json-stdout` works here too, so any section can feed a script or a
monitoring check without parsing terminal output.

## Why is this machine slow?

A benchmark says how fast a machine is. It does not say why it is slower than
it should be, and that is the question people actually arrive with:

```bash
pcbench --checkup            # about 15 seconds
pcbench --checkup --no-measure   # instant; reads state only, no load
```

It gathers evidence rather than scores, across every subsystem:

| Area | What it checks |
|------|----------------|
| **CPU** | Thermal throttling, boost disabled, power governor, SMT turned off, mitigation load, restricted affinity, load per core |
| **GPU** | Temperature against the point cards clock down, VRAM exhaustion, a discrete GPU that cannot be verified as the one in use |
| **Storage** | Volume headroom, SMART health and wear, sequential and random floors |
| **Memory** | Available RAM, swap pressure, total RAM, cgroup caps |
| **Software** | What is using the CPU *right now*, by name; uptime; virtualization and container limits |
| **History** | Whether this machine is measurably slower than its own past runs |

Then it measures briefly and holds load for eight seconds to catch
throttling. Findings are ranked by how much each is likely costing:

```
  The most likely cause is: the cpu is being thermally throttled right
  now. 4 other finding(s) may be contributing.

  6 finding(s): 2 critical, 3 warning, 1 info

  [ 1] CRITICAL The CPU is being thermally throttled right now
       evidence : thermal pressure reports 'throttled', CPU at 99 °C
       impact   : Clock speed is being cut to keep the chip within its limit,
                  so everything is slower — often by a third or more.
       fix      : Check that vents and fans are clear and the machine is on a
                  hard surface. On a laptop, a dust-blocked fan or dried
                  thermal paste is the usual cause.

  [ 2] CRITICAL / is 98% full (8.0 GB free)
       evidence : 8.0 GB of 500 GB free (1.6%)
       impact   : A nearly-full filesystem has to work harder to find
                  contiguous space for every write, and on an SSD it loses the
                  spare area the controller uses to spread wear.
       fix      : Free space until at least 15% is available.
```

Every finding carries **what was measured, why it matters, and what to do** —
because "your disk is slow" is not actionable and "your boot volume has 2 GB
free, so the filesystem can't find contiguous space to write into" is.

When nothing is wrong it says so, and still shows the numbers it based that
on:

```
  Nothing found that would explain a slowdown: no throttling, no
  contention, memory and disk headroom are fine, and the short
  measurement came out where it should.

  ── what this was based on ─────────────────────────────────────────────
  State      : 52 °C, nominal, on AC, load 0.19/core
  Memory     : 4.0 GB of 15 GB available (26%), swap 13% used
  Disk       : / — 22.9 GB free of 228 GB (10%)
  Uptime     : 1d 16h
  Busiest    : WindowServer 16%, fileproviderd 11%, Safari 8%
  Under load : throughput fell 0% over 8s
```

The last row is the strongest signal there is. "It used to be fast" is the
actual complaint most of the time, and a single snapshot cannot answer it —
but where this tool has run before, comparing against the same machine's own
record controls for the hardware entirely.

Two rules keep it honest. **An observation is not a verdict** — a spinning
disk is a specification, not a fault; long uptime is worth knowing, not a
problem. Findings say which they are, and severity reflects confidence as
much as impact. There is deliberately no CPU floor: a slow CPU with no
history is almost always just an old CPU. And **a check that could not run
says so**, because a clean report that quietly skipped half its checks is
worse than no report.

**All three platforms**, with the per-OS source for each check listed in the
[support matrix](docs/packages.md#platform-support-matrix). Processes come
from `ps` on macOS, `/proc` sampled twice on Linux — `ps` there reports CPU
averaged over a process's whole life, which is the wrong number for "what is
using it now" — and performance counters on Windows. GPU temperature and
VRAM need `nvidia-ml-py` and are NVIDIA-only; everything else is
standard-library.

Exit code is 1 when something is actively hurting performance and 0
otherwise, so a fleet check can gate on it.

## Choosing what to run

Twenty-two tests, thirteen profiles and twelve stats sections is a lot to read
before a first run, so there is a guided setup that asks instead. **It is what
you get by running the tool with no arguments at a terminal** — use the first
if you have not installed anything:

```bash
python3 benchmark.py            # straight from a checkout, no install needed
python3 -m pcbench              # the same thing, as a module
pcbench                         # after `pip install -e .`
```

`--menu` still asks for it explicitly, which is what you want when other flags
are present or when the input is piped. Going the other way, `--no-menu` runs
the default benchmark without asking, and so does `PCBENCH_NO_MENU=1`.

The menu is only the default when there is someone to answer it: with stdin or
stdout redirected — a pipe, a cron job, a container build, CI — a bare
`pcbench` runs the benchmark exactly as it always did, so nothing that scripts
this tool has to change.

Run it in the terminal directly. Piping or redirecting it (`| less`,
`> out.txt`) drops the arrow-key interface and falls back to typed answers,
because there is no TTY to put into raw mode — see
[when the arrow keys do nothing](#when-the-arrow-keys-do-nothing) below.

It is driven the way an OS installer is. Arrow keys move the highlighted row,
Enter selects, Esc steps back, `q` leaves. The only thing you ever type is a
real value — a duration, a custom repeat count — and nothing starts until you
accept the summary at the end.

```
====================================================================================
  pcbench 11.25 > Main menu
====================================================================================

  What would you like to do?

  > Run a benchmark              Measure how fast this machine is
    Read hardware stats          Battery, SSD endurance, temperatures, GPUs
    Watch or stress the machine  Live monitor, thermal throttling test, burn-in
    Check hardware health        RAM integrity, SMART status, drive wear
    Test the network             Internet speed, latency, or a two-node test
    Look at past runs            Rank the history, or compare two saved runs
    Shortcuts                    The flat list, if you already know what you want

  [up/down] move   [enter] select   [esc] back   [q] quit
```

Every screen takes the same keys. Lists — one choice or several — take all of
these:

| Key | What it does |
|-----|--------------|
| `↑` `↓`, or `k` `j` | Move the highlighted row; it wraps at both ends |
| `Home` `End` `PgUp` `PgDn` | Jump around a long list |
| `1`–`9` | Jump straight to that row |
| `Enter` | Select the row, or accept a screen of checkboxes |
| `Space` | Tick or untick a checkbox (on a one-choice screen it selects) |
| `a` | Tick everything, or clear everything |
| `Esc`, or `←` | Go back one screen |
| `q` | Leave without running anything |
| `Ctrl-C` | The same |

The few screens that ask for a value — a duration, a custom repeat count —
are a text field instead: type it, `Backspace` deletes, `Ctrl-U` clears, and
`Enter` accepts (empty keeps the default shown in grey). `Esc` still goes
back and `Ctrl-C` still leaves, but `q` is just the letter q there.

Picking *Run a benchmark* then asks what kind (quick pass, the full suite, a
profile, individual tests, AI/data science, storage I/O, reference standards),
how long each test should run, what else to measure, and what to write out.
Screens that pick several things at once are checkboxes — space ticks a row,
`a` ticks or clears everything, Enter accepts:

```
  Which tests?

  > [x] cpu_int      Integer math, single core (primes/s)
    [ ] cpu_float    Floating-point math, single core (iters/s)
    [x] cpu_multi    Integer math across every core (primes/s)
    [ ] cores        Per-core-count scaling curve and efficiency-core detection
    v 18 more

  [up/down] move   [space] toggle   [a] all/none   [enter] accept   [esc] back
```

Long lists scroll around the cursor, and screens that cannot apply to your
answers are skipped in both directions — so stepping back never lands on a
dead question.

The last screen is the whole point:

```
  This is what will run:

      pcbench --profile cpu --html

  Which means:
    - the cpu profile (7 tests)
    - 3 seconds per test, 3 repeats (the default)
    - self-contained html report
    - results saved to results/ as JSON and CSV

  > Run it now
    Go back and change something
    Quit without running anything
```

The screens are drawn on the terminal's alternate buffer, so once it exits the
only thing left in your scrollback is the command it built. That is deliberate:
the menu teaches the flags rather than replacing them, and the second time you
want that run, you type it.

### When the arrow keys do nothing

The menu needs a real terminal: stdin and stdout both a TTY, and a `TERM` that
is not `dumb`. Without that — a pipe, a redirect, a CI log, some IDE output
panes — the same screens fall back to numbered answers you type, which is the
right behaviour there rather than a failure. Check what it sees:

```bash
python3 -c "from pcbench import tui; print(tui.supported())"
```

The usual causes of an unexpected `False` are a redirect, `PCBENCH_NO_TUI`
being set, `TERM` unset or `dumb` (common under `cron` and some CI runners),
or a Windows console without VT support — Windows Terminal is fine.
[Troubleshooting](docs/troubleshooting.md#--menu-does-not-respond-to-the-arrow-keys)
has the fix for each.

The typed path takes the same answers by number, and is what makes the menu
scriptable:

```bash
printf '2\nbattery\n1\n1\n' | python3 benchmark.py --menu   # stats > battery > run
```

`PCBENCH_NO_TUI=1` forces it deliberately, if you prefer typing or your
terminal renders the full-screen version badly.

For everything else: `--list-tests`, `--list-stats`, `--list-devices`.

## Drive read/write speed

Sequential throughput and random IOPS are measured as part of any run, but if
that is the only question, ask it directly:

```bash
pcbench --drive-speed                 # every filesystem that can be measured
pcbench --drive-speed /Volumes/SSD    # or a specific one
pcbench --drive-speed --quick         # 1s x 2 instead of 3s x 3
```

```
  MOUNT                  KIND              WRITE         READ       RANDOM
  ------------------------------------------------------------------------
  /System/Volumes/Data   apfs          5835 MB/s    3811 MB/s   35952 IOPS
```

It writes a temporary file, reads it back with the page cache bypassed
(`F_NOCACHE` on macOS, `posix_fadvise` on Linux), deletes it, and exits — no
other test runs. If the cache could not be bypassed on some platform, the
table says so, because a read served from RAM is not a measurement of the
drive.

Sequential MB/s is what a large copy achieves. **Random IOPS is what makes a
machine feel fast**, since real workloads are dominated by small scattered
reads — a drive with excellent sequential numbers and poor IOPS still feels
slow. `--drive-speed` respects the same wear caps as every other disk test;
see [is it safe for my hardware?](#is-it-safe-for-my-hardware).

## Drive lifetime & wear

A benchmark says how fast storage is *today*. It says nothing about how long it
will keep working — and for an SSD that is the more consequential question,
because flash wears out by writing.

```
  APPLE SSD AP0256Z (NVMe)
    Status              : Verified
    Total written       : 25.77 TB
    Total read          : 54.67 TB
    Temperature         : 51 °C
    Health              : 98% (2% of rated life used)
    Power cycles        : 321
    Power on hours      : 855 (35.6 days)
    Spare blocks        : 100% (threshold 99%)
    Unsafe shutdowns    : 13
    Media errors        : 0
    Write rate          : 723.4 GB per power-on day
    Projected life left : ~41,895 more power-on hours
      which is ~28.7 years at 4h/day, ~14.3 at 8h/day, ~4.8 running continuously
```

Runs by default, costs milliseconds, needs no privileges, and is strictly
read-only — nothing writes to a drive, starts a self-test, or clears a log.
Disable with `--no-drive-life`.

**Where the data comes from.** Every platform records it; none make it easy the
same way. Linux uses `/sys/class/nvme`, then `nvme smart-log`, then `smartctl`
(SATA SSDs report the same facts under vendor-specific attribute names, which
are mapped). Windows uses `Get-StorageReliabilityCounter`.

macOS needed real work: **no command-line tool exposes this**. `system_profiler`
reports only a pass/fail status, and searching every IORegistry property for the
relevant keys returns nothing. The data sits behind an IOKit user client, so a
small helper (`smart_engine.c`, auto-compiled like the other native helpers)
reads NVMe log page 0x02 directly. One non-obvious detail: on Apple silicon the
user client is **not** published by the controller class — `IONVMeController`
and `AppleANS3CGv2Controller` both return `kIOReturnUnsupported`. It is
published by `IONVMeBlockStorageDevice`. Output was verified byte-for-byte
against a third-party SMART utility on the same machine.

**Two units that are easy to get wrong**, and are labelled explicitly here:

- **Write rate is per *power-on* day**, not per calendar day. A machine that
  sleeps has far fewer power-on days than calendar days, so "GB/day" would
  overstate the daily write load several-fold.
- **The projection is in power-on hours**, not calendar years. SMART counts
  power-on time and carries no manufacture date, so the drive's duty cycle is
  unknowable from the log. Dividing by 8760 would assume 24/7 operation and
  understate a laptop's calendar life by roughly ten times — so calendar
  figures are given against explicit daily-use assumptions instead.

**What gets flagged**, most urgent first: a controller critical-warning flag,
media/data-integrity errors, spare blocks below the drive's *own* threshold
(a failing drive, not a worn one), endurance past 80% and 95%, sustained
temperature above 70 °C, and unsafe shutdowns running at more than a quarter of
power cycles.

The figures also go into the CSV, so wear trends over months — which is exactly
what a per-run history file is good for — and can be gated on:

```bash
pcbench --assert 'drive_life.drives[0].health_pct>=20' \
        --assert 'drive_life.drives[0].media_errors==0'
```

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
this workload. Measured per test, `disk` produces 132,000/s and `latency`
83,000/s while every other test sits between 36 and 571/s — `--skip
disk,latency` takes a full run from 8,812/s to 985/s. Both do so by
construction: `disk` makes hundreds of thousands of blocking `pread()` calls to
measure IOPS, and `latency` *is* a context-switch benchmark. It is printed as data, and contention is judged from load average
and per-test condition sampling instead.

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

## Internet speed

Everything above stays on the machine or needs a target you name. "How fast is
my internet?" needs neither, which is exactly why it has to be asked for:

```bash
pcbench --internet
```

```
  Download    :    593.3 Mbit/s (70.7 MB/s, 20 MB in 0.3s)
  Upload      :     27.3 Mbit/s (3.2 MB/s, 5 MB in 1.5s)
  Ping        :     31.1 ms average (min 27.1, max 36.0, jitter 3.3, 0% loss)
                TCP handshake, also one round trip — ICMP got no reply
  DNS         :      3.4 ms median to resolve 3 name(s)
```

**Ping and jitter are usually the more useful half.** A 500 Mbit/s link with
30 ms of jitter makes calls stutter and shells feel laggy; a 50 Mbit/s link
with 2 ms does neither. Jitter here is the mean difference between
*consecutive* round trips (the RFC 3550 definition), not the spread — one
spike and a permanently unsteady connection have the same spread and very
different jitter.

The ping is ICMP where ICMP answers, exactly as the `ping` command measures
it. Many networks block or rate-limit ICMP, and on those a failed ping says
nothing about the connection — so it falls back to timing a TCP handshake,
which is also one round trip to the same host, and the line says which it
used. The fallback is often the *better* number, since ICMP is frequently
deprioritised while real traffic is not.

**This one sends traffic off the machine**, so it is never part of a benchmark
run — it is its own mode and has to be named every time. It downloads up to
`--internet-max-mb` (200 MB) and uploads a quarter of that, each also bounded
by `--internet-seconds` (5 s), so a slow or metered connection stops on bytes
and a fast one stops on time. The upload body is random bytes generated for
the purpose: no file, result, or machine identifier is ever sent.

The default endpoint is Cloudflare's public speed-test service, which needs no
account. Point it at your own with `--internet-server URL` — it needs to serve
`/__down?bytes=N` and accept a POST to `/__up`.

It is a single stream, not a speedtest.net client: no server selection, no
multi-connection saturation. On a fast link it therefore reads lower than
those tools, and says so. What it reports honestly is what one stream from
this machine actually achieves, which is the number that governs a download,
a `git clone`, or a container pull.

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

**Diagnosis**

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkup` | off | [Diagnose what is holding the machine back](#why-is-this-machine-slow), ranked by likely impact. Exits 1 on a critical finding |
| `--no-measure` | off | With `--checkup`, read state only — no benchmark, no load |

**Hardware stats & guided setup**

| Flag | Default | Meaning |
|------|---------|---------|
| `--stats [a,b]` | — | Report hardware facts and exit; no benchmark runs. Omit the list for every section |
| `--list-stats` | — | List the available `--stats` sections, then exit |
| `--menu` | on when bare + interactive | [Guided setup](#choosing-what-to-run) driven by the arrow keys; builds a command line, shows it, then runs it. Opens by default when `pcbench` is run with no arguments at a terminal |
| `--no-menu` | off | Run the default benchmark instead of opening that menu |

Sections for `--stats`: `cpu`, `memory`, `storage`, `drives`, `battery`,
`gpu`, `thermal`, `power`, `os`, `environment`, `numa`, `packages`.
`PCBENCH_NO_TUI=1` makes `--menu` use typed answers instead of the arrow
keys. `PCBENCH_NO_MENU=1` stops a bare `pcbench` opening it at all.

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
| `--drive-speed [P[,P]]` | — | Measure read/write and random IOPS per drive, then exit. Omit the list for every drive |

**Internet speed test** — sends traffic off this machine; never part of a run

| Flag | Default | Meaning |
|------|---------|---------|
| `--internet` | off | Measure download, upload, latency and jitter, then exit |
| `--internet-seconds N` | `5.0` | Time budget per direction |
| `--internet-max-mb MB` | `200` | Byte budget; upload uses a quarter of it |
| `--internet-server URL` | Cloudflare | Endpoint; must serve `/__down` and `/__up` |

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
python3 install.py --tier ai   # picks the build that reaches this hardware
```

The build matters more than the package name here. The plain `onnxruntime`
wheel on PyPI carries only `CPUExecutionProvider` on Windows and Linux, so
installing it on a machine with a discrete GPU gives a section that runs,
engages nothing, and reports the CPU. `install.py` inspects the hardware and
installs `onnxruntime-gpu[cuda,cudnn]` on NVIDIA, `onnxruntime-gpu` for ROCm,
`onnxruntime-directml` on Windows, or the plain wheel on macOS where Core ML
is already in it. For an Intel NPU or a Qualcomm Hexagon, add
`onnxruntime-openvino` or `onnxruntime-qnn` by hand.

On NVIDIA the tier also installs **TensorRT**, the provider ONNX Runtime tries
first and typically 30-50% faster than the plain CUDA one. Its version cannot
be guessed — ONNX Runtime pins a major and pip's newest runs ahead of it — so
the installer reads the required `libnvinfer` soname out of ONNX Runtime's own
provider library and constrains the install to match. On anything but NVIDIA
the package is not offered at all.

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

## GPU compute — NVIDIA, AMD, Intel, Apple

Two backends, because a modern GPU needs both and they measure different
hardware inside the same chip:

| Backend | Needs | Measures | Reaches |
|---|---|---|---|
| **OpenCL** (`pyopencl`) | `--tier gpu` | FMA throughput (GFLOPS), memory bandwidth | Shader cores, every vendor |
| **PyTorch** (CUDA / ROCm / XPU / MPS) | `pip install torch` | Dense matmul (TFLOPS) | **Tensor cores** — the AI-compute figure |

```
  NVIDIA GeForce RTX 5070 Ti [discrete]:         42,638 GFLOPS  (70 CUs, 635,971 MB/s)
  gfx1036 [integrated]      :          346.9 GFLOPS  (1 CUs, 47,198 MB/s)
      2 GPUs measured; scoring the discrete NVIDIA GeForce RTX 5070 Ti

  Matmul FP32 (NVIDIA GeForce RTX 5070 Ti):           41.2 TFLOPS
  Matmul FP16               :          168.5 TFLOPS  (4.09x fp32)
      fp16 is more than 2.5x fp32, which means dedicated matrix hardware
      (tensor cores or equivalent) is being used
```

The FP16 ratio is the interesting part. OpenCL only ever reaches the shader
cores, so a card whose value is its matrix hardware looks ordinary through it.
An RTX card shows roughly 4x fp32 at fp16; Apple silicon shows about 1.0x
because it has no separate matrix units in the GPU (its equivalent is the
Neural Engine, measured separately).

### Discrete is preferred over integrated

On a machine with both, the discrete GPU is scored — **even if the integrated
one measured faster**. Picking by throughput alone works almost always, and
"almost always" is the failure: a discrete card throttled by a power setting or
falling back to a slow driver path would silently lose to the iGPU and the
report would describe the wrong hardware. A reversal is now stated as the
finding it is:

```
      Note that the integrated Radeon Graphics measured higher (900 vs 400
      GFLOPS) — on a machine with a discrete card that usually means a driver
      or power setting is holding it back
```

Classification uses OpenCL's `host_unified_memory`, which is authoritative
where the driver reports it, with vendor name heuristics filling in where it
does not. Apple silicon is unified-memory but classed as discrete, since it is
the only GPU present and the fast one.

### What you need installed

```bash
python3 install.py --tier gpu     # pyopencl: shader GFLOPS + bandwidth, all vendors
pip install torch                 # matmul TFLOPS; CUDA, ROCm, Intel XPU, or MPS
```

On Windows the PyPI `torch` wheel is CPU-only, so the matmul is skipped exactly
as if torch were absent — name the CUDA index (`--index-url
https://download.pytorch.org/whl/cu130`) to get a build that reaches the GPU.
On Linux CUDA is bundled and on macOS MPS is built in.

Either backend works alone. With only PyTorch you still get the matmul figures
on a CUDA/ROCm/XPU device; with only OpenCL you get shader throughput on
everything. `pynvml` (in the same tier) adds NVIDIA temperature, power draw,
VRAM and utilisation.

**`pyopencl` is only the Python binding.** The driver it talks to is the
vendor's *ICD*, a system package, and pip cannot install one. A machine can
have a working GPU, a current driver and `pyopencl` installed and still
enumerate nothing — `PLATFORM_NOT_FOUND_KHR`, which reads like a hardware
fault and is not one. `install.py` checks afterwards and names the package for
your distribution:

```
OpenCL: pyopencl is installed but no driver is registered, so GPU
        shader throughput cannot be measured. That is a system
        package, not a pip one.

            sudo pacman -S opencl-nvidia
```

Windows ships the ICD with the display driver and macOS ships it with the OS,
so neither normally needs anything.

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

564 tests, standard library only (they run with or without the optional tiers).

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
