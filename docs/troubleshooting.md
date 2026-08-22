# Troubleshooting

Most problems here are environmental — power state, thermals, compilers,
permissions — rather than bugs.

## "System already busy" / "Running on BATTERY" — the run stops

**This is intentional** (exit code 3). Benchmarking on battery or under load
produces numbers that look like hardware differences but are not, and those
numbers then silently poison every later comparison.

Fix the condition — plug in, close background applications, let the machine
cool — or override:

```bash
python3 benchmark.py --force
```

The thresholds: battery power, load per core above 0.30, or an active thermal
throttle. Note that running the benchmark itself raises the load average, so
back-to-back runs may trip the guard; wait a minute between them.

## "VALIDATION FAILED" (exit code 4)

A workload computed the **wrong answer**. This is the most serious result the
tool can produce and is not a software bug in normal use. Common causes, in
rough order of likelihood:

1. **Unstable overclock / XMP-EXPO memory profile** — revert to stock and retest.
2. **Failing RAM** — run MemTest86 or `memtester`.
3. **Inadequate cooling** — a CPU past its thermal limit can compute
   incorrectly rather than merely slowly.
4. **Undervolting** applied to CPU or memory.
5. A miscompiled or mismatched Python/C runtime.

Reproduce with a single test to narrow it down:

```bash
python3 benchmark.py --only cpu_int --seconds 30 --repeats 5 --no-native
```

If it passes at stock settings and fails when overclocked, the overclock is
unstable.

## Disk read numbers look impossibly high

Check `cache_bypassed` in the JSON, or the console note. When it is `false`
the read was served from the OS page cache rather than the drive.

| Platform | Status |
|----------|--------|
| macOS | Bypassed via `F_NOCACHE` |
| Linux | Bypassed via `posix_fadvise(DONTNEED)` |
| Windows | **Not bypassed** — reads are an optimistic upper bound |

On Windows, or if bypass fails, trust the **write** figure (timed through
`fsync`) and treat reads as a floor. Increasing `--disk-mb` well beyond RAM
helps:

```bash
python3 benchmark.py --only disk --disk-mb 4096
```

## Disk test is skipped

**"needs ~N MB free"** — the guard requires free space ≥ 1.2× the file size.

```bash
python3 benchmark.py --disk-mb 64
python3 benchmark.py --output-dir /path/on/a/bigger/disk
```

## The native (C) engine section is missing

**"no C compiler found"** — the Python benchmarks still ran; only the
compiler-optimized comparison and the pointer-chase latency curve are missing.

- macOS: `xcode-select --install`
- Debian/Ubuntu: `sudo apt install build-essential`
- Fedora: `sudo dnf install gcc`
- Windows: MSVC Build Tools or MinGW-w64, or just use `--no-native`.

**"native build failed"** — the `detail` field carries compiler stderr. Build
by hand to see everything:

```bash
cc -O2 native_engine.c -o native_engine -lm -lpthread
```

On older POSIX toolchains a missing `-lpthread` is the usual cause.

**"native run failed"** — often a sandbox blocking the temp-file write. Run it
directly:

```bash
./native_engine --json --seconds 1 --repeats 1
```

Exit status 2 from the engine means *its* validation failed — see the
validation section above.

## "Neural Engine did NOT engage"

The speedup over CPU-only Core ML came in below 1.5x, meaning Core ML kept the
model on the CPU. The reported number is a CPU result, not an ANE one — which
is why the tool says so instead of publishing it as NPU performance.

Causes:

- **Not Apple silicon.** Intel Macs have no Neural Engine.
- **Model too small for this chip.** Dispatch overhead exceeded the work.
- **ANE busy** with another process.
- **macOS restrictions** in some VM or virtualized environments, where the ANE
  is not exposed at all.

The model geometry lives in `pcbench/coreml_model.py`
(`DEFAULT_CHANNELS`, `DEFAULT_SPATIAL`, `DEFAULT_LAYERS`); raising them
increases the work per inference.

## No accelerator section / "accelerator engine build failed"

GPU and NPU **benchmarks** require macOS plus the Command Line Tools. Install
them with:

```bash
xcode-select --install
```

Full Xcode is *not* required — Metal shaders are compiled at runtime precisely
to avoid it.

On Windows and Linux the inventory is shown and benchmarking is skipped by
design; see [technical.md](technical.md#what-is-and-isnt-covered).

Skip accelerators entirely with `--no-accel`, or individually with `--no-gpu` /
`--no-npu`.

## "no ML framework found" / AI section missing

Expected unless PyTorch or ONNX Runtime is installed — the AI training/inference
tier is opt-in precisely so the tool stays zero-dependency. To enable it:

```bash
pip install torch          # training + inference (CUDA/ROCm/MPS/CPU)
pip install onnxruntime    # inference-only fallback
python3 benchmark.py --ai
```

If a framework *is* installed but errors, the section shows the framework and
the error message; the rest of the run is unaffected.

## NPU section says "no accelerator beat the CPU by 1.5x"

The accelerator ran but was not faster than the CPU, so it is not reported as
engaged. This is a real result, not a bug — common reasons:

- **The CPU is genuinely fast at this workload.** Apple's CPU uses AMX for
  matrix multiply and often beats routing the same matmul through Core ML.
- **The provider fell back to CPU.** pcbench detects this and says so.
- **The model does not suit the NPU.** NPUs favour quantised/INT8 convolution
  work; a float32 matmul stack may stay on CPU or GPU.

For Intel/AMD NPUs specifically, install the vendor build — plain `onnxruntime`
does not include their providers:

```bash
pip install onnxruntime-openvino    # Intel
pip install onnxruntime-directml    # Windows, any DX12 NPU/GPU
pip install onnxruntime-qnn         # Qualcomm
```

## An NPU is detected but not benchmarked

Detection and benchmarking are separate. Detection works everywhere from PCI
IDs and drivers; benchmarking needs `onnxruntime` plus the matching execution
provider. Install one of the vendor packages above.

## ML workload failed validation

`nn_training`, `kmeans`, and `knn` each verify a known-correct outcome — loss
decreasing, clusters converging, points being their own nearest neighbour. A
failure here means the same thing as any other validation failure: suspect
unstable memory, an overclock, or cooling. See
[VALIDATION FAILED](#validation-failed-exit-code-4).

## "results are not writable" (exit code 5)

Running once with `sudo` (for real power readings) leaves **root-owned files**
in `results/`, so later unprivileged runs cannot save. The tool now detects
this before benchmarking rather than after, and prints the fix:

```bash
sudo chown -R $(whoami) results
```

Or write elsewhere with `--output-dir ~/bench-results`, or skip saving with
`--no-save`.

## A package failed to install

The installer installs one package at a time precisely so this is not fatal —
everything else still installs, and pcbench skips whatever is absent.

**`pyopencl`** is the usual culprit: it often has no prebuilt wheel and needs
system OpenCL headers.

```bash
# Debian / Ubuntu
sudo apt install opencl-headers ocl-icd-opencl-dev
# Fedora
sudo dnf install opencl-headers ocl-icd-devel
# macOS: OpenCL ships with the OS; no extra packages needed
```

**`numba`** pins specific numpy versions and can conflict; it is optional
within its tier, so `numpy` and `scipy` alone still give you BLAS and LAPACK.

Retry a single tier with `python3 install.py --tier compute`.

## Optional benchmarks do not appear after installing

The packages went into `./.venv`, but you are running the system Python. Use
the environment's interpreter:

```bash
.venv/bin/python benchmark.py          # macOS / Linux
.venv\Scripts\python.exe benchmark.py   # Windows
```

Confirm what the tool can see with `python3 install.py --list`.

## Temperature shows "unavailable"

| Platform | Requirement |
|----------|-------------|
| macOS | `sensors_engine` must compile — needs Command Line Tools (`xcode-select --install`) |
| Linux | Needs `/sys/class/hwmon` or `/sys/class/thermal`; absent in many containers and VMs |
| Windows | Needs WMI `MSAcpi_ThermalZoneTemperature`, which many OEMs never implement |

No temperature is ever invented — an unreadable sensor is reported as
unavailable rather than estimated.

## Power shows "(estimated)" instead of real watts

Real power metering is privileged:

- **macOS**: run the whole tool with `sudo` (`powermetrics` is root-only):
  `sudo python3 benchmark.py`. Without it you get a TDP-class estimate, clearly
  labelled.
- **Linux**: needs Intel/AMD RAPL at `/sys/class/powercap`; absent in many VMs
  and on ARM boards.
- **Windows**: no per-package metering is exposed, so only a TDP estimate is
  available.

The estimate is never presented as a measurement — the label always says which.

## Regression check says "first run on this machine"

There's no history for this hostname yet. Run the benchmark a few times (saving
each time) and subsequent runs will compare against the median of the prior
ones. Regression compares a machine only against **itself**; cross-machine
comparison is `--compare`.

## A "regression" appeared right after changing a flag

Metrics that depend on run settings are only compared against prior runs using
the same settings, so this should no longer happen. If you see disk or memory
metrics **skipped** in the regression section, that is why: no earlier run used
the current `--disk-mb` / `--mem-mb`. Run the same command twice and the
comparison becomes available.

Historically a `--quick` run (64 MB disk test) followed by a default run
(256 MB) reported a bogus -40% disk regression — bigger files exhaust an SSD's
SLC cache, which is a settings effect, not hardware failure.

## A regression was flagged but the machine is fine

Benchmarks are noisy. If a one-off background task slowed a run, re-run when
idle. Raise the sensitivity threshold if your environment is inherently
variable:

```bash
python3 benchmark.py --regression-threshold 20
```

## Network benchmark shows very high throughput

Loopback throughput (often >10 GB/s) is normal — it never touches a real NIC;
it measures the OS stack copying between two sockets in RAM. It's a diagnostic
of stack health, not your internet or LAN speed.

## GPU shows as "unknown" or is missing

- **Linux**: install `pciutils` for `lspci`, or the NVIDIA driver for
  `nvidia-smi`. Headless containers often expose nothing at all.
- **Windows**: `AdapterRAM` is a signed 32-bit field and misreports anything at
  or above 4 GB, so VRAM is omitted rather than shown wrong.
- **VMs**: virtual display adapters frequently report no useful model.

## Multi-core scaling is far below the core count

Expected. A 10-core chip rarely reaches 10×:

- **Hybrid cores** — efficiency cores are much slower than performance cores.
  An Apple M4 (4P + 6E) lands near 5×.
- **SMT / Hyper-Threading** — two threads sharing a core don't double
  throughput.
- **Process spawn cost** — counted in wall time, dominant on short runs.

Use `--seconds 5` or more. Compare against the native engine's threaded number:
if C scales much better than Python, the gap is interpreter overhead, not
hardware.

## Results marked "unstable" / high variance

The stability rating comes from the coefficient of variation. `unstable` (>10%)
means the number should not be trusted.

- Close background apps; plug in.
- Let the machine cool — thermal throttling depresses later repeats.
- Increase `--seconds 5 --repeats 5`.
- On a laptop, run `--sustained 5m` to see whether throttling is the cause.

## The cache sweep looks flat

Chips with very large caches and high memory bandwidth (Apple Silicon
especially) show a gentle curve rather than sharp cliffs. The sweep also starts
at 128 KB because smaller sizes measure interpreter overhead rather than the
cache.

For a precise hierarchy map, use the native engine's pointer-chase latency
table — it resolves L1 clearly, which the Python sweep cannot.

## `CPU model` shows an architecture string instead of a name

Some Linux ARM boards have no "model name" in `/proc/cpuinfo`. The tool falls
back to the device-tree model, then Hardware/Model, then the raw architecture.
Cosmetic only — benchmark numbers are unaffected.

## RAM or physical-core count shows `null`

The OS probe couldn't read it (unusual environment, container, locked-down
system). Installing `psutil` usually fixes it:

```bash
python3 -m pip install psutil
```

## `--compare` shows nothing

"No history found" means `results/benchmarks.csv` doesn't exist yet — run the
benchmark at least once without `--no-save`. If you used a custom
`--output-dir`, pass the same one to `--compare`.

If a `benchmarks.csv.v2.bak` appeared, your history was written by an older
version with a different schema; it was archived rather than corrupted. The old
data is intact in the `.bak` file.

## Unknown test name / invalid duration (exit 2)

Valid tests: `cpu_int`, `cpu_float`, `cpu_multi`, `compression`, `hashing`,
`json`, `memory`, `cache_sweep`, `disk`.
Valid durations: `90`, `30s`, `5m`, `1h`.

## Permission denied writing results

```bash
python3 benchmark.py --output-dir ~/bench-results
python3 benchmark.py --no-save
```

## Multiprocessing errors or repeated banners on Windows

Launch as a script (`python benchmark.py` or `pcbench`), not by importing and
calling `main()` without a `__main__` guard. `mp.freeze_support()` is already
called for frozen builds.

## Python too old

Errors mentioning `isqrt` or `fmean` mean Python < 3.8. Run with a newer
interpreter, e.g. `python3.12 benchmark.py`.

## Multi-core results look like a much slower machine (containers)

Inside a container or a cgroup, `os.cpu_count()` still reports the host's core
count while the scheduler hands out a fraction of one. Sixteen workers then
contend over half a core and the result looks like catastrophic hardware
failure.

pcbench detects this and reports it in the **Execution environment** section:

```
  Container        : Docker
  CPU quota        : 0.5 core(s) of 16 on the host
  Benchmarked with : 1 core(s)
```

Workloads are sized to the effective allowance automatically. If the section is
absent, nothing is limiting the run. On cgroup v1 systems the quota lives in
`/sys/fs/cgroup/cpu/cpu.cfs_quota_us`; on v2, `/sys/fs/cgroup/cpu.max`.

## Durable commits (fsync) look absurdly fast or absurdly slow

Both are informative, and neither is a bug:

- **Above 100,000 commits/s** is flagged. No storage device can persist that
  many 4 KiB writes per second; the drive or filesystem is acknowledging
  flushes without performing them, which means data loss on power failure.
  Common with virtual disks, `nobarrier` mounts, and some USB enclosures.
- **A few hundred per second on macOS** is correct. macOS is measured with
  `F_FULLFSYNC`, which actually reaches the medium, where a plain `fsync` would
  only reach the drive's volatile buffer and report a number two orders of
  magnitude higher. This is also why `fsync` is deliberately excluded from the
  composite score: it would measure the operating system's flush semantics
  rather than the hardware.

## The video benchmark is skipped

It needs `ffmpeg` with `libx264` on `PATH`, and it is excluded from the default
run. Install ffmpeg, then ask for it with `--only video` or `--profile media`.
The reported reason distinguishes "not found" from "the installed ffmpeg may
lack libx264".

## Thresholds fail with "was not measured in this run"

The metric the assertion names did not produce a value — usually because its
test was not selected, or was skipped for missing hardware or packages. This
is deliberate: treating an unmeasured metric as passing is how an acceptance
check quietly stops checking anything.

Check what the name resolves to. A bare name is the **score**; `name.rate` is
the raw figure, and they differ by orders of magnitude:

```bash
pcbench --assert 'sqlite>=250'        # score, baseline = 100
pcbench --assert 'sqlite.rate>=50000' # transactions per second
```

Every verdict prints which source it used, in brackets.

## The soak reported wrong answers (exit code 7)

The machine computed an incorrect result under sustained load. This is a
hardware finding, not a tool bug — the work units are validated against
independently known answers (Fermat's little theorem, compression round-trips,
SHA-256 digests, walking memory patterns), so a mismatch means the hardware
returned something wrong.

In rough order of likelihood: an unstable CPU or memory overclock (including
XMP/EXPO profiles), failing RAM, inadequate cooling, or marginal power
delivery. `time_to_first_error_s` narrows it down — failures within seconds
point at an overclock, while failures after hours point at heat or power.
Reset to stock clocks and re-run; if errors persist, test one memory module at
a time.

## Performance class says "unbalanced (subsystem drag)"

The composite is far below what this machine's own single-core score implies,
which means one subsystem is holding the rest back rather than the CPU being
slow. Look at the Bottleneck Analysis section for which category is lowest,
and at any absolute-floor warnings printed underneath the assessment.

The check is anchored on measured single-core performance rather than the CPU
model name, so it is equally valid on a Raspberry Pi and a 96-core server. If
too few single-threaded tests ran to anchor on, it says so instead of guessing.

## Config file settings are being ignored

In precedence order, a command-line flag always wins over `PCBENCH_*`
environment variables, which win over the config file. If a flag is on the
command line, the file cannot override it — that is by design.

Other causes: the file is not in the working directory or any parent (search
order is `pcbench.toml`, `.pcbench.toml`, `pcbench.json`, `.pcbench.json`);
`--no-config` is set; or the interpreter is older than 3.11, which cannot read
TOML — use a `.json` config there. An unknown setting is a hard error listing
every valid name, so a typo never silently does nothing.

## PMU counters say "unavailable"

`--counters` always collects resource counters (page faults, context switches,
peak RSS) — those need no privileges and work everywhere. Hardware PMU counters
are different, and the message states which case applies:

- **macOS / Windows.** Not reachable. macOS exposes the PMU only through a
  private framework needing root and an Apple entitlement; Windows needs a
  kernel driver. There is no workaround, and the tool does not substitute a
  weaker measurement and call it the same thing.
- **`perf` is not installed.** `apt install linux-tools-common
  linux-tools-$(uname -r)`, or `dnf install perf`.
- **`kernel.perf_event_paranoid` is above 2.** `sudo sysctl -w
  kernel.perf_event_paranoid=2` allows a process to measure itself.
- **Inside a container or VM.** Containers commonly drop `CAP_PERFMON` and VMs
  frequently expose no PMU at all. Neither shows up in `perf_event_paranoid`,
  which is why availability is confirmed with a trial run rather than assumed.

## STREAM numbers look impossibly high

Almost certainly the array is not large enough. STREAM requires each of its
three arrays to be roughly 4x the last-level cache; below that it measures
cache bandwidth. The array size is printed with the result:

```
  67 MB per array; STREAM requires roughly 4x the last-level cache, so this is
  valid for caches up to about 17 MB.
```

Server parts with 64 MB+ of L3 need `--stream-mb` raised well above the
default. If the result also says `VALIDATION FAILED`, the compiler optimised
the kernels away or the machine computed them incorrectly — the arrays are
checked against the arithmetic they should have produced precisely to catch it.

## LINPACK GFLOPS is lower than the vendor's figure

Expected, and stated with the result. N is capped so the run finishes quickly
and never approaches swap; HPL efficiency rises with N because the O(N³)
arithmetic increasingly dominates O(N²) memory traffic, which is why TOP500
submissions use matrices filling most of RAM. This figure is a consistent,
validated point of comparison, not a peak-performance claim.

If LINPACK is skipped entirely, NumPy is missing. A hand-written LU would
report a fraction of the machine's real capability and mean nothing, so it is
skipped rather than approximated.

## "CoreMark-style" — why not just CoreMark?

Because it is not CoreMark. It runs the same four kernels, which makes it
useful for comparing cores under an identical compiler-resistant integer
workload. Published CoreMark scores come from EEMBC's exact source under strict
reporting rules, and a number produced here must not be quoted as one. Use it
against other runs of this tool, not against published CoreMark figures.

## "NO SIGNIFICANT DIFFERENCE" but the number clearly changed

That is the finding. With a handful of repeats and normal run-to-run variance,
differences under roughly 5% are usually indistinguishable from noise, and the
verdict says how many repeats would resolve one that small:

```
  memory: NO SIGNIFICANT DIFFERENCE — the 0.8% gap is within run-to-run noise
          (p=0.156). Resolving a difference this small would take about 11
          repeats per side.
```

Re-run both sides with `--repeats 11` (or whatever it suggests). If the verdict
is `INCONCLUSIVE` instead, there were fewer than three repeats per side and no
statistical claim is made at all — that is deliberately distinct from "no
difference", because treating too-little-data as evidence of no effect is the
most common way performance comparisons mislead.

## A/B comparison warns about comparability

`--compare-runs` warns rather than refuses when the two runs differ in machine,
CPU, `--seconds`, `--repeats`, or resource confinement. Comparing across
machines is sometimes exactly the intent; the warning exists so the difference
is not read as a change comparison when it is a hardware comparison.

## LLM decode tokens/s seems low relative to prefill

That is the physics, and separating the two is why they are reported
separately. Generating one token requires reading *every* model weight, so
decode is bound by memory bandwidth, while prefill is a large matrix multiply
and is bound by compute. A ratio of 10-20x between them is normal.

Cross-check the decode figure against the STREAM Triad result: decode's
"achieved GB/s" should sit close to it. If it does, decode is at the memory
wall and no amount of extra compute will help — only faster memory or a smaller
(quantised) model will.

## The data-science tier is skipped or CPU-only

`--datascience` runs on NumPy alone, so a missing accelerated row means PyTorch
is not installed (`pip install torch`, or `python3 install.py`). A missing
dataframe section means none of pandas, polars, or duckdb is installed — the
`data` tier in `install.py` covers all three.

The transformer uses random weights and is never downloaded; if the model looks
small, it was sized to a fraction of available memory deliberately, so the
measurement is stable rather than impressive. Both backends run the same model
so their tokens/s are directly comparable.

## NUMA bandwidth matrix is skipped

Either the machine has one node (nothing remote to measure — the common case
and not an error), or `numactl` is not installed. It is required because the
kernel allocates memory locally by default, so without explicit placement the
remote cases would silently be measured as local and the matrix would come out
uniform and wrong.

## I/O job throughput looks too high

Check `cache_bypassed` in the JSON. If the OS cache could not be bypassed, the
figures include page-cache hits and a `caution` line says so. A sequential read
in the multi-GB/s range from a device that cannot sustain it is the signature.

For very high queue depths on very fast NVMe, expect figures below what `fio`
reports: Python has no portable asynchronous submission, so depth is reached
with blocking calls on threads, which costs more CPU per request. The note is
printed with every I/O run.

## Two-node network test cannot connect

The peer must be running `pcbench --net-server` and the port (default 51900)
must be open through any firewall between the machines. `--net-server` binds
all interfaces and says so — it is a real change in the machine's exposure and
should be stopped when finished.

Where `iperf3` is available on both ends it remains the better tool. This exists
for the very common case where nothing can be installed on either machine.

## Energy is reported as an estimate

No power meter was readable, so a TDP figure for the chip class was held over
the elapsed time. That is labelled rather than presented as a measurement:

- **Linux** reads RAPL (`/sys/class/powercap`) and gives exact joules from a
  cumulative counter — no sampling error at all.
- **macOS** needs `sudo` for `powermetrics`; without it only the TDP estimate
  is available.

## Package power looks far too low (and perf-per-watt too high)

Fixed in v11.1. Before it, the "under load" reading was generated with Python
threads, which the GIL serialises onto a single core — so on a 10-core machine
the all-core figure was really single-core power. An M1 Max reported 6.9 W
where the true all-core draw is several times that, and every perf-per-watt
number derived from it was inflated by roughly the core count.

The load now runs in processes. If you have saved runs from v11.0 or earlier
**with a real (non-estimated) power reading**, their `power_watts`,
`score_per_watt`, and energy figures are wrong and should be re-measured. Runs
where power was shown as `(estimated)` are unaffected — the TDP estimate never
depended on the load.

## STREAM Triad seems high for my chip

Check the cache line printed under the result. From v11.1 it states the ratio
and gives a verdict:

```
  268 MB per array against a 17 MB last-level cache (hw.perflevel0.l2cachesize)
  — 16.0x, satisfying STREAM's 4x rule, so this measures memory rather than
  cache bandwidth.
```

Arrays are now sized from the detected last-level cache with a generous floor,
rather than a fixed 64 MB. The floor matters on Apple silicon specifically:
macOS reports only L2, so an M1 Max looks like a 24 MB cache while also having
a 48 MB system-level cache the OS never mentions. Under the old fixed default
that machine ran STREAM at 1.4x its real cache and reported a Triad figure that
was partly cache bandwidth.

If the note says **BELOW**, it tells you what to pass to `--stream-mb`. Very
large server caches (EPYC parts with 256-384 MB of L3) are handled
automatically when sysfs reports them.

## Involuntary context switches look enormous

They usually are, and it is not a problem. The counter is reported as data with
no verdict attached, because for this workload it does not measure contention.
Measured per test:

| test | involuntary switches/s |
|---|---:|
| `disk` | 132,152 |
| `latency` | 83,027 |
| everything else | 36 – 571 |

Two tests produce the overwhelming majority, and both do so by construction: `disk`
issues hundreds of thousands of blocking `pread()` calls to measure random-read
IOPS, and each one that blocks is a preemption; `latency` *is* a context-switch
benchmark, so making them is its measurement. A whole-run total is therefore
close to a restatement of how many I/O operations the disk test completed,
which is already reported as IOPS.

Skipping them is the falsification test, and it is quick to repeat:
`--skip disk,latency` took a full run from 8,812/s to 985/s here, an 89% drop.

Versions up to v11.1 asserted "other processes were competing for CPU" from
this number. That conclusion was not supported and has been removed.

To judge contention, use the load average in the System Information block
against the core count, and the per-test interference notes — both are grounded
in the right measurement.

## "ml" is named as the bottleneck

Check whether the note says these are pure-Python workloads. `nn_training`,
`kmeans`, and `knn` run in the interpreter, so they re-measure the CPU rather
than an independent subsystem — measured on real hardware they track `cpu_int`
to within 2-4% (111.7 vs 113.5 on an M1 Max; 231.5 vs 223.4 on an M4).

From v11.3 the tool detects that and says so, rather than implying a separate
weakness:

```
  Verdict: ml scores lowest, but these are pure-Python workloads that track
  single-core CPU throughput — the finding is that this machine's cores are
  modest, not that a separate subsystem is weak
```

The action is not a hardware change. Install NumPy or PyTorch (`python3
install.py`, tiers `compute` and `data`) and the same work bypasses the
interpreter entirely — the `numeric` and `datascience` categories then show
what the silicon can actually do.

If `ml` scores *far* below `cpu_int` rather than tracking it, that is a real
finding and is reported plainly.

## "system load rose ... something else started competing for the CPU"

Fixed in v11.3 for tests that cause it themselves. `cpu_multi`, `cores`,
`mem_scaling`, `memory` and `disk` saturate every core by design, so the load
average rises because of them — measured at +0.19 per core for a 3-second
`cpu_multi` and +0.68 for a 10-second one, against a 0.25 threshold and 0.000
drift on an idle machine. Anyone running `--seconds 10` or longer for stabler
results got the warning on every multi-process test.

The load signal is now suppressed for those tests (the delta is still recorded
as data) and kept for single-threaded ones, where a rise really is external.
Temperature checks are unaffected — the tool heating the chip is precisely what
that check exists to catch.

If you still see this on a single-threaded test, it is real: something started
during the run.

## Core scaling says "hybrid design" on a CPU that is not hybrid

Fixed in v11.4. The analysis only saw the logical core count, so any knee in
the scaling curve was reported as "a hybrid design with slower efficiency
cores". On the many x86 CPUs with SMT / Hyper-Threading the knee is not that at
all — it is hyperthreads sharing execution units once every physical core is
busy, which typically contribute about 30% of a full core.

The two look identical in the curve and call for different actions (pin to
P-cores versus consider disabling SMT), so the physical core count is now used
to separate them:

- knee at the physical core count, with SMT present -> reported as SMT
- knee with no SMT on the machine -> reported as a hybrid layout
- knee somewhere else with SMT present -> no cause claimed, curve shown

Apple silicon is genuinely hybrid and has no SMT, so it is still reported as
hybrid.

## Core scaling reports far fewer linear workers than my CPU has

Usually the machine was busy. The curve needs the CPU mostly free: background
load steals workers unevenly and the marginal gain per worker becomes noise —
on a loaded machine successive workers can even show a negative marginal
contribution.

The pre-run state check normally refuses to run in that situation; if you
passed `--force` you have overridden it. Re-run on a quiet machine, and raise
`--seconds` (the default 0.6s per point for this test is deliberately short).

## Drive lifetime says "unavailable"

The reason is always stated. Common cases:

- **macOS** — the helper needs a compiler: `xcode-select --install`. It needs
  no privileges once built.
- **Linux** — install `nvme-cli` or `smartmontools`. Reading the SMART log
  usually needs root, so try `sudo`.
- **Windows** — many consumer drives and almost all USB enclosures do not
  expose reliability counters at all.
- **Any platform, external drive** — USB-SATA bridges rarely pass SMART
  commands through.

## Write rate looks impossibly high

It is per **power-on day**, not per calendar day. A laptop that sleeps most of
the time accumulates few power-on hours, so dividing lifetime writes by them
gives a large number. 25 TB over 855 power-on hours is 723 GB per power-on day
but only a fraction of that per calendar day.

The same distinction applies to the life projection, which is reported in
power-on hours with calendar equivalents at 4, 8 and 24 hours per day. SMART
records no manufacture date, so the tool cannot know your duty cycle — pick the
row that matches how you actually use the machine.

## Health percentage looks wrong for the drive's age

`percentage_used` is the *controller's own* wear estimate against its rated
endurance, not a measurement of remaining flash. Vendors compute it
differently, and it commonly sits at 0-1% for a long time and then moves in
steps. It can also exceed 100%, at which point the drive is past its rated life
but usually still working — the tool clamps health at 0% rather than showing a
negative number.

The figure to watch alongside it is **available spare**. Wear is a projection;
spare blocks falling below the drive's own threshold means the drive is
consuming its reserve now.

## A subsystem is flagged but the hardware is fine

Check the marker. From v11.8 these findings carry a severity:

- `i` — **expected for this machine.** An SD card on a single-board computer,
  or a hard disk, genuinely performs at these figures. Nothing is wrong.
- `!` — **worth investigating.** The number is out of line with what the rest
  of this machine's measurements imply.

Earlier versions asserted a fault from the absolute number alone, which flagged
a healthy Raspberry Pi 4 on an SD card twice (SD cards sustain 20-45 MB/s by
design) and told the owner of a working 5400rpm hard disk that "on an SSD it
indicates a real fault".

The distinction that does the work is the *pattern*, not the threshold: a
failing SSD loses random-access performance while keeping much of its
sequential throughput, so **low sequential and low random together** indicate a
slow medium, while **healthy sequential with collapsed random** is the
signature of a real fault.

## A regression is reported that vanishes on the next run

Check the note beside the finding. From v11.8 each change is judged against
that metric's own historical variability rather than a single percentage:

```
    ▼ Disk write   -32.7%  (4,605 → 3,100)   (within this metric's normal ±51% spread)
  ✓ Changes seen are within each metric's normal run-to-run variation.
```

Sequential disk throughput routinely swings tens of percent between runs on the
same machine, while integer CPU work varies under 1%. A single ±10% threshold
treats them as equally repeatable, so the noisy metrics produced most of the
findings.

With fewer than three prior runs there is nothing to estimate variability from,
and the finding says so — `(provisional — only 1 prior run)`. That is not the
same as "within normal variation", and the summary line distinguishes them. A
few more runs settle it.

A slowdown that is still flagged after several runs is real. In rough order of
likelihood: background load during the run, thermal throttling, a changed
setting, then hardware health.

## Reporting an issue

Capture a full machine-readable dump:

```bash
python3 benchmark.py --json-stdout --no-save --force > run.json
```

It contains the inventory, machine state, every raw sample, and any per-test
error messages. Also useful:

```bash
python3 -m unittest discover -s tests -v
```
