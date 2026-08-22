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
