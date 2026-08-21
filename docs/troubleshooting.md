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
