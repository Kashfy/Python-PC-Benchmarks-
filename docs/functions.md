# Function Reference

Public API of every module. See [technical.md](technical.md) for the
methodology behind these and [architecture.md](architecture.md) for how they
fit together.

---

## `pcbench.core` — timing, statistics, validation

Pure functions, no I/O — the most heavily unit-tested module.

#### `clock() -> float`
Monotonic high-resolution timestamp (`time.perf_counter`).

#### `warmup(chunk_func, seconds) -> int`
Runs `chunk_func` untimed for ~15% of `seconds` (clamped 0.05–1.0 s) to reach a
steady state before measurement. Always runs at least once. Returns the
iteration count.

#### `timed_loop(chunk_func, seconds) -> (elapsed, iterations)`
Calls `chunk_func` repeatedly until `seconds` elapse.

#### `summarize(samples) -> dict`
Reduces per-repeat rates to `{median, mean, stdev, cv, min, max, samples}`.
`cv` (stdev/mean) is a normalized stability indicator.

#### `stability_note(cv) -> str`
Maps a coefficient of variation to `excellent` / `good` / `fair` / `unstable`.

#### `ValidationError`
Raised when a workload computes an incorrect result — a hardware finding, not a
bug.

#### `check_exact(name, got, expected)` / `check_close(name, got, expected, rel_tol=1e-9)`
Assert a workload's output. `check_close` uses a relative tolerance because
platform math libraries differ in the last bits.

---

## `pcbench.system` — inventory and machine state

#### `arch_family(machine) -> str`
Normalizes `platform.machine()` to a canonical ISA family.

#### `cpu_model() -> str`
Human CPU name. `sysctl machdep.cpu.brand_string` (macOS); `/proc/cpuinfo`
model name → device-tree model → Hardware/Model (Linux); registry
`ProcessorNameString` → `PROCESSOR_IDENTIFIER` (Windows).

#### `total_ram_bytes() -> int`
`psutil` → `sysctl hw.memsize` / `/proc/meminfo` / `GlobalMemoryStatusEx`.
Returns 0 if undetectable.

#### `physical_cores() -> int | None`
`psutil` → `sysctl hw.physicalcpu` (macOS), distinct `(physical id, core id)`
pairs (Linux), PowerShell `Win32_Processor.NumberOfCores` with a `wmic`
fallback (Windows).

#### `cpu_frequency_mhz() -> float | None`
Base/max clock where exposed. Returns `None` on Apple Silicon, which publishes
no nominal clock.

#### `gil_status() -> dict`
`{free_threaded_build, gil_enabled}`. Python 3.13+ can be built free-threaded,
which changes multi-threaded scaling dramatically.

#### `inventory() -> dict`
The full static fact set: host, OS, architecture family/bits/endianness, CPU
model, core counts, base clock, RAM, Python version and GIL status.

#### `on_ac_power() -> bool | None`
True on AC, False on battery, None if unknown or no battery.

#### `load_average() -> tuple | None` · `thermal_pressure() -> str | None`
System load (not on Windows) and a best-effort throttling/temperature reading.

#### `machine_state() -> dict`
Volatile conditions: `{on_ac_power, load_average, load_per_core, thermal}`.

#### `state_warnings(state) -> list[str]`
Conditions that will distort results — battery power, load per core > 0.30,
active thermal throttling. A non-empty list stops the run unless `--force`.

---

## `pcbench.workloads` — the benchmarks

Every `bench_*` validates, warms up, repeats, and returns a dict with at least
`unit` and `rate`.

#### Chunks and constants
`cpu_integer_chunk() -> int` (returns the prime count, `EXPECTED_PRIME_COUNT ==
89`), `cpu_float_chunk() -> float` (returns the sum, `EXPECTED_FLOAT_SUM`),
`_is_prime(n)`, `_corpus(size)` (deterministic, compressible test data).

#### `bench_cpu_integer(seconds, repeats)` → primes/s
#### `bench_cpu_float(seconds, repeats)` → iters/s
#### `bench_cpu_multicore(seconds, workers=None)` → primes/s
One `spawn` process per logical core; returns aggregate rate, worker count,
wall time, and per-worker counts. `_multicore_worker` is module-level so it
stays picklable.

#### `bench_compression(seconds, repeats)` → MB/s
zlib level 6 round-trip, CRC-verified.

#### `bench_hashing(seconds, repeats)` → MB/s
SHA-256; reaches hardware crypto instructions where present.

#### `bench_json(seconds, repeats)` → MB/s
JSON parse throughput.

#### `bench_memory(seconds, repeats, buf_mb=64)` → MB/s
Sustained copy bandwidth; verifies the copy afterwards.

#### `bench_cache_sweep(total_seconds, ram_bytes)` → dict
Bandwidth across working sets from 128 KB to 128 MB, revealing cache tiers.
Returns `points`, `peak_mb_per_s`, `dram_mb_per_s`, `cache_to_dram_ratio`.

#### `bench_disk(seconds, repeats, file_mb, out_dir)` → dict
Sequential write (through `fsync`), sequential read, and 4 KiB random-read
IOPS. Returns `write_rate`, `read_rate`, `random_read_iops`, `cache_bypassed`
(the random phase), `sequential_cache_bypassed`, and `direct_method`. Skips
with a reason when free space is insufficient.

#### `_set_nocache(fd) -> bool` · `_drop_cache(fd) -> bool`
Page-cache defeat for the *sequential* phase. `_set_nocache` (macOS
`F_NOCACHE`) **must** run before the file is written — it prevents caching but
does not evict. `_drop_cache` (Linux `posix_fadvise`) evicts after `fsync`.

#### `DirectReader(path, shared_fd, block=4096)`
One thread's page-cache-bypassing reader for the *random* phase, where a single
eviction is useless because the file refills on first pass. `O_DIRECT` +
`preadv` on Linux, `F_NOCACHE` on macOS, `CreateFileW(FILE_FLAG_NO_BUFFERING)`
through `ctypes` on Windows, buffered where none is available. `.method`
records which, `.read(offset)` returns bytes read, and it is a context manager.
Owned by exactly one thread: the Windows and fallback paths carry a file
position.

#### `memory_filesystem(path) -> str | None`
Names the RAM-backed filesystem `path` sits on (`tmpfs`, `ramfs`, `devtmpfs`),
or None. Linux only — it is the only platform whose default temp directory is
commonly RAM, which is how `--no-save` used to report memory bandwidth as
storage throughput.

#### `direct_read_note(method, latency, ramfs=None) -> (bool, str)`
Decides whether the random-read figures really came from the device. Three ways
they might not have: `ramfs` is set and there is no device involved at all; the
`method` is `buffered` because the platform had no mechanism; or a median
latency under `IMPLAUSIBLE_DEVICE_US` (3 µs) shows the filesystem accepted the
flag and served from cache anyway, which btrfs-with-compression and network
filesystems both do.

#### `_random_read_iops(reader, size, budget) -> float`
4 KiB reads at random offsets through a `DirectReader`, batched so `clock()`
overhead stays off the measured path.

---

## `pcbench.sustained` — thermal behavior

#### `run_sustained(duration, window=5.0, workers=1) -> dict`
Runs continuous load, sampling per window. Returns `samples`, `peak_rate`,
`sustained_rate` (mean of the final 25%), `droop_percent`, `verdict`, and
post-run machine state.

#### `sparkline(values) -> str`
Compact unicode bar chart scaled to the values' own range.

Internals: `_run_single`, `_run_parallel` (creates the pool once and reuses it
so load stays continuous and heat accumulates), `_window_worker`, `_verdict`.

---

## `pcbench.scoring`

#### `BASELINES: dict`
Fixed reference rates; baseline == 100.

#### `compute_scores(results) -> dict`
`{"subscores": {...}, "composite": float}`. Skipped, errored, and zero-rate
entries are excluded (a zero would make `log()` raise).

#### `category_scores(subscores) -> dict`
Rolls subscores up into CPU / memory / disk headline numbers.

---

## `pcbench.report`

#### Console
`hr(title)`, `fmt(x)`, `print_system`, `print_results`, `print_native`,
`print_sustained`, `print_scores`, and `print_report(payload)` which calls them
in order and appends any validation failures.

#### `save_json(payload, out_dir) -> str`
Writes `benchmark_<host>_<timestamp>.json`.

#### `append_csv(payload, out_dir) -> str`
Appends one flattened row. If an existing file's header does not match
`CSV_FIELDS`, it is archived to `benchmarks.csv.v2.bak` — appending new columns
to an old file would silently misalign every row.

#### `save_html(payload, out_dir) -> str`
Self-contained HTML report: inline CSS, no external assets, no scripts, all
interpolated values HTML-escaped, and every field read defensively so the page
renders even when a probe returned nothing.

---

## `pcbench.compare`

#### `load_history(csv_path) -> list[dict]`
Reads the CSV history; returns `[]` if absent.

#### `latest_per_host(rows) -> list[dict]`
Keeps the newest run per hostname (ISO-8601 timestamps sort as text), ranked by
composite score.

#### `render_table(rows, all_runs=False) -> str`
Ranked comparison table with each machine's percentage of the best score.
Columns with no data anywhere are omitted.

---

## `pcbench.storage` — device enumeration and per-drive benchmarking

#### `inventory(need_mb=256) -> dict` · `targets(inv, requested, all_devices)`
What is mounted, what can be benchmarked, and which of it to measure. An
explicit path always wins over the skip heuristics: someone naming a mount
knows something the heuristics do not.

#### `run(targets_list, seconds, repeats, file_mb) -> dict`
Runs the standard disk workload plus an fsync test in a `.pcbench`
subdirectory of each target, removing it afterwards.

#### `render_speeds(result) -> str`
The `--drive-speed` table: sequential write, sequential read and 4 KiB random
IOPS per drive. Carries the cache-bypass warning, because a read served from
RAM is not a measurement of the drive.

---

## `pcbench.native`

#### `find_compiler() -> str | None` · `build(src, exe) -> (bool, str)`
#### `run(seconds, repeats, script_dir, threads=None) -> dict | None`
Compiles when the binary is missing or stale, runs with `--json`, returns
parsed output. `None` if the source is absent; `{"error": ...}` on failure.
Never raises.

---

## `pcbench.accel` — GPU / NPU

#### `detect_gpus() -> list[dict]`
Per-platform GPU enumeration: `system_profiler -json` (macOS); `nvidia-smi` →
`lspci` → `/sys/class/drm` (Linux); `Win32_VideoController` (Windows). Each
entry carries name and, where known, vendor, cores, VRAM, driver, Metal family.

#### `detect_npus(cpu_model="") -> list[dict]`
Apple Neural Engine is inferred from an Apple-silicon CPU model (every M-series
Mac has one). Windows matches PnP device names against Neural/NPU/AI Boost/
Hexagon; Linux checks `/dev/accel`, `/sys/class/accel`, and `/dev/amdxdna`.
Entries carry `benchmarkable`, which is only true for the ANE today.

#### `inventory(cpu_model="") -> dict`
`{gpus, npus, gpu_count, npu_count, benchmark_supported}`.

#### `build(src, exe)` · `run(seconds, script_dir, out_dir, gpu=True, ane=True)`
Compiles `accel_engine.m` with clang and the Foundation/Metal/CoreML
frameworks, generates the Core ML model, and runs the engine. Returns `None`
off Apple platforms, `{"error": ...}` on failure. Never raises.

#### `extract_rates(payload) -> dict`
Pulls `gpu_fp32`, `gpu_fp16`, `gpu_bandwidth`, and `npu` out for scoring.
Latency is deliberately excluded — lower-is-better values must not be scored as
throughput.

---

## `pcbench.coreml_model` — ANE model generator

Writes a `.mlmodel` protobuf directly, so no `coremltools` dependency is
needed.

#### `_varint`, `_tag`, `_msg`, `_uint`, `_str`, `_packed_uints`, `_packed_floats`
Minimal protobuf wire-format writers.

#### `build_model(channels=64, spatial=64, layers=12) -> bytes`
Serializes a convolution stack. Convolution is used because it is what the ANE
is built for and what Core ML most reliably offloads. **The defaults are large
on purpose**: Core ML keeps small models on the CPU, where the benchmark would
measure nothing.

#### `flops_per_inference(...) -> float` · `write_model(path, ...) -> str`
FLOPs for one forward pass, and a writer that reuses an identical existing
file.

---

## `pcbench.mlframework` — optional AI training/inference

The only module that may import a third-party framework, and only if installed.

#### `detect() -> dict`
`{pytorch, onnxruntime, available}` — versions if importable, without running
anything.

#### `run(seconds=3.0, batch=64) -> dict`
Prefers PyTorch (trains a small CNN, reports `train_samples_per_s` and
`infer_samples_per_s`, auto-selecting CUDA/ROCm/MPS/CPU); falls back to ONNX
Runtime (inference only). Returns `{"available": False, ...}` when neither is
present. Each timed region ends with a device `synchronize()` so async GPU work
is actually captured.

#### `extract_rates(payload) -> dict`
Pulls `ml_train` / `ml_infer` for scoring.

---

## `pcbench.mlbench` — pure-Python ML workloads

No framework required; every function is deterministic and self-validating.

#### `bench_nn_training(seconds, repeats)` → steps/s
Trains a real 32-24-4 MLP (forward, backprop, SGD). Also returns
`samples_per_s`, `mflops`, and `topology`. Validates that loss falls below 90%
of its initial value.

#### `bench_kmeans(seconds, repeats)` → distances/s
Lloyd's algorithm, 1,200 points × 8D, k=6. Reports `inertia_per_point` and
validates convergence.

#### `bench_knn(seconds, repeats)` → comparisons/s
Brute-force k-NN, 40 queries × 900 refs × 12D. Validates that each point is its
own nearest neighbour.

#### Internals
`_blobs` (deterministic clustered data), `_nn_dataset`, `_nn_init`,
`_nn_train_step` (one full training step), `nn_flops_per_step`, `_sq_dist`,
`_farthest_point_init` (maximin seeding — see
[technical.md](technical.md#k-means-clustering) for why random seeding is
unusable here), `_kmeans`, `_knn`.

---

## `pcbench.onnx_model` — ONNX model generator

Writes the ONNX protobuf directly, so only `onnxruntime` is needed — not the
`onnx` package. Verified against ONNX Runtime 1.29.

#### `build_model(dim=1024, layers=10, batch=32) -> bytes`
A `MatMul` + `Relu` stack sharing one weight initializer, with weights scaled
by `1/dim` so deep stacks cannot overflow.

#### `flops_per_inference(...)` · `write_model(path, ...)`
FLOPs for one forward pass (`2·batch·dim²·layers`), and a writer that reuses an
identical existing file.

---

## `pcbench.npu` — cross-vendor NPU benchmarking

#### `_preload_wheel_cuda() -> list[str]`
Loads the CUDA libraries that `onnxruntime-gpu[cuda,cudnn]` installs under
`site-packages/nvidia/*/lib`, which is on no loader search path. Without this
ONNX Runtime lists `CUDAExecutionProvider`, fails to `dlopen` its provider
library, and falls back to the CPU — reported as "the accelerator did not
engage" when the real problem is a missing `LD_LIBRARY_PATH`. `RTLD_GLOBAL` is
what makes the symbols visible to the provider library that loads later; on
Windows the directory is added with `os.add_dll_directory`. A no-op when the
wheels are absent, when CUDA comes from the system, and on macOS.

#### `_quiet()`
Context manager swallowing ONNX Runtime's own console output while a session is
built. A provider that cannot load is a *result* here, reported by name with a
reason; ORT also announces it in red with C++ source paths, straight into the
middle of the report.

#### `detect() -> dict`
Reports ONNX Runtime availability and which execution providers are installed,
split into all providers and accelerator providers. Sets the ORT logger to
fatal-only for the same reason as `_quiet`.

#### `run(seconds, out_dir) -> dict`
Benchmarks the CPU provider as a baseline, then every accelerator provider, and
returns per-device throughput, GFLOPS, speedup, and an `engaged` flag. Rejects
a result when ONNX Runtime silently fell back to CPU.

#### `extract_rates(payload) -> dict`
Returns the fastest **engaged** accelerator for scoring; an unengaged device is
never scored.

---

## `pcbench.power` — power & perf-per-watt

#### `estimate_tdp(cpu_model) -> int | None`
Rough package-TDP lookup, matching the part-number suffix before the family
name — a Ryzen 7 7800X3D and a 7840U are not both 45 W.

#### `measure(cpu_model="") -> dict`
A reading with its `source` and an `estimated` flag: `powermetrics` (macOS,
sudo), RAPL (Linux), else a labelled TDP estimate.

#### `measure_under_load(cpu_model="", load_s=1.5) -> dict`
Samples power while a background thread burns all cores — active draw, not idle.

#### `perf_per_watt(composite_score, power) -> dict | None`
`score_per_watt`, or None when power is unknown.

---

## `pcbench.network` — loopback stack

#### `run(duration=1.0) -> dict`
TCP loopback throughput (MB/s) plus ping/pong latency (`p50_us`, `p99_us`).
Sends nothing off-box. Never raises.

---

### Internet speed test (opt-in)

#### `internet_speed(server, seconds=5.0, max_mb=200) -> dict`
Download, upload, TCP latency and DNS against `server` (default
`DEFAULT_SPEED_SERVER`, Cloudflare's public endpoint). Never raises; failures
come back as `{"error": ...}` per section. Both directions are capped by time
and by bytes, and upload additionally by a quarter of the byte budget —
links are usually asymmetric, and a metered uplink should not be handed
hundreds of megabytes.

#### `_measure_download(server, seconds, max_bytes) -> dict`
Timed from the **first byte**, so DNS, the TCP handshake and TLS do not count
against the throughput they precede. Stops on whichever budget binds first.

#### `_measure_upload(server, seconds, max_bytes) -> dict`
Two-phase: a 1 MB probe measures the rate, then one body sized to the time
budget is sent. An upload cannot be cut short the way a download can — the
body is committed before timing starts — so the size is chosen from a
measurement rather than guessed. The body is `os.urandom` bytes, incompressible
so a transparent proxy cannot flatter the result, and containing nothing from
the machine.

#### `icmp_ping(host, count=3, timeout=1.0) -> dict`
Round-trip time via the system `ping`, which is setuid everywhere and needs
no privilege this tool would otherwise have to ask for. Per-packet `time=`
lines are parsed rather than the summary, which avoids every localisation
problem and yields the samples jitter needs. A hostname that is empty, starts
with `-`, or contains whitespace is refused before it can reach `ping` as an
option.

#### `ping_summary(icmp, tcp) -> dict`
One round-trip figure with the method that produced it. ICMP when it answers;
otherwise the TCP handshake, which is the same single round trip to a host
already serving the test. Reports `method` because they are not
interchangeable — ICMP is often deprioritised, so the TCP figure can be the
better estimate of what real traffic sees.

#### `_jitter(samples) -> float`
Mean absolute difference between consecutive round trips — RFC 3550. Used by
the two-node test, `tcp_latency` and the ping alike, so the word means one
thing throughout.

#### `render_internet(result) -> str`
Both directions, the ping with its method, and DNS. A section that failed and
a section that never ran are reported differently, and neither crashes the
report.

---

## `pcbench.regression` — run-over-run monitoring

#### `analyze(current_row, history, threshold=0.10) -> dict`
Compares the current flattened row against the median of this hostname's prior
runs (excluding itself). Returns `status` (`ok` / `regression` / `no_baseline`)
and per-metric `findings`, regressions first.

#### `render(result) -> str`
Human-readable summary with ▲/▼ markers.

---

## `pcbench.limits` — hardware-safety caps

Prevents the tool harming the machine it measures. See [safety.md](safety.md).

#### `safe_mem_mb(requested_mb, total_ram_bytes) -> (allowed_mb, notice)`
Clamps the memory buffer to 1/8 of RAM. The test allocates two buffers, so the
footprint stays under a quarter of RAM — clear of swap, which would stall the
machine and write heavily to the SSD.

#### `safe_disk_mb(requested_mb, free_bytes, repeats) -> (allowed_mb, notice)`
Clamps for free-space headroom (1.5x) and cumulative flash wear (16 GB per run
across all repeats).

#### `total_write_mb(file_mb, repeats) -> int`
Bytes this run will write, so wear is always disclosed.

#### `thermal_should_abort(thermal) -> (bool, reason)`
Whether a sustained run should stop early: throttled below 40% of nominal, or
above 100 °C.

---

## `pcbench.thermal` — temperatures, fans, battery

#### `read(script_dir=".") -> dict`
Current temperatures in **Celsius**. macOS builds and runs `sensors_engine`
(unprivileged IOHID); Linux reads `hwmon`/`thermal` plus fan tachometers;
Windows queries WMI. Returns `{}` when nothing is readable — a temperature is
never invented.

#### `cpu_celsius(script_dir) -> float | None` · `describe(temps) -> str`
The headline value, and a one-line summary such as `51.8 °C (normal)`.
Thresholds: warm at 75 °C, hot at 90 °C.

#### `battery_health() -> dict`
Charge cycles and capacity against design capacity. macOS reads
`AppleSmartBattery` via `ioreg`; Linux reads `/sys/class/power_supply`;
Windows uses WMI.

---

## `pcbench.cores` — per-core scaling analysis

#### `scaling_curve(seconds, max_workers) -> list[dict]`
Aggregate throughput at 1..N workers with each worker's marginal gain. Rate is
computed from the workers' own timed duration, not wall time, because pool
creation grows with worker count and would understate the higher counts.

#### `classify_cores(points) -> dict`
Reports how far scaling stays near-linear, whether the machine is hybrid, and
the fast/slow group ratio. **Deliberately does not report exact P/E core
counts** — see [technical.md](technical.md#core-scaling-analysis) for the
measurement that showed why.

#### `per_core_map(seconds) -> list[dict] | None`
Pins a worker to each core for an exact measurement. Returns `None` on macOS,
which exposes no thread-affinity API.

#### `analyze(seconds, max_workers) -> dict`
Curve, classification, and per-core map together.

---

## `pcbench.sysbench` — compilation, OS latency, CPU frequency

#### `bench_compile(repeats) -> dict`
Times a full C compile at `-O2` of a fixed source. One untimed compile runs
first so the compiler and headers are cached; otherwise the first result
measures the filesystem.

#### `_loop_overhead_ns(iterations) -> float`
Nanoseconds per iteration of an empty Python loop, subtracted from the syscall
figure so it reports the kernel transition rather than the transition plus a
bytecode dispatch. The loop cost varies several-fold between interpreters,
which would otherwise make an identical kernel look slower on the slower one.

#### `bench_syscall_latency(iterations) -> float`
Nanoseconds per trivial syscall (`os.getpid`, which CPython does not cache).

#### `bench_context_switch(iterations) -> float`
Nanoseconds per thread context switch, via a two-thread ping-pong.

#### `bench_process_spawn(iterations) -> float`
Milliseconds to create and reap a child process.

#### `bench_latency_suite() -> dict` · `cpu_frequency_mhz() -> float | None`
All latency figures together, and the live clock where the OS exposes it
(Linux `cpufreq`, Windows WMI; `None` on Apple silicon without root).

---

## `pcbench.optional` — optional-package registry

The single source of truth for tiers, consumed by `install.py` and mirrored by
the `pyproject.toml` extras.

#### `TIERS: dict` · `HEAVY: list[Package]`
Tier definitions, each package carrying its import name, pip name, purpose,
approximate size, and whether it is critical to its tier.

#### `status() -> dict`
Which packages are installed, per tier, with `complete` and `usable` flags.

#### `missing(tier_names=None) -> list[Package]` · `have(module) -> bool`
What still needs installing, and a cheap availability check used by benchmark
modules before importing. Uses `find_spec`, so heavyweight packages are never
imported merely to test for them.

#### `version_of(module) -> str | None` · `summary_line() -> str`

#### `tensorrt_distribution() -> str | None`
The TensorRT distribution ONNX Runtime *here* can actually load. ONNX Runtime
pins a TensorRT major and pip's newest runs ahead of it — `pip install
tensorrt` today fetches 11.x while ONNX Runtime 1.29 dlopens
`libnvinfer.so.10` — so the requirement is read out of the provider library's
bytes rather than hardcoded. Resolves to the `-libs` package only, never the
metapackage: ONNX Runtime uses the C++ library and never the Python bindings,
and the bindings are the part that lags (there is no TensorRT 10 binding wheel
for Python 3.14 at all). `None` when it cannot be determined, which means
installing would be 1.2 GB of libraries nothing loads.

#### `applies(pkg) -> bool` · `tier_packages(name) -> list[Package]`
A tier's packages less the ones this machine's hardware rules out. A TensorRT
that could never load is not "not installed yet", it is not applicable, and
listing it as missing on an AMD machine would be an instruction to download
1.2 GB for nothing.

#### `onnxruntime_distribution() -> str` · `pip_target(pkg) -> str | None`
Which wheel to install for a package whose distribution depends on the
hardware. `onnxruntime` is one import name and several distributions, and the
plain PyPI wheel carries only `CPUExecutionProvider` on Windows and Linux —
installing it on a machine with a discrete GPU gives an NPU section that
engages nothing and reports the CPU. Resolves to
`onnxruntime-gpu[cuda,cudnn]` on NVIDIA (the extras bring the CUDA runtime and
cuDNN), `onnxruntime-gpu` for ROCm, `onnxruntime-directml` on Windows, and the
plain wheel on macOS where Core ML is already in it. `pip_target` passes every
other package through unchanged, and returns `None` for a package this
machine's hardware rules out. It is called once per package immediately before
installing, not once for the batch: TensorRT's version depends on what
`onnxruntime` — earlier in the same tier — put on disk.

#### `opencl_icd_hint() -> str | None`
The command that registers an OpenCL driver on this machine, or None where the
ICD arrives with the driver (Windows) or the OS (macOS), or where no package
manager is recognised — a wrong command is worse than none. `pyopencl` is only
the binding; without an ICD the loader raises `PLATFORM_NOT_FOUND_KHR`, which
reads like a hardware fault and is not one, and pip cannot fix it.

---

## `pcbench.numeric` — BLAS / LAPACK (needs numpy, scipy)

#### `bench_matmul(seconds, repeats) -> dict`
Dense N x N multiply (2*N^3 FLOPs) in FP64 and FP32 through the platform BLAS,
which it names. The headline rate is FP64.

#### `bench_fft(seconds, repeats) -> dict`
Complex FFT throughput (~5*N*log2 N FLOPs) — memory-bound rather than
arithmetic-bound, so it probes a different limit from matmul.

#### `bench_lapack(seconds) -> dict`
Cholesky, SVD, and eigenvalue decompositions per second.

#### `run(seconds, repeats) -> dict` · `extract_rates(payload) -> dict`

---

## `pcbench.cryptobench` — hardware crypto and modern codecs

#### `bench_aes(seconds, repeats) -> dict`
AES-256-GCM throughput via OpenSSL, which dispatches to AES-NI or the ARMv8
crypto extensions. Validates an encrypt/decrypt round trip first.

#### `bench_zstd` · `bench_lz4` · `bench_blake3`
Zstandard and LZ4 throughput with compression ratios, and BLAKE3 hashing. Each
validates a round trip or digest before timing.

#### `run(seconds, repeats) -> dict` · `extract_rates(payload) -> dict`
Individual failures are captured per codec, so one missing package never loses
the others.

---

## `pcbench.gpucompute` — cross-platform GPU (needs pyopencl)

#### `devices() -> list[dict]`
Enumerates OpenCL devices — name, platform, type, compute units, memory, clock
— without benchmarking them.

#### `run(seconds) -> dict`
Benchmarks every OpenCL device: FMA throughput in GFLOPS and copy bandwidth in
MB/s, using kernels mirroring the Metal engine so the numbers are comparable.
Cross-validated on an M4: Metal 2,369 GFLOPS vs OpenCL 2,281. **Every failure
path still returns `matmul` and `nvidia`** — the two backends fail
independently, and a discrete card with a working CUDA build and no registered
ICD is an ordinary machine, not an edge case.

#### `_enumeration_note(error) -> str`
Turns an OpenCL enumeration failure into something actionable.
`PLATFORM_NOT_FOUND_KHR` is by far the most common and reads like a hardware
fault when it is not one: the loader is present, no vendor has registered a
driver with it, and the GPU is fine. Names the per-platform fix.

#### `torch_matmul(seconds) -> dict`
Dense N=4096 GEMM at fp32 and fp16 through CUDA, ROCm, XPU or MPS. This is the
AI-compute figure, and OpenCL cannot reach the matrix hardware that produces
it. Returns `{"skipped": True, "reason": ...}` when no torch GPU device exists.

#### `nvidia_telemetry() -> list[dict]`
NVIDIA temperature, power draw, fan, VRAM, and utilisation via `pynvml`. Each
metric is fetched independently so a card omitting one does not lose the rest.

#### `extract_rates(payload) -> dict`

---

## `pcbench.interference` — mid-run condition detection

Machine state is checked before a run, but a run takes minutes and the machine
can change underneath it. Every repeat inside a test is equally affected, so
statistics cannot rescue a disturbed measurement — it has to be labelled.

#### `sample(script_dir) -> dict`
Cheap snapshot of load per core and CPU temperature.

#### `compare_samples(before, after) -> dict`
Flags a test when load rose more than 0.25 per core, the CPU warmed more than
12 °C, or it was already above 85 °C.

#### `summarize(results) -> dict`
Run-level verdict listing which tests ran under changing conditions.

---

## `pcbench.diagnose` — bottleneck analysis and spec sheet

#### `analyse(scores) -> dict`
Compares category scores against the machine's **own median**, so the answer
is "what is weak *for this machine*" — which is what determines whether an
upgrade helps. Derived categories (`ai`, a roll-up of gpu/npu/ml) are excluded
so they cannot double-count.

#### `render(result) -> str`
Bar chart of subsystem scores with the bottleneck and strongest marked, plus
plain-language impact ("storage is the limit: application launches and builds
will feel slow regardless of CPU").

#### `spec_sheet(payload) -> str`
One-page Markdown summary — hardware, headline results, subsystem scores, and
the assessment. For a support ticket or a listing.

---

## `pcbench.checkup` — diagnosis

Answers "why is this slower than it should be?", which is a different job from
scoring. Gathers evidence, then ranks findings by likely impact.

#### `gather(script_dir=".") -> dict`
Every cheap source of evidence: inventory, machine state, processes, memory,
disks, power mode, provenance, drive health, uptime, confinement. A source
that fails is recorded as an error rather than omitted, so a check can report
its area as unexamined instead of clean.

#### `top_processes(limit=6) -> dict` · `memory_pressure()` · `uptime_seconds()` · `disk_headroom()` · `power_mode()`
The evidence the rest of the tool did not already collect. `top_processes`
sorts in Python because the sort flag differs between BSD and GNU `ps` while
the output does not. `disk_headroom` skips pseudo-filesystems, which report
themselves permanently full, and deduplicates volumes sharing one pool (APFS,
Btrfs) so a shared free-space figure is reported once.

#### `probe(seconds=1.0, droop_seconds=8.0) -> dict`
A deliberately small measurement. Not to score the machine — the benchmark
does that — but to catch the two faults that only appear under load: a
subsystem below any reasonable floor, and throughput that collapses as the
machine heats up.

#### Checks
`_check_thermal` (throttling, strain, temperature) · `_check_power` (battery,
power profile, boost disabled) · `_check_cpu` (SMT off, mitigation load,
restricted affinity) · `_check_gpu` (NVIDIA temperature, VRAM exhaustion,
untestable multi-GPU) · `_check_contention` (load per core, named processes) ·
`_check_memory` (available, swap) · `_check_ram_config` (total RAM, cgroup
cap) · `_check_storage` (volume headroom, SMART) · `_check_configuration`
(uptime, CPU quota, virtualization) · `_check_measured` (subsystem floors,
sustained droop) · `_check_regression` (slower than this machine's own
history).

There is deliberately no CPU *floor*. A slow CPU with no history is almost
always an old CPU, and calling a specification a fault is the mistake the
reference floors were rewritten to avoid.

#### `history_rows(output_dir="results") -> list[dict]`
Past runs for this hostname, oldest first. `_check_regression` compares the
probe against them — the strongest signal available that a machine actually
*got* slower, because it controls for the hardware entirely. Only
`cpu_int` is compared: the probe's disk and memory sizes differ from a full
run's, and comparing those reported a 74% disk regression on a healthy
machine.

#### Platform parsers
`parse_proc_stat` (Linux `/proc/PID/stat` — the comm field is parenthesised
and may contain spaces and parentheses, so the split starts after the last
`)`) · `parse_meminfo` · `parse_vm_stat` · `parse_swapusage` ·
`_parse_windows_processes`. Each is separated from its I/O so all three
platforms are testable from any one of them.

On Linux, processes are sampled from `/proc` twice rather than read from
`ps`, because `ps` reports %CPU averaged over the whole life of a process —
something that saturated a core for an hour and then went idle still reads
high, which is the wrong answer for "what is using the CPU right now".

#### `analyse(evidence) -> list[dict]`
Every finding the evidence supports, most serious first. A check that raises
becomes an "unexamined area" finding rather than taking the report down with
it. Each finding carries `severity`, `area`, `title`, `evidence`, `impact` and
`fix`.

#### `verdict(result) -> str` · `render(result) -> str`
One sentence naming the most likely cause, then each finding with its
evidence and remedy, then the numbers it was all based on — shown whether or
not anything fired.

**Severity** — `critical` means measurably hurting now; `warning` means a
common cause is present; `info` is context that shapes the other findings.
Info-only is still a clean bill: long uptime and a spinning disk are facts,
not explanations.

---

## `pcbench.health` — RAM integrity and drive SMART

#### `memory_integrity(size_mb, ram_bytes) -> dict`
Writes six adversarial bit patterns (all-zeros, all-ones, both alternating
patterns, both nibble patterns) and verifies each reads back. Catches stuck
bits and coupling between adjacent cells.

**Scope is reported with every result**: it tests only memory this process
could allocate, through the OS's virtual memory. A pass does **not** certify
the RAM — only a bootable tester owning the whole address space can.

#### `drive_health() -> dict`
Read-only drive self-assessment. macOS uses `system_profiler` (no privileges
needed); elsewhere `smartctl` if installed. **Never writes SMART data** — a
test asserts no mutating commands appear in the module.

---

## `pcbench.plugins` — user-supplied benchmarks

#### `discover(root) -> list[dict]`
Loads every `.py` in `plugins/`. A plugin needs `NAME`, `UNIT`, `BASELINE`, and
`run(seconds, repeats)` returning a dict with `rate`. Invalid or raising
plugins are reported and skipped, never fatal.

#### `run_all(plugins, seconds, repeats) -> dict` · `scores(results) -> dict`
Executes each plugin and scores it against its own declared baseline, so it
joins the composite like any built-in metric.

---

## `pcbench.cli`

#### `build_parser() -> ArgumentParser` · `main(argv=None) -> int` · `entry()`
Exit codes: `0` success, `2` bad arguments, `3` refused due to machine state,
`4` validation failure.

#### `parse_duration(text) -> float`
Parses `90`, `30s`, `5m`, `1h`. Rejects non-positive and malformed values.

#### `select_tests(only, skip) -> list[str]`
Resolves `--only`/`--skip` against `TESTS`, raising `ValueError` on unknown
names.

#### `_runners(args, info, disk_dir) -> dict`
Maps every test name to its callable. A test in `TESTS` without an entry here
is caught by the suite.

#### `--menu` dispatch
`main()` calls `wizard.run()`, then re-enters `main()` with the argv the
wizard returned — so a menu choice takes exactly the path a typed command
would. Flags given alongside `--menu` are named and discarded, since the
wizard builds a fresh command line.

---

## `pcbench.wizard` — the guided menu

Screens and what they build. The widgets themselves are `pcbench.tui`; this
module holds only the flow.

#### `run() -> list[str] | None`
Draws the menu inside `tui.screen()` and returns the argv to run, or `None`
if the user left. Prints the assembled command line afterwards, outside the
alternate buffer, so it survives in the scrollback.

#### `SHORTCUTS: list[tuple[str, list[str]]]`
The flat list of common tasks, reachable from the main menu. Every entry is a
real command line; a test asserts the parser accepts each one, so a dead menu
item cannot ship.

#### `_run_steps(steps, state) -> list[str]`
The step machine behind every branch. Each step mutates `state` and may raise
`tui.Back`; a step that returns `_SKIP` does not apply to the answers so far,
and the machine keeps moving in the direction it was already going. That is
what stops a skipped screen from bouncing the user forward again when they
step back through it. Walking off the front re-raises `Back`, which the main
menu catches.

#### `_benchmark()` · `_stats()` · `_watch()` · `_health()` · `_history()`
One step list each: e.g. the benchmark branch is kind → scope → depth →
extras → output → confirm. Scope only appears for a focused run (data
science, I/O, standards); depth is skipped when the chosen kind already
fixes it.

#### `_confirm(trail, argv, plan) -> None`
The last screen of every branch: the command line, a plain-English plan, and
run / go back / quit. Nothing has started when it is drawn.

#### `_validate(argv) -> str`
Parses the assembled argv with the real parser and returns an error string if
it would be rejected — so a wizard bug surfaces as a screen, not a traceback
after the user commits.

---

## `pcbench.tui` — raw-terminal widgets

No dependency: `termios`/`tty` on Unix, `msvcrt` on Windows, ANSI for drawing.
Every widget falls back to a typed prompt when the terminal cannot be driven
key by key, which is what keeps `--menu` scriptable.

#### `supported() -> bool`
True when stdin and stdout are both a TTY, `TERM` is usable, and
`PCBENCH_NO_TUI` is unset. On Windows it also enables
`ENABLE_VIRTUAL_TERMINAL_PROCESSING` via ctypes, and reports whether that
worked.

#### `screen()` (context manager)
Alternate buffer, hidden cursor, raw mode — all restored in a `finally`,
including on Ctrl-C, because a terminal left in raw mode breaks the user's
shell. Yields `False` and does nothing when `supported()` is false.

#### `read_key() -> str`
One keypress as a name (`UP`, `ENTER`, `BACKSPACE`, `ESC`) or the character
itself. `_read_posix` distinguishes a bare Esc from the start of an arrow
sequence by a 50 ms timeout, since nothing follows a bare Esc.

#### `select(title, question, options, ...) -> int`
One choice. Arrows or `j`/`k` move, a digit jumps, Enter selects, Esc raises
`Back`, `q` raises `Quit`.

#### `multiselect(title, question, options, default="none", ...) -> list[int]`
Checkboxes. Space toggles, `a` ticks or clears everything, Enter accepts.
`default` pre-ticks entries (`"all"`, `"none"`, or the same comma syntax the
typed path takes); `allow_empty=False` refuses an empty answer.

#### `text(title, question, label, default, validate=None, ...) -> str`
A one-line field. Printable keys type, Backspace deletes, Ctrl-U clears,
Enter accepts the default when nothing was typed. `q` is a character here,
not a command. `validate` re-asks with the error shown in place.

#### `parse_selection(answer, count, names=None) -> list[int]`
The typed path's grammar: `1,4`, ranges `1-6`, `all`, `none`, or names.
Duplicates collapse and order is kept.

#### `_window(count, cursor, room) -> (start, end)`
The slice of a long list to draw, keeping the cursor on screen.

#### `_widest_that_fits(candidates, width) -> str`
Picks the widest key-hint line that fits, so a narrow terminal loses the
optional hints rather than truncating the one that says how to leave.

---

## `native_engine.c`

**Platform layer** — `now_seconds()` (`QueryPerformanceCounter` / 
`clock_gettime`), `cpu_count()`, and a `thread_t` typedef over Win32 threads
and pthreads.

**Workloads** — `is_prime`, `cpu_integer_chunk` (returns the count for
validation), `cpu_float_chunk`.

**Statistics** — `median`, `stddev`.

**Runners** — `run_rate`, `run_memory` (memcpy bandwidth, verified),
`run_multithread(seconds, nthreads)` (real threads, no GIL or spawn cost),
`run_disk` (uses `F_NOCACHE` before writing on macOS, `posix_fadvise` on Linux).

**Latency** — `build_cycle` (Sattolo's algorithm, guaranteeing a single cycle
the prefetcher cannot predict) and `pointer_chase_ns(bytes, seconds)`.

**Output** — `print_human`, `print_json` (the contract `pcbench.native`
consumes), `main`. Exits 2 if validation failed.

---

## `accel_engine.m` (Apple)

**Shaders** — `kShaderSource` holds Metal Shading Language compiled at runtime
via `newLibraryWithSource:`: `fma_f32` / `fma_f16` (four independent dependent
FMA chains, so the compiler cannot vectorize the work away), `bandwidth`
(float4 copy), and `nop` (launch latency).

**`run_gpu(seconds)`** — device info plus FP32/FP16 GFLOPS, memory bandwidth,
and kernel launch latency.

**`run_coreml(compiled, units, shape, seconds, ok)`** — inferences/sec under one
`MLComputeUnits` setting, after a warm-up that absorbs model load, weight
conversion, and ANE program compilation.

**`run_ane(modelPath, seconds, flops, shape)`** — compiles the model, then runs
it CPU-only, CPU+ANE, and All. Reports throughput, effective GFLOPS, and the
speedup; flags `engaged` only above 1.5x, since Core ML never states placement
directly.
