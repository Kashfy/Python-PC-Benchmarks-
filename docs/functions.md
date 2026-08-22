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
IOPS. Returns `write_rate`, `read_rate`, `random_read_iops`, and
`cache_bypassed`. Skips with a reason when free space is insufficient.

#### `_set_nocache(fd) -> bool` · `_drop_cache(fd) -> bool`
Page-cache defeat. `_set_nocache` (macOS `F_NOCACHE`) **must** run before the
file is written — it prevents caching but does not evict. `_drop_cache`
(Linux `posix_fadvise`) evicts after `fsync`.

#### `_random_read_iops(fd, size, budget) -> float`
4 KiB reads at random offsets, batched so `clock()` overhead stays off the
measured path.

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

#### `detect() -> dict`
Reports ONNX Runtime availability and which execution providers are installed,
split into all providers and accelerator providers.

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
Rough package-TDP lookup by chip family.

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
Cross-validated on an M4: Metal 2,369 GFLOPS vs OpenCL 2,281.

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
