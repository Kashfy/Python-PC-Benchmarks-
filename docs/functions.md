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
