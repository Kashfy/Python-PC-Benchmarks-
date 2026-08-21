# Function Reference

Every public function in `benchmark.py` and `native_engine.c`, grouped by role.
Signatures reflect the source; see [technical.md](technical.md) for the
methodology behind them.

---

## `benchmark.py`

### Console formatting

#### `hr(title="") -> None`
Prints a 70-character rule, optionally with a centered title line. Used to
separate report sections.

#### `fmt_num(x) -> str`
Formats a number for the console: thousands-separated with no decimals when
≥ 1000, one decimal in `[10, 1000)`, three decimals below 10.

### Timed-loop core

#### `timed_loop(chunk_func, seconds) -> (elapsed, iterations)`
Calls `chunk_func` repeatedly until `seconds` elapse. Returns the actual
elapsed time and the number of chunks executed. The caller multiplies
`iterations` by the work-per-chunk constant to get a rate.

### Workloads

#### `_is_prime(n) -> bool`
Trial-division primality test up to `math.isqrt(n)`.

#### `cpu_integer_chunk() -> None`
One integer chunk: primality-tests every integer in `[50000, 51000)`.

#### `cpu_float_chunk() -> None`
One float chunk: 50,000 iterations of `sin(x)*cos(x) + sqrt(x)`.

#### `_multicore_worker(duration) -> int`
Top-level, picklable worker for `multiprocessing`. Runs integer chunks for
`duration` seconds and returns the number of primes tested. Times itself to
avoid cross-process clock issues.

### Benchmark runners

Each returns a dict containing at least `unit` and `rate` (the median), plus
the full statistics from `_summarize`.

#### `_summarize(samples) -> dict`
Reduces a list of per-repeat rates to `{median, mean, stdev, min, max,
samples}`. `stdev` is 0 for a single sample.

#### `bench_cpu_integer(seconds, repeats) -> dict`
Runs the integer workload `repeats` times. Unit `primes/s`.

#### `bench_cpu_float(seconds, repeats) -> dict`
Runs the float workload `repeats` times. Unit `iters/s`.

#### `bench_cpu_multicore(seconds, logical_cores) -> dict`
Runs the integer workload across `logical_cores` spawned processes. Returns the
aggregate `rate`, the worker count, wall time, and per-worker prime counts.
Unit `primes/s`.

#### `bench_memory(seconds, repeats, buf_mb=64) -> dict`
Measures memory copy bandwidth by repeatedly copying a `buf_mb` buffer. Unit
`MB/s`.

#### `_advise_dropcache(fd) -> None`
Best-effort hint (`posix_fadvise DONTNEED` on Linux) to drop the OS page cache
for a file descriptor before the read test. No-op where unsupported.

#### `bench_disk(seconds, repeats, file_mb, out_dir) -> dict`
Writes and reads a `file_mb` file in `out_dir`. Returns `write_rate` and
`read_rate` (medians) plus full `write`/`read` stat blocks. Returns
`{"skipped": True, "error": ...}` if free space is insufficient.

### Hardware / OS inventory

#### `_run(cmd) -> str`
Runs a command with a 5-second timeout and returns trimmed stdout, or `""` on
any error. Used to shell out to `sysctl` etc.

#### `_cpu_model() -> str`
Returns a human CPU name. Sources by OS: `sysctl machdep.cpu.brand_string`
(macOS); `/proc/cpuinfo` "model name", device-tree model, or "Hardware"/"Model"
(Linux); `PROCESSOR_IDENTIFIER` (Windows). Falls back to
`platform.processor()`/`platform.machine()`.

#### `_total_ram_bytes() -> int`
Total physical RAM in bytes. Uses `psutil` if available, else `sysctl
hw.memsize` (macOS), `/proc/meminfo` (Linux), or `GlobalMemoryStatusEx` via
`ctypes` (Windows). Returns 0 if undetectable.

#### `_physical_cores() -> int | None`
Physical (not logical) core count. Uses `psutil` if available, else `sysctl
hw.physicalcpu` (macOS) or `physical id`/`core id` pairs in `/proc/cpuinfo`
(Linux). Returns `None` if undetectable.

#### `_has_psutil() -> bool`
Whether `psutil` can be imported.

#### `_arch_family(machine) -> str`
Maps a raw `platform.machine()` string to a canonical ISA family (see the table
in [technical.md](technical.md)).

#### `gather_system_info() -> dict`
Assembles the full inventory: hostname, OS/release/version/platform,
architecture + `arch_family` + bits + byte order, CPU model, physical/logical
cores, total RAM (bytes and GB), Python version/implementation, and whether
psutil is present.

### Native engine integration

#### `_find_compiler() -> str | None`
Returns the first available compiler among `cc`, `clang`, `gcc`, or `None`.

#### `run_native_engine(seconds, repeats, script_dir) -> dict | None`
Compiles `native_engine.c` (only if the binary is missing or stale), runs it
with `--json`, and returns the parsed JSON. Returns `None` if the source is
absent, or an `{"error": ...}` dict if no compiler is found or the build/run
fails. Never raises.

### Scoring

#### `compute_scores(results) -> dict`
Normalizes each available rate against its `BASELINES` entry (baseline = 100)
and returns `{"subscores": {...}, "composite": <geometric mean>}`.

### Output

#### `print_console_report(info, results, scores, native) -> None`
Renders the full human-readable report: system info, benchmark results, the
native section (if present), and scores.

#### `save_json(payload, out_dir) -> str`
Writes the full payload to `results/benchmark_<host>_<timestamp>.json` and
returns the path.

#### `append_csv(payload, out_dir) -> str`
Appends one flattened row (headline rates + composite) to
`results/benchmarks.csv`, writing a header first if the file is new. Returns the
path.

### Entry point

#### `parse_args(argv=None) -> argparse.Namespace`
Defines and parses all CLI flags (see [packages.md](packages.md) /
`README.md` for the flag table).

#### `main(argv=None) -> int`
Orchestrates a full run and returns a process exit code (0 success, 2 for a
bad `--only` value).

---

## `native_engine.c`

### Timing & platform

- `now_seconds()` — monotonic high-resolution clock
  (`QueryPerformanceCounter` on Windows, `clock_gettime(CLOCK_MONOTONIC)` on
  POSIX).

### Workloads

- `is_prime(n)` — trial-division primality test.
- `cpu_integer_chunk()` — primality over `[50000, 51000)`, accumulating into a
  `volatile` sink so the optimizer can't delete the work.
- `cpu_float_chunk()` — 50,000 iterations of `sin*cos + sqrt`, into a
  `volatile` double sink.

### Statistics

- `median(v, n)` — insertion-sorts a small array and returns the median.
- `stddev(v, n)` — sample standard deviation (0 for n < 2).

### Runners

- `run_rate(func, seconds, units_per_chunk)` — the timed-loop core; returns
  work-units per second.
- `run_memory(seconds, buf_mb)` — `memcpy` bandwidth in MB/s.
- `run_disk(seconds, file_mb, *out_write, *out_read)` — sequential write/read
  MB/s via a temp file (`GetTempPathA`/`fopen` on Windows, `mkstemp`+`fsync` on
  POSIX).

### Output & entry

- `print_human(r, n)` — human-readable table.
- `print_json(r, n, seconds, repeats)` — the JSON contract consumed by
  `benchmark.py`.
- `main(argc, argv)` — parses `--json/--seconds/--repeats/--mem-mb/--disk-mb`,
  runs all workloads `repeats` times, and prints results.
