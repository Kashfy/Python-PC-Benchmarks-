# Packages, Dependencies & Toolchain

What the tool needs to run, what it optionally uses, and the CLI surface.

## Runtime requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | **3.8 or newer** | Uses f-strings, `math.isqrt`, `statistics.fmean`, and `from __future__ import annotations`. |
| OS | Windows, macOS, or Linux | All three have dedicated code paths. |
| C compiler | **Optional** | Only needed for the native engine (`cc`, `clang`, or `gcc`; MSVC `cl` for manual builds). |

**No `pip install` is required.** `benchmark.py` imports only the Python
standard library.

## Python standard-library modules used

All are bundled with CPython — nothing to install:

| Module | Used for |
|--------|----------|
| `argparse` | CLI parsing |
| `json` | JSON output + parsing the native engine's output |
| `math` | `isqrt`, `sin`/`cos`/`sqrt`, `exp`, `log` (scoring & workloads) |
| `multiprocessing` | Multi-core CPU test (`spawn` context) |
| `os` | File descriptors, `cpu_count`, `posix_fadvise`, env vars |
| `platform` | OS / arch / CPU / Python identification |
| `re` | Parsing `/proc/meminfo`, sanitizing filenames |
| `shutil` | `which` (compiler discovery), `disk_usage` (space guard) |
| `statistics` | `median`, `fmean`, `stdev` |
| `subprocess` | Running `sysctl` and the native engine |
| `sys` | `maxsize` (bitness), `byteorder`, exit codes |
| `tempfile` | Disk-test scratch files, fallback scratch dir |
| `time` | `perf_counter` timing |
| `datetime` | UTC timestamps |
| `csv` | Appending to `benchmarks.csv` (imported inside `append_csv`) |
| `ctypes` | Windows RAM query (`GlobalMemoryStatusEx`), imported lazily |

## Optional dependency: `psutil`

`psutil` is **not required**. If it is importable, it is used to improve two
fields — total RAM (`virtual_memory().total`) and physical core count
(`cpu_count(logical=False)`). Without it, the tool falls back to native OS
probes (`sysctl`, `/proc/*`, `ctypes`). The `system.psutil_available` field in
the output records which path was taken.

To install it (only if you want the marginally richer detection):

```bash
python3 -m pip install psutil
```

## Native engine toolchain

`native_engine.c` is standard C11 with no third-party libraries. It links only
the C runtime and, on POSIX, the math library (`-lm`).

| Platform | Build command |
|----------|---------------|
| macOS / Linux | `cc -O2 native_engine.c -o native_engine -lm` |
| Windows (MinGW) | `gcc -O2 native_engine.c -o native_engine.exe` |
| Windows (MSVC) | `cl /O2 native_engine.c` |

`benchmark.py` does this automatically via `run_native_engine()`, rebuilding
only when the binary is missing or older than the source. Pass `--no-native` to
skip it entirely.

## Command-line interface

```
python3 benchmark.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seconds N` | `3.0` | Target duration per test, per repeat |
| `--repeats M` | `3` | Repeats per test (median reported) |
| `--disk-mb K` | `256` | Disk-test file size (MB) |
| `--mem-mb K` | `64` | Memory-test buffer size (MB) |
| `--only a,b` | all | Subset of `cpu_int,cpu_float,cpu_multi,memory,disk` |
| `--quick` | off | Fast preset: 1s × 2 repeats, 64 MB disk |
| `--no-native` | off | Skip the native C engine |
| `--no-save` | off | Don't write JSON/CSV |
| `--output-dir D` | `results` | Output directory for JSON/CSV |
| `--json-stdout` | off | Print the full payload as JSON to stdout |

The native engine accepts `--json`, `--seconds`, `--repeats`, `--mem-mb`, and
`--disk-mb`.

## Output artifacts

Written to `--output-dir` (default `results/`, git-ignored):

| File | Contents |
|------|----------|
| `benchmark_<host>_<timestamp>.json` | Full structured payload for one run |
| `benchmarks.csv` | One appended row per run for spreadsheet comparison |

## Version support notes

- `statistics.fmean` requires Python **3.8+**; `math.isqrt` requires **3.8+**.
  These set the floor.
- `os.posix_fadvise` exists only on Linux; its absence elsewhere is handled.
- `multiprocessing` `spawn` is the default on Windows/macOS and forced on
  Linux here for consistent behavior.
