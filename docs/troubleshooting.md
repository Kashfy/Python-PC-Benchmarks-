# Troubleshooting

Common issues, what causes them, and how to fix them. Most problems are
environmental (compiler, permissions, thermal state) rather than bugs.

## The native (C) engine section is missing or shows an error

**"no C compiler found (cc/clang/gcc); skipped"**
No compiler is on `PATH`. The Python benchmarks still ran — only the
compiler-optimized comparison is skipped. Install one if you want it:

- macOS: `xcode-select --install`
- Debian/Ubuntu: `sudo apt install build-essential`
- Fedora: `sudo dnf install gcc`
- Windows: install MSVC (Visual Studio Build Tools) or MinGW-w64, then run from
  a Developer Command Prompt, or just use `--no-native`.

**"native build failed"**
The `detail` field carries the compiler's stderr. Usually a missing math
library or a toolchain misconfig. Try building by hand to see the full error:

```bash
cc -O2 native_engine.c -o native_engine -lm
```

**"native run failed" / "native run error"**
The binary built but exited non-zero or produced unparseable output — often a
sandbox blocking the temp-file write. Run it directly to inspect:

```bash
./native_engine --json --seconds 1 --repeats 1
```

To skip the engine entirely: `python3 benchmark.py --no-native`.

## Disk test is skipped

**"not enough free space for N MB disk test"**
The guard requires free space ≥ 1.2 × the test file. Lower the size or point
the scratch dir elsewhere:

```bash
python3 benchmark.py --disk-mb 64
python3 benchmark.py --output-dir /path/on/a/bigger/disk
```

## Disk read numbers look impossibly high

Read throughput above what the drive can physically do (e.g. tens of GB/s on a
SATA SSD) means the read was served from the **OS page cache**, not the device.
The tool drops the cache on Linux (`posix_fadvise`) but cannot on macOS/Windows
without elevated privileges. Mitigations:

- Increase `--disk-mb` well beyond a few hundred MB so the file exceeds cache.
- Treat read numbers as a **floor**, and trust the write number (post-`fsync`)
  as the more device-accurate figure.

## Multi-core scaling is lower than the core count

Expected. A 10-core chip rarely gives 10× the single-core rate because of:

- **Hybrid cores** — efficiency cores are slower than performance cores.
- **SMT/Hyper-Threading** — two logical threads share one physical core.
- **Spawn overhead** — process startup is counted in wall time; it dominates on
  very short runs.

Use `--seconds 5` or more for a cleaner scaling figure. See
[technical.md](technical.md#cpu-multi-core--primes-s--scaling-factor).

## Results vary a lot between runs (high stdev)

Benchmarks are sensitive to machine state. To stabilize:

- Close background apps and browser tabs.
- Plug in laptops — battery power profiles throttle the CPU.
- Let the machine cool between runs (thermal throttling depresses later runs).
- Increase `--seconds` and `--repeats` (e.g. `--seconds 5 --repeats 5`).

A high `stdev` relative to the `median` in the JSON is the signal to do this.

## `CPU model` shows an architecture string instead of a name

On some Linux ARM boards `/proc/cpuinfo` has no "model name" line. The tool
falls back to the device-tree model, then "Hardware"/"Model", then the raw
architecture. This is cosmetic — the benchmark numbers are unaffected. The
`architecture` and `arch_family` fields still identify the chip class.

## RAM or physical-core count shows `null` / `unknown`

The OS-specific probe couldn't read it (unusual environment, container, or
locked-down system). Installing `psutil` often fixes detection:

```bash
python3 -m pip install psutil
```

Benchmarks run normally regardless; only the inventory field is affected.

## `Unknown test(s): ...` and the tool exits

An invalid name was passed to `--only`. Valid values are exactly:
`cpu_int`, `cpu_float`, `cpu_multi`, `memory`, `disk`.

## `RuntimeError` about multiprocessing / repeated banner output on Windows

This happens if the module is run in a way that bypasses the `__main__` guard.
Always launch as a script (`python benchmark.py`), not by importing and calling
`main()` from an interactive shell without a `if __name__ == "__main__"` guard.
The script already calls `mp.freeze_support()` for frozen/`.exe` builds.

## Permission denied writing results

The default `results/` directory couldn't be created or written. Redirect
output:

```bash
python3 benchmark.py --output-dir ~/bench-results
# or don't write files at all:
python3 benchmark.py --no-save
```

## Python version too old

Errors mentioning `isqrt`, `fmean`, or f-strings mean Python < 3.8. Install a
newer Python 3 and run with it explicitly (e.g. `python3.12 benchmark.py`).

## Still stuck?

Capture a full machine-readable dump to share when reporting an issue:

```bash
python3 benchmark.py --json-stdout --no-save > run.json
```

That file contains the system inventory, every raw sample, and any per-test
error messages.
