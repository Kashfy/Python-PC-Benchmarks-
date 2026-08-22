# Architecture

How the tool is put together, and why.

## Layout

```
Python-PC-Benchmarks-/
├── benchmark.py        # zero-install launcher (python3 benchmark.py)
├── install.py          # optional-package installer (venv by default)
├── pyproject.toml      # installable package + per-tier extras
├── native_engine.c     # optional native engine (C): threads, pointer chase
├── accel_engine.m      # optional GPU/NPU engine (Apple: Metal + Core ML)
├── sensors_engine.m    # optional temperature reader (Apple: IOHID)
├── pcbench/
│   │   ── core plumbing ──
│   ├── core.py         # timing, statistics, warm-up, validation
│   ├── limits.py       # hardware-safety caps (memory/disk/wear/thermal)
│   ├── optional.py     # registry of optional packages, grouped in tiers
│   ├── config.py       # config files (TOML/JSON) and PCBENCH_* variables
│   ├── stats.py        # confidence intervals, Mann-Whitney U, A/B verdicts
│   ├── cli.py          # argument parsing and run orchestration
│   │   ── inventory and state ──
│   ├── system.py       # hardware inventory, CPU features, machine state
│   ├── thermal.py      # temperatures in Celsius, fans, battery health
│   ├── power.py        # power draw + perf-per-watt
│   ├── container.py    # container / cgroup / cloud / CI confinement
│   ├── provenance.py   # governor, mitigations, hugepages, SMT, microcode
│   ├── counters.py     # PMU counters via perf, plus rusage everywhere
│   ├── numa.py         # NUMA topology and local/remote bandwidth matrix
│   ├── storage.py      # mount enumeration, device classification
│   ├── monitor.py      # live telemetry mode (no benchmarking)
│   │   ── stdlib benchmarks ──
│   ├── workloads.py    # cpu, memory, disk (queue depth), real-world
│   ├── apps.py         # application-shaped: sqlite, fsync, raytrace, video
│   ├── standards.py    # STREAM, LINPACK/HPL, CoreMark-style
│   ├── iobench.py      # fio-shaped configurable storage jobs
│   ├── datascience.py  # LLM prefill/decode, input pipeline, dataframes
│   ├── mlbench.py      # pure-Python ML: NN training, k-means, k-NN
│   ├── cores.py        # per-core scaling analysis and hybrid detection
│   ├── sysbench.py     # compile benchmark, OS latency, CPU frequency
│   ├── network.py      # loopback stack + opt-in external tests
│   ├── interference.py # per-test detection of changed conditions
│   ├── health.py       # RAM integrity, drive SMART (read-only)
│   ├── diagnose.py     # bottleneck analysis, spec sheet
│   ├── plugins.py      # discover and run user benchmarks
│   ├── sustained.py    # thermal / sustained-load mode
│   ├── soak.py         # long-duration burn-in with error accounting
│   │   ── optional-package benchmarks ──
│   ├── numeric.py      # BLAS matmul, FFT, LAPACK (numpy, scipy)
│   ├── cryptobench.py  # AES-GCM, Zstandard, LZ4, BLAKE3
│   ├── gpucompute.py   # cross-platform GPU via OpenCL, NVIDIA telemetry
│   ├── mlframework.py  # PyTorch / ONNX training and inference
│   │   ── accelerators ──
│   ├── accel.py        # GPU/NPU inventory + Apple accelerator engine
│   ├── coreml_model.py # generates the ANE benchmark model (raw protobuf)
│   ├── npu.py          # cross-vendor NPU via ONNX Runtime providers
│   ├── onnx_model.py   # generates the ONNX benchmark model (raw protobuf)
│   ├── native.py       # build + run the C engine
│   │   ── output ──
│   ├── scoring.py      # baselines, subscores, composite
│   ├── report.py       # console / JSON / CSV / HTML output
│   ├── compare.py      # cross-device ranking from CSV history
│   ├── reference.py    # performance classes and the balance check
│   ├── gates.py        # pass/fail thresholds for CI and monitoring
│   ├── export.py       # Prometheus, JUnit XML, SQLite, Markdown
│   └── regression.py   # run-over-run regression detection
├── plugins/            # drop-in user benchmarks (auto-discovered)
├── tests/              # 417 stdlib unittest cases
└── docs/
```

The package split exists so the pieces can be unit tested in isolation:
`core`, `scoring`, `compare`, and `system.arch_family` are pure functions with
no I/O, which is where most of the test suite aims.

### Why a package *and* a `benchmark.py` shim

Being able to copy one file onto a strange machine and run it is a genuine
virtue for this kind of tool, and the package structure would normally cost
that. [`benchmark.py`](../benchmark.py) preserves it: it inserts the repo root
on `sys.path` and calls the CLI, so a bare checkout still works with
`python3 benchmark.py` — no install, no `PYTHONPATH`.

## Three-tier design

```
                 ┌────────────────────────────────────────────┐
                 │                  cli.py                     │
                 │  orchestration · state guard · payload      │
                 └───────┬────────────────────────┬───────────┘
                         │                        │
      in-process         │                        │  subprocess + JSON
      Python workloads   ▼                        ▼
        ┌────────────────────────┐   ┌──────────────────────────┐
        │  workloads.py           │   │  native_engine.c          │
        │  cpu · memory · disk    │   │  cc -O2                   │
        │  real-world · sweep     │   │  threads · pointer chase  │
        │  multiprocessing pool   │   │                           │
        └────────────────────────┘   └──────────────────────────┘
                                                  │
                                     ┌──────────────────────────┐
                                     │  accel_engine.m  (Apple) │
                                     │  Metal compute · Core ML │
                                     │  GPU GFLOPS · ANE infer  │
                                     └──────────────────────────┘
```

### The accelerator tier

Accelerators need vendor frameworks, which sits awkwardly against the
zero-dependency rule. The split adopted:

* **Inventory** — all platforms, no dependencies (`system_profiler`, `lspci` /
  `nvidia-smi` / DRM, `Win32_VideoController` / `Win32_PnPEntity`).
* **Benchmarking** — Apple only, via `accel_engine.m`. Metal shaders are
  compiled at **runtime** so only the Command Line Tools are required; the
  offline `metal` compiler ships with full Xcode.
* **Elsewhere** — inventory plus an explicit note. Compute benchmarks for CUDA,
  ROCm, oneAPI, or OpenCL would need hardware that cannot be tested here, and
  shipping untested code for someone else's GPU is worse than naming the gap.

The Neural Engine is a special case: it has **no public API for arbitrary
work**. Core ML alone decides whether a model runs on CPU, GPU, or ANE, so the
benchmark runs one model under several `MLComputeUnits` settings and treats the
speedup over CPU-only as the evidence of ANE engagement. The model itself is
generated by `coreml_model.py`, which writes the `.mlmodel` protobuf byte by
byte rather than depending on `coremltools`.

### Dependency tiers

The design rule is a **stdlib core that always works**, with optional packages
adding depth rather than being required:

| Tier | Requires | Adds |
|------|----------|------|
| **Core** | nothing | CPU, memory, disk, ML, compile, latency, sensors |
| **Native** | a C compiler | Compiler-optimised CPU, real memory latency |
| **Apple** | Command Line Tools | Metal GPU, Core ML ANE, IOHID temperatures |
| **compute** | numpy, scipy, numba | BLAS matmul, FFT, LAPACK |
| **gpu** | pyopencl, nvidia-ml-py | GPU compute on any vendor, NVIDIA telemetry |
| **crypto** | cryptography, zstandard, lz4, blake3 | AES-NI, modern codecs |
| **system** | psutil, rich, matplotlib, … | Better sensors, charts, tables |

`optional.py` is the single registry describing these; `install.py` and the
`pyproject.toml` extras both derive from it, so a tier cannot be added in one
place and forgotten in another — a test asserts they stay in sync.

Every optional benchmark obeys the same contract: **nothing is imported at
module load**, absence returns `{"available": False, "note": ...}` rather than
raising, and absent capabilities are omitted from the composite rather than
scored as zero.

### Three tiers of ML measurement

The tool measures ML performance at three levels, deliberately, because each
answers a different question:

1. **`mlbench.py` — pure Python, always runs.** Real neural-network training,
   clustering, and search with zero dependencies. Answers "what can this
   machine do out of the box?"
2. **`npu.py` — ONNX Runtime, opt-in.** Routes one model to whichever
   accelerator the machine has (Intel/AMD/Qualcomm/Apple/GPU) and reports which
   actually engaged. Answers "does the NPU work, and is it faster?"
3. **`mlframework.py` — PyTorch, opt-in.** A full training loop on CUDA / ROCm
   / MPS / CPU. Answers "what does a real ML stack achieve here?"

### The AI framework tier (the one optional dependency)

Everything above is zero-dependency. Real *training* — forward, backward,
optimizer step — genuinely cannot be done without an ML framework, so
`mlframework.py` is the single module allowed to import one, and only if the
user already installed it. It prefers PyTorch (which trains and runs on CUDA /
ROCm / MPS / CPU), falls back to ONNX Runtime (inference only), and reports
`available: False` when neither is present. This is also the only path that can
benchmark non-Apple GPUs, since it rides the framework's own backend.

### Post-processing tier

`power`, `network`, and `regression` run alongside the benchmarks rather than
as timed workloads. Power is sampled *under load* (a background CPU burn) so it
reflects active draw; regression compares the freshly-built result row against
the CSV history for the same hostname; network exercises only loopback so it
never leaves the machine.

`cli.py` is the control plane. The C engine is an optional data plane that
measures two things Python cannot express meaningfully:

- **True multi-threaded scaling** — real threads with no GIL and no process
  spawn cost, showing the hardware's parallel ceiling.
- **Memory latency** — a dependent-load pointer chase. An L1 hit is ~1 ns while
  CPython's overhead per bytecode is tens of ns, so in Python the signal is
  buried; in C the L1/L2/L3/DRAM steps are unmistakable.

## Design principles

1. **Zero required dependencies.** Standard library only. `psutil` is used when
   present but never required; a C compiler is optional.

2. **Portable by construction.** Every OS-specific probe branches for Windows,
   macOS, and Linux with a fallback. The C engine has `_WIN32` branches for
   timing, threads, and temp files.

3. **Fail soft.** One failing probe is recorded as an error entry; the run
   continues.

4. **Refuse to produce misleading numbers.** Two mechanisms:
   - the **state guard** stops a run that would be distorted by battery power
     or existing load (exit 3),
   - **validation** treats a wrong computed answer as a hardware finding
     (exit 4), not a crash.

5. **Report honestly.** Where a measurement can't be trusted — e.g. the page
   cache could not be bypassed on this platform — the output says so rather
   than presenting the number as clean.

## Execution flow

```
parse_args()
   ├─ --list-tests / --init-config / --list-devices / --compare → print, exit 0
   ├─ config.apply()   config file + PCBENCH_* (command line still wins)
   ├─ --monitor?  → telemetry session, summary, exit 0 (no benchmarking)
   ├─ --quick presets, validate --only/--skip/--assert, parse durations
   │      malformed assertions fail here, not after ten minutes of work
   │
inventory() + machine_state() + state_warnings()
   └─ warnings and not --force  → print and exit 3
container.detect()      cgroup quota / affinity / memory cap / cloud / CI
   └─ workloads sized to *effective* cores and RAM, not the host's
_autoscale()            shrink test sizes on small machines (unless disabled)
   │
for each selected test:
     runners[name]()
       ├─ ValidationError → {"validation_failed": True}   (exit 4 later)
       └─ Exception       → {"error": ...}                (run continues)
   │
native.run()            unless --no-native
standards.run()         STREAM + CoreMark-style from the native payload,
                        LINPACK via NumPy; unless --no-standards
accel.inventory()       always (cheap)
accel.run()             Apple only, unless --no-accel/--no-gpu/--no-npu
   └─ headline GPU/NPU rates folded into `results` so they score normally
counters.*()            rusage always; PMU tier only with --counters
numa.run()              if --numa / --numa-bandwidth
datascience.run()       if --datascience
iobench.run()           if --io / --io-job
power.energy_to_solution()  if --energy
storage.run()           if --disk-all / --disk-path
run_sustained()         if --sustained
soak.run()              if --soak   (last: longest phase, exit 7 on errors)
compute_scores()        normalize vs BASELINES, geometric mean
reference.assess()      performance class + balance vs the single-core anchor
gates.evaluate()        --fail-under / --assert   (exit 6 on failure)
   │
build payload  →  console report | --json-stdout
   ├─ save_json() · append_csv() · save_html() · save_spec_sheet()
   └─ save_prometheus() · save_junit() · save_sqlite() · save_markdown()
```

## Data model

One `payload` dict per run, written verbatim to JSON:

```jsonc
{
  "tool": "pcbench", "version": "3.0",
  "timestamp_utc": "2026-08-21T02:08:29Z",
  "config":    { "seconds": 3.0, "repeats": 3, "tests": [...] },
  "system":    { "arch_family": "ARM64", "cpu_model": "Apple M4", ... },
  "state":     { "on_ac_power": true, "load_average": [...], "thermal": "..." },
  "warnings":  [],
  "results":   { "cpu_int": {...}, "disk": {...}, "cache_sweep": {...} },
  "native":    { "results": [...], "latency": [...] },
  "accelerators": { "gpus": [...], "npus": [...], "benchmark_supported": true },
  "accel":     { "results": [...], "gpu": {...}, "ane": {...} },
  "sustained": { "peak_rate": ..., "droop_percent": 13.4, ... },
  "soak":      { "units_completed": 1.2e7, "errors": 0, "verdict": "STABLE ..." },
  "storage":   { "devices": [ { "mount": "/mnt/data", "disk": {...} } ] },
  "confinement": { "container": "Docker", "cpu_quota_cores": 2.0, ... },
  "reference": { "class": "workstation", "flag": "balanced", ... },
  "standards": { "stream": {...}, "linpack": {...}, "coremark_style": {...} },
  "provenance": { "mitigations": {...}, "frequency": {...}, "smt": {...} },
  "counters":  { "pmu": { "ipc": 1.9, ... }, "resources": {...} },
  "numa":      { "topology": {...}, "bandwidth": { "matrix": {...} } },
  "datascience": { "llm": {...}, "dataloader": {...}, "dataframes": {...} },
  "io":        { "jobs": [ { "name": "database", "iops": 49548, ... } ] },
  "energy":    { "joules": 412.5, "units_per_joule": 4850.0, ... },
  "gates":     [ { "name": "composite>=250", "passed": true, ... } ],
  "scores":    { "subscores": {...}, "composite": 349.0 }
}
```

The CSV is a **flattened projection** of that payload — one row per run.
Because the schema grew in v3, `append_csv` compares the existing header and
archives a mismatched file to `benchmarks.csv.v2.bak` rather than appending
misaligned rows.

## Cross-process concerns

The multi-core test uses the **`spawn`** start method explicitly so behavior is
identical on all three platforms. `spawn` re-imports the module in each worker,
so `_multicore_worker` lives at module top level and stays picklable, and
`mp.freeze_support()` runs under `__main__` for frozen Windows builds. Each
worker times *itself* — `perf_counter` epochs are per-process, so a shared
deadline would be meaningless.

The sustained mode creates its pool **once** and reuses it across sampling
windows, so spawn cost isn't charged to the measurement and the load stays
continuous — a pause between windows would let the chip cool and mask the
throttling the test exists to find.

## Native engine contract

- Python invokes `native_engine[.exe] --json --seconds N --repeats M --threads T`.
- The engine prints one JSON object with `results` (array of
  `{name, unit, rate, stdev}`), `latency` (array of `{label, bytes, ns}`), and
  `validated`.
- Non-zero exit or unparseable output becomes an `{"error": ...}` entry — never
  fatal.
- The binary rebuilds only when missing or older than the source.

See [functions.md](functions.md) for the API and [technical.md](technical.md)
for methodology.
