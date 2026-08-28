# Documentation

Reference documentation for the PC Benchmark & Diagnostics tool. Start with the
project [README](../README.md) for a quick start, then dive in here.

| Document | What's inside |
|----------|---------------|
| [architecture.md](architecture.md) | Module layout, two-tier design, execution flow, data model, and the Python↔native contract. |
| [technical.md](technical.md) | Measurement methodology, warm-up, cache-bypass, validation, statistics, scoring, thermal droop, GPU/NPU. |
| [functions.md](functions.md) | Per-function reference for every module and the C engine. |
| [packages.md](packages.md) | Requirements, stdlib modules used, optional `psutil`, toolchain, full CLI, exit codes. |
| [glossary.md](glossary.md) | **Every acronym explained** — units, chips, instruction sets, ML terms, software stacks. |
| [ml-algorithms.md](ml-algorithms.md) | **Every ML algorithm** — architecture, equations, and why each was chosen. |
| [safety.md](safety.md) | **Hardware safety** — what the tool never does, and how memory/disk/wear/thermal are capped. |
| [troubleshooting.md](troubleshooting.md) | Common problems and fixes — guard rails, validation failures, cache inflation, compilers, variance. |

## Quick answers

- **How do I pick what to run without reading every flag?** → [README: choosing what to run](../README.md#choosing-what-to-run) (`pcbench --menu`)
- **The menu ignores my arrow keys** → [troubleshooting: --menu](troubleshooting.md#--menu-does-not-respond-to-the-arrow-keys)
- **Why did it refuse to run?** → [troubleshooting: guard rails](troubleshooting.md#system-already-busy--running-on-battery--the-run-stops)
- **What does "VALIDATION FAILED" mean?** → [troubleshooting: validation](troubleshooting.md#validation-failed-exit-code-4)
- **Why is my disk read speed absurd?** → [technical: defeating the page cache](technical.md#disk-and-defeating-the-page-cache)
- **Why isn't multi-core 10× faster?** → [technical: scaling factor](technical.md#cpu-multi-core-and-the-scaling-factor)
- **What is the composite score?** → [technical: scoring](technical.md#scoring)
- **Why didn't the Neural Engine engage?** → [troubleshooting: ANE](troubleshooting.md#neural-engine-did-not-engage)
- **Why no GPU benchmark on Windows/Linux?** → [technical: coverage](technical.md#what-is-and-isnt-covered)
- **How do I get real AI training numbers?** → [technical: AI framework tier](technical.md#ai-training--inference-optional-framework-tier)
- **Is the power number real or estimated?** → [technical: power](technical.md#power--perf-per-watt)
- **A package failed to install — now what?** → [troubleshooting: install](troubleshooting.md#a-package-failed-to-install)
- **What do the optional packages add?** → [packages: tiers](packages.md#optional-package-tiers)
- **What does TFLOPS / IOPS / NPU / GEMM mean?** → [glossary.md](glossary.md)
- **How does the neural-network training actually work?** → [ml-algorithms.md](ml-algorithms.md#1-neural-network-training)
- **Can this damage my SSD / RAM / laptop?** → [safety.md](safety.md) (short answer: no)
- **Does it support Intel / AMD NPUs?** → [technical: cross-vendor NPU](technical.md#cross-vendor-npu-via-onnx-runtime)
- **What ML workloads run without a framework?** → [technical: ML workloads](technical.md#machine-learning-workloads-pure-python)
