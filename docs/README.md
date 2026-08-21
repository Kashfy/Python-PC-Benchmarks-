# Documentation

Reference documentation for the PC Benchmark & Diagnostics tool. Start with the
project [README](../README.md) for a quick start, then dive in here.

| Document | What's inside |
|----------|---------------|
| [architecture.md](architecture.md) | Module layout, two-tier design, execution flow, data model, and the Python↔native contract. |
| [technical.md](technical.md) | Measurement methodology, warm-up, cache-bypass, validation, statistics, scoring, thermal droop, GPU/NPU. |
| [functions.md](functions.md) | Per-function reference for every module and the C engine. |
| [packages.md](packages.md) | Requirements, stdlib modules used, optional `psutil`, toolchain, full CLI, exit codes. |
| [safety.md](safety.md) | **Hardware safety** — what the tool never does, and how memory/disk/wear/thermal are capped. |
| [troubleshooting.md](troubleshooting.md) | Common problems and fixes — guard rails, validation failures, cache inflation, compilers, variance. |

## Quick answers

- **Why did it refuse to run?** → [troubleshooting: guard rails](troubleshooting.md#system-already-busy--running-on-battery--the-run-stops)
- **What does "VALIDATION FAILED" mean?** → [troubleshooting: validation](troubleshooting.md#validation-failed-exit-code-4)
- **Why is my disk read speed absurd?** → [technical: defeating the page cache](technical.md#disk-and-defeating-the-page-cache)
- **Why isn't multi-core 10× faster?** → [technical: scaling factor](technical.md#cpu-multi-core-and-the-scaling-factor)
- **What is the composite score?** → [technical: scoring](technical.md#scoring)
- **Why didn't the Neural Engine engage?** → [troubleshooting: ANE](troubleshooting.md#neural-engine-did-not-engage)
- **Why no GPU benchmark on Windows/Linux?** → [technical: coverage](technical.md#what-is-and-isnt-covered)
- **How do I get real AI training numbers?** → [technical: AI framework tier](technical.md#ai-training--inference-optional-framework-tier)
- **Is the power number real or estimated?** → [technical: power](technical.md#power--perf-per-watt)
- **Can this damage my SSD / RAM / laptop?** → [safety.md](safety.md) (short answer: no)
