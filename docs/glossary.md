# Glossary

Every acronym and unit this tool prints, in plain language. Grouped by where
you will meet them.

---

## Units of measurement

| Term | Stands for | What it means here |
|------|-----------|--------------------|
| **FLOPS** | **Fl**oating-point **Op**erations **Per S**econd | How many decimal-number calculations a chip does per second. The standard measure of compute for scientific and AI work. |
| **MFLOPS / GFLOPS / TFLOPS** | Mega- / Giga- / Tera-FLOPS | Millions / billions / trillions of FLOPS. 1 TFLOPS = 1,000 GFLOPS = 1,000,000 MFLOPS. |
| **IOPS** | **I**nput/**O**utput **O**perations **Per S**econd | How many separate read or write requests a drive handles per second. Small random reads determine how *responsive* a machine feels — more so than raw MB/s. |
| **MB/s** | **M**ega**b**ytes per second | Data transfer rate. Here 1 MB = 1024 × 1024 bytes. |
| **RPM** | **R**evolutions **P**er **M**inute | Fan speed. |
| **TBW** | **T**era**b**ytes **W**ritten | An SSD's rated lifetime write endurance. A 300 TBW drive is warranted to accept 300 TB of writes. |
| **ns / µs / ms** | nano- / micro- / milli-seconds | Billionths / millionths / thousandths of a second. Memory latency is measured in ns, disk and inference latency in µs and ms. |
| **W** | **W**att | Unit of power draw. |
| **°C** | Degrees **C**elsius | All temperatures in this tool are Celsius. |
| **p50 / p99** | 50th / 99th **p**ercentile | p50 is the median (half of results are faster). p99 means 99% were faster — the "worst realistic case". A good p50 with a bad p99 causes visible stutter. |
| **CV** | **C**oefficient of **V**ariation | Standard deviation ÷ mean. A normalized measure of how much repeated runs disagree. Under 2% is excellent; over 10% means the number should not be trusted. |

---

## Processors and chips

| Term | Stands for | What it means |
|------|-----------|---------------|
| **CPU** | **C**entral **P**rocessing **U**nit | The main general-purpose processor. |
| **GPU** | **G**raphics **P**rocessing **U**nit | A massively parallel processor, originally for graphics, now also for AI and general compute. |
| **NPU** | **N**eural **P**rocessing **U**nit | A chip block built specifically to run neural networks efficiently. Fast and power-efficient at AI work, useless for general computing. |
| **ANE** | **A**pple **N**eural **E**ngine | Apple's NPU, present in every Apple-silicon Mac, iPhone, and iPad. Reachable only through Core ML. |
| **VPU** | **V**ersatile **P**rocessing **U**nit | Intel's older name for its NPU; the Linux driver is still called `intel_vpu`. |
| **XDNA** | AMD's NPU architecture | The engine behind "Ryzen AI". Derived from Xilinx AI Engine technology. |
| **SoC** | **S**ystem **o**n a **C**hip | CPU, GPU, NPU, memory controller, and more on one piece of silicon. All Apple-silicon chips and most phone chips are SoCs. |
| **ISA** | **I**nstruction **S**et **A**rchitecture | The vocabulary of machine instructions a chip understands — x86-64, ARM64, RISC-V. Programs compiled for one ISA do not run on another. |
| **P-core / E-core** | **P**erformance / **E**fficiency core | Modern chips mix fast power-hungry cores with slow efficient ones. This is why 10 cores rarely give 10× the speed of one. |
| **SMT / Hyper-Threading** | **S**imultaneous **M**ulti**t**hreading | One physical core presenting itself as two logical cores. Adds throughput, but far less than a second real core. |
| **TDP** | **T**hermal **D**esign **P**ower | The heat output (in watts) a chip's cooling is designed for — a rough proxy for power draw. |

---

## CPU instruction-set features

These appear under "CPU features" and explain *why* certain benchmarks are fast.

| Term | Stands for | Effect |
|------|-----------|--------|
| **SIMD** | **S**ingle **I**nstruction, **M**ultiple **D**ata | One instruction operating on many values at once. The foundation of fast math. |
| **AES-NI / AES** | **A**dvanced **E**ncryption **S**tandard **N**ew **I**nstructions | Hardware encryption. Makes disk encryption and HTTPS dramatically faster. |
| **SHA-NI / SHA-256** | **S**ecure **H**ash **A**lgorithm instructions | Hardware hashing. This is why the SHA-256 benchmark can hit thousands of MB/s. |
| **AVX / AVX2 / AVX-512** | **A**dvanced **V**ector **Ex**tensions | Intel/AMD SIMD, processing 256 or 512 bits at once. |
| **NEON** | (not an acronym) | ARM's 128-bit SIMD engine. |
| **SVE / SVE2** | **S**calable **V**ector **E**xtension | ARM's newer variable-width SIMD. |
| **AMX** | **A**pple **M**atrix e**x**tension (also Intel Advanced Matrix Extensions) | A dedicated matrix-multiply unit. Often beats the GPU for small matrix work. |
| **FMA** | **F**used **M**ultiply-**A**dd | Computes `a × b + c` in one instruction — the core operation of matrix multiply and neural networks. |
| **BF16** | **B**rain **F**loat **16** | A 16-bit number format designed for AI, trading precision for speed. |
| **Int8 matmul** | 8-bit integer matrix multiply | Quantized AI inference — much faster and lower-power than float. |

---

## Memory and storage

| Term | Stands for | What it means |
|------|-----------|---------------|
| **RAM** | **R**andom **A**ccess **M**emory | Main working memory. Fast, volatile, erased at power-off. |
| **DRAM** | **D**ynamic **RAM** | The technology main memory is built from. Slow relative to cache (~70–100 ns). |
| **L1 / L2 / L3 cache** | Level 1/2/3 | Small, very fast memories inside the CPU. L1 is tiny and ~1 ns; L3 is larger and slower. The cache sweep and latency curve reveal these tiers. |
| **Working set** | — | How much memory a program actively touches. When it outgrows a cache level, performance drops sharply. |
| **Page cache** | — | The OS keeping recently-read file data in RAM. Makes disk reads *look* impossibly fast unless deliberately bypassed. |
| **SSD** | **S**olid **S**tate **D**rive | Flash-based storage, no moving parts. |
| **NVMe** | **N**on-**V**olatile **M**emory **e**xpress | The fast protocol modern SSDs use over PCIe. |
| **SLC cache** | **S**ingle-**L**evel **C**ell cache | A fast write buffer on consumer SSDs. Once exhausted by a large write, speed drops — which is why a 256 MB disk test scores lower than a 64 MB one. |
| **fsync** | **f**ile **sync**hronize | A system call forcing buffered writes out to the physical device. Without it, "write speed" only measures writing to RAM. |
| **F_NOCACHE / O_DIRECT** | — | macOS and Linux flags requesting that file I/O bypass the page cache, so reads measure the drive rather than memory. |
| **PCI / PCIe** | **P**eripheral **C**omponent **I**nterconnect (**e**xpress) | The bus connecting GPUs, SSDs, and NPUs. Devices are identified by PCI vendor and device IDs. |

---

## Machine learning

See [ml-algorithms.md](ml-algorithms.md) for how each algorithm actually works.

| Term | Stands for | What it means |
|------|-----------|---------------|
| **ML** | **M**achine **L**earning | Programs that improve at a task by learning from data rather than explicit rules. |
| **AI** | **A**rtificial **I**ntelligence | The broader field; used loosely here for neural-network workloads. |
| **NN** | **N**eural **N**etwork | A model of layered interconnected "neurons" whose connection strengths are learned. |
| **MLP** | **M**ulti-**L**ayer **P**erceptron | The classic fully-connected neural network — the kind this tool trains. |
| **CNN** | **C**onvolutional **N**eural **N**etwork | A network using convolution filters; standard for images. Used in the PyTorch and Core ML benchmarks. |
| **Training** | — | Adjusting a model's weights so it makes fewer mistakes. Computationally ~3× the cost of inference. |
| **Inference** | — | Running an already-trained model to get an answer. What NPUs are optimized for. |
| **Backprop** | **Back**ward **prop**agation of errors | The algorithm computing how much each weight contributed to the error, enabling learning. |
| **SGD** | **S**tochastic **G**radient **D**escent | The rule that nudges each weight opposite to its gradient. |
| **Gradient** | — | The slope of the error with respect to a weight: which way, and how strongly, to adjust it. |
| **Loss** | — | A number measuring how wrong the model is. Training minimizes it. |
| **Epoch / step / batch** | — | A *batch* is a group of examples processed together; a *step* is one weight update; an *epoch* is one pass over all data. |
| **k-NN** | **k**-**N**earest **N**eighbours | Classify or retrieve by finding the k most similar known examples. |
| **k-means** | — | Group data into k clusters by repeatedly assigning points to the nearest cluster centre. |
| **Inertia / WCSS** | **W**ithin-**C**luster **S**um of **S**quares | Total squared distance from each point to its cluster centre. Lower means tighter clusters. |
| **GEMM** | **GE**neral **M**atrix **M**ultiply | Dense matrix multiplication — the operation that dominates nearly all neural-network compute. |
| **Quantization** | — | Using lower-precision numbers (e.g. 8-bit instead of 32-bit) for speed and efficiency. |

---

## Numerical computing

| Term | Stands for | What it means |
|------|-----------|---------------|
| **BLAS** | **B**asic **L**inear **A**lgebra **S**ubprograms | The standard low-level matrix/vector library. Every platform ships a tuned one — Accelerate on macOS, OpenBLAS or MKL elsewhere — and nearly all numeric performance rests on it. |
| **LAPACK** | **L**inear **A**lgebra **PACK**age | Higher-level routines built on BLAS: solving systems, decompositions, eigenvalues. |
| **SVD** | **S**ingular **V**alue **D**ecomposition | Factors a matrix into rotation–scale–rotation. Underlies PCA, recommendation systems, and least-squares fitting. |
| **Cholesky** | (named after André-Louis Cholesky) | Factors a symmetric positive-definite matrix into `L·Lᵀ`. About twice as fast as general factorization, so it is preferred wherever it applies. |
| **Eigenvalues** | — | The scaling factors along a matrix's characteristic directions. Central to stability analysis, vibration, and PCA. |
| **FFT** | **F**ast **F**ourier **T**ransform | Converts a signal between time and frequency. Costs ~5·N·log₂N operations and is memory-bound rather than arithmetic-bound, so it stresses a different limit from matrix multiply. |
| **SPD** | **S**ymmetric **P**ositive-**D**efinite | A matrix property required by Cholesky and one that makes eigenvalue problems well-conditioned. |
| **FP32 / FP64** | 32- / 64-bit **f**loating **p**oint | Single and double precision. FP64 is more accurate and usually slower; consumer GPUs are often far weaker at it. |
| **JIT** | **J**ust-**I**n-**T**ime compilation | Compiling code at runtime rather than ahead of time. What `numba` does to Python. |

---

## Cryptography and compression

| Term | Stands for | What it means |
|------|-----------|---------------|
| **AES** | **A**dvanced **E**ncryption **S**tandard | The dominant symmetric cipher. Modern CPUs implement it in hardware. |
| **GCM** | **G**alois/**C**ounter **M**ode | An AES mode giving both encryption and authentication in one pass. What TLS and disk encryption typically use. |
| **Zstandard (zstd)** | — | A modern compression codec from 2016 offering far better speed *and* ratio than zlib. |
| **LZ4** | **L**empel-**Z**iv 4 | A compression codec optimized for raw speed over ratio. |
| **BLAKE3** | — | A modern cryptographic hash, heavily parallel and much faster than SHA-256 in software. |
| **Compression ratio** | — | Original size ÷ compressed size. 7.29× means the output is about one-seventh the input. |
| **Round trip** | — | Compress-then-decompress, or encrypt-then-decrypt. Used here to validate that data survives intact. |

---

## GPU compute and packaging

| Term | Stands for | What it means |
|------|-----------|---------------|
| **OpenCL** | **Open** **C**omputing **L**anguage | A vendor-neutral GPU/accelerator compute API implemented by NVIDIA, AMD, Intel, and Apple. How this tool benchmarks GPUs portably. |
| **Kernel** (GPU) | — | A small program run by thousands of GPU threads at once. Unrelated to an OS kernel. |
| **CU** | **C**ompute **U**nit | An OpenCL device's independent processing block — roughly a GPU "core cluster". |
| **Work item / global size** | — | One GPU thread, and the total number launched. |
| **QD** | **Q**ueue **D**epth | How many I/O requests are outstanding at once. QD1 measures latency; higher depths reveal a drive's real ceiling. |
| **venv** | **v**irtual **env**ironment | An isolated Python installation. The installer uses one so packages never touch your system Python. |
| **Wheel** | — | A prebuilt Python package. When none exists for your platform, pip must compile from source, which often fails without system libraries. |
| **pip extras** | — | Optional dependency groups, e.g. `pip install pcbench[compute]`. |

---

## AI software stacks

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Core ML** | — | Apple's ML framework. The **only** public way to reach the Neural Engine. |
| **Metal** | — | Apple's low-level GPU programming API. |
| **MPS** | **M**etal **P**erformance **S**haders | Apple's library of optimized GPU routines, including matrix multiply. (In PyTorch, "mps" also names the Apple GPU backend.) |
| **ONNX** | **O**pen **N**eural **N**etwork e**x**change | A vendor-neutral file format for ML models. |
| **ONNX Runtime** | — | The engine that runs ONNX models, with plug-in backends for many accelerators. |
| **EP** | **E**xecution **P**rovider | An ONNX Runtime backend targeting specific hardware (CPU, CUDA, OpenVINO, DirectML…). |
| **OpenVINO** | **Open** **V**isual **I**nference and **N**eural network **O**ptimization | Intel's toolkit; the route to Intel's NPU. |
| **Vitis AI** | — | AMD's stack for its XDNA NPU. |
| **QNN** | **Q**ualcomm **N**eural **N**etwork SDK | The route to Qualcomm's Hexagon NPU. |
| **DirectML** | **Direct** **M**achine **L**earning | Microsoft's hardware-agnostic ML API on any DirectX 12 device. |
| **CUDA** | **C**ompute **U**nified **D**evice **A**rchitecture | NVIDIA's GPU computing platform. |
| **ROCm** | **R**adeon **O**pen **C**ompute | AMD's equivalent of CUDA. |
| **TensorRT** | — | NVIDIA's optimizing inference engine. |

---

## Software and system

| Term | Stands for | What it means |
|------|-----------|---------------|
| **API** | **A**pplication **P**rogramming **I**nterface | The published way for programs to ask a system to do something. |
| **SDK** | **S**oftware **D**evelopment **K**it | The tools and libraries for building on a platform. |
| **CLI** | **C**ommand **L**ine **I**nterface | Text commands and flags — how you drive this tool. |
| **TUI** | **T**ext **U**ser **I**nterface | A full-screen interface drawn in the terminal and driven by keys rather than typed commands — what `--menu` gives you. |
| **raw mode** | — | A terminal setting where each keypress reaches the program immediately, instead of a whole line at Enter. It is what makes arrow keys work; the tool always restores the previous setting on exit. |
| **alternate screen buffer** | — | A second, scrollback-free screen a program can draw on and then drop, leaving what was there before untouched. `--menu` uses it, so only the command it built stays in your history. |
| **GIL** | **G**lobal **I**nterpreter **L**ock | A CPython lock allowing only one thread to run Python at a time. Python 3.13+ can be built without it. |
| **spawn / fork** | — | Two ways to start a worker process. This tool uses `spawn` everywhere for identical behaviour across platforms. |
| **JSON** | **J**ava**S**cript **O**bject **N**otation | A structured text format; used for the full result payload. |
| **CSV** | **C**omma-**S**eparated **V**alues | A spreadsheet-friendly table; one row per run. |
| **HTML** | **H**yper**T**ext **M**arkup **L**anguage | Web-page format; used for the shareable report. |
| **protobuf** | **Proto**col **Buf**fers | Google's binary data format. Core ML and ONNX model files are protobufs — this tool writes them directly to avoid heavy dependencies. |
| **POSIX** | **P**ortable **O**perating **S**ystem **I**nterface | The Unix standard macOS and Linux both follow. |
| **ICMP** | **I**nternet **C**ontrol **M**essage **P**rotocol | The protocol `ping` uses. Many networks block it, so a failed ping is usually a policy, not a fault. |
| **jitter** | — | The mean difference between *consecutive* round trips (RFC 3550). It, not average latency, is what decides whether calls and remote shells feel steady. |
| **RTT** | **R**ound-**T**rip **T**ime | How long a packet takes to reach a host and come back — what a ping reports. |
| **WMI** | **W**indows **M**anagement **I**nstrumentation | Windows' system-information service. |
| **RAPL** | **R**unning **A**verage **P**ower **L**imit | Intel/AMD on-chip energy counters — the source of real power readings on Linux. |
| **SMC** | **S**ystem **M**anagement **C**ontroller | The Mac subsystem managing fans, power, and temperature sensors. |
| **IOHID / IOKit** | **I/O** **H**uman **I**nterface **D**evice | The macOS driver framework; its thermal usage page is how this tool reads temperatures without root. |
| **DMI / SMBIOS** | **D**esktop **M**anagement **I**nterface | A firmware table describing the machine; used to detect virtual machines. |
| **VM** | **V**irtual **M**achine | An emulated computer. Benchmarks inside one are not comparable to bare metal. |
| **SMART** | **S**elf-**M**onitoring, **A**nalysis and **R**eporting **T**echnology | Drive self-diagnostics. This tool never writes SMART data. |

---

## Statistics

| Term | Meaning |
|------|---------|
| **Median** | The middle value. Used for headline results because one bad run cannot drag it, unlike an average. |
| **Mean** | The ordinary average. |
| **Standard deviation** | How spread out repeated measurements are. |
| **Geometric mean** | The *n*-th root of *n* values multiplied together. Used for the composite score so no single category dominates. |
| **Baseline** | The fixed reference each score is measured against (baseline = 100). |
| **Regression** | A metric getting measurably *worse* than this machine's own history. |
| **Warm-up** | Untimed work run first, so cold caches and low clock speeds do not distort the measurement. |
