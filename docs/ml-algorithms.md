# Machine-Learning Algorithms

Every ML workload this tool runs, what it computes, the mathematics behind it,
and why it was chosen. Acronyms are expanded in [glossary.md](glossary.md).

The tool measures ML at three levels:

| Tier | Module | Dependencies | Answers |
|------|--------|--------------|---------|
| **1. Pure Python** | `mlbench.py` | none | What can this machine do out of the box? |
| **2. Numeric** | `numeric.py` | numpy, scipy | What can the CPU do through a real BLAS? |
| **3. Accelerator** | `npu.py`, `accel_engine.m`, `gpucompute.py` | onnxruntime / Apple frameworks / pyopencl | Does the NPU/GPU work, and is it faster? |
| **4. Framework** | `mlframework.py` | PyTorch (optional) | What does a real ML stack achieve? |

**Why more than one tier.** Tier 1 measures CPython, not silicon: on an Apple
M4 it reaches ~113 MFLOPS while the same chip does ~450 GFLOPS FP64 through
BLAS. Tier 1 numbers are still directly comparable *between* machines running
the same Python, which is what makes them useful; the higher tiers show what
the hardware can actually do.

---

# Tier 1 — Pure Python (always runs)

No NumPy, no PyTorch, no installation. These run identically on every machine
and are directly comparable.

---

## 1. Neural-network training

**File:** `pcbench/mlbench.py` · **Reported as:** steps/s, samples/s, MFLOPS

This is **genuine training**, not a simulation: weights are initialized,
gradients are computed by backpropagation, and the weights are updated. The
loss measurably falls.

### Architecture

A multi-layer perceptron (MLP) with one hidden layer:

```
input (32) ──► hidden (24, tanh) ──► output (4, softmax)
```

Batch size 24. Written out in explicit loops rather than with a matrix
library, so the arithmetic being timed is visible.

### Forward pass

**Hidden layer** — weighted sum plus bias, then a `tanh` non-linearity:

```
z⁽¹⁾ⱼ = bⱼ + Σᵢ xᵢ · Wᵢⱼ          for j = 1…24
h ⱼ  = tanh(z⁽¹⁾ⱼ)
```

`tanh` squashes any input into (−1, 1). Without a non-linearity, stacked
layers would collapse into a single linear function and the network could
learn nothing a straight line could not.

**Output layer** — another weighted sum, then softmax to turn scores into
probabilities:

```
z⁽²⁾ₖ = cₖ + Σⱼ hⱼ · Vⱼₖ          for k = 1…4

           exp(z⁽²⁾ₖ − max z⁽²⁾)
softmax:  pₖ = ─────────────────────────
             Σₘ exp(z⁽²⁾ₘ − max z⁽²⁾)
```

Subtracting `max z⁽²⁾` is a numerical-stability trick: it leaves the result
mathematically unchanged (the constant cancels) while preventing `exp()` from
overflowing to infinity on large inputs.

### Loss — cross-entropy

For a true class *y*, the loss is the negative log of the probability the
model assigned to it:

```
L = −ln(p_y)
```

A confident correct answer (p_y ≈ 1) gives L ≈ 0. A confident *wrong* answer
gives a very large loss. With 4 balanced classes, an untrained network scores
about `ln(4) ≈ 1.386` — which the test suite verifies.

### Backward pass — backpropagation

Backpropagation is the chain rule applied layer by layer, from the output
backwards.

**Output gradient.** Softmax combined with cross-entropy has an elegant
derivative — the reason this pairing is standard:

```
∂L/∂z⁽²⁾ₖ = pₖ − 1[k = y]
```

That is: the predicted probability minus 1 for the correct class, and just the
predicted probability for the others.

**Output weights and biases:**

```
∂L/∂Vⱼₖ = hⱼ · ∂L/∂z⁽²⁾ₖ
∂L/∂cₖ  = ∂L/∂z⁽²⁾ₖ
```

**Propagate into the hidden layer**, using the derivative of `tanh`:

```
d/dx tanh(x) = 1 − tanh²(x)

∂L/∂hⱼ    = Σₖ Vⱼₖ · ∂L/∂z⁽²⁾ₖ
∂L/∂z⁽¹⁾ⱼ = ∂L/∂hⱼ · (1 − hⱼ²)
```

Because `hⱼ = tanh(z⁽¹⁾ⱼ)` is already computed, the derivative costs one
multiply — no re-evaluation of `tanh`.

**Input weights:**

```
∂L/∂Wᵢⱼ = xᵢ · ∂L/∂z⁽¹⁾ⱼ
∂L/∂bⱼ  = ∂L/∂z⁽¹⁾ⱼ
```

### Weight update — stochastic gradient descent

Each weight moves a small step *against* its gradient, averaged over the batch:

```
W ← W − (η / N) · Σ_batch ∂L/∂W        with learning rate η = 0.08
```

Dividing by the batch size N keeps the step size independent of batch size.

### Initialization

Weights are drawn uniformly from `±√(1/fan_in)` — Xavier/Glorot-style scaling.
Too large and activations saturate `tanh` (where the gradient vanishes); too
small and the signal dies out through the layers.

### Validation

Before timing, the network trains 60 steps and the loss **must** fall below 90%
of its starting value. Healthy hardware always achieves this, so a failure
indicates a floating-point or memory fault rather than a slow machine.

### Cost

```
FLOPs per step ≈ 3 × batch × 2 × (32×24 + 24×4)
```

The factor 3 accounts for backward being roughly twice the forward cost;
the factor 2 counts a multiply-add as two operations.

---

## 2. K-means clustering

**File:** `pcbench/mlbench.py` · **Reported as:** distances/s

The canonical *unsupervised* algorithm: it finds structure in unlabelled data.
1,200 points in 8 dimensions, k = 6 clusters, 6 iterations.

### Objective

K-means minimizes **inertia**, also called within-cluster sum of squares:

```
J = Σᵢ ‖ xᵢ − μ_c(i) ‖²
```

where `μ_c(i)` is the centre of the cluster point `xᵢ` belongs to. Lower means
tighter, better-separated clusters.

### Lloyd's algorithm

Two alternating steps, repeated:

**1. Assignment** — each point joins its nearest centre:

```
c(i) = argmin_k ‖ xᵢ − μₖ ‖²
```

**2. Update** — each centre moves to the mean of its members:

```
μₖ = (1 / |Cₖ|) · Σ_{i ∈ Cₖ} xᵢ
```

Each step provably cannot increase J, so the algorithm always converges —
though only to a *local* minimum, which makes the starting centres critical.

Squared distance is used throughout, never `√`: comparisons give identical
results and a square root per distance would be wasted work.

### Why initialization mattered here

Choosing starting centres at random frequently draws two from the same blob,
stranding the algorithm in a poor local minimum. Measured on this dataset,
random seeding converged to **20.5** inertia per point — against a theoretical
ideal of `dims × σ² = 8 × 0.6² = 2.88`.

That made the validation check useless: it could not distinguish an unlucky
seed from a genuine hardware fault.

The fix is **farthest-point (maximin) initialization**:

```
μ₁ = x₁
μₖ = argmax_x  min_{j<k} ‖ x − μⱼ ‖²
```

Each new centre is the point furthest from all chosen so far, which lands one
per well-separated cluster and is fully deterministic. Result: **2.867 per
point** against the 2.88 ideal — so a high inertia now genuinely means
something is wrong.

### Cost

```
distance evaluations = points × k × iterations = 1200 × 6 × 6 = 43,200 per run
```

Each is 8 subtract-square-add operations. The workload is dominated by
sequential memory access, making it sensitive to cache behaviour.

---

## 3. K-nearest-neighbours search

**File:** `pcbench/mlbench.py` · **Reported as:** comparisons/s

Brute-force similarity search: 40 queries against 900 reference points in 12
dimensions, k = 5. This is the operation underneath vector databases, semantic
search, and retrieval-augmented generation.

### Method

For each query **q**, compute the squared Euclidean distance to every reference
point and keep the k smallest:

```
d(q, r) = Σ_{d=1}^{12} (q_d − r_d)²
```

k-NN has no training phase — the data *is* the model, which is why it is called
a lazy learner. Its cost is entirely at query time:

```
O(queries × references × dimensions)
```

The implementation keeps a sorted k-element list, replacing the current worst
neighbour whenever a closer one appears. For small k this beats sorting all
900 distances.

### Validation

Every reference point must be returned as its own nearest neighbour — its
distance to itself is exactly 0. A failure means the arithmetic or the memory
holding it is faulty.

---

## Shared design decisions

**Fixed seeds everywhere.** Datasets are generated from constant seeds
(`random.Random(7)`, etc.), so every machine processes byte-identical data and
results are directly comparable.

**They measure Python too.** Being pure Python, these benchmark the
interpreter as well as the silicon. That is intentional — they compare machines
running the same Python version, while tiers 2 and 3 cover compiled and
accelerated speed.

---

# Tier 2 — Accelerator workloads

## 4. Convolutional network (Apple Neural Engine)

**File:** `pcbench/coreml_model.py` · **Reported as:** inferences/s, GFLOPS

A 12-layer convolutional stack — 64 channels, 64×64 spatial, 3×3 kernels —
generated as a Core ML model.

### Convolution

Each output element is a weighted sum over a local 3×3 patch across all input
channels:

```
y[c_out, i, j] = b[c_out] + Σ_{c_in} Σ_{u=0}^{2} Σ_{v=0}^{2}
                     w[c_out, c_in, u, v] · x[c_in, i+u−1, j+v−1]
```

"Same" padding keeps the output the same height and width as the input, so
layers can stack without shrinking.

Convolution is used rather than plain matrix multiply because it is what NPUs
are physically built to accelerate, and what Core ML most reliably offloads.

### Cost

```
FLOPs per layer = C_out × C_in × K × K × H × W × 2
                = 64 × 64 × 3 × 3 × 64 × 64 × 2  ≈ 302 MFLOP
Total (12 layers) ≈ 3.62 GFLOP per inference
```

### Why the model is large

Core ML decides *for itself* whether to use the ANE, and small models are kept
on the CPU because dispatch overhead exceeds the work. Measured on an M4:

| Model | Speedup vs CPU-only | Verdict |
|-------|--------------------|---------|
| 16 channels, 32×32 | **0.92×** | never left the CPU |
| 64 channels, 64×64, 12 layers | **~6×** | ANE genuinely engaged |

Since no API reports placement, the speedup over a CPU-only run *is* the
evidence. Below 1.5× the tool reports the ANE as **not engaged** rather than
publishing a CPU result as NPU performance.

---

## 5. Matrix-multiply stack (cross-vendor NPU)

**File:** `pcbench/onnx_model.py` · **Reported as:** inferences/s, GFLOPS

Ten `MatMul` + `ReLU` layers, 1024×1024, batch 32 — an ONNX model run through
whichever execution provider the machine has (Intel OpenVINO, AMD Vitis AI,
Qualcomm QNN, DirectML, CUDA, Core ML).

### Operations

```
MatMul:  Y = X · W          Yᵢⱼ = Σₖ Xᵢₖ · Wₖⱼ
ReLU:    f(x) = max(0, x)
```

**ReLU** (Rectified Linear Unit) is the most common activation in modern
networks: trivially cheap, and its gradient is either 0 or 1, which avoids the
vanishing-gradient problem that `tanh` suffers in deep stacks.

### Why weights are scaled by 1/dim

Each layer multiplies activation magnitude by roughly `dim`. Left unscaled, a
10-layer stack at dim = 1024 amplifies by ~10³⁰ and overflows to infinity —
the model would return NaN instead of a timing. Scaling weights to `±1/dim`
keeps activations near unity:

```
E[output] ≈ dim × (1/dim) × x = x
```

### Cost

```
FLOPs = 2 × batch × dim² × layers = 2 × 32 × 1024² × 10 ≈ 0.67 GFLOP
```

---

## 6. Dense GEMM (GPU)

**File:** `accel_engine.m` · **Reported as:** TFLOPS

A 2048×2048 dense matrix multiply via MetalPerformanceShaders, in FP32 and
FP16.

```
C = A · B        FLOPs = 2 · N³ = 2 × 2048³ ≈ 17.2 GFLOP
```

GEMM is the single most meaningful "AI performance" number for a GPU because
every fully-connected and convolution layer ultimately reduces to it. A
vendor-tuned kernel is used deliberately, so the result measures the hardware
rather than the quality of hand-written shader code.

---

# Tier 3 — Framework workload

## 7. Convolutional network training (PyTorch)

**File:** `pcbench/mlframework.py` · **Reported as:** training/inference samples/s

Runs only if PyTorch is installed. Automatically selects CUDA (NVIDIA), ROCm
(AMD), MPS (Apple), or CPU — the only path in this tool that benchmarks
non-Apple GPUs.

```
Conv2d(3→32, 3×3) ─ ReLU ─ Conv2d(32→64, 3×3) ─ ReLU ─ MaxPool(2)
  ─ Conv2d(64→64, 3×3) ─ ReLU ─ Flatten ─ Linear(16384→256) ─ ReLU
  ─ Linear(256→10)
```

Trained on 32×32 inputs with cross-entropy loss and SGD — the same mathematics
as Tier 1, at realistic scale and with a real autograd engine.

**MaxPool** halves each spatial dimension by keeping only the largest value in
each 2×2 window, reducing computation and granting small translation
invariance.

### Measuring asynchronous hardware honestly

GPU frameworks queue work and return immediately. Each timed region therefore
ends with an explicit device synchronize:

```python
torch.cuda.synchronize()   # or torch.mps.synchronize()
```

Without it the timer would measure how fast work is *submitted*, not how fast
it runs — inflating results by orders of magnitude.

Training is measured as forward + backward + optimizer step; inference as
forward only under `torch.no_grad()`, which skips building the gradient graph.

---

## Tier 2 — BLAS numerics

## 8. Dense matrix multiply, FFT, and LAPACK

**File:** `pcbench/numeric.py` · **Requires:** numpy (scipy for LAPACK)

Every algorithm above ultimately reduces to linear algebra, so measuring it
directly — through the platform's tuned BLAS — gives the CPU's real numeric
ceiling.

**Matrix multiply.** A dense N x N product is `2·N³` FLOPs:

```
Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
```

Measured at N = 1536 in FP64 and FP32. The BLAS in use is named in the report
(Accelerate, OpenBLAS, or MKL).

**FFT.** An N-point complex transform costs about `5·N·log₂N` FLOPs. Unlike
matrix multiply it is bound by memory access rather than arithmetic, so it
probes a different limit.

**LAPACK decompositions** — the operations behind regression, PCA, and
simulation:

| Routine | Computes | Used for |
|---------|----------|----------|
| Cholesky | `A = L·Lᵀ` for symmetric positive-definite `A` | Solving systems ~2x faster than general factorization |
| SVD | `A = U·Σ·Vᵀ` | PCA, least squares, recommendation systems |
| Eigenvalues | `A·v = λ·v` | Stability analysis, vibration, PCA |

The test matrix is made symmetric positive-definite (`A·Aᵀ + n·I`) because
Cholesky requires it and it keeps the eigenvalue problem well-conditioned.

## Summary

| # | Algorithm | Type | Measures | Tier |
|---|-----------|------|----------|------|
| 1 | MLP training | Supervised, training | Backprop + SGD throughput | Pure Python |
| 2 | K-means | Unsupervised, clustering | Distance computation | Pure Python |
| 3 | k-NN | Supervised, lazy | Similarity search | Pure Python |
| 4 | CNN inference | Inference | Apple Neural Engine | Core ML |
| 5 | MatMul stack | Inference | Any vendor NPU/GPU | ONNX Runtime |
| 6 | Dense GEMM | Raw compute | GPU matrix throughput | Metal |
| 7 | CNN training | Supervised, training | Real framework training | PyTorch |
| 8 | GEMM / FFT / LAPACK | Raw numerics | CPU's real BLAS ceiling | numpy, scipy |

Every Tier 1 workload validates its own output — loss decreasing, clusters
converging, points being their own nearest neighbour — so a wrong answer is
reported as a **hardware fault**, not a slow result. See
[technical.md](technical.md#validation--benchmark-as-diagnostic).
