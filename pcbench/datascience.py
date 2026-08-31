"""Workloads that decide whether a machine is good for data and ML work.

The existing ML tier measures training steps for a small MLP and some classical
algorithms. That was the right thing to measure in 2015. It is not what anyone
sizes hardware against now, and it misses the four numbers that actually
determine whether a machine is usable for this work:

* **LLM decode throughput.** Generating one token requires reading *every*
  model weight. Decode is therefore bound by memory bandwidth, not by compute,
  and a GPU with huge FLOPS but modest bandwidth generates text slowly. This is
  the single most misunderstood number in current hardware selection.
* **LLM prefill throughput.** Processing the prompt is a large matrix multiply
  and *is* compute-bound. Prefill and decode scale with different hardware
  properties, so reporting one "inference" figure hides the thing being asked
  about.
* **Input-pipeline throughput.** Most training runs are not limited by the
  accelerator at all — they are limited by the CPU decoding and augmenting
  samples fast enough to keep it fed. A machine with a fast GPU and a slow
  pipeline trains at the speed of the pipeline, and nobody measures it.
* **Dataframe throughput.** Before any model is trained, the data has to be
  filtered, grouped, and joined. For most practitioners this is where the day
  actually goes.

**On backends.** Everything here runs on NumPy, which is a far lower bar than
PyTorch and makes the numbers available on any machine with a working BLAS.
PyTorch is used when present, because it is the only way to reach a GPU. The
transformer is built from random weights rather than a downloaded checkpoint:
throughput depends on the *shape* of the computation, not the values in it, and
requiring a multi-gigabyte download would put this out of reach of exactly the
constrained machines that most need measuring.
"""

from __future__ import annotations

import math
import os
import time

from .core import clock

MB = 1024 * 1024
GB = 1024 * MB


def available() -> dict:
    """Which backends and libraries are usable here."""
    status: dict = {"numpy": False, "torch": False, "torch_device": None,
                    "dataframes": []}
    try:
        import numpy  # noqa: F401
        status["numpy"] = True
    except Exception:
        pass
    try:
        import torch
        status["torch"] = True
        status["torch_version"] = torch.__version__
        status["torch_device"] = best_torch_device()
    except Exception:
        pass
    for name in ("pandas", "polars", "duckdb"):
        try:
            __import__(name)
            status["dataframes"].append(name)
        except Exception:
            pass
    return status


def best_torch_device() -> str:
    """Fastest device PyTorch can reach here."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Transformer model sizing
# --------------------------------------------------------------------------- #
class ModelSpec:
    """Shape of the synthetic decoder-only transformer being measured."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 vocab: int = 32_000):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = 4 * d_model
        self.vocab = vocab

    @property
    def params(self) -> int:
        """Parameter count.

        Per layer: QKV and output projections (4 d^2) plus the MLP's two
        matrices (8 d^2). Plus one tied embedding/output matrix.
        """
        per_layer = 12 * self.d_model * self.d_model
        return self.n_layers * per_layer + self.vocab * self.d_model

    def bytes(self, dtype_size: int = 4) -> int:
        return self.params * dtype_size

    def describe(self) -> str:
        return (f"{self.n_layers} layers, d_model={self.d_model}, "
                f"{self.n_heads} heads, {self.params / 1e6:.0f}M parameters")


def choose_model(memory_bytes: int = 0, dtype_size: int = 4) -> ModelSpec:
    """Pick a model that fits comfortably in the memory available.

    Sized to occupy a small fraction of memory rather than to be impressive:
    the goal is a stable measurement of the machine's prefill and decode
    characteristics, and a model that triggers swapping measures the swap file.
    """
    budget = memory_bytes or (4 * GB)
    # Aim for the weights to use at most ~6% of memory, leaving ample room for
    # activations, the KV cache, and everything else running on the machine.
    target_bytes = max(32 * MB, int(budget * 0.06))

    for d_model, n_layers, n_heads in ((2048, 12, 16), (1536, 10, 12),
                                       (1024, 8, 16), (768, 6, 12),
                                       (512, 4, 8), (256, 2, 4)):
        spec = ModelSpec(d_model, n_layers, n_heads)
        if spec.bytes(dtype_size) <= target_bytes:
            return spec
    return ModelSpec(256, 2, 4)


# --------------------------------------------------------------------------- #
# NumPy transformer
# --------------------------------------------------------------------------- #
def _np_weights(spec: "ModelSpec", np):
    """Random weights, scaled so activations neither vanish nor explode.

    Correct initialisation matters even though the outputs are meaningless:
    activations that overflow to inf or collapse to zero can change floating
    point performance markedly on some hardware (denormal handling in
    particular), which would make the benchmark measure numerical accident.
    """
    rng = np.random.default_rng(1234)
    scale = 1.0 / math.sqrt(spec.d_model)
    layers = []
    for _ in range(spec.n_layers):
        layers.append({
            "qkv": (rng.standard_normal(
                (spec.d_model, 3 * spec.d_model), dtype=np.float32) * scale),
            "proj": (rng.standard_normal(
                (spec.d_model, spec.d_model), dtype=np.float32) * scale),
            "fc1": (rng.standard_normal(
                (spec.d_model, spec.d_ff), dtype=np.float32) * scale),
            "fc2": (rng.standard_normal(
                (spec.d_ff, spec.d_model), dtype=np.float32)
                * (1.0 / math.sqrt(spec.d_ff))),
        })
    return layers


def _np_layer_norm(x, np):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5)


def _np_softmax(x, np):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _np_forward(x, layers, spec: "ModelSpec", np, cache=None, offset: int = 0):
    """One forward pass. With ``cache`` supplied, runs in incremental mode."""
    seq = x.shape[0]
    for i, w in enumerate(layers):
        h = _np_layer_norm(x, np)
        qkv = h @ w["qkv"]
        q, k, v = np.split(qkv, 3, axis=-1)

        if cache is not None:
            # Decode: append this step's K/V and attend over everything so far.
            # This is what makes decode memory-bound — the cache grows and every
            # weight matrix is re-read for a single token.
            ck, cv = cache[i]
            ck[offset:offset + seq] = k
            cv[offset:offset + seq] = v
            k = ck[:offset + seq]
            v = cv[:offset + seq]

        q = q.reshape(seq, spec.n_heads, spec.d_head).transpose(1, 0, 2)
        k = k.reshape(-1, spec.n_heads, spec.d_head).transpose(1, 2, 0)
        v = v.reshape(-1, spec.n_heads, spec.d_head).transpose(1, 0, 2)

        scores = (q @ k) / math.sqrt(spec.d_head)
        if cache is None and seq > 1:
            # Causal mask, so prefill does the same work a real model does.
            mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), 1)
            scores = scores + mask
        attn = _np_softmax(scores, np) @ v
        attn = attn.transpose(1, 0, 2).reshape(seq, spec.d_model)
        x = x + attn @ w["proj"]

        h = _np_layer_norm(x, np)
        # GELU's tanh approximation, as used by GPT-family models.
        u = h @ w["fc1"]
        gelu = 0.5 * u * (1.0 + np.tanh(0.7978845608 * (u + 0.044715 * u ** 3)))
        x = x + gelu @ w["fc2"]
    return x


def llm_numpy(spec: "ModelSpec", prefill_tokens: int = 256,
              decode_tokens: int = 32) -> dict:
    """Prefill and decode throughput on the CPU through NumPy/BLAS."""
    import numpy as np

    layers = _np_weights(spec, np)
    rng = np.random.default_rng(7)

    # --- Prefill: the whole prompt at once, compute-bound ------------------
    x = rng.standard_normal((prefill_tokens, spec.d_model),
                            dtype=np.float32) * 0.1
    _np_forward(x[:8], layers, spec, np)          # warm BLAS and caches
    start = clock()
    out = _np_forward(x, layers, spec, np)
    prefill_s = clock() - start
    if not np.isfinite(out).all():
        return {"skipped": True,
                "reason": "prefill produced non-finite activations"}

    # --- Decode: one token at a time against a growing KV cache ------------
    max_len = prefill_tokens + decode_tokens
    cache = [(np.zeros((max_len, spec.d_model), dtype=np.float32),
              np.zeros((max_len, spec.d_model), dtype=np.float32))
             for _ in range(spec.n_layers)]
    step = rng.standard_normal((1, spec.d_model), dtype=np.float32) * 0.1
    _np_forward(step, layers, spec, np, cache, 0)   # warm
    start = clock()
    for i in range(decode_tokens):
        _np_forward(step, layers, spec, np, cache, i + 1)
    decode_s = clock() - start

    return _summarise(spec, prefill_tokens, prefill_s,
                      decode_tokens, decode_s, backend="numpy (BLAS)",
                      device="cpu", dtype_size=4)


# --------------------------------------------------------------------------- #
# PyTorch transformer
# --------------------------------------------------------------------------- #
def llm_torch(spec: "ModelSpec", prefill_tokens: int = 256,
              decode_tokens: int = 32, device: str | None = None) -> dict:
    """Same measurement through PyTorch, which can reach a GPU."""
    try:
        import torch
    except Exception:
        return {"skipped": True, "reason": "PyTorch is not installed"}

    device = device or best_torch_device()
    dev = torch.device(device)
    # fp16 on an accelerator is what inference actually uses; fp32 on CPU
    # because fp16 CPU kernels are usually emulated and would mislead.
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    dtype_size = 2 if dtype == torch.float16 else 4

    try:
        torch.manual_seed(1234)
        scale = 1.0 / math.sqrt(spec.d_model)
        layers = []
        for _ in range(spec.n_layers):
            layers.append({
                "qkv": torch.randn(spec.d_model, 3 * spec.d_model,
                                   device=dev, dtype=dtype) * scale,
                "proj": torch.randn(spec.d_model, spec.d_model,
                                    device=dev, dtype=dtype) * scale,
                "fc1": torch.randn(spec.d_model, spec.d_ff,
                                   device=dev, dtype=dtype) * scale,
                "fc2": torch.randn(spec.d_ff, spec.d_model, device=dev,
                                   dtype=dtype) / math.sqrt(spec.d_ff),
            })
    except (RuntimeError, MemoryError) as e:
        return {"skipped": True,
                "reason": f"could not allocate the model on {device}: {e}"}

    def sync() -> None:
        # Accelerator work is asynchronous; without a barrier the timer would
        # measure how fast Python can enqueue kernels.
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def forward(x, cache=None, offset=0):
        seq = x.shape[0]
        for i, w in enumerate(layers):
            h = torch.nn.functional.layer_norm(x, (spec.d_model,))
            qkv = h @ w["qkv"]
            q, k, v = qkv.chunk(3, dim=-1)
            if cache is not None:
                ck, cv = cache[i]
                ck[offset:offset + seq] = k
                cv[offset:offset + seq] = v
                k, v = ck[:offset + seq], cv[:offset + seq]
            q = q.view(seq, spec.n_heads, spec.d_head).transpose(0, 1)
            k = k.view(-1, spec.n_heads, spec.d_head).transpose(0, 1)
            v = v.view(-1, spec.n_heads, spec.d_head).transpose(0, 1)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=(cache is None and seq > 1))
            attn = attn.transpose(0, 1).reshape(seq, spec.d_model)
            x = x + attn @ w["proj"]
            h = torch.nn.functional.layer_norm(x, (spec.d_model,))
            x = x + torch.nn.functional.gelu(h @ w["fc1"]) @ w["fc2"]
        return x

    try:
        with torch.inference_mode():
            x = torch.randn(prefill_tokens, spec.d_model,
                            device=dev, dtype=dtype) * 0.1
            forward(x[:8])
            sync()
            start = clock()
            out = forward(x)
            sync()
            prefill_s = clock() - start
            if not torch.isfinite(out).all():
                return {"skipped": True,
                        "reason": "prefill produced non-finite activations"}

            max_len = prefill_tokens + decode_tokens
            cache = [(torch.zeros(max_len, spec.d_model, device=dev,
                                  dtype=dtype),
                      torch.zeros(max_len, spec.d_model, device=dev,
                                  dtype=dtype))
                     for _ in range(spec.n_layers)]
            step = torch.randn(1, spec.d_model, device=dev, dtype=dtype) * 0.1
            forward(step, cache, 0)
            sync()
            start = clock()
            for i in range(decode_tokens):
                forward(step, cache, i + 1)
            sync()
            decode_s = clock() - start
    except (RuntimeError, MemoryError) as e:
        return {"skipped": True, "reason": f"PyTorch run failed: {e}"}

    return _summarise(spec, prefill_tokens, prefill_s, decode_tokens,
                      decode_s, backend=f"pytorch {torch.__version__}",
                      device=device, dtype_size=dtype_size,
                      dtype=str(dtype).replace("torch.", ""))


def _summarise(spec: "ModelSpec", prefill_tokens: int, prefill_s: float,
               decode_tokens: int, decode_s: float, backend: str,
               device: str, dtype_size: int, dtype: str = "float32") -> dict:
    """Derive the interpretable figures from two timings."""
    prefill_rate = prefill_tokens / prefill_s if prefill_s > 0 else 0.0
    decode_rate = decode_tokens / decode_s if decode_s > 0 else 0.0

    # A forward pass costs roughly 2 FLOPs per parameter per token.
    prefill_tflops = (2.0 * spec.params * prefill_tokens / prefill_s / 1e12
                      if prefill_s > 0 else 0.0)
    # Decode re-reads every weight for each token, so tokens/s times the model
    # size is the memory bandwidth the decode phase actually achieved.
    weight_bytes = spec.bytes(dtype_size)
    decode_bandwidth = (decode_rate * weight_bytes / 1e9)

    return {
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "model": spec.describe(),
        "parameters": spec.params,
        "weights_mb": round(weight_bytes / MB, 1),
        "prefill_tokens_per_s": round(prefill_rate, 1),
        "prefill_tflops": round(prefill_tflops, 3),
        "decode_tokens_per_s": round(decode_rate, 1),
        "decode_bandwidth_gb_s": round(decode_bandwidth, 1),
        "prefill_decode_ratio": (round(prefill_rate / decode_rate, 1)
                                 if decode_rate else None),
        "note": ("prefill is compute-bound (large GEMMs); decode re-reads every "
                 "weight per token and is bound by memory bandwidth"),
    }


def llm(memory_bytes: int = 0, prefill_tokens: int = 256,
        decode_tokens: int = 32, use_torch: bool = True) -> dict:
    """Run the LLM benchmark on the best available backend, and on NumPy."""
    status = available()
    if not status["numpy"] and not status["torch"]:
        return {"skipped": True,
                "reason": "needs NumPy or PyTorch",
                "hint": "pip install numpy (or run install.py)"}

    # One model spec for every backend. Sizing each to its own dtype would
    # produce two tokens/s figures for two different models side by side,
    # which reads as a backend comparison and is not one. Sized against fp32
    # so the CPU path, which cannot use fp16 meaningfully, still fits.
    spec = choose_model(memory_bytes, dtype_size=4)

    out: dict = {"model": spec.describe(), "parameters": spec.params}
    if status["torch"] and use_torch:
        out["accelerated"] = llm_torch(spec, prefill_tokens, decode_tokens)
    if status["numpy"]:
        out["cpu"] = llm_numpy(spec, prefill_tokens, min(decode_tokens, 16))

    best = out.get("accelerated") or out.get("cpu") or {}
    if not best.get("skipped"):
        out["decode_tokens_per_s"] = best.get("decode_tokens_per_s")
        out["prefill_tokens_per_s"] = best.get("prefill_tokens_per_s")
    return out


# --------------------------------------------------------------------------- #
# Input pipeline
# --------------------------------------------------------------------------- #
def dataloader(seconds: float = 2.0, batch_size: int = 32,
               workers: int = 0) -> dict:
    """Samples per second through a typical training input pipeline.

    Almost every "my GPU is underutilised" report is this: the accelerator
    finishes each batch before the CPU can produce the next one, so the
    expensive part of the machine idles. The pipeline modelled here is the
    standard vision one — decode, random crop, horizontal flip, normalise to
    float, collate — which is CPU-bound and parallel, exactly like the real
    thing.
    """
    try:
        import numpy as np
    except Exception:
        return {"skipped": True, "reason": "NumPy is not installed"}

    height = width = 224
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def one_sample(seed: int):
        r = np.random.default_rng(seed)
        top = int(r.integers(0, 256 - height + 1))
        left = int(r.integers(0, 256 - width + 1))
        crop = raw[top:top + height, left:left + width]
        if r.random() < 0.5:
            crop = crop[:, ::-1]
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)

    def one_batch(index: int):
        return np.stack([one_sample(index * batch_size + i)
                         for i in range(batch_size)])

    one_batch(0)                                   # warm
    start = clock()
    batches = 0
    while clock() - start < seconds:
        one_batch(batches + 1)
        batches += 1
    elapsed = clock() - start
    single = batches * batch_size / elapsed

    result = {
        "unit": "samples/s",
        "rate": single,
        "single_worker_samples_per_s": round(single, 1),
        "batch_size": batch_size,
        "resolution": f"{height}x{width}",
        "note": "decode, random crop, flip, normalise, collate — the standard "
                "vision input pipeline, which is CPU-bound",
    }

    # Estimate the pipeline's ceiling across cores, which is what decides
    # whether an accelerator can be kept fed.
    cores = workers or (os.cpu_count() or 1)
    result["estimated_all_cores_samples_per_s"] = round(single * cores, 1)
    result["workers_assumed"] = cores
    result["caveat"] = ("the all-core figure assumes perfect scaling; real "
                        "dataloaders lose some of it to collation and IPC")
    return result


# --------------------------------------------------------------------------- #
# Batch-size scaling
# --------------------------------------------------------------------------- #
def batch_scaling(memory_bytes: int = 0,
                  sizes: list[int] | None = None) -> dict:
    """Throughput against batch size, to locate the point of diminishing return.

    Small batches leave the accelerator's matrix units idle between launches;
    large ones stop helping once the units saturate and start costing memory.
    The knee is the batch size worth using, and it is hardware-specific rather
    than something to copy from a blog post.
    """
    sizes = sizes or [1, 2, 4, 8, 16, 32, 64, 128]
    status = available()
    if not (status["numpy"] or status["torch"]):
        return {"skipped": True, "reason": "needs NumPy or PyTorch"}

    spec = choose_model(memory_bytes, dtype_size=2 if status["torch"] else 4)
    points: list[dict] = []

    if status["torch"]:
        result = _batch_scaling_torch(spec, sizes)
    else:
        result = _batch_scaling_numpy(spec, sizes)
    points = result.get("points", [])
    if not points:
        return {"skipped": True,
                "reason": result.get("reason", "no batch size completed")}

    best = max(points, key=lambda p: p["samples_per_s"])
    # The knee: smallest batch reaching 95% of peak throughput. Anything larger
    # costs memory for no speed.
    knee = next((p for p in points
                 if p["samples_per_s"] >= 0.95 * best["samples_per_s"]), best)
    return {
        "points": points,
        "device": result.get("device", "cpu"),
        "model": spec.describe(),
        "best_batch": best["batch"],
        "best_samples_per_s": round(best["samples_per_s"], 1),
        "recommended_batch": knee["batch"],
        "note": (f"batch {knee['batch']} reaches 95% of peak throughput; "
                 f"larger batches cost memory without meaningful speedup"),
    }


def _batch_scaling_numpy(spec: "ModelSpec", sizes: list[int]) -> dict:
    import numpy as np

    layers = _np_weights(spec, np)
    rng = np.random.default_rng(3)
    points = []
    for batch in sizes:
        try:
            x = rng.standard_normal((batch, spec.d_model),
                                    dtype=np.float32) * 0.1
            _np_forward(x, layers, spec, np)
            start = clock()
            _np_forward(x, layers, spec, np)
            elapsed = clock() - start
        except (MemoryError, ValueError):
            break
        if elapsed <= 0:
            continue
        points.append({"batch": batch,
                       "samples_per_s": batch / elapsed,
                       "ms_per_batch": round(elapsed * 1000, 2)})
    return {"points": points, "device": "cpu"}


def _batch_scaling_torch(spec: "ModelSpec", sizes: list[int]) -> dict:
    try:
        import torch
    except Exception:
        return {"points": [], "reason": "PyTorch is not installed"}

    device = best_torch_device()
    dev = torch.device(device)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    scale = 1.0 / math.sqrt(spec.d_model)
    torch.manual_seed(3)
    try:
        w1 = torch.randn(spec.d_model, spec.d_ff, device=dev, dtype=dtype) * scale
        w2 = torch.randn(spec.d_ff, spec.d_model, device=dev, dtype=dtype) * scale
    except (RuntimeError, MemoryError) as e:
        return {"points": [], "reason": str(e)}

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    points = []
    with torch.inference_mode():
        for batch in sizes:
            try:
                x = torch.randn(batch, spec.d_model, device=dev, dtype=dtype)
                for _ in range(2):
                    torch.nn.functional.gelu(x @ w1) @ w2
                sync()
                start = clock()
                for _ in range(spec.n_layers):
                    torch.nn.functional.gelu(x @ w1) @ w2
                sync()
                elapsed = clock() - start
            except (RuntimeError, MemoryError):
                break
            if elapsed <= 0:
                continue
            points.append({"batch": batch,
                           "samples_per_s": batch / elapsed,
                           "ms_per_batch": round(elapsed * 1000, 2)})
    return {"points": points, "device": device}


# --------------------------------------------------------------------------- #
# Accelerator memory
# --------------------------------------------------------------------------- #
#: Bytes per parameter at common quantisations, for the "will it fit?" table.
_QUANT_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}

#: Room for the KV cache, activations, and the runtime itself. Loading weights
#: into exactly the free memory fails in practice.
_OVERHEAD = 1.25


def accelerator_memory(system_ram_bytes: int = 0) -> dict:
    """Usable accelerator memory, and which model sizes fit in it."""
    total = None
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            total = torch.cuda.get_device_properties(0).total_memory
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            # Apple silicon shares one pool with the CPU; the GPU may address
            # most but not all of it.
            total = getattr(torch.mps, "recommended_max_memory", lambda: 0)()
            if not total:
                total = int(system_ram_bytes * 0.75) if system_ram_bytes else None
    except Exception:
        pass

    if not total:
        total = system_ram_bytes or None
        device = "cpu"
    if not total:
        return {"skipped": True, "reason": "memory size could not be determined"}

    fits = {}
    for name, bytes_per_param in _QUANT_BYTES.items():
        billions = total / (bytes_per_param * 1e9 * _OVERHEAD)
        fits[name] = round(billions, 1)

    return {
        "device": device,
        "total_bytes": int(total),
        "total_gb": round(total / GB, 1),
        "largest_model_billions": fits,
        "unified_memory": device == "mps",
        "note": (f"largest model that fits, in billions of parameters, "
                 f"allowing {int((_OVERHEAD - 1) * 100)}% for the KV cache, "
                 f"activations, and runtime"),
    }


# --------------------------------------------------------------------------- #
# Dataframe operations
# --------------------------------------------------------------------------- #
_DF_ROWS = 2_000_000


def dataframes(rows: int = _DF_ROWS) -> dict:
    """Filter, group-by, join, and sort throughput across installed engines.

    Deliberately the same four operations on the same generated data for every
    engine, so the comparison is between engines rather than between benchmarks.
    These are the TPC-H shapes that dominate real analytical work.
    """
    status = available()
    engines = status["dataframes"]
    if not engines:
        return {"skipped": True,
                "reason": "none of pandas, polars, or duckdb is installed",
                "hint": "pip install pandas polars duckdb"}
    try:
        import numpy as np
    except Exception:
        return {"skipped": True, "reason": "NumPy is not installed"}

    rng = np.random.default_rng(42)
    data = {
        "id": np.arange(rows, dtype=np.int64),
        "category": rng.integers(0, 1000, size=rows, dtype=np.int32),
        "region": rng.integers(0, 20, size=rows, dtype=np.int32),
        "value": rng.random(rows).astype(np.float64) * 1000.0,
    }
    lookup_rows = 1000
    lookup = {
        "category": np.arange(lookup_rows, dtype=np.int32),
        "weight": rng.random(lookup_rows).astype(np.float64),
    }

    results: dict = {"rows": rows, "engines": {}}
    for name in engines:
        try:
            results["engines"][name] = _bench_engine(name, data, lookup, np)
        except Exception as e:
            results["engines"][name] = {"error": f"{type(e).__name__}: {e}"}

    ranked = [(n, r["total_s"]) for n, r in results["engines"].items()
              if isinstance(r, dict) and r.get("total_s")]
    if ranked:
        ranked.sort(key=lambda kv: kv[1])
        results["fastest"] = ranked[0][0]
        if len(ranked) > 1:
            results["speedup_over_slowest"] = round(
                ranked[-1][1] / ranked[0][1], 2)
    return results


def _bench_engine(name: str, data: dict, lookup: dict, np) -> dict:
    """Four operations, timed individually, on one engine."""
    timings: dict[str, float] = {}
    checks: dict[str, int] = {}

    if name == "pandas":
        import pandas as pd
        df = pd.DataFrame(data)
        lk = pd.DataFrame(lookup)

        t = clock()
        filtered = df[df["value"] > 500.0]
        timings["filter"] = clock() - t
        checks["filter_rows"] = len(filtered)

        t = clock()
        grouped = df.groupby("category", observed=True)["value"].agg(
            ["sum", "mean", "count"])
        timings["groupby"] = clock() - t
        checks["groups"] = len(grouped)

        t = clock()
        joined = df.merge(lk, on="category", how="inner")
        timings["join"] = clock() - t
        checks["join_rows"] = len(joined)

        t = clock()
        df.sort_values("value")
        timings["sort"] = clock() - t

    elif name == "polars":
        import polars as pl
        df = pl.DataFrame(data)
        lk = pl.DataFrame(lookup)

        t = clock()
        filtered = df.filter(pl.col("value") > 500.0)
        timings["filter"] = clock() - t
        checks["filter_rows"] = filtered.height

        t = clock()
        grouped = df.group_by("category").agg([
            pl.col("value").sum().alias("sum"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").count().alias("count")])
        timings["groupby"] = clock() - t
        checks["groups"] = grouped.height

        t = clock()
        joined = df.join(lk, on="category", how="inner")
        timings["join"] = clock() - t
        checks["join_rows"] = joined.height

        t = clock()
        df.sort("value")
        timings["sort"] = clock() - t

    elif name == "duckdb":
        import duckdb
        con = duckdb.connect()
        con.register("t", _arrow_or_dict(data, np))
        con.register("lk", _arrow_or_dict(lookup, np))

        t = clock()
        checks["filter_rows"] = con.execute(
            "SELECT count(*) FROM t WHERE value > 500.0").fetchone()[0]
        timings["filter"] = clock() - t

        t = clock()
        checks["groups"] = len(con.execute(
            "SELECT category, sum(value), avg(value), count(*) "
            "FROM t GROUP BY category").fetchall())
        timings["groupby"] = clock() - t

        t = clock()
        checks["join_rows"] = con.execute(
            "SELECT count(*) FROM t JOIN lk USING (category)").fetchone()[0]
        timings["join"] = clock() - t

        t = clock()
        con.execute("SELECT value FROM t ORDER BY value").fetchall()
        timings["sort"] = clock() - t
        con.close()
    else:
        return {"error": f"unknown engine {name}"}

    rows = len(data["id"])
    total = sum(timings.values())
    return {
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
        "rows_per_s": {k: round(rows / v) if v > 0 else None
                       for k, v in timings.items()},
        "total_s": round(total, 4),
        "checks": checks,
    }


def _arrow_or_dict(data: dict, np):
    """DuckDB reads NumPy dicts directly; a DataFrame is not required."""
    return {k: v for k, v in data.items()}


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def run(memory_bytes: int = 0, seconds: float = 2.0,
        skip_llm: bool = False, skip_dataframes: bool = False,
        prefill_tokens: int = 256, decode_tokens: int = 32) -> dict:
    """The whole data-science tier."""
    out: dict = {"available": available()}
    if not skip_llm:
        out["llm"] = llm(memory_bytes, prefill_tokens, decode_tokens)
        out["batch_scaling"] = batch_scaling(memory_bytes)
    out["accelerator_memory"] = accelerator_memory(memory_bytes)
    out["dataloader"] = dataloader(min(seconds, 2.0))
    if not skip_dataframes:
        out["dataframes"] = dataframes()
    return out


def extract_rates(result: dict | None) -> dict:
    """Scoreable rates from the data-science tier."""
    if not result:
        return {}
    out = {}
    llm_result = result.get("llm") or {}
    if llm_result.get("decode_tokens_per_s"):
        out["llm_decode"] = float(llm_result["decode_tokens_per_s"])
    if llm_result.get("prefill_tokens_per_s"):
        out["llm_prefill"] = float(llm_result["prefill_tokens_per_s"])
    dl = result.get("dataloader") or {}
    if not dl.get("skipped") and dl.get("rate"):
        out["dataloader"] = float(dl["rate"])
    df = result.get("dataframes") or {}
    engines = df.get("engines") or {}
    best = [e.get("total_s") for e in engines.values()
            if isinstance(e, dict) and e.get("total_s")]
    if best:
        # Operations per second across the four-query suite, so a faster engine
        # scores higher.
        out["dataframe"] = 4.0 / min(best)
    return out


def render(result: dict | None) -> str:
    """Terminal block for the data-science tier."""
    if not result:
        return ""
    lines: list[str] = []

    llm_result = result.get("llm") or {}
    if llm_result.get("model"):
        lines.append(f"  LLM model           : {llm_result['model']} "
                     f"(identical on every backend below)")
    for key, label in (("accelerated", "Accelerated"), ("cpu", "CPU")):
        entry = llm_result.get(key)
        if not isinstance(entry, dict):
            continue
        if entry.get("skipped"):
            lines.append(f"  LLM {label:<12}: skipped — {entry['reason']}")
            continue
        lines.append(f"    {label} ({entry['device']}, {entry['dtype']}, "
                     f"{entry['weights_mb']:,.0f} MB of weights)")
        lines.append(f"      prefill : {entry['prefill_tokens_per_s']:>10,.0f} "
                     f"tok/s   ({entry['prefill_tflops']:.2f} TFLOPS, "
                     f"compute-bound)")
        lines.append(f"      decode  : {entry['decode_tokens_per_s']:>10,.0f} "
                     f"tok/s   ({entry['decode_bandwidth_gb_s']:.0f} GB/s "
                     f"achieved, bandwidth-bound)")

    mem = result.get("accelerator_memory") or {}
    if mem and not mem.get("skipped"):
        fits = mem.get("largest_model_billions", {})
        lines.append(f"  Accelerator memory  : {mem.get('total_gb')} GB "
                     f"({mem.get('device')}"
                     f"{', unified' if mem.get('unified_memory') else ''})")
        lines.append(f"      largest model that fits: "
                     f"{fits.get('fp16', 0)}B at fp16, "
                     f"{fits.get('int8', 0)}B at int8, "
                     f"{fits.get('int4', 0)}B at int4")

    bs = result.get("batch_scaling") or {}
    if bs.get("points"):
        lines.append(f"  Batch scaling       : peak {bs['best_samples_per_s']:,.0f} "
                     f"samples/s at batch {bs['best_batch']}; "
                     f"batch {bs['recommended_batch']} reaches 95% of it")

    dl = result.get("dataloader") or {}
    if dl and not dl.get("skipped"):
        lines.append(f"  Input pipeline      : "
                     f"{dl['single_worker_samples_per_s']:,.0f} samples/s per "
                     f"worker, ~{dl['estimated_all_cores_samples_per_s']:,.0f} "
                     f"across {dl['workers_assumed']} cores")

    df = result.get("dataframes") or {}
    if df.get("skipped"):
        lines.append(f"  Dataframes          : skipped — {df['reason']}")
    elif df.get("engines"):
        lines.append(f"  Dataframes ({df['rows']:,} rows, seconds for "
                     f"filter/groupby/join/sort)")
        for name, entry in df["engines"].items():
            if entry.get("error"):
                lines.append(f"      {name:<8}: error — {entry['error']}")
                continue
            t = entry["timings_s"]
            lines.append(f"      {name:<8}: {t.get('filter', 0):.3f} "
                         f"{t.get('groupby', 0):.3f} {t.get('join', 0):.3f} "
                         f"{t.get('sort', 0):.3f}   total {entry['total_s']:.3f}s")
        if df.get("fastest"):
            lines.append(f"      fastest: {df['fastest']}"
                         + (f" ({df['speedup_over_slowest']}x the slowest)"
                            if df.get("speedup_over_slowest") else ""))
    return "\n".join(lines)
