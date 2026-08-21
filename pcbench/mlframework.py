"""Optional AI training & inference benchmarks via a real ML framework.

This is the **one** place the tool will use a third-party dependency, and only
if the user already has one installed — it is never required and never
installed automatically. The rest of pcbench stays pure standard library.

Why a framework is unavoidable here: real *training* means forward pass,
backpropagation, and an optimizer step. Core ML and Metal can measure inference
and raw compute, but not a genuine training loop. PyTorch or ONNX Runtime is
the honest way to report training throughput — and, as a bonus, it works across
NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and plain CPU, which the Apple-native
path cannot.

Everything degrades gracefully: if no framework is importable, this module
reports ``available: False`` and the run continues.
"""

from __future__ import annotations

import os
import time


def _torch_device(torch):
    """Pick the best device this PyTorch build can actually use."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "Apple GPU (MPS)"
    return "cpu", "CPU"


def _sync(torch, device: str) -> None:
    """Block until queued GPU work finishes, so timing is honest.

    GPU frameworks are asynchronous: without this the timer would measure how
    fast work is *submitted*, not how fast it runs.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        getattr(torch, "mps", None) and torch.mps.synchronize()


def _bench_torch(seconds: float, batch: int) -> dict:
    import torch
    import torch.nn as nn

    device, dev_name = _torch_device(torch)
    torch.manual_seed(0)

    # A small but representative CNN — the shape of real image models, big
    # enough to exercise the hardware without needing a dataset on disk.
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    x = torch.randn(batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    params = sum(p.numel() for p in model.parameters())

    # ---- Training throughput (forward + backward + optimizer step) ----
    model.train()
    for _ in range(3):                              # warm up / cudnn autotune
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    _sync(torch, device)

    start = time.perf_counter()
    steps = 0
    while time.perf_counter() - start < seconds:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        steps += 1
    _sync(torch, device)
    train_elapsed = time.perf_counter() - start
    train_sps = steps * batch / train_elapsed

    # ---- Inference throughput (forward only, no gradients) ----
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        _sync(torch, device)
        start = time.perf_counter()
        n = 0
        while time.perf_counter() - start < seconds:
            model(x)
            n += 1
        _sync(torch, device)
        infer_elapsed = time.perf_counter() - start
    infer_sps = n * batch / infer_elapsed

    return {
        "available": True,
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "device": device,
        "device_name": dev_name,
        "batch_size": batch,
        "model_params": params,
        "train_samples_per_s": round(train_sps, 1),
        "infer_samples_per_s": round(infer_sps, 1),
    }


def _bench_onnx(seconds: float, batch: int) -> dict:
    """Inference-only fallback when ONNX Runtime is present but PyTorch isn't.

    ONNX Runtime cannot train, so this reports inference throughput only and
    says so. The model comes from :mod:`pcbench.onnx_model`, which writes the
    protobuf directly — so the heavyweight ``onnx`` package is not required.
    """
    import tempfile

    import numpy as np
    import onnxruntime as ort

    from . import onnx_model

    providers = [p for p in ort.get_available_providers()
                 if p != "AzureExecutionProvider"]
    path = onnx_model.write_model(
        os.path.join(tempfile.gettempdir(), "pcbench_ml.onnx"))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    dim = onnx_model.DEFAULT_DIM
    feed = {"input": np.random.rand(onnx_model.DEFAULT_BATCH, dim)
            .astype(np.float32)}

    for _ in range(3):
        sess.run(None, feed)
    start = time.perf_counter()
    n = 0
    while time.perf_counter() - start < seconds:
        sess.run(None, feed)
        n += 1
    elapsed = time.perf_counter() - start

    active = sess.get_providers()
    return {
        "available": True,
        "framework": "onnxruntime",
        "framework_version": ort.__version__,
        "device": active[0] if active else "cpu",
        "device_name": (active[0].replace("ExecutionProvider", "")
                        if active else "CPU"),
        "providers": providers,
        "batch_size": onnx_model.DEFAULT_BATCH,
        "infer_samples_per_s": round(n * onnx_model.DEFAULT_BATCH / elapsed, 1),
        "gflops": round(n * onnx_model.flops_per_inference() / elapsed / 1e9, 1),
        "note": "ONNX Runtime does not train; inference only",
    }


def detect() -> dict:
    """Report which ML framework, if any, is importable — without running it."""
    info = {"pytorch": None, "onnxruntime": None}
    try:
        import torch
        info["pytorch"] = torch.__version__
    except Exception:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    info["available"] = bool(info["pytorch"] or info["onnxruntime"])
    return info


def run(seconds: float = 3.0, batch: int = 64) -> dict:
    """Run the AI framework benchmark with whatever is installed.

    Prefers PyTorch (which can train); falls back to ONNX Runtime (inference
    only). Returns ``{"available": False, ...}`` when neither is present.
    """
    try:
        import torch  # noqa: F401
        return _bench_torch(seconds, batch)
    except ImportError:
        pass
    except Exception as e:
        return {"available": True, "framework": "pytorch",
                "error": f"{type(e).__name__}: {e}"}

    try:
        import onnxruntime  # noqa: F401
        return _bench_onnx(seconds, batch)
    except ImportError:
        pass
    except Exception as e:
        return {"available": True, "framework": "onnxruntime",
                "error": f"{type(e).__name__}: {e}"}

    return {
        "available": False,
        "note": ("no ML framework found. Install one for real AI training/"
                 "inference numbers: pip install torch  (or onnxruntime)"),
    }


def extract_rates(payload: dict | None) -> dict:
    """Pull training/inference throughput out for scoring."""
    rates: dict[str, float] = {}
    if not payload or not payload.get("available") or payload.get("error"):
        return rates
    t = payload.get("train_samples_per_s")
    i = payload.get("infer_samples_per_s")
    if isinstance(t, (int, float)) and t > 0:
        rates["ml_train"] = float(t)
    if isinstance(i, (int, float)) and i > 0:
        rates["ml_infer"] = float(i)
    return rates
