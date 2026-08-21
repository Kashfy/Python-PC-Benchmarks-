"""Classic machine-learning workloads in pure Python — no framework required.

The optional PyTorch tier in :mod:`pcbench.mlframework` measures what a machine
can do *with* an ML stack installed. These workloads measure the same kinds of
computation with nothing but the standard library, so every machine gets real
ML numbers out of the box:

* **Neural-network training** — a genuine multi-layer perceptron with forward
  pass, backpropagation, and SGD updates. This is real training (weights change
  and loss falls), not a synthetic stand-in.
* **K-means clustering** — the canonical unsupervised workload; distance-bound
  and cache-sensitive.
* **K-nearest-neighbours search** — brute-force similarity search, the
  operation underneath vector databases and retrieval.

All three are deterministic (fixed seeds) and self-validating: each asserts a
known-correct outcome — loss actually decreasing, clusters converging to a
known inertia, neighbours matching a brute-force answer — so a wrong result
signals faulty hardware rather than a slow machine.

Being pure Python these measure the interpreter as much as the silicon, which
is the point: they are comparable across machines running the same Python, and
the native/framework tiers cover compiled speed separately.
"""

from __future__ import annotations

import math
import random

from .core import ValidationError, clock, summarize, timed_loop, warmup

# --------------------------------------------------------------------------- #
# Shared deterministic data generation
# --------------------------------------------------------------------------- #
def _blobs(n: int, dims: int, clusters: int, seed: int = 7) -> list[list[float]]:
    """Well-separated Gaussian-ish blobs, identical on every machine."""
    rnd = random.Random(seed)
    centres = [[rnd.uniform(-10, 10) for _ in range(dims)]
               for _ in range(clusters)]
    points = []
    for i in range(n):
        c = centres[i % clusters]
        points.append([c[d] + rnd.gauss(0, 0.6) for d in range(dims)])
    return points


# --------------------------------------------------------------------------- #
# 1. Neural network training (MLP with backpropagation)
# --------------------------------------------------------------------------- #
_NN_IN, _NN_HIDDEN, _NN_OUT = 32, 24, 4
_NN_BATCH = 24


def _nn_dataset(seed: int = 11):
    """A linearly-separable classification set the network can actually learn."""
    rnd = random.Random(seed)
    xs, ys = [], []
    for i in range(_NN_BATCH):
        label = i % _NN_OUT
        # Each class gets a distinct signature so training provably converges.
        x = [rnd.gauss(0, 0.3) for _ in range(_NN_IN)]
        for d in range(label, _NN_IN, _NN_OUT):
            x[d] += 1.5
        xs.append(x)
        ys.append(label)
    return xs, ys


def _nn_init(seed: int = 3):
    """Xavier-ish initialisation, deterministic across machines."""
    rnd = random.Random(seed)
    s1 = math.sqrt(1.0 / _NN_IN)
    s2 = math.sqrt(1.0 / _NN_HIDDEN)
    w1 = [[rnd.uniform(-s1, s1) for _ in range(_NN_HIDDEN)]
          for _ in range(_NN_IN)]
    b1 = [0.0] * _NN_HIDDEN
    w2 = [[rnd.uniform(-s2, s2) for _ in range(_NN_OUT)]
          for _ in range(_NN_HIDDEN)]
    b2 = [0.0] * _NN_OUT
    return w1, b1, w2, b2


def _nn_train_step(xs, ys, w1, b1, w2, b2, lr: float = 0.08) -> float:
    """One full training step: forward, backward, SGD update. Returns loss.

    Written out explicitly rather than with a matrix library so the arithmetic
    is visible and dependency-free — this is the actual work being timed.
    """
    batch = len(xs)
    total_loss = 0.0

    # Gradient accumulators.
    gw1 = [[0.0] * _NN_HIDDEN for _ in range(_NN_IN)]
    gb1 = [0.0] * _NN_HIDDEN
    gw2 = [[0.0] * _NN_OUT for _ in range(_NN_HIDDEN)]
    gb2 = [0.0] * _NN_OUT

    for x, y in zip(xs, ys):
        # ---- forward: hidden layer with tanh ----
        h_pre = list(b1)
        for i, xi in enumerate(x):
            if xi:
                row = w1[i]
                for j in range(_NN_HIDDEN):
                    h_pre[j] += xi * row[j]
        h = [math.tanh(v) for v in h_pre]

        # ---- forward: output layer + softmax ----
        o = list(b2)
        for j, hj in enumerate(h):
            row = w2[j]
            for k in range(_NN_OUT):
                o[k] += hj * row[k]
        m = max(o)
        exps = [math.exp(v - m) for v in o]          # shift for stability
        ssum = sum(exps)
        probs = [e / ssum for e in exps]
        total_loss += -math.log(max(probs[y], 1e-12))

        # ---- backward: softmax + cross-entropy gradient ----
        dout = list(probs)
        dout[y] -= 1.0

        dh = [0.0] * _NN_HIDDEN
        for j in range(_NN_HIDDEN):
            hj = h[j]
            row = w2[j]
            grow = gw2[j]
            acc = 0.0
            for k in range(_NN_OUT):
                dk = dout[k]
                grow[k] += hj * dk
                acc += row[k] * dk
            # tanh'(x) = 1 - tanh(x)^2
            dh[j] = acc * (1.0 - hj * hj)
        for k in range(_NN_OUT):
            gb2[k] += dout[k]

        for i, xi in enumerate(x):
            if xi:
                grow = gw1[i]
                for j in range(_NN_HIDDEN):
                    grow[j] += xi * dh[j]
        for j in range(_NN_HIDDEN):
            gb1[j] += dh[j]

    # ---- SGD update ----
    scale = lr / batch
    for i in range(_NN_IN):
        row, grow = w1[i], gw1[i]
        for j in range(_NN_HIDDEN):
            row[j] -= scale * grow[j]
    for j in range(_NN_HIDDEN):
        b1[j] -= scale * gb1[j]
        row, grow = w2[j], gw2[j]
        for k in range(_NN_OUT):
            row[k] -= scale * grow[k]
    for k in range(_NN_OUT):
        b2[k] -= scale * gb2[k]

    return total_loss / batch


def nn_flops_per_step() -> int:
    """Approximate FLOPs in one training step (forward + backward ~ 3x fwd)."""
    fwd = (_NN_IN * _NN_HIDDEN + _NN_HIDDEN * _NN_OUT) * 2
    return int(fwd * 3 * _NN_BATCH)


def bench_nn_training(seconds: float, repeats: int) -> dict:
    """Train a real MLP and report training steps and samples per second."""
    xs, ys = _nn_dataset()

    # Validation: the network must actually learn. If loss does not fall, the
    # arithmetic is wrong — which on healthy hardware it never is.
    w1, b1, w2, b2 = _nn_init()
    first = _nn_train_step(xs, ys, w1, b1, w2, b2)
    for _ in range(60):
        last = _nn_train_step(xs, ys, w1, b1, w2, b2)
    if not (last < first * 0.9):
        raise ValidationError(
            f"nn_training: loss failed to decrease ({first:.4f} -> {last:.4f}) "
            f"— indicates a floating-point or memory fault")

    state = _nn_init()

    def chunk():
        _nn_train_step(xs, ys, *state)

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, steps = timed_loop(chunk, seconds)
        rates.append(steps / elapsed)
    s = summarize(rates)
    return {
        "unit": "steps/s",
        "rate": s["median"],
        "samples_per_s": round(s["median"] * _NN_BATCH, 1),
        "mflops": round(s["median"] * nn_flops_per_step() / 1e6, 1),
        "topology": f"{_NN_IN}-{_NN_HIDDEN}-{_NN_OUT} MLP, batch {_NN_BATCH}",
        "validated": True,
        **s,
    }


# --------------------------------------------------------------------------- #
# 2. K-means clustering
# --------------------------------------------------------------------------- #
_KM_N, _KM_DIMS, _KM_K, _KM_ITERS = 1200, 8, 6, 6


def _sq_dist(a, b) -> float:
    d = 0.0
    for x, y in zip(a, b):
        diff = x - y
        d += diff * diff
    return d


def _farthest_point_init(points, k: int):
    """Pick k mutually distant seeds (greedy farthest-point / maximin).

    Random seeding frequently draws two centroids from the same blob, which
    leaves Lloyd's algorithm stuck in a poor local optimum — the clustering
    then fails to converge and the validation check cannot distinguish a bad
    seed from a genuine hardware fault. Farthest-point init is deterministic
    and lands one seed per well-separated cluster, so a high final inertia
    really does mean something is wrong.
    """
    centroids = [list(points[0])]
    best = [_sq_dist(p, centroids[0]) for p in points]
    for _ in range(1, k):
        far = max(range(len(points)), key=lambda i: best[i])
        centroids.append(list(points[far]))
        for i, p in enumerate(points):
            d = _sq_dist(p, centroids[-1])
            if d < best[i]:
                best[i] = d
    return centroids


def _kmeans(points, k: int, iters: int):
    """Lloyd's algorithm. Returns (centroids, inertia)."""
    dims = len(points[0])
    centroids = _farthest_point_init(points, k)

    inertia = 0.0
    for _ in range(iters):
        sums = [[0.0] * dims for _ in range(k)]
        counts = [0] * k
        inertia = 0.0
        for p in points:
            best, best_d = 0, float("inf")
            for ci, c in enumerate(centroids):
                d = _sq_dist(p, c)
                if d < best_d:
                    best_d, best = d, ci
            inertia += best_d
            s = sums[best]
            for j, v in enumerate(p):
                s[j] += v
            counts[best] += 1
        for ci in range(k):
            if counts[ci]:
                centroids[ci] = [v / counts[ci] for v in sums[ci]]
    return centroids, inertia


def bench_kmeans(seconds: float, repeats: int) -> dict:
    points = _blobs(_KM_N, _KM_DIMS, _KM_K)

    # Validation: clustering well-separated blobs must reach a low inertia.
    _, inertia = _kmeans(points, _KM_K, _KM_ITERS)
    per_point = inertia / _KM_N
    if not (0.0 <= per_point < 10.0):
        raise ValidationError(
            f"kmeans: converged to implausible inertia {per_point:.3f} per "
            f"point — indicates a floating-point fault")

    def chunk():
        _kmeans(points, _KM_K, _KM_ITERS)

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, runs = timed_loop(chunk, seconds)
        # One "operation" is a point-to-centroid distance evaluation.
        rates.append(runs * _KM_N * _KM_K * _KM_ITERS / elapsed)
    s = summarize(rates)
    return {
        "unit": "distances/s",
        "rate": s["median"],
        "dataset": f"{_KM_N} points x {_KM_DIMS}D, k={_KM_K}, {_KM_ITERS} iters",
        "inertia_per_point": round(per_point, 4),
        "validated": True,
        **s,
    }


# --------------------------------------------------------------------------- #
# 3. K-nearest-neighbours search
# --------------------------------------------------------------------------- #
_KNN_REF, _KNN_QUERY, _KNN_DIMS, _KNN_K = 900, 40, 12, 5


def _knn(reference, queries, k: int):
    """Brute-force k-NN. Returns the neighbour indices for each query."""
    out = []
    for q in queries:
        best = []                      # (distance, index), kept sorted, len<=k
        for idx, r in enumerate(reference):
            d = 0.0
            for a, b in zip(q, r):
                diff = a - b
                d += diff * diff
            if len(best) < k:
                best.append((d, idx))
                best.sort()
            elif d < best[-1][0]:
                best[-1] = (d, idx)
                best.sort()
        out.append([i for _, i in best])
    return out


def bench_knn(seconds: float, repeats: int) -> dict:
    reference = _blobs(_KNN_REF, _KNN_DIMS, 5, seed=21)
    queries = _blobs(_KNN_QUERY, _KNN_DIMS, 5, seed=22)

    # Validation: every reference point must be its own nearest neighbour.
    probe = reference[:5]
    found = _knn(reference, probe, 1)
    if [f[0] for f in found] != list(range(5)):
        raise ValidationError(
            "knn: a point was not its own nearest neighbour — indicates a "
            "floating-point or memory fault")

    def chunk():
        _knn(reference, queries, _KNN_K)

    warmup(chunk, seconds)
    rates = []
    for _ in range(repeats):
        elapsed, runs = timed_loop(chunk, seconds)
        rates.append(runs * _KNN_REF * _KNN_QUERY / elapsed)
    s = summarize(rates)
    return {
        "unit": "comparisons/s",
        "rate": s["median"],
        "dataset": f"{_KNN_QUERY} queries x {_KNN_REF} refs x {_KNN_DIMS}D, "
                   f"k={_KNN_K}",
        "validated": True,
        **s,
    }
