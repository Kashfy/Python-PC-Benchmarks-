"""Generate a Core ML model for Neural Engine benchmarking — no dependencies.

The Apple Neural Engine cannot be programmed directly; there is no public API
for submitting arbitrary work to it. The only supported path is Core ML, which
decides for itself whether a given model runs on CPU, GPU, or ANE. Benchmarking
the ANE therefore requires an actual model file.

Rather than depend on ``coremltools`` (a large package that would break the
tool's zero-dependency guarantee), this module writes the ``.mlmodel``
protobuf directly. The format is stable and the subset needed for a
convolution stack is small.

**Model size matters.** Core ML only schedules work on the ANE when the model
is big enough to be worth the dispatch cost. A small model is silently kept on
the CPU and produces a benchmark that measures nothing: measured here, a
16-channel 32x32 model ran at 0.92x of CPU-only speed (i.e. never left the
CPU), while the 64-channel 64x64 default below reaches 5.4x — proof the ANE is
actually engaged.
"""

from __future__ import annotations

import os
import struct

# Core ML ArrayFeatureType.ArrayDataType
_FLOAT32 = 65568

# Defaults chosen to be large enough that Core ML dispatches to the ANE.
DEFAULT_CHANNELS = 64
DEFAULT_SPATIAL = 64
DEFAULT_LAYERS = 12
KERNEL = 3


# --------------------------------------------------------------------------- #
# Minimal protobuf writer (wire format only — no schema compiler needed)
# --------------------------------------------------------------------------- #
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _msg(field: int, payload: bytes) -> bytes:
    """Length-delimited field (nested message, string, or packed array)."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _uint(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _str(field: int, text: str) -> bytes:
    return _msg(field, text.encode("utf-8"))


def _packed_uints(field: int, values) -> bytes:
    return _msg(field, b"".join(_varint(v) for v in values))


def _packed_floats(field: int, values) -> bytes:
    return _msg(field, struct.pack(f"<{len(values)}f", *values))


# --------------------------------------------------------------------------- #
# Core ML message construction
# --------------------------------------------------------------------------- #
def _feature(name: str, shape) -> bytes:
    """FeatureDescription{name, type: FeatureType{multiArrayType}}."""
    array_type = _msg(5, _packed_uints(1, shape) + _uint(2, _FLOAT32))
    return _str(1, name) + _msg(3, array_type)


def _conv_layer(name: str, src: str, dst: str, channels: int) -> bytes:
    """One NeuralNetworkLayer holding a ConvolutionLayerParams."""
    n_weights = channels * channels * KERNEL * KERNEL
    # Small non-zero weights keep activations finite across a deep stack;
    # the values are irrelevant to timing but must not produce NaN/Inf.
    weights = [0.01] * n_weights
    bias = [0.0] * channels

    conv = _uint(1, channels)              # outputChannels
    conv += _uint(2, channels)             # kernelChannels
    conv += _uint(3, 1)                    # nGroups
    conv += _packed_uints(20, [KERNEL, KERNEL])   # kernelSize
    conv += _packed_uints(30, [1, 1])             # stride
    conv += _packed_uints(40, [1, 1])             # dilationFactor
    conv += _msg(51, b"")                  # same padding (keeps H,W constant)
    conv += _uint(70, 1)                   # hasBias
    conv += _msg(90, _packed_floats(1, weights))
    conv += _msg(91, _packed_floats(1, bias))

    layer = _str(1, name) + _str(2, src) + _str(3, dst) + _msg(100, conv)
    return _msg(1, layer)                  # NeuralNetwork.layers


def build_model(channels: int = DEFAULT_CHANNELS,
                spatial: int = DEFAULT_SPATIAL,
                layers: int = DEFAULT_LAYERS) -> bytes:
    """Serialize a convolution-stack .mlmodel.

    Convolution is used because it is the operation the ANE is built for and
    the one Core ML most reliably offloads.
    """
    body = b""
    src = "input"
    for i in range(layers):
        dst = "output" if i == layers - 1 else f"h{i}"
        body += _conv_layer(f"conv{i}", src, dst, channels)
        src = dst
    # arrayInputShapeMapping = EXACT_ARRAY_MAPPING, so [1,C,H,W] is taken as-is.
    network = body + _uint(5, 1)

    shape = [1, channels, spatial, spatial]
    description = (_msg(1, _feature("input", shape))
                   + _msg(10, _feature("output", shape)))

    # specificationVersion 4 covers everything used here.
    return _uint(1, 4) + _msg(2, description) + _msg(500, network)


def flops_per_inference(channels: int = DEFAULT_CHANNELS,
                        spatial: int = DEFAULT_SPATIAL,
                        layers: int = DEFAULT_LAYERS) -> float:
    """Total FLOPs for one forward pass.

    Each output element of a conv layer costs ``in_ch * K * K`` multiply-adds,
    and a multiply-add is two floating-point operations.
    """
    per_layer = (channels * channels * KERNEL * KERNEL
                 * spatial * spatial * 2)
    return float(per_layer * layers)


def write_model(path: str, channels: int = DEFAULT_CHANNELS,
                spatial: int = DEFAULT_SPATIAL,
                layers: int = DEFAULT_LAYERS) -> str:
    """Write the model, reusing an existing file of the same size."""
    blob = build_model(channels, spatial, layers)
    if os.path.isfile(path) and os.path.getsize(path) == len(blob):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return path
