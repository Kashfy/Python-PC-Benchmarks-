"""Generate an ONNX model for NPU benchmarking — no ``onnx`` package needed.

ONNX Runtime is the one runtime that reaches every vendor's NPU through its
execution providers (OpenVINO for Intel, Vitis AI for AMD, QNN for Qualcomm,
DirectML for anything DX12, Core ML for Apple). Benchmarking those requires a
model file.

Building one normally means installing the ``onnx`` package on top of
``onnxruntime``. Instead this writes the ModelProto wire format directly — the
same approach used for Core ML in :mod:`pcbench.coreml_model` — so the NPU
benchmark needs only ``onnxruntime`` itself.

The graph is a stack of ``MatMul`` + ``Relu``. Matrix multiply is the operation
NPUs are built to accelerate, and neither op takes attributes, which keeps the
encoder small and robust.
"""

from __future__ import annotations

import os
import struct

# TensorProto.DataType
_FLOAT = 1

# IR version 7 pairs with opset 13 — the most widely supported combination
# across ONNX Runtime builds and vendor execution providers.
_IR_VERSION = 7
_OPSET = 13

# Sized so an NPU is actually given enough work to be worth dispatching to.
# The Core ML experiment showed a too-small model is silently kept on the CPU,
# producing a benchmark that measures nothing.
DEFAULT_DIM = 1024
DEFAULT_LAYERS = 10
DEFAULT_BATCH = 32


# --------------------------------------------------------------------------- #
# Protobuf wire-format primitives
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


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited: nested message, string, or bytes."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _uint(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _str(field: int, text: str) -> bytes:
    return _ld(field, text.encode("utf-8"))


def _packed_ints(field: int, values) -> bytes:
    return _ld(field, b"".join(_varint(v) for v in values))


# --------------------------------------------------------------------------- #
# ONNX message construction
# --------------------------------------------------------------------------- #
def _tensor_value_info(name: str, shape) -> bytes:
    """ValueInfoProto{name, type: TypeProto{tensor_type}}."""
    dims = b"".join(_ld(1, _uint(1, d)) for d in shape)   # Dimension.dim_value
    shape_proto = _ld(2, dims)                            # Tensor.shape
    tensor_type = _uint(1, _FLOAT) + shape_proto          # Tensor.elem_type
    type_proto = _ld(1, tensor_type)                      # TypeProto.tensor_type
    return _str(1, name) + _ld(2, type_proto)


def _initializer(name: str, shape, values) -> bytes:
    """TensorProto carrying weights as little-endian raw_data."""
    raw = struct.pack(f"<{len(values)}f", *values)
    return (_packed_ints(1, shape)      # dims
            + _uint(2, _FLOAT)          # data_type
            + _str(8, name)             # name
            + _ld(9, raw))              # raw_data


def _node(op_type: str, inputs, outputs, name: str) -> bytes:
    """NodeProto. Neither MatMul nor Relu takes attributes."""
    body = b"".join(_str(1, i) for i in inputs)
    body += b"".join(_str(2, o) for o in outputs)
    body += _str(3, name)
    body += _str(4, op_type)
    return body


def build_model(dim: int = DEFAULT_DIM, layers: int = DEFAULT_LAYERS,
                batch: int = DEFAULT_BATCH) -> bytes:
    """Serialize a MatMul/Relu stack as a complete ONNX ModelProto."""
    # Deterministic weights scaled by 1/dim. Without that scaling each layer
    # multiplies activation magnitude by roughly `dim`, so a deep stack
    # overflows to infinity and the model produces NaN rather than a timing.
    mag = 1.0 / dim
    weights = [mag if (i % 7) else -mag for i in range(dim * dim)]

    # One weight tensor shared by every layer. ONNX allows several nodes to
    # consume the same initializer, which keeps the file at a few MB instead
    # of multiplying it by the layer count.
    inits = [_initializer("W", [dim, dim], weights)]

    nodes, graph_body = [], b""
    prev = "input"
    for i in range(layers):
        mm_out = f"m{i}"
        nodes.append(_node("MatMul", [prev, "W"], [mm_out], f"matmul{i}"))
        act_out = "output" if i == layers - 1 else f"h{i}"
        nodes.append(_node("Relu", [mm_out], [act_out], f"relu{i}"))
        prev = act_out

    graph_body += b"".join(_ld(1, n) for n in nodes)          # node
    graph_body += _str(2, "pcbench_npu")                      # name
    graph_body += b"".join(_ld(5, t) for t in inits)          # initializer
    graph_body += _ld(11, _tensor_value_info("input", [batch, dim]))
    graph_body += _ld(12, _tensor_value_info("output", [batch, dim]))

    opset = _ld(8, _str(1, "") + _uint(2, _OPSET))            # opset_import
    return (_uint(1, _IR_VERSION)
            + _str(2, "pcbench")                              # producer_name
            + opset
            + _ld(7, graph_body))                             # graph


def flops_per_inference(dim: int = DEFAULT_DIM, layers: int = DEFAULT_LAYERS,
                        batch: int = DEFAULT_BATCH) -> float:
    """FLOPs for one forward pass: each MatMul is 2*batch*dim*dim."""
    return float(2 * batch * dim * dim * layers)


def write_model(path: str, dim: int = DEFAULT_DIM,
                layers: int = DEFAULT_LAYERS,
                batch: int = DEFAULT_BATCH) -> str:
    """Write the model, reusing an identical existing file."""
    blob = build_model(dim, layers, batch)
    if os.path.isfile(path) and os.path.getsize(path) == len(blob):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return path
