---
title: "Lesson 2 — Model Formats"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 2
tags: ["onnx", "gguf", "coreml", "litert", "executorch", "pte", "conversion"]
author: "Sudipta Pathak"
prerequisites: ["01-edge-runtime-map"]
---

# Lesson 2 — Model Formats

## Why this lesson exists

The runtime layer expects a model in a specific format. The format determines what's representable, what's expressible, and what the runtime can optimize. This lesson is a tour of the five formats you'll encounter — ONNX, GGUF, Core ML, LiteRT, ExecuTorch — plus the conversion paths between them and the things that go wrong in each.

The unifying theme: a "model file" is more than weights. It's also the graph (which ops execute in what order), the dtypes, sometimes the quantization scheme, sometimes runtime hints (preferred backend, supported dynamic shapes). Different formats represent these things differently; conversion that preserves all of them is harder than conversion that preserves just the weights.

The lesson is reading. The Hands-on converts a small PyTorch model into each of the major formats and shows what the conversion produces.

## What's in a model file

A complete on-device model file typically contains:

1. **Weights**: the trained parameters, in some dtype (FP16, BF16, INT8, INT4, ...).
2. **The computation graph**: nodes (ops) and edges (tensor flow) describing the forward pass.
3. **Dtype and shape information**: per-tensor metadata.
4. **Quantization metadata** (if quantized): scales, zero-points, per-group structure.
5. **Sometimes**: tokenizer state (LLM), preprocessing parameters (vision), runtime hints, vendor-specific opcodes.

Different formats include different subsets and arrange them differently on disk. The least-common-denominator subset (1 + 2) is what everyone supports; (3) is universal but spelled differently; (4) and (5) vary widely.

## ONNX

ONNX (Open Neural Network Exchange) is a Microsoft-led, open-spec format that uses Protocol Buffers as its serialization. It was designed in 2017 explicitly as an interchange format between frameworks.

A `.onnx` file is a protobuf-serialized `ModelProto` containing:

- A `GraphProto`: the computation graph (nodes, inputs, outputs, initializers).
- A list of opsets and their versions (the canonical ONNX operator catalog has ~190 op types as of opset 20).
- Weights stored as `TensorProto` initializers inside the graph or external files (the `external_data` mechanism for large weight tensors).

ONNX strengths:
- **Wide producer support**: PyTorch, TensorFlow, JAX, scikit-learn, XGBoost all export to ONNX.
- **Wide consumer support**: ONNX Runtime, ExecuTorch (via the `to_executorch` converter), Core ML (via `coremltools.convert`), LiteRT (via `tf2onnx` round-trip), QNN, NCNN, MNN — almost every edge runtime can import ONNX.
- **Well-documented spec.** When something fails, you can read the spec and reason about what should happen.

ONNX limitations:
- **Opset drift.** Producers and consumers may target different opset versions. PyTorch 2.x exports to opset 17–20; a runtime that supports up to opset 14 will fail. Backward-compatibility is decent within the same opset family but not universal.
- **Custom-op friction.** Operations not in the standard catalog become "custom ops" that the runtime must know about. PyTorch's recent ops (FlashAttention variants, rotary embeddings before they got standardized) often hit this.
- **Quantization representation.** ONNX supports quantized models via specific ops (QuantizeLinear, DequantizeLinear), but the representation is verbose and sometimes hard to optimize through. Many runtimes prefer to consume FP16 ONNX and quantize themselves.

ONNX use case: when you need a format that crosses framework and runtime boundaries with reasonable fidelity. The default intermediate when converting PyTorch → something-not-PyTorch.

## GGUF

GGUF (GGML Universal File) is the format `llama.cpp` invented for itself, then standardized across the ggml-derived ecosystem. It's a binary format designed specifically for quantized LLM weights.

A `.gguf` file contains:

- A header with metadata (model architecture, vocab size, layer count, attention parameters).
- A KV-store of metadata (tokenizer settings, special tokens, chat template).
- A list of tensors with name, shape, dtype, offset.
- The tensor data, stored in the file (often using the file as memory-mapped backing).

The quantization story is baked in: GGUF supports many quant formats (F32, F16, Q8_0, Q5_K, Q4_K_M, Q4_0, etc.) at the tensor level. Different tensors within the same model can use different quantization (typical: embedding and lm_head at higher precision; attention/FFN weights at INT4).

GGUF strengths:
- **Native quantization across many bit-widths.** The format was designed around quantized inference; conversions don't have to twist into ONNX's QDQ patterns.
- **Memory-mappable.** llama.cpp mmaps the file directly; no separate weight-loading step.
- **Self-contained.** Tokenizer, chat template, all metadata in one file.
- **Stable.** The format has been the LLM standard since ~2023.

GGUF limitations:
- **LLM-specific.** Architecturally biased toward transformer LLMs. Not the right format for vision/audio.
- **Implicit graph.** The graph isn't stored as nodes-and-edges; it's implicit in the architecture name + tensor names. Loading a GGUF file means knowing how Llama-architecture / Mistral-architecture / etc. work. Adding a new architecture means changing the runtime, not just the file.
- **Single-runtime-dominated.** llama.cpp / ggml are the primary GGUF consumers. MLX recently added GGUF read support; other runtimes are catching up.

GGUF use case: quantized LLM deployment via llama.cpp or MLX. The lingua franca of on-device LLM weights in 2026.

## Core ML

Core ML is Apple's model format. Two variants:

- `.mlmodel`: the older, single-file flat-buffer format. Supports basic ML models (linear, neural nets, tree-based).
- `.mlpackage`: a directory bundle (since Core ML 5, ~2021) containing a `Manifest.json`, `Data/com.apple.CoreML/Model.mlmodel`, `Data/com.apple.CoreML/weights/*` (separated weight files), and metadata. The current standard.

The Manifest points at one or more "specifications" — different versions of the model targeting different compute units or precisions. A single `.mlpackage` can contain a FP16 spec and an INT8 spec, with the runtime picking based on the chosen compute units at load time.

Core ML strengths:
- **Tightly integrated with Apple's compiler.** Loading a `.mlpackage` triggers a compilation step that produces an ANE-optimized binary; the compiler knows about Apple-specific ops and tile sizes.
- **Compute-unit targeting.** Native concept; you ask for `.all` or `.cpuAndGPU` and Core ML routes per op.
- **Best path to the ANE.** The only way to access the ANE in production.

Core ML limitations:
- **Apple-only.** Doesn't exist on Android or Linux.
- **Conversion friction.** Producing a Core ML file from PyTorch requires `coremltools`; some PyTorch ops don't have clean Core ML equivalents and need workarounds.
- **Versioning quirks.** Each macOS / iOS version supports a specific minimum Core ML spec version; deploying to older OS versions requires producing an older-spec model.

Core ML use case: Apple deployment, especially vision/audio with ANE acceleration.

## LiteRT (TFLite)

LiteRT (rebranded from TensorFlow Lite in 2024) is Google's mobile model format. A `.tflite` file is a FlatBuffer-serialized model — flatbuffers chosen specifically for zero-copy mmap loading.

A `.tflite` file contains:
- The graph (nodes, ops, tensors).
- The weights, inline or via `tflite-buffer` files.
- Quantization metadata per-tensor (TFLite has a rich quantization model with per-channel scales, per-tensor scales, asymmetric quantization).
- An "op resolver" hint that tells the runtime which op implementations to use.

LiteRT strengths:
- **Tiny binary footprint.** The LiteRT runtime is a few MB.
- **Quantization-native.** First-class support for INT8 (per-tensor and per-axis), with QAT integration via TF's tooling.
- **Mature delegate model.** GPU delegate, NNAPI delegate, Hexagon delegate, Core ML delegate (yes, LiteRT can use Core ML as an execution backend on Apple platforms).
- **Wide Android adoption.** The default ML runtime in many Android apps.

LiteRT limitations:
- **TensorFlow-shaped graph IR.** PyTorch models convert via ONNX → TF → LiteRT, which is multi-step and lossy.
- **Op coverage.** Less than ONNX's catalog; some recent ops require custom-op implementations.
- **LLM story is weak.** LiteRT's LLM path exists (via MediaPipe Tasks LlmInference) but lags llama.cpp on most metrics.

LiteRT use case: Android mobile deployment, especially vision/audio models. Has Apple support via Core ML delegate but Core ML is usually more direct on Apple.

## ExecuTorch

ExecuTorch (`.pte` files) is Meta's edge format, designed as PyTorch's native edge story. The `.pte` (PyTorch Edge) format is a binary representation of a PyTorch program, post-`torch.export` and post-quantization, with backend-specific code generation already applied.

A `.pte` file contains:
- The graph in PyTorch's Edge IR.
- Weights, typically quantized.
- Per-region backend delegate metadata (which parts of the graph go to XNNPACK, Core ML, QNN, etc.).
- A small "program" that the ExecuTorch runtime executes.

ExecuTorch strengths:
- **Direct PyTorch path.** No ONNX intermediate. `torch.export` + `to_edge` + `to_backend` produces a `.pte` file in one toolchain.
- **Backend delegation.** A single `.pte` can have regions targeting XNNPACK (CPU), Core ML (ANE/GPU), QNN (Hexagon), etc. The runtime composes these.
- **Production scale at Meta.** Powers WhatsApp, Instagram, Quest ML inference. Real-world battle-tested at significant scale.

ExecuTorch limitations:
- **Newer than the others.** First stable release in early 2024. Some rough edges; some backends not yet production-ready.
- **PyTorch-shaped.** Models from TF, JAX need to go through PyTorch first, which is awkward.
- **Tooling ergonomics.** The convert-and-deploy pipeline has more steps than e.g. Core ML's.

ExecuTorch use case: PyTorch models deployed cross-platform (iOS + Android) where you want to stay in PyTorch's ecosystem from training through deployment.

## The MLX-native format

For completeness — covered in Module 4: MLX's native format is a custom binary that stores per-tensor weights with per-group quantization metadata. It's not interchange-format; it's the file MLX writes after `mlx_lm.convert`. You can read it back with `mlx-lm` and that's about it. The benefit is that MLX's loader is fast and the layout matches MLX's runtime expectations exactly.

## Conversion paths and what's lost

A simplified map of common conversion paths and the typical issues:

```
PyTorch ──► ONNX ──► ORT/LiteRT/Core ML (etc.)
            │
            └─ losses: opset gaps, custom ops, dynamic shapes

PyTorch ──► Core ML (via coremltools)
            │
            └─ losses: ANE-incompatible ops, dtype precision

PyTorch ──► ExecuTorch (.pte) ──► XNNPACK/Core ML/QNN backends
            │
            └─ losses: backend coverage gaps; some ops force CPU

HF Transformers ──► GGUF (via convert_hf_to_gguf.py)
            │
            └─ losses: only LLM architectures supported; chat
              templates manually configured

TF/Keras ──► LiteRT (native)
            │
            └─ losses: minimal; this is the smoothest path

PyTorch ──► HF Optimum ──► ONNX ──► LiteRT (via tf2lite)
            │
            └─ losses: two conversions, accumulated; only when
              you really must
```

The general rule: minimize the number of conversion steps. Each conversion is an opportunity for ops to be approximated, intermediate precisions to change, or shapes to be specialized.

Common conversion failure modes:

- **Unsupported op**. The source has an op the target doesn't. Fix: decompose the op manually in the source before exporting, or write a custom op for the target.
- **Dynamic shape unsupported**. The source allows variable shape (e.g., variable batch size); the target expects fixed shapes. Fix: specify a fixed shape at conversion time, or use the target's "dynamic input" mechanism if it has one.
- **Quantization scheme mismatch**. The source uses per-group quantization; the target only supports per-channel. Fix: re-quantize for the target.
- **Numerical drift**. The conversion preserves the graph but uses a different precision for intermediates; output diverges from the source by a fraction of a percent. Usually harmless; occasionally meaningful for downstream code that checks for exact values.

## What you should believe after this lesson

Three sentences:

**1. The five edge model formats** — ONNX, GGUF, Core ML, LiteRT, ExecuTorch — each have a clear native home (interchange, LLMs, Apple, Android, PyTorch-edge respectively) and a different set of strengths. Picking the format that matches the deployment minimizes the conversion friction.

**2. Conversion paths accumulate lossiness**; the rule is to minimize conversion steps. Direct paths (PyTorch → Core ML, PyTorch → ExecuTorch, HF → GGUF) are more reliable than chained paths (PyTorch → ONNX → tf2lite → LiteRT).

**3. Quantization representation varies enormously across formats** — ONNX uses QDQ patterns, GGUF has native multi-format tensors, Core ML/LiteRT have per-tensor quant metadata, ExecuTorch encodes it in the Edge IR. Re-quantizing for the target format is often safer than trying to preserve the source's exact scheme.

## Hands-on (at home)

Convert a small model into multiple formats and inspect the file structures.

```python
# convert_multi_format.py
# pip install torch onnx coremltools executorch
import torch
import torch.nn as nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TinyClassifier().eval()
example = torch.randn(1, 64)

# 1. ONNX.
torch.onnx.export(model, example, "tiny.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"])
import os
print(f"ONNX: {os.path.getsize('tiny.onnx')} bytes")

# 2. Core ML (if on Apple).
try:
    import coremltools as ct
    traced = torch.jit.trace(model, example)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="input", shape=(1, 64))])
    ml.save("tiny.mlpackage")
    print(f"Core ML: tiny.mlpackage created")
except ImportError:
    print("Core ML: coremltools not installed")

# 3. ExecuTorch (.pte).
try:
    from torch.export import export
    from executorch.exir import to_edge
    exported = export(model, (example,))
    edge = to_edge(exported)
    pte = edge.to_executorch()
    with open("tiny.pte", "wb") as f:
        f.write(pte.buffer)
    print(f"ExecuTorch: {os.path.getsize('tiny.pte')} bytes")
except (ImportError, Exception) as e:
    print(f"ExecuTorch: {type(e).__name__}: {e}")

# 4. Inspect each.
import onnx
m = onnx.load("tiny.onnx")
print(f"\nONNX graph has {len(m.graph.node)} nodes:")
for n in m.graph.node:
    print(f"  {n.op_type}")
```

This produces three different files for the same trivial model. The ONNX one is human-readable when you print the graph; the others are binary and require their respective libraries to inspect. The sizes will be different — ONNX includes the graph verbose representation, Core ML includes the compiled-for-Apple metadata, ExecuTorch includes backend-specific code.

For GGUF specifically, you need an LLM (the format is LLM-focused). The conversion is via `convert_hf_to_gguf.py` from llama.cpp, covered in detail in Lesson 9.

## Further reading

- ONNX specification (github.com/onnx/onnx) — the spec is readable and the operator catalog is the single most useful ONNX reference.
- GGUF specification (in `llama.cpp/gguf.md`) — short, concrete.
- Apple's "Core ML Format Specification" — the `.mlmodel` and `.mlpackage` layouts.
- ExecuTorch documentation on `to_executorch()` and `.pte` files.
- LiteRT (TFLite) format documentation at tensorflow.org/lite/models.

Next lesson: **Core ML deep dive.** We zoom in on Core ML as the Apple deployment runtime, covering the `coremltools` conversion in depth, the compute-units system, ANE targeting, and the debugging workflow for "this should be on the ANE but it's running on CPU" issues.
