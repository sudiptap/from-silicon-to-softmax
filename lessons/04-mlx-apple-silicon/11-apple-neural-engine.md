---
title: "Lesson 11 — The Apple Neural Engine"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 11
tags: ["ane", "neural-engine", "coreml", "vision", "whisper", "power"]
author: "Sudipta Pathak"
prerequisites: ["10-quantization-apple-silicon"]
---

# Lesson 11 — The Apple Neural Engine

## Why this lesson exists

The ANE (Apple Neural Engine) is the third compute surface on M-series alongside the CPU and GPU. Apple's marketing makes it sound like the obvious destination for any ML workload — "16-core Neural Engine, 38 TOPS" on M3 Pro. The reality is more nuanced. The ANE is great at some workloads and inaccessible (or counterproductive) for others. Most LLM inference in 2026 still runs on the GPU even though, in principle, the ANE is more power-efficient.

This lesson is the realistic ANE picture: what it is at the hardware level, what it can run, the Core ML bridge that's the only way to reach it, and the latency-vs-flexibility tradeoff that drives the deployment decision.

The lesson is short because the ANE story is short. The Hands-on runs a small vision model through Core ML's ANE path and observes the power and latency tradeoffs.

## What the ANE is

The ANE is a fixed-function accelerator block on the M-series die, separate from the CPU and GPU. It has:

- A grid of multiply-accumulate units organized for convolution and matmul workloads.
- Local SRAM for tile staging (size not officially documented; estimated at a few MB per core).
- Its own memory paths to the unified memory pool — meaning it doesn't directly compete with CPU/GPU for memory bandwidth (much).
- A small instruction set focused on the operations it supports (no general-purpose compute).

Throughput is impressive on paper: 38 TOPS at INT8 on M3-class chips. That number is comparable to a mid-range desktop GPU's INT8 throughput. The catch is what the ANE can actually run.

## What the ANE supports

The ANE handles:

- **2D convolutions** (the bulk of what makes it powerful for vision).
- **Matrix multiplications** (some shapes, with constraints).
- **Common activation functions** (ReLU, sigmoid, tanh, swish).
- **Normalization layers** (batch norm, layer norm with some constraints).
- **Element-wise operations** (add, multiply, etc.).
- **Pooling** (max, average).

What it doesn't:

- **Custom kernels.** No programmable surface. You can only run the ops Apple has implemented.
- **Arbitrary control flow.** If/else branches based on tensor values don't compile to ANE.
- **Many recently-introduced LLM ops.** Rotary position embeddings, sliding-window attention masks, MoE routing, custom attention patterns — most don't have ANE implementations.
- **FP32.** ANE is FP16 / INT8 only (with BF16 added on M4-class chips).
- **Very large activations.** Tile sizes are fixed; activations larger than ~few hundred MB get split or fall back.

The pattern: convolutional vision models map cleanly to the ANE. Standard 2017-era CNNs (ResNet, EfficientNet, MobileNet, VGG-like) run on ANE with the full throughput uplift. LLMs do not.

## The Core ML bridge

The ANE is reachable only through Core ML. Specifically:

1. You produce a Core ML model file (`.mlpackage` directory or `.mlmodel` file).
2. You load it via the Core ML runtime (Swift `MLModel` API, Python `coremltools` for inspection).
3. You specify the compute units to use: `.cpuOnly`, `.cpuAndGPU`, `.all` (includes ANE).
4. The Core ML runtime, at load time, *compiles* the model for the chosen compute units. The compiler decides per-op which unit runs it. If you allow `.all`, ANE-supported ops run on ANE; the rest run on GPU; the rest on CPU.

You don't have direct control over what runs on which unit. The compiler decides based on its model of what each unit is good at. You can hint via the compute units selector but not enforce a specific op-to-unit mapping.

Producing a Core ML model:

- From PyTorch: `coremltools.convert(your_pt_model, ...)` — covers most common architectures.
- From TensorFlow: similar.
- From ONNX: similar.
- From scratch: define the model in `coremltools`' MIL (Model Intermediate Language) DSL.

The conversion path is mature for vision and audio models. For LLMs, the path exists (you can convert Llama to Core ML) but is fragile — most LLM conversions fall back substantially to GPU or even CPU for the non-trivial ops, negating the ANE's advantage.

## When the ANE wins

The ANE-wins cases in 2026 are well-defined:

**Vision models.** Image classification, object detection, semantic segmentation, depth estimation, pose estimation — anything with a ResNet/EfficientNet/Vision-Transformer-friendly architecture runs noticeably faster *and* much lower power on the ANE than on the GPU. Examples: Apple's Vision framework (the OS-level CV API) uses the ANE under the hood.

**Audio models.** Whisper, in particular, has been ported to ANE via Core ML (Apple's official "whisperkit" project). The ANE path is significantly faster than the GPU path for batch-1 transcription on M-series, because Whisper's encoder is conv-heavy and maps cleanly. Decoder is less ANE-friendly but the encoder dominates compute.

**Stable Diffusion image generation.** Apple's Core ML Stable Diffusion port runs the U-Net's convolutions on ANE. Throughput is ~30% better than the GPU-only path on M3 Pro, at lower power.

**Always-on / background workloads.** The ANE's power efficiency matters when the model runs continuously (background noise suppression, on-screen text recognition, photo analysis). The ANE consumes ~1/5 of the GPU's power for equivalent work, which adds up over hours of background processing.

## When the ANE loses (LLMs specifically)

LLM decoding on the ANE is technically possible but rarely competitive in 2026:

- **Op coverage gaps.** Standard LLM ops (RoPE, sliding-window attention, custom causal masks) often don't have ANE implementations; Core ML routes them to GPU. The CPU/GPU/ANE back-and-forth introduces latency that wipes out the ANE's compute advantage.
- **Tile-size mismatch.** ANE's matrix-multiply primitives are tuned for batch-many shapes (vision: batch of images). Autoregressive LLM decoding is batch-1 — one token at a time. The ANE can't usefully parallelize across the time dimension; it just runs one tiny matmul per dispatch with high overhead per dispatch.
- **Kernel-launch overhead.** Each ANE dispatch has a fixed overhead. For the very-short kernels in decode (1×4096 × 4096×4096 matmul takes microseconds of compute), the overhead dominates.
- **Memory paths.** The ANE has its own paths but reads from the same DRAM as GPU/CPU. Bandwidth doesn't go up; just power efficiency.

The net result: an LLM running on the ANE via Core ML often hits 5–15 tokens/sec where the same model on GPU via MLX hits 30–50 tokens/sec. The ANE consumes less power per token, but the user-facing latency is much worse. For a chat application, the GPU wins decisively.

There are research efforts (Apple's own "ANE-friendly LLM" papers) to redesign LLM architectures to better fit the ANE — fewer custom ops, batch-friendly shapes, careful tile sizes. These are interesting but not yet mainstream. As of 2026, LLMs run on the GPU on Apple Silicon.

## The Core ML compute-units decision

If you're shipping a Core ML model in a Mac/iOS app, you pick a compute-units strategy:

- **`.cpuOnly`**: deterministic, no acceleration. For testing and for cases where you specifically need predictability.
- **`.cpuAndGPU`**: includes the GPU, excludes the ANE. Good for ops that ANE doesn't handle well; predictable performance.
- **`.all`**: include everything. The runtime picks per-op. Usually fastest *and* most power-efficient for vision/audio; sometimes slower for LLMs.

For vision/audio, default to `.all`. For LLMs, `.cpuAndGPU` is often safer until you've measured the specific model on the specific hardware.

## Power matters more than you'd think

For battery-powered devices (iPhones, iPads, MacBooks on battery), power efficiency translates directly to runtime. The ANE consumes ~1–2 watts at full load; the GPU consumes 8–15 watts. A model that runs on ANE at the same throughput as on GPU lets you do 5× as much work on the same battery charge. For background workloads (the photo library indexer, on-device search, dictation) this is the whole reason the ANE exists.

For plugged-in laptop and desktop work, the power difference matters less — you're not constrained by battery. The throughput advantage on vision (~30%) and on Whisper (~2×) still applies, but for LLMs (where ANE is slower), the choice is GPU.

## What you should believe after this lesson

Three sentences:

**1. The ANE is a fixed-function neural-network accelerator** that excels at convolution-heavy workloads (vision, audio encoders) and is essentially the only sensible deployment target for those models on Apple Silicon. For LLMs, op coverage gaps and batch-1 inefficiency mean the GPU wins despite the ANE's higher power efficiency.

**2. The ANE is reachable only through Core ML**, which compiles a model for chosen compute units and decides per-op which unit runs it. You don't have direct ANE programming; you ship a Core ML model and let the runtime route.

**3. The compute-units choice for production deployment is workload-dependent** — `.all` for vision/audio (let the ANE shine), `.cpuAndGPU` for LLMs (skip the ANE round-trip that hurts more than it helps). Power-efficient background workloads benefit from the ANE; user-facing low-latency LLM chat does not.

## Hands-on (at home)

Run a small vision model through Core ML's ANE path and observe the difference.

```python
# coreml_ane_demo.py
# pip install coremltools torch torchvision
import coremltools as ct
import torch
import torchvision.models as models
import time
import numpy as np

# Pick a vision model that's ANE-friendly.
pt_model = models.efficientnet_b0(weights="DEFAULT").eval()

# Convert to Core ML.
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(pt_model, example)
ml_model_all = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.ALL,  # include ANE
)
ml_model_gpu = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
)
ml_model_cpu = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

def bench(m, name):
    # Warm.
    for _ in range(5): m.predict({"image": x})
    t0 = time.time()
    for _ in range(100): m.predict({"image": x})
    dt = (time.time() - t0) / 100
    print(f"{name:25s}: {dt*1000:6.2f} ms/iter")

bench(ml_model_cpu, "Core ML, CPU only")
bench(ml_model_gpu, "Core ML, CPU + GPU")
bench(ml_model_all, "Core ML, CPU + GPU + ANE")
```

Expected on M3 Pro:
- CPU only: ~30–60 ms/iter.
- CPU + GPU: ~5–10 ms/iter.
- CPU + GPU + ANE: ~3–6 ms/iter (best).

The ANE path is the fastest by a meaningful margin, and (you can verify with `powermetrics` or `asitop`) consumes less power.

Now try the same with an LLM:

```bash
# Convert a small LLM to Core ML and run via ANE.
# This is non-trivial; many people use the apple/ml-stable-diffusion conversion
# scripts as a reference. The shorter version is to use llama-models-coreml
# from the community, which provides pre-converted small LLMs.
```

You'll observe that an LLM via Core ML/ANE runs significantly slower than the same model in mlx-lm. That's the LLM-on-ANE story.

## Further reading

- Apple's Core ML documentation (developer.apple.com/documentation/coreml) — the official entry point.
- `coremltools` documentation — the Python conversion library.
- "Deploying Transformers on the Apple Neural Engine" (Apple ML research blog) — Apple's own perspective on the LLM-on-ANE problem.
- "WhisperKit" project — the canonical example of an audio model carefully ported to the ANE.
- "Stable Diffusion on Apple Silicon" (Apple ML research blog) — the canonical vision generative model ported to ANE.

Next lesson: **Picking your tool — MLX vs PyTorch MPS vs Core ML vs llama.cpp.** The module wrap. A practical decision tree backed by the benchmarks accumulated through this module, organized around "given my workload, which runtime is the right answer" and the handoff to Module 5 (Mobile & Edge Runtimes).
