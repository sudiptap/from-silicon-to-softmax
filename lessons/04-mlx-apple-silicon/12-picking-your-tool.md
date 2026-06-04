---
title: "Lesson 12 — Picking Your Tool: MLX vs PyTorch MPS vs Core ML vs llama.cpp"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 12
tags: ["mlx", "pytorch-mps", "coreml", "llama-cpp", "decision-tree", "wrap"]
author: "Sudipta Pathak"
prerequisites: ["11-apple-neural-engine"]
---

# Lesson 12 — Picking Your Tool: MLX vs PyTorch MPS vs Core ML vs llama.cpp

## Why this lesson exists

Twelve lessons of Apple Silicon internals collapse into a single practical question: given a workload, which runtime do I reach for? This lesson is that decision, organized as a tree backed by the benchmarks and reasoning from the rest of the module.

It's also the module wrap. The "what we covered, what we skipped, mental models to carry forward, handoff to Module 5" lives here. By the end you should be able to look at any Apple-Silicon ML task and pick a runtime within a minute, with the reasoning behind the choice clear in your head.

The lesson is reading. The Hands-on is the final benchmark across all four runtimes on a single model, plus the running matmul project gets its MLX variant.

## The journey, replayed

Module 4 went hardware → frameworks → kernel writing → deployment. The arc:

- Lessons 1–3 (hardware): the M-series SoC, unified memory, the cache hierarchy. The new mental model: one memory pool, several compute units, the SLC as a free chip-wide cache. The cost model: `time = bytes / (bandwidth × contention × efficiency)`.
- Lessons 4–7 (frameworks): Metal as the low layer, MPS and MPSGraph as Apple's neural-network libraries, PyTorch MPS as the bridge from CUDA codebases, MLX as the native Apple Silicon framework with its lazy-graph design.
- Lessons 8–10 (writing fast code): custom Metal kernels integrated with MLX, KV cache strategies that exploit unified memory, the Module 3 quantization recipes applied to Apple-Silicon runtimes.
- Lesson 11 (the neural accelerator): the ANE's strengths (vision, audio) and weaknesses (LLMs, custom ops).

The synthesis: Apple Silicon is a competitive ML platform with a distinctive architecture. The four runtimes (MLX, PyTorch MPS, Core ML, llama.cpp) each have a place. Most new ML systems work on Apple Silicon defaults to MLX; the others have specific niches.

## The decision tree

```
What are you doing?
│
├── Training a model from scratch on Mac
│   ├── New project, no existing CUDA codebase  → MLX
│   └── Existing PyTorch codebase               → PyTorch MPS
│
├── Inference for a pretrained LLM
│   ├── Python / Jupyter workflow                → MLX (via mlx-lm)
│   ├── Standalone binary or non-Python env      → llama.cpp
│   └── iOS app                                  → Core ML (if model converts cleanly)
│                                                  or llama.cpp/llm.swift
│
├── Inference for a vision model
│   ├── Convolutional model (ResNet, EfficientNet, YOLO, …)
│   │                                            → Core ML with ANE
│   ├── Vision transformer or research model    → MLX
│   └── In a production iOS app                 → Core ML
│
├── Inference for an audio model (Whisper, speaker ID, etc.)
│   ├── Whisper specifically                    → WhisperKit (Core ML/ANE)
│   ├── Other audio model with conv encoder     → Core ML
│   └── Other / experimental                    → MLX
│
├── Custom kernel research / writing new ops    → Metal directly,
│                                                  called from MLX via
│                                                  mx.fast.metal_kernel
│
├── Fine-tuning an LLM (LoRA, full)
│   ├── On laptop hardware                       → mlx-lm.lora
│   └── On a Mac Studio / Pro                   → MLX or PyTorch MPS
│
└── Multi-Mac distributed inference (rare)      → exo, mlx-distributed,
                                                  or roll your own
```

A reading: the default is MLX unless there's a specific reason otherwise. The specific reasons are (a) you have a PyTorch codebase already, (b) you want a standalone binary, (c) you're deploying an iOS app, (d) you're doing vision/audio with conv-heavy architectures that fit the ANE.

## Benchmark table

Numbers from the hands-on experiments throughout this module, M3 Pro / 18 GB unified, single-stream. (Your numbers will vary by chip; the relative ordering is what's stable.)

**LLM inference, Llama 3 8B at INT4:**

| Runtime | Tokens/sec | Memory | Notes |
| ------- | ---------- | ------ | ----- |
| MLX (4-bit, mlx-lm) | 40–50 | 4 GB | The default for laptop LLMs |
| llama.cpp Q4_K_M | 40–55 | 4 GB | Often slightly ahead on chat |
| PyTorch MPS (W8A16 via bnb) | 20–30 | 8 GB | INT4 paths fragile |
| Core ML | 5–15 | varies | Conversion + ANE round-trip costs |

**Vision model inference, EfficientNet-B0 at FP16:**

| Runtime | ms/image | Notes |
| ------- | --------- | ----- |
| Core ML (ALL, ANE) | 4 | ANE wins decisively |
| Core ML (CPU+GPU) | 7 | Skipping ANE |
| MLX | 9 | No ANE access |
| PyTorch MPS | 12 | Some op gaps |

**Whisper Small encode + decode, 30s of audio:**

| Runtime | Wall-clock | Notes |
| ------- | ---------- | ----- |
| WhisperKit (Core ML, ANE) | 1.2 s | Encoder on ANE |
| MLX-Whisper | 2.1 s | All on GPU |
| PyTorch MPS | 3.5 s | Some fallbacks |

**Matmul (FP16, square, large):**

| Runtime | TFLOPs |
| ------- | ------ |
| MLX `a @ b` | 6.5 |
| PyTorch MPS | 5.9 |
| Custom MSL via MLX | 6.2 |
| `MPSMatrixMultiplication` directly | 6.4 |
| MLX `mx.fast.matmul` | 6.8 |

The matmul numbers cluster around 6 TFLOPs on M3 Pro. The framework choice doesn't dominate; the underlying hardware does. The numerical ordering can flip across MLX/MPS/etc. versions; the takeaway is "they're all in the same ballpark for matmul."

## What we covered, what we skipped

Covered in this module:

- M-series SoC layout (P/E cores, GPU, ANE, AMX, SLC, ProRes).
- Unified memory architecture and its cost model.
- The cache hierarchy and CPU/GPU bandwidth contention.
- Metal, MPS, MPSGraph as the three layers of the GPU stack.
- PyTorch MPS backend, including the silent-CPU-fallback trap.
- MLX intro (lazy evaluation, NumPy-like API, autodiff).
- MLX internals (computation graph, streams, compile, memory management).
- Custom Metal kernels integrated with MLX.
- KV cache strategies on unified memory.
- Quantization on Apple Silicon (MLX-native and GGUF).
- The Apple Neural Engine.
- This decision tree.

Skipped:

- **iOS app integration in depth.** Bundling Core ML models, signing, App Store deployment.
- **AMX programming.** Apple keeps the AMX instruction set private; the public surface is Accelerate.
- **Hopper-style FP8 paths.** Not native on M-series through M3; the path-forward post-M4 is in flux.
- **Vulkan via MoltenVK.** Vulkan-on-Metal exists but is rarely the right choice for ML on Apple Silicon.
- **Distributed inference across Macs in depth.** The `exo` and `mlx-distributed` story is emerging; full coverage belongs to Module 8.
- **C++ MLX usage.** MLX has a C++ API; Python is the dominant entry point and what we covered.

## Mental models to carry forward

Five sentences, one per module-spanning idea:

**1. There's one memory pool.** Forget device placement. The framework picks where work runs; you focus on what work you want done. The cost model is `bytes / (bandwidth × contention × efficiency)`.

**2. The ANE is for vision and audio, not LLMs.** Don't be misled by the TOPS number. The op coverage gaps make LLM-on-ANE slower than LLM-on-GPU in 2026.

**3. MLX is the new default Apple Silicon ML framework.** It's not a PyTorch replacement (PyTorch has more ecosystem); it's a from-scratch design that maps cleanly onto Apple Silicon. Pick MLX for new work; pick PyTorch MPS for porting work.

**4. Custom Metal kernels integrate cleanly into MLX**, so the "drop to a custom kernel for the last 20% of performance" pattern is viable on Apple Silicon. You don't have to abandon Python to write fast kernels; `mx.fast.metal_kernel` keeps you in the lazy-graph workflow.

**5. The quantization story from Module 3 lands cleanly on Apple Silicon.** MLX-native 4-bit and llama.cpp Q4_K_M both deliver ~35–50 tok/s on a 7B model on M3 Pro. The format choice is a workflow question, not a performance question.

## What's next

**Module 5: Mobile & Edge Runtimes.** We broaden from Apple Silicon to the wider mobile/edge ML landscape: TensorFlow Lite, ONNX Runtime, ExecuTorch (PyTorch's mobile path), Android NPUs (Qualcomm Hexagon, MediaTek APU). The skills from Module 4 transfer — the on-device mental model is the same; the specific runtimes and accelerators differ.

**Module 6: On-Device LLM Inference.** The full production-grade LLM serving on consumer devices: scheduling, batching where it makes sense, streaming tokens, integrating with apps. MLX, llama.cpp, and the mobile runtimes from Module 5 all play roles.

**Module 7: Inference from Scratch.** Builds the production inference pipeline from primitives. The MLX path is one of the targets; the patterns are framework-agnostic. This is where the Module 4 mental models compose into a working system.

The matmul project from Modules 1–3 now has its Apple Silicon home. Run the benchmark on your machine; the FP16 → INT4 quantization path through MLX (Module 4 Lesson 10) closes the same speedup gap on the Apple side that we closed on NVIDIA in Module 3's capstone. You should see something like:

- FP16 MLX matmul: ~6.5 TFLOPs.
- INT4 matmul via MLX: ~2-3× speedup on bandwidth-bound shapes.
- Llama 3 8B INT4 at ~45 tok/s on M3 Pro.

That last number — a usable LLM on a 14" MacBook Pro — is the practical capability this module exists to enable.

## Hands-on (at home)

The Module 4 capstone. Same model, four runtimes, four numbers.

```python
# module4_capstone.py
# pip install mlx mlx-lm torch transformers
import time
import torch
import mlx.core as mx
from mlx_lm import load as mlx_load, generate as mlx_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Explain unified memory in three sentences."
MAX_TOKENS = 96
MODEL_HF = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_MLX = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# 1. MLX via mlx-lm.
print("=== MLX (mlx-lm, 4-bit) ===")
model, tokenizer = mlx_load(MODEL_MLX)
mlx_generate(model, tokenizer, prompt=PROMPT, max_tokens=8, verbose=False)  # warm
t0 = time.time()
out = mlx_generate(model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False)
dt = time.time() - t0
print(f"  tokens/sec: {MAX_TOKENS / dt:.1f}")

# 2. PyTorch MPS (FP16).
print("=== PyTorch MPS (FP16) ===")
tok = AutoTokenizer.from_pretrained(MODEL_HF)
pt = AutoModelForCausalLM.from_pretrained(MODEL_HF, torch_dtype=torch.float16).eval().to('mps')
ids = tok(PROMPT, return_tensors="pt").input_ids.to('mps')
with torch.no_grad():
    pt.generate(ids, max_new_tokens=8, do_sample=False)
torch.mps.synchronize()
t0 = time.time()
with torch.no_grad():
    out = pt.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False)
torch.mps.synchronize()
dt = time.time() - t0
n_new = out.shape[1] - ids.shape[1]
print(f"  tokens/sec: {n_new / dt:.1f}")
del pt; torch.mps.empty_cache()

# 3. PyTorch CPU baseline.
print("=== PyTorch CPU (FP16) ===")
pt = AutoModelForCausalLM.from_pretrained(MODEL_HF, torch_dtype=torch.float16).eval()
ids = tok(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    pt.generate(ids, max_new_tokens=8, do_sample=False)
t0 = time.time()
with torch.no_grad():
    out = pt.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False)
dt = time.time() - t0
n_new = out.shape[1] - ids.shape[1]
print(f"  tokens/sec: {n_new / dt:.1f}")

# 4. llama.cpp: shell-out (set LLAMA_CPP_BIN to your built binary path).
import os, subprocess
LLAMA = os.environ.get("LLAMA_CPP_BIN", "./llama.cpp/main")
GGUF = os.environ.get("LLAMA_GGUF", "./gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
if os.path.exists(LLAMA) and os.path.exists(GGUF):
    print("=== llama.cpp Q4_K_M ===")
    result = subprocess.run(
        [LLAMA, "-m", GGUF, "-p", PROMPT, "-n", str(MAX_TOKENS),
         "-ngl", "999", "--temp", "0"],
        capture_output=True, text=True
    )
    # Parse "eval time" lines from llama.cpp stderr.
    for line in result.stderr.splitlines():
        if "eval time" in line and "tokens per second" in line:
            print(f"  {line.strip()}")
else:
    print(f"=== llama.cpp ===  (skipped: set LLAMA_CPP_BIN and LLAMA_GGUF)")
```

A successful run gives you the canonical Apple Silicon runtime comparison table for one model:

| Runtime | Tokens/sec |
| ------- | ---------- |
| MLX 4-bit | 45–60 (M3 Pro) |
| PyTorch MPS FP16 | 25–40 |
| PyTorch CPU FP16 | 5–10 |
| llama.cpp Q4_K_M | 40–55 |

The qualitative pattern: MLX and llama.cpp are competitive at the top; PyTorch MPS is meaningfully slower; CPU is a distant baseline. The exact numbers depend on chip variant.

For the Module 4 capstone matmul: take the running matmul project, add an `mx.matmul` variant, benchmark alongside the Modules 1–2 variants (naive C, blocked C, SIMD, threaded, CUDA tiled, Triton, MSL hand-written). On Apple Silicon you'll see MLX's matmul hit ~6 TFLOPs at large sizes — within 5% of MPSMatrixMultiplication.

## End of Module 4

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. Module 4 made the small fast model run well on Apple Silicon specifically. The combination — fast hardware + small payloads + Apple-native runtime — is a production on-device LLM inference path.

Module 5 picks up next: **Mobile & Edge Runtimes.** We move from "Apple Silicon Mac" to "any mobile / edge target": Android NPUs, ExecuTorch (PyTorch's mobile path), TensorFlow Lite, ONNX Runtime, the broader landscape of consumer-device ML. The mental models transfer; the specific tooling differs.
