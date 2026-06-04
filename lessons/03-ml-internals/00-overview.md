---
title: "Module 3 — ML Internals & Optimization"
date: "2026-06-04"
module: "ml-internals"
order: 0
tags: ["quantization", "int8", "int4", "fp8", "awq", "gptq", "pruning", "distillation", "overview"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal", "gpu-computing"]
---

# ML Internals & Optimization

## Why this module exists

Modules 1 and 2 made matmul fast. That's half the story. The other half is making the *thing being multiplied* smaller, and making sure the smaller version still produces sensible numbers.

A modern open-weights LLM ships at ~16 GB in FP16. The same model at INT4 fits in ~4 GB and runs on a laptop. The transformation is not free — there is a recipe, a calibration step, and a fairly delicate dance between the format and the kernel that consumes it — but it is the difference between "needs an H100" and "runs on the laptop you're holding." Quantization is the single biggest practical lever in on-device and edge ML. It is also the lever that is most often misunderstood: people quote a number like "INT4" without specifying which INT4 (per-channel? per-group? what group size? AWQ-scaled? GPTQ-corrected? GGUF's Q4_K_M?), and the answers vary by 5–10 perplexity points on the same model.

This module is the recipe book. We cover the precision landscape from FP32 down to 1-bit, the math that makes a quantized matmul still equal (approximately) the FP matmul, the algorithms (AWQ, GPTQ, calibration) that pick the scales well, and the adjacent compression levers (pruning, distillation) that round out the story. We also touch the Rust-on-GPU frontier — relevant because the tooling for low-precision kernels is rapidly leaving Python.

## How this fits

Module 3 of the depth track. Modules 1 and 2 were about the *substrate*: caches, SIMD, threads, warps, shared memory, tensor cores. Module 3 is about the *payload*: the numbers being computed on. The two compose. A FlashAttention kernel in FP16 with a hand-rolled tiled CUDA matmul (from Module 2) doesn't get faster from Module 3 — it gets *cheaper to deploy*, because the weights it loads shrink by 4× or 8×, and the memory bandwidth pressure (which usually dominates on inference) drops in lockstep.

The output of this module: the ability to take a published model checkpoint, decide on a compression budget for a target machine, run the conversion, validate it, and predict the throughput implications. This is the gating skill for Modules 4–7 (Apple Silicon internals, mobile/edge runtimes, on-device LLMs, inference from scratch). You cannot run a 70B model on a laptop without it.

## The roadmap

Eleven lessons.

### The arithmetic baseline

1. **The arithmetic of neural nets: where the FLOPs actually go** — per-layer FLOP budget, the matmul-dominated reality, what compression actually shrinks (memory traffic, more than compute).

### The precision landscape

2. **FP16 vs BF16 vs FP8: the precision landscape** — IEEE 754 refresher, what each format keeps and throws away, the dynamic-range / precision tradeoff, FP8 (E4M3 vs E5M2) and MXFP, when each format is used and why.

### Integer quantization

3. **INT8 quantization: math and recipes** — symmetric vs asymmetric, per-tensor vs per-channel, scale/zero-point math, the dequant fusion pattern in matmul kernels, why INT8 is essentially free.
4. **INT4 quantization: per-channel, per-group, the formats** — group sizes, the bitpack story, GGUF Q4_0/Q4_K_M, AWQ packing, GPTQ packing, what the format actually looks like on disk.

### Picking the scales well

5. **Calibration data and quantization-aware training basics** — PTQ calibration (MinMax, MSE, percentile), why one batch matters, when QAT is worth the training cost.
6. **AWQ: activation-aware weight quantization** — the salient-channels insight, the activation-driven scaling trick, why it survives INT4 where naive quantization collapses.
7. **GPTQ: error-correcting quantization** — the OBS/OBQ heritage, the layer-by-layer Hessian update, why it's the most common workhorse in 2026.

### Other compression levers

8. **Pruning: structured vs unstructured, lottery tickets** — N:M sparsity (2:4 on NVIDIA), the unstructured fantasy vs the structured reality, why pruning is a junior partner to quantization in practice.
9. **Distillation: from logits to step-by-step** — Hinton soft-label distillation, modern step-by-step (rationale distillation, MiniLLM, on-policy variants), when distillation beats quantization and when it doesn't.

### The Rust frontier

10. **The Rust GPU frontier: cubecl, rust-gpu, Burn** — where Rust-on-GPU sits in 2026, what cubecl does that CUDA C++ doesn't, what's production-ready and what's research-ware.

### Wrap

11. **Module wrap: picking your compression budget** — the decision tree (target device → budget → format → algorithm → validation), what we covered and skipped, handoff to Module 4.

---

## What this module deliberately won't cover

- **Sparse attention algorithms.** Those are an *architecture* topic, not a *compression* topic — covered in Module 7 (Inference from Scratch) where sliding-window, sink, and routed attention belong.
- **Mixture-of-experts.** Same reasoning — MoE is architectural sparsity, covered with the rest of the MoE machinery in Module 7.
- **Mixed-precision training.** This module is inference-focused. Training-time mixed precision (gradient scaling, BF16 master weights, FP8 training paths) is its own discipline; we touch it where the formats overlap but don't go deep.
- **1-bit / ternary models.** BitNet, 1.58-bit, and friends — mentioned in passing in the precision lesson but not as a separate deep dive. The serious 1-bit work currently requires training from scratch with the format baked in, which puts it outside the post-hoc compression frame of this module.
- **Hardware INT4 paths in detail.** Hopper FP8 / Blackwell FP4 — we name them and explain the kernel surface, but the deepest tensor-core micro-code work belongs to a different curriculum.

## How to work through it

Every lesson is fully readable as prose on a phone. The hands-on sections at the end of each lesson assume:

- An NVIDIA GPU for the FP8 / INT8 / INT4 paths via PyTorch or vLLM. A 4090 is plenty; an H100 unlocks FP8 paths cleanly; a 3060/3080 still demonstrates INT8.
- An Apple Silicon Mac for the MLX / `llama.cpp` / GGUF side. M1 onward works; M3 Pro+ is comfortable for 13B-class models at INT4.
- A model checkpoint to quantize — Llama 3 8B, Qwen 2.5 7B, or Mistral 7B are the standard study subjects.

The running project from Modules 1 and 2 carries forward: the matmul kernels gain an INT8 and an INT4 variant. New this module: a full LLM (small one — TinyLlama or Phi-2) goes through the full quantize → validate → benchmark loop, end-to-end. By the end of Module 3, your laptop runs an LLM that produces sensible text and you understand every byte of the format that made it fit.

A note on tempo: this module reads faster than Module 2 because there is less novel hardware to introduce — most of the substrate is from Module 2. The density is in the *recipes* and the *failure modes*. Read the AWQ and GPTQ lessons twice; their tradeoffs come up in every subsequent on-device chapter.
