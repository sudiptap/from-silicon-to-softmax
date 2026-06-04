---
title: "Lesson 5 — PyTorch's MPS Backend"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 5
tags: ["pytorch", "mps", "operator-coverage", "cpu-fallback", "porting"]
author: "Sudipta Pathak"
prerequisites: ["04-metal-mps"]
---

# Lesson 5 — PyTorch's MPS Backend

## Why this lesson exists

PyTorch has had a Metal-based GPU backend on Apple Silicon since version 1.12 (mid-2022). For someone with an existing PyTorch codebase trained on CUDA, the natural first attempt at running on a Mac is `model.to('mps')`. This often works. Sometimes it works fast. Sometimes it falls back to CPU silently and runs 10× slower than expected. Sometimes it errors out on an op that doesn't have an MPS implementation.

This lesson is the operator-coverage story, what to expect when porting, and the realistic role of PyTorch MPS in 2026: it's a competent but secondary path; MLX is the framework that natively belongs on Apple Silicon. PyTorch MPS is the bridge if you're maintaining a CUDA-first codebase and need to run on Mac too.

The lesson is reading. The Hands-on benchmarks the same model on PyTorch CPU vs PyTorch MPS vs MLX so the relative positioning is concrete.

## The PyTorch MPS architecture

PyTorch's MPS backend is C++ code that, for each PyTorch op:

1. Translates the op into one or more MPSGraph operations (or, for simpler ops, directly into MPS calls or a custom Metal kernel).
2. Caches the resulting graph so subsequent invocations with the same shape don't re-build.
3. Executes the cached graph against the actual tensor data.

The graph caching is important. The first run of an op with a new shape is slow (compiling the graph); subsequent runs are fast. This is the analog of `torch.compile`'s caching behavior — except it happens transparently per-op rather than over a whole region.

The choice of "MPSGraph vs custom Metal kernel" per op is the backend authors' call. As of 2026:

- Most matrix multiplications go through MPSGraph (or directly to the `MPSMatrixMultiplication` path).
- Element-wise ops (`+`, `*`, `relu`, etc.) are custom Metal kernels.
- Reductions (`sum`, `mean`, `max`) are custom kernels.
- Convolutions go through `MPSCNNConvolution`.
- Some recently-added ops (FlashAttention, RoPE, specialized norms) are custom Metal kernels written by the PyTorch team.

The op coverage has grown steadily since 2022 but still isn't 100%. Operators that aren't implemented for MPS automatically fall back to CPU when called — *silently*. Your model runs, but each unimplemented op causes the tensor to move from MPS-resident to CPU, the op runs on CPU, and the result moves back. This is the silent-CPU-fallback trap.

## The silent CPU fallback

To see whether your model is hitting CPU fallbacks, set the environment variable:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

This *enables* the fallback (otherwise unsupported ops would error). And then set:

```bash
TORCH_LOGS=+mps
```

or use the PyTorch profiler to see which ops are running on CPU.

A common pattern: a model uses some recently-added op (e.g., `aten::_scaled_dot_product_flash_attention` for SDPA on PyTorch 2.0+) that has MPS support, but the model's preprocessing uses `torch.fft.fft2` which doesn't have MPS support. The preprocessing falls back, copies tensors back and forth, and dominates latency.

Symptoms of CPU fallback:

- A model runs much slower on MPS than the bandwidth/compute math predicts.
- The Activity Monitor shows GPU utilization is low (~10-30%) even though MPS is "selected."
- Per-op profiling shows large gaps between GPU activity that are actually CPU work + tensor migrations.

Fixes:

- Replace the falling-back op with one that has MPS support. For example, `torch.nn.functional.fft.fft2` → use a different approach, or wrap in `@torch.compile` which may handle it.
- Move the offending preprocessing entirely to CPU (do it before the data hits the model) so there's no migration cost.
- Switch to MLX if the offending op has an MLX equivalent and the framework migration is bearable.

## What works well on PyTorch MPS

Most standard ML workloads are fine:

- **Transformer inference at small to medium scale.** A 1B-7B model with standard ops (linear, layer norm, RoPE, SDPA) runs cleanly on MPS. Throughput is 70–90% of MLX on the same model.
- **CNN inference and training.** Conv-heavy models are well-covered. PyTorch MPS for ResNet-class training is a viable workflow on Mac.
- **Small-to-medium model training.** Models that fit easily in unified memory and use standard ops train reliably on MPS. Convergence behavior matches CUDA; the loss curves look the same.

What doesn't work well:

- **Very recent transformer ops.** FlashAttention variants beyond SDPA, custom attention masks for sliding-window/sink models, MoE routing — these often hit CPU fallback or have correctness issues that aren't caught until you run.
- **Fine-grained custom kernels.** PyTorch's custom-op extension API exists for MPS but is much less mature than the CUDA equivalent. Writing a custom kernel for PyTorch MPS is significantly more work than writing one for CUDA.
- **Very large models.** A 70B-parameter model in FP16 (140 GB) doesn't fit anywhere on Apple Silicon directly; the path is INT4 + a runtime that supports it. PyTorch MPS's INT4 inference paths exist (via `bitsandbytes` workarounds or HF's quanto, both fragile on Mac) but are not the smooth deployment story MLX or llama.cpp offer.
- **Distributed training.** PyTorch DDP doesn't have a good Mac story. If you want multi-Mac distributed inference, MLX's distributed primitives or `exo` are more credible paths.

## Performance: PyTorch MPS vs MLX vs CPU

The typical ratios on a Llama-class workload at INT4 / INT8 / FP16, M3 Pro:

- **CPU (PyTorch CPU):** baseline. Reasonable for very small models (<500M params); painful above 1B.
- **PyTorch MPS:** 5–15× faster than CPU on standard transformer inference. Slow when CPU fallbacks bite.
- **MLX:** 1.1–1.4× faster than PyTorch MPS on typical LLM workloads. Wins more on quantized formats and custom attention.
- **llama.cpp (GGUF Q4_K_M):** comparable to MLX, often slightly faster on the most common workload (single-stream chat at 7B).

So MLX vs PyTorch MPS isn't a 2× win — it's a ~20% win plus better quantization support plus less risk of silent CPU fallback. The choice is mostly determined by what your codebase already uses.

## When PyTorch MPS is the right answer

The case for PyTorch MPS:

- You have an existing PyTorch model trained on CUDA and need to run inference on Mac. `model.to('mps')` is one line; rewriting in MLX is a project.
- You're training a model that needs to also run on CUDA in production. PyTorch MPS lets you debug on Mac with the same code that will run on the H100 later.
- You're using PyTorch ecosystem libraries (transformers, diffusers, accelerate) and don't want to port them.

The case against:

- You're starting fresh with an Apple Silicon target. MLX is more idiomatic, has fewer surprises, and tracks Apple's hardware capabilities more aggressively.
- You're shipping a Mac app. Core ML or MLX-via-Swift integrates cleaner than PyTorch.
- You're at the bleeding edge of LLM inference (long context, custom attention, novel quantization). MLX moves faster on these than PyTorch MPS.

The realistic 2026 picture: PyTorch MPS exists, is maintained, and is the right answer for the PyTorch-codebase-meets-Mac-laptop case. For new Apple-Silicon ML work, MLX is the default.

## What you should believe after this lesson

Three sentences:

**1. PyTorch's MPS backend works well for standard transformer and CNN workloads but has a silent CPU-fallback trap** — operators it doesn't implement fall back transparently to CPU, often making the model 5–10× slower than expected without any error. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` and profile to find them.

**2. PyTorch MPS performance is 70–90% of MLX on equivalent workloads**, with the gap larger on quantized formats and custom attention variants. The 10–30% gap is mostly because PyTorch dispatches through MPSGraph for many ops while MLX has direct custom Metal kernels.

**3. The choice between PyTorch MPS and MLX is mostly an existing-codebase question** — use PyTorch MPS if you're porting an existing PyTorch model to Mac; use MLX if you're starting fresh with an Apple Silicon target.

## Hands-on (at home)

Benchmark a Llama-class model on PyTorch CPU vs MPS vs MLX. Use a small model (Llama 3.2 1B or Qwen 2.5 0.5B) so the comparison fits on every machine.

```python
# bench_pytorch_mps_vs_mlx.py
# pip install torch mlx mlx-lm transformers
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
prompt = "Explain how a transformer attention layer works in two sentences."
ids = tok(prompt, return_tensors="pt").input_ids
max_new = 96

def bench_pt(device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).eval().to(device)
    ids_d = ids.to(device)
    # warm
    with torch.no_grad():
        model.generate(ids_d, max_new_tokens=8, do_sample=False)
    if device == 'mps': torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids_d, max_new_tokens=max_new, do_sample=False)
    if device == 'mps': torch.mps.synchronize()
    dt = time.time() - t0
    n_new = out.shape[1] - ids_d.shape[1]
    del model
    return n_new / dt

def bench_mlx():
    from mlx_lm import load, generate as mlx_generate
    model, tokenizer = load(model_id)
    # warm
    mlx_generate(model, tokenizer, prompt=prompt, max_tokens=8, verbose=False)
    t0 = time.time()
    out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new, verbose=False)
    dt = time.time() - t0
    n_new = len(tokenizer.encode(out)) - ids.shape[1]
    return n_new / dt

print(f"PyTorch CPU:   {bench_pt('cpu'):.1f} tok/s")
print(f"PyTorch MPS:   {bench_pt('mps'):.1f} tok/s")
print(f"MLX (FP16):    {bench_mlx():.1f} tok/s")
```

Expected on M3 Pro:
- PyTorch CPU: 5–10 tok/s.
- PyTorch MPS: 40–80 tok/s.
- MLX (FP16): 50–100 tok/s, ~20% faster than PyTorch MPS.

The relative ordering is the lesson: MPS beats CPU by ~10×; MLX beats MPS by ~20%. On a quantized model (Module 4 Lesson 10 covers this), the MLX advantage widens because MLX has better INT4 paths.

Part 2 — see a CPU fallback explicitly.

```python
# cpu_fallback_demo.py
import os, torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

x = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
# torch.linalg.eigh has spotty MPS support — this often falls back to CPU.
try:
    eig = torch.linalg.eigh(x @ x.T)
    print("eigh ran on MPS (or fell back to CPU silently — check Activity Monitor)")
except Exception as e:
    print(f"eigh errored: {e}")
```

The exact ops that fall back change with PyTorch versions. The point is: an unsupported op doesn't crash; it migrates the tensor to CPU, runs, and migrates back. The migration is what kills latency.

## Further reading

- PyTorch's MPS backend docs (pytorch.org) — the official compatibility matrix.
- The `pytorch/pytorch` GitHub repo's `aten/src/ATen/native/mps` directory — the source for every MPS-implemented op.
- "PyTorch on Apple Silicon" blog posts from the PyTorch team — episodic but useful for major-version updates.
- The `mps_fallback_ops_summary.py` script in the PyTorch test suite — generates the list of currently-falling-back ops.

Next lesson: **MLX intro.** What MLX is, what makes it different from PyTorch MPS at a design level (lazy evaluation, unified-memory native, NumPy-like API), and when it's the right choice. After this lesson the next one goes deep on MLX's internals.
