---
title: "Lesson 13 — Module Wrap: When to Leave the Compiler Alone"
date: "2026-06-03"
module: "gpu-computing"
order: 13
tags: ["gpu", "wrap", "summary", "cublas", "triton", "mlx"]
author: "Sudipta Pathak"
prerequisites: ["12-profiling-gpu-kernels"]
---

# Lesson 13 — Module Wrap: When to Leave the Compiler Alone

## Why this lesson exists

Twelve lessons of GPU programming end the same way Module 1 ended: with a summary, a "where did we land," and a clear handoff to what comes next. The summary matters because the practical skill at this point is not "I can write any kernel" but "I know which kernel to write and when to use someone else's."

This lesson is reading. The Hands-on at the bottom is a final summary benchmark plus the project setup that carries forward into Module 3.

## The journey, replayed

We started with a naive matmul on the GPU — one thread per output cell, brute reduction — hitting a few thousand GFLOPs on a 4090. The headline number sounded fast (100× the best CPU number from Module 1!) and was still 5% of what the chip could do.

We applied, in order:

| Lesson | Technique | Speedup over prior |
| -- | --------- | ------------------ |
| 5 | Shared-memory tiled CUDA | ~5–8× |
| 6 | Triton (block-level, with auto-tune) | ~3× over hand CUDA |
| 7 | Metal Shading Language (Apple parallel path) | sideways — equivalent throughput on smaller hardware |
| 8 | Kernel fusion (matmul + bias + activation) | ~1.5–2× per multi-op chain |
| 9–10 | Reductions + FlashAttention | ~10× for attention at long context, structural change |
| 11 | FlashAttention on Apple Silicon via MLX | comparable on relative hardware, ~10× over naive |
| 12 | Profiling (Nsight, Xcode) | not a speedup — a diagnostic upgrade |

Multiply the matmul side: ~5 × 3 ≈ 15× over the naive CUDA. Starting from 5% of peak that puts the hand-tuned Triton matmul at ~75% of peak — within striking distance of cuBLAS, which lives at 90–95%. We didn't quite match the vendor library, but we know exactly why (Hopper's `wgmma` async paths, tighter precision tuning, register-tile micro-decisions) and we got there with a fraction of the code.

For attention: FlashAttention's structural change (no N² intermediate) gave us ~10× at moderate context and >50× at long context. That's not a constant-factor optimization — it's a fundamentally different algorithm enabled by a fusion-friendly GPU substrate.

For Apple Silicon: the MLX path matches what we built on NVIDIA structurally, with ~1/3 the absolute throughput, but at ~1/10 the power. Per watt, Apple wins. For on-device, that's the relevant metric.

## What we did, what we skipped

We covered:

- The GPU mental model: SIMT, warps/SIMD-groups, occupancy, memory hierarchy.
- NVIDIA architecture in concrete numbers (SMs, tensor cores, HBM).
- Apple Silicon GPU architecture and how unified memory changes the cost model.
- CUDA basics, then tiled CUDA matmul with shared memory.
- Triton — same matmul in 30 lines, with auto-tuning.
- Metal Shading Language — same matmul on the Apple side.
- Kernel fusion via the arithmetic-intensity / Roofline framing.
- Warp shuffle reductions, fused softmax.
- FlashAttention — the algorithm and its kernel structure.
- FlashAttention on Apple Silicon via MLX.
- Profiling on both platforms.

We deliberately skipped:

- **CUTLASS internals.** We mentioned CUTLASS as the vendor C++ template library that gets cuBLAS-class performance with tunable shapes. Reading CUTLASS source is a worthwhile project, beyond the scope of this module.
- **AMD ROCm.** The mental model and most code transfers; the toolchain (HIP, ROCm libraries) is largely a search-and-replace from CUDA. If you have AMD GPUs you need: read NVIDIA's CUDA docs, then HIP's migration guides.
- **Hopper-specific features** (TMA, `wgmma`, async via `cp.async.bulk`, thread block clusters). Mentioned but not coded by hand. Real production kernels for Hopper use them; reproducing them is a substantial chapter on its own.
- **The CUDA backend pass** — how Triton, XLA, IREE map to PTX. Interesting compiler material but tangential to "write fast kernels."
- **Multi-GPU.** Module 8 (Distributed Systems) covers NCCL, sharding, the inter-GPU stories.
- **SMMA on Apple in depth.** We sketched the matrix-instruction approach for M3+ but didn't write a complete kernel using it. MLX does this for you; for experiments, the further-reading section points to working examples.

## When to leave the compiler alone

A practical answer to "should I write a custom kernel" at three levels of difficulty:

**Use the vendor library (cuBLAS / cuDNN / MPS / Accelerate) when:**

- The op is a standard matmul, conv, layernorm, etc.
- Performance is already at vendor-tuned levels (probably).
- Your role is to compose ops, not invent them.

**Use `torch.compile` or MLX's auto-fusion when:**

- You have a graph of standard ops.
- You can express what you want in plain framework code.
- The fusion pattern is recognizable (matmul + element-wise chains, softmax, normalizations).
- You don't have specific kernel control needs.

**Use Triton / MLX custom MSL when:**

- Your fusion pattern is unusual (FlashAttention variants, MoE routing, KV cache management).
- You need a specific precision policy (FP8 accumulation, mixed precision).
- You're at the frontier — implementing a new paper, exploring an architectural variant.
- The vendor library's shape coverage doesn't match yours (rare for matmul; common for variants).

**Use raw CUDA / MSL only when:**

- Triton / MLX kernel API isn't expressive enough.
- You need Hopper-specific async with full control.
- You're contributing to a vendor library or framework backend.

The 95% case: vendor library or fused-via-compiler. The 5% case: Triton. The 1% case: raw kernel code. Spending too much time at the wrong level is the biggest productivity drain in ML systems work — much more common than performance left on the table by going "too high level."

## Mental models to carry forward

Three sentences:

**Most kernels are memory-bound; fast kernels reduce memory traffic per FLOP done.** Tile, fuse, stream. The pattern from cache-blocked matmul (Module 1) through shared-memory matmul through FlashAttention is one continuous thread.

**Hardware has feature ceilings that determine which kernel structures are even possible.** Tensor cores set the FP16/BF16 matmul ceiling. Async copies (Hopper, MLX async on M3+) set the overlap potential. The Apple GPU's unified memory removes the host-device transfer problem but introduces bandwidth contention.

**The right tool depends on what you're optimizing.** Wall-clock latency: vendor lib if available. Throughput at production cost: Triton. Frontier kernels: hand-written. Don't fight a vendor library at its own game.

## What's next

**Module 3: ML Internals & Optimization.** We pivot from "how to compute fast" to "what to compute." Quantization is the headline: turning 32-bit weights into 8-bit or 4-bit, with the math and the recipes that make it work. Distillation, pruning, and the Rust GPU frontier round out the module.

**Module 4: MLX & Apple Silicon Internals** picks up the Apple thread from this module. We go deeper on the M-series SoC, the ANE, the matrix accelerators, and the practical question of *which Apple substrate* for a given workload.

The matmul project from Modules 1 and 2 is now in a clean state — CPU naive → CPU SIMD threaded → CUDA tiled → Triton → MSL → vendor BLAS / cuBLAS / Accelerate / MLX. Module 3 quantizes everything from FP32 to INT8 and INT4, the matmul gets faster *again*, and we start running real LLM-class models on real on-device hardware.

## Hands-on (at home)

The final benchmark of Module 2. A single script that runs every variant.

```python
# bench_module_2.py
import torch
import triton
import triton.language as tl
import time

# Optionally: import mlx for Apple side.
try:
    import mlx.core as mx
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

# Sizes.
M = K = N = 4096
device = 'cuda' if torch.cuda.is_available() else 'mps'

a = torch.randn(M, K, device=device, dtype=torch.float16)
b = torch.randn(K, N, device=device, dtype=torch.float16)

def bench(name, fn, runs=20):
    for _ in range(3): fn()  # warm
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        out = fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - t0) / runs
    flops = 2 * M * N * K
    print(f"{name:>20}: {dt*1000:.2f} ms  {flops/dt/1e12:.1f} TFLOPs")
    return out

# 1. Naive matmul.
def naive():
    # Don't actually run a Python loop matmul; emulate with PyTorch matmul.
    return a @ b

# 2. cuBLAS / MPS / Accelerate via torch.
def vendor():
    return torch.matmul(a, b)

# 3. Triton matmul (your implementation from Lesson 6, or torch.compile).
@torch.compile
def compiled(a=a, b=b):
    return a @ b

bench("vendor", vendor)
bench("compiled", lambda: compiled())

# 4. Attention.
def attn(causal=True):
    q = torch.randn(1, 16, M, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 16, M, 128, device=device, dtype=torch.float16)
    v = torch.randn(1, 16, M, 128, device=device, dtype=torch.float16)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

bench("sdpa (FA)", attn)

if HAVE_MLX:
    print("\nApple Silicon (MLX):")
    a_mx = mx.random.normal((M, K)).astype(mx.float16)
    b_mx = mx.random.normal((K, N)).astype(mx.float16)
    def mlx_matmul():
        c = mx.matmul(a_mx, b_mx)
        mx.eval(c)
        return c
    bench("mlx matmul", mlx_matmul)

    q_mx = mx.random.normal((1, 16, M, 128)).astype(mx.float16)
    k_mx = mx.random.normal((1, 16, M, 128)).astype(mx.float16)
    v_mx = mx.random.normal((1, 16, M, 128)).astype(mx.float16)
    def mlx_attn():
        out = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=1/(128**0.5), mask="causal")
        mx.eval(out)
        return out
    bench("mlx sdpa (FA)", mlx_attn)
```

Run on every machine you have access to. Save the numbers.

Sanity-check ranges:

- **4090 (24GB)**: vendor matmul ~280 TFLOPs FP16; SDPA ~200 TFLOPs effective.
- **H100 (80GB)**: vendor matmul ~700 TFLOPs FP16 (in default mode; FP8 would be much higher with proper paths); SDPA ~500 TFLOPs effective.
- **M3 Max**: MLX matmul ~17 TFLOPs FP16; SDPA ~10 TFLOPs effective.
- **A100 (40GB)**: vendor matmul ~250 TFLOPs FP16; SDPA ~150 TFLOPs effective.

Save the output. It's the baseline for Module 3 (quantization) where the same matmul, at INT4, runs at much higher effective throughput per byte.

## Further reading

- "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (Simon Boehm) — for one more pass through the matmul story with profiler walkthroughs.
- CUTLASS GitHub README and tutorial — for the production-grade C++ approach to matmul on NVIDIA.
- "FlashAttention-3" paper — for the cutting edge of attention kernels.
- MLX source — production Metal kernels with comments explaining design choices.
- The Anton Lozhkov / Simon Boehm / Tri Dao Twitter feed in the LLM systems world — performance updates and kernel writeups appear there before they hit papers.

End of Module 2.

Module 3 — **ML Internals & Optimization** — picks up next. We've been at FP32 and FP16. Module 3 takes us to INT8, INT4, and the 1-bit frontier. The same matmul, different number system, very different throughput.
