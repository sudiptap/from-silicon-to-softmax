---
title: "Lesson 8 — FlashAttention v1 → v2 → v3"
date: "2026-06-04"
module: "inference-from-scratch"
order: 8
tags: ["flashattention", "io-aware", "softmax-tiling", "kernel", "hopper", "wgmma"]
author: "Sudipta Pathak"
prerequisites: ["07-cross-attention"]
---

# Lesson 8 — FlashAttention v1 → v2 → v3

## Why this lesson exists

Standard attention computes `softmax(Q K^T / sqrt(d)) V` by *materializing* the full `[N, N]` score matrix in memory, applying softmax, then doing the second matmul. For long sequences, this matrix dominates memory: a 32K-context attention at FP16 with 32 heads is 32 × 32K × 32K × 2 = 64 GB. The math says 64 GB; you have ≤80 GB.

FlashAttention (Dao et al, 2022; "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") solves this by *never materializing* the full score matrix. It tiles the computation so each tile fits in fast on-chip SRAM, computes the softmax incrementally (the "online softmax" trick), and produces the same exact result as standard attention with `O(N)` extra memory instead of `O(N²)`.

The win compounds: long-context inference is now feasible; the kernel runs faster than the standard approach because it doesn't waste bandwidth moving the `[N, N]` matrix to/from HBM; the algorithm is the foundation of every modern attention implementation in production.

V2 (2023) improved parallelism. V3 (2024) added Hopper-specific async paths. This lesson is the algorithm, the IO-awareness mental model, and the version progression.

The lesson is reading. The Hands-on uses FlashAttention via PyTorch's SDPA and measures the memory and throughput vs naive attention.

## The naive attention problem

Standard scaled-dot-product attention:

```python
scores = Q @ K.T / sqrt(d)              # [N, N]; this is the problem
weights = softmax(scores, dim=-1)        # [N, N]
output = weights @ V                     # [N, d]
```

For `N = 32K`, the `[N, N]` matrix is `32K × 32K × 4 bytes = 4 GB` per head per layer (FP32) — and you typically need it in SRAM-friendly form. Most GPUs don't have enough on-chip memory; the matrix lives in HBM, requiring slow HBM reads per element. The kernel is bandwidth-bound on the `O(N²)` matrix movement, not on the compute.

The roofline arithmetic-intensity analysis:
- FLOPs: `4 × N² × d` for the two matmuls.
- HBM traffic: `O(N²)` for the score matrix + `O(N × d)` for Q, K, V, O.
- Arithmetic intensity at long N: dominated by score traffic, giving `O(d)` FLOPs per byte.

A typical 4090 has 165 TFLOPs FP16 / 1 TB/s = 165 FLOPs/byte to be compute-bound. For `d=128`, the naive kernel has `≈ d = 128` FLOPs/byte — just barely compute-bound. The actual implementation is much worse because the score matrix needs to round-trip HBM: write, read for softmax, read for output matmul. Effective intensity drops to ~`d/3 ≈ 40`. Solidly memory-bound. The kernel runs at a fraction of peak.

## The FlashAttention v1 idea

Two interacting tricks:

**1. Tile the computation.** Split Q into blocks of `B_q` rows; split K, V into blocks of `B_k` rows. The output of the attention for a Q block can be computed by iterating over all K, V blocks and accumulating partial results.

**2. Online softmax.** The softmax denominator needs to know the sum of `exp(score)` over all positions. Normally you'd compute all scores first, then sum, then divide. The online softmax computes the sum incrementally as you process tiles, updating the partial sum and the partial output together using a numerically-stable formula.

The online softmax update:
- For each K, V block, compute the partial scores `S_block = Q_block @ K_block.T`.
- Track the running max `m_running` and running sum `l_running`.
- For a new block with max `m_new`:
  - Update `m_global = max(m_running, m_new)`.
  - Rescale the running sum: `l_running = l_running × exp(m_running - m_global)`.
  - Rescale the running output: `o_running = o_running × exp(m_running - m_global)`.
  - Add this block's contribution: `l_running += sum(exp(S_block - m_global))`, `o_running += exp(S_block - m_global) @ V_block`.
- After all blocks: `output = o_running / l_running`.

The math is numerically equivalent to the standard softmax-then-multiply path. The implementation reads each K, V block once, keeps the tile in SRAM, and never writes the full `[N, N]` matrix to HBM.

Memory traffic: `O(N × d)` for Q, K, V, O. The `N²` term disappears.

## The wins, quantified

For FlashAttention v1 on an A100 with FP16, N=4096, d=64:
- HBM bytes accessed: `O(N × d)` vs standard's `O(N²)`. ~10× reduction.
- Wall-clock: 2-4× faster for medium N; 7-10× faster for long N (where the standard kernel becomes severely memory-bound).
- Peak memory: drops from `O(N²)` (the score matrix) to `O(N × d)`. For N=32K, this is the difference between OOM and fitting.

The exact-equivalence property: FlashAttention produces *bit-identical* results (or within last-bit floating-point rounding) to standard attention. Not approximate; not a different algorithm. Same math, better implementation.

## FlashAttention v2

V1 had a parallelism limitation: it parallelized across Q blocks, but within a Q block the K/V iteration was sequential. For modern GPUs with many SMs, this left compute on the table at moderate sequence lengths.

V2 (Dao, 2023) restructures the parallelism:
- Parallelize across the *sequence-length* dimension of Q.
- For each Q block, multiple SMs can work in parallel on different K, V blocks.
- Better warp-level work distribution within each SM.

Throughput improvement: 1.5-2× over v1 on A100. Same correctness, same memory profile, better hardware utilization.

V2 is the default FlashAttention in PyTorch 2.x's `scaled_dot_product_attention` and in most production attention kernels through 2023-2024.

## FlashAttention v3

V3 (Shah et al, 2024) targets Hopper (H100) specifically:
- **Asynchronous tensor cores**: H100's `wgmma` instructions let the tensor cores run *while the kernel issues other work*. V3 overlaps the GEMM (matmul) operations with the softmax computation, hiding latency.
- **TMA (Tensor Memory Accelerator)**: Hopper's dedicated DMA engine for moving tensor tiles between HBM and SRAM. V3 uses TMA explicitly to overlap data movement with compute.
- **FP8 support**: Hopper's FP8 tensor cores can do attention at FP8, doubling throughput over FP16 — if the precision works for your model (V3 has both FP8 and BF16 paths).

Throughput: 1.5-2× over v2 on H100. Combined with v2's gains over v1, the FlashAttention story is ~6× from the original 2022 version through v3.

V3 also handles GQA and MLA cleanly (the K/V broadcasting from Lessons 4 and 5) without separate code paths.

## Why this is a permanent fixture

FlashAttention is the production attention kernel because:
1. It's exact (no quality loss).
2. It's faster than standard attention at every N.
3. It uses less memory at every N (and the memory advantage compounds at long context).
4. It's available in every major runtime (PyTorch, MLX, llama.cpp's port, vLLM, ExecuTorch).

The original 2022 paper changed the conversation about attention's `O(N²)` memory cost — it shifted from a hard architectural constraint to "use FlashAttention." Long-context models (32K, 128K, 1M) became feasible because the memory cost was now linear in N. The training cost of long-context LLMs dropped commensurately.

## The lesson it teaches

FlashAttention's broader lesson for ML systems: **the algorithm and the kernel are not separable.** The naive attention algorithm has cost `O(N²)` *as written*; FlashAttention has the same mathematical complexity but vastly better runtime cost because the *data movement* dominates real performance, not the FLOP count.

This is the IO-aware view: kernels should be analyzed by their HBM traffic, not their FLOP count. The roofline model (Module 1 Lesson 1; Module 3 Lesson 1) is the formal framework.

For people building inference engines: assume FlashAttention is available; design around it; don't accept any other attention path unless you have a specific reason.

## What you should believe after this lesson

Three sentences:

**1. FlashAttention computes `softmax(QK^T)V` without materializing the `[N, N]` score matrix** — via tiling and online softmax. The result is bit-equivalent to standard attention with `O(N)` memory instead of `O(N²)`, and 2-10× faster because of reduced HBM traffic.

**2. V2 (2023) improved parallelism within Q blocks; V3 (2024) added Hopper-specific async paths (wgmma, TMA, FP8)** for another ~3× cumulative throughput improvement on H100. The algorithm has been continuously refined for hardware capabilities.

**3. FlashAttention is the production attention kernel** — every major runtime uses it; the original `O(N²)` memory constraint that defined attention's long-context limit no longer exists. Building anything else is reserved for specific kernel-research contexts.

## Hands-on (at home)

Use PyTorch's FlashAttention via SDPA and compare to manual attention.

```python
# flash_attention_demo.py
import torch
import torch.nn.functional as F
import time
import math

# Conditional on having a recent enough PyTorch + CUDA.
device = 'cuda' if torch.cuda.is_available() else 'mps'
dtype = torch.float16

def naive_attention(q, k, v):
    # q, k, v: [B, h, N, d]
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)
    return weights @ v

def flash_attention(q, k, v):
    # SDPA uses FlashAttention v2/v3 internally when shapes and dtypes allow.
    return F.scaled_dot_product_attention(q, k, v)

B, h, N, d = 1, 8, 4096, 64
q = torch.randn(B, h, N, d, device=device, dtype=dtype)
k = torch.randn(B, h, N, d, device=device, dtype=dtype)
v = torch.randn(B, h, N, d, device=device, dtype=dtype)

# Warm and bench naive.
for _ in range(3): naive_attention(q, k, v)
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
t0 = time.time()
for _ in range(10): naive_attention(q, k, v)
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
print(f"naive attention: {(time.time()-t0)*100:.2f} ms/iter")

# Warm and bench flash.
for _ in range(3): flash_attention(q, k, v)
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
t0 = time.time()
for _ in range(10): flash_attention(q, k, v)
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
print(f"flash attention: {(time.time()-t0)*100:.2f} ms/iter")

# Verify equivalence.
a = naive_attention(q, k, v)
b = flash_attention(q, k, v)
print(f"max abs diff: {(a - b).abs().max().item():.2e}")
```

You should see FlashAttention 2-5× faster than naive at `N=4096` and the gap widens at longer N. The difference between the outputs should be at the level of floating-point rounding (~1e-3 in FP16).

For the memory-explosion demonstration, try `N=8192` or `N=16K` with the naive implementation — depending on your GPU you may OOM, while FlashAttention happily runs.

## Further reading

- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al, 2022) — the original paper. Section 3 has the algorithm pseudocode.
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023).
- "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (Shah et al, 2024).
- "Online Softmax" (Milakov & Gimelshein, 2018) — the original online softmax algorithm that FlashAttention uses.
- Module 2 Lesson 10 "FlashAttention Demystified" — this curriculum's earlier coverage of the kernel-implementation details.

Next lesson: **Linear and sub-quadratic attention.** A different attack on `O(N²)` — instead of optimizing the implementation, change the algorithm. Performer (random features) and Linformer (low-rank projection) trade some expressivity for linear complexity. The lesson covers what they achieve, what they lose, and why they haven't displaced FlashAttention.
