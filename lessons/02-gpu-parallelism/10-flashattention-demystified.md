---
title: "Lesson 10 — FlashAttention Demystified"
date: "2026-06-03"
module: "gpu-computing"
order: 10
tags: ["flashattention", "attention", "online-softmax", "tiling", "io-aware"]
author: "Sudipta Pathak"
prerequisites: ["09-reduction-problem"]
---

# Lesson 10 — FlashAttention Demystified

## Why this matters

FlashAttention is the most important GPU kernel of the LLM era. Before it, attention's memory cost grew as O(N²) — a 32K-context model needed gigabytes of intermediate memory per attention call, blowing past HBM and ruining throughput. After it, attention's memory cost is O(N) — the same model runs on the same hardware with the same accuracy, ten to a hundred times faster. Every modern LLM serving stack — vLLM, SGLang, TRT-LLM, MLX-LM, llama.cpp — uses FlashAttention or a derivative.

The technical content is also the cleanest possible illustration of every concept in this module so far: tiling, shared memory, kernel fusion, the online softmax reduction. If you understand FlashAttention you understand what fast GPU kernels actually look like.

This lesson is the algorithmic essence and the kernel structure. We don't reproduce the full Triton code — that's available in Tri Dao's repo and best read after this lesson — but you'll come out knowing exactly what's going on and why.

## Concept

### The attention operation, recap

Given queries Q (N × d), keys K (N × d), values V (N × d):

```
S = Q @ K^T / sqrt(d)        # (N, N) scores
P = softmax(S, dim=-1)       # (N, N) probabilities
O = P @ V                     # (N, d) output
```

For a single head; multi-head is repeated h times in parallel.

For N = 32768 and d = 128 (typical for a Llama-class head): S and P are each 32768 × 32768 = ~4 GB in FP32, ~1 GB in FP16. Per head. With 32 heads per batch on a 4090's 24 GB HBM, you can barely fit one sequence. Worse, every read of these 1-GB tensors saturates HBM bandwidth, making attention 10–100× slower than it needs to be.

### The standard fix that didn't work

The pre-FlashAttention solution: chunk the operation. Split Q into chunks, compute S for one chunk at a time, softmax, multiply by V, accumulate. This *reduces peak memory* but doesn't reduce *total memory traffic* — you still read each of K and V once per Q-chunk, and the softmax normalization requires a global reduction over the row.

The output ends up needing two passes: one to compute the per-row max and sum (for softmax), then a second to compute the normalized output. Two passes = 2× the HBM reads of K and V. Not great.

### The FlashAttention insight: online softmax

The online softmax trick from Lesson 9 lets us compute softmax in *one streaming pass* by maintaining two running values: the max-seen-so-far and a sum-of-exps-rescaled-by-the-current-max.

```
m_i = max over the first i elements
d_i = sum over the first i elements of exp(x_j - m_i)
```

When we process a new chunk of elements:

```
m_new = max(m_i, max of new chunk)
d_new = d_i * exp(m_i - m_new) + sum over new chunk of exp(x_j - m_new)
```

The same recursion can be applied to the *output*. Define the running unnormalized output:

```
O_i = sum over j ≤ i of exp(s_j - m_i) * V_j
```

When new data arrives:

```
O_new = O_i * exp(m_i - m_new) + sum over new chunk of exp(s_j - m_new) * V_j
```

At the end, the normalized output is `O_final / d_final`.

This is the algorithm. Process Q and the corresponding KV blocks in chunks, maintain running `(m, d, O)`, never materialize the full S or P. At the end, divide once.

### The tiling structure

FlashAttention's kernel structure:

```
For each Q-block of size B_r × d:
    Initialize m = -inf, d = 0, O = 0  (running stats and partial output)
    For each KV-block of size B_c × d:
        Load Q-block and KV-block into shared memory
        Compute S = Q @ K^T (matmul: B_r × B_c)
        Apply causal mask if needed
        Compute m_new = max(m, rowmax(S))
        Compute exp(S - m_new) into a B_r × B_c tile
        Update d = d * exp(m - m_new) + rowsum(exp(S - m_new))
        Update O = O * exp(m - m_new) + (exp(S - m_new)) @ V
        m = m_new
    Output O / d to global memory  (the normalization step)
```

Important properties:

- **No full N × N tensor is ever stored.** S exists only as a B_r × B_c tile in registers.
- **Each KV chunk is loaded once per Q chunk.** With B_r ≈ 64–128 and B_c chosen to fit shared memory, this is much fewer HBM reads than the naive O(N²) version.
- **The two matmuls (Q@K^T and (P_tile)@V) are inside the same kernel.** Fully fused.

Block size choices: B_r and B_c need to make Q-block, K-block, V-block, and S-tile fit in shared memory (~100 KB on a 4090). For d=128 FP16: each block is 128 × 128 × 2 bytes = 32 KB. Three of them (Q, K, V) plus the S tile is 4 × 32 KB = 128 KB — tight. Usually B_r = 64, B_c = 64 or 128 in practice.

### The IO complexity argument

FlashAttention's paper formalizes the gains as an IO complexity analysis. Naive attention reads O(N²) bytes from HBM (the S matrix). FlashAttention reads O(N · d) bytes plus O(N² · d / B_c) bytes for the KV scans. With B_c large enough, the second term dominates only at small d, and the practical bandwidth reduction is 5–20×.

The headline measurement from the paper: 7.6× speedup at sequence length 4096 vs the naive PyTorch implementation, scaling to larger gains at longer sequences. Numbers have improved since with FlashAttention-2 (better work partitioning) and FlashAttention-3 (Hopper-specific async).

### FlashAttention-2: more work per program

The original FlashAttention had one Q block per program (i.e., one CUDA block); the inner loop over KV blocks was serial within that block. FlashAttention-2 splits work the other way: more programs in flight (better SM occupancy) and reduces non-matmul work that was limiting throughput on Ampere.

The high-level algorithm is the same; the kernel structure changes for better scheduling.

### FlashAttention-3: Hopper async

FlashAttention-3 (Hopper-only) uses TMA for async tile loads and `wgmma` for async tensor-core MMA. This lets the kernel overlap KV-tile loads with the previous tile's compute — explicit pipelining instead of relying on the scheduler. The result: ~75% of H100 peak FP8 utilization, which is the highest sustained number I've seen for any LLM kernel.

The conceptual lesson remains: same algorithm, structural pipelining, ~2× the throughput.

### Decoding-specific flavors

For inference, two important variants:

- **FlashAttention for decoding (single-token at a time).** Q is 1 × d (one new token). K and V are the full KV cache. The algorithm degenerates — there's nothing to gain from tiling Q. Different optimization: **paged KV cache** + sequence parallelism + FlashDecoding (parallel across the sequence dimension for one query).

- **FlashAttention with KV cache** — full prefill is the standard FA case; decode-step is the degenerate case. Modern serving systems use FA for prefill and a paged variant (FlashAttention or vLLM's PagedAttention) for decode.

## Code walkthrough

A simplified Triton kernel skeleton — not production code (real FlashAttention is more careful about edge cases, masking, and block schedules), but enough to see the structure.

```python
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    L_ptr,                              # store the log-sum-exp for backward
    stride_qb, stride_qh, stride_qm, stride_qd,
    # ... similar for K, V, O ...
    B, H, N, D: tl.constexpr,
    BLOCK_M: tl.constexpr,              # B_r
    BLOCK_N: tl.constexpr,              # B_c
):
    pid_bh = tl.program_id(0)           # batch-head index
    pid_m = tl.program_id(1)            # which Q block

    # Load Q block into registers.
    q_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, D)
    q_ptrs = Q_ptr + ... compute offsets using pid_bh and q_offs_m ...
    Q = tl.load(q_ptrs, mask=q_offs_m[:, None] < N, other=0.0)

    # Initialize running stats.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Walk KV blocks.
    for start_n in range(0, N, BLOCK_N):
        # Load K and V blocks.
        k_offs_n = start_n + tl.arange(0, BLOCK_N)
        K = tl.load(K_ptr + ..., mask=k_offs_n[:, None] < N, other=0.0)
        V = tl.load(V_ptr + ..., mask=k_offs_n[:, None] < N, other=0.0)

        # Compute scores.
        S = tl.dot(Q, K.T)               # BLOCK_M × BLOCK_N
        S = S * (1.0 / tl.sqrt(D))

        # Causal mask (if needed): mask out S where col > row.
        # ...

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(S, axis=1))    # max along rows
        alpha = tl.exp(m_i - m_new)                    # rescale factor
        P = tl.exp(S - m_new[:, None])                 # un-normalized prob
        d_i = d_i * alpha + tl.sum(P, axis=1)
        O = O * alpha[:, None] + tl.dot(P, V)
        m_i = m_new

    # Normalize and store.
    O = O / d_i[:, None]
    o_ptrs = O_ptr + ...
    tl.store(o_ptrs, O.to(O_ptr.dtype.element_ty), mask=q_offs_m[:, None] < N)

    # Store log-sum-exp for backward pass.
    L = m_i + tl.log(d_i)
    tl.store(L_ptr + ..., L, mask=q_offs_m < N)
```

The key parts to recognize:

- **`Q` is loaded once** per (batch, head, Q-block).
- **The inner loop streams K and V blocks** — each loaded once, processed, discarded.
- **`m_i`, `d_i`, `O`** are the three running quantities updated each iteration.
- **`alpha = exp(m_i - m_new)`** is the rescaling factor that lets us merge the old and new statistics correctly.
- **Final division by `d_i`** normalizes the softmax. We do this once at the end.

The full Triton implementation (Tri Dao's `flash-attention`) handles: causal masking, dropout, ALiBi, multi-query / grouped-query attention, backward pass, and various block-size autotuning.

## Mental model & pitfalls

Single sentence: **FlashAttention reshapes attention from "materialize an N×N matrix, normalize, multiply by V" into "stream through K and V, updating a running max-and-denominator while accumulating a running output, normalize at the end" — and that reframing eliminates the N² memory pressure that bottlenecked attention before.**

Pitfalls when adapting the idea to a new kernel:

- **Numerical stability of the online recurrence.** The subtraction `S - m_new` and `alpha = exp(m_i - m_new)` keep things in a safe range. Naive implementations that compute `exp(S)` directly will overflow. Always do it the FlashAttention way.
- **Backward pass requires recomputation.** The backward pass recomputes `S` and `P` from saved `m_i`, `d_i`, and Q/K/V, since we never stored P. This trades extra compute for memory savings; usually it's a win.
- **Block size constraints.** Q-block, K-block, and S-tile must fit shared memory. On low-shared-memory GPUs, you can't always run the largest beneficial block.
- **Causal masking inside the inner loop.** Masking adds compute and can ruin tensor-core utilization if done poorly. FlashAttention skips entirely-future blocks and masks only the diagonal block, which keeps the optimization clean.
- **Decoding latency vs prefill throughput.** FA is great for prefill but degenerate for one-token decode. Don't expect the same speedup at decode-time as at prefill-time.

## Hands-on (at home)

1. **Run `flash-attention` (the official Triton/CUDA package)** if you have NVIDIA hardware:

```bash
pip install flash-attn --no-build-isolation
```

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 16, 4096, 128, device='cuda', dtype=torch.float16)
k = torch.randn(2, 16, 4096, 128, device='cuda', dtype=torch.float16)
v = torch.randn(2, 16, 4096, 128, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v, causal=True)
print(out.shape, out.dtype)
```

2. **Compare to naive attention:**

```python
def naive_attention(q, k, v, causal=True):
    s = torch.einsum('bhsd,bhtd->bhst', q, k) / (q.shape[-1] ** 0.5)
    if causal:
        mask = torch.triu(torch.ones_like(s), diagonal=1).bool()
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.einsum('bhst,bhtd->bhsd', p, v)
```

Time both for sequence length 4096, then 8192, then 16384 (if memory allows). Expected: at 4096, FA is ~3× faster; at 16384, ~15× faster (because naive runs out of memory or thrashes HBM).

3. **Compare max abs difference.** Should be <1e-2 in FP16.

4. **Read the actual Triton kernel** in `flash_attn/flash_attn_triton.py`. After this lesson, every line should be recognizable.

5. **(Optional) FlashAttention on Apple via MLX:**

```python
import mlx.core as mx
import mlx.nn as nn

# MLX's scaled_dot_product_attention uses a FlashAttention-style algorithm internally.
q = mx.random.normal((2, 16, 4096, 128)).astype(mx.float16)
k = mx.random.normal((2, 16, 4096, 128)).astype(mx.float16)
v = mx.random.normal((2, 16, 4096, 128)).astype(mx.float16)

out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/(128**0.5))
mx.eval(out)
```

`mx.fast.scaled_dot_product_attention` is MLX's optimized attention; it implements the same algorithmic ideas.

6. **(Optional, deeper)** Implement a simplified FlashAttention kernel in Triton from scratch, using the skeleton above as a starting point. Aim for correctness first; benchmark second. Even a naive port will be 2–3× faster than the materialized version.

## Further reading

- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022) — the original paper. Surprisingly readable; the core algorithm and complexity argument are clear by section 3.
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023) — the improved version. Architectural decisions explained.
- "FlashAttention-3" (Shah et al., 2024) — the Hopper async version.
- "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018) — the precursor paper that introduced the online softmax recursion. Brief and worth reading.
- Tri Dao's `flash-attention` repo on GitHub — the implementation. The README has the version compatibility matrix; the `csrc/` directory has the production CUDA/CUTLASS path; the Triton version is in `flash_attn_triton.py`.

Next lesson: porting FlashAttention to Apple Silicon. Same algorithm, different toolchain.
