---
title: "Lesson 11 — FlashAttention on Apple Silicon"
date: "2026-06-03"
module: "gpu-computing"
order: 11
tags: ["flashattention", "mlx", "metal", "mpsgraph", "apple-silicon", "on-device"]
author: "Sudipta Pathak"
prerequisites: ["10-flashattention-demystified"]
---

# Lesson 11 — FlashAttention on Apple Silicon

## Why this matters

The most important kernel in modern LLMs is FlashAttention. The most important deployment target for on-device AI is Apple Silicon. The two together are the foundation of practical on-device LLM serving — running Llama-class models on a phone, an iPad, or a Mac with usable latency. Every fast on-device LLM framework (MLX-LM, llama.cpp's Metal backend, Apple's own on-device models in iOS) uses some flavor of FlashAttention on the Metal/MLX path.

The good news: the algorithm doesn't change. The online softmax reduction, the Q-block-by-KV-block streaming, the running `(m, d, O)` recursion — all identical. What changes is the kernel substrate. Triton doesn't compile to Metal (yet); the explicit-async primitives are different (and less developed) on Apple; the matrix accelerator (SMMA on M3+) is shaped differently from NVIDIA's tensor cores.

This lesson covers what's the same, what's different, and how to actually run FlashAttention-class attention on Apple Silicon today.

## Concept

### The three Apple paths

For attention on Apple Silicon, you have three practical options:

**1. MLX's `mx.fast.scaled_dot_product_attention`.** MLX ships an optimized attention kernel implemented in Metal that uses the FlashAttention algorithm internally. Production-ready. Used by MLX-LM and many on-device pipelines. This is the right default.

**2. MPSGraph's attention op.** Apple's Metal Performance Shaders graph framework includes an attention operator. Generally faster than rolling your own, slower than MLX's path for many shapes. Stable and supported by Apple directly.

**3. A custom MSL kernel.** Write the FlashAttention algorithm directly in Metal Shading Language, dispatch via MLX's `fast.metal_kernel` API or via raw Metal. Maximum control, more work.

For most on-device LLM work the answer is (1). You write your model in MLX, call `scaled_dot_product_attention`, and you get FlashAttention performance without writing kernels. Cases where (3) is needed: novel attention variants (sliding window with a specific layout, modified scoring), aggressive low-precision (FP8 isn't yet widely supported on Apple GPUs), or experiments at the kernel level.

### What's the same as the CUDA / Triton story

- The algorithm: online softmax, Q-block streaming over KV blocks, no full N×N materialization.
- The tile structure: Q-block (B_r × d), K-block (B_c × d), V-block (B_c × d) in threadgroup memory.
- The running statistics: `m_i`, `d_i`, `O_i` updated per KV block.
- The memory-traffic argument: ~10× reduction at long sequences.

### What's different

**No 128-thread tensor-core MMA across a warp.** On NVIDIA, the basic matrix instruction does an 8×8 (or larger) MMA per warp, with the operands distributed across the 32 threads. On Apple M3+, the analog is `simdgroup_matrix_multiply_accumulate` doing 8×8 MMA per SIMD-group. Smaller per-op throughput; you need more SIMD-groups in flight to fill the SMs.

**Threadgroup memory is generous but no TMA.** Apple's GPUs have ~32 KB of threadgroup memory per threadgroup, similar to NVIDIA's 100 KB shared per block but smaller per-thread. No equivalent of Hopper's TMA for async bulk copies — Apple's async memory primitives are simpler.

**Memory bandwidth is lower.** ~400 GB/s on M3 Max vs 3 TB/s on H100. This means **attention is more memory-bound on Apple than on NVIDIA**. The FlashAttention algorithm's memory-traffic reduction translates into an *even larger* relative speedup on Apple, because the gap between "naive memory-bound" and "FlashAttention memory-traffic" matters more when bandwidth is scarce.

**Unified memory removes KV cache transfer cost.** A common pattern on NVIDIA: keep KV cache in HBM. On Apple: the KV cache is in unified DRAM, accessible by GPU and CPU. CPU-side resampling, repacking, or pruning of the cache can happen without copying.

### MLX's attention API

```python
import mlx.core as mx

q = mx.random.normal((B, H, N, D))  # batch, heads, seq, head_dim
k = mx.random.normal((B, H, N, D))
v = mx.random.normal((B, H, N, D))

# Causal, scaled dot-product attention.
out = mx.fast.scaled_dot_product_attention(
    q, k, v,
    scale=1.0 / (D ** 0.5),
    mask="causal",
)
mx.eval(out)
```

Behind the scenes this invokes MLX's FlashAttention-style kernel. The supported parameters cover the standard cases: causal mask, additive masks, GQA (key/value with fewer heads than query — broadcast handled internally), and FP16/BF16/FP32 precisions.

### When to write your own MSL attention

You'll consider a custom kernel when:

- You need an unusual mask pattern (sliding window with attention sinks, custom block-sparse).
- You're experimenting with a novel attention variant (DeepSeek's MLA, linear attention).
- You're targeting a model-shape that MLX's path doesn't handle optimally.
- You want to fuse attention with surrounding ops (positional embeddings, RoPE, rotary application).

For most other cases, MLX's path is the right answer.

## Code walkthrough

### The MLX path — usage

```python
import mlx.core as mx
import time

B, H, N, D = 1, 32, 4096, 128
q = mx.random.normal((B, H, N, D)).astype(mx.float16)
k = mx.random.normal((B, H, N, D)).astype(mx.float16)
v = mx.random.normal((B, H, N, D)).astype(mx.float16)
scale = 1.0 / (D ** 0.5)

# Warm up.
for _ in range(3):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out)

t0 = time.time()
for _ in range(50):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
mx.eval(out)
dt = (time.time() - t0) / 50

flops = 4 * B * H * N * N * D / 2  # causal halves the work
print(f"{dt*1000:.2f} ms  {flops / dt / 1e12:.1f} TFLOPs FP16")
```

Expected on M3 Max for this shape: ~6–8 ms per call, ~3–5 TFLOPs FP16 effective. This is the on-device frontier; what production MLX-LM runs.

### The naive baseline (for comparison)

```python
def naive_attention_mlx(q, k, v, scale, causal=True):
    # q, k, v: (B, H, N, D)
    s = q @ k.transpose(0, 1, 3, 2) * scale  # (B, H, N, N)
    if causal:
        mask = mx.tril(mx.ones((q.shape[2], q.shape[2])))
        s = mx.where(mask == 0, mx.array(-mx.inf), s)
    p = mx.softmax(s, axis=-1)
    return p @ v
```

Bench this on the same shape. Expected: ~50–80 ms for N=4096 — and it will *fail* (OOM or thrash) for N=8192 because of the N² intermediate.

The ratio: ~10× speedup for FlashAttention at this shape. At longer contexts the ratio grows; at shorter (say N=512), the naive version is comparable because the N² overhead is small.

### A sketch of a custom MSL FlashAttention kernel

Showing the structure, not a complete production kernel. The pattern follows the Triton version from Lesson 10:

```metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void flash_attn_kernel(
    device const half* Q                [[buffer(0)]],
    device const half* K                [[buffer(1)]],
    device const half* V                [[buffer(2)]],
    device       half* O                [[buffer(3)]],
    constant int& B                     [[buffer(4)]],
    constant int& H                     [[buffer(5)]],
    constant int& N                     [[buffer(6)]],
    constant int& D                     [[buffer(7)]],
    constant float& scale               [[buffer(8)]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]],
    uint3 t_pos                         [[thread_position_in_threadgroup]],
    uint tid                            [[thread_index_in_threadgroup]])
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int HEAD_D  = 128;        // assume d=128 for this kernel

    threadgroup half sQ[BLOCK_M][HEAD_D];
    threadgroup half sK[BLOCK_N][HEAD_D];
    threadgroup half sV[BLOCK_N][HEAD_D];

    // pid_bh selects (batch, head); pid_m selects the Q block.
    int pid_bh = tg_pos.x;
    int pid_m  = tg_pos.y;
    int batch  = pid_bh / H;
    int head   = pid_bh % H;

    // Load Q block from global to threadgroup.
    // ... cooperative loads using tid to spread the work ...
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread accumulators (registers).
    float m_i[BLOCK_M / WARP];       // running max, partitioned
    float d_i[BLOCK_M / WARP];       // running denominator
    float O_i[BLOCK_M / WARP][HEAD_D];

    // Initialize to -inf, 0, 0
    // ...

    for (int start_n = 0; start_n < N; start_n += BLOCK_N) {
        // Load K and V blocks.
        // ... cooperative loads of sK and sV ...
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T (BLOCK_M × BLOCK_N), via SMMA where supported.
        // ...

        // Apply causal mask if needed.
        // ...

        // Online softmax update: m_new, alpha, P, d, O all updated.
        // ...

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize O_i by d_i, store to global O.
    // ...
}
```

This is non-trivial code at full implementation, but the structure matches the CUDA/Triton version exactly. Working examples for M3+ live in MLX's own source.

## Mental model & pitfalls

Single sentence: **FlashAttention's algorithm is platform-independent; on Apple Silicon, use MLX's built-in for production and only write your own MSL kernel for experiments or unusual variants.**

Pitfalls:

- **Forgetting that `mx.fast.scaled_dot_product_attention` exists.** Implementing attention manually with `q @ k.T` style code in MLX gives you the slow N² version; the fast path requires the named function.
- **Using FP32 inputs.** The fast path is tuned for FP16/BF16 (matching the matrix instructions). Pass FP32 inputs and you'll fall back to a slower path or take a manual conversion hit.
- **Confusing MLX with PyTorch MPS.** PyTorch's MPS backend (`torch.tensor(...).to("mps")`) and MLX are different code. MPS's attention kernel exists but is less optimized than MLX's. For new on-device work, MLX is the better choice.
- **Memory layout assumptions.** MLX expects `(batch, heads, seq, head_dim)`. PyTorch is the same. Some inference kernels use `(batch, seq, heads, head_dim)`. Mixing them silently produces wrong outputs.
- **Causal mask cost.** The `mask="causal"` path is well-optimized; arbitrary additive masks fall back to a slower path. Use the structured mask when possible.

## Hands-on (at home)

Requires an Apple Silicon Mac.

1. **Run the MLX FlashAttention call** as in the code walkthrough. Verify speed.

2. **Implement the naive comparison.** Bench at N = 1024, 4096, 8192. Note where the naive version OOMs (probably around 4096–8192 depending on RAM).

3. **Sweep head dim.** Try D = 64, 128, 256. MLX's path is best-tuned for the common values (64, 128).

4. **Try GQA.** When K and V have fewer heads than Q (typical for Llama 3+ and similar), MLX broadcasts internally. Pass q with H=32, k and v with H=8, head_dim=128:

```python
q = mx.random.normal((1, 32, N, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, N, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, N, 128)).astype(mx.float16)
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/(128**0.5), mask="causal")
```

The KV memory budget drops by 4× compared to MHA. This is a major win for on-device inference.

5. **Run an end-to-end MLX-LM model** to see the kernel in production:

```bash
pip install mlx-lm
python -m mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --prompt "Explain memory bandwidth in one paragraph"
```

Watch tokens/sec. At Q4 quantization with FlashAttention, expect ~80–200 tokens/sec on an M3 Max for a 3B model. This is *fast*. It's also the assembly of every concept in this module so far.

6. **(Optional) Profile in Xcode.** Open Xcode → File → New → Tool, add a small driver that calls into MLX. Profile under the Metal System Trace instrument. See the kernel boundaries, GPU utilization, and memory bandwidth.

7. **(Optional, hard) Write your own simplified FlashAttention in MSL.** Use the sketch above. Start with a fixed head_dim=64, no GQA, FP16 only. Aim to match (or get within 2×) MLX's path. This is a meaningful project — plan a weekend.

## Further reading

- MLX source: `mlx/backend/metal/kernels/scaled_dot_product_attention.metal` and related files. The reference implementation.
- MLX-LM source on GitHub — production usage of `mx.fast.scaled_dot_product_attention` across many model architectures.
- Apple WWDC sessions on Metal performance and ML inference (year over year) — relevant updates each cycle.
- "Awni Hannun's MLX talks and blog posts" — Awni leads MLX development at Apple and writes accessibly about performance details.

Next lesson: profiling GPU kernels. We have a fast matmul, a fast attention, and now we need to see what they're actually doing at runtime — the tools that turn vague "is this fast" into precise "this is bandwidth-bound at 78% of peak".
