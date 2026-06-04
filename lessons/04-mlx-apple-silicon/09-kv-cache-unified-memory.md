---
title: "Lesson 9 — KV Cache Strategies on Unified Memory"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 9
tags: ["kv-cache", "unified-memory", "autoregressive", "decoding", "in-place", "paged-attention"]
author: "Sudipta Pathak"
prerequisites: ["08-custom-metal-kernels"]
---

# Lesson 9 — KV Cache Strategies on Unified Memory

## Why this lesson exists

The KV cache is the data structure that turns a quadratic-in-context attention computation into a linear-in-context one. During autoregressive decoding, each new token would naively recompute attention against every prior token; with a KV cache, you instead store every prior token's K and V tensors and have the new token attend against the cache. Module 3 Lesson 1 introduced the size budget; this lesson is about the layout, the update pattern, and the strategies that work well — specifically on Apple Silicon, where unified memory changes the design space.

The unified-memory difference matters here because the dominant KV-cache pattern in CUDA-land is *paged attention* (vLLM's invention), which uses non-contiguous blocks to allow efficient memory packing across requests. That design was motivated by CUDA's separate VRAM and the need to dynamically allocate and free KV pages across concurrent requests in a server setting. On Apple Silicon, where there's only one memory pool and most workloads are single-user, the simpler designs win — but there's still a real choice between three or four layouts, with different latency/throughput/memory tradeoffs.

The lesson is reading. The Hands-on implements two KV-cache layouts in MLX and measures the cost difference.

## What goes in the KV cache

For each transformer layer, the attention mechanism produces K and V tensors of shape `[batch, n_heads, seq_len, d_head]` (or with various dimension permutations depending on the framework). During training and prefill, these are computed for the whole sequence at once. During decode, only the latest token's K and V are new; everything from previous decode steps must be cached.

The size: `2 (K+V) × n_layers × n_heads × seq_len × d_head × bytes_per_elem`.

For Llama 3 8B (32 layers, 8 KV heads (it's GQA), `d_head=128`) at FP16:
- per-token: 2 × 32 × 8 × 128 × 2 = 131 KB.
- per 1K context: 131 MB.
- per 32K context: 4.2 GB.

The KV cache dominates memory for long-context inference. This is the constraint everything below tries to manage.

## Layout option 1: contiguous, pre-allocated max-length buffer

The simplest layout: allocate a buffer for the maximum sequence length you'll handle (say, 8K tokens) and write into it sequentially.

```
K shape: [n_layers, n_heads, MAX_LEN, d_head]
V shape: [n_layers, n_heads, MAX_LEN, d_head]
```

Each decode step writes one slot at index `current_len`, then attention reads slots `0..current_len`. Trivial to implement; great memory locality (contiguous loads); fast on Apple Silicon because the unified-memory layout means you can update in place without sync.

Cost: you pay for `MAX_LEN` whether or not you use it. If your typical query is 200 tokens but `MAX_LEN=8192`, you've allocated 40× more than you need. For single-user laptop inference this is fine; for multi-tenant serving it's wasteful.

This is the default for `mlx-lm` and most laptop LLM workflows. The simplicity wins.

## Layout option 2: contiguous, grow as needed

Allocate a small initial buffer; when it fills up, allocate a larger one and copy. The classic `ArrayList` / `std::vector` pattern.

```
K_buffer: initially [n_layers, n_heads, INITIAL_CAP, d_head]
when seq_len > INITIAL_CAP:
    new_buffer = allocate(2 × INITIAL_CAP); copy old → new
```

Cost: occasional growth events that copy the whole cache. On Apple Silicon, the copy is bandwidth-bound; copying a 1 GB cache at 200 GB/s costs 5 ms — fast in absolute terms but a noticeable hiccup in a streaming decode loop.

The amortized cost is fine (each token's data is copied O(log N) times across the geometric growth), but the latency spikes are user-visible.

Not commonly used; the pre-allocated max-length pattern is preferred even with its memory waste.

## Layout option 3: paged attention

The vLLM design: split the KV cache into fixed-size blocks (typically 16 tokens per block). Maintain a per-sequence "block table" mapping logical token positions to physical block IDs. Multiple sequences can share blocks (e.g., the prefix of a few prompts that share a common system prompt). Memory is allocated and freed block-by-block.

The win on CUDA: efficient memory utilization in a multi-tenant server, where many requests with different lengths share a GPU. Without paged attention, you'd over-allocate for each request's worst-case length, wasting VRAM you can't easily reclaim.

The cost: every attention computation has to follow the block table indirection. The kernel becomes more complex (the standard FlashAttention kernel doesn't work directly; you need a variant that handles block-table lookups).

On Apple Silicon for single-user inference, paged attention is overkill. The simpler contiguous layout has better latency and the memory waste is bearable because you control the workload. For an Apple Silicon LLM *server* handling many concurrent users (rare in 2026 but emerging), paged attention starts to make sense.

MLX has experimental paged-attention support; production deployments still mostly use the contiguous layout.

## Layout option 4: ring buffer with sliding window

For models that use sliding-window attention (Mistral-style local attention with window size W=4096), the KV cache only needs to hold the most recent W tokens. A ring buffer of fixed size W works:

```
K_buffer: [n_layers, n_heads, W, d_head]  // fixed size
write index: i = current_step % W
attention attends to: tokens [(current_step - W + 1) ... current_step]
```

The ring buffer never grows. Memory is bounded. The attention kernel needs to handle the wrap-around (the "logical position 0" of the window may be at physical index 2048 in the buffer).

This is what `llama.cpp` does for Mistral / Gemma sliding-window models on Apple Silicon. The memory savings are huge — you cap the KV cache at W × per-token-cost regardless of generation length.

For non-sliding-window models (most others), this doesn't apply.

## In-place update is essentially free on Apple Silicon

A subtle but real win: on Apple Silicon, in-place update of the KV cache buffer is essentially free. The GPU writes into the unified memory buffer at index `current_len`; the next attention kernel reads from the same buffer. There's no copy, no cache invalidation hassle, no separate "device" version of the buffer to keep in sync.

On CUDA, in-place updates work but you have to be careful about cache coherence (writing from one stream while reading from another can race); the framework's KV cache abstractions handle this for you. On MLX, the abstraction is just "write into the buffer at the right index" — the lazy graph handles ordering, and the unified memory means there's no separate copy to sync.

This is why MLX's KV cache code is short and clean (~100 lines per layout) compared to the multi-thousand-line vLLM KV cache management.

## Quantizing the KV cache

KV cache quantization is the next compression lever after weight quantization. The math is the same as Module 3's weight quantization (scale + zero-point, per-head granularity), applied dynamically as the cache fills.

In MLX:

- INT8 KV cache: 2× memory reduction, ~0.1 perplexity hit on most models. Reliable.
- INT4 KV cache: 4× memory reduction, ~0.5 perplexity hit. More aggressive; sometimes hurts on long context.

The recipe: as each new K and V is produced, quantize it per-head with the typical max-based scale. Store the scale (FP16, one per head per token) alongside the quantized value. The attention kernel dequantizes on-the-fly during the read.

Performance: a quantized KV cache is fewer bytes to read during attention, which is bandwidth-bound. INT8 KV cache gives ~20–30% throughput improvement on attention-heavy long-context workloads. INT4 KV cache gives more but with the accuracy hit.

`mlx-lm` supports INT8 KV cache out of the box (`--kv-bits 8`). INT4 is experimental but available.

For very long context (32K+) where the KV cache dominates memory, KV quantization is essential — without it, you can't fit the cache.

## A simple MLX KV cache implementation

A working sketch of the simplest layout (contiguous, pre-allocated):

```python
import mlx.core as mx

class KVCache:
    def __init__(self, n_layers, n_heads, max_seq_len, d_head, dtype=mx.float16):
        self.k = mx.zeros((n_layers, n_heads, max_seq_len, d_head), dtype=dtype)
        self.v = mx.zeros((n_layers, n_heads, max_seq_len, d_head), dtype=dtype)
        self.current_len = 0

    def update(self, layer, new_k, new_v):
        # new_k, new_v: [1, n_heads, n_new, d_head] for `n_new` new tokens
        n_new = new_k.shape[2]
        start = self.current_len
        end = start + n_new
        # MLX doesn't have in-place assignment of slices in the public API
        # without going through an op; the conventional way is to use
        # mx.array's __setitem__ on the underlying storage. In practice,
        # mlx-lm uses an internal API for this.
        self.k = mx.concatenate([self.k[:, :, :start], new_k, self.k[:, :, end:]], axis=2)
        # (This isn't actually in-place; real impl uses lower-level updates.)
        self.current_len = end

    def read(self, layer):
        return (self.k[layer, :, :self.current_len], self.v[layer, :, :self.current_len])
```

(The real `mlx-lm` KV cache uses lower-level update primitives that *are* in-place; the above is illustrative of the layout, not the optimal implementation.)

The attention computation:

```python
# In the attention forward pass for one decode step:
q = compute_q(x)  # shape [1, n_heads, 1, d_head] for a single new token
new_k, new_v = compute_k_v(x)  # shape [1, n_heads, 1, d_head]
cache.update(layer, new_k, new_v)
k, v = cache.read(layer)  # full K and V up to current length
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/(d_head**0.5))
```

The `mx.fast.scaled_dot_product_attention` is MLX's optimized SDPA kernel (FlashAttention-style); it handles the long-K-and-V case efficiently.

## What you should believe after this lesson

Three sentences:

**1. Unified memory simplifies KV cache design significantly** — in-place updates are essentially free, no host-device sync, no separate VRAM allocator to worry about. The most common Apple Silicon pattern (contiguous pre-allocated max-length buffer) is simple and competitive with the more elaborate CUDA-era designs.

**2. Paged attention is overkill for single-user Apple Silicon inference**, but starts to matter for multi-tenant servers (which are uncommon on Apple Silicon today). For sliding-window models, the ring buffer pattern caps memory regardless of generation length.

**3. KV cache quantization (INT8, sometimes INT4) is essential for long-context inference**, where the cache often dominates memory. MLX supports INT8 KV cache out of the box; the recipe mirrors Module 3's weight quantization but applied dynamically as the cache grows.

## Hands-on (at home)

Measure the bandwidth cost of KV cache reads at increasing context length.

```python
# kv_cache_bandwidth.py
# pip install mlx
import mlx.core as mx
import time

n_layers, n_heads, d_head = 32, 8, 128

for seq_len in [128, 512, 2048, 8192, 16384]:
    k = mx.random.normal((1, n_heads, seq_len, d_head), dtype=mx.float16)
    v = mx.random.normal((1, n_heads, seq_len, d_head), dtype=mx.float16)
    q = mx.random.normal((1, n_heads, 1, d_head), dtype=mx.float16)
    mx.eval(k, v, q)

    # SDPA against the full cache, one query token.
    for _ in range(5):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/(d_head**0.5))
        mx.eval(out)

    t0 = time.time()
    for _ in range(100):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/(d_head**0.5))
        mx.eval(out)
    dt = (time.time() - t0) / 100

    # Bytes read: K + V, each of size n_heads * seq_len * d_head * 2.
    bytes_read = 2 * n_heads * seq_len * d_head * 2
    bw_gbs = bytes_read / dt / 1e9
    print(f"seq_len={seq_len:6d}  attention={dt*1000:6.3f} ms  ~{bw_gbs:.1f} GB/s effective")
```

You'll see attention time grow linearly with seq_len (the FlashAttention algorithm); the effective bandwidth (bytes read / time) should approach 60–80% of your chip's peak bandwidth at longer contexts where the attention is bandwidth-bound. Below ~512 tokens, the kernel launch overhead dominates and the bandwidth number looks lower.

Try the same experiment with `--kv-bits 8` via `mlx-lm`:

```bash
mlx_lm.generate \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --prompt "Write a 500-word essay about chess." \
    --max-tokens 800 \
    --kv-bits 8
```

Compare to the default (no `--kv-bits`). At 800 tokens generated from a short prompt, the difference is small; at 8K+ context you'd see meaningful memory savings without throughput loss.

## Further reading

- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al, 2023) — the vLLM paper that introduced paged attention.
- `mlx-lm`'s `cache.py` — the production KV cache implementation; ~200 lines, very readable.
- "Streaming LLMs" / Attention Sinks (Xiao et al, 2023) — for the trick of keeping the first few tokens' KV alongside a sliding window.
- Mistral's "Sliding Window Attention" technical report — for the design context of why sliding windows came about.
- "FlashAttention" papers (cited in Module 2) — for the attention kernel that consumes the KV cache.

Next lesson: **Quantization on Apple Silicon.** Bringing Module 3's quantization recipes to the actual Apple runtimes: MLX's native quantized formats, llama.cpp's GGUF, the conversion paths, and which delivers the best tokens/sec at INT4.
