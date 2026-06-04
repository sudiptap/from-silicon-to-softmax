---
title: "Lesson 17 — KV Cache Memory Layout and Arithmetic Intensity"
date: "2026-06-04"
module: "inference-from-scratch"
order: 17
tags: ["kv-cache", "layout", "arithmetic-intensity", "roofline", "bandwidth"]
author: "Sudipta Pathak"
prerequisites: ["16-why-kv-cache-exists"]
---

# Lesson 17 — KV Cache Memory Layout and Arithmetic Intensity

## Why this lesson exists

The KV cache is fundamentally a memory-layout problem. How you arrange the K and V tensors in DRAM determines how efficiently the attention kernel can load them. Get the layout right and you read tensors at near-peak HBM bandwidth; get it wrong and the strided access patterns cut effective bandwidth by 2-4×.

This lesson covers the layout choices that the runtimes have settled on, the arithmetic-intensity math that says decode is bandwidth-bound, and the roofline picture that quantifies how much performance you're leaving on the table.

The lesson is reading. The Hands-on profiles the bandwidth of KV cache reads under different layouts.

## The tensor layout

For each transformer layer, K and V are stored as 4D tensors:

```
K: [batch, n_kv_heads, sequence_length, d_head]
V: [batch, n_kv_heads, sequence_length, d_head]
```

Or equivalently, with sequence as the leading dimension:

```
K: [sequence_length, batch, n_kv_heads, d_head]
```

The choice of layout affects access patterns. The dominant attention kernel access pattern is "for each Q, load all K and V across the sequence dimension." The layout should make this contiguous.

For PyTorch's `[batch, n_kv_heads, sequence, d_head]`, the sequence dimension is not innermost. Reading "all K for one head" requires strided access. For consumer GPUs this is fine because cache lines absorb the strides; for the largest models the layout becomes a meaningful optimization.

Production runtimes use one of two layouts:
- **`[layer, batch, n_kv_heads, seq, d_head]`**: standard PyTorch tensor; accessible via standard tensor operations.
- **Block-based layouts** (paged attention, RadixAttention): break the sequence into fixed-size blocks, store each block as a contiguous chunk. Each block contains all of `[batch, n_kv_heads, BLOCK_SIZE, d_head]`. Trades some access pattern complexity for memory-management flexibility.

## Per-token reads during decode

During decode at position `i`, the attention computation for one head reads:
- The current Q: `d_head` floats. Negligible.
- All K for this head, positions 0 to `i-1`: `i × d_head` floats.
- All V for this head, positions 0 to `i-1`: `i × d_head` floats.

For 32 heads, 32 layers, position 4096, FP16: `32 × 32 × 2 × 4096 × 128 × 2 bytes = 2 GB` of cache reads. At 200 GB/s effective bandwidth (M3 Pro), that's 10 ms just for the KV reads in one decode step.

Compare to the compute: the attention scoring is `Q × K^T` which is `n_heads × N × d_head` ops per query token. For 32 heads, position 4096, d_head=128: `32 × 4096 × 128 = 16 MFLOPs`. At 7 TFLOPs of M3 Pro, 2 microseconds. Compute is irrelevant; bandwidth is everything.

This is the canonical "decode is bandwidth-bound" calculation.

## Arithmetic intensity calculation

Arithmetic intensity (AI) = FLOPs / bytes loaded.

For one decode step's attention:
- FLOPs: ~`4 × n_heads × N × d_head` (the QK^T matmul, softmax, weighted sum, output).
- Bytes loaded: `2 × n_heads × N × d_head × bytes_per_elem` (K + V cache reads).

AI = `4 × n_heads × N × d_head / (2 × n_heads × N × d_head × bytes_per_elem)` = `2 / bytes_per_elem`.

For FP16 (2 bytes): AI = 1 FLOP/byte.

For comparison:
- 4090: peak 165 TFLOPs / 1 TB/s ≈ 165 FLOPs/byte before becoming compute-bound.
- H100: peak 700 TFLOPs / 3 TB/s ≈ 233 FLOPs/byte.
- M3 Pro: peak 7 TFLOPs / 300 GB/s ≈ 23 FLOPs/byte.

Decode is at AI = 1, ~100× below the compute-bound transition. The GPU is running at ~1% of its compute peak; 99% of its FLOP capacity is idle.

This is the structural reason for everything in this curriculum about quantization, batching, and speculative decoding. They're all attempts to push the AI of decode higher, toward compute-bound territory where the chip can do useful work with the bandwidth it's burning.

## The FFN side

The FFN (gate, up, down projections in SwiGLU) is the other half of each decoder layer's compute. For a single decode token:
- Each linear layer reads its weight matrix from memory and multiplies it by the input vector.
- For Llama-7B-class: gate is `[D, 11008]`, up is `[D, 11008]`, down is `[11008, D]`. Total parameters per FFN: `~33D × D / 4 ≈ 90M` weights per layer per FFN.
- Bytes loaded per layer for FFN: ~`90M × 2 = 180 MB` per layer per decode step at FP16.

Across 32 layers: ~5.6 GB per decode step.

Compare to KV cache reads at 4K context: ~2 GB.

The FFN dominates the bandwidth-bound cost at moderate context; the KV cache dominates at very long context (where the cache grows). The crossover for typical 7B-class models is around 8K-16K context.

This means quantization (which compresses weights) helps decode more than KV quantization at moderate context, and KV quantization helps more at very long context. The right optimization depends on the deployment's context profile.

## Batching changes the arithmetic

At batch size `B`, decode arithmetic intensity becomes:

- FLOPs: `4 × B × n_heads × N × d_head` (B× more work per step).
- Bytes loaded: `2 × n_heads × N × d_head × bytes` (K, V cache loaded once, shared across batch) + `B × d_model × bytes` (per-batch Q, hidden state — small).

AI ≈ `B × 2 / bytes_per_elem` = `B` FLOPs/byte at FP16.

At batch 16: AI = 16. At batch 64: AI = 64. We're approaching the compute-bound transition. This is why batching is the single biggest decode-throughput win on server-side serving.

For on-device (batch 1) there's nowhere to batch. The bandwidth-bound regime is unavoidable without other tricks (speculative decoding, smaller models).

## Layout for paged attention

vLLM's paged attention (Lesson 18) uses a block-based layout:

```
K_cache: shape [n_blocks, n_heads, block_size, d_head]
V_cache: shape [n_blocks, n_heads, block_size, d_head]
```

`n_blocks` is the total number of memory blocks the runtime has allocated. `block_size` is typically 16 (16 tokens per block).

The block table per sequence: `[n_seq, max_blocks_per_seq]` mapping logical positions to physical blocks.

This layout supports dynamic allocation (allocate a new block when a sequence grows) and sharing (multiple sequences pointing to the same blocks for shared prefixes). Lessons 18 and 19 cover this in detail.

## What you should believe after this lesson

Three sentences:

**1. KV cache reads at decode are the dominant bandwidth cost at long context**, and the FFN weight reads dominate at short-to-medium context. The crossover for 7B-class models is around 8K-16K tokens; beyond that, KV cache is the binding constraint.

**2. Decode arithmetic intensity is `2 / bytes_per_elem` per single decode step** — `1` for FP16, far below the `100+` needed to be compute-bound on modern GPUs. The chip is running at ~1% of its compute capacity, with 99% of FLOP units idle.

**3. Batching multiplies AI by the batch size** — at batch 64 or higher, decode becomes compute-bound. This is the structural reason batching is the dominant server-side throughput optimization; for on-device (batch 1) you have to use other techniques to fill the FLOP gap.

## Hands-on (at home)

Measure decode bandwidth utilization.

```python
# decode_bandwidth.py
import torch
import torch.nn.functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'mps'

# Simulate a single decode step's KV reads.
n_heads, d_head, N, n_layers = 32, 128, 4096, 32
dtype = torch.float16

K_cache = torch.randn(n_layers, n_heads, N, d_head, device=device, dtype=dtype)
V_cache = torch.randn(n_layers, n_heads, N, d_head, device=device, dtype=dtype)
q = torch.randn(n_layers, n_heads, 1, d_head, device=device, dtype=dtype)

# Bench: one decode step across all layers.
def one_decode_step():
    out = torch.zeros(n_layers, n_heads, 1, d_head, device=device, dtype=dtype)
    for layer in range(n_layers):
        scores = (q[layer] @ K_cache[layer].transpose(-2, -1)) / (d_head ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out[layer] = weights @ V_cache[layer]
    return out

# Warm.
for _ in range(5):
    one_decode_step()
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()

t0 = time.time()
for _ in range(100):
    one_decode_step()
torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
dt = (time.time() - t0) / 100

# Bytes moved per step: K + V cache reads.
bytes_per_step = 2 * n_layers * n_heads * N * d_head * 2  # FP16
gbps = bytes_per_step / dt / 1e9
print(f"per decode step: {dt*1000:.2f} ms")
print(f"bytes per step: {bytes_per_step/1e9:.2f} GB")
print(f"effective bandwidth: {gbps:.0f} GB/s")
print(f"(M3 Pro peak: 300 GB/s; 4090 peak: 1000 GB/s)")
```

You should see effective bandwidth in the realistic range for your hardware (60-80% of peak). The decode time is what it is; the FLOP utilization is 1-5%; the bandwidth utilization is 60-80%.

## Further reading

- "Efficiently Scaling Transformer Inference" (Pope et al, 2022) — the roofline analysis for inference at scale.
- vLLM's paged attention paper — for the block-based layout.
- Horace He's "Making Deep Learning Go Brrrr From First Principles" — for the roofline framing applied to PyTorch code.
- "FlashAttention" papers — for the IO-aware analysis that motivates the KV cache layout choices.

Next lesson: **PagedAttention.** vLLM's virtual-memory analog for the KV cache. Instead of contiguous allocations, treat the cache as blocks that can be allocated and freed dynamically. This solves the memory fragmentation problem and enables aggressive cross-request sharing.
