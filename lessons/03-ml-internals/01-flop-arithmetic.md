---
title: "Lesson 1 — The Arithmetic of Neural Nets: Where the FLOPs Actually Go"
date: "2026-06-04"
module: "ml-internals"
order: 1
tags: ["flops", "transformer", "arithmetic-intensity", "roofline", "memory-bound"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — The Arithmetic of Neural Nets: Where the FLOPs Actually Go

## Why this lesson exists

Before we shrink a model, we should know what we are shrinking. "Make the model smaller" is not specific enough. The interesting questions are: smaller in *what*? Parameter count? Activation memory? Weight bytes loaded per token? Compute? KV cache? Each of these has a different scaling law, a different lever, and a different ceiling. Compression budgets that don't distinguish them produce nonsense — the engineer says "I quantized to INT4 and got a 1.5× speedup" when they expected 4×, and they don't know why.

This lesson is the budget. We count the FLOPs and bytes that a transformer layer actually performs, layer by layer. We separate compute from memory traffic. We compute the arithmetic intensity — the FLOPs-per-byte ratio that determines whether a kernel is compute-bound or memory-bound. The punch line, which the rest of the module rests on: **inference is memory-bound for batch sizes that matter on-device, and that is precisely why quantization helps so much.** Compression doesn't speed up the compute; it speeds up the *loading*.

The lesson is reading. The Hands-on at the bottom is a one-page Python script that profiles the FLOP and byte budget of a transformer block for a chosen model.

## The transformer block, counted

The standard decoder-only transformer block has two halves: attention and the feed-forward network (FFN, also called MLP). For a model with hidden size `d`, sequence length `n`, batch size `B`, head dimension `d_h`, and FFN expansion factor `f` (typically 4 for older models, often `8/3` for SwiGLU-style models like Llama), the per-block FLOP counts are:

**Attention (one block, one forward pass):**

- Q, K, V projections: `3 × B × n × d × d` → `3 B n d²` multiply-adds → `6 B n d²` FLOPs.
- QK^T (per head): `B × n × n × d_h` per head; summed across `d/d_h` heads gives `B n² d` → `2 B n² d` FLOPs.
- Softmax: ~`5 B n² (d/d_h)` FLOPs (small, often ignored).
- AV (attention × values): same `2 B n² d` FLOPs as QK^T.
- Output projection: `B × n × d × d` → `2 B n d²` FLOPs.

Sum: roughly `8 B n d² + 4 B n² d` FLOPs.

**FFN (Llama-style SwiGLU, with three matrices of shape `d × f·d`):**

- Gate, up, down projections: `3 × 2 B n d (f·d)` = `6 B n d² · f` FLOPs.
- Element-wise: tiny, ignored.

For `f = 8/3` (Llama's SwiGLU), that's `16 B n d²` FLOPs per block.

**Total per block:** `~8 B n d² + 4 B n² d + 16 B n d² = ~24 B n d² + 4 B n² d`.

Two takeaways:

1. **The `d²` term — the projections — dominates for short sequences.** For Llama 3 8B (`d = 4096`, `f = 14336/4096 ≈ 3.5`), a context of `n = 2048`, the projection term is `~24 × 2048 × 4096² ≈ 8.2 × 10¹¹` FLOPs per block, while the attention term is `4 × 2048² × 4096 ≈ 6.9 × 10¹⁰` — an order of magnitude smaller. Total per block ≈ 0.9 TFLOPs.
2. **The `n²` term — attention — overtakes the projections only at long context.** Crossover is around `n ≈ 6d` (rough). At 64K context, `n²` wins; this is exactly the regime where FlashAttention's structural fix matters most (see Module 2, Lesson 10).

Multiply per-block by the number of blocks (32 for Llama 3 8B), and you have the full forward pass: ~29 TFLOPs to encode a 2K-context prompt through Llama 3 8B at batch 1.

## The KV cache, which most people forget

During autoregressive decoding, you don't recompute attention against the whole prefix every step. You cache K and V for each prior token and only compute Q-fresh for the new token. The cache size:

```
KV cache bytes = 2 × n_layers × 2 × n_tokens × d × bytes_per_elem
                 (K and V)       (tokens)
```

For Llama 3 8B at FP16, 2K context: `2 × 32 × 2 × 2048 × 4096 × 2 = 2.1 GB`. At 32K context: 33 GB. **This is why long context is so expensive.** The KV cache often eats more memory than the weights themselves. It also means that decoding FLOPs scale linearly with sequence length (one new token attends to all prior K/V), not quadratically as in training — but the *memory* to hold the K/V grows linearly with sequence length.

Compression of the KV cache (INT8 / INT4 KV cache, sliding-window) is the other big lever Module 3 sets up. The weights take their byte cut; the KV cache takes its byte cut; both contribute.

## Arithmetic intensity: FLOPs per byte

The roofline model — referenced lightly in Module 1 (matmul) and Module 2 (kernel fusion) — sharpens to a single ratio:

```
Arithmetic intensity = FLOPs / bytes of memory traffic
```

A kernel running on a chip with peak `P` FLOPs/s and peak `B` bytes/s of memory bandwidth is bottlenecked by whichever is smaller of:

- `P` (compute ceiling)
- `B × arithmetic_intensity` (memory-bound ceiling)

If the kernel has high arithmetic intensity (many FLOPs per byte loaded), the memory-bound ceiling is high and compute dominates: you are *compute-bound*. If it has low arithmetic intensity (few FLOPs per byte), the memory-bound ceiling is low: you are *memory-bound*, and adding compute does nothing — you are waiting on memory.

The crossover ratio is `P / B`. For a 4090: ~150 TFLOPs FP16 / 1 TB/s = 150 FLOPs/byte. For an H100: ~700 TFLOPs / 3 TB/s = 230 FLOPs/byte. For Apple M3 Max: ~17 TFLOPs / 400 GB/s = 42 FLOPs/byte.

A matmul of shape `M × K × N` has FLOPs `2 M N K` and (at FP16) loads `2 (M K + K N + M N)` bytes. Intensity = `M N K / (M K + K N + M N)`. For a large square matmul (say `4096 × 4096`), that's ~`(4096³) / (3 × 4096²) ≈ 1365` FLOPs/byte. Comfortably compute-bound on every chip.

A matmul where one matrix is tiny — say a single token (`M=1`) by a `4096 × 4096` weight — has intensity `(N K) / (K + N K + N) ≈ K N / (K N) = 1`. **Catastrophically memory-bound.** The 4090 can do 150 FLOPs per byte loaded; you are asking it to do 1.

That second case is the *normal case* for autoregressive LLM decoding at batch size 1. Every new token is a `1 × d` by `d × d` matmul against each weight matrix. The chip spends almost all its time waiting on weight loads from HBM. The compute units are idle. **This is why an INT4 model on a 4090 generates tokens roughly 2–4× faster than the same model in FP16, even though the tensor cores run at the same rate for both formats.** You're not faster at multiplying; you're faster at loading the operands.

(This also explains why batch size helps: at batch 32, the `1 × d × d × d` matmul becomes `32 × d × d × d`. Intensity scales with batch. Past some batch size, you're back to compute-bound and quantization stops helping wall-clock latency — though it still helps memory footprint.)

## The four levers Module 3 pulls

A model has four scaling axes that compression can touch:

| Axis | What it is | What compresses it | When it matters |
| ---- | ---------- | ------------------ | --------------- |
| **Weights bytes** | Total bytes of model parameters | Weight quantization (INT8/INT4) | Always — limits whether the model fits |
| **Activation bytes** | Intermediate tensors during forward | Activation quantization (FP8/INT8 activations), recomputation | Long sequences, large batch |
| **KV cache bytes** | Cached K/V tensors for decoding | KV quantization (INT8/INT4 KV) | Long context, multi-user serving |
| **Compute FLOPs** | Multiplies and adds | Pruning, distillation, smaller architecture | Compute-bound regimes (training, large-batch inference) |

The first three are *memory* levers. The last one is the *compute* lever. Inference at batch 1 is memory-bound, so the memory levers move the needle. Training at batch 1024 across GPUs is compute-bound, so the compute levers move the needle there. Practice rule: if you're optimizing on-device inference, you're almost certainly fighting memory traffic, and quantization is your hammer. If you're optimizing training cost, you're fighting compute, and the architecture/distillation choices matter more.

## A worked example: Llama 3 8B on an M3 Pro

Let's price it out. The model has 8.0B parameters; in FP16 that's 16 GB of weights. The M3 Pro has 18 GB of unified memory; the model fits, barely, but leaves almost nothing for the KV cache or activations or the OS.

At INT4, the model is 4 GB. The KV cache at 8K context is 8 × 32 × 2 × 8192 × 4096 × 2 bytes / 2 (FP16 → 1 byte saved per element? no, KV cache is usually kept in FP16 unless explicitly quantized) — without KV cache quantization, ~16 GB. Even with weights at INT4, KV at 8K context eats more than the weights. Quantize KV to INT8 → 8 GB. Sliding window or shorter context → less.

The point: **quantization budgeting is multi-dimensional**, and the right plan depends on the device's memory ceiling and the workload's context profile. A model that fits at INT4 8K context might not fit at INT4 32K context unless you also quantize the KV cache. We come back to this in Lessons 3, 4, and 11.

## What you should believe after this lesson

Three sentences:

**1. Inference at batch 1 is almost always memory-bound, not compute-bound.** This is the single most important fact in on-device ML systems. Every design choice flows from it.

**2. Compression's main win is loading fewer bytes per token, not doing fewer FLOPs.** INT4 doesn't make multiplication faster; it makes the operand fetch faster.

**3. The model is not one budget — it's three (weights, activations, KV cache) plus FLOPs.** Each has its own compression strategy, and a sloppy plan picks one and ignores the others.

## Hands-on (at home)

A script that prints the FLOP and byte budget for a chosen open-weights LLM. Use `transformers` to introspect the config; we don't need to load the weights. (You will need to load them for the roofline ratio in part 3, but part 1 and 2 work on any laptop.)

```python
# flop_budget.py
from transformers import AutoConfig

def block_flops(d, f, n_layers, n_ctx, batch=1):
    # Per-block forward pass.
    attn_proj = 8 * batch * n_ctx * d**2          # Q,K,V,O projections
    attn_score = 4 * batch * n_ctx**2 * d         # QK^T + AV
    ffn = 6 * batch * n_ctx * d * (f * d)         # gate, up, down
    return (attn_proj + attn_score + ffn) * n_layers

def kv_cache_bytes(d, n_layers, n_ctx, bytes_per_elem=2):
    return 2 * n_layers * 2 * n_ctx * d * bytes_per_elem

models = {
    "meta-llama/Llama-3-8B": ("llama-3-8b", 2048),
    "meta-llama/Llama-3-70B": ("llama-3-70b", 2048),
    "Qwen/Qwen2.5-7B": ("qwen-2.5-7b", 2048),
}

for hf_id, (label, n_ctx) in models.items():
    try:
        cfg = AutoConfig.from_pretrained(hf_id)
    except Exception as e:
        print(f"  skip {hf_id}: {e}")
        continue
    d = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    # SwiGLU: ffn ratio = intermediate_size / hidden_size, applied 1.5x for gate+up+down
    f = cfg.intermediate_size / d
    flops = block_flops(d, f, n_layers, n_ctx)
    kv = kv_cache_bytes(d, n_layers, n_ctx)
    params = sum(p for p in [
        d * d * 4 * n_layers,                          # Q,K,V,O
        d * cfg.intermediate_size * 3 * n_layers,      # SwiGLU
        d * cfg.vocab_size,                            # embedding/lm_head
    ])
    print(f"{label:24s}  d={d:5d}  L={n_layers:3d}  ctx={n_ctx}")
    print(f"    FLOPs (1 forward):  {flops/1e12:.2f} TFLOPs")
    print(f"    KV cache @ FP16:    {kv/1e9:.2f} GB")
    print(f"    Weights @ FP16:     {2*params/1e9:.2f} GB")
    print(f"    Weights @ INT4:     {0.5*params/1e9:.2f} GB")
```

Run it. Sanity-check the Llama 3 8B numbers against the prose above. Then increase `n_ctx` to 32K and observe the KV cache balloon.

Part 2 — measure your own device's arithmetic-intensity ceiling.

```python
# roofline.py
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'mps'
dtype = torch.float16

# Compute peak: large square matmul, batch-1 not enough.
M = K = N = 4096
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)
for _ in range(3): a @ b
if device == 'cuda': torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): a @ b
if device == 'cuda': torch.cuda.synchronize()
tflops = (2 * M * N * K * 20) / (time.time() - t0) / 1e12
print(f"Square matmul TFLOPs: {tflops:.1f}")

# Memory-bound: 1 × d × d matmul.
v = torch.randn(1, M, device=device, dtype=dtype)
W = torch.randn(M, N, device=device, dtype=dtype)
for _ in range(3): v @ W
if device == 'cuda': torch.cuda.synchronize()
t0 = time.time()
for _ in range(200): v @ W
if device == 'cuda': torch.cuda.synchronize()
tflops_mb = (2 * 1 * N * M * 200) / (time.time() - t0) / 1e12
print(f"1xD matmul TFLOPs:   {tflops_mb:.2f}  (this is your memory-bound ceiling)")
print(f"Ratio:                {tflops/tflops_mb:.1f}x — that's how much room quantization has to recover")
```

The ratio is the multiplier quantization is *trying* to claw back. On a 4090 you'll see something like 100× between square-matmul TFLOPs and 1×D-matmul TFLOPs. INT4 won't close all of it (you're still moving bytes, just fewer of them), but it captures a 2–3× chunk of it on real decoding workloads.

## Further reading

- "The Hardware Lottery" (Sara Hooker) — for the broader frame that the hardware decides which model shapes are viable.
- Chinchilla scaling laws paper — for the training-compute math, which sits adjacent to but doesn't directly inform inference compression.
- "Efficiently Scaling Transformer Inference" (Pope et al, Google) — for the canonical breakdown of inference cost components, including the arithmetic-intensity framing applied to PaLM-scale models.
- Horace He's blog post "Making Deep Learning Go Brrrr From First Principles" — for the same memory-vs-compute framing with PyTorch-flavored examples.

Next lesson: **FP16 vs BF16 vs FP8 — the precision landscape.** Before we go to integers, we look at what's already happening in floating point — because most of what people call "INT8 quantization" is actually a hybrid that does the matmul in integer and the accumulation in FP, and you can't reason about either without a clear picture of the float side.
