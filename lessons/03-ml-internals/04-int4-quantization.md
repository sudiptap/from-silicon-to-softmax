---
title: "Lesson 4 — INT4 Quantization: Per-Channel, Per-Group, and the Formats"
date: "2026-06-04"
module: "ml-internals"
order: 4
tags: ["int4", "quantization", "gguf", "awq", "gptq", "bitpack", "group-size"]
author: "Sudipta Pathak"
prerequisites: ["03-int8-quantization"]
---

# Lesson 4 — INT4 Quantization: Per-Channel, Per-Group, and the Formats

## Why this lesson exists

INT8 is the format that almost always works. INT4 is the format that *makes the model fit on your laptop*. The compression factor goes from 2× (FP16 → INT8) to 4× (FP16 → INT4) — Llama 3 8B drops from 16 GB to 4 GB. That's the difference between "fits with room to spare on a 16 GB MacBook with 8 GB free for the OS" and "doesn't fit at all." Every meaningful on-device LLM in 2026 ships at INT4 or smaller.

But INT4 is also where naive approaches stop working. 4 bits is 16 levels of resolution. A per-channel scale (which sufficed at INT8) is no longer enough — within a single channel of 4096 weights, the range varies enough that quantizing all 4096 to the same 16 levels destroys too much information. The fix is a finer granularity (per-group scales) and smarter scale selection (AWQ, GPTQ — coming in Lessons 6 and 7). This lesson establishes the format and the layout; the algorithms that pick the scales well come after.

There's also a real *file-format* component to INT4 that doesn't exist at INT8. INT8 is a byte; you can store it in standard NumPy and PyTorch tensors. INT4 is half a byte; it has to be packed two-per-byte, which means every framework defines its own packing convention. GGUF (`llama.cpp`), AWQ packing, GPTQ packing, and bitsandbytes NF4 each have a slightly different on-disk layout. This lesson teaches the layouts.

The lesson is reading. The Hands-on packs a real Llama linear layer into a custom INT4 format and verifies the round-trip.

## Why per-channel collapses at INT4

Recall the INT8 setup from Lesson 3: per output channel, you store one scale `s_c`, and within that channel you have 256 quantization levels (-128 to 127, or 0 to 255). If the channel's value distribution has a heavy tail or outliers, 256 levels is still enough resolution to represent the bulk distribution and the tail.

At INT4, you have 16 levels (-8 to 7). The same heavy-tail problem now means the bulk distribution gets compressed into 2 or 3 levels while the tail gets the rest. The model's effective resolution collapses. Empirically, naive per-channel INT4 on Llama 3 8B causes a perplexity jump of 1–3 points (catastrophic in LLM terms).

The fix: shrink the unit over which one scale applies. Per-group INT4 with group size 128 means you store one scale per 128 consecutive input-dimension elements. If a tensor has shape `[4096, 4096]`, instead of 4096 scales (one per output channel), you have `4096 × (4096/128) = 131072` scales. Each scale serves 128 values, which is small enough that the local value distribution is well-behaved even when the channel-wide distribution has outliers.

Storage cost of the scales: at FP16, 131072 × 2 = 262 KB per matrix. Compared to the weight tensor itself at INT4 (~8 MB), the scales are 3% overhead. Acceptable.

Group sizes in the wild: 32 (most aggressive, smallest groups, used in GGUF Q4_K), 64 (AWQ default), 128 (GPTQ default, most common). Smaller groups → better accuracy → larger scale overhead. The 128 default is the empirical sweet spot for 7B–70B parameter models.

## The bitpack

INT4 weights are stored two per byte. The packing convention varies:

**Convention A (low-nibble-first):** byte `i` holds `q[2i]` in bits 0–3 and `q[2i+1]` in bits 4–7. The natural "little-endian nibble" layout.

**Convention B (high-nibble-first):** byte `i` holds `q[2i]` in bits 4–7 and `q[2i+1]` in bits 0–3. Used by some CUDA kernels because the high-nibble unpacks faster on certain hardware.

**Convention C (interleaved-128):** GPTQ packs 8 INT4 values into a single INT32 word, packed in some specific order designed for the matmul kernel's loading pattern. Looks weird if you just print bits; makes sense if you know how the kernel reads them.

The unsigned vs signed convention also varies. Some formats store INT4 as unsigned `[0, 15]` with a per-group zero-point (asymmetric quantization). Some store as signed `[-8, 7]` with no zero-point (symmetric).

Practically: you should never pack/unpack INT4 by hand for production code. Use the format library (`gguf`, `auto-gptq`, `autoawq`, `bitsandbytes`). What this lesson teaches is the *mental model* so you can read someone else's format spec when you have to debug a conversion.

## The major formats

A tour of what you'll encounter:

### GGUF Q4_0 (llama.cpp)

The simplest GGUF quant. Block size 32, symmetric, one FP16 scale per block. On-disk layout: for every 32 INT4 values, one FP16 scale precedes them.

```
[scale_0: 2 bytes] [16 packed bytes: 32 INT4 values] [scale_1: 2 bytes] [16 packed bytes] ...
```

Effective bits per weight: 4 + (16/32) = 4.5. The 0.5 bit of overhead is the per-block FP16 scale.

### GGUF Q4_K_M (llama.cpp)

The current default for llama.cpp's INT4. More complex: block of 256 values, broken into 8 sub-blocks of 32, with a *super-block scale* (FP16) and *sub-block scales* (6-bit, packed). Effective bits per weight: ~4.5, but distributed more carefully than Q4_0; better perplexity at the same compression. Also designates some weight matrices (the most sensitive, like the embedding) to be quantized at higher precision (Q6_K), hence the "_M" (medium) designation. There are also Q4_K_S (small, more aggressive) and Q4_K_L (large, less aggressive).

### AWQ (4-bit, group=128)

Stores INT4 weights as `[N, K/8]` of INT32 (8 INT4 values per word) plus per-group FP16 scales of shape `[N, K/128]` plus per-group INT8 zero-points of shape `[N, K/128]`. The AWQ packing of 8 values per INT32 follows a specific permutation that the CUDA kernel expects. The "AW" in AWQ — activation-aware — refers to the *scale selection* (Lesson 6), not the packing.

### GPTQ (4-bit, group=128)

Same per-group structure as AWQ but with a different packing permutation and slightly different scale-selection algorithm (Hessian-driven; Lesson 7). The packed layout: `[N, K/8]` of INT32 weights, `[N, K/128]` of FP16 scales, `[N, K/128]` of INT8 zero-points.

### bitsandbytes NF4 (4-bit, non-linear)

A different design entirely. Instead of evenly-spaced quantization levels, NF4 (Normal Float 4) uses 16 *unevenly-spaced* levels chosen to match the empirical distribution of LLM weights (which is approximately Gaussian, so more levels are placed near zero where most weights live). The packing is straightforward (2 per byte), the levels are a fixed lookup table, and per-block scales (block=64) handle the per-block range. NF4 is the default for QLoRA training.

The choice between formats:

- **llama.cpp / on-device** → GGUF Q4_K_M
- **PyTorch + Hugging Face** → AWQ or GPTQ (`auto-gptq`, `autoawq`)
- **vLLM serving** → AWQ, GPTQ, or FP8 depending on the model
- **QLoRA fine-tuning** → NF4 (via bitsandbytes)

The accuracy delta between Q4_K_M, AWQ, and GPTQ on most LLMs is small (~0.1–0.3 perplexity). The choice is mostly determined by the runtime you're deploying to, not by accuracy.

## The dequant-fused matmul kernel

The INT4 matmul kernel is structurally the same as INT8 (Lesson 3) with an extra unpacking step. For a per-group quantized weight matrix `W` (group size 128, per-group scale `s_g` and per-group zero `z_g`), the math per output channel `c` and per batch index `b`:

```
W[c, k] ≈ s_g[c, k/128] × (q_w[c, k] - z_g[c, k/128])

Y[c, b] = sum_k W[c, k] × A[b, k]
        = sum_g s_g[c, g] × sum_{k in group g} (q_w[c, k] - z_g[c, g]) × A[b, k]
```

The inner sum is INT4 × FP16 (activations stay in FP16 for W4A16) → FP16 accumulator → per-group scale folded in at the end. The kernel streams INT4 weights from HBM at maximum bandwidth, unpacks two-per-byte on-chip, does the math at FP16. The bandwidth win is the full 4× (FP16 → INT4); the compute side is unchanged.

For W4A4 (both INT4) deployments — much rarer — the math gets harder because INT4 × INT4 doesn't have native tensor-core support on most chips. Some research kernels handle it; production almost never does.

## What you should believe after this lesson

Three sentences:

**1. Per-channel granularity that worked at INT8 stops working at INT4; per-group with group size 128 is the empirical sweet spot for most models.** The scale overhead is small (~3% of the weight tensor); the accuracy recovery is enormous.

**2. Every INT4 format has its own packing convention** (Q4_0, Q4_K_M, AWQ, GPTQ, NF4) and you should never roll your own — use the format library that ships with your runtime.

**3. INT4 matmul is the same memory-bandwidth game as INT8, just more aggressive** — 4× weight compression in exchange for unpacking work that happens on-chip and effectively for free.

## Hands-on (at home)

A from-scratch per-group INT4 quantization and round-trip check. Not for production — just to build the muscle memory for the layout.

```python
# int4_quant.py
import torch
import torch.nn as nn

torch.manual_seed(0)

def quantize_int4_per_group(W: torch.Tensor, group_size: int = 128):
    """Symmetric per-group INT4 quantization. W is [out, in] FP16/FP32."""
    out, in_ = W.shape
    assert in_ % group_size == 0
    # Reshape so the last axis is the group.
    Wg = W.reshape(out, in_ // group_size, group_size)
    # Per-group scale: max |w| / 7 (signed INT4 is -8..7).
    s = Wg.abs().amax(dim=-1, keepdim=True) / 7.0   # [out, in/group, 1]
    q = (Wg / s).round().clamp(-8, 7).to(torch.int8)
    return q.reshape(out, in_), s.squeeze(-1)        # int8 storing INT4 values

def dequantize_int4_per_group(q: torch.Tensor, s: torch.Tensor, group_size: int = 128):
    """Inverse. q is int8 of shape [out, in], s is [out, in/group]."""
    out, in_ = q.shape
    qg = q.float().reshape(out, in_ // group_size, group_size)
    sg = s.unsqueeze(-1)
    Wg = qg * sg
    return Wg.reshape(out, in_)

def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack 2 INT4 values per byte (low-nibble first)."""
    assert q.dtype == torch.int8
    out, in_ = q.shape
    assert in_ % 2 == 0
    q_u = (q + 8).to(torch.uint8)  # shift to [0, 15] for clean packing
    low = q_u[:, 0::2]
    high = q_u[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed  # [out, in/2]

def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of pack_int4."""
    low = packed & 0xf
    high = (packed >> 4) & 0xf
    q_u = torch.zeros(packed.shape[0], in_features, dtype=torch.uint8)
    q_u[:, 0::2] = low
    q_u[:, 1::2] = high
    return (q_u.to(torch.int16) - 8).to(torch.int8)

# Test on a Llama-sized linear layer.
W = torch.randn(4096, 4096, dtype=torch.float32) * 0.02

q, s = quantize_int4_per_group(W, group_size=128)
packed = pack_int4(q)
q_back = unpack_int4(packed, 4096)
W_back = dequantize_int4_per_group(q_back, s, group_size=128)

print(f"original: {W.shape}, {W.element_size()*W.numel()/1e6:.2f} MB FP32")
print(f"packed:   {packed.shape}, {packed.numel()/1e6:.2f} MB (uint8 storing INT4)")
print(f"scales:   {s.shape}, {s.element_size()*s.numel()/1e6:.4f} MB")
print(f"round-trip max abs error: {(W - W_back).abs().max().item():.4e}")
print(f"round-trip mean abs error: {(W - W_back).abs().mean().item():.4e}")

# Compare to per-channel (group_size = in_features).
q_pc, s_pc = quantize_int4_per_group(W, group_size=4096)
W_pc = dequantize_int4_per_group(q_pc, s_pc, group_size=4096)
print(f"\nper-channel INT4 max abs error: {(W - W_pc).abs().max().item():.4e}")
print(f"per-channel INT4 mean abs error: {(W - W_pc).abs().mean().item():.4e}")
print("(Per-group should be ~10x better on max error.)")
```

You should see per-group max error ~1e-3 and per-channel max error ~1e-2. On a real LLM, that 10× difference is the gap between "model works" and "model outputs garbage."

Part 2 (laptop, requires `llama.cpp`): convert a real model.

```bash
# Convert Llama 3.2 1B (small enough for quick experimentation) to GGUF Q4_K_M.
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
# Download the FP16 GGUF first from HuggingFace, then quantize:
./quantize models/llama-3.2-1b-instruct-f16.gguf models/llama-3.2-1b-instruct-q4_k_m.gguf Q4_K_M
ls -la models/                # See the 4x size reduction
./main -m models/llama-3.2-1b-instruct-q4_k_m.gguf -p "The quick brown fox"
```

Read the converter output. It tells you which layers it quantized at Q4 vs Q6 vs F16 (token embedding and output projection are usually kept at higher precision — they're sensitive and small).

## Further reading

- `llama.cpp`'s `ggml-quants.c` — the canonical source for Q4_0 / Q4_K_M layouts. Heavy reading but definitive.
- AWQ paper (Lin et al, 2023) — Section on packing format and the kernel.
- GPTQ paper (Frantar et al, 2022) — Algorithm in Section 3; packing detail in the code release.
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al, 2023) — the NF4 design rationale.
- Tim Dettmers' blog on 4-bit quantization — for the broader case for INT4 on consumer hardware.

Next lesson: **Calibration data and quantization-aware training basics.** Now that we know the formats, we look at how to pick the scales well. The naive max-based scales we used above are the starting point; calibration-driven percentile or MSE-minimizing scales are what good recipes use, and QAT is the "if all else fails" option.
