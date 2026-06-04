---
title: "Lesson 15 — Long-Context RoPE Scaling"
date: "2026-06-04"
module: "inference-from-scratch"
order: 15
tags: ["rope", "long-context", "yarn", "ntk-aware", "linear-scaling", "longrope"]
author: "Sudipta Pathak"
prerequisites: ["14-rope"]
---

# Lesson 15 — Long-Context RoPE Scaling

## Why this lesson exists

Vanilla RoPE works well within the trained context length. Beyond it, attention scores at long positions become "novel" — the model hasn't seen these rotation patterns during training, and quality degrades. The empirical cliff is roughly the trained length: a model trained at 8K context degrades sharply at 16K.

The fix is a family of *scaling methods* that modify the RoPE frequency table so the model can use rotations it didn't train on. Linear scaling (the first attempt) compresses positions linearly. NTK-aware scaling redistributes the frequencies. YaRN combines several ideas. LongRoPE (Microsoft, 2024) goes further with learned position adjustments.

This lesson covers the scaling family, what each method does, and the practical recipes that ship in modern LLMs (Llama 3.1's 128K context, Qwen 2.5's 1M context).

The lesson is reading. The Hands-on applies linear and NTK-aware scaling and observes the perplexity impact at long context.

## The problem, more precisely

Vanilla RoPE rotates by `θ_k × position` for the `k`-th frequency. At positions beyond training:
- The fastest-rotating pairs (small `k`, high frequency) have already cycled through many full rotations during training; they've seen "all" of their angular range.
- The slowest-rotating pairs (large `k`, low frequency) haven't even completed one rotation during training. At position 32K vs training position 8K, the slowest pair has rotated 4× as much — into angular regions it never saw.

The slowest pairs are the ones that encode "long-range" position information. They're also the ones that haven't seen the long-position rotations. Quality at long context degrades because these dimensions become noise.

The scaling methods all attack this asymmetry: they modify the frequencies (especially the slow ones) so that long positions still produce in-distribution rotations.

## Linear scaling (the naive fix)

The simplest scaling: divide all positions by the scaling factor `s` before computing rotations.

```
α_k(i) = i × θ_k         # vanilla
α_k(i) = (i / s) × θ_k   # linear scaling, factor s
```

If you trained at context 8K and want 32K, set `s = 4`. Now position 32K "looks like" position 8K to the rotation; the rotation is in-distribution.

Trade-off: the resolution between adjacent positions is compressed by 4×. Position 7000 and 7001 now have rotation difference `(1/4) × θ_k` instead of `θ_k`. The model can't distinguish nearby positions as sharply.

Empirically: linear scaling extends context but degrades quality on tasks requiring fine-grained position discrimination. It's the baseline, useful for quick experiments but rarely the right production choice.

## NTK-aware scaling

NTK-aware scaling (Reddit user "bloc97" was an early proponent; named after Neural Tangent Kernel analyses) addresses the linear-scaling resolution problem:

```
θ_k(scaled) = θ_k × (s)^(-2k/(d-2))
```

This *interpolates* between the high frequencies (left mostly alone, preserving local resolution) and the low frequencies (scaled down more aggressively, accommodating long-range positions).

The intuition: high frequencies don't need scaling because they've seen all their angular range. Low frequencies need scaling because they haven't.

Result: better preservation of local-position resolution while still extending the effective context.

Code 2 paper by NousResearch + Eleuther AI: "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning." A small amount of continued pretraining further closes the quality gap.

## YaRN (Yet another RoPE extension)

YaRN (Peng et al, 2023) refines NTK-aware with several additions:

1. **NTK-by-parts**: don't scale frequencies that have completed several full rotations during training (they don't need scaling); scale frequencies that haven't completed even one rotation aggressively.
2. **Length-based attention scaling**: apply a temperature factor to attention scores at long context. Compensates for the entropy increase that comes with longer context.
3. **Brief continued pretraining**: a few thousand steps of training at the new context length to let the model adapt to the scaled rotations.

YaRN is the most rigorous of the scaling methods and is used in production. Mistral's "Mistral 7B Instruct v0.3" and Yi-200K use YaRN.

Empirically: YaRN at 8K → 32K context degrades perplexity by maybe 5-10%. At 8K → 128K, the degradation is larger but still manageable.

## LongRoPE

LongRoPE (Ding et al, Microsoft, 2024) takes a different angle: instead of analytically scaling the frequencies, *search* for the optimal per-position adjustment using a held-out dataset. The optimization finds the rotation pattern that best preserves perplexity at the long context.

LongRoPE extends Llama-class models to 2M context with quality competitive to the in-distribution baseline. It's more complex than YaRN but produces better extrapolation.

Phi-3 (Microsoft, 2024) uses LongRoPE for its long-context variants.

## What recent flagship LLMs use

Roughly:
- **Llama 3.1 / 3.2 / 3.3**: extended from 8K → 128K via YaRN-like scaling + continued pretraining.
- **Mistral / Ministral**: combined RoPE + sliding window. Sliding window bounds the actual attention range; RoPE handles position within the window.
- **Qwen 2.5**: variable approach across model sizes; 128K context via NTK-aware scaling + extended pretraining.
- **Gemma 2**: similar to Llama's approach.
- **Phi-3 (long-context)**: LongRoPE.
- **DeepSeek-V3**: YaRN-like scaling + MLA (the MLA RoPE-K is a separate small dimension with its own scaling story).

The general 2026 pattern: NTK-aware or YaRN is the default; LongRoPE for the most ambitious context lengths.

## A practical workflow

If you have a model trained at context `N_train` and want to deploy at `N_target = s × N_train`:

1. **Start with NTK-aware scaling.** Lowest engineering effort; no retraining needed.
2. **If quality is too low**, switch to YaRN with a few thousand steps of continued pretraining at the target context. Larger engineering cost but better quality.
3. **If you need extreme lengths** (4× training or beyond), consider LongRoPE.
4. **Validate** with a long-context benchmark (Needle-in-Haystack, RULER, LongBench).

Most production deployments do step 1 for moderate extensions; step 2 for serious deployments. The engineering involved is non-trivial but well-trodden.

## A subtle point on inference

The scaling is applied at inference time by modifying the cos/sin tables. The model weights don't change. So you can take a pre-trained model and apply different scaling at deploy time:

```python
# At model load:
cos, sin = precompute_freqs_cis(d_head, max_target_len,
                                 base=10000, scaling='ntk-aware', scale_factor=4)
# These tables replace the model's original cos/sin tables.
```

This is how llama.cpp, vLLM, and MLX support long-context extensions: load the model, swap in scaled tables, run.

## What you should believe after this lesson

Three sentences:

**1. Vanilla RoPE degrades sharply beyond its training context** because the long-position rotations are out-of-distribution. Scaling methods modify the RoPE frequencies so long positions produce in-distribution rotations.

**2. The progression: linear scaling (naive) → NTK-aware (interpolate frequencies) → YaRN (NTK-by-parts + attention temperature + brief retraining) → LongRoPE (learned per-position adjustments).** Each step trades engineering complexity for quality at very long context.

**3. The 2026 production default is NTK-aware or YaRN** for moderate extensions (2-8×); LongRoPE for ambitious extensions (16x+). Llama 3.1's 128K context, Mistral's long-context variants, and Qwen 2.5's 1M context all use members of this family.

## Hands-on (at home)

Apply linear and NTK-aware scaling and observe the rotation patterns.

```python
# rope_scaling.py
import torch
import math

def vanilla_freqs(d, max_len, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(max_len).float()
    return torch.outer(pos, inv_freq)

def linear_scaled_freqs(d, max_len, base=10000, scale=4):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(max_len).float() / scale  # divide by scale
    return torch.outer(pos, inv_freq)

def ntk_scaled_freqs(d, max_len, base=10000, scale=4):
    # NTK: increase the base so the frequencies stretch out.
    base_scaled = base * scale ** (d / (d - 2))
    inv_freq = 1.0 / (base_scaled ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(max_len).float()
    return torch.outer(pos, inv_freq)

d, max_len, scale = 64, 32768, 4  # extend 8K → 32K
freqs_v = vanilla_freqs(d, max_len)
freqs_l = linear_scaled_freqs(d, max_len, scale=scale)
freqs_n = ntk_scaled_freqs(d, max_len, scale=scale)

# Compare the angles at position 16000 (in the extended range).
print(f"Position 16000 angles (first 5 frequency pairs):")
print(f"  vanilla: {freqs_v[16000, :5].numpy()}")
print(f"  linear:  {freqs_l[16000, :5].numpy()}")
print(f"  ntk:     {freqs_n[16000, :5].numpy()}")
print()
print("Note: vanilla angles at 16K are 'novel' to a model trained at 8K.")
print("Linear scaling pushes them back to angles seen during training (positions 0-8K).")
print("NTK preserves high-frequency local resolution while scaling low frequencies more.")
```

For a real test of quality vs scaling method, take a small open model (Llama 3.2 1B), apply each scaling variant via `llama.cpp` flags (`--rope-scaling linear`, `--rope-scaling ntk_aware`, `--rope-scaling yarn`), run perplexity at 16K and 32K context, compare. You'll see the quality hierarchy match what the methods promise.

## Further reading

- "NTK-Aware Scaled RoPE" (bloc97, Reddit/HuggingFace 2023) — community-developed; widely cited.
- "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al, 2023).
- "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" (Ding et al, 2024).
- "RULER: What's the Real Context Size of Your Long-Context Language Models?" (Hsieh et al, 2024) — long-context evaluation methodology.
- "Llama 3.1 paper" — Meta's description of their RoPE scaling for 128K context.

End of Part 2. Next: Part 3 begins with **Why the KV cache exists** — the prefill vs decode asymmetry that defines inference, and the data structure (the KV cache) that resolves it.
