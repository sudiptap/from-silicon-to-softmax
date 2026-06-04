---
title: "Lesson 4 — Mixed Precision and Per-Channel Quantization"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 4
tags: ["mixed-precision", "per-channel", "smoothquant", "calibration", "activation-quantization"]
author: "Sudipta Pathak"
prerequisites: ["03-aggressive-quantization-recipes"]
---

# Lesson 4 — Mixed Precision and Per-Channel Quantization

## Why this lesson exists

Lesson 3 was about picking among formats. This lesson is about the techniques *within* a format that recover accuracy when the format alone isn't quite enough — per-channel scaling, SmoothQuant-style migration of difficulty between activations and weights, and calibration data choices.

These techniques mostly run at conversion time, not inference time. They cost engineering effort once; the inference path stays the same. On-device specifically, they buy you the headroom to use more aggressive base formats (Q3_K_M instead of Q4_K_M) without losing perplexity, which directly translates to faster decode and more KV cache headroom.

The lesson is reading. The Hands-on applies SmoothQuant-style migration to a small model and measures the perplexity improvement at INT3.

## The per-channel insight, recapped

Module 3 Lesson 3 established that LLM activations have *outlier channels* — a small fraction (~0.1–1%) of channels carry activations 10–100× larger than the median. These outliers dominate the per-tensor max-based scale, crushing the bulk distribution into a handful of integer levels.

Three responses, in increasing sophistication:

1. **Per-channel scales.** One scale per output channel of the weight matrix. Each channel's scale is sized for its own values; outlier channels don't contaminate other channels' scales. Works fine at INT8; insufficient at INT4 because within a single channel of 4096 weights, the range still varies enough to lose information.

2. **Per-group scales.** Per-block of (typically) 128 weights within a channel. This is what Q4_K_M, AWQ, GPTQ all use. Recovers most of the per-channel-only accuracy loss.

3. **Mixed precision per layer / per channel.** Keep outlier-channel weights or activations in higher precision; quantize the rest aggressively. This is the on-device frontier in 2026.

## SmoothQuant and AWQ, on-device

SmoothQuant (Module 3 Lesson 3) is the trick of migrating outlier difficulty from activations to weights via a per-channel diagonal rescaling. AWQ (Module 3 Lesson 6) applies the same idea differently — using per-channel scaling to allocate more INT4 resolution to high-activation channels.

On-device, both techniques are useful because:

- They run entirely offline, at conversion time. No runtime cost.
- They let you use more aggressive base formats (Q3 instead of Q4) without losing the perplexity gap.
- The conversion-time cost is minutes per model on a Mac; cheap.

The practical recipe for an aggressive on-device deployment:

1. Pick the model (Lesson 2).
2. Decide on the target format (Lesson 3). Aim for one step more aggressive than you initially considered — say, Q3 instead of Q4.
3. Apply SmoothQuant or AWQ-style scaling during conversion to recover the lost accuracy.
4. Validate against the FP16 baseline; aim for perplexity within ~0.3 of FP16.

The result: a deployment that's smaller and faster than the off-the-shelf Q4_K_M variant, at comparable quality.

## The calibration-data question

For all these techniques, the conversion uses calibration data — a small sample (typically 128–1024 examples of 512 tokens each) of representative text that the algorithm uses to estimate activation distributions and pick the scales.

The right calibration data:

- **For general-purpose models**, use a mix from C4 / WikiText / The Pile. The standard recipe.
- **For task-specific models**, use task-specific calibration data. A code model should be calibrated on code; a multilingual model should include multiple languages.
- **For chat models**, use chat-formatted data (with the model's chat template applied) so the activations match the deployment distribution.

The wrong calibration data:

- Using only short, simple prompts when the deployment will see long, complex ones. Activation outliers may be different.
- Using only the model's pretraining-style data (raw text) when the deployment is chat. The chat-template tokens (`<|im_start|>`, etc.) produce different activations.
- Skipping calibration entirely (the "naive" quantization path). Saves engineering time; costs 0.3–1.0 perplexity for the cheaper algorithms.

The middle ground: there's diminishing return past ~512 calibration samples for most models. 128 is the floor; 512 is the standard; 1024+ is rarely worth the extra time.

## Per-layer quantization sensitivity, measured

A useful experiment: quantize one layer at a time, measure the perplexity hit. The most-sensitive layers are easy to identify.

For a typical Llama-class model:

- **Layer 0 (the first attention layer)**: very sensitive. Quantizing alone to INT4 from FP16 costs 0.2–0.5 perplexity.
- **Layers 1–4**: moderately sensitive.
- **Middle layers (5 to N-5)**: low sensitivity. Quantizing these to INT3 or INT2 costs less than expected.
- **Last few layers**: moderately sensitive again.
- **LM head**: very sensitive (since errors map directly to output logits).

This sensitivity profile is roughly what the Q4_K_M variants already exploit (keeping embedding and LM head at Q6); the custom-recipe path can extend it (e.g., keep the first and last 3 layers at Q5 and the middle layers at Q3).

The diagnostic loop:

```python
# Pseudo-code: layer sensitivity test.
baseline_ppl = eval_perplexity(model_fp16)
for layer_idx in range(n_layers):
    quantize_only(model, layer_idx, target_bits=4)
    ppl = eval_perplexity(model)
    print(f"layer {layer_idx}: perplexity {ppl:.3f} (delta {ppl - baseline_ppl:.3f})")
    restore(model, layer_idx)
```

Running this on Llama 3.2 3B takes maybe an hour and gives you a clear per-layer sensitivity profile. Use it to design custom quantization recipes.

## Activation quantization specifically

So far we've discussed weight quantization. The other half — quantizing the activations as they flow through the model — is mostly irrelevant on-device for single-stream inference (the activations are tiny per token, and you're bandwidth-bound on weights, not on activations). It matters for two niche cases:

**1. Multi-batch inference on-device** (rare): when batch > 1, activations become a more significant fraction of bandwidth and quantizing them helps. Mostly relevant for tablet-class devices serving multiple lightweight chat sessions.

**2. KV cache quantization** (covered in Lesson 6): the K and V tensors are activations from earlier steps; quantizing them is a real win for long-context workloads.

Outside these cases, leave activations in FP16 and focus the quantization effort on weights.

## A note on speculative tunings

In 2026 there are a few experimental approaches worth knowing about:

- **OmniQuant**: a learnable per-layer scale + shift that's optimized via gradient descent on calibration data. Slower to compute than AWQ/GPTQ; can give 0.1–0.2 perplexity improvement at INT4. Worth the effort for production-critical deployments.
- **HQQ**: calibration-free, no representative data needed. Fastest to apply; ~0.2 perplexity worse than AWQ at the same compression.
- **QuIP#**: Hadamard rotation + lattice coding. Better than AWQ/GPTQ at very low bits (2–3 bit). Niche but the right answer if you must go below INT3.

None of these are mainstream in 2026 production deployments. AWQ / GPTQ / Q4_K_M plus per-layer customization remains the dominant pattern.

## What you should believe after this lesson

Three sentences:

**1. Per-group scales (group=128) plus per-layer precision stepping plus SmoothQuant-style activation-aware migration is the standard recipe for on-device quantization in 2026.** This combination buys you one bit of effective precision compared to naive approaches — enough to use Q3 where you'd otherwise need Q4.

**2. Calibration data choice matters more than calibration data quantity** — use chat-formatted, task-representative samples; 128–512 examples is enough. Skipping calibration costs perplexity that's often not worth the time savings.

**3. Activation quantization is a side issue for single-stream on-device** — the bandwidth is dominated by weight loads, not activation reads. The exception is the KV cache, which is a stored activation and very much worth quantizing (Lesson 6).

## Hands-on (at home)

Apply AWQ to a small model, measure perplexity at INT3 with vs without AWQ.

```python
# awq_int3_demo.py
# pip install autoawq transformers datasets torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import time

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Configure AWQ for INT3, more aggressive than default Q4.
quant_config = {"zero_point": True, "q_group_size": 64, "w_bit": 3, "version": "GEMM"}
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Naive Q3 (no AWQ): just for comparison; do this manually since most libs default to AWQ-style.
# We'll skip the naive baseline and just measure AWQ at Q3.

model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="float16")
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(f"{model_id.split('/')[-1]}-AWQ-Q3")

# Perplexity check (very abbreviated; use lm-eval for a serious eval).
def perplexity(m, t, n=20):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total, tokens = 0.0, 0
    for i in range(n):
        text = ds[i]["text"]
        if not text.strip(): continue
        ids = t(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(m.device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            total += m(ids, labels=ids).loss.item() * ids.shape[1]
        tokens += ids.shape[1]
    return float(torch.tensor(total / tokens).exp())

print(f"AWQ Q3 perplexity: {perplexity(model, tokenizer):.3f}")
```

Compare to:
- Same model at Q4 via AWQ (the default in Lesson 6 of Module 3).
- Same model at Q3 via llama.cpp's Q3_K_M (no AWQ).

You should see AWQ-Q3 outperform plain Q3_K_M by ~0.3–0.5 perplexity, putting it roughly halfway between Q4_K_M and Q3_K_M in quality but at Q3 size.

For the layer-sensitivity test, this is more involved. The MLX or `auto-gptq` libraries each have hooks that let you quantize a single layer at a time; the experiment takes 1–2 hours but gives you a sensitivity profile you can use to design a custom recipe.

## Further reading

- "SmoothQuant" (Xiao et al, 2022) — already cited; foundational.
- "AWQ" (Lin et al, 2023) — Module 3 Lesson 6's cite.
- "OmniQuant" (Shao et al, 2023) — for the learnable-scale approach.
- "HQQ" (Badri & Lefebvre, 2023) — for the calibration-free path.
- "QuIP#" (Tseng et al, 2024) — for the Hadamard-rotation path at very low bits.

Next lesson: **Pruning and distillation as a complement.** When quantization isn't enough to fit the model, the next levers are removing weights (pruning) or training a smaller architecture from a larger teacher (distillation). We look at when each is worth the engineering cost on-device.
