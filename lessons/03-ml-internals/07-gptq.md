---
title: "Lesson 7 — GPTQ: Error-Correcting Quantization"
date: "2026-06-04"
module: "ml-internals"
order: 7
tags: ["gptq", "quantization", "int4", "hessian", "obs", "error-correction"]
author: "Sudipta Pathak"
prerequisites: ["06-awq"]
---

# Lesson 7 — GPTQ: Error-Correcting Quantization

## Why this lesson exists

AWQ (Lesson 6) gets to deployable INT4 by pre-scaling weights so the high-activation columns occupy more of the integer grid. GPTQ gets to the same place via a different mechanism: quantize the weights one column at a time, and after each column, *adjust the remaining columns* to compensate for the rounding error you just introduced. Where AWQ is "make the quantization grid better-aligned to the weights you're quantizing," GPTQ is "let the quantization grid be what it is, but spread the errors out so they cancel."

Both algorithms produce INT4 models within ~0.2 perplexity of FP16; both ship at production scale in 2026. GPTQ is the older of the two (2022 vs 2023 for AWQ), is the default in the most widely-deployed serving stacks (vLLM, TGI), and has a deeper theoretical pedigree — it's a direct descendant of the Optimal Brain Surgeon / Optimal Brain Quantizer line of work from the 1990s. This lesson is the algorithm, its origin, and what it means in practice.

The lesson is reading. The Hands-on uses `auto-gptq` to quantize a real model and compares the result to AWQ from Lesson 6.

## The lineage: Optimal Brain Surgeon

In 1993, Hassibi and Stork proposed Optimal Brain Surgeon (OBS) for *pruning* neural nets. The idea: remove a weight, then adjust the remaining weights to minimize the increase in loss. They derived a closed-form update using the inverse Hessian of the loss. For a single weight `w_q` you decide to remove:

```
δw = -w_q / H⁻¹[q, q] × H⁻¹[:, q]
```

The vector `δw` is the optimal update to all other weights given that `w_q` is forced to 0. The increase in loss is `w_q² / (2 × H⁻¹[q, q])`. So you can iteratively pick weights to prune by lowest expected loss increase, and adjust the rest.

OBQ (Optimal Brain Quantizer, Frantar & Alistarh 2022) generalizes this from "force `w_q` to 0" to "force `w_q` to its nearest quantization level." The same Hessian-based update applies. OBQ produced excellent quantization quality but was prohibitively expensive (cubic in the number of weights) for billion-parameter models.

GPTQ (Frantar et al, 2022, originally for OPT-175B) is OBQ made tractable for LLMs through three key approximations:

1. **Layer-wise:** quantize one linear layer at a time, treating each as an independent reconstruction problem. The "Hessian" becomes the empirical second-moment matrix of the layer's inputs, `H = 2 X X^T / N`, computed from calibration data. This avoids needing the true loss Hessian (which would require backprop through the whole network).

2. **Column-by-column ordering:** within a layer, quantize columns of the weight matrix one at a time in a specific order. The error from quantizing column `q` is spread into the remaining unquantized columns via the Hessian update.

3. **Cholesky decomposition trick:** the inverse Hessian update can be reformulated using a Cholesky factorization, which makes the per-column update O(d²) instead of O(d³). This is what makes GPTQ tractable on Llama-class models.

The result: an algorithm that quantizes a 7B model in ~30–90 minutes on a single GPU and produces INT4 weights with perplexity within ~0.2 of the FP16 baseline.

## The algorithm in one paragraph

For each linear layer `Y = W X` (we use weights `W` of shape `[out, in]` and inputs `X` of shape `[in, n_samples]`):

1. Run a forward pass on calibration data, capturing `X` for this layer.
2. Compute `H = 2 X X^T / N` (a `[in, in]` matrix). Add a small ridge `λ I` for stability.
3. Compute the Cholesky factor of `H⁻¹` (cheaper than `H⁻¹` directly).
4. For each output channel (row of `W`) independently:
   a. For each input dimension (column of `W`) in order:
      - Round `W[c, q]` to its nearest INT4 level.
      - Compute the rounding error `e = W[c, q] - quantized`.
      - Update all remaining `W[c, q+1:]` by `-e × cholesky_factor[q, q+1:]`.
   b. Store the quantized row.
5. Done with this layer. Move to the next.

The per-output-channel processing is independent, so it can be parallelized across the output dimension. The within-channel sequential dependency (each column's update depends on the previous columns' errors being propagated) is what makes GPTQ stateful — you can't trivially parallelize across columns.

The "in order" choice matters; GPTQ uses *activation-order* — quantize columns in order of decreasing diagonal of the Hessian (i.e., the columns with the largest activations first). This is the "act-order" variant, which is what virtually all production GPTQ deployments use. It gives ~0.1 perplexity better than no-ordering.

## What error correction does

The Hessian update `-e × H⁻¹[q, q+1:]` is the analytic expression for "given that column `q` was rounded by `e`, what's the optimal adjustment to the remaining columns to minimize the output reconstruction error?"

Concretely: suppose `H[q, q+1]` is large, meaning columns `q` and `q+1` of `X` are correlated. If column `q` rounded up by some `e`, the output got pushed up by `e × X[q]`. To cancel that, you'd want to round column `q+1` down — and you'd want to round it down *more* if its input is more correlated with column `q`'s input. The Hessian-update formula picks the exact magnitude.

The visible effect: GPTQ's per-weight rounding errors look noisy and chaotic if you inspect them individually, but the *cumulative* error on the layer's output is much smaller than naive rounding. The errors are deliberately structured to cancel.

## GPTQ in practice

The standard GPTQ recipe:

- **Format:** 4-bit, group_size=128, symmetric quantization, act-order ordering.
- **Calibration data:** 128–1024 samples from C4, WikiText, or task-relevant data. 512 samples × 512 tokens is the typical recipe.
- **Time:** 30–90 minutes for Llama-7B on a 4090. ~3–8 hours for Llama-70B.
- **Output:** INT4 weights packed in GPTQ's format (`auto-gptq` library handles serialization), with per-group FP16 scales and INT8 zero-points.

The library landscape:

- **`auto-gptq`**: the standard Python library, supports most HF models, integrates with Transformers.
- **AutoGPTQ-style kernels in vLLM, TGI, llama.cpp** (GGUF has a GPTQ-style algorithm internally): the format is widely interoperable.
- **ExLlamaV2**: an even faster GPTQ kernel implementation for NVIDIA inference; widely used in the local LLM scene.

## AWQ vs GPTQ: when to pick which

This is the most common practical question. The honest answer in 2026:

**They are nearly interchangeable for inference quality on most Llama-class models.** Perplexity differences are within run-to-run noise (a few hundredths to a tenth of a point). Throughput differences depend more on the kernel implementation than the format.

The decision tree:

- **You're using a runtime that supports one and not the other** → use what's supported. Most do support both now (vLLM, TGI, ExLlama), so this rarely binds.
- **You need to quantize quickly (one-shot deployment)** → AWQ. 5–15 min for 7B vs 30–90 min for GPTQ.
- **You're squeezing out the last 0.1 perplexity** → try both, eval, pick the winner. The winner varies per model.
- **You're quantizing a model with a non-standard architecture** → GPTQ tends to be more robust because it's a layer-by-layer algorithm without strong activation-distribution assumptions. AWQ relies on the salient-channels structure being prominent.
- **You're targeting on-device (llama.cpp)** → GGUF Q4_K_M, which uses a GPTQ-inspired but distinct algorithm internally.

## What's behind the wins (and what isn't)

What GPTQ does buy you:
- Robust handling of all weights, including the boring middle of the distribution. AWQ focuses on salient channels and lets the rest take naive INT4; GPTQ optimizes everywhere.
- A clean theoretical framework (OBQ heritage) that makes the algorithm easy to reason about and modify.
- Format compatibility with the widely-deployed GGUF Q4_K_M family.

What it doesn't buy you:
- Much over AWQ on standard LLMs. The headline "GPTQ vs AWQ" comparison hides that both are ~95% of the way from naive INT4 back to FP16.
- A free lunch at sub-INT4. Both GPTQ and AWQ collapse at INT2 / INT3; the algorithms that work there (QuIP#, OmniQuant) are different.

## What you should believe after this lesson

Three sentences:

**1. GPTQ quantizes columns one at a time and propagates the rounding error into the remaining columns via a Hessian-based update, so that the cumulative layer-output error is minimized rather than each weight's individual rounding error.** It's "error-correcting" in the literal sense: each weight's error is mostly canceled by adjustments to others.

**2. GPTQ and AWQ are the two production INT4 algorithms; they're nearly interchangeable in quality, with AWQ being faster to run and GPTQ being more robust to non-standard architectures.** Pick by runtime support, calibration time budget, and a quick eval on the specific model.

**3. GPTQ's lineage (OBS → OBQ → GPTQ) is one of the cleaner examples of an old idea finding a new application** — a 1993 pruning algorithm becoming the default LLM quantization algorithm in 2024, made tractable by a Cholesky trick.

## Hands-on (at home)

Quantize a real model with GPTQ and compare to the AWQ result from Lesson 6.

```python
# gptq_quantize.py
# pip install auto-gptq optimum
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "Qwen/Qwen2.5-1.5B"
quant_path = "Qwen2.5-1.5B-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPTQConfig handles calibration internally; provide a dataset or rely on c4.
quant_cfg = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=True,  # act-order
    tokenizer=tokenizer,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=quant_cfg,
)
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Saved to {quant_path}")
```

This is slower than AWQ — expect 15–30 minutes for Qwen 1.5B on a 4090.

Compare perplexity to the AWQ version from Lesson 6:

```python
# gptq_vs_awq_ppl.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(model, tokenizer, n_samples=64, max_len=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total_loss, total_tokens = 0.0, 0
    for i in range(n_samples):
        text = ds[i]["text"]
        if not text.strip(): continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            out = model(ids, labels=ids)
        total_loss += out.loss.item() * ids.shape[1]
        total_tokens += ids.shape[1]
    return float(torch.tensor(total_loss / total_tokens).exp())

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
for name, path in [
    ("FP16", "Qwen/Qwen2.5-1.5B"),
    ("AWQ", "Qwen2.5-1.5B-AWQ"),
    ("GPTQ", "Qwen2.5-1.5B-GPTQ"),
]:
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda")
    ppl = perplexity(model, tok)
    print(f"{name:6s}: ppl = {ppl:.3f}")
    del model
    torch.cuda.empty_cache()
```

You should see all three within ~0.3 perplexity of each other, with the INT4 variants each within ~0.2 of the FP16 baseline. The relative ordering of AWQ vs GPTQ can flip from model to model; on Qwen 2.5 1.5B, expect them to be statistically tied.

For a richer comparison, also run MMLU or a domain-specific eval (HumanEval, GSM8K) — these often show more separation than WikiText perplexity.

## Further reading

- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al, 2022) — the paper. Section 3 has the algorithm pseudocode; Section 4 has the Cholesky trick.
- "Optimal Brain Surgeon and General Network Pruning" (Hassibi & Stork, 1993) — the 30-year-old lineage piece, well worth reading for the original derivation.
- "OBQ: Optimal Brain Quantization" (Frantar & Alistarh, 2022) — the direct precursor to GPTQ.
- `auto-gptq` GitHub README — for the API and the kernel implementations.
- ExLlamaV2 README — for a high-performance GPTQ inference kernel widely used in local LLM hosting.

Next lesson: **Pruning — Structured vs Unstructured, Lottery Tickets.** We pivot from quantization (changing the precision) to pruning (removing weights entirely). The pruning story is less exciting than quantization in 2026 — most of the practical wins come from structured N:M sparsity, and even that is a junior partner to INT4 quantization — but understanding why pruning matters less than it once did is itself a lesson in what compression is for.
