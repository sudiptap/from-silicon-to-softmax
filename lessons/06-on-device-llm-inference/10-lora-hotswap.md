---
title: "Lesson 10 — LoRA Hot-Swap"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 10
tags: ["lora", "adapter", "hot-swap", "fine-tuning", "merge", "apple-foundation"]
author: "Sudipta Pathak"
prerequisites: ["09-speculative-decoding"]
---

# Lesson 10 — LoRA Hot-Swap

## Why this lesson exists

A common deployment pattern: you want to serve N specialized variants of a base model — different personas, task specializations, domain experts — without paying N times the memory cost. Ship N full models (5 GB each at 3B INT4): 25 GB just for 5 variants. Not viable on-device.

The solution: ship the base model once, plus N small "adapters" that customize it. LoRA (Low-Rank Adaptation) is the canonical adapter mechanism: ~10–100 MB per adapter for a 3B base. At request time, swap in the relevant adapter; the base stays resident. Apple's on-device foundation model is built around exactly this pattern — the OS ships one base model, and individual apps' specializations are tiny adapters.

This lesson covers LoRA hot-swap from the on-device deployment perspective: the math behind LoRA, the merge-vs-keep-separate tradeoff, the operational patterns, and Apple's specific implementation.

The lesson is reading. The Hands-on swaps two LoRA adapters on a base model in MLX and measures the latency cost.

## What LoRA is

LoRA (Hu et al, 2021) trains a *low-rank update* on top of a frozen base model. For each weight matrix `W` of shape `[out, in]`, LoRA learns two small matrices `A` of shape `[r, in]` and `B` of shape `[out, r]` such that the effective weight is `W + B @ A`. The rank `r` is small — typically 4 to 64.

Parameter count comparison:
- Original `W`: `out × in` parameters. For a 4096 × 4096 layer, 16M params.
- LoRA update `B @ A`: `out × r + r × in` = `(out + in) × r` params. For `r=16`, 131k params. ~125× smaller.

Summing across all the attention and FFN layers in a 3B model: a typical LoRA at rank 16 is ~10–30 MB total. At FP16. Negligible compared to the base.

The trained adapter is just `(A, B)` for each adapted weight matrix. During inference, you either:

1. **Merge**: compute `W_eff = W + B @ A` once at load time. Same memory as the base model; same inference path; you can only use one adapter at a time.

2. **Keep separate**: store `W` and `(A, B)` separately. During each matmul, compute `x @ W + (x @ A^T) @ B^T`. Two matmuls instead of one; one of them is cheap (rank-r). Memory cost is base + multiple adapters, but you can switch adapters instantly.

The choice between merge and keep-separate is the central LoRA hot-swap question.

## The merge-vs-keep-separate tradeoff

**Merge:**
- Pros: zero inference-time overhead; the merged model is indistinguishable from a fully-trained model.
- Cons: switching adapters requires re-loading or re-merging the base. Either keep N merged models in memory (expensive) or pay the merge cost (~seconds for a 3B model) on each switch.

**Keep separate:**
- Pros: instant adapter switch (just point to a different adapter file). Multiple adapters can be active simultaneously (e.g., apply two compatible adapters).
- Cons: per-layer inference overhead. For typical ranks the overhead is 5–15% throughput.

For deployments that pick an adapter at session start and stay on it (chat app where the user picks a persona once, then chats), merge wins.

For deployments that need to switch frequently or use multiple adapters in one session, keep-separate wins. The throughput cost is real but the operational flexibility is worth it.

## The on-device serving pattern

A common Apple-Intelligence-style pattern:

1. The OS ships one base model (~3B parameters at INT4, ~1.5 GB).
2. Individual apps ship adapter files (~10–100 MB each).
3. When an app requests inference, the runtime loads the app's adapter on top of the base.
4. Inference runs with the adapter applied (either merged or kept-separate, depending on the runtime).
5. When the next app requests inference, swap to its adapter.

Memory advantage: N apps share one base. Each app pays ~10–100 MB for its adapter instead of ~1.5 GB.

Operational considerations:
- The adapter must be compatible with the base (same architecture, same hidden dim, etc.). Apps can't bring arbitrary adapters; they must be trained against the OS's base model.
- The OS controls the base; updates ship via OS update. Apps don't control base version.
- For Apple specifically, the Foundation Models framework exposes a curated set of adapter slots; you can't ship arbitrary LoRA weights.

For non-Apple deployments (your own server or app), you have more flexibility but the same fundamental tradeoffs.

## When LoRA hot-swap is the right pattern

The clear cases:

**1. Multiple personas / brands.** A chat app that supports several conversational styles (cheerful assistant, professional expert, sarcastic friend). Each persona is an adapter; all share the base.

**2. Per-user adaptation.** Users can fine-tune their own persona (small calibration set, brief training). The result is a per-user adapter the size of a small image file.

**3. Per-task specialization.** Code completion, math solving, document QA each get their own adapter. The base handles general; adapters handle specialization.

**4. A/B testing.** Ship multiple adapter variants; route different users to different ones; measure outcomes. Much cheaper than shipping multiple full models.

When LoRA hot-swap is the wrong pattern:

- **You need a fundamentally different model architecture.** LoRA is a fine-tuning technique; it doesn't change the architecture. If your task requires a different base, LoRA won't help.
- **You need extreme specialization.** A LoRA adapter can shift a model's style and behavior but can't teach radically new capabilities. For "I need this model to be world-class at this narrow task," full fine-tuning or distillation may be needed.
- **You only have one model variant.** No point in the infrastructure if N=1.

## QLoRA: training LoRA on a quantized base

A separate but related technique: QLoRA (Dettmers et al, 2023) trains LoRA adapters on a quantized base model. The base stays quantized (INT4 / NF4); only the LoRA `A` and `B` matrices are in FP16 and trainable. This makes fine-tuning possible on memory-constrained hardware — a 7B base + small LoRA fits in ~5 GB of GPU memory.

For on-device deployment, QLoRA matters because it makes per-user fine-tuning feasible on consumer hardware. A user trains a personalization adapter locally on their laptop in minutes; the result is a small adapter file they own.

The MLX `mlx-lm.lora` command supports QLoRA-style training natively:

```bash
mlx_lm.lora \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train --data path/to/user_data.jsonl \
    --iters 1000 --batch-size 4 --lora-rank 16
```

The output is a `.safetensors` adapter file (~10 MB). Load it at inference time:

```bash
mlx_lm.generate --model meta-llama/Llama-3.2-3B-Instruct \
    --adapter-path path/to/adapter \
    --prompt "..."
```

## Adapter file formats

The adapter format isn't standardized; each framework uses its own:

- **HuggingFace PEFT**: `adapter_model.bin` or `adapter_model.safetensors` plus an `adapter_config.json`. The de-facto standard for PyTorch-trained adapters.
- **MLX**: `safetensors` with MLX-specific naming conventions.
- **llama.cpp**: GGUF-format LoRA adapters (less common; the format supports it but the ecosystem mostly uses PEFT format).

Conversion between formats is generally possible via the model conversion tools, with the usual caveats around opset / op coverage.

Apple's Foundation Models framework uses its own internal adapter format; you can't directly ship a PEFT adapter to it. The "adapters" Apple supports are a curated set of behaviors the OS provides.

## Multi-LoRA serving

A more advanced pattern: serve *multiple* adapters simultaneously. Two variants:

**1. Adapter composition.** Apply two compatible adapters at once: `W_eff = W + B1@A1 + B2@A2`. Useful for combining a "persona" adapter with a "task" adapter (e.g., "math solving, in a cheerful style").

**2. Per-request adapter routing.** A server-side pattern: receive a request with an adapter ID; load the corresponding adapter; serve the request. Used in vLLM-style multi-tenant serving where each request might use a different fine-tune.

For on-device, composition is occasionally useful; per-request routing is rarely relevant (you're serving one user, who has at most one adapter active at a time).

## What you should believe after this lesson

Three sentences:

**1. LoRA adapters are ~100× smaller than the base model** (10–100 MB for a 3B base) and let you share one resident base across multiple specialized variants — personas, tasks, per-user customizations. This is the core mechanism behind Apple's on-device Foundation Models and similar one-base-many-adapters deployment patterns.

**2. The merge-vs-keep-separate tradeoff is the central choice**: merge for zero inference overhead at the cost of slow adapter switching; keep-separate for instant switching at ~5–15% throughput cost. For single-persona-per-session apps, merge wins; for multi-task apps, keep-separate wins.

**3. QLoRA makes per-user adapter training feasible on consumer hardware** — a user trains a personalization adapter on their laptop in minutes, produces a small file they own. This is the foundation for personalized on-device LLM behavior.

## Hands-on (at home)

Apply two different LoRA adapters to the same base in MLX.

```bash
# Train two adapters (or download pre-trained ones if available).
# We'll fake-train with a tiny dataset for demonstration; in practice use real data.
echo '{"text": "User: Tell me about cooking\nAssistant: I love sharing cooking tips!"}' > cooking.jsonl
echo '{"text": "User: Tell me about coding\nAssistant: I enjoy precise technical answers."}' > coding.jsonl

mlx_lm.lora \
    --model mlx-community/Qwen2.5-0.5B-Instruct \
    --train --data cooking.jsonl \
    --iters 100 --batch-size 1 --lora-rank 8 \
    --adapter-path adapters/cooking

mlx_lm.lora \
    --model mlx-community/Qwen2.5-0.5B-Instruct \
    --train --data coding.jsonl \
    --iters 100 --batch-size 1 --lora-rank 8 \
    --adapter-path adapters/coding

# Inference with each adapter.
echo "=== cooking persona ==="
mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct \
    --adapter-path adapters/cooking \
    --prompt "Tell me about Python." --max-tokens 80

echo "=== coding persona ==="
mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct \
    --adapter-path adapters/coding \
    --prompt "Tell me about Python." --max-tokens 80
```

You'll see two different responses to the same prompt — the cooking adapter biases toward warm/conversational; the coding adapter toward technical/precise. The adapter files are tiny (~5 MB each); the base model is reused.

For measuring the merge-vs-keep-separate latency:

```python
# Pseudo-code timing comparison.
# Fuse / merge:
model_merged = fuse_adapter(base, cooking_adapter)
t0 = time.time(); generate(model_merged, prompt); print("merged:", time.time() - t0)

# Keep separate (default in mlx-lm with --adapter-path):
t0 = time.time(); generate(base, prompt, adapter=cooking_adapter); print("separate:", time.time() - t0)
```

The keep-separate path is typically 5–10% slower per token. The merge path is faster per token but the merge itself takes seconds.

## Further reading

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al, 2021) — the foundational paper.
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al, 2023) — quantized-base fine-tuning.
- HuggingFace PEFT library documentation — the standard adapter ecosystem.
- Apple's "On-Device Foundation Models" / "Adapters" documentation — the OS-level multi-adapter story.
- "Apple Intelligence Foundation Language Models" technical report (2024) — Apple's own description of their on-device model + adapter architecture.

Next lesson: **On-device multimodal.** We pivot from text-only to vision-language and audio-language models. The vision encoder cost dominates the per-request budget; we look at Moondream, Florence-2, SmolVLM, Phi-3 Vision, and the engineering patterns (encoder sharing, vision caching) that make multimodal on-device feasible.
