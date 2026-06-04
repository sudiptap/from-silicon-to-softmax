---
title: "Lesson 5 — Pruning and Distillation as a Complement"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 5
tags: ["pruning", "distillation", "sheared-llama", "depth-pruning", "width-pruning"]
author: "Sudipta Pathak"
prerequisites: ["04-mixed-precision"]
---

# Lesson 5 — Pruning and Distillation as a Complement

## Why this lesson exists

Quantization is the dominant compression lever. But it has a floor — beyond ~Q2, accuracy degrades fast — and sometimes the model still doesn't fit. The remaining levers are *pruning* (remove weights from an existing model) and *distillation* (train a smaller model to imitate a larger one). Module 3 Lessons 8 and 9 covered the algorithms; this lesson is the on-device deployment perspective: when is the engineering cost worth it, what does the workflow actually look like, and which approaches deliver in practice.

The headline answer: in 2026, distillation is the more productive lever for on-device LLM deployment, because the SLM landscape (Lesson 2) is rich enough that someone has usually already done the work for you. Pruning of LLMs remains rare in production deployments, though structured pruning sees occasional use in specific niches.

The lesson is reading. The Hands-on shows two approaches: load a distilled-from-scratch model (the common path), and apply Wanda-style pruning to an existing model (the rare-but-useful path).

## The hierarchy of compression effort

In ascending order of engineering cost:

1. **Pick a smaller off-the-shelf model** (Lesson 2). Zero conversion engineering; you change the model.
2. **Quantize an existing model** (Lessons 3–4). Hours of conversion engineering; you change the format.
3. **Apply post-training pruning** to an existing model. Days of engineering plus a small fine-tune to recover accuracy.
4. **Distill from scratch** into a smaller architecture. Weeks-to-months of engineering plus significant compute.
5. **Train a smaller model from scratch** with carefully curated data. Months of engineering plus large compute.

Most on-device deployments live in steps 1–2. Step 3 (pruning) appears in specific niches. Steps 4–5 are for organizations that ship at scale (Apple, Microsoft, Google) and have invested in the infrastructure.

## When pruning works on LLMs

Module 3 Lesson 8 made the case that unstructured pruning is mostly disappointing for LLMs because (a) general sparse-matmul kernels are slow on real hardware, and (b) the perplexity hit at meaningful sparsity is significant. Two specific pruning approaches escape these limitations:

**1. Layer pruning (depth pruning).** Remove whole transformer blocks. The middle blocks of a model often turn out to be quite redundant; removing 4–8 layers from a 32-layer model and fine-tuning briefly can recover most of the lost capability. The remaining model is smaller, faster (fewer layers to traverse), and uses standard dense matmul (no sparse kernel needed).

The "Sheared LLaMA" approach (Xia et al, 2023) is the canonical example: take Llama 7B, prune to ~3B by removing layers and shrinking hidden dimensions, fine-tune on a few billion tokens. The result rivals Llama 3B trained from scratch.

For on-device use: layer-pruned models are deployment-friendly because they look like any other dense model to the runtime. Most runtimes don't even know they were pruned.

**2. 2:4 structured sparsity.** Module 3 Lesson 8 covered this; rarely worth it on phones (no NPU support for it as of 2026) but real on NVIDIA-based desktops and laptops with Ampere+ tensor cores.

**3. Width pruning** (removing channels / attention heads). Similar to layer pruning in flavor — the model shrinks, no sparse kernel needed. Less commonly applied to LLMs than layer pruning; some 2024–2025 papers explore it.

The trick that makes pruning useful on-device: the pruned model is dense at a smaller size. The runtime is happy.

## When distillation works

Distillation (Module 3 Lesson 9) is the dominant compression technique for getting capable small models. The catch is that you need access to a strong teacher and you have to actually train. Two common scenarios:

**1. You have a frontier-scale teacher** (Llama 3.1 70B, Claude, GPT-4-class). Generate synthetic training data, train a small student. This is what Phi-3 / Phi-3.5 did, what Apple's on-device foundation model did, what many of the strongest SLMs did.

The cost: tens of thousands of dollars in inference (running the teacher to generate data) plus a few thousand to train the student. Possible for a well-resourced startup; not feasible for a hobbyist.

**2. You pick a distilled model someone else made.** Phi-3.5, Apple's on-device model, distilled variants of Llama published by Meta. In 2026 the "use someone else's distilled model" path is the default for almost every deployment. The hard work is done; you select from a menu.

This is why Lesson 2's model-picking is the right framing for most deployments. The distillation effort that produced Phi-3.5 mini is what makes it a credible 3.8B parameter model with 7B-class capability. You benefit from that effort by picking the model, not by replicating the training.

## When DIY distillation is worth it

There are scenarios where rolling your own distillation pays off:

- **Domain-specific tasks** where general SLMs don't perform well. Distilling a domain-specialized teacher (a fine-tuned 70B model on legal / medical / code text) into a smaller domain student.
- **Specific output formats or styles** that off-the-shelf SLMs don't produce. Distilling for "must produce JSON of this schema" or "must follow this brand voice."
- **Privacy-sensitive deployments** where you can't use any cloud teacher and need to distill from an internal model.
- **Aggressive size targets** (sub-1B) where off-the-shelf options are limited.

For these, the effort is justified. For everything else, pick an existing distilled model.

## Wanda: a recent practical pruning method

If you do choose to prune, the simplest method worth knowing is Wanda (Pruning by Weights and Activations, Sun et al, 2023). The idea: instead of pruning by magnitude alone, prune by `|weight| * ||activation||`. The product of weight magnitude and the L2 norm of the corresponding activation is a better importance estimator than weight magnitude alone.

The algorithm:

1. Compute activation statistics from calibration data (one forward pass).
2. For each linear layer, compute the importance score `|W| * ||A||` per weight.
3. Prune the lowest-importance weights (typically structured, e.g., per-output-channel groups, to keep dense matmul kernels working).
4. Brief fine-tune to recover (optional but improves quality).

Wanda is calibration-only — no gradient updates required. It runs fast (minutes for a 7B model) and is reasonably effective.

For LLMs targeting on-device, layer pruning is usually a better choice than Wanda's per-weight pruning, because layer pruning doesn't require any custom sparse kernel handling. But Wanda is the recommended approach if you're going the per-weight route.

## What to actually do on-device

Decision tree for the compression-beyond-quantization question:

- **The smaller off-the-shelf model is good enough** → use it. Skip pruning/distillation entirely.
- **There's a distilled variant of your model** (or a distilled cousin that fits your needs) → use it. The teacher work has been done.
- **You need a domain specialist and no off-the-shelf option exists** → DIY distillation. Plan for weeks of effort.
- **You need to fit a specific size and only a slightly-too-large model exists** → layer pruning. Take 4–8 layers off a 7B model, fine-tune briefly to recover.
- **You're on NVIDIA hardware with Ampere+ and care about throughput** → 2:4 structured sparsity. Small but real gain.

For most deployments the answer is "use an existing distilled model." The other options are real but reserved for specific situations.

## What you should believe after this lesson

Three sentences:

**1. The dominant compression lever on-device is "pick a smaller model"** (often a distilled one), with quantization second and pruning a distant third. The SLM landscape is rich enough that someone has usually already done the distillation work for you.

**2. Layer pruning (depth pruning) is the one pruning approach that works on-device** — it produces a smaller dense model that runs on standard kernels with no sparse-matmul slowdown. Wanda is the recommended weight-pruning method if you go that route, but it's rarely the right choice for LLMs.

**3. DIY distillation pays off only in specific scenarios** — domain specialists, brand-specific output styles, privacy-bound deployments, sub-1B targets. For general use cases, picking an off-the-shelf distilled model is more cost-effective by orders of magnitude.

## Hands-on (at home)

Two short exercises.

**1. Try an off-the-shelf distilled model.** Phi-3.5 mini is a strong distilled 3.8B model.

```bash
mlx_lm.generate --model mlx-community/Phi-3.5-mini-instruct-4bit \
    --prompt "If a train leaves at 10:00 AM at 60 mph and arrives at 1:00 PM, how far did it travel?" \
    --max-tokens 100 --temp 0
```

Compare to a non-distilled model of similar size (Llama 3.2 3B):

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --prompt "If a train leaves at 10:00 AM at 60 mph and arrives at 1:00 PM, how far did it travel?" \
    --max-tokens 100 --temp 0
```

For reasoning tasks Phi-3.5 typically wins; for chat naturalness Llama often wins. The distillation pipeline shapes the strengths.

**2. Apply Wanda-style pruning to a small model.** This is a demonstration; production usage would need more care.

```python
# wanda_demo.py
# Requires a model already loaded in PyTorch.
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Compute activation statistics for a couple of layers.
activations = {}
def make_hook(name):
    def hook(_, input, _o):
        a = input[0].detach()
        if name not in activations:
            activations[name] = torch.zeros(a.shape[-1])
        activations[name] += (a ** 2).sum(dim=(0, 1)).cpu()
    return hook

hooks = []
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and 'mlp' in name:
        hooks.append(mod.register_forward_hook(make_hook(name)))

# Run calibration data.
for prompt in ["Hello", "The weather today is", "Once upon a time"]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)

for h in hooks: h.remove()

# Wanda-style importance: |W| * sqrt(activation_l2).
# Prune the lowest-importance 50% per column.
for name, mod in model.named_modules():
    if name in activations:
        W = mod.weight.data.abs()
        a_norm = activations[name].sqrt()
        importance = W * a_norm.unsqueeze(0)
        threshold = importance.quantile(0.5)
        mask = importance >= threshold
        mod.weight.data *= mask.float()
        print(f"{name}: pruned {((~mask).float().mean() * 100):.1f}% of weights")

# Verify the model still produces sensible output (it should, with mild degradation).
ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20)
print(tokenizer.decode(out[0]))
```

You'll see ~50% of MLP weights zeroed; the model still produces output, but quality degrades visibly. Without the brief fine-tune step (omitted here for brevity) the quality loss is larger than what a proper Wanda paper deployment would show. The exercise is to feel the workflow; production usage requires the fine-tune.

## Further reading

- "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning" (Xia et al, 2023).
- "Wanda: A Simple and Effective Pruning Approach for Large Language Models" (Sun et al, 2023).
- "MiniLLM" (Gu et al, 2023) — Module 3 Lesson 9; the on-policy distillation paper.
- "Phi-3 Technical Report" (Microsoft, 2024) — the modern distillation-driven SLM playbook.
- "Compact Language Models via Pruning and Knowledge Distillation" (various authors, 2024) — combining the two.

Next lesson: **KV cache for tiny memory budgets.** We pivot from the static compression story (weights) to the dynamic compression story (KV cache during decode). At long context, the KV cache often dominates memory; managing it well is what makes 32K-context inference possible on consumer devices.
