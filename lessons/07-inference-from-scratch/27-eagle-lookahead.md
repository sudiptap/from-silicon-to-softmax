---
title: "Lesson 27 — EAGLE & Lookahead Decoding"
date: "2026-06-04"
module: "inference-from-scratch"
order: 27
tags: ["eagle", "lookahead-decoding", "self-speculation", "jacobi", "n-gram"]
author: "Sudipta Pathak"
prerequisites: ["26-medusa"]
---

# Lesson 27 — EAGLE & Lookahead Decoding

## Why this lesson exists

Medusa (Lesson 26) showed self-speculation works without a separate draft model. Two important refinements followed: **EAGLE** (Li et al, 2024) replaces Medusa's independent heads with a small autoregressive sub-network for higher acceptance; **lookahead decoding** (Fu et al, 2024) sidesteps the need for *any* additional parameters or training by using the model's own forward-pass predictions to bootstrap n-gram speculation.

Both are practical contributions to the speculative-decoding landscape; both ship in production runtimes. This lesson covers what each does and when each is the right choice.

The lesson is reading. The Hands-on demonstrates lookahead-style n-gram bootstrapping.

## EAGLE — autoregressive self-speculation

Medusa's independent heads have a limitation: head 2 (predicting token at position +2) doesn't know what head 1's prediction (position +1) will be. Each head guesses based on the base hidden state alone.

EAGLE adds a small autoregressive component: a tiny transformer (1-2 layers) that:
1. Takes the base model's last-token hidden state as input.
2. Takes the previously-speculated tokens as additional input.
3. Produces the next speculated token.

So position-2's prediction conditions on position-1's prediction. Higher acceptance rate because each token's prediction is coherent with the prior.

The EAGLE network is small (~10-50M parameters) and runs quickly. Training: distill from the target model's predictions on a corpus.

EAGLE-2 (2024 update) adds dynamic tree exploration: instead of a fixed-shape candidate tree, EAGLE-2 grows the tree adaptively based on confidence at each branch.

Throughput: EAGLE achieves 3-4× speedup on Llama-class models; EAGLE-2 a bit more. Competitive with draft-based speculative decoding while keeping single-model deployment.

## Lookahead decoding

Lookahead decoding (Fu et al, 2024) is the most clever of the self-speculation variants because it requires *zero additional parameters or training*.

The mechanism: at each step, instead of generating one token via the standard forward pass, generate a *small parallel set* of tokens using the model's own forward pass.

The trick: Jacobi iteration. Set up `k` candidate positions; initialize them with placeholder tokens; run the forward pass; update each position based on the model's prediction; repeat until convergence (or a fixed number of iterations).

Pseudo-code:

```python
def lookahead_step(model, prompt, k=4, n_iter=4):
    # Initialize k candidate tokens (could be EOS, could be a guess).
    candidates = torch.full((k,), eos_token_id)
    for iteration in range(n_iter):
        # Run the model on (prompt + candidates).
        logits = model(torch.cat([prompt, candidates]))
        # Update each candidate position based on the new prediction.
        for i in range(k):
            candidates[i] = logits[len(prompt) + i].argmax()
    return candidates
```

After a few iterations, the candidate positions stabilize at the tokens the model would have generated autoregressively. The fixed point of the Jacobi iteration matches autoregressive generation.

The win: each iteration is one forward pass; instead of `k` sequential decode steps, you do `n_iter` parallel-position forward passes. If `n_iter < k`, you've reduced total forward passes.

Combined with the n-gram pool (looking up "what 4-grams has this model emitted recently?" and using them as candidate initializations), lookahead decoding achieves 1.5-2× speedup without any model changes.

## What lookahead decoding doesn't need

Notable absences from lookahead decoding:
- **No additional model.** The same target model does everything.
- **No additional training.** No fine-tuning required.
- **No additional parameters.** The model artifact is unchanged.
- **No additional infrastructure** beyond a small lookahead branch in the inference loop.

This is the lightest-weight speculative decoding variant. Speedups are smaller (1.5-2× vs 3-4× for EAGLE), but the engineering cost is near zero.

## When to use which

A rough decision tree:

- **Maximum throughput, willing to train**: EAGLE or EAGLE-2.
- **Maximum throughput, have a small model in the family**: draft-based speculative decoding.
- **Single-model deployment without training**: lookahead decoding.
- **Single-model deployment, willing to train**: Medusa or EAGLE.
- **Lightweight gain on existing deployment**: lookahead decoding or n-gram speculation (Module 6 Lesson 9).

For most production deployments in 2026:
- If you're building a new system with a model family that has small variants: draft-based speculative.
- If you're augmenting an existing deployment without changes: lookahead.
- If you need the best of both: EAGLE-2.

## What you should believe after this lesson

Three sentences:

**1. EAGLE adds a small autoregressive sub-network** (1-2 layers, ~10-50M params) that produces speculative tokens conditioning on previous speculated tokens. Higher acceptance than Medusa's independent heads; 3-4× throughput improvement.

**2. Lookahead decoding requires zero additional parameters or training** — it uses Jacobi iteration over `k` parallel candidate positions, plus an n-gram pool of recently-emitted token sequences. Achieves 1.5-2× speedup with near-zero engineering cost.

**3. The right choice depends on engineering budget**: lookahead for zero-touch deployment, draft-based for maximum throughput with existing small models, EAGLE-2 for the best single-model option after training investment.

## Hands-on (at home)

Demonstrate a simplified lookahead step.

```python
# lookahead_demo.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16).eval()

prompt = "The capital of France is "
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids

def standard_decode(model, prompt_ids, n_new):
    seq = prompt_ids
    for _ in range(n_new):
        with torch.no_grad():
            logits = model(seq).logits[0, -1]
        seq = torch.cat([seq, logits.argmax().view(1, 1)], dim=1)
    return seq

def lookahead_decode_step(model, prompt_ids, k=4, n_iter=3):
    """Generate k tokens via Jacobi iteration. Returns the k tokens."""
    # Initialize k candidate positions with the prompt's last token (a guess).
    init_token = prompt_ids[0, -1].item()
    candidates = torch.full((k,), init_token, dtype=torch.long)
    seq = torch.cat([prompt_ids, candidates.unsqueeze(0)], dim=1)
    
    for it in range(n_iter):
        with torch.no_grad():
            logits = model(seq).logits[0]
        # The logit at position i predicts the token at position i+1.
        # The k candidate positions are at indices [len(prompt_ids), len(prompt_ids)+1, ..., len(prompt_ids)+k-1].
        # Their predictions come from logits at indices [len(prompt_ids)-1, len(prompt_ids), ..., len(prompt_ids)+k-2].
        new_candidates = logits[len(prompt_ids[0])-1: len(prompt_ids[0])-1+k].argmax(dim=-1)
        if torch.equal(new_candidates, candidates):
            print(f"  Converged at iteration {it}")
            break
        candidates = new_candidates
        seq = torch.cat([prompt_ids, candidates.unsqueeze(0)], dim=1)
    return candidates

# Standard.
std = standard_decode(model, prompt_ids, n_new=4)
print(f"Standard: {tokenizer.decode(std[0])}")

# Lookahead step (gives 4 tokens in ~3 forward passes instead of 4).
la = lookahead_decode_step(model, prompt_ids, k=4, n_iter=3)
seq_la = torch.cat([prompt_ids, la.unsqueeze(0)], dim=1)
print(f"Lookahead: {tokenizer.decode(seq_la[0])}")
```

This is a simplified demonstration; production lookahead decoding has the n-gram pool, longer windows, and integration with the regular decode loop.

## Further reading

- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al, 2024).
- "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (Li et al, 2024).
- "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (Fu et al, 2024).
- "Cascade Speculative Drafting for Even Faster LLM Inference" (Chen et al, 2024) — hierarchical drafting.

Next lesson: **Multi-Token Prediction (MTP).** DeepSeek-V3 introduced training-time MTP where the model is explicitly trained to predict multiple future tokens. This isn't quite the same as inference-time speculation but enables related tricks. We close Part 4 with MTP and how it composes with everything else.
