---
title: "Lesson 25 — Speculative Decoding: Draft + Verify"
date: "2026-06-04"
module: "inference-from-scratch"
order: 25
tags: ["speculative-decoding", "draft-model", "verification", "rejection-sampling"]
author: "Sudipta Pathak"
prerequisites: ["24-constrained-decoding"]
---

# Lesson 25 — Speculative Decoding: Draft + Verify

## Why this lesson exists

Module 6 Lesson 9 covered speculative decoding from the on-device deployment perspective: when it pays off, when it doesn't, the throughput math. This lesson goes one level deeper: the *algorithm itself*, the rejection sampling that makes it provably exact, the implementation details, and how the verification step works in practice.

The thing that makes speculative decoding particularly elegant: it produces *exactly* the same output distribution as the main model alone — never approximate. The draft model proposes; the main model accepts or rejects each draft token; the accepted prefix is appended; the rejected token is replaced by a sample from a *corrected* distribution. The output is provably identical to autoregressive sampling from the main model.

This lesson is the derivation.

The lesson is reading. The Hands-on implements speculative decoding from scratch.

## The algorithm

Setup:
- Target model `M_target` (large, slow).
- Draft model `M_draft` (small, fast).
- Both produce probability distributions over the same vocabulary.

For each speculative step:

1. **Draft**: starting from current sequence `x`, draft model generates `k` tokens `t_1, t_2, ..., t_k`. Record draft probabilities `q_i(t_i | x, t_<i)` for each.

2. **Verify**: pass the *entire* sequence `x + t_1 + ... + t_k` through `M_target` in one forward pass. This produces `k+1` logits: one after each prefix length 0 through k. Convert to probabilities `p_i(• | x, t_<i)`.

3. **Accept or reject** each draft token `t_i` (in order):
   - Compute acceptance probability `α_i = min(1, p_i(t_i) / q_i(t_i))`.
   - Sample `u_i ~ Uniform(0, 1)`. If `u_i ≤ α_i`, accept `t_i`. Otherwise, reject and stop.
   
4. **If a token is rejected at position `i`**: replace `t_i` with a sample from the *adjusted* distribution `p_i(•) - q_i(•)` clipped to non-negative and renormalized. Append this corrected token; stop.

5. **If all `k` drafts are accepted**: also sample one more token from the target's distribution at position `k+1` (using the logits from the verify pass).

Net result: 1 to `k+1` tokens added per main-model pass.

## Why this is exact

The key mathematical property: the procedure samples from the target distribution `p`, even though most of the work was done with the draft `q`.

The proof sketch: rejection sampling. The acceptance probability `α = min(1, p/q)` is chosen so the accepted samples have density proportional to `p`. The corrected distribution `(p - q)` for the rejected case fills in the missing probability mass.

Formal proof in the original speculative decoding paper (Leviathan, Kalman, Matias 2022). The crucial point is the output distribution is exactly `p`, never some approximation of it. Speculative decoding is loss-free.

## The acceptance rate

Acceptance probability for token `t`: `α = min(1, p(t)/q(t))`.

Key cases:
- `q(t) = p(t)`: `α = 1`. Always accept.
- `q(t) > p(t)`: `α = p(t)/q(t) < 1`. Probabilistically accept.
- `q(t) < p(t)`: `α = 1`. Always accept (the draft is being conservative; the target also wants this token).

The empirical acceptance rate depends on how similar the draft and target distributions are. Typical numbers:
- Draft model is a smaller version of target (Llama 3.2 1B drafting for Llama 3.1 8B): 70-85% acceptance.
- Draft model is unrelated: 30-50% acceptance.
- Draft model is a distilled version of target: 80-90%.

For acceptance rate `r`, the expected number of tokens per main pass is approximately `r × k + 1` for small `k` (because if you accept all `k`, you also get a free target sample).

## The expected speedup

If the main model takes `T_main` per pass and the draft takes `T_draft` per token, then `k` draft tokens cost `k × T_draft` plus one `T_main` for verification.

Without speculation: 1 token per `T_main`. Throughput: `1/T_main` tok/s.

With speculation: `(r × k + 1)` tokens per `(k × T_draft + T_main)`. Throughput: `(r × k + 1) / (k × T_draft + T_main)`.

Speedup: `(r × k + 1) × T_main / (k × T_draft + T_main)`.

If `T_draft << T_main` (small draft), the cost is mostly `T_main`; you get roughly `r × k + 1` tokens per main pass. For `r = 0.8`, `k = 4`: 4.2 tokens per main pass. ~4× speedup.

If `T_draft` is a non-trivial fraction of `T_main` (small main model relative to draft), the math gets worse. This is why speculative decoding shines for *large* main models.

## Choosing the draft model

Three approaches:

**1. Smaller variant of the same family.** Llama 3.2 1B as draft for Llama 3.1 8B target. Same training data, same tokenizer, same general behavior. Acceptance rate high (70-85%). Simple to deploy (just load both models).

**2. Trained-as-draft.** A model specifically trained to mimic the target. Higher acceptance rate (80-90%) at the cost of training effort.

**3. Self-speculation** (Medusa / EAGLE, Lessons 26-27). The target model with additional heads to predict multiple tokens. No separate draft model.

**4. N-gram speculation** (Lesson 9 in Module 6). A lookup table of common token sequences. Near-zero overhead; lower acceptance rate (typically 30-50%).

The choice depends on what you have available and the workload. For general chat, smaller variant + acceptance ~75% is typical.

## The KV cache wrinkle

Speculative decoding involves running the target model on `x + k draft tokens`. The KV cache must include both the original `x` (already cached) and the draft tokens (just added).

If a draft token gets rejected, its K and V have been written to the cache but shouldn't be reused for future tokens. The runtime must *roll back* the cache to before the rejected token.

For paged attention (Lesson 18), this is cheap — just decrement the position counter and free any blocks that contain rejected tokens. For contiguous caches, you reset the "current length" pointer.

This rollback is part of why speculative decoding is more involved to implement than naive decode.

## Implementation in production runtimes

Production support:
- **vLLM**: speculative decoding with draft models (`--num-speculative-tokens 4 --speculative-model llama-3.2-1b`).
- **TensorRT-LLM**: similar.
- **SGLang**: built-in.
- **llama.cpp**: `--draft-model` and `--draft N` options.

The acceptance rate varies by model pair and prompt; runtimes log it so you can tune.

## What you should believe after this lesson

Three sentences:

**1. Speculative decoding samples from the *target* distribution exactly** — never approximate. The draft proposes; the target accepts via rejection sampling (`α = min(1, p/q)`); rejected tokens are replaced by samples from a corrected distribution. Output distribution is mathematically identical to autoregressive sampling from the target.

**2. The throughput speedup is roughly `r × k + 1` per main model pass** where `r` is acceptance rate and `k` is drafted tokens. For `r = 0.8, k = 4`: ~4 tokens per pass, ~3-4× speedup. The speedup is real only when the draft model is much faster than the target (typically a 1:10 size ratio).

**3. Implementation requires KV cache rollback** for rejected tokens. PagedAttention makes this cheap; contiguous caches need careful pointer management.

## Hands-on (at home)

Implement speculative decoding from scratch (small models for tractability).

```python
# speculative_decoding.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use a small target and an even smaller draft for demonstration.
target_id = "Qwen/Qwen2.5-1.5B-Instruct"
draft_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(target_id)
target = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=torch.float16).eval()
draft = AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=torch.float16).eval()

def speculative_decode(target, draft, prompt_ids, k=4, max_new=100, temp=0.7):
    seq = prompt_ids.clone()
    n_generated = 0
    while n_generated < max_new:
        # 1. Draft k tokens.
        draft_tokens = []
        draft_probs = []
        for _ in range(k):
            with torch.no_grad():
                logits = draft(seq).logits[0, -1] / temp
            p = F.softmax(logits, dim=-1)
            t = torch.multinomial(p, 1)
            draft_tokens.append(t.item())
            draft_probs.append(p[t.item()].item())
            seq = torch.cat([seq, t.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # 2. Verify with target: get its distributions at each draft position.
        with torch.no_grad():
            target_logits = target(seq[:, :-1+0]).logits[0, -k-1:-1] / temp  # last k+1 positions
        target_probs = F.softmax(target_logits, dim=-1)
        
        # 3. Accept / reject.
        accepted = 0
        for i in range(k):
            p_target = target_probs[i, draft_tokens[i]].item()
            p_draft = draft_probs[i]
            alpha = min(1.0, p_target / max(p_draft, 1e-10))
            if torch.rand(1).item() < alpha:
                accepted += 1
            else:
                # Reject: replace with sample from (p_target - p_draft) clipped.
                seq = seq[:, :-(k-i)]  # roll back
                adjusted = (target_probs[i] - draft_probs[i]).clamp(min=0)
                adjusted = adjusted / adjusted.sum()
                new_t = torch.multinomial(adjusted, 1).item()
                seq = torch.cat([seq, torch.tensor([[new_t]])], dim=1)
                accepted_total = i + 1  # i draft + 1 replacement
                n_generated += accepted_total
                break
        else:
            # All k accepted; also sample one more from target's position k+1.
            with torch.no_grad():
                bonus_logits = target(seq).logits[0, -1] / temp
            bonus = torch.multinomial(F.softmax(bonus_logits, dim=-1), 1).item()
            seq = torch.cat([seq, torch.tensor([[bonus]])], dim=1)
            n_generated += k + 1
        print(f"  speculative step: accepted {accepted}/{k}")
    return seq

prompt = tokenizer("Explain the mechanism of action of caffeine in two sentences:", return_tensors="pt").input_ids
out = speculative_decode(target, draft, prompt, k=4, max_new=80)
print("\nGenerated:", tokenizer.decode(out[0]))
```

This is a simplified single-batch CPU implementation; production speculative decoding has KV cache reuse for both models, GPU batching, and many more optimizations. The point is to see the accept/reject mechanism in action.

For real measurements, use vLLM or llama.cpp's speculative-decoding mode and watch the reported acceptance rate.

## Further reading

- "Fast Inference from Transformers via Speculative Decoding" (Leviathan, Kalman, Matias, 2022) — the foundational paper with the rejection-sampling proof.
- "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al, 2023) — independent DeepMind paper on the same idea.
- vLLM speculative decoding docs.
- Module 6 Lesson 9 — the on-device deployment view.

Next lesson: **Medusa — multi-head speculative.** A self-speculation variant: add extra heads to the main model that predict the second, third, ... future tokens directly. No separate draft model required.
