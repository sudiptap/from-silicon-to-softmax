---
title: "Lesson 22 — Greedy, Temperature, Top-k, Top-p, Min-p"
date: "2026-06-04"
module: "inference-from-scratch"
order: 22
tags: ["sampling", "temperature", "top-k", "top-p", "nucleus", "min-p"]
author: "Sudipta Pathak"
prerequisites: ["21-streaming-llm-attention-sinks"]
---

# Lesson 22 — Greedy, Temperature, Top-k, Top-p, Min-p

## Why this lesson exists

The transformer's forward pass produces logits — one real number per vocabulary token. These logits define a probability distribution via softmax: `P(token) = softmax(logits)`. But which token to actually emit isn't determined by the distribution alone; you need a *sampling method*.

The choice of sampling method changes everything users see. Greedy decoding produces deterministic, often-repetitive outputs. Temperature scaling tunes randomness. Top-k, top-p, and min-p truncate the distribution differently. Each has a use case; the right combination depends on the application.

This lesson is the practical sampling catalog: what each method does, when to use which, and how they compose.

The lesson is reading. The Hands-on samples from the same distribution with all five methods and shows the qualitative difference.

## The setup

After the final layer, the model produces logits `z ∈ R^V` where `V` is the vocabulary size. The probability of token `i` is:

```
P(i) = softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Sampling means picking a token according to (some version of) this distribution. The choice of "some version" is what differs between sampling methods.

## Greedy

The simplest: pick the most likely token.

```python
token = argmax(logits)
```

Deterministic. Fast. Often the right choice for tasks with a single correct answer (math, code, structured output where you want reproducibility).

Failure modes:
- **Repetition loops.** The model picks a token that increases the probability of the same token being picked next, ad infinitum.
- **Boring outputs.** Without any randomness, the model always picks the safest token; outputs tend toward generic.

For code completion and structured output, greedy is the default. For creative writing, chat, and most general tasks, you want some randomness.

## Temperature

A scalar that reshapes the distribution before sampling.

```python
probs = softmax(logits / T)
token = random_sample(probs)
```

- **T = 1.0**: the model's default distribution. Standard.
- **T < 1.0** (typically 0.7-0.9): sharper distribution. More likely to pick the top tokens. Less random.
- **T > 1.0** (typically 1.0-2.0): flatter distribution. More chance to pick less-likely tokens. More random.
- **T → 0**: approaches greedy.
- **T → ∞**: approaches uniform random over the whole vocabulary.

Temperature alone is rarely the right choice. At T=1.0 the long tail of low-probability tokens still gets some chance; at T=2.0 you sample from extreme tail tokens that produce nonsense. Combined with top-k / top-p truncation, temperature tunes the "how aggressively to pick from the top of the truncated distribution."

## Top-k

Truncate the distribution to the top-k most likely tokens; sample from those.

```python
topk_logits, topk_indices = logits.topk(k)
probs = softmax(topk_logits)
token = topk_indices[random_sample(probs)]
```

- **k = 1**: greedy.
- **k = 50** (common default): the next token is chosen from the 50 most likely.
- **k = vocabulary_size**: no truncation; equivalent to plain sampling.

Top-k is conceptually clean (pick from the top N candidates) but the right k depends on the distribution's shape. A peaked distribution (one very likely token, rest very unlikely) — k=10 already includes a lot of noise. A flat distribution — k=10 might exclude meaningful candidates.

The fix that handles this adaptively is top-p.

## Top-p (nucleus sampling)

Truncate to the smallest set of tokens whose cumulative probability exceeds `p`. Sample from that set.

```python
sorted_logits, sorted_indices = logits.sort(descending=True)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
nucleus_mask = cumulative_probs <= p
# Keep at least one token.
nucleus_mask[..., 0] = True
nucleus_logits = sorted_logits[nucleus_mask]
nucleus_indices = sorted_indices[nucleus_mask]
probs = softmax(nucleus_logits)
token = nucleus_indices[random_sample(probs)]
```

- **p = 0.9** (common default): pick from the smallest set covering 90% of the probability mass.
- **p = 1.0**: no truncation.

Top-p adapts to the distribution: a peaked distribution gets fewer candidates (the top 1-2 tokens cover 90%); a flat distribution gets more.

For most chat / generation tasks, top-p is preferred over top-k. The standard default is `temperature=0.7, top_p=0.9`.

## Min-p

A more recent variant: truncate by *relative probability*. Keep tokens whose probability is at least `min_p × max_probability`.

```python
max_prob = probs.max()
threshold = min_p * max_prob
mask = probs >= threshold
```

- **min_p = 0.1** (a typical setting): tokens at least 10% as likely as the most likely token survive.

Min-p has nice properties:
- **Adapts to peaked vs flat distributions naturally.** A peaked distribution with one dominant token has few survivors; a flat distribution has many.
- **More resistant to noise** at low temperatures than top-p, where one outlier token can bump the cumulative probability past the threshold incorrectly.

Min-p is gaining adoption in 2024-2026. Many local LLM hosting tools default to min-p over top-p now.

## Composing them

Production sampling usually composes several methods:

```python
# Common pipeline:
logits = logits / temperature              # temperature
logits = apply_top_k(logits, k=50)         # top-k truncation
logits = apply_top_p(logits, p=0.9)        # top-p truncation
probs = softmax(logits)
token = random_sample(probs)
```

Or, instead of top-k + top-p:
```python
logits = logits / temperature
probs = softmax(logits)
probs = apply_min_p(probs, min_p=0.1)
token = random_sample(probs)
```

A reasonable starting point for most tasks: `temperature=0.7, min_p=0.1`.

## What each method is good for

A rough guide:

- **Greedy (T=0)**: math, code, structured output where reproducibility matters.
- **Low temperature (T=0.3-0.5)**: factual tasks where you want the most likely answer but some flexibility.
- **Medium temperature + top-p (T=0.7-0.9, p=0.9)**: general chat, summarization, writing.
- **Higher temperature + min-p (T=0.9-1.2, min_p=0.05-0.1)**: creative writing, brainstorming, more diverse generation.
- **Very high temperature (T > 1.5)**: usually a mistake; outputs become incoherent.

The defaults in popular APIs:
- OpenAI: `temperature=1.0, top_p=1.0` (no truncation).
- Anthropic Claude: similar.
- llama.cpp default: `temperature=0.8, top_k=40, top_p=0.95`.

Tune per application.

## Repetition penalties

A common addition: penalize tokens that have recently appeared.

```python
# Multiply the logit of recently-used tokens by some penalty factor < 1.
for token in recent_tokens:
    logits[token] /= repetition_penalty   # e.g., 1.1
```

This combats the repetition loop failure mode of greedy decoding. Many runtimes apply it after temperature and before top-k/top-p.

Common: `repetition_penalty = 1.1` or `1.2`. Too high and the model avoids saying the same word twice, which is unnatural.

## What you should believe after this lesson

Three sentences:

**1. The forward pass produces logits; sampling turns logits into tokens.** Greedy is deterministic. Temperature scales the sharpness. Top-k, top-p, and min-p truncate the distribution before sampling, each in slightly different ways.

**2. The standard production recipe is "temperature + top-p" or "temperature + min-p."** Defaults around `temperature=0.7, top_p=0.9` or `min_p=0.1` work for most chat applications.

**3. Different applications want different sampling**: greedy / very low temperature for math and code; medium temperature with top-p for chat; higher temperature for creative writing. Repetition penalty helps with the loop failure mode common in low-temperature settings.

## Hands-on (at home)

Sample from the same distribution with all five methods.

```python
# sampling_demo.py
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# Fake distribution: peaked plus long tail.
vocab_size = 100
logits = torch.zeros(vocab_size)
logits[0] = 5.0    # very probable
logits[1] = 3.0    # likely
logits[2:10] = 1.0  # plausible
logits[10:50] = 0.0  # random
logits[50:] = -1.0   # unlikely

def sample(logits, n_samples=10000, temperature=1.0, top_k=None, top_p=None, min_p=None):
    logits = logits / temperature
    if top_k is not None:
        v, idx = logits.topk(top_k)
        new_logits = torch.full_like(logits, float('-inf'))
        new_logits[idx] = v
        logits = new_logits
    if top_p is not None:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        keep = cum <= top_p
        keep[0] = True
        new_logits = torch.full_like(logits, float('-inf'))
        new_logits[sorted_indices[keep]] = sorted_logits[keep]
        logits = new_logits
    probs = logits.softmax(dim=-1)
    if min_p is not None:
        max_prob = probs.max()
        keep = probs >= min_p * max_prob
        probs = probs * keep
        probs = probs / probs.sum()
    return torch.multinomial(probs.unsqueeze(0).expand(n_samples, -1), 1).squeeze()

# Sample 10K times with each method; count the picks.
methods = {
    "greedy (T=0 ≈ T=0.01)": lambda: sample(logits, temperature=0.01),
    "temp=1.0":                lambda: sample(logits, temperature=1.0),
    "temp=0.7":                lambda: sample(logits, temperature=0.7),
    "top-k=10":                lambda: sample(logits, top_k=10),
    "top-p=0.9":               lambda: sample(logits, top_p=0.9),
    "min-p=0.1":               lambda: sample(logits, min_p=0.1),
}
for name, fn in methods.items():
    samples = fn()
    counts = torch.bincount(samples, minlength=vocab_size)
    top5 = counts.topk(5)
    print(f"{name:25s}: top 5 token IDs = {top5.indices.tolist()}, counts = {top5.values.tolist()}")
```

Output shows how each method shapes the distribution. Greedy picks token 0 every time. Temperature 1.0 spreads across many. Top-k=10 is restricted to the top 10. Top-p and min-p adapt to the distribution shape.

For real-world testing: hook into your favorite local LLM runtime, run the same prompt with each sampling method, observe the qualitative output difference.

## Further reading

- "Hierarchical Neural Story Generation" (Fan et al, 2018) — introduced top-k sampling.
- "The Curious Case of Neural Text Degeneration" (Holtzman et al, 2019) — introduced nucleus (top-p) sampling.
- Various LLM hosting tool docs (llama.cpp, vLLM, oobabooga) — for the sampling parameter conventions.
- "Min-p Sampling" — community-driven; multiple blog posts and Reddit threads document the discovery.

Next lesson: **Beam search and why LLMs largely abandoned it.** Beam search dominated neural translation in the 2017-2020 era; it's largely absent from modern LLM serving. We trace why — and the small remaining contexts where beam search is still the right choice.
