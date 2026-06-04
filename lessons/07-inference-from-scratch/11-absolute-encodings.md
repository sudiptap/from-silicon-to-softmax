---
title: "Lesson 11 — Absolute Encodings: Sinusoidal, Learned, Integer/Binary"
date: "2026-06-04"
module: "inference-from-scratch"
order: 11
tags: ["positional-encoding", "sinusoidal", "learned", "absolute", "extrapolation"]
author: "Sudipta Pathak"
prerequisites: ["10-why-positions-matter"]
---

# Lesson 11 — Absolute Encodings: Sinusoidal, Learned, Integer/Binary

## Why this lesson exists

The first generation of positional encodings is the *absolute* family: each position gets a unique vector that's added to the token embedding before the attention layers. The model learns to interpret these position vectors alongside the content embeddings.

Three variants dominated 2017-2020: **sinusoidal** (the original transformer; deterministic), **learned** (treated as additional learnable parameters; what BERT used), and **integer/binary** (more recent, niche). They have similar performance characteristics with different practical properties — most importantly around extrapolation to lengths longer than training.

This lesson is the catalog of absolute encodings: what each does, their differences, and why the field eventually moved past them in favor of relative encodings (Lesson 12) and RoPE (Lesson 14).

The lesson is reading. The Hands-on visualizes sinusoidal vs learned encodings and tests extrapolation.

## Sinusoidal encodings (the original)

The 2017 transformer's positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

where `pos` is the token's position (0, 1, 2, ...) and `i` ranges over the dimension. The encoding for each position is a `d`-dimensional vector of sines and cosines at exponentially-spaced frequencies.

The rationale (per the paper): the function has the property that `PE(pos + k)` is a linear function of `PE(pos)` — making it easy in principle for the model to learn to attend by relative position. The exponential frequency spacing covers a range from "captures local position" (high-frequency sinusoids) to "captures global position" (low-frequency sinusoids).

Properties:
- **Deterministic**: no learned parameters. The encoding is the same for every model trained on the same dimension.
- **Theoretically supports arbitrary length**: the function is well-defined for any `pos`.
- **Empirically poor extrapolation**: in practice, models trained on context length `N` don't generalize cleanly to lengths much larger than `N`. The sinusoidal pattern is well-defined at longer positions but the model never saw those exact patterns during training.

The encoding is added (element-wise) to the input token embeddings:

```
x_with_pos = x_embedding + PE
```

Then `x_with_pos` is what enters the first transformer layer.

## Learned encodings

The learned variant: instead of computing PE via a fixed formula, allocate `MAX_LEN × d` learnable parameters and let gradient descent pick them.

```
PE_table = nn.Embedding(MAX_LEN, d)  # learnable
PE_for_seq = PE_table(positions)     # [N, d]
x_with_pos = x_embedding + PE_for_seq
```

BERT used this. GPT-2 used this. The chief practical difference from sinusoidal: the values are learned during pretraining.

Properties:
- **Slightly more parameters** (`MAX_LEN × d` extra).
- **Better in-distribution performance** for sequences within `MAX_LEN`.
- **Catastrophic at sequences longer than `MAX_LEN`**: the PE table doesn't have entries for positions beyond `MAX_LEN`. The model literally cannot represent positions it never saw.

This is the fundamental problem with absolute positional encodings: the model can't handle positions it wasn't trained on. Sinusoidal has the values but the model never learned to use them at those positions; learned doesn't have the values at all.

Empirically, both perform comparably to within-context-length. Both fail at out-of-context-length.

## Integer / binary encodings

A more recent variant (used in some smaller research models): represent the position as integers or binary patterns directly in the encoding.

```
PE(pos, i) = i-th bit of binary(pos)   # binary encoding
```

This guarantees unique representations for each position and works at arbitrary length (the bit encoding is well-defined for any integer).

In practice, this hasn't seen wide adoption. The binary representations are too "sharp" for the soft attention mechanism to use efficiently; gradient descent struggles to learn meaningful patterns from them.

## The extrapolation problem

The big practical issue with absolute encodings: they don't extrapolate.

For a model trained on contexts up to 2048 tokens, both sinusoidal and learned encodings cause severe degradation when fed sequences of 4096, 8192, etc. The reason is different for the two:
- **Sinusoidal**: the encoding values for positions > 2048 are well-defined but the model's other parameters weren't trained alongside them. Attention scores at long positions look like noise; outputs are degraded.
- **Learned**: the encoding values for positions > MAX_LEN literally don't exist. Either you allocate MAX_LEN very large (wastes parameters; most aren't trained well) or you can't run at long context at all.

Workarounds at the time:
- Train with the longest context you anticipate. Doesn't help if you want to use the model at longer.
- Truncate inputs to the trained context length. Cap on capability.
- Interpolate or extend the PE function. Hacky; quality cost.

The frustration with these workarounds is what motivated the relative-encoding shift (Lesson 12) and eventually RoPE (Lesson 14), which extrapolate much better.

## Why absolute encodings persisted as long as they did

Sinusoidal and learned encodings were the standard from 2017 through ~2020. Why not move past them sooner?

Several reasons:
1. **They worked well in-distribution.** At the contexts most models were trained for (512-2048 tokens), the limitations didn't bite hard.
2. **Simplicity.** Add a vector at the embedding layer; no architectural surgery to the attention mechanism.
3. **Drop-in compatibility.** Models could be swapped between sinusoidal and learned without changing other components.

The shift began when models started training for longer contexts (T5, GPT-3) and the extrapolation cliff became impossible to ignore.

## Are absolute encodings dead?

Not quite. Some scenarios where absolute encodings still make sense:

- **BERT-style classification** with fixed-size inputs (e.g., 512 tokens): the extrapolation issue doesn't matter because you never see longer inputs.
- **Domain-specific small models** trained on a known fixed-length distribution.
- **Research / educational contexts** where you want to start with the simplest implementation.

For general-purpose LLMs in 2026, RoPE is the default; absolute encodings are legacy.

## What you should believe after this lesson

Three sentences:

**1. Absolute positional encodings (sinusoidal, learned, integer/binary)** add a position-specific vector to each token's embedding before attention layers. They work well within the trained context length and fail catastrophically beyond it.

**2. Sinusoidal is the original (2017)** — deterministic, no parameters, fixed frequencies. **Learned** allocates `MAX_LEN × d` trainable parameters. They perform comparably in-distribution; both fail at extrapolation.

**3. The extrapolation problem is what motivated the move to relative encodings and RoPE.** Absolute encodings persist in BERT-style fixed-length contexts but are legacy for modern LLMs.

## Hands-on (at home)

Visualize sinusoidal and learned encodings; test extrapolation behavior.

```python
# absolute_encodings.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

def sinusoidal_pe(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

# Visualize the sinusoidal encoding.
pe = sinusoidal_pe(64, 16)
plt.figure(figsize=(8, 4))
plt.imshow(pe.T, aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Sinusoidal positional encoding (64 positions × 16 dims)')
plt.tight_layout()
plt.savefig('sinusoidal_pe.png')
print("Saved sinusoidal_pe.png")

# Demonstrate that the sinusoidal encoding extends "for free" to longer lengths
# (it's a well-defined function), even though the model wouldn't have been
# trained on it.
pe_long = sinusoidal_pe(256, 16)
print(f"Sinusoidal PE shape at length 256: {pe_long.shape} — well-defined")

# Learned encoding: needs an explicit max_len.
learned_pe = nn.Embedding(64, 16)
print(f"Learned PE table shape: {learned_pe.weight.shape}")
print("Querying position 200 would crash:")
try:
    learned_pe(torch.tensor([200]))
except Exception as e:
    print(f"  Error: {e}")
```

The sinusoidal encoding "extends" naturally to any length (the function is defined for any integer); the learned encoding hard-fails beyond `MAX_LEN`. But sinusoidal's quality also degrades because the rest of the model wasn't trained for the longer positions.

For the extrapolation experiment: take a small GPT-2 model (trained at 1024 context with learned encoding), feed it 2048-token input, observe the output garbage. Then redo with a RoPE-based model (e.g., Llama 3.2) and see it handle the longer input cleanly.

## Further reading

- "Attention Is All You Need" Section 3.5 — sinusoidal encoding formula.
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al, 2018) — learned positional encodings in production.
- "Position Information in Transformers" (Dufter et al, 2022) — survey of positional encoding approaches.
- "On the Position Embeddings in BERT" (Wang & Chen, 2020) — analysis comparing learned to sinusoidal.

Next lesson: **Relative position encodings (Shaw et al, T5 bias).** The first generation of variants that addresses extrapolation: instead of encoding absolute position, encode the *distance* between two tokens. This shifts the problem from "what position is this token at" to "how far apart are these two tokens" — which generalizes better.
