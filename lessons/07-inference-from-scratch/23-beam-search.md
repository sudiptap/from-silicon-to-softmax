---
title: "Lesson 23 — Beam Search and Why LLMs Largely Abandoned It"
date: "2026-06-04"
module: "inference-from-scratch"
order: 23
tags: ["beam-search", "decoding", "machine-translation", "deprecated"]
author: "Sudipta Pathak"
prerequisites: ["22-sampling-methods"]
---

# Lesson 23 — Beam Search and Why LLMs Largely Abandoned It

## Why this lesson exists

Beam search was the dominant decoding algorithm for neural sequence models from ~2014 to ~2020. It powered Google Translate, summarization systems, and most encoder-decoder applications. Then transformers and LLMs arrived; the field shifted to sampling; beam search faded.

This lesson covers what beam search does, why it dominated for translation, and the specific reasons LLMs walked away from it. The reasons matter: they tell you when beam search is still the right choice (some niches) and why it's the wrong default for chat-style LLMs.

The lesson is reading. The Hands-on implements beam search and compares to sampling.

## The algorithm

Beam search is a *constrained tree search* over the space of possible output sequences. The "beam" is a set of partial sequences (the "beams") that you keep alive at each step.

```
beam_size = 5  # how many partial sequences to track
beams = [(start_token, 0.0)]  # (sequence, cumulative_log_prob)

while not all beams complete:
    candidates = []
    for sequence, score in beams:
        logits = model(sequence)
        log_probs = logits.log_softmax(dim=-1)
        for token_id in range(vocab_size):
            new_seq = sequence + [token_id]
            new_score = score + log_probs[token_id].item()
            candidates.append((new_seq, new_score))
    # Keep the top-beam_size candidates.
    candidates.sort(key=lambda x: -x[1])
    beams = candidates[:beam_size]

best_sequence = beams[0][0]  # highest cumulative log prob
```

The intuition: at each step, expand all beams; for each, consider all vocabulary tokens; pick the top-`beam_size` resulting partial sequences (ranked by cumulative log probability); discard the rest. Continue until all beams hit end-of-sequence.

For `beam_size = 1`, beam search reduces to greedy.

## Why it dominated translation

Two reasons beam search was *the* algorithm for neural translation:

**1. There's a "correct" answer to optimize toward.** Translation has a target — the human reference translation. Beam search finds high-probability sequences; high-probability sequences correlate with the reference. The metric (BLEU score) tracks this directly.

**2. Translation models had short outputs and tightly-peaked distributions.** A typical translation is 20-50 tokens; the model is fairly sure about each token given the source. Greedy decoding misses some opportunities because of local choices; a small beam (k=5) usually catches the small global re-orderings that help. Greedy + a tiny beam was a clear improvement over greedy alone.

In this regime, beam search delivers:
- 1-3 BLEU points over greedy.
- Reproducible outputs (deterministic).
- Modest compute overhead (`beam_size` × greedy cost).

For 2015-era seq2seq translation, beam=4 or beam=5 was standard.

## Why LLMs walked away

The shift came with two changes:

**1. LLMs do open-ended generation, not constrained translation.** A chat response has many valid forms. Beam search finds the highest-probability sequence — but "highest probability" doesn't mean "best." High-probability sequences tend to be generic, repetitive, and "safe" — exactly what users don't want from a chat assistant.

**2. LLM distributions are flatter and longer.** Generating 500 tokens at moderate uncertainty per token means the highest-probability sequence is not particularly stable or interesting. Sampling produces more varied outputs that subjectively read better.

The empirical result: beam search on a chat LLM gives outputs that feel mechanical, repeat phrases, and lack creativity. Sampling — even at low temperature — produces noticeably better results for chat.

The third reason, more practical: **beam search is expensive**. Each beam keeps its own KV cache; for `beam=5`, you need 5× the cache memory. For long outputs, this is a serious cost.

## When beam search is still right

Some specific niches:

**Constrained outputs where you want the highest-probability valid sequence.**
- Math problem solving where there's a single correct answer.
- Structured output (JSON, function calls) where you want the most likely valid completion.

**Short, focused outputs.**
- Translation (still!).
- Summarization where exactness matters.
- Code completion of a single statement.

**Reproducibility-critical settings.**
- Reproducible research outputs.
- Cases where the same input must always produce the same output.

For these, beam=4 or beam=5 is reasonable. For everything else (chat, creative writing, brainstorming, open generation), sampling wins.

## Implementation considerations

A few practical points:

**Length normalization.** Without normalization, beam search prefers shorter sequences (cumulative log probability is more negative for longer sequences). Common fix: divide cumulative log prob by sequence length raised to some power (`α = 0.6` is typical).

```python
score = cumulative_log_prob / (length ** alpha)
```

**Stop tokens.** Beam search needs to know when to stop. Usually: terminate a beam when it emits EOS; collect terminated beams; continue with non-terminated until all done.

**Diverse beam search.** Standard beam search often produces beams that are minor variants of each other. Diverse beam search penalizes similar candidates to produce more variety. Useful when you want N distinct candidates for downstream selection.

## What you should believe after this lesson

Three sentences:

**1. Beam search is a constrained tree search over output sequences** — keep `beam_size` partial sequences alive; at each step, expand and prune to the top-`beam_size`. Reduces to greedy at `beam_size=1`.

**2. LLMs largely abandoned beam search because (a) the highest-probability sequence isn't what users want for open-ended generation,** (b) LLM distributions are flatter and the beam doesn't help much, and (c) beam search costs `beam_size×` the KV cache memory.

**3. Beam search persists in constrained settings**: translation, math with a known correct answer, structured output, reproducibility-critical research. For chat / creative / open generation, sampling with temperature + top-p / min-p is the right choice.

## Hands-on (at home)

Implement beam search and compare to greedy.

```python
# beam_search.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "The future of artificial intelligence is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

def beam_search(model, input_ids, beam_size=4, max_new=20):
    # Track (sequence, cumulative_log_prob).
    beams = [(input_ids, 0.0)]
    for _ in range(max_new):
        candidates = []
        for seq, score in beams:
            with torch.no_grad():
                logits = model(seq).logits[0, -1]
            log_probs = logits.log_softmax(dim=-1)
            # Top beam_size tokens per beam (avoid exploding candidates).
            top_v, top_i = log_probs.topk(beam_size)
            for v, i in zip(top_v, top_i):
                new_seq = torch.cat([seq, i.unsqueeze(0).unsqueeze(0)], dim=1)
                candidates.append((new_seq, score + v.item()))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam_size]
    return beams

def greedy_decode(model, input_ids, max_new=20):
    seq = input_ids
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(seq).logits[0, -1]
        token = logits.argmax()
        seq = torch.cat([seq, token.unsqueeze(0).unsqueeze(0)], dim=1)
    return seq

print("Greedy:")
out = greedy_decode(model, input_ids)
print(" ", tokenizer.decode(out[0]))

print("\nBeam search (beam=4):")
beams = beam_search(model, input_ids, beam_size=4)
for i, (seq, score) in enumerate(beams):
    print(f"  Beam {i} (score {score:.2f}): {tokenizer.decode(seq[0])}")
```

You'll see the beams cluster around similar continuations; the top beam isn't dramatically different from greedy. For creative prompts, beam search outputs feel less interesting than nucleus sampling.

For HuggingFace's built-in beam search, use `model.generate(input_ids, num_beams=4, max_new_tokens=20)`.

## Further reading

- "Sequence to Sequence Learning with Neural Networks" (Sutskever et al, 2014) — beam search applied to early seq2seq.
- "The Curious Case of Neural Text Degeneration" (Holtzman et al, 2019) — empirical comparison of beam search vs sampling; documents beam search's failure modes on open generation.
- "Diverse Beam Search" (Vijayakumar et al, 2018) — for the diverse variant.
- "On Decoding Strategies for Neural Text Generators" (Wiher et al, 2022) — survey.

Next lesson: **Constrained decoding — grammars, JSON schema, regex-guided.** A more recent technique: instead of sampling and hoping, mask invalid tokens at each step so the output is guaranteed to follow a specific format. The infrastructure for structured output in 2026 LLMs.
