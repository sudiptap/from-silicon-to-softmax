---
title: "Lesson 49 — Test-Time Compute Scaling"
date: "2026-06-04"
module: "inference-from-scratch"
order: 49
tags: ["test-time-compute", "best-of-n", "majority-voting", "reasoning", "scaling-laws"]
author: "Sudipta Pathak"
prerequisites: ["48-tree-attention"]
---

# Lesson 49 — Test-Time Compute Scaling

## Why this lesson exists

Until 2024, the dominant scaling law was: bigger model + more training data = better quality. Inference cost was fixed (one forward pass).

OpenAI's o1 (September 2024) and DeepSeek's R1 (January 2025) introduced a new axis: spend more compute *at inference time* — either by generating many candidate answers and selecting/aggregating, or by generating long chains of thought before the final answer. The result: dramatically improved performance on reasoning-heavy benchmarks at the cost of more tokens per question.

This new scaling axis — *test-time compute* — has reshaped inference systems. Suddenly inference can use 10× or 100× more compute per question; the model architecture and serving stack need to support it.

This lesson covers the canonical test-time compute strategies (best-of-N, majority voting, reasoning chains) and the cost-vs-quality math.

The lesson is reading. The Hands-on implements majority-voting on a math benchmark.

## The basic strategies

Three main strategies, in increasing sophistication:

**1. Best-of-N (BoN)**. Generate N candidate answers; pick the best by some criterion (a reward model, or the most consistent, or the highest-probability).

**2. Majority voting / self-consistency**. Generate N answers; pick the one that appears most often (for tasks with a discrete answer like math problems).

**3. Reasoning chains**. Generate a long chain-of-thought before the final answer. The reasoning helps the model decompose the problem; the final answer is more reliable.

Combinations:
- BoN + reasoning: generate N reasoning chains; pick the best one's answer (or majority vote on the answers).
- BoN + reward model: a separate model scores candidates; pick the highest-scoring.
- Reflection / iterative refinement: the model reviews its own answer and refines.

## The compute-quality frontier

The 2024-2025 finding: across many models and tasks, performance scales roughly *log-linearly* with test-time compute. Doubling compute (more samples, longer reasoning) gives a fixed quality improvement.

Specifically:
- BoN with N=16 vs N=1 on math benchmarks: ~10-15% absolute accuracy improvement.
- Reasoning chains 10× longer: similar improvement.
- The improvements compound: BoN + reasoning is roughly additive on the log scale.

This is the "test-time compute scaling law": quality is bounded by `q = q_0 + c × log(test_time_compute)`. The constant `c` depends on the task; harder tasks have larger `c`.

## What this changes for inference

The serving implications:

**1. Variable per-question cost.** A simple question might need 1 forward pass; a reasoning question might need 100. The serving stack needs to handle this variance.

**2. KV cache for reasoning chains.** A 10K-token reasoning chain has 10K-token KV cache. Long-context support becomes essential.

**3. Branching and verification.** BoN requires generating N candidates; verification needs to compare them. Tree attention (Lesson 48) becomes the right substrate.

**4. Throughput optimization at high compute per question.** Each question takes more compute; throughput optimizations (continuous batching, KV sharing across branches) matter more.

## The cost trade

A naive calculation:
- Standard inference: 1 forward pass per question. Cost: X.
- BoN with N=16: 16 forward passes per question. Cost: 16X. Quality improvement: ~10%.
- Reasoning chains 10× longer: 10X cost. Quality improvement: ~10%.

For a domain where the 10% quality improvement is worth 10-16× the cost (e.g., math research, code generation, hard reasoning), this is a clear win.

For domains where a 90% answer is fine (casual chat, simple Q&A), the extra compute is wasted.

## What o1 and R1 do differently

OpenAI's o1:
- Generates long reasoning chains (often 10K+ tokens) before the final answer.
- Trained via RLHF where the reward is correctness of the final answer; the model learns to reason through problems.
- Inference cost: many times higher than standard chat models.
- Quality on hard benchmarks (math, science): dramatically higher.

DeepSeek's R1 (open-weights):
- Similar approach with explicit reasoning chains.
- Released the model with open weights; the community can study and replicate.
- Trained via RL on verifiable problems (math, code) with correctness rewards.
- Quality comparable to o1 on many tasks.

The 2025-2026 picture: reasoning models are a separate model class. Standard chat models for fast cheap responses; reasoning models for hard problems with budget for the extra tokens.

## Inference system implications

Reasoning models stress the inference stack:
- **KV cache memory dominates** because the reasoning chains are long.
- **Decode throughput matters more** than for chat (more tokens per question).
- **Streaming UX changes** — users see "thinking..." for many seconds before the answer; the UI needs to express this.
- **Cost per query is 10-50× a chat query** — API pricing reflects this.

Some serving stacks have specialized support for reasoning models:
- TensorRT-LLM has long-context modes specifically for reasoning.
- vLLM has reasoning-model-friendly KV cache management.

## What you should believe after this lesson

Three sentences:

**1. Test-time compute scaling is a new axis** — performance scales log-linearly with compute spent at inference (best-of-N, longer reasoning chains). The scaling has been demonstrated on math, code, and reasoning benchmarks.

**2. The 2024-2025 reasoning-model wave (o1, R1)** trained models to generate long reasoning chains before answers; performance on hard benchmarks improves dramatically at 10-50× the per-query cost of standard chat.

**3. The serving implications include longer KV caches, more focus on decode throughput, and UI changes** for the "thinking..." phase. Reasoning models stress the inference stack in ways standard chat models don't.

## Hands-on (at home)

Implement majority voting for a math task.

```python
# majority_voting.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter
import re

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).eval()

questions = [
    {"q": "What is 23 × 47?", "correct": "1081"},
    {"q": "What is 14 + 27?", "correct": "41"},
]

def generate_answer(question, N=8, temperature=0.7):
    prompt = (
        f"<|im_start|>system\nYou are a helpful math assistant. End your answer with the number only.\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    answers = []
    for _ in range(N):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=100, do_sample=True, temperature=temperature)
        decoded = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # Extract the last number.
        nums = re.findall(r'\d+', decoded)
        if nums:
            answers.append(nums[-1])
    # Majority vote.
    counter = Counter(answers)
    return counter.most_common(1)[0][0] if counter else None, counter

for q in questions:
    answer, distribution = generate_answer(q["q"], N=8)
    correct = answer == q["correct"]
    print(f"Q: {q['q']}")
    print(f"  Majority answer: {answer} (correct: {correct})")
    print(f"  Distribution: {distribution}")
```

You'll see: for easy questions, all samples agree; for hard ones, the distribution is spread and majority voting picks the most common (often correct) answer.

For more advanced test-time compute (reasoning chains), prompt the model to produce its reasoning before the answer; the longer reasoning improves quality at the cost of more tokens.

## Further reading

- "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (Snell et al, 2024).
- "Let's Verify Step by Step" (Lightman et al, OpenAI, 2023) — process reward models for verifying reasoning steps.
- DeepSeek R1 technical report (DeepSeek, 2025).
- OpenAI o1 blog posts (limited public technical details).

Next lesson: **Reasoning models (o1, R1) — KV pressure when chain-of-thought runs 10k+ tokens.** We close Part 8 by zooming in on the specific inference challenges of reasoning models, where generating multi-thousand-token reasoning per query is the norm.
