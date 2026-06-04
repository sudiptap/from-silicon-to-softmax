---
title: "Lesson 15 — Context Windows as a Resource"
date: "2026-06-04"
module: "agents"
order: 15
tags: ["context-engineering", "prompt-caching", "budget", "cost-quality"]
author: "Sudipta Pathak"
prerequisites: ["14-self-reflection-verifiers"]
---

# Lesson 15 — Context Windows as a Resource

## Why this lesson exists

The context window is the agent's working memory. Treating it as "unlimited because the API accepts 1M tokens" is the path to expensive, slow, low-quality agents. Treating it as a finite resource — like RAM — that you have to budget is the right mental model.

This lesson covers context engineering: budgeting the context, structuring for prompt caching, the cost-quality tradeoffs.

The lesson is reading. The Hands-on calculates context budget for a typical agent.

## Context as RAM

The analogy:
- **You have a fixed budget**: 128K-1M tokens.
- **Each thing in context has a cost**: tokens consumed.
- **Each thing in context has a benefit**: information available to reason on.
- **Quality degrades when the budget is exceeded** or even approached (the "lost in the middle" effect).

So budgeting matters. Like RAM:
- Frequently-used items should stay loaded (cache).
- Rarely-used items go to long-term storage (vector DB).
- Total resident items must fit in the budget.

## Where context comes from

A typical agent's context:

1. **System prompt** (instructions, persona): typically 500-5000 tokens.
2. **Tool schemas**: 500-5000 tokens for a moderate tool set.
3. **Conversation history**: variable; can grow to many thousands.
4. **RAG context** (retrieved chunks): 1000-10000 tokens.
5. **Current user message**: typically <1000 tokens.
6. **Scratchpad**: variable.

Total: easily 20K-50K tokens for a non-trivial agent at any single LLM call.

At $5/M input tokens, each call is $0.10-$0.25. For a multi-step agent doing 10 calls per task: $1-2.50 per task.

For high-volume agents (millions of calls per day): the context tax dominates costs.

## Prompt caching

Anthropic, OpenAI, and Google all support *prompt caching*: cached prefixes don't count against per-call costs / latency.

The model:
- You mark certain parts of the prompt as cacheable (e.g., system prompt + tool schemas).
- First call: full cost.
- Subsequent calls within cache lifetime (typically 5-60 minutes): the cached prefix is 90% cheaper.

For agents with stable prefixes (system + tools), caching cuts costs dramatically. A 5000-token prefix used across 1000 calls: full cost on first call, 10% cost on the other 999.

Implementation:
- **Anthropic**: `cache_control: {"type": "ephemeral"}` on cacheable content blocks.
- **OpenAI**: automatic for prefixes ≥ 1024 tokens.
- **Google**: explicit caching API.

Structure your prompts so the stable parts are at the *beginning*; cache from there. Variable parts (user message, recent context) go at the end.

## The "stable to variable" ordering

For prompt caching:

```
[Stable]
- System prompt
- Tool schemas
- Long-lived context (loaded once)

[Variable]
- Conversation history
- Current user message
- Tool results
```

The stable section is cached. The variable section is sent fresh each call.

This is a meaningful design constraint: design your prompts top-down (stable → variable), not bottom-up (most recent stuff first).

## Compaction (Lesson 8) interacts

Memory compaction (summarization, sliding window) is a context-budget mechanism. The strategies:

- **Aggressive compaction**: keep context small; cheap per call; risk losing info.
- **Permissive compaction**: keep context large; expensive per call; preserve info.

Tune for your application. Latency-sensitive: aggressive. Quality-sensitive: permissive.

## The "lost in the middle" effect

LLMs attend less to content in the middle of long contexts. The beginning and end get more attention; the middle is at risk of being ignored.

Implications for context engineering:
- **Important instructions at the start AND end of the system prompt** (some practitioners include "remember: <key constraint>" near the end).
- **Critical RAG context near the end of the user turn**, not buried in middle.
- **Shorter is better than longer** when in doubt.

This degrades less in newer long-context models but still matters. Test with your specific model.

## The cost-quality curve

A typical curve:

- **0-10% of context budget**: cheap; not enough info; quality may suffer from missing context.
- **10-50%**: cost increases linearly; quality plateaus.
- **50-90%**: cost still rises; quality starts to degrade (lost-in-middle).
- **>90%**: cost very high; quality often worse than at 50%.

The sweet spot: 20-50% utilization. Keep context concentrated on what matters.

## Budgeting per call

A practical exercise: for each agent call, write down what's in context:

```
- System prompt: 1000 tokens (cached)
- Tool schemas: 500 tokens (cached)
- Conversation history: ~2000 tokens
- Most recent user message: ~100 tokens
- Retrieved RAG chunks: ~2000 tokens

Total: ~5600 tokens.
Of those, 1500 cached → 4100 paid per call.
At $5/M tokens → $0.02 per call.
```

For 100 calls per task (multi-step agent): $2 per task. At 10K tasks/day: $20K/day. Add output costs.

Now figure out where to trim:
- Reduce RAG chunks from 5 to 3 → save 800 tokens × $5/M = $0.004 per call.
- Switch from full history to summary → save 1000 tokens, similar savings.

Cumulative savings can be 30-50% with good context engineering.

## What you should believe after this lesson

Three sentences:

**1. Context is a finite resource** — treat it like RAM. Budget per call; identify what's cacheable vs variable; structure prompts stable-to-variable for prompt caching. 90% cost reduction on cached prefixes is common.

**2. Quality degrades with very long contexts** (lost-in-middle). The sweet spot for cost and quality: 20-50% of context budget utilized. Compaction (Lesson 8) is the mechanism for keeping context concentrated.

**3. Per-call budgeting is a real engineering exercise**: list what's in context; assess what's cacheable; trim what's not load-bearing. Cumulative savings of 30-50% are achievable.

## Hands-on (at home)

Calculate context budget for a typical agent.

```python
# context_budget.py
from collections import OrderedDict

# Tokens count (estimate).
def count_tokens(text):
    return len(text) // 4  # rough; actual depends on tokenizer

components = OrderedDict([
    ("system_prompt", "You are a helpful assistant. " * 100),  # ~500 tokens
    ("tool_schemas", "{...tool schema details...}" * 50),  # ~600 tokens
    ("conversation_history", "{...past turns...}" * 200),  # ~2400 tokens
    ("retrieved_chunks", "{...RAG chunks...}" * 200),  # ~2400 tokens
    ("current_user_message", "What's the weather?"),  # ~5 tokens
])

print(f"{'Component':30s} {'Tokens':>10s} {'$/M=5 calls=100':>15s}")
total_tokens = 0
for name, content in components.items():
    tokens = count_tokens(content)
    total_tokens += tokens
    cost_per_call = tokens * 5e-6
    cost_per_100_calls = cost_per_call * 100
    print(f"{name:30s} {tokens:>10d} {cost_per_100_calls:>15.2f}")

print(f"\n{'TOTAL':30s} {total_tokens:>10d}")
print(f"\nWith prompt caching of (system_prompt + tool_schemas):")
cached_tokens = count_tokens(components['system_prompt']) + count_tokens(components['tool_schemas'])
paid_tokens = total_tokens - cached_tokens + 0.1 * cached_tokens  # 90% off cached
cost_per_call = paid_tokens * 5e-6
print(f"  Effective tokens per call: {paid_tokens:.0f}")
print(f"  Cost per call: ${cost_per_call:.4f}")
print(f"  Per 100 calls: ${cost_per_call * 100:.2f}")
```

You'll see the magnitude of savings from caching. For a real agent, run this exercise on your own context structure.

## Further reading

- "Lost in the Middle" (Liu et al, 2023).
- Anthropic's prompt caching documentation.
- OpenAI's prompt caching documentation.
- "Context Engineering for Agents" — emerging blog-post topic in 2024-2026.

Next lesson: **Subagent isolation.** When to spin up a fresh context vs continue in the same one. The pattern of delegating subtasks to "fresh-eyes" sub-agents.
