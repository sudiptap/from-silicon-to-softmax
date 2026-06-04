---
title: "Module 11 — Agents from Scratch"
date: "2026-06-04"
module: "agents"
order: 0
tags: ["agents", "llm", "tool-use", "rag", "memory", "planning", "multi-agent", "mcp", "production", "overview"]
author: "Sudipta Pathak"
prerequisites: ["inference-from-scratch", "ml-platforms"]
---

# Agents from Scratch

## Why this module exists

The agent tutorial landscape splits into two unhelpful piles. One pile is *prompt cookbooks* — "here's how to phrase your prompt so the agent does the right thing." The other is *framework demos* — "here's how to call LangGraph / CrewAI / Swarm to make a thing." Neither teaches you how agents actually work.

This module takes the third path: build every meaningful agent primitive **from scratch**, in the smallest amount of code that captures it, then scale it across machines.

By the end you should be able to read a multi-agent system's source on a Friday and have an opinion on its failure modes by Monday — because you'll have built each piece yourself.

## How this fits

The final module of the depth track. Module 7 built the inference engine; this module builds the application layer that consumes it. The trained, served, monitored models from Modules 1-10 become the *substrate* for agents — the LLM is the "engine" but the agent is the "vehicle."

The output: the ability to design and implement production agent systems. Tool use, memory, RAG, planning, multi-agent coordination, MCP — the standard agent toolkit, built from first principles.

## The teaching philosophy

Three rules, adapted from Module 7:

**1. Plain language first, formalism second.** Every concept starts with a sentence you could say at a coffee machine — *"ReAct is a while loop that lets the model decide between thinking and acting at each step"* — before any formalism shows up.

**2. Build it on one machine before you scale it.** A single-process implementation is the unit test for your mental model. Every primitive that *can* be implemented in a single process is built there first.

**3. Then take it distributed.** After the single-process version works, we discuss the scaling story: durable state, multi-worker, observability across processes.

## The roadmap

Twenty-four lessons across ten parts.

### Part 1 — The agent loop (3 lessons)

1. **What makes a system agentic + ReAct from scratch** — the minimal "what is an agent" framing plus a ReAct loop in under 200 lines.
2. **The minimal loop: perceive → reason → act → observe** — the canonical four-step cycle; variations and failure modes.
3. **LLM-as-planner vs LLM-as-worker; state machines vs flexible loops** — when to use which control pattern.

### Part 2 — Tool use (3 lessons)

4. **Native function calling** — what's in the API payload; JSON-schema decoding; constrained-output integration.
5. **Parallel tool calls** — async patterns, fan-out / fan-in; tool-selection at scale.
6. **Designing tools well** — schema, side-effect taxonomy, reversibility, error recovery.

### Part 3 — Memory (2 lessons)

7. **The agent memory hierarchy** — short-term (context), working (scratchpad), long-term (vector/key-value); when each matters.
8. **Memory compaction and eviction** — summarization, sliding window, bounded-context strategies.

### Part 4 — Retrieval-Augmented Generation (4 lessons)

9. **RAG: the core pipeline from scratch** — load, chunk, embed, retrieve, generate.
10. **Chunking strategies** — fixed, semantic, recursive, document-aware.
11. **Hybrid search and reranking** — BM25 + dense + cross-encoder reranker; when each component helps.
12. **Agentic RAG** — the agent decides what to retrieve, when, and how; Self-RAG, Corrective RAG.

### Part 5 — Planning and reasoning (2 lessons)

13. **Plan-and-execute and Tree of Thoughts** — multi-step planning patterns.
14. **Self-reflection, critic loops, and verifier agents** — separating do from check.

### Part 6 — Context engineering (2 lessons)

15. **Context windows as a resource** — budget like RAM; cache-friendly prompts; the cost-quality tradeoffs.
16. **Subagent isolation** — when to spin up a fresh context vs continue in the same one.

### Part 7 — Multi-agent systems (2 lessons)

17. **The orchestrator-worker pattern** — debate, role-based crews, handoff protocols.
18. **Communication failure modes** — deadlock, divergence, role drift; the things that break multi-agent.

### Part 8 — Build it from scratch (2 lessons)

19. **A minimal agent in 100 lines** — framework-agnostic; the patterns every framework implements.
20. **MCP from scratch** — the Model Context Protocol; custom server, custom client; the agent interop landscape.

### Part 9 — Vertical agents (2 lessons)

21. **Coding agents** — Claude Code, Cursor, Aider patterns; the specific architectures for code-editing agents.
22. **Browser and computer-use agents** — Playwright loops, OSWorld, the screen-as-input pattern.

### Part 10 — Production (2 lessons)

23. **Tracing, evaluation, and observability** — Langfuse, OpenTelemetry; trajectory eval vs outcome eval; LLM-as-judge.
24. **Distributed agent scheduling + module wrap** — queue-backed workers, durable state, cost control; the final wrap of Module 11 and the curriculum.

---

## What this module deliberately won't cover

- **Specific framework deep-dives** in code (LangGraph, CrewAI, AutoGen). The patterns generalize; the framework details age fast.
- **Prompt engineering recipes**. Mentioned where it shapes architecture, not as a separate topic.
- **Specific agent benchmarks** in depth. Mentioned in evaluation; the benchmark landscape moves quickly.
- **Safety alignment training** — adjacent to agents but in the RL/training territory, not the systems-engineering layer.
- **Specific commercial agent products** (vs the patterns they implement).

## How to work through it

Every lesson is fully readable as prose. The hands-on sections require:

- An API key to a hosted LLM (OpenAI, Anthropic, or local Ollama).
- Python with `openai` or `anthropic` package.
- For RAG lessons: an embedding model (sentence-transformers, OpenAI embeddings).
- For vector-DB lessons: a local FAISS or pgvector.
- For browser-agent lessons: Playwright.
- For distributed scheduling: a Redis instance (or local).

The hands-on sections build progressively: the ReAct loop in Lesson 1 grows into the multi-agent system in Lesson 17, then into the production scheduler in Lesson 24.

A note on tempo: this module is agent-systems-heavy. Each lesson has the architectural pattern, the minimum-viable implementation, and the production considerations. The minimum-viable code is short (50-200 lines per lesson) so you can fit it in your head; the production considerations are where the depth is.

The capstone (Lesson 24): a distributed agent-scheduling design and the module + curriculum wrap.
