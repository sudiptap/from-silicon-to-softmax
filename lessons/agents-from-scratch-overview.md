---
title: "Agents from Scratch: Series Overview"
date: "2026-05-26"
excerpt: "A tutorial series on the full design space of LLM agents — the agent loop, tool use, memory, RAG, planning, context engineering, multi-agent systems, and the distributed infrastructure that actually ships them. Built from scratch on one machine, then taken distributed."
module: "agents"
order: 0
tags: ["agents", "llm", "tool-use", "rag", "memory", "planning", "multi-agent", "mcp", "production", "overview"]
author: "Sudipta Pathak"
prerequisites: []
---

# Agents from Scratch

## Why another agent series

The agent tutorial landscape splits into two unhelpful piles. One pile is *prompt cookbooks* — "here's how to phrase your prompt so the agent does the right thing." The other is *framework demos* — "here's how to call LangGraph / CrewAI / Swarm to make a thing." Neither teaches you how agents actually work.

This series takes the third path: build every meaningful agent primitive **from scratch**, in the smallest amount of code that captures it, then scale it across machines.

By the end you should be able to read a multi-agent system's source on a Friday and have an opinion on its failure modes by Monday — because you'll have built each piece yourself.

## The teaching philosophy

The same three rules from the [Inference series](/curriculum/inference-from-scratch-overview) apply here, adapted for agents.

### 1. Plain language first, formalism second

You can't reason about a planner you can only call. Every concept starts with a sentence you could say to a colleague at a coffee machine — *"ReAct is a `while` loop that lets the model decide between thinking and acting at each step"* — before any formalism shows up.

When the formalism arrives (state machines, protocols, schemas), we walk through it piece by piece, no skipped steps. If a message gets transformed before it's appended to context, we say exactly what changes and why.

### 2. Build it on one machine before you scale it

A single-process implementation is the unit test for your mental model. If you can't write a ReAct loop in 100 lines of plain Python, you have no business reasoning about durable distributed agent scheduling.

Every primitive that *can* be implemented in a single process is built there first — no framework, no abstractions you didn't write yourself. You'll see the naive version, the bug, the fix, and the production version. In that order.

### 3. Then take it distributed

After the single-process version works, we rewrite it for the realistic case: multiple agent workers behind a queue, durable state in Redis or Postgres, tracing across processes, scheduling under cost and rate-limit constraints. This is where most agent content drops off and where production actually lives.

You'll see the message protocols, the failure modes (deadlock, divergence, role drift), and how the patterns scale — or don't.

---

## The roadmap

Ten parts, ~60 tutorials, sequenced so each one earns the next. Topics get linked here as they ship.

### Part 1 — The agent loop

The foundation. Before frameworks, before tools — what *is* an agent?

1. What makes a system "agentic" — autonomy, tools, feedback
2. The minimal loop: perceive → reason → act → observe
3. **ReAct from scratch** in <200 lines, no framework
4. LLM-as-planner vs LLM-as-worker
5. State machines vs flexible loops — when each wins

### Part 2 — Tool use

The thing that makes an agent useful. The thing that makes it dangerous.

6. Native function calling — what's actually in the API payload
7. JSON-schema decoding and grammar-constrained outputs
8. **Parallel tool calls** — async patterns, fan-out/fan-in
9. Tool selection at scale — when you have 100+ tools (retrieval, hierarchies)
10. Tool result feedback — error recovery, retries, idempotency
11. Designing tools well — schema, side-effect taxonomy, reversibility

### Part 3 — Memory

The agent's view of the world beyond the current context.

12. Why the context window isn't enough — the agent memory hierarchy
13. Conversation memory — buffer, summary, sliding window
14. Vector memory — embeddings as a retrieval index
15. Key-value & structured memory — when you don't need similarity
16. Episodic vs semantic memory
17. **Memory compaction & eviction** — the bounded-context problem

### Part 4 — Retrieval-Augmented Generation

RAG deserves its own part. It's the single most common production agent pattern, and every interview asks about it. We build the whole stack, including the bits everyone hand-waves through.

18. Why RAG — the parametric vs non-parametric memory split
19. The core pipeline from scratch — load, chunk, embed, retrieve, generate
20. Chunking strategies — fixed, semantic, recursive, document-aware
21. Embedding models — what they encode, where they fail
22. **Vector DBs from the inside** — HNSW, IVF, pgvector
23. Hybrid search — BM25 + dense + reranking with cross-encoders
24. Query transformation — HyDE, multi-query, step-back prompting
25. Advanced retrieval — parent-child, contextual retrieval, ColBERT
26. **Agentic RAG** — the agent decides what to retrieve, when, and how
27. Self-RAG & Corrective RAG
28. Graph RAG — knowledge graph + vector hybrid
29. **RAG evaluation** — faithfulness, context precision/recall, RAGAs

### Part 5 — Planning & reasoning

When one step isn't enough.

30. Single-step vs multi-step planning
31. Plan-and-execute
32. Tree of Thoughts
33. **LATS** — Language Agent Tree Search
34. Self-reflection & critic loops
35. **Verifier agents** — separating "do" from "check"

### Part 6 — Context engineering

The 2025–26 hot topic. Treating the context window as a resource you have to budget.

36. Context windows as a resource — budget like RAM
37. Compaction strategies — summarization, eviction, layered context
38. **Subagent isolation** — when to spin up a fresh context
39. Cache-friendly prompts — structuring for prompt caching
40. Context pollution — drift, recency bias, instruction decay

### Part 7 — Multi-agent systems

Multiple agents, multiple failure modes.

41. The orchestrator-worker pattern
42. Debate & adversarial agents
43. Role-based crews (CrewAI-style)
44. Handoff protocols — OpenAI Swarm patterns
45. **A2A and the interop landscape**
46. Communication failure modes — deadlock, divergence, role drift

### Part 8 — Build it from scratch

The frameworks everyone uses, taken apart.

47. A minimal agent loop in 100 lines
48. **LangGraph internals** — rebuild the executor
49. AutoGen architecture
50. OpenAI Agents SDK / Swarm internals
51. Claude Agent SDK & Claude Code's loop
52. **MCP from scratch** — protocol, custom server, custom client

### Part 9 — Vertical agents

Where the patterns meet real product surfaces.

53. **Coding agents** — Claude Code, Cursor, Aider patterns
54. Browser & computer-use agents — OSWorld, Playwright loops
55. Research agents — Deep Research patterns
56. **Data agents** — NL2SQL, autonomous analytics
57. Long-running & scheduled agents — cron, event-driven, durable execution

### Part 10 — Production agents

This is the part where the systems work pays off. Distributed scheduling, observability, eval pipelines, cost control — agents as infrastructure, not as demos.

58. Trajectory eval vs outcome eval; LLM-as-judge
59. Benchmarks — SWE-bench, GAIA, WebArena, OSWorld
60. **Tracing & replay** — Langfuse / custom OTEL pipelines
61. Safety — prompt injection, sandboxing, HITL checkpoints
62. **Cost & latency engineering** — model routing, parallelism, caching
63. **Distributed agent scheduling** — Ray, queue-backed workers, statefulness
64. A/B testing agents in production

---

## How this series fits the bigger picture

This is one of four breadth tracks I'm building, with [ML infrastructure](/curriculum) as the depth bar — a T-shaped portfolio.

- **Pre-Training from Scratch** — *planned*
- **Post-Training from Scratch** — *planned*
- **[Inference from Scratch](/curriculum/inference-from-scratch-overview)** — *shipping now*
- **Agents from Scratch** — *this series*

Every breadth track lands in the same place: a "Production X" part that uses the infra depth — distributed scheduling, observability, queue infrastructure, eval pipelines. That's the bridge back to the vertical bar of the T.

---

## How to follow along

You can read these in order — they're sequenced for that — or jump to the part most relevant to what you're building. RAG (Part 4) and Production Agents (Part 10) are the parts most interview loops care about; if you're optimizing for that, start there and double back for foundations.

Code lives alongside the prose. Single-process versions run on anything with a CPU and an API key. Distributed sections assume access to a small cluster, a managed Redis/Postgres, and a vector DB — instructions for free-tier setups are included where they apply.

If you find an error, an explanation that didn't land, or a topic that's missing — open an issue or reach out. This series is meant to be lived in, not just published.
