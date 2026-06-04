---
title: "Lesson 7 — The Agent Memory Hierarchy"
date: "2026-06-04"
module: "agents"
order: 7
tags: ["memory", "context", "scratchpad", "vector", "key-value"]
author: "Sudipta Pathak"
prerequisites: ["06-designing-tools"]
---

# Lesson 7 — The Agent Memory Hierarchy

## Why this lesson exists

An agent's memory is what it knows beyond the immediate input. For LLM agents, the natural "memory" is the context window — but context is bounded (128K-1M tokens), expensive to fill (more tokens = more cost), and degrades with very long contexts ("lost in the middle" effect).

A real agent has a *hierarchy* of memory:

1. **Context window**: short-term, current request. Bounded.
2. **Working memory / scratchpad**: notes the agent keeps during a task.
3. **Long-term memory**: persistent across sessions — vector DB, key-value store, structured DB.

This lesson covers the hierarchy, when each layer matters, and the patterns for managing each.

The lesson is reading. The Hands-on builds a small agent with vector memory.

## The hierarchy

By analogy with CPU memory (Module 1):
- **CPU registers** ↔ **Context window**: fastest, smallest.
- **CPU cache** ↔ **Working memory / scratchpad**: medium-speed; structured; agent-managed.
- **DRAM** ↔ **Vector memory / KV store**: larger; needs explicit retrieval.
- **Disk / network** ↔ **External knowledge** (DBs, APIs): unbounded; slower access.

The agent's job: keep the most-relevant info in faster tiers; retrieve from slower tiers when needed.

## Context window (short-term)

What lives here: the current conversation, the recent tool calls and observations, the system prompt.

Properties:
- Limited by model's context window (128K-1M tokens in 2026).
- All tokens add to cost and latency.
- Quality degrades for very long contexts (the model attends less to mid-context info).

Best practices:
- Keep system prompt small and stable (for prompt caching, Lesson 15).
- Truncate old turns when conversation gets long (Lesson 8).
- Use structured formatting so important info is visually grouped.

## Working memory / scratchpad

A *scratchpad* is the agent's notes about the current task: what it's tried, what it's discovered, what it plans next. Distinct from the message history (which is the LLM's literal context); the scratchpad is what the agent writes about its task.

Implementation:
- A plain text or JSON document the agent reads and writes.
- Tools like `read_scratchpad`, `write_to_scratchpad`, `update_scratchpad`.
- The scratchpad's contents are pulled into the context when relevant.

Why have a scratchpad separate from message history?
- The message history is full of low-information stuff (system messages, intermediate tool results, formatting).
- The scratchpad is high-information: distilled facts.
- Compaction (Lesson 8) can summarize message history; scratchpad survives.

For long-running tasks (an agent that works on something for hours), the scratchpad is essential.

## Long-term memory

What lives here: information that persists across sessions or tasks.

Two main types:

**Vector memory**: embeddings + similarity search. For unstructured information (past conversations, documents, notes).

**Key-value or structured memory**: for facts. "User Alice prefers Python; last logged in on Tuesday."

The agent reads from long-term memory when relevant; writes to it after meaningful events.

Implementation:
- Vector DB (Pinecone, Weaviate, Chroma, pgvector): store agent's past interactions; retrieve by similarity.
- KV store (Redis, DynamoDB): store agent-managed facts; retrieve by key.
- Hybrid: facts in KV; full text / context in vector DB.

## When each layer matters

Different agent types use different memory layers:

**Stateless single-turn agents** (search a question, return an answer): just context. No need for scratchpad or long-term.

**Multi-turn chatbots**: context + summarized history. Scratchpad optional. Long-term memory if user-specific.

**Task-completion agents** (write this report, fix this bug): context + scratchpad. Scratchpad tracks what's been tried.

**Personal assistant agents** (remember my preferences): context + KV memory. The KV stores user-specific facts.

**Research agents that learn over time**: context + scratchpad + vector memory + KV memory. The full hierarchy.

## Vector memory: write

Writing to vector memory:
1. Take a chunk of text (a conversation summary, a discovered fact).
2. Embed it: `embedding = model.embed(text)`.
3. Store: `vector_db.add(id=uuid, vector=embedding, metadata={"text": text, "timestamp": now()})`.

The agent writes when something important happens (user states a preference, agent discovers a fact, task completes).

## Vector memory: read

Reading:
1. Form a query (from the current context).
2. Embed: `query_emb = model.embed(query)`.
3. Search: `results = vector_db.search(query_emb, k=5)`.
4. Inject results into context.

The agent reads at the start of each task / turn — "what relevant memories do I have?"

## The memory tools

A typical agent with memory has tools:

```
remember(content) → store something in long-term memory
recall(query) → search long-term memory; return relevant
update_fact(key, value) → set a fact in KV memory
get_fact(key) → retrieve a fact from KV memory
list_facts() → list all known facts
```

The agent's reasoning uses these like any other tools. The architecture choice is whether memory ops are explicit (the agent decides) or implicit (the system auto-injects relevant memory each turn).

Implicit is more reliable for chatbots; explicit gives the agent more control.

## The "memory contamination" risk

A subtlety: long-term memory can contaminate the agent's reasoning if not managed.

If past failed attempts are stored as facts, the agent might be biased against trying similar approaches. If user preferences are stored too granularly, the agent might over-fit to old preferences that have changed.

Mitigations:
- **Decay**: older memories have lower weight in retrieval.
- **Confidence scores**: each memory has a confidence; low-confidence memories don't dominate.
- **Explicit overrides**: the user can correct stored facts.

## What you should believe after this lesson

Three sentences:

**1. Agent memory is a hierarchy**: context window (short-term, bounded), working memory / scratchpad (per-task, agent-managed), long-term memory (vector DB and/or KV store, persistent). Different agent types use different layers; most don't need all of them.

**2. Vector memory** is for unstructured content (past conversations, discovered facts); KV memory is for structured facts. The agent reads at the start of each task; writes when important events happen.

**3. Memory contamination is a real risk** — stale facts, over-fitting to old preferences, biases from past failures. Mitigations: decay, confidence scores, explicit overrides.

## Hands-on (at home)

Build a small agent with vector memory.

```python
# vector_memory_agent.py
# pip install sentence-transformers chromadb openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

client = OpenAI()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
db_client = chromadb.Client()
memory = db_client.create_collection("agent_memory")

def remember(content: str):
    emb = embedder.encode([content])[0].tolist()
    memory.add(documents=[content], embeddings=[emb], ids=[str(memory.count())])

def recall(query: str, k: int = 3):
    emb = embedder.encode([query])[0].tolist()
    results = memory.query(query_embeddings=[emb], n_results=k)
    return results['documents'][0]

# Tools for the agent.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact for future recall.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search past memories for relevant info.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
]

# A standard ReAct loop (omitted for brevity; see Lesson 4) that uses these tools.

# Demo: store and recall.
remember("Alice prefers Python.")
remember("Bob's email is bob@example.com.")
remember("The deployment script is at /scripts/deploy.sh.")

print(recall("What language does Alice use?"))
# Should return Alice's preference.
```

The agent can `remember` and `recall` as part of its tool set; over time, the vector memory accumulates relevant facts.

For a real production agent, use a managed vector DB (Pinecone) instead of in-memory Chroma; consider hybrid search (vector + keyword) for better retrieval.

## Further reading

- "Generative Agents: Interactive Simulacra of Human Behavior" (Park et al, 2023) — memory-rich agent design.
- "MemGPT: Towards LLMs as Operating Systems" (Packer et al, 2023) — explicit memory hierarchy.
- LangChain memory documentation.

Next lesson: **Memory compaction and eviction.** Summarization, sliding window, bounded-context strategies. When the message history gets too long, what do you do?
