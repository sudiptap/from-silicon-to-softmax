---
title: "Lesson 19 — Prefix Caching & Cross-Request KV Reuse"
date: "2026-06-04"
module: "inference-from-scratch"
order: 19
tags: ["prefix-caching", "kv-cache", "system-prompt", "chat", "sgl"]
author: "Sudipta Pathak"
prerequisites: ["18-paged-attention"]
---

# Lesson 19 — Prefix Caching & Cross-Request KV Reuse

## Why this lesson exists

LLM workloads have heavy prefix structure. A chatbot's requests all start with the same system prompt. A coding assistant's requests all include the same large file context. A few-shot RAG application's requests all share the same retrieval context plus task instructions. In all these cases, recomputing K/V for the shared prefix on every request is pure waste.

Prefix caching detects these shared prefixes across requests and reuses the cached K/V. When a new request arrives, the runtime checks if its prefix matches any cached prefix; if so, it skips that portion of prefill and starts attention from the cache.

This lesson covers the prefix-caching mechanism, the data structures (trie-based, hash-based), the situations where it pays off, and the operational story of using it well.

The lesson is reading. The Hands-on demonstrates prefix caching with vLLM and measures the speedup.

## When prefix caching wins

The clear wins:

**1. Chatbots with system prompts.** A typical system prompt is 200-2000 tokens; every request includes it; caching saves the prefill cost of recomputing it.

**2. Few-shot prompting.** "Here are 5 examples of the task...now do this one." The examples are the same for every request; caching saves them.

**3. RAG with shared retrieval contexts.** When multiple queries hit the same retrieved document, the document's KV can be cached and shared.

**4. Code assistants.** The current file context is the same across the user's recent edits; cache it.

**5. Long-document Q&A.** "Here is a 30K-token document. Answer questions about it." Multiple questions; the document's KV is cached once.

The savings: prefix prefill is `O(N_prefix² × D)` for attention plus `O(N_prefix × D²)` for FFN. For a 2K-token prefix, this is dozens of milliseconds saved per request. Over a server's lifetime, this is enormous compute saved.

When prefix caching is a wash:
- Requests with mostly unique prefixes (single-turn user queries with no common context).
- Very short prefixes where the prefill cost was negligible anyway.

## The detection problem

Given a new request, how do you know which cached prefix to use? Three approaches:

**1. Hash-based.** Hash the first N tokens of the prompt; if there's a cache entry with that hash, use it. Simple, fast, exact.

**2. Trie-based.** Maintain a trie indexed by token sequences. The trie's nodes hold KV cache blocks. A new request walks down the trie matching its prefix; the deepest matching node tells you the cached prefix length and the blocks to reuse. This is what SGLang's RadixAttention (Lesson 44) does.

**3. Application-level hints.** The user tells the runtime "this prompt has prefix X." Most precise but requires app cooperation.

In practice, hash-based is common in vLLM; trie-based is the SGLang approach. Both work.

## The block-level granularity

Prefix sharing happens at the *block* level (paged attention blocks; Lesson 18). If the shared prefix is 200 tokens and the block size is 16, the runtime can share 12 full blocks plus a partial block at the end. The partial block is usually copy-on-write'd to the new request (since the request will write the next 13th block when generating).

The interaction with paged attention is what makes prefix caching practical: blocks are the unit of allocation, sharing, and freeing.

## The eviction problem

Cached prefixes occupy KV memory. The runtime can't keep every prefix forever; eventually it runs out of blocks. Eviction policies:

- **LRU**: evict the least-recently-used prefix when memory pressure forces it.
- **Size-aware**: prefer to evict larger prefixes that haven't been used recently.
- **Reference-counted**: never evict a prefix that's currently in use by an active request.

vLLM uses an LRU-ish policy on the prefix cache; cached blocks that aren't in active use can be evicted when the free pool runs low.

## What you actually save

The math for a chat application:
- System prompt: 500 tokens.
- Average user turn: 100 tokens.
- Average response: 200 tokens.
- 1000 requests per hour.

Without prefix caching:
- Each request prefills 600 tokens (system + user).
- Prefill cost: 600 tokens × ~25 ms per 100 tokens = 150 ms per request.
- Total prefill compute: 150 ms × 1000 = 150 seconds per hour.

With prefix caching:
- System prompt prefilled once; cached.
- Each request prefills only 100 user tokens.
- Prefill cost: ~25 ms per request.
- Total prefill compute: 25 seconds per hour.

Compute savings: ~6× on prefill. For the user, TTFT drops from ~150 ms to ~25 ms — visible improvement.

For RAG with 30K-token shared documents, the savings are much bigger; the prefill cost dominates without caching.

## The chat-template gotcha

A subtle issue: chat templates can break prefix sharing. If your system prompt template is:

```
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```

Then the *exact* token sequence at the start depends on both the system prompt and the user message — and they're interleaved in token positions. The system prompt's tokens aren't a true prefix of the conversation because of the chat-template structure.

In practice, the template is usually:
- System prompt → fixed prefix tokens (same across requests with the same system prompt).
- User message → comes after the system prompt's `<|im_end|>` token.

So the shared prefix is `<|im_start|>system\n{system_prompt}\n<|im_end|>\n`. This is the longest shared prefix; the cache stores up to this point.

Some templates make sharing harder (the chat formatting interleaves user turn boundaries with assistant turn boundaries). Designing prompts with caching in mind is a real consideration for production deployments.

## A note on prefix caching for multi-turn conversations

For a multi-turn chat:

Turn 1: `[system, user1, assistant1]`.
Turn 2: `[system, user1, assistant1, user2, assistant2]`.
Turn 3: `[system, user1, assistant1, user2, assistant2, user3, assistant3]`.

Each turn's prefix is the previous turn's full conversation. If the runtime keeps the cache from turn N around when turn N+1 arrives, the entire prior conversation is cached. Only the new user message needs prefill.

This is how production chat servers achieve very fast TTFT mid-conversation: the cache from the last turn is reused for the current turn. The cache grows by ~one user turn + ~one assistant turn per round.

The catch: if the user takes 5 minutes between turns, the runtime may have evicted the conversation's cache to serve other requests. The next message has cold prefix; the user sees the slower TTFT. Some deployments avoid this by pinning active-user caches.

## What you should believe after this lesson

Three sentences:

**1. Prefix caching reuses KV cache for shared prefixes across requests** — system prompts, few-shot examples, RAG documents. The block-level granularity from paged attention makes the sharing efficient; the savings on prefill compute are typically 5-10× for chat applications.

**2. The detection mechanism varies**: hash-based (vLLM), trie-based (SGLang's RadixAttention), or application-level hints. Hash and trie both work; the choice depends on the runtime.

**3. Multi-turn chat is the killer app**: each turn's prefix is the previous full conversation, so the cache from turn N serves turn N+1's prefill in microseconds rather than tens of milliseconds. Production chat servers depend on this for fast mid-conversation TTFT.

## Hands-on (at home)

Measure the speedup with vLLM's prefix caching enabled.

```bash
# Install vLLM (requires NVIDIA GPU).
pip install vllm

# Run vLLM server with prefix caching enabled.
vllm serve mlx-community/Qwen2.5-1.5B-Instruct --enable-prefix-caching
```

```python
# prefix_caching_bench.py
import openai
import time

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

LONG_SYSTEM = "You are an expert assistant.  " * 200  # ~1000 tokens
USER_MESSAGES = [
    "What is the capital of France?",
    "What is the square root of 144?",
    "Explain quantum tunneling in one paragraph.",
    "What's the difference between TCP and UDP?",
    "Write a haiku about programming.",
]

def measure_ttft(user_msg, model="Qwen/Qwen2.5-1.5B-Instruct"):
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=100,
        stream=True,
    )
    first_token_time = None
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - t0
                break
    return first_token_time * 1000

# Cold cache (first request).
print(f"Request 1: TTFT = {measure_ttft(USER_MESSAGES[0]):.0f} ms (cold cache)")

# Subsequent requests should be much faster (prefix cached).
for i, msg in enumerate(USER_MESSAGES[1:], start=2):
    print(f"Request {i}: TTFT = {measure_ttft(msg):.0f} ms")
```

Expected: request 1's TTFT includes prefill of the 1000-token system prompt (~50 ms); subsequent requests should have TTFT ~5-15 ms because the system prompt is cached.

Disable prefix caching (`--no-enable-prefix-caching`) to confirm the speedup is from caching.

## Further reading

- vLLM documentation on prefix caching.
- "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al, 2023) — introduces RadixAttention (the trie-based variant).
- "Memory-Efficient LLM Serving with PagedAttention" — vLLM paper section on cross-request sharing.
- "Prompt Caching" — OpenAI / Anthropic API docs; the same idea exposed as an API feature.

Next lesson: **KV cache quantization.** Beyond paged allocation and prefix sharing, the KV cache can be compressed via quantization to fit more sequences in the same memory. INT8 / INT4 / FP8 each have practical implementations and tradeoffs.
