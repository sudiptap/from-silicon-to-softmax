---
title: "Lesson 44 — RadixAttention: SGLang's Prefix Tree"
date: "2026-06-04"
module: "inference-from-scratch"
order: 44
tags: ["radix-attention", "sglang", "prefix-tree", "kv-sharing", "agent"]
author: "Sudipta Pathak"
prerequisites: ["43-prefill-decode-disaggregation"]
---

# Lesson 44 — RadixAttention: SGLang's Prefix Tree

## Why this lesson exists

Prefix caching (Lesson 19) shares KV cache between requests that have the same prefix. The vLLM implementation uses a hash-based lookup: hash the first N tokens; if cached, reuse.

RadixAttention (SGLang, Zheng et al, 2023) generalizes this with a *radix tree* (trie) of token sequences. Multiple requests' KV caches are organized as a tree where each node is a token sequence; siblings share their parent's prefix. The tree allows much more flexible sharing — overlap between two requests' middle sections, not just their prefixes.

This lesson covers the radix-tree mechanism, the use cases it enables (agent reasoning, multi-turn conversation, tree-based search), and why SGLang adopted it as their core scheduling primitive.

The lesson is reading. The Hands-on visualizes a small radix tree.

## The radix tree

A radix tree (trie) where:
- Each *node* represents a sequence of tokens.
- Each *edge* labels the connection with the next token.
- The path from root to any node represents a token sequence.

For example, three requests with prompts:
- "The cat sat on the mat"
- "The cat sat on the rug"
- "The dog sat on the mat"

The radix tree:

```
ROOT
├── "The cat sat on the "
│    ├── "mat"
│    └── "rug"
└── "The dog sat on the mat"
```

The first two requests share `"The cat sat on the "`; only the last token differs. The third request shares only `"The "`.

Each radix-tree node has its associated KV cache for the tokens in that node. When a new request arrives, find its longest matching prefix in the tree; reuse the KV caches along that path; only compute KV for the unmatched suffix.

## The cache management

The radix tree replaces a flat hash table:

- **Insert**: when a new request arrives, walk the tree to find the longest matching prefix; create new nodes for the unmatched suffix.
- **Lookup**: same — walk to find longest match.
- **Eviction**: nodes can be evicted when memory pressure forces it. LRU eviction; child nodes evicted before parents.

The radix tree is more expressive than flat prefix caching:
- Mid-sequence sharing (two requests share their *middle* tokens, e.g., the same system prompt embedded in different positions). The radix tree captures this if the token sequences truly overlap.
- Multi-turn conversations naturally form long chains; the chain structure efficiently expresses many turns of the same conversation.

For agents that do tree-search reasoning (best-of-N with shared root, beam search, MCTS), the tree structure matches the search structure exactly. Each search branch shares the parent's KV cache.

## When RadixAttention shines

The strong cases:

**1. Agent workflows with tree search.** An LLM agent that considers multiple action options at each step has a tree of decisions; the shared decision history is in the radix tree's shared nodes.

**2. Multi-turn conversations.** Each turn is the previous conversation + new user message. The tree's chain structure captures this with maximum sharing.

**3. Few-shot prompting.** Many requests sharing the same few-shot examples; the radix tree shares them naturally.

**4. RAG with overlapping retrieved contexts.** Two queries hitting the same retrieved document share the document's KV via the tree.

**5. Best-of-N sampling.** Generate N candidate completions from the same prompt; they all share the prefix; the tree branches at the divergence points.

## SGLang's full design

SGLang is more than just radix attention — it's a "structured generation" framework that combines:

- RadixAttention for KV cache sharing.
- A request DSL for composing complex prompts (multi-turn, branching, structured output).
- Constrained decoding integration.
- Continuous batching + chunked prefill.

The DSL example:

```python
@sgl.function
def two_step_reasoning(s, question):
    s += "Q: " + question + "\n"
    s += "Reasoning: " + sgl.gen("reasoning", max_tokens=200)
    s += "\nAnswer: " + sgl.gen("answer", max_tokens=50)
```

The framework tracks the shared prefix structure automatically; multiple calls with the same question share the prompt via radix attention.

## Comparison to vLLM's prefix caching

vLLM uses hash-based prefix caching (Lesson 19); SGLang uses RadixAttention. The differences:

| Aspect | vLLM hash | SGLang Radix |
| ------ | --------- | ------------ |
| Lookup | Hash table | Tree walk |
| Mid-sequence sharing | No | Yes (within reason) |
| Multi-turn efficiency | Good | Excellent |
| Tree-search workloads | Limited | Native |
| Implementation complexity | Lower | Higher |
| Memory overhead | Lower | Slightly higher (tree bookkeeping) |

For typical chat workloads, both work well. For agent / tree-search / structured-generation workloads, SGLang's radix structure is meaningfully better.

## What you should believe after this lesson

Three sentences:

**1. RadixAttention organizes the KV cache as a tree of token sequences** where siblings share their parent's prefix. More flexible than flat prefix caching; supports mid-sequence sharing, multi-turn chains, and tree-based search workflows natively.

**2. SGLang adopted RadixAttention as its core scheduling primitive** because the framework targets structured generation (agents, multi-turn chat, branching reasoning) where the tree structure matches the workload pattern.

**3. For typical chat workloads, vLLM's hash-based prefix caching is sufficient**; RadixAttention's advantages emerge for tree-search, agent reasoning, and best-of-N sampling. Choose the runtime based on workload pattern.

## Hands-on (at home)

A tiny radix tree for token sequences.

```python
# radix_tree.py
class RadixNode:
    def __init__(self):
        self.children = {}  # token_id → RadixNode
        self.tokens = []    # tokens stored at this node (for visualization)
        self.kv_cache = None  # placeholder for the KV cache associated with this node

    def insert(self, tokens, kv_cache):
        node = self
        for t in tokens:
            if t not in node.children:
                node.children[t] = RadixNode()
            node = node.children[t]
            node.tokens.append(t)
        node.kv_cache = kv_cache

    def longest_match(self, tokens):
        node = self
        matched = 0
        for t in tokens:
            if t in node.children:
                node = node.children[t]
                matched += 1
            else:
                break
        return matched, node

# Demo.
tree = RadixNode()
# Three "prompts" represented as token ID sequences.
tree.insert([1, 2, 3, 4, 5], kv_cache="cache_for_prompt_1")
tree.insert([1, 2, 3, 4, 6], kv_cache="cache_for_prompt_2")
tree.insert([1, 2, 7, 8, 9], kv_cache="cache_for_prompt_3")

# Look up: new request with prompt [1, 2, 3, 4, 10].
new_request = [1, 2, 3, 4, 10]
matched, node = tree.longest_match(new_request)
print(f"New request: {new_request}")
print(f"Longest match: {matched} tokens (path: {new_request[:matched]})")
print(f"Need to compute KV for: {new_request[matched:]}")
# Should match 4 tokens (the [1, 2, 3, 4] prefix is shared with prompts 1 and 2).
```

For SGLang in practice:

```bash
pip install sglang
# Launch the SGLang server.
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
# Use the SGLang DSL to write structured generation programs.
```

## Further reading

- "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al, 2023).
- SGLang GitHub documentation.
- "Programming Frameworks for LLMs: Past, Present, and Future" — survey of LLM serving frameworks.

Next lesson: **CUDA Graphs for decode.** A kernel-launch-overhead optimization. Each decode step launches many small kernels; the per-launch overhead adds up. CUDA Graphs lets you capture a graph of kernels once and replay it without per-launch cost.
