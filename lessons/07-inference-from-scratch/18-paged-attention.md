---
title: "Lesson 18 — PagedAttention"
date: "2026-06-04"
module: "inference-from-scratch"
order: 18
tags: ["paged-attention", "vllm", "kv-cache", "block-allocation", "fragmentation"]
author: "Sudipta Pathak"
prerequisites: ["17-kv-cache-layout"]
---

# Lesson 18 — PagedAttention

## Why this lesson exists

The naive KV cache allocation: pre-allocate `max_context × per_token_cost` for every request, regardless of actual length. The problem: most requests don't use the full context. A 50-token request in a system with `max_context=4096` wastes ~99% of the allocation. Across many concurrent requests on a server, this fragmentation wastes most of the memory.

PagedAttention (Kwon et al, 2023, the foundation of vLLM) treats the KV cache like virtual memory: split into fixed-size *blocks* (typically 16 tokens each); allocate blocks dynamically as sequences grow; maintain a *block table* per sequence mapping logical positions to physical blocks; free blocks when sequences finish.

The result: memory utilization goes from ~30% with naive allocation to ~95%+ with paged. On a server with N GB of HBM, you can serve 3× more concurrent requests with the same hardware.

This lesson covers the algorithm and the engineering it enables (cross-request sharing in Lesson 19; the SGLang RadixAttention variant in Lesson 44).

The lesson is reading. The Hands-on builds a simple paged KV cache from scratch.

## The fragmentation problem

In a server serving many LLM requests concurrently, requests have variable lengths:
- Some prompts are short (50 tokens); some are long (3000 tokens).
- Some responses are short (20 tokens); some are long (500 tokens).
- Mid-flight requests cancel; new requests arrive.

Naive allocation: reserve `max_context` for every request. If `max_context = 4096` and you serve 10 concurrent requests, you allocate `10 × 4096 × per_token_cost` of KV memory — even if those 10 requests are at average length 200. ~95% of the allocation sits unused.

Paged allocation: reserve only the blocks each request actually uses. A 200-token request uses 13 blocks of 16; a 3000-token request uses 188 blocks. The total allocation matches the actual demand.

## The block table

Each sequence has a *block table*: a small array mapping logical positions to physical block indices.

```
sequence_id 0, length 200:
  block_table = [3, 7, 12, 41, 92, ...]  # 13 entries (200 tokens / 16 = 12.5, round up)
```

Logical position `i` corresponds to physical block `block_table[i // 16]`, with offset `i % 16` within that block.

When the sequence grows beyond its current allocation, the runtime:
1. Allocates a new free block from the pool.
2. Appends its index to the block table.
3. Writes the new K, V into the block.

When a sequence finishes:
1. Free all its blocks back to the pool.
2. Discard the block table.

The pool is a single allocator across all concurrent requests. Sequences come and go; blocks circulate. The result is a dense, dynamic, high-utilization KV store.

## The attention kernel modification

Standard attention iterates over tokens in a contiguous range. Paged attention iterates with one level of indirection: for each logical position, look up the physical block via the block table.

```python
for each query position q_pos:
    for each logical key position k_pos:
        block_idx = block_table[k_pos // BLOCK_SIZE]
        offset = k_pos % BLOCK_SIZE
        k = K_pool[block_idx, :, offset, :]
        v = V_pool[block_idx, :, offset, :]
        score = q @ k.T / sqrt(d)
        accumulate output
```

The indirection adds modest overhead — maybe 5-10% slower than contiguous attention on the same shapes — in exchange for the memory savings.

vLLM's actual implementation uses a custom CUDA kernel that fuses the block-table lookup with the FlashAttention-style computation. Other paged-attention implementations exist (TensorRT-LLM, SGLang's RadixAttention).

## Block size choice

Why 16 tokens per block? It's a tradeoff:

- **Smaller blocks** (e.g., 4 tokens): finer granularity, less waste at the end of each sequence, but more block-table lookups per attention step and more allocation overhead.
- **Larger blocks** (e.g., 64 tokens): cheaper indirection (fewer lookups), but more waste at the tail of sequences shorter than the block size, and less flexibility for cross-request sharing.

The 16-token sweet spot was empirically established. Some configurations use 8 or 32; 16 is the most common.

## Cross-request sharing (preview)

A subtle but huge benefit of paged attention: multiple sequences can point to the *same* blocks for shared content.

Example: a chat application where every request starts with the same system prompt. The first request processes the system prompt, fills blocks 0-3 of its allocation. The second request, when it sees the same system prompt, reuses blocks 0-3 — no recomputation, no extra memory.

This is *prefix caching*, covered in detail in Lesson 19. Paged attention makes it natural to implement; without paged allocation, sharing would require per-request copying.

## The reference-counting wrinkle

When multiple sequences share blocks, freeing a sequence's blocks requires care. The runtime maintains *reference counts*:

```
block_idx → refcount
```

A block is reusable only when its refcount hits 0. Adding a sequence that shares a block increments the refcount; finishing the sequence decrements. Free pool returns blocks at refcount 0.

This is the same machinery as OS virtual memory; the analogy that gave the technique its name.

## Copy-on-write for divergence

When two sequences share a block and one writes to it (because new tokens are generated), the shared block can't be mutated. Standard fix: copy-on-write.

When sequence A wants to write to a shared block:
1. Allocate a new block.
2. Copy the shared block's contents.
3. Update A's block table to point to the new block.
4. Decrement the old block's refcount.
5. Write A's new K, V into the new block.

The runtime detects the conflict (refcount > 1) and handles the copy automatically.

For speculative decoding (Lessons 25-27), branching beam search, and tree-based decoding, copy-on-write is the mechanism that makes parallel exploration efficient.

## What you should believe after this lesson

Three sentences:

**1. PagedAttention treats the KV cache like virtual memory**: split into fixed-size blocks (typically 16 tokens), allocate dynamically per sequence via a block table, free when sequences finish. Memory utilization goes from ~30% (naive) to ~95% (paged).

**2. The attention kernel pays a small overhead** (5-10%) for the block-table indirection, in exchange for the memory savings that let you serve 3× more concurrent requests. The kernel is custom; vLLM ships one tuned via CUDA; SGLang and TensorRT-LLM have their own.

**3. The block-allocation infrastructure enables cross-request sharing** (Lesson 19) and tree-based decoding (Lesson 26), via the same reference-counting + copy-on-write mechanics that OS virtual memory uses.

## Hands-on (at home)

A minimal paged KV cache.

```python
# paged_kv.py
import torch

BLOCK_SIZE = 16

class PagedKVCache:
    def __init__(self, n_blocks, n_kv_heads, d_head, n_layers, dtype=torch.float16, device='cuda'):
        # The block pool: one big tensor.
        self.K_pool = torch.zeros(n_layers, n_blocks, n_kv_heads, BLOCK_SIZE, d_head,
                                   dtype=dtype, device=device)
        self.V_pool = torch.zeros_like(self.K_pool)
        self.free_blocks = list(range(n_blocks))
        # Per-sequence: block table.
        self.block_tables = {}  # sequence_id → list of block indices
        self.refcounts = [0] * n_blocks

    def alloc_block(self):
        if not self.free_blocks:
            raise RuntimeError("Out of blocks!")
        b = self.free_blocks.pop()
        self.refcounts[b] = 1
        return b

    def add_sequence(self, seq_id):
        self.block_tables[seq_id] = []

    def append(self, seq_id, layer, k, v):
        # k, v: [n_kv_heads, d_head]
        bt = self.block_tables[seq_id]
        if not bt or bt[-1] is None:
            new_b = self.alloc_block()
            bt.append(new_b)
            position_in_block = 0
        else:
            # Check if last block is full.
            position_in_block = (sum(BLOCK_SIZE for _ in bt[:-1]) + position_in_last_block(bt, BLOCK_SIZE))
            # Simplified: just track count.
            ...
        # Write k, v into the block.
        # (This is where the indexing gets fiddly; the production code uses a per-seq cursor.)
        pass

    def read(self, seq_id, layer):
        bt = self.block_tables[seq_id]
        # Gather K and V from all blocks.
        K = self.K_pool[layer, bt]   # [n_blocks_used, n_kv_heads, BLOCK_SIZE, d_head]
        V = self.V_pool[layer, bt]
        # Reshape to [n_kv_heads, n_blocks * BLOCK_SIZE, d_head] for attention.
        return K, V

    def free_sequence(self, seq_id):
        for b in self.block_tables[seq_id]:
            self.refcounts[b] -= 1
            if self.refcounts[b] == 0:
                self.free_blocks.append(b)
        del self.block_tables[seq_id]

# This is a sketch; production implementations are several thousand lines because
# the kernel side (paged attention CUDA kernel) is complex.
print("PagedAttention sketch defined; see vLLM source for the full implementation.")
```

For the real thing, read vLLM's `attention/backends/paged_attn.py` and the CUDA kernel in `csrc/attention/`.

## Further reading

- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al, 2023) — the foundational vLLM paper.
- vLLM GitHub source — the production implementation.
- "FlashAttention with PagedAttention" — recent work combining the two for better memory and compute efficiency.
- "Operating Systems: Three Easy Pieces," chapter on Virtual Memory — for the OS-side analog if you haven't seen it.

Next lesson: **Prefix caching & cross-request KV reuse.** Paged attention enables this; let's go deep on how to share KV across requests with the same prefix, when it pays off, and the operational implications.
