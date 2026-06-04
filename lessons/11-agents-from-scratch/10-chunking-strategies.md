---
title: "Lesson 10 — Chunking Strategies"
date: "2026-06-04"
module: "agents"
order: 10
tags: ["chunking", "rag", "fixed", "semantic", "recursive", "document-aware"]
author: "Sudipta Pathak"
prerequisites: ["09-rag-core-pipeline"]
---

# Lesson 10 — Chunking Strategies

## Why this lesson exists

Chunking is the single most consequential preprocessing decision in a RAG pipeline. The chunking strategy determines what gets retrieved together; bad chunking means relevant info is split across chunks (lost in retrieval) or grouped with irrelevant text (lost in noise).

The chunking landscape: fixed-size, recursive, semantic, document-aware. Each has tradeoffs.

This lesson covers each, when to use which, and the practical patterns.

The lesson is reading. The Hands-on compares chunking strategies on a real document.

## Fixed-size chunking

Split documents into chunks of N tokens (e.g., 500), with overlap (e.g., 100).

```python
def fixed_chunk(text, chunk_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks
```

Pros: simple; predictable; works on any text.
Cons: cuts across natural boundaries (sentences, paragraphs); produces incoherent chunks.

The fixed-size approach is the baseline; everything else tries to improve on it.

## Recursive chunking

Split by hierarchy of separators: first by `\n\n` (paragraphs), then by `\n` (lines), then by `. ` (sentences), then by spaces. Each level is fallback if the previous produces chunks above the size limit.

```python
def recursive_chunk(text, chunk_size=500, separators=["\n\n", "\n", ". ", " "]):
    if len(text) <= chunk_size:
        return [text]
    
    for sep in separators:
        parts = text.split(sep)
        if all(len(p) <= chunk_size for p in parts):
            # All small enough; reassemble into chunks of <= chunk_size.
            chunks = []
            current = ""
            for p in parts:
                if len(current) + len(p) + len(sep) <= chunk_size:
                    current = current + sep + p if current else p
                else:
                    if current: chunks.append(current)
                    current = p
            if current: chunks.append(current)
            return chunks
    
    # All separators failed; fall back to fixed.
    return fixed_chunk(text, chunk_size)
```

Pros: respects natural boundaries when possible; falls back gracefully.
Cons: still some chunks may cut awkwardly.

This is LangChain's `RecursiveCharacterTextSplitter`. Production default for many RAG systems.

## Semantic chunking

Use embeddings to find topic boundaries. Compute embeddings for sliding window of sentences; chunk at points where the embedding similarity drops sharply.

```python
def semantic_chunk(text, embedder, threshold=0.7):
    sentences = text.split('. ')
    embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks
```

Pros: chunks are semantically coherent; topic boundaries are respected.
Cons: chunk sizes are variable; relies on embedding quality; computes embeddings during chunking.

Semantic chunking became popular in 2023+. Works well; the variable chunk size sometimes complicates downstream.

## Document-aware chunking

For structured documents (Markdown, HTML, PDFs with sections), respect the structure:
- Chunk at section boundaries.
- Keep headers with their content.
- For tables, keep them whole.
- For code blocks, keep them whole.

```python
def markdown_chunk(text, max_chunk_size=500):
    # Split at headers; each section is a chunk (further chunked if too large).
    sections = re.split(r'^(#{1,6} .+)$', text, flags=re.MULTILINE)
    # ... merge headers with their content; respect max_chunk_size.
```

Pros: best chunks for structured content; preserves the document's intended hierarchy.
Cons: more code per document type; for unstructured text, no improvement over recursive.

Tools like `Unstructured` provide document-aware loaders that combine extraction + chunking.

## Late chunking

A 2024 technique: embed the *whole document* first (with a long-context embedder), then chunk the resulting embeddings.

The idea: the embedder sees the full document context for each token; the resulting chunk embeddings carry "what does this chunk mean in the context of the document," not just "what does this chunk say in isolation."

Pros: chunks reflect document context.
Cons: requires long-context embedding model; not all embedders support it.

For documents where context-dependency matters (legal, medical), late chunking shows improvements.

## Chunk size and overlap

The dimensions:
- **Chunk size**: 200-1000 tokens typical. Smaller for QA (precise retrieval); larger for context-heavy tasks.
- **Overlap**: 10-20% of chunk size. Helps when answer spans a boundary.

Tune for your retrieval task:
- High-precision queries (specific facts): smaller chunks.
- Reasoning queries (need context): larger chunks.
- Mixed: medium (500 tokens) with some overlap.

## Multi-resolution chunking

Embed the same content at multiple chunk sizes; retrieve from both.
- Small chunks (100 tokens) for precise matching.
- Large chunks (1000 tokens) for context.

At retrieval, get top-k small chunks plus their large-chunk parents. Pass both to the LLM.

This is the "parent-child" pattern. Used in production for accuracy-critical RAG.

## Metadata

Each chunk carries metadata:
- Source filename.
- Page / section.
- Author / date.
- Custom tags.

Metadata helps filtering at retrieval: "search only chunks from this document type" or "search only chunks from the last 6 months."

## Anti-patterns

**Tiny chunks** (~50 tokens): retrieve fragments; LLM lacks context to answer.

**Huge chunks** (~5000 tokens): retrieval is imprecise; many chunks are mostly irrelevant.

**Ignoring document structure**: a PDF table chunked by 500-token blocks becomes garbled.

**No overlap**: an answer sentence split between two chunks is harder to retrieve.

**Re-chunking on every update**: if you only added one new document, don't re-chunk the entire corpus.

## What you should believe after this lesson

Three sentences:

**1. Chunking strategy is the most consequential RAG preprocessing decision.** Bad chunking → relevant info lost or buried. The strategies: fixed, recursive, semantic, document-aware, late chunking. Each has tradeoffs.

**2. Recursive chunking (split by paragraph → line → sentence)** is the safe default — respects natural boundaries, falls back gracefully. Semantic and document-aware are better for specific document types; late chunking shines for context-heavy content.

**3. Multi-resolution chunking (parent-child pattern)** combines precise retrieval (small chunks) with rich context (large chunks). Used in accuracy-critical RAG; production pattern for high-stakes deployments.

## Hands-on (at home)

Compare chunking strategies on a real document.

```python
# chunking_comparison.py
text = open("article.txt").read()  # any reasonable-length text

# Strategy 1: fixed.
fixed = fixed_chunk(text, chunk_size=500, overlap=100)

# Strategy 2: recursive (using LangChain).
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
recursive = splitter.split_text(text)

# Strategy 3: semantic (compute embeddings; chunk at low-similarity boundaries).
# (See semantic_chunk above.)

# Compare.
print(f"Fixed: {len(fixed)} chunks; sizes: {[len(c) for c in fixed[:5]]}")
print(f"Recursive: {len(recursive)} chunks; sizes: {[len(c) for c in recursive[:5]]}")
# print(f"Semantic: ...")

# Inspect a fixed-chunk boundary vs recursive boundary.
print(f"\nFixed chunk 1 ending: ...{fixed[1][-100:]}")
print(f"Recursive chunk 1 ending: ...{recursive[1][-100:]}")
```

You'll likely see recursive chunks ending at sentence boundaries (cleaner); fixed chunks cutting mid-sentence.

For a deeper experiment, build the same RAG pipeline with each chunking strategy; query a set of test questions; see which strategy retrieves the best chunks.

## Further reading

- LangChain text splitters documentation.
- "Late Chunking" (2024 papers).
- "Optimal Chunking for RAG" — various blog posts.

Next lesson: **Hybrid search and reranking.** BM25 + dense + cross-encoder reranker; when each component helps; the production pattern that beats pure dense retrieval.
