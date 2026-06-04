---
title: "Lesson 9 — RAG: The Core Pipeline from Scratch"
date: "2026-06-04"
module: "agents"
order: 9
tags: ["rag", "retrieval", "embeddings", "pipeline", "generation"]
author: "Sudipta Pathak"
prerequisites: ["08-memory-compaction"]
---

# Lesson 9 — RAG: The Core Pipeline from Scratch

## Why this lesson exists

RAG (Retrieval-Augmented Generation) is the most common production agent pattern. The idea: the LLM doesn't know everything in its training data; for current / specific / proprietary information, retrieve relevant context from a knowledge base; pass it into the prompt; generate the answer.

This lesson builds the core RAG pipeline from scratch — load, chunk, embed, retrieve, generate — in under 100 lines. The next three lessons go deeper on chunking, retrieval strategies, and agentic RAG.

The lesson is reading. The Hands-on builds the full pipeline.

## The pipeline

```
Documents
   │
   ▼
[ Load ]      → read files (PDF, HTML, text)
   │
   ▼
[ Chunk ]     → split into ~500-token pieces
   │
   ▼
[ Embed ]     → produce vector per chunk
   │
   ▼
[ Index ]     → store vectors in a DB (FAISS, pgvector, Pinecone)

QUERY TIME:

User question
   │
   ▼
[ Embed query ]
   │
   ▼
[ Retrieve ]  → top-k similar chunks from index
   │
   ▼
[ Generate ]  → LLM call with question + retrieved chunks
   │
   ▼
Answer
```

The user question's embedding is compared to chunk embeddings; the top-k most-similar chunks are inserted into the LLM's prompt; the LLM generates an answer using those chunks.

## Why RAG works

Two reasons:

**Capacity**: the LLM's training data is fixed; new information (post-training-cutoff news, proprietary docs) isn't in it. RAG provides external memory.

**Specificity**: the LLM has world knowledge but not your company's specific docs. RAG retrieves your-specific context.

The bet: it's cheaper / faster to retrieve relevant chunks at query time than to fine-tune the model on your corpus. For most cases, this is true.

## Loading documents

Documents come in many formats:
- Text files (`.txt`, `.md`).
- HTML pages.
- PDFs.
- Word docs.
- Slides.
- Spreadsheets.

Each needs a loader. The Python ecosystem has many:
- **Unstructured**: handles many formats; quality varies.
- **PyPDF2 / pdfplumber**: for PDFs.
- **BeautifulSoup**: for HTML.
- **LangChain document loaders**: convenience wrappers around many.

The output: a list of `Document` objects, each with `text` and `metadata` (source filename, page number, etc.).

## Chunking

Long documents don't fit in the LLM's context; even if they did, you don't want to retrieve a 100-page document for a single-sentence question. Chunk into pieces.

Common chunk sizes: 200-1000 tokens. The tradeoff:
- Smaller chunks: more precise retrieval (find the exact relevant passage), but may miss context.
- Larger chunks: more context per chunk, but less precise retrieval.

Common chunk *overlap*: 10-20% of chunk size. Adjacent chunks share some text. Helps when the answer spans a chunk boundary.

Lesson 10 covers chunking strategies in detail.

## Embedding

Convert each chunk to a vector (typically 256-1024 dimensions). The embedding model has been trained so that semantically similar text produces similar vectors.

Popular embedding models in 2026:
- **OpenAI text-embedding-3-large** (3072-dim).
- **OpenAI text-embedding-3-small** (1536-dim).
- **Cohere embed-v3** (1024-dim).
- **Voyage embeddings**.
- **sentence-transformers/all-MiniLM-L6-v2** (384-dim, open-source, fast).
- **BGE embeddings** (open-source, strong).

For most production deployments, OpenAI or Cohere embeddings give the best out-of-box quality. Open-source models are competitive and self-hostable.

## Indexing

Store the chunk vectors in a vector index. Options:

**FAISS** (Facebook): in-memory; fast; popular for prototypes.

**Chroma**: lightweight; embedded or hosted.

**Pinecone**: managed service.

**Weaviate**: hybrid (vector + keyword); managed or self-hosted.

**pgvector**: PostgreSQL extension; useful when you already have Postgres.

**Qdrant**: Rust-based; high performance.

For prototyping: FAISS or Chroma. For production: managed (Pinecone) or pgvector.

The index supports nearest-neighbor search: given a query vector, return the top-k closest chunk vectors.

## Retrieval

At query time:
1. Embed the user's question.
2. Query the index for the top-k most-similar chunks.
3. Return them.

The choice of k: typically 3-10. More chunks = more context; harder for the LLM to focus.

The similarity metric: cosine similarity (most common) or Euclidean distance. Both work; cosine is dominant.

## Generation

Insert the retrieved chunks into the prompt and call the LLM:

```python
prompt = f"""Answer the question based on the provided context.

Context:
{retrieved_chunks_joined}

Question: {user_question}

Answer:"""

response = llm.complete(prompt)
```

The prompt structure varies. Some patterns:
- **Stuffed**: all chunks concatenated; LLM picks.
- **Refined**: feed chunks one at a time; LLM refines the answer.
- **Map-reduce**: LLM answers from each chunk; another LLM consolidates.

For typical Q&A, stuffed works.

## A complete pipeline in <100 lines

```python
# rag_pipeline.py
# pip install sentence-transformers faiss-cpu openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# ----- Build the index -----
documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "Python was created by Guido van Rossum in 1991.",
    "The Eiffel Tower is 330 meters tall.",
    "JavaScript was created by Brendan Eich in 1995.",
]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents)
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)  # inner product (= cosine if normalized)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# ----- Query time -----
def rag_query(question, k=2):
    q_emb = embedder.encode([question])
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    retrieved = [documents[i] for i in indices[0]]
    
    context = "\n".join(f"- {chunk}" for chunk in retrieved)
    prompt = f"""Answer based on the context.

Context:
{context}

Question: {question}
Answer:"""
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, retrieved

answer, retrieved = rag_query("Who created Python?")
print(f"Retrieved: {retrieved}")
print(f"Answer: {answer}")
```

The full pipeline. With 5 documents this is overkill; with 100K it's the production pattern.

## What this doesn't do

The naive pipeline:
- Doesn't handle ambiguous queries well (Lesson 12 covers query transformation).
- Doesn't combine BM25 + dense (Lesson 11 covers hybrid search).
- Doesn't rerank (Lesson 11).
- Doesn't update the index when documents change.
- Doesn't handle multi-hop queries (need multiple retrievals).

The next lessons fill in these gaps.

## What you should believe after this lesson

Three sentences:

**1. The core RAG pipeline is load → chunk → embed → index, then at query time: embed → retrieve → generate.** Under 100 lines for a basic implementation; modern stacks (FAISS + sentence-transformers + OpenAI API) make this fast.

**2. RAG works because it provides external memory for current and proprietary content** that's not in the LLM's training data. Cheaper / faster than fine-tuning for most use cases.

**3. The naive pipeline misses many production concerns** — query understanding, hybrid search, reranking, freshness, multi-hop. The next three lessons cover these.

## Hands-on (at home)

Build the RAG pipeline above. Try with your own documents:

```bash
pip install sentence-transformers faiss-cpu openai
# Set OPENAI_API_KEY in your env.

# Replace `documents` with your own — could be paragraphs from a book, your notes, etc.
```

Query it; observe what's retrieved and the final answer. Note where it succeeds and where it fails (ambiguous queries, queries needing multiple chunks).

For a more realistic dataset, scrape a small Wikipedia subset (a few hundred articles); chunk; embed; query about specific facts.

## Further reading

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al, 2020) — the original RAG paper.
- LangChain RAG documentation.
- "Building RAG-based LLM Applications" (Anthropic, Hugging Face tutorials).

Next lesson: **Chunking strategies.** Fixed, semantic, recursive, document-aware. The chunking strategy dramatically affects retrieval quality.
