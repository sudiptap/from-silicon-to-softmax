---
title: "Lesson 11 — Hybrid Search and Reranking"
date: "2026-06-04"
module: "agents"
order: 11
tags: ["hybrid-search", "bm25", "dense", "reranking", "cross-encoder"]
author: "Sudipta Pathak"
prerequisites: ["10-chunking-strategies"]
---

# Lesson 11 — Hybrid Search and Reranking

## Why this lesson exists

Pure dense (vector) retrieval has weaknesses: it can miss exact keyword matches (specific names, codes, technical terms). Pure keyword search (BM25) misses semantic similarity ("the same idea in different words").

The 2024+ production pattern: **hybrid search** combines both. **Reranking** with a cross-encoder further improves quality by re-scoring the top candidates with a more expensive but more accurate model.

This lesson covers BM25, hybrid fusion, cross-encoder reranking, and when each helps.

The lesson is reading. The Hands-on builds a hybrid + rerank pipeline.

## BM25: keyword retrieval

BM25 (Best Match 25) is a classic IR ranking function. Given a query and a corpus:
1. For each document, compute a BM25 score based on term frequencies.
2. Rank by score.

The score rewards documents that have rare-in-the-corpus query terms many times.

```python
from rank_bm25 import BM25Okapi
documents = ["The cat sat on the mat", "The dog ran in the park", "Cats and dogs"]
tokenized = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized)

query = "where did the cat sit"
scores = bm25.get_scores(query.split())
print(scores)  # higher score for documents matching query terms
```

BM25 is fast (sparse vector operations), handles exact matches well, and is interpretable.

Its weakness: misses synonymy. A query for "felines" won't match documents that only say "cats" (no shared tokens).

## Dense retrieval: semantic

Dense (vector) retrieval (Lesson 9) handles synonymy: "felines" embeds close to "cats."

Its weakness: misses rare exact matches. A query for "version 2.4.1" might not retrieve documents that have this exact string but no broader semantic context.

## Hybrid search

The fix: run both BM25 and dense; fuse the results.

**Reciprocal Rank Fusion (RRF)**: a simple, effective fusion algorithm.

```python
def rrf_fusion(rankings, k=60):
    """rankings is a list of dicts: {doc_id: rank} for each retriever."""
    scores = {}
    for ranking in rankings:
        for doc_id, rank in ranking.items():
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

Each retriever returns ranked results. RRF combines: each document's RRF score is the sum of `1/(k+rank)` across retrievers (k=60 is standard).

A document that ranks high in both gets a high RRF score; a document that's high in one but not the other gets a medium score.

Production hybrid:
1. Run BM25; get top 50.
2. Run dense; get top 50.
3. RRF-fuse; take top 20.

This top 20 is sent to the reranker.

## Cross-encoder reranking

After hybrid retrieval gives top 20 candidates, a cross-encoder re-scores them.

**Bi-encoder** (what we've used so far): encodes query and document independently; compares vectors. Fast (vectors precomputed) but the comparison is independent.

**Cross-encoder**: encodes (query, document) together, attending to both; produces a single score. Much more accurate but slower (must run for each (query, document) pair).

The pattern:
1. Retrieve top 20 with bi-encoder (fast).
2. Rerank with cross-encoder (slower but only 20 pairs).
3. Take top 3-5 as final context.

Popular cross-encoder models:
- **BGE reranker** (BAAI): open-source.
- **Cohere Rerank API**: hosted.
- **Voyage Rerank API**: hosted.

```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

candidates = [
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "London is the capital of England.",
]
query = "What is the capital of France?"

# Score each (query, candidate) pair.
scores = cross_encoder.predict([(query, c) for c in candidates])
# Higher scores = more relevant.
ranked = sorted(zip(scores, candidates), reverse=True)
print(ranked)
```

The cross-encoder typically improves NDCG@10 by 5-15 points over bi-encoder alone — substantial for production RAG.

## When hybrid helps

Hybrid (BM25 + dense) helps when:
- Queries mix semantic and keyword matching (typical real queries).
- The corpus has technical terms, codes, names that benefit from exact matching.
- The corpus has paraphrases / synonyms that benefit from semantic matching.

Dense-only suffices when:
- Queries are well-formed natural-language questions.
- The corpus is uniformly written.
- Cost / latency matter and you can tolerate some misses.

For most production: use hybrid. The marginal cost is low; the accuracy gain is meaningful.

## When reranking helps

Reranking helps when:
- Retrieval returns many plausible candidates.
- The top-k from retrieval is noisy.
- You can afford the latency (cross-encoder adds 100-500ms for 20 candidates).

For chatbots: usually worth it.
For real-time agents (every ms counts): trade-off; skip reranking or use a smaller reranker.

## Multi-query expansion

Another hybrid: generate multiple query variations; retrieve for each; combine.

**HyDE (Hypothetical Document Embeddings)**: use the LLM to generate a hypothetical answer; embed that; retrieve. The hypothetical answer is closer in embedding space to the actual relevant documents than the question is.

**Multi-query**: generate 3-5 query variations; retrieve for each; combine. Covers different phrasings.

**Step-back prompting**: generate a more general version of the question; retrieve for that; helps with specific questions about general topics.

These help when single-query retrieval misses relevant chunks.

## The full pipeline

Production RAG (2026):

```
Query
  │
  ▼
[ Query understanding ]  → expand to multiple queries (HyDE, multi-query)
  │
  ▼
[ BM25 retrieval ] + [ Dense retrieval ]  (parallel)
  │
  ▼
[ RRF fusion ]  → top 20
  │
  ▼
[ Cross-encoder rerank ]  → top 5
  │
  ▼
[ LLM generation ]
  │
  ▼
Answer
```

Each stage is a 5-15% improvement; together they're a significant gain over the naive single-retriever pipeline from Lesson 9.

## What you should believe after this lesson

Three sentences:

**1. Pure dense retrieval misses exact-match queries; pure BM25 misses semantic.** Hybrid (BM25 + dense + RRF fusion) handles both; standard 2024+ production pattern.

**2. Cross-encoder reranking re-scores the top-N hybrid candidates** with a model that attends to (query, candidate) jointly. ~5-15 NDCG@10 improvement; latency cost of 100-500ms for 20 candidates.

**3. The full production RAG pipeline**: query expansion → hybrid retrieval → rerank → generate. Each stage adds meaningful gain over the naive single-retriever baseline; production deployments increasingly use all of them.

## Hands-on (at home)

Build a hybrid + rerank pipeline.

```python
# hybrid_rerank.py
# pip install rank-bm25 sentence-transformers faiss-cpu
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

documents = [
    "The capital of France is Paris.",
    "Paris is famous for the Eiffel Tower.",
    "Python was created in 1991 by Guido van Rossum.",
    "The Eiffel Tower is 330 meters tall.",
    "France has 67 million inhabitants.",
]

# BM25.
tokenized = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized)

# Dense.
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Cross-encoder.
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def hybrid_search(query, k=3):
    # BM25.
    bm25_scores = bm25.get_scores(query.split())
    bm25_top = np.argsort(bm25_scores)[::-1][:5]
    
    # Dense.
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    _, dense_top = index.search(q_emb, 5)
    dense_top = dense_top[0]
    
    # RRF fusion.
    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_top):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank)
    for rank, doc_id in enumerate(dense_top):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank)
    
    candidates = sorted(rrf_scores.items(), key=lambda x: -x[1])[:10]
    candidate_docs = [documents[doc_id] for doc_id, _ in candidates]
    
    # Rerank.
    scores = ce.predict([(query, d) for d in candidate_docs])
    reranked = sorted(zip(scores, candidate_docs), reverse=True)[:k]
    return reranked

results = hybrid_search("What is the height of the Eiffel Tower?")
for score, doc in results:
    print(f"{score:.3f}  {doc}")
```

You should see the most relevant document at the top (the one explicitly about the Eiffel Tower's height).

## Further reading

- "BM25: A Probabilistic Retrieval Model" — the foundational IR work.
- "Reciprocal Rank Fusion" (Cormack et al, 2009).
- "ColBERT" (Khattab & Zaharia, 2020) — late interaction; better than bi-encoder, faster than cross-encoder.
- "Beyond Semantic Search: How to Make Search Engines Truly Intelligent" — practical tutorials.

Next lesson: **Agentic RAG.** The agent decides what to retrieve, when, and how. Self-RAG, Corrective RAG, the iterative-retrieval patterns.
