---
title: "Lesson 10 — Feature Stores"
date: "2026-06-04"
module: "ml-platforms"
order: 10
tags: ["feature-store", "feast", "tecton", "train-serve-skew", "online", "offline"]
author: "Sudipta Pathak"
prerequisites: ["09-ci-cd-for-models"]
---

# Lesson 10 — Feature Stores

## Why this lesson exists

Classic ML — recommendation systems, fraud detection, churn prediction — uses *features*: pre-computed signals about entities (users, items, transactions). The features have to be available at both training time (over a historical window) and at serving time (for the current request, in milliseconds).

The "train/serve skew" problem: features computed differently in training vs serving produce subtle bugs. The model trained on average-of-last-7-days behaves differently when serving sees average-of-last-1-day instead.

Feature stores solve this: a single system that computes features once and serves them consistently for both training and inference.

LLMs have made feature stores less central than they were 5 years ago (LLMs do their own feature extraction from raw inputs), but for classical ML and recommendation, feature stores remain essential.

This lesson covers the patterns.

The lesson is reading. The Hands-on sketches a feature store.

## What features are

A feature is a structured signal: "user X's average click-through rate over the last 7 days," "item Y's price percentile in its category," "is_this_user_new_this_week."

For each feature:
- **Definition**: how to compute it from raw data.
- **Type**: float, int, vector, string.
- **Entity**: what it's about (user, item, session).
- **Freshness requirement**: hourly, daily, real-time.

A typical feature store has 100s to 10000s of features.

## The two sides: offline and online

A feature store has two storage layers:

**Offline store**: historical feature values, for training. Backed by a data warehouse (BigQuery, Snowflake) or parquet on S3. Supports time-travel ("what was feature X for user Y at time T").

**Online store**: latest feature values, for serving. Backed by a key-value store (Redis, DynamoDB, Cassandra). Low-latency lookups (<10 ms).

The same feature is computed once but served from two stores. The feature store framework ensures both are consistent.

## Train/serve skew, illustrated

Suppose feature `user_avg_clicks_7d` is computed as:
- Training: a SQL query against the historical events table, computed offline.
- Serving: a real-time aggregation over Redis-cached recent events.

If the implementations drift (different time-window definitions, different aggregation logic, different handling of null), the model trained on offline features performs poorly when fed online features.

Feature store solution: define the feature once. The feature store's offline backfill and online materialization both use the same definition.

## Feast

Feast is the open-source feature store. The conceptual API:

```python
from feast import FeatureView, Field, Entity
from feast.types import Float32

user = Entity(name="user_id")

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="avg_clicks_7d", dtype=Float32),
        Field(name="total_purchases", dtype=Float32),
    ],
    source=BigQuerySource(
        query="SELECT ...",
        timestamp_field="event_timestamp",
    ),
)
```

Feast:
- Reads the SQL/Spark/etc. source to compute feature values.
- Materializes them: writes to the offline store (training) and the online store (serving).
- Provides a retrieval API for training (point-in-time joins) and serving (low-latency lookup).

## Tecton

Tecton is the commercial managed feature store (by former Uber Michelangelo engineers). Similar concepts; productized.

For orgs running multi-team ML, Tecton's enterprise features (governance, monitoring, multi-cloud) often justify the cost. Smaller orgs use Feast or roll their own.

## Real-time features

A subset of features need to be very fresh (sub-second). Examples:
- "Time since user's last action."
- "Number of failed logins in the past 5 minutes" (fraud detection).

These can't be precomputed in batch; they're computed on-the-fly as events arrive (via Kafka + Flink) and cached in the online store.

Feature stores integrate with stream processors for this: the feature is defined; the stream processor computes it; the online store holds the latest value.

## Why LLMs change this

For LLMs, features are mostly text (the prompt, conversation history). The model does the feature extraction internally. The classical "user_avg_clicks_7d"-style features are less central.

But hybrid systems (LLM + recommendation, LLM + traditional ML) still use feature stores for the non-LLM parts. RAG (retrieval-augmented generation) uses an embedding-vector "feature" (embeddings of documents); the feature-store pattern applies to those.

In 2026, vector databases (Pinecone, Weaviate, pgvector) have become the "feature store for embeddings" — analogous to feature stores for traditional features.

## Anti-patterns

**Duplicating feature logic** in training and serving code. The whole point of a feature store is to avoid this; if you duplicate, you'll have skew.

**Stale features**. The online store didn't update; serving uses old values. Monitor freshness.

**Over-engineering early**. A feature store has operational cost. For small ML projects (one model, a few features), it's overkill. Add when you have multiple models sharing features.

**Ignoring point-in-time correctness**. Training uses the historical value of a feature *as of the training-label timestamp*, not the current value. Get this wrong and you "leak" future information into training.

## What you should believe after this lesson

Three sentences:

**1. Feature stores solve the train/serve skew problem** by defining features once and serving them from both an offline store (for training, time-travel-capable) and online store (for serving, low-latency). The dominant tools: Feast (open-source), Tecton (managed).

**2. LLMs reduce the central role of classical feature stores** (the model does feature extraction internally), but hybrid systems and RAG (vector "features") still benefit from feature-store-style patterns.

**3. Real-time features** (sub-second freshness) integrate with stream processors (Kafka + Flink); the feature store's online layer holds the latest value with low-latency lookups.

## Hands-on (at home)

A toy feature store with Feast.

```python
# feast_demo.py
# pip install feast pandas
import pandas as pd
from datetime import datetime, timedelta

# Create a small Feast feature store programmatically.
from feast import FeatureStore, FeatureView, Field, Entity
from feast.types import Float32
from feast.value_type import ValueType
from feast.data_source import FileSource

# In practice, you'd structure this as a Feast repo on disk.
# This is a minimal demo.

# Define entity.
user = Entity(name="user_id", value_type=ValueType.INT64)

# Define source (parquet file).
df = pd.DataFrame({
    "user_id": [1, 2, 3, 1, 2],
    "avg_clicks_7d": [10.5, 5.2, 3.8, 12.0, 6.0],
    "event_timestamp": [
        datetime.now() - timedelta(days=2),
        datetime.now() - timedelta(days=2),
        datetime.now() - timedelta(days=2),
        datetime.now(),
        datetime.now(),
    ],
})
df.to_parquet("/tmp/user_features.parquet")

# In a real Feast setup, you'd `feast apply` and `feast materialize`.
# Here, just demonstrate the conceptual workflow.

print("Feast workflow:")
print("  1. Define entities and feature views in feature_repo/")
print("  2. `feast apply` registers them")
print("  3. `feast materialize-incremental NOW` updates online store")
print("  4. At training: `store.get_historical_features(entities, features)`")
print("  5. At serving: `store.get_online_features(entities, features)`")
```

For a real Feast setup, see the Feast documentation tutorial; the example repo walks through end-to-end.

## Further reading

- Feast documentation.
- Tecton documentation.
- "Feature Stores: A Tour" — surveys.
- Uber's Michelangelo blog posts (the inspiration for many feature stores).

Next lesson: **Model serving infrastructure.** Bridges into Module 7 (inference); the platform-side view of serving — fleet management, autoscaling, request routing.
