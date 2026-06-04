---
title: "Lesson 4 — Data Infrastructure for ML"
date: "2026-06-04"
module: "ml-platforms"
order: 4
tags: ["data-infrastructure", "ray-data", "spark", "streaming", "dataset-as-product"]
author: "Sudipta Pathak"
prerequisites: ["03-hyperparameter-sweeps"]
---

# Lesson 4 — Data Infrastructure for ML

## Why this lesson exists

ML data infrastructure is "boring" but consumes more engineering time than the modeling. Cleaning data, deduping, sharding, tokenizing, augmenting, sampling — these turn raw inputs into model-ready batches. For LLM-scale pretraining (10T+ tokens), the data pipeline itself is a major engineering effort; for vision and audio, less but still substantial.

This lesson covers the data infrastructure patterns: Ray Data for in-Python parallelism, Spark for ETL at scale, streaming pipelines for real-time data, and the "dataset as a product" mindset that makes data work tractable.

The lesson is reading. The Hands-on builds a small Ray Data pipeline.

## The pipeline stages

A typical ML data pipeline:

1. **Acquisition**: collect raw data (scrape, ingest from databases, receive from streams).
2. **Cleaning**: dedupe, filter spam, normalize formats, fix encoding.
3. **Annotation** (if needed): human or model-based labeling.
4. **Transformation**: tokenize, resize, compute features.
5. **Sharding**: split into many files for parallel reading.
6. **Serving to training**: data loader pulls from the sharded files at training time.

For LLM pretraining specifically, deduping and quality filtering dominate the engineering effort. Common Crawl alone is 100+ TB raw; the filtered + deduplicated version usable for training is 10-30 TB. The filtering is the value.

## Ray Data

Ray Data is a distributed dataset abstraction built on Ray:

```python
import ray
import ray.data

ds = ray.data.read_parquet("s3://my-bucket/data/*.parquet")
ds = ds.map_batches(preprocess_fn, batch_size=1000)
ds = ds.filter(quality_filter)
ds = ds.repartition(1024)
ds.write_parquet("s3://my-bucket/processed/")
```

The operations (map, filter, repartition, etc.) execute distributed across Ray workers. Lazy by default; the actual computation happens when you `.iter_*()` or write.

Ray Data is well-suited for ML preprocessing because:
- It integrates with Ray Train (Lesson 10 of Module 9 — KubeRay).
- It handles heterogeneous data (text + images + tabular).
- Streaming execution doesn't materialize the whole dataset in memory.
- Python-native; you don't write Scala/Java for ETL.

## Spark for ML

Apache Spark has long been the standard for ETL at scale. For ML, Spark is used for:
- Large-scale cleaning and joining (joining a 1B-row table with another).
- SQL-style aggregations on training data.
- Feature engineering on large datasets.

Spark's distributed-execution model (driver + executors; lazy-evaluated DAG) is well-suited for the ETL phase.

For ML-specific tasks (tokenization, image preprocessing), Spark's Python integration (PySpark) works but has overhead vs Ray Data. The split tends to be:
- Spark: structured data, joins, aggregations.
- Ray Data: ML-specific preprocessing, batch inference.

Many production stacks use both: Spark for the data warehouse layer; Ray Data for the ML-pipeline layer.

## Streaming pipelines

For real-time data (clickstreams, sensor data, financial transactions), the data flow is continuous, not batch.

Standard tools:
- **Kafka**: message bus; producers write events; consumers process them.
- **Flink**: stateful stream processing.
- **Spark Streaming / Structured Streaming**: micro-batch streaming.

For ML, streaming data feeds:
- Online feature stores (Lesson 10).
- Continuous model retraining.
- Real-time inference pipelines.

The patterns:
- **Lambda architecture**: a batch layer (Spark) processes historical data; a speed layer (Flink) processes recent. Combine for fresh features. Complex but flexible.
- **Kappa architecture**: only the streaming layer; everything goes through it. Simpler; requires the stream to be replayable.

Most production ML in 2026 uses streaming for fresh feature computation; batch for model training.

## Dataset as a product

A philosophical shift in recent years: treat datasets as products with versions, owners, SLAs, and consumers.

Practices:
- **Version every dataset**. v1, v2, v3 of a dataset; new versions don't break existing consumers without notice.
- **Document inputs and outputs**. A dataset card analogous to a model card.
- **Designate owners**. Someone is on-call for the dataset; if it goes stale or quality drops, they're paged.
- **Test data quality**. Statistical tests run when a new version is produced; alerts if metrics drift.

Tools:
- **Great Expectations**: data quality testing library.
- **Pachyderm / DVC**: data versioning systems.
- **Datafold / Monte Carlo**: data observability.

For LLM training in particular, dataset quality is a major lever; treating datasets as products is the only way to manage it at scale.

## Anti-patterns

**Reading raw data at training time**. Materialize the processed form; training reads from processed files. Don't re-tokenize 10T tokens every training run.

**Tiny files**. Object stores penalize small reads. Bundle into larger files (typically 100MB-1GB each).

**Single-machine preprocessing**. For 100GB+ datasets, a single-machine pandas script takes hours. Distribute via Ray Data or Spark.

**No deduplication**. LLM pretraining especially: duplicates inflate the effective dataset and hurt training.

**No quality monitoring**. Datasets drift silently; data sources change format; bugs in preprocessing flip subtly. Without monitoring, you don't know.

## What you should believe after this lesson

Three sentences:

**1. ML data infrastructure has multiple stages**: acquisition, cleaning, annotation, transformation, sharding, training-serving. For LLM-scale, the data pipeline is a major engineering effort; for smaller workloads, less but still substantial.

**2. The tool choice depends on the workload**: Spark for structured ETL at scale, Ray Data for ML-specific preprocessing and Python-native pipelines, streaming tools (Kafka + Flink) for real-time data feeds.

**3. The "dataset as a product" mindset** — versioning, owners, SLAs, quality testing — is the only sustainable way to manage ML data at scale. Tools (Great Expectations, DVC, observability platforms) support it; the discipline is what matters.

## Hands-on (at home)

Build a small Ray Data pipeline.

```python
# ray_data_demo.py
# pip install 'ray[data]' pyarrow
import ray
import ray.data

# Create a small dataset.
data = [{"text": f"sample text {i}", "label": i % 2} for i in range(1000)]
ds = ray.data.from_items(data)

# Map: tokenize-like operation.
def tokenize(batch):
    return {"text": batch["text"], "label": batch["label"],
            "length": [len(t) for t in batch["text"]]}

ds = ds.map_batches(tokenize)

# Filter: keep only long examples.
def keep_long(row):
    return row["length"] > 12
ds = ds.filter(keep_long)

# Aggregate / inspect.
print(f"Total rows after filter: {ds.count()}")
print(f"Sample row: {ds.take(1)}")

# Write to disk.
ds.write_parquet("/tmp/processed/")

# Read back to confirm.
ds2 = ray.data.read_parquet("/tmp/processed/")
print(f"Read back {ds2.count()} rows")
```

For larger datasets, the same pipeline runs distributed across a Ray cluster. The single-machine version is conceptual.

For a real LLM-data-prep pipeline, see HuggingFace's `datatrove` library or AllenAI's `dolma`.

## Further reading

- Ray Data documentation.
- "Spark: The Definitive Guide" (Chambers & Zaharia).
- "Designing Data-Intensive Applications" (Kleppmann).
- "Great Expectations" documentation.
- "Data-centric AI" — Andrew Ng's materials on dataset-as-product.

Next lesson: **Checkpoint storage at scale.** S3 multipart, multi-node sync, fast restore. The filesystem performance no one budgets for until it's the bottleneck.
