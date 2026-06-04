---
title: "Module 10 — ML Platform Engineering"
date: "2026-06-04"
module: "ml-platforms"
order: 0
tags: ["ml-platform", "mlops", "experiment-tracking", "observability", "ci-cd", "ray", "mlflow", "wandb", "overview"]
author: "Sudipta Pathak"
prerequisites: ["cluster-orchestration"]
---

# ML Platform Engineering

## Why this module exists

A trained model is a starting point, not a finish line.

The work of getting from "one good training run on a notebook" to "a production system that ships better models every quarter" is *platform engineering*: experiment tracking, model registry, eval gates, observability, cost attribution, CI/CD for models, and the data infrastructure that feeds it all.

This is where ML stops being research and starts being software engineering at scale. It's also where most teams accumulate the highest leverage and the worst tech debt — because nobody on the team owns it explicitly, so it grows by accretion until nobody understands the whole pipeline.

## How this fits

Module 10 of the depth track. Module 9 (Cluster Orchestration) gets jobs running reliably; Module 10 makes the next 1000 jobs run reliably, lets you learn from them, and ship the winners. Module 11 (Agents from Scratch) consumes what this module produces — deployed models with reliable serving.

The output of this module: the ability to design and operate an ML platform — the set of tools, conventions, and infrastructure that supports the team's day-to-day ML work without each engineer reinventing it.

## The roadmap

Fourteen lessons.

### Tracking what you ran

1. **Experiment tracking** — W&B and MLflow architecture; what they store, how they scale, what fails.
2. **Model registry & versioning** — semantic versioning for models, lineage, promotion workflows.
3. **Hyperparameter sweeps at scale** — Ray Tune, Optuna, distributed Bayesian optimization.

### Feeding the beast

4. **Data infrastructure for ML** — Ray Data, Spark for ML, streaming pipelines, the dataset-as-a-product mindset.
5. **Checkpoint storage at scale** — S3 multipart, multi-node sync, fast restore; the filesystem performance no one budgets for.

### Watching the run

6. **Training observability** — Prometheus + Grafana for GPU metrics, DCGM, NaN canaries, anomaly detection.
7. **Cost monitoring** — per-job and per-experiment cost attribution; "how much did this paper cost us" should have a one-click answer.

### Shipping the run

8. **Workflow orchestration** — Argo Workflows, Flyte, Kubeflow Pipelines; chaining train → eval → deploy.
9. **CI/CD for models** — eval gates, automatic rollback, canary deploys; how to make a model deploy as boring as a code deploy.
10. **Feature stores** — Feast, Tecton; the train/serve skew problem and what solves it.
11. **Model serving infrastructure** — bridges into Module 7 (Inference); the platform's view of serving.

### Living with the run

12. **A/B testing infrastructure** — traffic splitting, statistical rigor, attribution; how to actually know a new model is better.
13. **Compliance & governance** — model cards, lineage tracking, audit trails; the boring stuff that matters in regulated industries.
14. **Incident response + module wrap** — what does "the model is down" mean, how do you debug it, who pages whom. Plus the module wrap.

---

## What this module deliberately won't cover

- **Specific vendor product details** in depth (the W&B vs MLflow vs Neptune vs Comet comparison). We cover the underlying patterns; the vendor choice is workload- and budget-specific.
- **Model training algorithms** — those are Modules 3-8.
- **Inference systems** in depth — that's Module 7.
- **General software engineering practices** beyond ML-specific concerns.
- **Specific cloud provider services** (SageMaker, Vertex AI, Azure ML) in detail. These offer integrated platforms; the patterns are the same as DIY but in a managed package.

## How to work through it

Every lesson is fully readable as prose. The hands-on sections vary in setup requirements:

- Lessons 1-3, 7-8: most exercises run locally with open-source tools (MLflow, W&B free tier, Optuna).
- Lessons 4-5: need access to a cluster and storage for the meaningful parts.
- Lessons 9-12: production-feel; hands-on is sketches or local simulations.

The mental models are framework-agnostic. The specifics of MLflow vs W&B vs Vertex AI matter less than understanding the patterns each implements.

A note on tempo: this module is platform-engineering-heavy. Each lesson has a "what problem is this for" component, a "what's the standard solution" component, and an "anti-pattern" component. The anti-patterns are often the most useful — they're the mistakes you make once and remember forever.

The capstone (Lesson 14): an incident-response scenario plus the module wrap with a worked platform design.
