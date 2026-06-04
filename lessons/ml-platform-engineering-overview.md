---
title: "ML Platform Engineering: Module Overview"
date: "2026-05-26"
excerpt: "Make ML reproducible, observable, deployable. Experiment tracking, model registry, training observability, workflow orchestration, CI/CD for models, cost attribution — the infrastructure that turns one good training run into a reliable model factory."
module: "ml-platforms"
order: 0
tags: ["ml-platform", "mlops", "experiment-tracking", "observability", "ci-cd", "ray", "mlflow", "wandb", "overview"]
author: "Sudipta Pathak"
prerequisites: ["cluster-orchestration"]
---

# ML Platform Engineering

## Why this module exists

A trained model is a starting point, not a finish line.

The work of getting from "one good training run on a notebook" to "a production system that ships better models every quarter" is platform engineering: experiment tracking, model registry, eval gates, observability, cost attribution, CI/CD for models, and the data infrastructure that feeds it all.

This is where ML stops being research and starts being software engineering at scale. It's also where most teams accumulate the highest leverage and the worst tech debt — because nobody on the team owns it explicitly, so it grows by accretion until nobody understands the whole pipeline.

## How this fits

This is the sixth module of the [depth track](/curriculum) — *From Silicon to Softmax*. It sits on top of [Cluster Orchestration](/curriculum/cluster-orchestration-overview) (which gets jobs running) and connects to the [Inference](/curriculum/inference-from-scratch-overview) and [Agents](/curriculum/agents-from-scratch-overview) breadth tracks (which consume what this module produces).

If Cluster Orchestration is "how do I get a job running reliably," ML Platform Engineering is "how do I get the *next 1000* jobs running reliably, learn from them, and ship the winners."

## The roadmap

14 tutorials. Topics get linked here as they ship.

### Tracking what you ran

1. **Experiment tracking** — W&B and MLflow architecture; what they store, how they scale, what fails
2. **Model registry & versioning** — semantic versioning for models, lineage, promotion workflows
3. **Hyperparameter sweeps at scale** — Ray Tune, Optuna, distributed Bayesian optimization

### Feeding the beast

4. **Data infrastructure for ML** — Ray Data, Spark for ML, streaming pipelines, the dataset-as-a-product mindset
5. **Checkpoint storage at scale** — S3 multipart, multi-node sync, fast restore; the file-system performance no one budgets for

### Watching the run

6. **Training observability** — Prometheus + Grafana for GPU metrics, DCGM, NaN canaries, anomaly detection
7. **Cost monitoring** — per-job and per-experiment cost attribution; the question "how much did this paper cost us" should have a one-click answer

### Shipping the run

8. **Workflow orchestration** — Argo Workflows, Flyte, Kubeflow Pipelines; chaining train → eval → deploy
9. **CI/CD for models** — eval gates, automatic rollback, canary deploys; how to make a model deploy as boring as a code deploy
10. **Feature stores** — Feast, Tecton; the train/serve skew problem and what solves it
11. **Model serving infrastructure** — bridges into the [Inference series](/curriculum/inference-from-scratch-overview); the platform's view of serving

### Living with the run

12. **A/B testing infrastructure** — traffic splitting, statistical rigor, attribution; how to actually know a new model is better
13. **Compliance & governance** — model cards, lineage tracking, audit trails; the boring stuff that matters in regulated industries
14. **Incident response for model regressions** — what does "the model is down" mean, how do you debug it, who pages whom

---

## What I'm filling in over time

This is the topic scaffold. Each tutorial is a build — pick a real open-source platform component, build a minimum version of it from scratch, then compare to the production tool. Same pattern as the rest of the site: you don't understand a system until you've built a small version of it yourself.
