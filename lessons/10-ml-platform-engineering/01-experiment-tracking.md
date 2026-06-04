---
title: "Lesson 1 — Experiment Tracking"
date: "2026-06-04"
module: "ml-platforms"
order: 1
tags: ["experiment-tracking", "wandb", "mlflow", "comet", "neptune", "metadata"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — Experiment Tracking

## Why this lesson exists

Every ML training run produces: a model, metrics over time, the code that ran, the data it used, the hyperparameters, system info, and (often) artifacts like checkpoints and plots. Without a system to capture all of this, two months later you can't reproduce the run, can't compare it to others, and can't tell why the production model behaves the way it does.

Experiment tracking is the solution. The tools (W&B, MLflow, Comet, Neptune, Aim) all do roughly the same thing — capture run metadata into a queryable database — with different scaling characteristics, UIs, and integration patterns.

This lesson covers what experiment tracking is, the patterns the standard tools implement, and the choices to make when adopting it.

The lesson is reading. The Hands-on uses MLflow to log a small experiment.

## What gets tracked

A "run" in experiment-tracking parlance typically captures:

- **Hyperparameters**: learning rate, batch size, model architecture choices.
- **Metrics over time**: training loss, eval accuracy, GPU utilization — logged per step or per epoch.
- **System info**: GPU type, package versions, git commit, command line.
- **Artifacts**: model checkpoints, plots, sample outputs, dataset references.
- **Tags / notes**: human-attached metadata for organization.

The schema is loose by design; different teams emphasize different things. The contract: log enough to reproduce and to compare.

## The architecture

Most tracking tools have a similar architecture:

1. **Client library**: a Python (or other) library you call from your training script.
2. **Backend server**: receives logs from clients; stores them in a database; serves a query API.
3. **Artifact store**: blob storage for checkpoints, plots, large files.
4. **Web UI**: lets you browse, compare, filter runs.

The client → server protocol is HTTP (or gRPC). Logs are batched and sent asynchronously to avoid blocking the training.

For W&B: SaaS server; you point at api.wandb.ai (or self-host).
For MLflow: self-hosted typically; the server is open-source.

## The patterns

A typical training script with tracking:

```python
import mlflow

mlflow.set_experiment("my-experiment")
with mlflow.start_run(run_name="lr=1e-4-bs=64"):
    mlflow.log_params({"learning_rate": 1e-4, "batch_size": 64, "model": "llama-3.2-1b"})
    
    for step in range(10000):
        loss = train_step()
        if step % 100 == 0:
            mlflow.log_metric("loss", loss, step=step)
            mlflow.log_metric("lr", current_lr, step=step)
    
    # Save artifacts.
    torch.save(model.state_dict(), "model.pt")
    mlflow.log_artifact("model.pt")
```

The `log_params`, `log_metric`, `log_artifact` API is standard across tools. The integration is light — a few lines added to existing training code.

## What separates the tools

The tools differ on:

**Scaling**: W&B handles millions of runs and thousands of metrics per run with reasonable latency. MLflow's stock backend (SQLite/Postgres) struggles past ~100K runs. For frontier ML orgs that run thousands of experiments per day, the backend choice matters.

**Metric volume**: high-frequency logging (every step at 100Hz) generates millions of points per run. W&B and Comet handle this well; MLflow needs careful backend tuning.

**Artifact storage**: W&B and Comet manage artifact storage automatically (S3/GCS under the hood). MLflow lets you configure; you bring your own bucket.

**Collaboration**: W&B has good team features (sharing, comments). MLflow is more bare-bones.

**Privacy / on-prem**: MLflow is fully self-hosted by default. W&B has self-hosted options for regulated industries.

**Cost**: MLflow free (you operate the server). W&B free up to small team usage; paid for larger.

In 2026:
- Startups and research teams: W&B (free tier + easy onboarding).
- Larger teams with budget: W&B paid or Comet.
- Privacy-sensitive / cost-conscious: MLflow self-hosted.
- HuggingFace ecosystem: HF Hub is increasingly a tracking + sharing destination.

## What scaling looks like

A production ML org at scale typically has:
- 100-10,000 active experiments per week.
- Hundreds of TB of metric/artifact data total.
- Concurrent reads from researchers comparing runs, writes from active jobs.
- API integrations from CI/CD pipelines, dashboards, alerting.

The backend must handle this. Self-hosted MLflow at scale typically uses:
- Postgres for metadata.
- S3 (or equivalent) for artifacts.
- Multiple MLflow tracking-server replicas behind a load balancer.

W&B at scale: their SaaS handles it; the largest orgs pay for dedicated tenants.

## Anti-patterns

Common mistakes:

**Logging everything**: tempting but expensive. Per-step logging of 100 metrics × 1M steps = 100M points per run. Often kills the backend. Aggregate or sample.

**Not logging the code version**: a run without its git commit + diff is irreproducible. Always log `git rev-parse HEAD` and any diff vs that commit.

**Not logging the data version**: a model is a function of its training data; the data version is part of the run. Dataset hashes / references must be logged.

**Mixing experiment tracking with monitoring**: tracking is for *historical* runs; production monitoring (Prometheus, Grafana) is for *live* services. Different requirements, different tools. Don't try to use W&B as your production dashboard.

**Forgetting to clean up**: stale runs accumulate; the backend grows; queries slow down. Set retention policies.

## What you should believe after this lesson

Three sentences:

**1. Experiment tracking captures hyperparameters, metrics, system info, and artifacts** for every training run — the bare minimum for reproducibility and cross-run comparison. Standard tools (W&B, MLflow, Comet, Neptune) implement the same patterns with different scaling characteristics.

**2. The tool choice depends on scale, privacy, cost, and team workflow**: W&B for ease-of-use; MLflow for self-hosted control; HF Hub for sharing-focused teams. In 2026, W&B is the most common at mid-scale; MLflow at large self-hosted scale.

**3. Anti-patterns include over-logging (kills the backend), under-logging (kills reproducibility), and confusing tracking with production monitoring** (they have different requirements; use different tools). Always log code version and data version.

## Hands-on (at home)

Track an experiment with MLflow.

```python
# experiment_tracking_demo.py
# pip install mlflow scikit-learn
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("iris-classification")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_est in [10, 50, 100, 200]:
    with mlflow.start_run(run_name=f"rf-{n_est}-trees"):
        mlflow.log_param("n_estimators", n_est)
        model = RandomForestClassifier(n_estimators=n_est, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"n_estimators={n_est}, acc={acc:.4f}")
```

Then launch the MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

You'll see the four runs; can compare metrics, click into each to see params and artifacts.

## Further reading

- MLflow documentation (mlflow.org).
- Weights & Biases documentation (docs.wandb.ai).
- "Machine Learning Engineering" (Andriy Burkov) — chapter on tracking.
- "MLOps community" Slack and meetups for practical patterns.

Next lesson: **Model registry & versioning.** Beyond tracking individual runs: how to manage the trained-model artifacts, version them, and promote between dev → staging → prod.
