---
title: "Lesson 2 — Model Registry and Versioning"
date: "2026-06-04"
module: "ml-platforms"
order: 2
tags: ["model-registry", "versioning", "lineage", "promotion", "stages"]
author: "Sudipta Pathak"
prerequisites: ["01-experiment-tracking"]
---

# Lesson 2 — Model Registry and Versioning

## Why this lesson exists

Experiment tracking captures every training run. But not every training run becomes a deployed model; some are exploratory, some are dead ends, some are checkpoints in a longer process. The *model registry* is the layer above tracking that identifies which trained models are deployable, manages their versions, tracks lineage (which run produced this model? which dataset trained it?), and supports promotion (dev → staging → prod).

This lesson covers the model registry concepts, the common tools (MLflow Model Registry, W&B Models, custom registries), and the operational patterns.

The lesson is reading. The Hands-on registers a model with MLflow.

## What a model registry stores

For each registered model:

- **Name**: a stable identifier ("text-classifier", "llama-3-instruct").
- **Versions**: numbered releases (1, 2, 3, ...). Each version is a frozen artifact.
- **Stages**: current state of each version (None, Staging, Production, Archived).
- **Lineage**: link back to the training run that produced this version (and through that, the data and code).
- **Metadata**: human-attached description, tags, evaluation metrics, contact info.

## Versioning conventions

A model version is a frozen artifact:
- Weights file.
- Config file (architecture, tokenizer, preprocessing).
- Sometimes: serving code, requirements, etc.

The versioning conventions vary:

**Sequential integers** (MLflow default): v1, v2, v3, ... Simple; no semantic meaning to the numbers.

**Semantic versioning** (some teams): MAJOR.MINOR.PATCH (e.g., 1.0.0, 1.1.0, 2.0.0). MAJOR for architecture changes; MINOR for retraining; PATCH for hot-fixes. Adapted from software versioning.

**Date-based** (some teams): YYYY-MM-DD.N. Easy to read; shows when the model was produced.

Pick one and stick with it. The convention is less important than consistency.

## Stages and promotion

The standard stage flow:

1. **None**: just registered; not yet vetted.
2. **Staging**: passed initial eval; deployed to staging environment for further testing.
3. **Production**: deployed to production traffic.
4. **Archived**: no longer in use.

Each transition is a "promotion": vet the model against criteria, then change its stage label.

The actual deployment usually points at "the Production version of model X." When you promote v3 from Staging to Production:
- The label moves from "v2 is Production" to "v3 is Production."
- The serving system, which reads the registry, picks up v3 on its next refresh.
- v2 might move to Archived or stay as a rollback target.

For canary deploys: the promotion can be gradual (10% traffic to v3, 90% to v2; ramp up). The registry supports this; the serving layer enforces it.

## Lineage

Lineage tracks: this model came from this training run, which used this dataset version, with this code commit.

Full lineage:
- **Model version** → links to:
- **Training run** → which logged:
  - Code git SHA.
  - Dataset version (or hash).
  - Hyperparameters.
  - Environment (Python packages, framework versions).
- Each of those is independently traceable.

Lineage matters for:
- **Reproducibility**: rebuild this exact model.
- **Debugging**: production model misbehaves; trace back to what produced it.
- **Compliance**: regulated industries need full provenance (Lesson 13).

## Approval workflows

For production-critical models, promotions go through review:
- An automated check (passes evals).
- A human review (model card, performance numbers).
- A formal sign-off.

Tools support this via:
- **Pull-request-style approvals**: MLflow Model Registry has webhooks; can integrate with GitHub PRs.
- **Manual approval gates** in CI/CD pipelines.
- **Required reviewers** before promotion.

The level of process depends on the stakes: a hobby project doesn't need this; a healthcare ML system does.

## Tools

The common options:

**MLflow Model Registry**: built into MLflow. Free, open-source, self-hosted. Standard for teams already using MLflow.

**W&B Models** (formerly Artifacts): W&B's registry layer. Integrated with their tracking.

**HuggingFace Hub**: more share-focused; the registry for open-source models. Many companies use HF Hub as their internal model registry too.

**Vertex AI Model Registry / SageMaker Model Registry / Azure ML Model Registry**: cloud-native options integrated with cloud-specific serving.

**Custom**: many large orgs have built their own. A model registry isn't fundamentally hard; you can build a minimal one in a few hundred lines.

The choice depends on what you're already using and the desired integration depth.

## The "model" as an interface

A model registry treats the model as an interface — inputs, outputs, metadata. Internally, the model can be:
- PyTorch state dict.
- TensorFlow SavedModel.
- ONNX file.
- GGUF for llama.cpp.
- Core ML for iOS.

The registry stores the file(s) and the format; consumers (serving systems) handle the appropriate loading.

For an org with multiple deployment targets (server inference + iOS app + Android), one registered model version might have multiple file formats: the same logical model, exported to PyTorch + ONNX + Core ML + GGUF. The registry tracks all of them.

## What you should believe after this lesson

Three sentences:

**1. A model registry sits above experiment tracking** — it identifies which trained models are deployable, manages their versions, tracks lineage, and supports stage-based promotion (None → Staging → Production → Archived).

**2. Versioning conventions vary** (sequential, semantic, date-based); consistency matters more than the specific scheme. Lineage tracks which run + dataset + code produced each version, enabling reproducibility and debugging.

**3. The standard tools** (MLflow Model Registry, W&B Models, HuggingFace Hub, cloud-native options) implement similar patterns. The choice depends on existing infrastructure; many large orgs build custom registries because the surface area is small.

## Hands-on (at home)

Register a model with MLflow.

```python
# model_registry_demo.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Train a model.
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

# Register it.
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier")
    print(f"Run ID: {run.info.run_id}")

# Promote to Staging.
client = mlflow.tracking.MlflowClient()
mv = client.get_latest_versions("iris-classifier", stages=["None"])[0]
client.transition_model_version_stage(
    name="iris-classifier",
    version=mv.version,
    stage="Staging",
)
print(f"Version {mv.version} now in Staging")

# List all versions.
for v in client.search_model_versions("name='iris-classifier'"):
    print(f"  v{v.version}: stage={v.current_stage}, run={v.run_id}")
```

In the MLflow UI (`mlflow ui`), you'll see the model registered with its versions and stage labels.

For real use, the promotion would be gated by a CI step (run eval; promote if it passes).

## Further reading

- MLflow Model Registry documentation.
- W&B Model Registry documentation.
- HuggingFace Hub documentation.
- "MLOps: Continuous Delivery and Automation Pipelines in Machine Learning" (Google whitepaper).

Next lesson: **Hyperparameter sweeps at scale.** Ray Tune, Optuna, the algorithms (grid, random, Bayesian, Hyperband) and how to scale them across a cluster.
