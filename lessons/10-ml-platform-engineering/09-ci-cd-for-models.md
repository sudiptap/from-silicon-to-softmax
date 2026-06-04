---
title: "Lesson 9 — CI/CD for Models"
date: "2026-06-04"
module: "ml-platforms"
order: 9
tags: ["ci-cd", "eval-gates", "canary", "rollback", "deployment"]
author: "Sudipta Pathak"
prerequisites: ["08-workflow-orchestration"]
---

# Lesson 9 — CI/CD for Models

## Why this lesson exists

Software CI/CD is mature: every commit triggers tests; passing tests trigger deploys; failing deploys auto-rollback. Model CI/CD is younger and trickier — the artifact (a model) is non-deterministic; the tests (evaluations) are slow and expensive; the failure modes are silent (a model can produce bad outputs without erroring).

This lesson covers what model CI/CD looks like in 2026: eval gates, canary deployments, automatic rollback, the patterns that have stabilized.

The lesson is reading. The Hands-on builds a simple eval-gated deploy.

## The pieces

Model CI/CD has analogues of software CI/CD:

- **CI**: every commit to the training repo triggers a small eval on a recent model. Catches regressions in training code.
- **Eval gates**: a model can't promote to production unless it passes pre-defined evaluations.
- **Canary deploys**: route 1%-10% of production traffic to the new model; monitor; expand if good.
- **Rollback**: if metrics regress, automatically revert to the previous model.
- **Observability**: production metrics tracked per model version; regressions caught.

The flow:

```
Commit → Training (workflow from Lesson 8)
              │
              ▼
       Eval on held-out set
              │
              ▼
       Pass eval gate? ── No → fail; alert; no deploy
              │ Yes
              ▼
       Promote in registry → Deploy to staging
              │
              ▼
       Integration tests
              │
              ▼
       Canary deploy (5% traffic)
              │
              ▼
       Monitor for 1 hour
              │
              ▼
       Metrics OK? ── No → rollback
              │ Yes
              ▼
       Full deploy (100% traffic)
```

## Eval gates

The eval gate: a set of automated tests that a new model must pass before deployment.

Examples:
- **Accuracy on held-out set**: > 0.90 (or whatever threshold).
- **No regression on critical eval slices**: e.g., performance on a specific subgroup must not drop.
- **Output safety checks**: no harmful outputs on a curated red-team set.
- **Latency under threshold**: model serves at <100 ms p99.
- **Output diversity**: doesn't collapse to a single response.

For LLMs specifically, the eval gate might include:
- MMLU score > X.
- HumanEval score > Y.
- A small custom benchmark for the org's specific use case.
- Helpfulness/harmlessness checks via a judge model.

The bar: better than the current production model, or above an absolute threshold. Whichever applies.

## Canary deploys

After passing the eval gate, the model deploys to a small fraction of traffic:
- 5% of users see the new model; 95% see the old.
- Production metrics tracked per cohort.
- If the new model is at least as good (or better), expand to 25%, 50%, 100%.
- If worse, rollback.

This catches issues that offline evaluation missed — distribution shift, real-user feedback, integration bugs.

Tools:
- **Argo Rollouts**: K8s-native progressive delivery.
- **Flagger**: similar.
- **Custom routing layer**: many ML platforms have their own (the serving layer routes traffic to model A vs model B based on user ID or random hash).

## Automatic rollback

If metrics regress during canary:
- Stop the rollout.
- Revert to the previous version.
- Alert.

The automation requires:
- Clear regression criteria (which metric, what threshold, over what window).
- Reliable production metrics (Lesson 12).
- Fast rollback (the previous version's artifacts must still be available).

Frequent rollback is bad — indicates eval gates are too loose. Rare rollback is good but the mechanism still needs to work.

## Model CI

CI for models is trickier than CI for code:
- Training takes hours / days; you can't run it on every commit.
- Eval is slow; you can't run the full benchmark on every PR.

Compromises:
- **Quick smoke evals on every PR**: a 5-minute evaluation on a subset; catches obvious regressions.
- **Full eval nightly or weekly**: comprehensive but infrequent.
- **Targeted evals on relevant PRs**: a PR that touches the tokenizer triggers tokenizer-specific evals.

For training-code changes specifically, "does this training script still produce a sensible model after 100 steps" is a useful smoke test. A passable model in 100 steps is faster than waiting for convergence; catches "I broke the gradient" bugs.

## The CD challenge: model artifacts are large

A 70B model is ~140 GB in BF16. Deploying involves:
- Pull the model from the registry.
- Load it into the serving infrastructure.
- Warm up (often a few minutes for the model to JIT-compile or initialize).
- Switch traffic.

This is slower than software CD. A typical model deploy is 5-30 minutes vs seconds for code.

Optimizations:
- **Pre-warm**: have a "ready" replica with the new model loaded before switching traffic.
- **Side-by-side**: keep the old model running while warming the new one; switch when ready.
- **Local-cached weights**: keep model weights on local NVMe per inference node; only redownload when version changes.

For very large models (405B+), the deploy is multi-hour. Special infrastructure needed.

## What you should believe after this lesson

Three sentences:

**1. Model CI/CD has analogues of software CI/CD** — eval gates, canary deploys, automatic rollback — but with ML-specific challenges (non-deterministic artifacts, slow tests, silent failures). The standard pattern: workflow runs training → eval gate → canary → full deploy with rollback on regression.

**2. Eval gates encode "this model is good enough"**: held-out accuracy, critical-slice performance, safety checks, latency. The threshold is "better than current production" or "above absolute bar."

**3. Canary deploys + automatic rollback are the production safety net** — catch what offline eval missed. Tools (Argo Rollouts, Flagger) implement progressive delivery on K8s; custom routing layers do the same for orgs with mature ML platforms.

## Hands-on (at home)

Build a simple eval-gated deploy with a Bash script.

```bash
#!/bin/bash
# eval_gated_deploy.sh

set -e

NEW_MODEL_VERSION=$1
EVAL_THRESHOLD=0.90

# Run eval (placeholder; in practice this runs a full eval suite).
ACCURACY=$(python -c "
import random
print(random.uniform(0.85, 0.95))
")
echo "New model accuracy: $ACCURACY"

# Compare to threshold.
if (( $(echo "$ACCURACY > $EVAL_THRESHOLD" | bc -l) )); then
    echo "Passed eval gate; promoting to production"
    # In real life: update the model registry; trigger deploy.
    mlflow models promote --name my-model --version $NEW_MODEL_VERSION --stage Production
else
    echo "Failed eval gate (got $ACCURACY, threshold $EVAL_THRESHOLD); not deploying"
    exit 1
fi
```

For a real production setup, the eval is a full benchmark suite; the deploy involves canary rollouts; the rollback is automatic via production-metric monitoring.

## Further reading

- Argo Rollouts documentation.
- Flagger documentation.
- "Continuous Delivery for Machine Learning" (Martin Fowler's blog, by Sato et al).
- "ML Test Score" (Google) — checklist of practices for ML system reliability.

Next lesson: **Feature stores.** Feast, Tecton; the train/serve skew problem and what solves it.
