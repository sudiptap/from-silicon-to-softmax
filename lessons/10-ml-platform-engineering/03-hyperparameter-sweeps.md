---
title: "Lesson 3 — Hyperparameter Sweeps at Scale"
date: "2026-06-04"
module: "ml-platforms"
order: 3
tags: ["hyperparameter", "sweep", "ray-tune", "optuna", "bayesian", "hyperband"]
author: "Sudipta Pathak"
prerequisites: ["02-model-registry"]
---

# Lesson 3 — Hyperparameter Sweeps at Scale

## Why this lesson exists

Most ML projects involve tuning hyperparameters: learning rate, batch size, model size, regularization. A single training run gives one data point; the question "what's the best learning rate" requires many runs comparing them.

For small projects, you tune by hand. For larger ones, hyperparameter *sweeps* — automated multi-run searches — are the answer. The major frameworks (Ray Tune, Optuna, Weights & Biases sweeps) all implement the standard algorithms (grid search, random search, Bayesian optimization, Hyperband, ASHA).

This lesson covers the algorithms, the frameworks, and how to scale sweeps across a cluster.

The lesson is reading. The Hands-on runs an Optuna sweep.

## The algorithms

**Grid search**: try every combination from a discrete grid. Simple; exhaustive within the grid. Combinatorial explosion (5 LRs × 5 batches × 5 dropouts = 125 runs). Good for low-dimensional spaces and final-stage fine-tuning.

**Random search**: sample randomly from the hyperparameter ranges. Surprisingly effective — Bergstra & Bengio showed random often beats grid for similar budget because many hyperparameters don't matter much; random covers the important ones.

**Bayesian optimization** (TPE, GP-based): use the results of past trials to inform the next trial. Build a surrogate model of "hyperparameters → score"; sample where the model predicts high score with high uncertainty. More sample-efficient than random.

**Hyperband / ASHA**: early-stop bad trials. Allocate small budget to many trials; keep the best; allocate more budget to those; repeat. Hyperband and ASHA (Asynchronous Successive Halving) are the standard early-stopping algorithms.

**Population-based training (PBT)**: maintain a population of models; periodically replace bad ones with mutations of good ones. Used for very long-running optimization (RL hyperparameters). Less common for standard ML tuning.

For most workloads, the sweet spot is **Bayesian + ASHA**: use Bayesian sampling for what to try next; ASHA to early-stop poorly-performing trials.

## The frameworks

**Optuna**: Python-native; lightweight; widely used. Implements TPE, CMA-ES, and others. Easy to integrate into existing scripts.

**Ray Tune**: integrated with Ray's distributed runtime. Best for distributed multi-trial sweeps; supports many search algorithms and schedulers.

**W&B Sweeps**: SaaS-flavored sweeps integrated with W&B tracking. Each trial is a tracked run; the UI shows the sweep progress.

**Vertex AI Vizier / SageMaker Automatic Model Tuning**: cloud-managed equivalents.

The choice:
- Local / small-scale: Optuna.
- Distributed: Ray Tune.
- W&B users: W&B Sweeps.

## A sweep example

A typical sweep workflow with Optuna:

```python
import optuna

def objective(trial):
    # Sample hyperparameters.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Train model with these hyperparameters.
    model = train(lr=lr, batch_size=batch_size, dropout=dropout)
    
    # Return the metric to maximize / minimize.
    return evaluate(model)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
print(f"Best trial: {study.best_trial.params}")
```

`suggest_float`, `suggest_categorical`, etc. define the search space. `TPESampler` is the Bayesian sampler. After 100 trials, you get the best hyperparameters.

For distributed Optuna (multiple machines), use a shared backend storage (SQLite, MySQL, Postgres, RedshiftDB). Workers pull trial parameters from the storage; report results back; the sampler stays consistent across workers.

## Distributed sweep architecture

For 100+ trials × hours per trial = days of compute, distribute across cluster:

1. **Centralized parameter server**: stores the sweep state (which trials done, which parameters tried, which results received). Optuna's SQLite/Postgres backend; Ray Tune's centralized scheduler.

2. **Workers**: each worker queries the parameter server, gets the next trial's parameters, runs the trial, reports the result.

3. **Scheduler logic**: the parameter server's sampler picks parameters based on past results (Bayesian) or with early-stopping (ASHA).

For Ray Tune, the integration with Ray Cluster is automatic — `tune.run()` distributes trials across the Ray workers.

For Optuna, you launch N worker processes (each on a different GPU or node), each running `study.optimize(...)` against the shared backend.

## Early stopping (ASHA)

ASHA's algorithm:
- Run all trials with a small budget (e.g., 5 epochs).
- Keep the top fraction (e.g., top 1/3); discard the rest.
- Run the survivors with more budget (e.g., 15 epochs).
- Repeat until budget exhausted.

Compared to "run all trials to completion," ASHA spends much less compute on bad trials. For typical ML workloads, the budget savings is 3-5×.

The risk: a trial that looks bad early might become good later (slow learners). ASHA loses these. The tradeoff is usually worth it; the savings outweigh the rare missed-discovery.

## Anti-patterns

**Searching too many dimensions at once**. With 10 hyperparameters, you need O(10^N) trials to cover the space. Search a few at a time; fix the rest at known-good values.

**Not using early stopping**. Letting bad trials run to completion wastes compute. Use ASHA or similar.

**Not logging the sweep**. Each trial should be tracked individually (Lesson 1) so you can dig into specific trials post-hoc.

**Optimizing the wrong metric**. The sweep's objective should be what you actually care about; not training loss (you care about generalization, not training loss) but eval-set metric.

**Re-running the same sweep**. If hyperparameters are stable across model versions, store the best ones from the last sweep; only re-run when something changes (new dataset, new architecture).

## What you should believe after this lesson

Three sentences:

**1. The sweep algorithms are**: grid (low-dim final-stage), random (surprisingly good baseline), Bayesian (sample-efficient for moderate dimensions), Hyperband/ASHA (early-stops bad trials, 3-5× compute savings). Combine Bayesian + ASHA for typical ML workloads.

**2. The frameworks (Optuna, Ray Tune, W&B Sweeps)** implement these algorithms with different scaling characteristics: Optuna is lightweight Python-native; Ray Tune handles distributed clusters; W&B Sweeps integrates with W&B tracking.

**3. For distributed sweeps, the workers query a centralized parameter server**, get next trial's params, run, report results. The scheduler logic (Bayesian sampling, ASHA early-stopping) lives in the parameter server; workers are stateless.

## Hands-on (at home)

Run an Optuna sweep on a small classification task.

```python
# optuna_sweep.py
# pip install optuna scikit-learn
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=0
    )
    return cross_val_score(model, X, y, cv=3).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualize.
optuna.visualization.matplotlib.plot_optimization_history(study)
```

The output shows the best hyperparameters found in 50 trials. For larger models, the same pattern; the trial is just slower.

For Ray Tune distributed, the pattern is similar but with `tune.run(...)` and a Ray cluster.

## Further reading

- Optuna documentation.
- Ray Tune documentation.
- "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (Li et al, 2018).
- "Algorithms for Hyper-Parameter Optimization" (Bergstra et al, 2011) — TPE paper.

Next lesson: **Data infrastructure for ML.** Ray Data, Spark for ML, streaming pipelines, the dataset-as-a-product mindset.
