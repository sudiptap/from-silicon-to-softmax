---
title: "Lesson 8 — Workflow Orchestration"
date: "2026-06-04"
module: "ml-platforms"
order: 8
tags: ["workflow", "argo", "flyte", "kubeflow-pipelines", "dag"]
author: "Sudipta Pathak"
prerequisites: ["07-cost-monitoring"]
---

# Lesson 8 — Workflow Orchestration

## Why this lesson exists

An ML training run isn't usually a single command. The real workflow is: prepare data → train → evaluate → if good, deploy → run regression tests → if pass, route traffic. Each step is its own job; the steps have dependencies; some are conditional.

Workflow orchestrators (Argo Workflows, Flyte, Kubeflow Pipelines, Airflow) manage these chains: define the DAG, submit, the orchestrator runs each step in order, retries failures, surfaces results.

This lesson covers the major workflow orchestrators and the patterns they implement.

The lesson is reading. The Hands-on builds a small Argo Workflow.

## The shape of an ML workflow

A typical training-and-deploy workflow:

```
[ Prepare Data ]
       │
       ▼
[ Train Model ]
       │
       ▼
[ Evaluate Model ]
       │
       ▼
[ Decision: pass eval? ]
    │           │
    │ yes       │ no
    ▼           ▼
[ Register ]  [ Notify failure ]
       │
       ▼
[ Deploy to Staging ]
       │
       ▼
[ Run Integration Tests ]
       │
       ▼
[ Promote to Prod ]
```

This is a DAG with conditional branches. The orchestrator runs it; if any step fails, retries (configurable times); if all succeeds, the model lands in production.

## Argo Workflows

Argo Workflows is a Kubernetes-native workflow engine. Each workflow step is a pod; the orchestrator (Argo) handles the DAG.

Example (simplified):

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: train-and-deploy
spec:
  entrypoint: main
  templates:
  - name: main
    dag:
      tasks:
      - name: prepare-data
        template: prepare
      - name: train
        template: train
        dependencies: [prepare-data]
      - name: evaluate
        template: evaluate
        dependencies: [train]
      - name: deploy
        template: deploy
        dependencies: [evaluate]
        when: "{{tasks.evaluate.outputs.parameters.accuracy}} > 0.9"
  
  - name: prepare
    container: {image: my-pipeline:latest, command: [python, prepare.py]}
  - name: train
    container: {image: my-pipeline:latest, command: [python, train.py]}
  - name: evaluate
    container: {image: my-pipeline:latest, command: [python, eval.py]}
    outputs:
      parameters:
      - name: accuracy
        valueFrom: {path: /tmp/accuracy.txt}
  - name: deploy
    container: {image: my-pipeline:latest, command: [python, deploy.py]}
```

Argo runs each template as a pod. The `when:` clause conditionally executes `deploy` only if accuracy > 0.9.

Argo handles retries, parallel execution, parameter passing, and a UI showing the workflow status.

## Flyte

Flyte (Lyft) is a more ML-focused workflow engine. The user-facing API is Python (not YAML):

```python
from flytekit import task, workflow

@task
def prepare() -> str:
    # Prepare data, return its path.
    return "/data/prepared"

@task
def train(data_path: str) -> str:
    # Train model, return its path.
    return "/models/v1"

@task
def evaluate(model_path: str) -> float:
    # Evaluate model, return accuracy.
    return 0.92

@task
def deploy(model_path: str) -> None:
    # Deploy.
    pass

@workflow
def train_and_deploy():
    data = prepare()
    model = train(data_path=data)
    acc = evaluate(model_path=model)
    deploy(model_path=model)
```

The Python decorators register tasks with Flyte; the workflow runs on a cluster. Flyte handles caching (skipping unchanged steps), retries, and the UI.

Flyte's strength: typed Python interfaces, type-safe parameter passing, strong caching.

## Kubeflow Pipelines

Kubeflow Pipelines is the original ML-focused workflow on K8s. The Python SDK compiles workflows to Argo (older versions) or to a native runtime.

Has gone through significant evolution; in 2026 the v2 API is most common.

The user-facing pattern is similar to Flyte: Python decorators for steps; the SDK compiles to YAML; the orchestrator runs.

## Airflow

Airflow predates the Kubernetes era. Strong in data engineering; less ML-specific.

For ML, Airflow can orchestrate workflows but feels heavy and not GPU-aware natively. Many orgs that use Airflow for data engineering also use it for ML; many start fresh with Argo or Flyte.

## The choice

A rough decision matrix:

- **Argo Workflows**: K8s-native, language-agnostic (YAML); good for mixed-language pipelines and tight K8s integration.
- **Flyte**: Python-first, ML-oriented, strong typing, good caching; preferred for Python-heavy ML.
- **Kubeflow Pipelines**: Python-first, similar to Flyte; deep integration with other Kubeflow components.
- **Airflow**: traditional data engineering; works for ML but feels old-fashioned.

For a new ML platform: Flyte or Argo. For data engineering integration: Airflow. For Kubeflow ecosystem: Kubeflow Pipelines.

## What workflow orchestration gets you

The wins:

**Reproducibility**: the workflow is code; rerun anytime.

**Resilience**: failed steps retry; failed workflows can resume from the failed step.

**Visibility**: the UI shows current state, history, failures.

**Composition**: workflows can call other workflows; pipelines compose.

**Caching**: unchanged steps are skipped; the cache key is inputs + code version.

For ML specifically, caching is huge: re-running a training workflow doesn't re-prepare data if the data hasn't changed.

## Anti-patterns

**Monolithic workflows**: one workflow doing everything (data + train + eval + deploy + monitor). When something fails, the whole workflow is suspect. Break into smaller, composable workflows.

**No type safety**: parameter passing via untyped strings makes refactoring dangerous. Use typed APIs (Flyte, KFP v2).

**Workflow as a programming language**: complex logic inside the workflow definition (loops, conditionals, calculations). Hard to debug. Move logic into tasks; keep the workflow simple.

**Hardcoded paths**: workflows hardcoded to specific GCS / S3 paths. Make them parameterized; allows reuse across environments.

## What you should believe after this lesson

Three sentences:

**1. Workflow orchestrators (Argo, Flyte, Kubeflow Pipelines, Airflow) chain ML pipeline steps** — data prep, train, eval, deploy — into DAGs with dependencies, conditional branches, retries, and UI visibility.

**2. For ML in 2026**: Flyte and Argo are the most common new-platform choices. Flyte is Python-first and ML-oriented; Argo is K8s-native and language-agnostic. Kubeflow Pipelines remains relevant for Kubeflow ecosystems.

**3. The wins are reproducibility, resilience, visibility, composition, and caching.** Anti-patterns: monolithic workflows, untyped parameters, complex logic-in-workflow, hardcoded paths.

## Hands-on (at home)

Build a small Argo Workflow.

```bash
# Install Argo (on a K8s cluster).
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/quick-start-postgres.yaml

# Submit a workflow.
cat > hello-workflow.yaml <<'EOF'
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: hello-
spec:
  entrypoint: main
  templates:
  - name: main
    dag:
      tasks:
      - name: step-one
        template: print-hello
      - name: step-two
        template: print-hello
        dependencies: [step-one]
  - name: print-hello
    container:
      image: busybox
      command: ["echo", "hello from a workflow step"]
EOF
kubectl apply -f hello-workflow.yaml -n argo

# Watch.
argo list -n argo
argo get @latest -n argo
```

The workflow runs two steps in sequence; the UI (port-forward to argo-server) shows the DAG.

For real ML, replace the busybox containers with your training/eval images; pass model paths via outputs.

## Further reading

- Argo Workflows documentation.
- Flyte documentation.
- Kubeflow Pipelines documentation.
- Airflow documentation (for the data engineering side).

Next lesson: **CI/CD for models.** Eval gates, automatic rollback, canary deploys; how to make a model deploy as boring as a code deploy.
