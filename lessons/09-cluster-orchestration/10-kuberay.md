---
title: "Lesson 10 — KubeRay: Ray on Kubernetes"
date: "2026-06-04"
module: "cluster-orchestration"
order: 10
tags: ["kuberay", "ray", "kubernetes", "distributed-python", "ray-train", "ray-serve"]
author: "Sudipta Pathak"
prerequisites: ["09-slurm-vs-kubernetes"]
---

# Lesson 10 — KubeRay: Ray on Kubernetes

## Why this lesson exists

Ray (UC Berkeley, now Anyscale) is a Python-native distributed computing framework. Unlike Slurm or Kubernetes (which orchestrate at the OS / container level), Ray orchestrates at the *Python function call* level: you decorate a function with `@ray.remote` and it runs on any worker in the cluster.

For ML, Ray provides:
- **Ray Train**: distributed training abstractions.
- **Ray Serve**: model serving with autoscaling.
- **Ray Tune**: hyperparameter tuning.
- **Ray Data**: distributed data processing.

KubeRay is the Ray-on-Kubernetes operator. It manages Ray clusters as K8s resources, integrating Ray with the K8s ecosystem you've learned.

This lesson covers KubeRay and why Ray sits awkwardly between "library you import" and "infrastructure you deploy."

The lesson is reading. The Hands-on launches a Ray cluster via KubeRay.

## What Ray is, conceptually

Ray's mental model: a cluster of Python workers, and a single "driver" Python process. The driver coordinates; workers execute. Computation is expressed as Python function calls that may run remotely.

```python
import ray

ray.init()  # connect to the cluster

@ray.remote
def expensive_function(x):
    return x ** 2

# Run 100 calls in parallel across workers.
results = ray.get([expensive_function.remote(x) for x in range(100)])
```

For ML:

```python
import ray.train
from ray.train.torch import TorchTrainer

def train_loop(config):
    # Your standard PyTorch training code.
    ...

trainer = TorchTrainer(
    train_loop_per_worker=train_loop,
    scaling_config=ray.train.ScalingConfig(num_workers=8, use_gpu=True),
)
result = trainer.fit()
```

Ray distributes the training across the 8 workers; you don't write the distributed-launching boilerplate (no `torchrun`, no manual env var setup).

## KubeRay's role

KubeRay is the K8s operator that runs Ray clusters as K8s resources. The CRDs:

- **RayCluster**: a running Ray cluster (long-lived).
- **RayJob**: a one-shot training job (creates a RayCluster, runs the driver, deletes when done).
- **RayService**: a Ray Serve deployment (autoscaling model serving).

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: my-training
spec:
  entrypoint: python train.py
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: head
            image: rayproject/ray:2.30.0
            resources:
              limits: {cpu: "4", memory: "8Gi"}
    workerGroupSpecs:
    - replicas: 4
      groupName: workers
      template:
        spec:
          containers:
          - name: worker
            image: rayproject/ray:2.30.0
            resources:
              limits: {nvidia.com/gpu: 8, memory: "256Gi"}
```

The operator creates the head and worker pods, sets up the Ray cluster, runs the driver, and cleans up.

## What Ray offers that PyTorchJob doesn't

PyTorchJob orchestrates *processes* (one per pod, one per GPU). Ray orchestrates *Python tasks* (any number of remote function calls).

Where Ray shines:

**Heterogeneous workloads**: a workflow that has data preprocessing (CPU), training (GPU), and evaluation (CPU+GPU mix) can be expressed as one Ray script. PyTorchJob would need separate jobs for each phase.

**Dynamic scaling**: Ray autoscales workers as needed. Adding workers mid-job works; PyTorchJob's world size is fixed.

**Hyperparameter tuning**: Ray Tune runs many training trials, dynamically scheduling them across the cluster. PyTorchJob would require submitting each trial as a separate job.

**Mixed Python + ML**: any Python distributed workflow benefits from Ray. PyTorchJob is PyTorch-specific.

**Serving with autoscaling**: Ray Serve handles autoscaling and request-routing for inference. PyTorchJob is training-only.

## What PyTorchJob offers that Ray doesn't

The flip side:

**Lower-level control**: PyTorchJob is close to the metal; Ray adds an abstraction layer.

**Better for "just distributed training"**: if all you want is multi-GPU PyTorch, Ray's overhead doesn't pay for itself.

**Lighter weight**: Ray has its own runtime (Raylet) on every node; PyTorchJob is just pods.

**Smaller ecosystem**: many distributed-training papers' reference implementations target torch.distributed directly; using Ray requires adapting.

## When to use Ray on K8s

The clear cases:

- **Hyperparameter tuning at scale** with Ray Tune.
- **Mixed Python pipelines** (data + training + eval) as one job.
- **RAG and agent workloads** where the orchestration is at the Python-task level.
- **Model serving** with Ray Serve (autoscaling, request routing).
- **Reinforcement learning** training (Ray's original use case; the framework is RL-friendly).

When PyTorchJob is fine:
- Pure distributed pretraining or fine-tuning.
- You want to stay close to torch.distributed.

When Slurm is the right choice:
- Large dedicated cluster, HPC-style.

## The serving-side view: Ray Serve

Ray Serve is one of the production options for LLM and ML model serving on K8s. It supports:
- Autoscaling based on traffic.
- Heterogeneous deployments (different replicas for different model variants).
- Composition (chain models, route between them).
- Native LLM serving via the `serve.llm` API (2024+).

For LLM serving on K8s, alternatives include:
- vLLM directly (deployed as a K8s Deployment).
- Triton Inference Server.
- KServe.
- Ray Serve.

Each has tradeoffs; Ray Serve wins for complex compositional pipelines (RAG, agent flows) and for autoscaling.

## What you should believe after this lesson

Three sentences:

**1. Ray is a Python-native distributed computing framework**; KubeRay runs Ray clusters as K8s resources. The abstraction is at the Python function level (vs PyTorchJob's process-level abstraction).

**2. Ray shines for hyperparameter tuning, heterogeneous Python pipelines, RL, RAG/agent workloads, and autoscaling model serving** — places where the orchestration is naturally at the task level. For pure distributed pretraining, PyTorchJob is usually simpler.

**3. KubeRay integrates Ray with the K8s ecosystem** (CRDs for RayCluster, RayJob, RayService) — gets you Ray's productivity within K8s's operational model. Common choice in modern ML platforms.

## Hands-on (at home)

Launch a Ray cluster via KubeRay.

```bash
# Install KubeRay operator.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0

# Create a small RayCluster.
cat > ray-cluster.yaml <<'EOF'
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: small-cluster
spec:
  rayVersion: 2.30.0
  headGroupSpec:
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.30.0
          resources:
            limits: {cpu: "1", memory: "2Gi"}
  workerGroupSpecs:
  - replicas: 2
    groupName: small
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.30.0
          resources:
            limits: {cpu: "1", memory: "2Gi"}
EOF
kubectl apply -f ray-cluster.yaml

# Once ready, port-forward to the head pod and submit a job.
kubectl port-forward svc/small-cluster-head-svc 10001:10001 &
ray job submit --address http://localhost:10001 -- python -c "import ray; ray.init('auto'); print(ray.nodes())"
```

You should see the Ray cluster's node info — the head and the two workers.

For ML, use Ray Train's TorchTrainer (with GPU resources) for a distributed training example.

## Further reading

- Ray documentation (docs.ray.io).
- KubeRay GitHub.
- "Ray for the Curious" — gentle intro.
- "Ray for Production ML" — Anyscale blog.

Next lesson: **Topology-aware scheduling.** Why naive scheduling halves throughput on tight-coupled jobs, and how Slurm + K8s schedulers can be made topology-aware.
