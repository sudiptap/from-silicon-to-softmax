---
title: "Lesson 1 — Kubernetes for ML"
date: "2026-06-04"
module: "cluster-orchestration"
order: 1
tags: ["kubernetes", "k8s", "ml", "gang-scheduling", "stateful"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — Kubernetes for ML

## Why this lesson exists

Kubernetes (K8s) was built for web services: stateless containers, many replicas, scale up and down based on traffic. ML workloads break those assumptions in several ways:

- ML jobs are *stateful* — checkpoints, intermediate results, growing KV caches.
- ML jobs need *gang scheduling* — either all 32 GPUs start together or none do; partial start is useless.
- ML jobs care about *GPU topology* — the 32 GPUs need to be in specific NVLink/IB configurations.
- ML jobs are *long-running* — hours to weeks per job, not seconds per request.
- ML jobs are *batch* — submit, wait for completion, get results; not "always-on serving."

Stock Kubernetes handles some of these poorly; specific extensions and patterns address them. This lesson is the introduction: what's different about ML on K8s, the patterns that have emerged, and why "just run a pod" is wrong.

The lesson is reading. The Hands-on launches a single-pod ML job on a local Kubernetes.

## The Kubernetes basics, quickly

The relevant K8s objects:

- **Pod**: the unit of scheduling. One or more containers that run together on the same node, sharing network and (optionally) storage.
- **Deployment**: manages a set of replica pods for stateless workloads (web services).
- **StatefulSet**: manages pods with stable identity and ordered startup. Closer to what ML needs.
- **Job**: runs pods to completion; restarts on failure.
- **DaemonSet**: runs one pod per node. Used for cluster-wide agents.
- **Service**: stable network endpoint for a set of pods.
- **PersistentVolume / PersistentVolumeClaim**: storage attached to pods.

For ML specifically: the relevant objects are typically `Job` (one-shot training), custom resources from operators (PyTorchJob, MPIJob), and `StatefulSet` (long-running serving with state).

## What ML adds

**Custom resources (CRDs)**: operators like KubeRay, Kubeflow Training Operator, MPI Operator add new object types that K8s manages. E.g., a `PyTorchJob` CRD that creates the right combination of pods, services, env vars for a PyTorch distributed training run.

**GPU device plugin**: NVIDIA's plugin reports GPUs as schedulable resources (`nvidia.com/gpu: 4`). Without this, K8s doesn't know about GPUs.

**GPU operator**: a higher-level NVIDIA package that installs drivers, the device plugin, monitoring, and more. Standard for production K8s clusters with GPUs. Lesson 2 covers this.

**Gang scheduling**: ensure all pods of a job start together. Stock K8s starts pods independently; if 30 of your 32 pods start and 2 are waiting for resources, the 30 sit idle waiting. Gang scheduling (Volcano, Kueue) refuses to start any until all can start. Lesson 5.

**Topology awareness**: NVLink within a node vs IB across nodes vs slow PCIe paths between racks. Stock K8s doesn't see topology; specialized schedulers do. Lesson 11.

**Job-level retry and checkpoint**: ML jobs can run for days. Resilience to node failures requires checkpointing and orchestrator-level restart. Lesson 13.

## What "just run a pod" gets wrong

A naive ML deployment:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: train-job
spec:
  containers:
  - name: trainer
    image: my-pytorch:latest
    command: ["torchrun", "--nproc_per_node=8", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 8
```

This runs `torchrun` on 8 GPUs of one node. Problems:

1. **Single-node only.** No mechanism for multi-node coordination.
2. **No checkpoint/restart.** If the pod dies, you lose the training.
3. **No gang scheduling.** If you scale to multiple pods, K8s starts them independently; partial start wastes resources.
4. **No topology.** The 8 GPUs are arbitrary; might be from different NUMA nodes.
5. **Resource accounting.** Once the pod is running, K8s considers it "active"; no notion of "queued" vs "running."

The fixes are the rest of this module: operators (PyTorchJob, MPIJob), gang scheduling (Kueue, Volcano), checkpointing (operator-level retry), topology-aware scheduling.

## The patterns that work

A 2026 production ML cluster typically has:

1. **NVIDIA GPU Operator** installed (Lesson 2).
2. **Kueue or Volcano** for job queueing (Lessons 4-5).
3. **A training operator** for the framework you use (PyTorchJob from Kubeflow; KubeRay for Ray-based jobs).
4. **Network policies** for tenant isolation.
5. **Persistent storage** (a CSI driver for a shared filesystem; Lesson 14).
6. **Monitoring** (Prometheus + Grafana for cluster-level; per-job logging).

This stack is non-trivial to assemble. Cloud providers (EKS, GKE, AKS) increasingly offer "ML-ready" cluster blueprints; CoreWeave and Lambda offer purpose-built ML clouds.

## Slurm as the alternative

For HPC-style workloads, Slurm is the alternative. Slurm has:
- Built-in gang scheduling.
- Topology awareness (was its native concern for decades).
- Mature batch queuing.
- Less flexible than K8s for non-ML workloads.

Lesson 8 covers Slurm; Lesson 9 covers the Slurm-vs-K8s choice. The short version: Slurm for tight-coupled training workloads on dedicated GPU clusters; K8s for mixed workloads (training + serving + other services).

## What you should believe after this lesson

Three sentences:

**1. Kubernetes was built for stateless web services**; ML workloads break the assumptions (stateful, gang-scheduled, topology-sensitive, long-running, batch). Stock K8s handles ML poorly without specific extensions.

**2. The standard ML on K8s stack** adds: NVIDIA GPU Operator, a batch scheduler (Kueue/Volcano), a training operator (Kubeflow/KubeRay/MPI Operator), persistent storage, and monitoring. This is non-trivial; cloud providers increasingly offer ML-ready blueprints.

**3. Slurm is the alternative for HPC-style ML workloads**; tight-coupled training on dedicated GPU clusters. K8s wins for mixed workloads; Slurm wins for pure training at scale. We cover both later in the module.

## Hands-on (at home)

Launch a single-pod ML job on a local Kubernetes.

```bash
# Install kind (Kubernetes in Docker) if you don't have a cluster.
go install sigs.k8s.io/kind@latest  # or via brew, etc.
kind create cluster --name ml-demo

# Apply a simple job that prints "hello from container" using a small image.
cat > hello-job.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: hello
spec:
  template:
    spec:
      containers:
      - name: hello
        image: busybox
        command: ["echo", "hello from a Kubernetes job"]
      restartPolicy: Never
  backoffLimit: 0
EOF

kubectl apply -f hello-job.yaml
kubectl logs -l job-name=hello
kubectl delete -f hello-job.yaml
```

For a real ML pod (requires a GPU node; not available in kind):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-pod
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
    command: ["python", "-c", "import torch; print(torch.cuda.is_available())"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

On a GPU-enabled cluster (EKS with GPU nodes, GKE GPU pool, or a bare-metal cluster), the output will be `True`. On a non-GPU cluster, K8s won't schedule the pod because no node has `nvidia.com/gpu`.

## Further reading

- Kubernetes documentation (kubernetes.io/docs).
- "Kubernetes Up & Running" (Hightower, Burns, Beda) — the foundational K8s book.
- "Designing Distributed Systems" (Burns) — for the architectural patterns.
- NVIDIA's "Running GPUs on Kubernetes" overview.

Next lesson: **NVIDIA GPU Operator + Device Plugin.** What gets installed; what it manages; what breaks when you skip it.
