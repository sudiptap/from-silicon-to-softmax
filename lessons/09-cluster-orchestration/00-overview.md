---
title: "Module 9 — Cluster Orchestration"
date: "2026-06-04"
module: "cluster-orchestration"
order: 0
tags: ["kubernetes", "slurm", "scheduling", "kueue", "volcano", "kuberay", "infrastructure", "overview"]
author: "Sudipta Pathak"
prerequisites: ["distributed-systems"]
---

# Cluster Orchestration

## Why this module exists

Most "ML systems" courses stop one layer too high. They teach you FSDP, ZeRO, tensor parallel, NCCL — *how* distributed training works algorithmically. Then they hand you a `torchrun` command and call it production.

In real life the bottleneck is rarely *how* you shard the model. It's:

- How do you get the job onto a cluster you share with ten other teams?
- What happens when a node dies 6 hours into a 48-hour run?
- How do you stop one runaway job from starving everyone else?
- How do you take advantage of cheap spot capacity without losing 12 hours of training?

This is the **orchestration layer**. It sits between the distributed primitives (Module 8) and the ML platform (Module 10). It's where most candidates lose points in infra interviews because the surface area is large and the patterns are vendor-specific.

This module is the bridge. By the end, you should know how to schedule a 32-node training job on a shared cluster, recover from failures, share resources across teams, and choose between Slurm and Kubernetes for a given workload.

## How this fits

Module 9 of the depth track. Module 8 covered the network primitives (RDMA, NCCL, parallelism); Module 10 covers ML platform engineering (experiment tracking, model registry). Module 9 sits between: the cluster manager that takes "I want 32 GPUs" and turns it into "your job is running on these specific GPUs."

The output of this module: the ability to deploy a distributed training or inference job onto a Kubernetes or Slurm cluster, handle failures, share resources across multiple users, and reason about cost/throughput tradeoffs.

## The roadmap

Fifteen lessons.

### Foundations

1. **Kubernetes for ML** — what's actually different about ML workloads (gang scheduling, GPU topology, long-running stateful jobs, why "just run a pod" is wrong).
2. **NVIDIA GPU Operator + Device Plugin** — what they install, what they manage, what breaks when you skip them.
3. **The cluster-side view of a training job** — pods, containers, devices, init containers, sidecars.

### Batch scheduling on Kubernetes

4. **Kueue** — Kubernetes-native job queueing, ResourceFlavors, ClusterQueues.
5. **Volcano** — gang scheduling, fair-share, queue management.
6. **MPI Operator** — running NCCL/MPI jobs on K8s.
7. **Training Operator (Kubeflow)** — PyTorchJob, MPIJob, the operator pattern for ML.

### The HPC side

8. **Slurm** — the HPC standard, why frontier labs still use it.
9. **Slurm vs Kubernetes** — the genuine tradeoffs.
10. **KubeRay** — Ray on Kubernetes for training, tuning, and serving.

### Scheduling intelligence

11. **Topology-aware scheduling** — NVLink, NUMA, switch awareness; why naive scheduling halves throughput.
12. **Multi-tenancy** — quotas, priorities, preemption, fair-share; sharing a $50M cluster across teams.
13. **Spot / preemptible scheduling** — checkpoint + restart at the orchestrator layer.

### The substrate

14. **Storage for clusters** — Lustre, WekaFS, parallel S3 patterns; why your training is IO-bound and you didn't notice.
15. **Cluster networking topology + module wrap** — fat trees, IB rails, RoCE; how rack-aware scheduling changes everything; the module wrap.

---

## What this module deliberately won't cover

- **Specific cloud provider control planes** in depth (EKS, GKE, AKS specifics). We mention them where the choice matters; the operational details are vendor-specific and change frequently.
- **Application-layer ML platform** (experiment tracking, model registry, monitoring). That's Module 10.
- **General Kubernetes administration** beyond ML-relevant patterns. Plenty of other resources for that.
- **Cost management** in depth — important, but tied to specific cloud/account configurations.
- **Security and compliance** in depth — important but a separate discipline (network policies, RBAC, image signing, etc.).

## How to work through it

Every lesson is fully readable as prose. The hands-on sections vary in compute requirements:

- Lessons 1-3 (Kubernetes basics): a Minikube or kind cluster on your laptop suffices.
- Lessons 4-7 (batch scheduling): a small managed cluster (EKS, GKE) helps; many examples can be simulated with kind.
- Lesson 8 (Slurm): a single-node Slurm install via Docker is sufficient for the basics.
- Lessons 11-13 (scheduling intelligence): requires a real multi-node cluster to see the effects.

If you don't have a cluster, the prose still stands. The mental models are framework-agnostic; the operational specifics are well-documented.

A note on tempo: this module is operations-heavy. Each lesson has a clear "what is this for" component, a "how does the API look" component, and a "what goes wrong" component. The "what goes wrong" is often the most useful part.

The capstone (Lesson 15): a complete production cluster design for a hypothetical 256-GPU ML cluster, with rationale for each choice.
