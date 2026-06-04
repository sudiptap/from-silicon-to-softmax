---
title: "Cluster Orchestration: Module Overview"
date: "2026-05-26"
excerpt: "Running jobs on iron. Kubernetes for ML, Slurm, Volcano/Kueue, MPI Operator, KubeRay, topology-aware scheduling, multi-tenancy, spot/preemptible orchestration — the layer between 'I can write FSDP' and 'my 256-GPU training run finished without paging anyone at 3am'."
module: "cluster-orchestration"
order: 0
tags: ["kubernetes", "slurm", "scheduling", "kueue", "volcano", "kuberay", "infrastructure", "overview"]
author: "Sudipta Pathak"
prerequisites: ["distributed-systems"]
---

# Cluster Orchestration

## Why this module exists

Most "ML systems" courses stop one layer too high. They teach you FSDP, ZeRO, tensor parallel, NCCL — *how* distributed training works algorithmically. Then they hand you a `torchrun` command and call it production.

But in real life the bottleneck is rarely *how* you shard the model. It's:

- How do you get the job onto a cluster you share with ten other teams?
- What happens when a node dies 6 hours into a 48-hour run?
- How do you stop one runaway job from starving everyone else?
- How do you take advantage of cheap spot capacity without losing 12 hours of training?

This is the **orchestration layer**, and it's where most candidates lose points in infra interviews. This module covers it.

## How this fits

This is the fifth module of the [depth track](/curriculum) — *From Silicon to Softmax*. It sits between Distributed Systems (the primitives — NCCL, RDMA, parallelism patterns) and ML Platform Engineering (the meta-layer — experiment tracking, model registry, deployment).

If you can already explain how FSDP shards optimizer state, this is the module that teaches you how to get FSDP running reliably on 32 nodes you don't own.

## The roadmap

15 tutorials. Topics get linked here as they ship.

### Foundations

1. **Kubernetes for ML** — what's actually different about ML workloads (gang scheduling, GPU topology, long-running stateful jobs, why "just run a pod" is wrong)
2. **NVIDIA GPU Operator + Device Plugin** — what they install, what they manage, what breaks when you skip them
3. The cluster-side view of a training job — pods, containers, devices, init containers, sidecars

### Batch scheduling on Kubernetes

4. **Kueue** — Kubernetes-native job queueing, ResourceFlavors, ClusterQueues
5. **Volcano** — gang scheduling, fair-share, queue management
6. **MPI Operator** — running NCCL/MPI jobs on K8s (the model behind `mpirun`)
7. **Training Operator (Kubeflow)** — PyTorchJob, MPIJob, the operator pattern for ML

### The HPC side

8. **Slurm** — the HPC standard, why frontier labs still use it
9. **Slurm vs Kubernetes** — the genuine tradeoffs (not "Slurm bad" or "K8s ate the world")
10. **KubeRay** — Ray on Kubernetes for training, tuning, and serving

### Scheduling intelligence

11. **Topology-aware scheduling** — NVLink, NUMA, switch awareness; why naive scheduling halves throughput
12. **Multi-tenancy** — quotas, priorities, preemption, fair-share; how to share a $50M cluster across ten teams
13. **Spot / preemptible scheduling** — checkpoint + restart at the orchestrator layer; how to recover 95% of cost without losing your training run

### The substrate

14. **Storage for clusters** — Lustre, WekaFS, parallel S3 patterns; why your training is IO-bound and you didn't notice
15. **Cluster networking topology** — fat trees, IB rails, RoCE; how rack-aware scheduling changes everything

### Provider patterns *(running themes throughout)*

How each topic above looks on EKS, GKE, AKS, CoreWeave, and bare-metal — with the gotchas that only show up in one.

---

## What I'm filling in over time

This is the topic scaffold. Tutorials will arrive as I write them — expect a mix of single-cluster builds (one tutorial, one cluster) and comparative deep-dives (Slurm-vs-K8s on the same workload). Open an issue or reach out if there's an order you'd like to see things written in.
