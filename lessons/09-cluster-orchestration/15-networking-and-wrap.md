---
title: "Lesson 15 — Cluster Networking Topology + Module Wrap"
date: "2026-06-04"
module: "cluster-orchestration"
order: 15
tags: ["networking", "fat-tree", "ib-rails", "roce", "module-wrap"]
author: "Sudipta Pathak"
prerequisites: ["14-storage-for-clusters"]
---

# Lesson 15 — Cluster Networking Topology + Module Wrap

## Why this lesson exists

Modules 8 and 9 covered networking and orchestration from the algorithmic and scheduler perspectives. This lesson is the physical-network view: fat trees, IB rails, RoCE topologies — the wire-level reality that the schedulers reason about.

Then the module wrap: a worked design for a hypothetical 256-GPU ML cluster, applying everything Module 9 covered.

## Network topologies

The dominant ML cluster topology is the **fat tree** (or "Clos network"):

- **Leaf switches** at the rack level: each rack has 1-2 leaf switches; nodes connect to them.
- **Spine switches** above: leaves connect upward to spines. Spines are "fat" — high bandwidth aggregating many leaf connections.
- **Super-spine** (in very large clusters): another layer above the spine.

For a 256-GPU cluster (32 nodes × 8 GPUs):
- 2-3 racks of nodes.
- Top-of-rack (ToR) leaf switches with 32+ ports.
- 2-4 spine switches with high cross-section bandwidth.
- Connections sized so any-to-any communication has near-line-rate bandwidth.

The "fatness" requirement: the spine should have enough bandwidth that any pattern of communication between leaves doesn't bottleneck the spine. A well-designed fat tree provides full bisection bandwidth — every pair of nodes can communicate at line rate even when many pairs do so simultaneously.

## IB rails

For very high-end ML clusters with InfiniBand, the typical design is "multi-rail":
- Each GPU has its own dedicated NIC connected to one of N "rails."
- Rails are independent IB networks; each handles a fraction of the traffic.
- NCCL routes traffic across the rails.

NVIDIA's DGX H100 reference design: 8 GPUs, 8 dedicated NICs, 8-rail IB. Each GPU has its own dedicated network channel. This is the gold standard for tight-coupled multi-node ML.

## RoCE vs IB topology

For RoCE deployments (Ethernet-based), the topology is the same fat tree but with Ethernet switches. The lossless-Ethernet configuration (PFC, ECN) must be applied throughout.

RoCE typically allows mixing other Ethernet traffic on the same fabric (storage, management); IB is usually a dedicated network. This is one of the operational pros of RoCE.

## Rack-aware scheduling

The scheduler should know about rack topology and prefer placing multi-node jobs within a rack. Lesson 11 covered the topology awareness; here's how it interacts with the physical layer:

- A 4-node job in 1 rack: all communication within ToR. Best.
- A 4-node job spanning 2 racks: most traffic crosses the spine. ~2× slower.
- A 4-node job spanning 4 racks: even slower.

For Slurm, the `topology.conf` describes the rack topology; jobs with `--switches=1` get rack-local placement.

For K8s, Volcano's task-topology plugin does the same; cloud providers (AWS Capacity Blocks, GCP placement groups) provide instance-type-level guarantees.

## The "spine oversubscription" trap

Some cluster designs save money by oversubscribing the spine: the leaves' total downlink bandwidth exceeds the spine's uplink capacity. This works for typical web workloads (most traffic stays within a rack) but breaks ML.

For ML, the spine needs to handle all-reduces between far racks at full speed. Oversubscription kills multi-rack ML throughput.

The check: when designing or evaluating an ML cluster, look at the cross-section bandwidth (worst-case any-to-any throughput). It should equal the per-node bandwidth × node count.

## The Llama 3.1 405B training network

As an example of a frontier-training network, Meta's reported Llama 3.1 setup:
- 16K H100 GPUs.
- 800 Gbps NDR InfiniBand within and between racks.
- Multi-rail IB (8 rails per node).
- Specifically engineered topology to support 3D parallelism's communication patterns.

This is the high end. Most production deployments are far smaller; the principles scale down.

## Module wrap: a worked 256-GPU cluster design

Let's design a 256-GPU ML cluster from scratch.

**Compute**:
- 32 nodes × 8 H100 GPUs per node.
- Each node: 2 × 64-core EPYC CPUs, 2 TB RAM, 8 × 3.84 TB NVMe.

**Intra-node**:
- NVLink-4 between the 8 GPUs (via NVSwitch). 900 GB/s any-to-any.
- AMD Genoa CPUs; 2 NUMA domains; 4 GPUs per NUMA.

**Inter-node**:
- 8-rail NDR InfiniBand per node (one 400 Gbps rail per GPU).
- 2 racks of 16 nodes each.
- Top-of-rack: 64-port NDR switches (Mellanox QM9700-class).
- 4 spine switches connecting the two ToRs at full cross-section bandwidth.

**Storage**:
- Parallel filesystem: WekaFS, 10 nodes, ~500 TB usable, ~200 GB/s read.
- Object store: S3 (or equivalent), petabyte-scale, for datasets.
- Local NVMe per node: scratch and checkpoint staging.

**Orchestration**:
- Kubernetes with NVIDIA GPU Operator.
- Kueue for admission control + Volcano for gang scheduling.
- Kubeflow Training Operator (PyTorchJob).
- KubeRay for tuning and serving workloads.

**Multi-tenancy**:
- 4 teams; each with 64-GPU soft quota.
- Borrowing via Kueue cohorts.
- 2 priority classes: production (preempts) and research (preemptible).

**Spot policy**:
- Cluster is on-demand (research lab; not chasing cloud spot pricing).
- Could add a "burst" cluster of spot instances for evaluation runs.

**Network policies**:
- Each team in its own namespace.
- Network policies prevent cross-namespace access.
- RBAC controls who can submit to each queue.

**Monitoring**:
- DCGM exporter for per-GPU metrics.
- Prometheus + Grafana.
- Per-job logging via Loki.

This cluster supports:
- A single 256-GPU job for frontier training (e.g., 70B from scratch, 405B fine-tuning with 4-way DP).
- Smaller jobs for experimentation.
- Mixed workloads with multi-tenancy.

## What we covered, what we skipped

Covered: Kubernetes for ML, GPU Operator, training job anatomy, Kueue, Volcano, MPI Operator, Training Operator, Slurm, Slurm vs K8s, KubeRay, topology-aware scheduling, multi-tenancy, spot scheduling, storage, networking topology + this worked design.

Skipped:
- **Cloud-specific deep dives** (EKS, GKE, AKS specifics).
- **Specific monitoring stacks** (mentioned Prometheus; didn't go deep).
- **CI/CD for ML** (Module 10).
- **Specific security configurations** (network policies, RBAC details).
- **Hyperscale-specific patterns** (Meta, Google, Microsoft internal stacks; mostly proprietary).

## Mental models to carry forward

Five sentences:

**1. Stock Kubernetes doesn't handle ML well**; the standard stack adds NVIDIA GPU Operator, a batch scheduler (Kueue/Volcano), a training operator (Kubeflow/KubeRay), and persistent storage. Slurm is the alternative for HPC-style dedicated clusters.

**2. Topology matters**: same hardware, different scheduling decisions, 2× throughput difference. Make the scheduler topology-aware; label nodes with rack info; use multi-rail IB for tight coupling.

**3. Multi-tenancy is hard but mature**: quotas + priorities + preemption + fair-share + borrowing. Kueue + Volcano + Slurm all provide these. The mechanism choice depends on the cluster's workload mix.

**4. Spot/preemptible instances** save 50-90% if your training checkpoints and resumes. Elastic training and mixed on-demand+spot are the patterns for production-grade spot usage.

**5. Storage is the silent killer of training throughput** — ML jobs often saturate the parallel filesystem before they saturate the GPUs. Multi-tier storage (NVMe + parallel FS + object store) with pre-staging is the standard pattern.

## What's next

**Module 10: ML Platform Engineering.** The layer above orchestration: experiment tracking, model registry, CI/CD for models, monitoring, observability. Where the ML lifecycle meets DevOps.

**Module 11: Agents from Scratch.** The application layer. Tool use, planning, multi-step reasoning, the agent architectures.

## End of Module 9

Module 1 made the CPU fast. Module 2 the GPU. Module 3 the model. Module 4 Apple Silicon. Module 5 mobile/edge. Module 6 on-device. Module 7 the inference engine. Module 8 the network and distributed primitives. Module 9 the cluster orchestration that ties hardware and jobs together.

Module 10 picks up at the ML lifecycle layer: experiment tracking, model registry, deployment, monitoring. The stuff that turns ML from "training works" into "models reliably deployed in production."
