---
title: "Lesson 9 — Slurm vs Kubernetes"
date: "2026-06-04"
module: "cluster-orchestration"
order: 9
tags: ["slurm", "kubernetes", "comparison", "decision-tree", "tradeoffs"]
author: "Sudipta Pathak"
prerequisites: ["08-slurm"]
---

# Lesson 9 — Slurm vs Kubernetes

## Why this lesson exists

The "Slurm vs Kubernetes" debate is one of the most contentious in ML infrastructure. Different shops have wildly different opinions; some have switched both ways. The truth: both work; the right choice depends on workload mix, team skills, and operational constraints.

This lesson is the honest comparison. Where each wins, where each loses, and how to decide.

The lesson is reading. The Hands-on is conceptual (no setup); you've seen both schedulers in earlier lessons.

## The dimensions of comparison

A useful framework: rate Slurm and K8s on the dimensions that matter for ML.

### 1. Job submission and overhead

**Slurm**: minimal overhead. Submit → schedule → run, <1 second typically.

**Kubernetes**: pod creation involves API calls, scheduling, image pull, init containers, etc. 30-60 seconds for a fresh job is normal; cached images can drop this to ~10 seconds.

**Winner**: Slurm for short jobs (interactive shells, benchmarks). Tie for long jobs.

### 2. Gang scheduling

**Slurm**: native. Multi-node jobs always start atomically.

**Kubernetes**: requires Kueue, Volcano, or another batch scheduler. Stock K8s doesn't have gang scheduling.

**Winner**: Slurm (built-in). K8s competitive with the right add-ons.

### 3. Topology awareness

**Slurm**: native. `topology.conf` defines the network topology; the scheduler is aware.

**Kubernetes**: limited. Some via node labels and pod affinity rules; more sophisticated via specialized schedulers (Volcano with task-topology plugin).

**Winner**: Slurm (built-in). K8s catching up.

### 4. Long-running services

**Slurm**: poor. Designed for batch jobs; running a serving endpoint is awkward.

**Kubernetes**: native. Deployments, Services, Ingress — all built for this.

**Winner**: Kubernetes, decisively.

### 5. Mixed workloads

**Slurm**: poor. Slurm clusters are typically dedicated to batch HPC/ML.

**Kubernetes**: excellent. ML training + serving + non-ML services + databases — all on the same cluster.

**Winner**: Kubernetes.

### 6. Cloud-native

**Slurm**: weak. Cloud deployments via wrappers (AWS ParallelCluster, Azure CycleCloud) but feels bolted on. Auto-scaling node pools is awkward.

**Kubernetes**: native. EKS, GKE, AKS, managed K8s on every major cloud. Cluster autoscalers, spot/preemptible node pools, etc.

**Winner**: Kubernetes.

### 7. Container support

**Slurm**: via Apptainer (formerly Singularity) or Charliecloud. Works but feels secondary to the conda/module model.

**Kubernetes**: container-native. Pods are containers; the whole API is container-first.

**Winner**: Kubernetes.

### 8. Team familiarity

**Slurm**: HPC sysadmins and academic researchers know it. Industry ML engineers usually don't.

**Kubernetes**: every cloud engineer knows it. ML engineers increasingly know it.

**Winner**: Kubernetes for industry; Slurm for HPC/academic.

### 9. Job complexity

**Slurm**: shell scripts. Easy to debug; standard tools.

**Kubernetes**: YAML CRDs. More structured but more verbose; learning curve.

**Winner**: Slurm for simple jobs; K8s for production pipelines (the structure pays off at scale).

### 10. Multi-tenancy

**Slurm**: mature fair-share, accounting, QoS classes. Very fine-grained control.

**Kubernetes**: improving (Kueue, Volcano), but historically weak. Catching up.

**Winner**: Slurm, but K8s closing the gap.

### 11. Failure handling

**Slurm**: requeue + checkpointing. Simple but works.

**Kubernetes**: built-in restart policies; operator-level retries. Often more sophisticated; sometimes overcomplicated.

**Winner**: tie.

## The decision tree

A practical rubric:

```
Are you running a mix of services and training?
├── Yes → Kubernetes
└── No → continue

Are you primarily on cloud with frequent autoscaling?
├── Yes → Kubernetes
└── No → continue

Do you have dedicated HPC-style hardware (bare-metal, IB-connected)?
├── Yes → Slurm (HPC operations team will thank you)
└── No → continue

Is your team primarily K8s-fluent?
├── Yes → Kubernetes (with Kueue/Volcano for batch)
└── No → Slurm (lower operational burden)

Default → Kubernetes (broader ecosystem, more career-portable skills)
```

## The hybrid pattern

Many large ML orgs run both:

- **Slurm for the core training cluster**: dedicated hardware, frontier model training, HPC-style workloads.
- **Kubernetes for everything else**: serving, fine-tuning, evaluation, services, experimentation.

The two are usually separate clusters; jobs flow between them via models being saved to a shared object store.

OpenAI, Anthropic, Meta — all believed to use Slurm for big training, K8s for the rest. The hybrid is the pragmatic answer at scale.

## The "K8s ate Slurm" claim

A common belief: K8s is gradually replacing Slurm for ML. Evidence:
- Kueue and Volcano have closed the batch-scheduling gap.
- KubeRay (Lesson 10) provides Ray on K8s for the things Ray does well.
- New ML startups default to K8s.
- Cloud providers push K8s.

The counter-evidence:
- The largest training jobs (Llama 3 405B, GPT-4-class) still run on Slurm.
- HPC labs continue to invest in Slurm.
- Slurm's per-job overhead remains lower.

The honest assessment in 2026: K8s is winning at the mid-scale and the cloud; Slurm holds the very-large-scale HPC and the dedicated-hardware niche. Both will be around for years.

## What you should believe after this lesson

Three sentences:

**1. Slurm wins for tight-coupled batch ML training on dedicated hardware**; Kubernetes wins for mixed workloads, cloud-native deployments, and serving alongside training. The right choice depends on what's on the cluster beyond ML.

**2. The hybrid pattern is common at large ML orgs**: Slurm for the core training cluster, Kubernetes for everything else. The two clusters communicate via shared storage / object stores.

**3. K8s is closing the gap on Slurm's traditional strengths** (gang scheduling via Volcano/Kueue, topology awareness improving). The "K8s ate Slurm" narrative is partially true; for new mid-scale ML deployments K8s is increasingly the default. Slurm holds the very-large-scale HPC niche.

## Hands-on (at home)

A conceptual exercise: pick three deployment scenarios and decide Slurm vs Kubernetes.

**Scenario 1**: A startup with 50 GPUs running both LLM serving and continuous fine-tuning. Mostly on cloud.
*Answer*: Kubernetes. Mixed workloads; cloud-native; small enough that the K8s overhead isn't a concern.

**Scenario 2**: A research lab with a dedicated 1024-GPU cluster doing only pretraining runs.
*Answer*: Slurm. Pure HPC; dedicated hardware; the operations team is HPC-experienced.

**Scenario 3**: A large company with 10K+ GPUs across multiple data centers running training, inference, experimentation, and non-ML services.
*Answer*: hybrid. Slurm for the dedicated training pods; K8s for everything else.

For your own use case: walk through the decision tree above. The answer is usually clear once you list the workload mix and constraints.

## Further reading

- "Slurm vs Kubernetes" blog posts (many; quality varies).
- "Running Slurm Workloads on Kubernetes" — Project that bridges the two.
- "Slinky" (SchedMD's Slurm-on-K8s) — runs Slurm controllers inside K8s for hybrid environments.

Next lesson: **KubeRay.** Ray on Kubernetes — the framework for distributed training, tuning, and serving with Python-native APIs. The "third way" between Slurm and K8s-native.
