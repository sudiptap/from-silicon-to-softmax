---
title: "Lesson 12 — Multi-Tenancy"
date: "2026-06-04"
module: "cluster-orchestration"
order: 12
tags: ["multi-tenancy", "quotas", "fair-share", "preemption", "priority"]
author: "Sudipta Pathak"
prerequisites: ["11-topology-aware-scheduling"]
---

# Lesson 12 — Multi-Tenancy

## Why this lesson exists

A $50M ML cluster doesn't belong to one team. Real production clusters serve multiple teams (research, product, infra), multiple projects, multiple priorities. The orchestration layer's job is to ensure each team gets their fair share without one runaway job starving everyone else.

This lesson covers the multi-tenancy mechanisms — quotas, priorities, preemption, fair-share — and the practical patterns for sharing clusters across organizations.

The lesson is reading. The Hands-on configures multi-tenant quotas in Kueue.

## The problem space

Without multi-tenancy controls:
- Team A submits a 256-GPU job; it eats the whole cluster.
- Teams B and C wait. And wait.
- Eventually someone complains; an admin manually intervenes.

The goals:
- Each team has a guaranteed minimum allocation.
- Idle capacity gets used (sharing across teams).
- Burst usage is allowed up to a cap.
- Higher-priority work preempts lower-priority work when needed.
- The scheduler enforces fairness automatically; no manual intervention.

## Quotas

The basic mechanism: each team gets a *quota* — a guaranteed allocation they can always use. Quotas come in two flavors:

**Soft quota**: a guideline. The team can exceed it if other teams aren't using their share. Borrowing is allowed.

**Hard quota**: an absolute cap. Even if the cluster is empty otherwise, the team can't exceed this.

In Kueue terms (Lesson 4):
- `nominalQuota`: soft quota; can be borrowed from.
- `borrowingLimit`: how much can be borrowed.
- `lendingLimit`: how much can be lent to others.

In Slurm terms:
- `Account` with `MaxJobsPerAccount`, `GrpTRES`: similar concepts.

The standard pattern: configure soft quotas summing to the cluster size; let teams burst when others are idle.

## Priorities

Priorities determine which job runs when resources are contested. Two levels:

**Job-level priority**: within a queue, higher-priority jobs run before lower-priority ones.

**Queue-level priority**: across queues, higher-priority queues' jobs run before lower-priority ones.

For ML, common priority classes:
- **Production**: highest. Serving deployments and time-critical pipelines.
- **Research**: medium. Active experiments.
- **Background**: low. Sweeps, exploratory work that can wait.

Higher-priority work doesn't necessarily preempt lower-priority work running; it just goes ahead in the queue. Preemption is a separate decision (next).

## Preemption

Preemption: kill a running job to make room for a higher-priority one.

For ML, preemption is fraught:
- Killing a training job loses its progress (unless it checkpointed recently).
- Even with checkpoints, restart takes minutes and re-uses some compute.

Mitigation:
- **Check the time-to-preempt**: schedulers can be configured to wait for a grace period before killing.
- **Use checkpointing**: jobs checkpoint frequently (every N minutes) so preemption doesn't lose more than N minutes.
- **Avoid preemption for short-running jobs**: prefer to wait for short jobs to finish naturally.

Both Kueue and Volcano support priority-based preemption with configurable policies.

## Fair-share

Fair-share is the algorithm that decides priority over time:
- A user/team's "share" of past usage.
- Heavy users get lower priority (their share is "used up").
- Light users get higher priority (they have "share to spend").

The window over which usage is accounted matters:
- Short window (hours): responsive to recent usage but volatile.
- Long window (weeks): smoother but less responsive.

Slurm has mature fair-share (decades of HPC operations). Kueue has basic queue fair-share; Volcano has DRF (Dominant Resource Fairness, see below).

## Dominant Resource Fairness (DRF)

DRF (Ghodsi et al, 2011) is the standard multi-resource fairness algorithm. The idea:

- Each user/team has a *dominant resource* — the resource type they request most relative to its total availability.
- Fairness is computed per dominant resource.

Example:
- Cluster: 1000 CPUs, 100 GPUs.
- Team A: requests 10 CPUs + 0 GPUs per job. Dominant: CPU (1%).
- Team B: requests 1 CPU + 10 GPUs per job. Dominant: GPU (10%).

DRF allocates such that the *dominant share* is equalized:
- Team A: 50% of CPUs = 500 jobs.
- Team B: 50% of GPUs = 5 jobs.

This handles the multi-resource case better than naive CPU-only or GPU-only fair-share.

Volcano implements DRF. Kueue's fair-share is per-resource; less sophisticated.

## Borrowing and lending

The cohort mechanism (Kueue, Lesson 4; similar in Slurm via account sharing) allows queues to borrow from each other when idle.

Operational pattern:
- Each team has a soft quota = their guaranteed share.
- All teams in a "shared pool" can borrow from each other.
- When the owning team needs their share back, borrowed jobs are preempted (or wait to complete).

This dramatically improves utilization. A cluster running pure isolated quotas typically sees 40-60% utilization; with borrowing, 70-85%.

## Pre-emptible vs guaranteed tiers

A common pattern in cloud-ML:

- **Guaranteed tier**: standard pricing; jobs aren't preempted. For production serving and critical training.
- **Pre-emptible tier**: cheaper (50-90% discount on cloud spot); jobs can be preempted with short notice. For batch ML where checkpointing is in place.

The cluster orchestrator routes jobs to the right tier based on the user's choice. Workloads tagged "pre-emptible" go to spot instance node pools; "guaranteed" goes to on-demand.

This isn't strictly "multi-tenancy" but is multi-tenancy-adjacent: different teams with different latency/cost preferences use different tiers.

## Network policies and isolation

Beyond resource fairness, multi-tenant clusters need security isolation:

- **Network policies**: prevent team A's pods from accessing team B's pods.
- **RBAC (Role-Based Access Control)**: who can submit jobs to which queue, view which namespaces, etc.
- **Pod security policies**: prevent containers from doing privileged things.

These are general Kubernetes patterns; ML-specific quirks are minimal.

## What you should believe after this lesson

Three sentences:

**1. Multi-tenancy in ML clusters requires quotas (soft + hard), priorities (job-level + queue-level), preemption (with care), and fair-share** (DRF for multi-resource cases). Kueue + Volcano + Slurm all provide these mechanisms with varying maturity.

**2. Borrowing and lending across queues** (cohorts in Kueue, sharing in Slurm) is what gets utilization from 40-60% (isolated quotas) to 70-85%. The owning team can reclaim with preemption; pre-emptible jobs need checkpointing.

**3. Pre-emptible vs guaranteed tiers** is the cloud-ML pattern for different cost/latency preferences. Different teams or different workloads in the same team route to different tiers based on their needs.

## Hands-on (at home)

Configure multi-tenant quotas in Kueue.

```yaml
# Two teams sharing the cluster.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: gpu-flavor
spec: {}
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-a-queue
spec:
  namespaceSelector: {matchLabels: {team: a}}
  cohort: shared-pool
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: gpu-flavor
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 16  # team A's guaranteed share
        borrowingLimit: 16  # can borrow up to 16 more from cohort
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-b-queue
spec:
  namespaceSelector: {matchLabels: {team: b}}
  cohort: shared-pool
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: gpu-flavor
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 16
        borrowingLimit: 16
```

Total cluster: 32 GPUs (the sum of nominal quotas). Each team is guaranteed 16; either team can use up to 32 if the other is idle (via borrowing).

Configure priority classes:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: production
value: 1000
preemptionPolicy: PreemptLowerPriority
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: research
value: 100
```

Then in your jobs:
```yaml
spec:
  priorityClassName: production
```

Production jobs preempt research jobs if needed.

## Further reading

- "Dominant Resource Fairness: Fair Allocation of Multiple Resource Types" (Ghodsi et al, 2011).
- Kueue cohorts documentation.
- Slurm Fair Tree and Fair Share documentation.
- "Kubernetes Multi-Tenancy" working group docs.

Next lesson: **Spot / preemptible scheduling.** Recovering 50-90% cost via spot instances; checkpoint + restart patterns at the orchestrator layer; how to lose only minutes when a node disappears.
