---
title: "Lesson 5 — Volcano"
date: "2026-06-04"
module: "cluster-orchestration"
order: 5
tags: ["volcano", "gang-scheduling", "fair-share", "batch", "kubernetes"]
author: "Sudipta Pathak"
prerequisites: ["04-kueue"]
---

# Lesson 5 — Volcano

## Why this lesson exists

Volcano (CNCF, originating from Huawei's batch scheduler) is the other major batch scheduler for Kubernetes — alongside Kueue. Volcano is older (2019+), more featureful, and historically the choice for HPC-style ML workloads. Where Kueue focuses on admission control and resource fairness, Volcano emphasizes gang scheduling and queue management with rich scheduling policies.

This lesson covers Volcano's design and when to pick it over Kueue.

The lesson is reading. The Hands-on installs Volcano and runs a gang-scheduled job.

## What Volcano provides

Volcano is a replacement for the default Kubernetes scheduler. Instead of just queueing (Kueue), Volcano takes over the entire scheduling decision for batch jobs.

Key features:
- **Gang scheduling**: a Job's pods are scheduled all-or-nothing. Never start 30 of 32; either all 32 start or none.
- **Queue priorities**: jobs in higher-priority queues run first.
- **Fair-share**: round-robin across queues to avoid one team starving the others.
- **Preemption**: lower-priority jobs preempted by higher-priority ones.
- **DRF (Dominant Resource Fairness)**: a sophisticated fairness algorithm that accounts for multiple resource types (GPU, CPU, memory).
- **Plugins for HPC frameworks**: MPI Operator, PyTorchJob, TensorFlow, Spark, all integrate with Volcano.

## Gang scheduling in detail

The problem: distributed ML training needs N pods to start together. If only some start, those waste resources without making progress.

Stock K8s doesn't gang-schedule. It starts pods independently as resources become available. For an N-pod job, you might end up with 30 running and 2 pending — the 30 are wasted.

Volcano's gang scheduler:
1. The job specifies "I need all N pods or none."
2. Volcano evaluates: are there enough resources to start all N?
3. If yes: start all N atomically (within a scheduling cycle).
4. If no: keep waiting; no pods start.

The implementation: Volcano introduces a `PodGroup` resource. The Job's pods reference the PodGroup; Volcano's scheduler only starts pods if the whole PodGroup can be scheduled.

This is the most important feature for ML on K8s. Without it, multi-node training fails in predictable ways.

## Queue management

Volcano queues have:
- **Weight**: for fair-share. A queue with weight 2 gets 2× the resources of a queue with weight 1, all else equal.
- **Capability**: max resources the queue can use (hard cap).
- **Reclaimable**: whether resources can be borrowed (similar to Kueue's cohorts).
- **Priority**: queue-level priority for inter-queue arbitration.

Example:
```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: research
spec:
  weight: 1
  capability:
    nvidia.com/gpu: 100
  reclaimable: true
```

A job submitted to this queue can use up to 100 GPUs; the queue weight 1 determines its share when contending with other queues.

## Plugins

Volcano has a plugin architecture for scheduling decisions:
- **predicates**: filter nodes based on pod spec (similar to K8s built-in).
- **priorities**: score nodes for ranking.
- **preempt**: who can preempt whom.
- **reclaim**: when resources can be reclaimed from another queue.
- **task-topology**: topology-aware scheduling (Lesson 11).

The plugin chain is configurable. Different cluster admins enable different plugins.

## Volcano vs Kueue

The honest comparison:

| Aspect | Volcano | Kueue |
| ------ | ------- | ----- |
| Maturity | Older (2019+), CNCF graduated | Newer (2022+), CNCF incubating |
| Gang scheduling | First-class | Via partner schedulers |
| Fair-share | Built-in (DRF) | Limited |
| Replaces default scheduler | Yes | No (admission-only) |
| Integration with training operators | Good | Good |
| Cloud vendor support | Mostly DIY | Google/AWS push Kueue |

Use Volcano when:
- You need strong gang scheduling guarantees.
- HPC-style fair-share across queues matters.
- You're OK replacing the default scheduler.

Use Kueue when:
- You want admission control without replacing the scheduler.
- Your cluster has mixed workloads (ML + non-ML).
- You value GKE/EKS-aligned tooling.

For pure ML clusters at scale, Volcano remains popular. For mixed-workload clusters, Kueue is increasingly the default.

## Production deployments

Volcano in production:
- **Huawei Cloud** (Volcano's origin) — large internal use.
- **CNCF projects** like Kubeflow and KubeRay have Volcano integration.
- **HPC labs** running K8s often use Volcano for the gang scheduling.

Kueue has gained ground in 2023-2026 as the more "Kubernetes-native" choice, but Volcano remains a strong option for ML-first clusters.

## What you should believe after this lesson

Three sentences:

**1. Volcano replaces the default Kubernetes scheduler** with a batch-oriented one that natively supports gang scheduling, queue management, fair-share, and preemption. Most useful for HPC-style ML workloads.

**2. Gang scheduling is Volcano's killer feature**: pods of a multi-node training job start atomically (all or none). Without it, distributed training partial starts waste resources.

**3. Volcano vs Kueue is a real choice**: Volcano is more featureful and replaces the scheduler; Kueue is admission-only and more Kubernetes-aligned. Volcano for ML-first clusters; Kueue for mixed-workload clusters.

## Hands-on (at home)

Install Volcano and run a gang-scheduled job.

```bash
# Install Volcano.
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# Wait for the controllers to be ready.
kubectl get pods -n volcano-system

# Submit a gang-scheduled job.
cat > volcano-job.yaml <<'EOF'
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: gang-job
spec:
  minAvailable: 3
  schedulerName: volcano
  tasks:
  - replicas: 3
    name: hello
    template:
      spec:
        containers:
        - name: hello
          image: busybox
          command: ["sleep", "30"]
          resources:
            requests: {cpu: "1"}
        restartPolicy: Never
EOF
kubectl apply -f volcano-job.yaml

# Check.
kubectl get vcjob
kubectl get pods -l volcano.sh/job-name=gang-job
```

You'll see the 3 pods start together (gang). If you reduce cluster capacity below what the job needs, no pods start (vs default K8s, which would start some).

## Further reading

- Volcano documentation (volcano.sh).
- "Volcano: Collision of the Volcano Scheduler with HPC and AI Workloads" — KubeCon talks.
- "Kueue vs Volcano" comparison posts (various).

Next lesson: **MPI Operator.** Running NCCL/MPI-based jobs on Kubernetes. The bridge between traditional HPC programming model and K8s orchestration.
