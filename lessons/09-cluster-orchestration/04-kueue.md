---
title: "Lesson 4 — Kueue"
date: "2026-06-04"
module: "cluster-orchestration"
order: 4
tags: ["kueue", "batch", "queueing", "kubernetes", "resource-flavors"]
author: "Sudipta Pathak"
prerequisites: ["03-training-job-anatomy"]
---

# Lesson 4 — Kueue

## Why this lesson exists

Stock Kubernetes doesn't have a real batch job queue. When you submit a Job, it tries to start immediately; if resources aren't available, the pod sits `Pending` indefinitely with no notion of "queued vs running vs preempted." For a shared ML cluster with many users submitting jobs, this is unworkable.

Kueue (Kubernetes-native job queueing, started in 2022 by Google and a community of contributors) adds the missing layer: ResourceFlavors, ClusterQueues, LocalQueues, and Workload objects. It's the standard batch scheduler for ML on K8s in 2026.

This lesson covers what Kueue is, its API model, and how it integrates with the training operators (PyTorchJob, MPIJob, RayJob).

The lesson is reading. The Hands-on installs Kueue and queues a job.

## The problem Kueue solves

Without Kueue, on a shared cluster:
- 10 users submit jobs at once.
- 8 of them get scheduled (resources available).
- 2 of them sit `Pending`.
- A 9th job completes; its resources free up.
- The next-pending pod gets scheduled — but it might not be the one that's been waiting longest.
- No fairness, no priority, no preemption.

Kueue addresses all of these.

## The Kueue API

Three main resource types:

**ResourceFlavor**: a "shape" of resources available in the cluster. E.g., "h100-spot" (Spot H100 instances), "h100-on-demand", "a100-on-demand". A flavor is defined by node selectors / taints; it represents a type of capacity.

**ClusterQueue**: a queue with quotas across flavors. E.g., "research-team-queue" with quota of "100 GPU-hours of h100-on-demand per day." The cluster admin configures these.

**LocalQueue**: a per-namespace pointer to a ClusterQueue. Users in namespace "team-a" submit jobs to a LocalQueue that points at the team's ClusterQueue.

**Workload**: an internal Kueue object tracking a job's queue position, requested resources, and admission status. You don't create Workloads directly; the operators do.

The flow:
1. User submits a Job (or PyTorchJob, MPIJob, etc.) with a `kueue.x-k8s.io/queue-name` label pointing at a LocalQueue.
2. Kueue creates a Workload tracking the job.
3. Kueue's controller checks the ClusterQueue's quota. If resources are available, the Workload is *admitted*.
4. Once admitted, Kueue lets Kubernetes proceed with normal scheduling.
5. If not admitted, the Workload stays queued. Other admitted workloads complete; eventually this one gets admitted.

The key word: **admission**. Kueue gates job admission based on quota; once admitted, normal K8s scheduling happens.

## ResourceFlavors and node selectors

A ResourceFlavor maps to a slice of cluster capacity:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: h100-on-demand
spec:
  nodeLabels:
    cloud.google.com/gke-accelerator: nvidia-h100-80gb
    spot: "false"
```

This flavor represents H100 GPU nodes that aren't spot instances. A ClusterQueue can have quota in this flavor; jobs that request it land on H100 on-demand nodes.

For multi-cloud or multi-instance-type setups, you'd have many ResourceFlavors. The flavor tells Kueue both "which capacity bucket" and "where to schedule" (via the node selectors).

## ClusterQueue configuration

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: research-queue
spec:
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: h100-on-demand
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 64
    - name: h100-spot
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 128
```

This queue has 64 H100 on-demand GPUs and 128 H100 spot GPUs of quota. Jobs submitted to this queue are admitted as long as the running total doesn't exceed the quotas.

The flavor ordering matters: Kueue prefers earlier-listed flavors. So jobs land on on-demand first; spot is the overflow.

## Cohorts (cross-queue sharing)

Multiple ClusterQueues can share resources via a *cohort*: a label that links queues. If queue A's nominalQuota is unused, queue B (in the same cohort) can borrow it.

This lets you have soft quotas: "team A is guaranteed 50 GPUs but can use up to 100 if team B isn't using theirs."

The borrowed resources are reclaimed when the lending team needs them; the borrowing job may be preempted.

## Priority and preemption

Kueue supports priority classes:
- Higher-priority workloads can preempt lower-priority ones if quota is exhausted.
- Within the same priority, oldest-first scheduling.

Preemption is triggered by Kueue: it asks Kubernetes to delete the lower-priority pods so the higher-priority job can take their resources.

For ML training, preemption is meaningful only if jobs checkpoint frequently. Otherwise preempting a job loses its progress. Kueue assumes preemption-tolerant workloads; checkpoint-frequently is your responsibility.

## Integration with training operators

Kueue integrates with several training operators:
- **Kubeflow Training Operator**: PyTorchJob, TFJob, MPIJob, MXJob, XGBoostJob.
- **KubeRay**: RayJob (Lesson 10).
- **Argo Workflows**.
- **Stock Kubernetes Jobs**.

For each, you set the `kueue.x-k8s.io/queue-name` label, and Kueue handles admission.

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: my-training
  labels:
    kueue.x-k8s.io/queue-name: research-queue
spec:
  # ...
```

The operator creates the pods; Kueue gates them via the queue.

## What you should believe after this lesson

Three sentences:

**1. Kueue is the missing batch-queue layer for Kubernetes** — adds ResourceFlavors (capacity buckets), ClusterQueues (quotas), LocalQueues (per-namespace queue handles), and Workloads (internal queue items). Jobs are admitted based on quota; admitted jobs proceed with normal K8s scheduling.

**2. The cohort mechanism** allows queues to borrow unused quota from each other; the borrowing is preempted when the lending queue needs it back. This enables soft quotas and overall higher utilization.

**3. Kueue integrates with training operators** (PyTorchJob, MPIJob, RayJob) via a label; the operator creates the pods, Kueue gates admission. This is the standard ML-on-K8s batch scheduler in 2026.

## Hands-on (at home)

Install Kueue and queue a job.

```bash
# Install Kueue.
VERSION=v0.7.0
kubectl apply --server-side -f \
    https://github.com/kubernetes-sigs/kueue/releases/download/${VERSION}/manifests.yaml

# Create a ResourceFlavor.
kubectl apply -f - <<'EOF'
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: default-flavor
spec: {}
EOF

# Create a ClusterQueue.
kubectl apply -f - <<'EOF'
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: default-queue
spec:
  namespaceSelector: {}
  resourceGroups:
  - coveredResources: ["cpu", "memory"]
    flavors:
    - name: default-flavor
      resources:
      - name: cpu
        nominalQuota: 4
      - name: memory
        nominalQuota: 8Gi
EOF

# Create a LocalQueue in your namespace.
kubectl apply -f - <<'EOF'
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: user-queue
  namespace: default
spec:
  clusterQueue: default-queue
EOF

# Submit a job through Kueue.
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: queued-job
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  template:
    spec:
      containers:
      - name: hello
        image: busybox
        command: ["sleep", "10"]
        resources:
          requests: {cpu: "1", memory: "1Gi"}
      restartPolicy: Never
EOF

# Watch.
kubectl get workloads -A
kubectl get jobs -A
```

You'll see the Workload created; if quota is available, the Job starts. Submit 10 such jobs at once and watch them queue up.

For GPU testing, replace the resource requests with `nvidia.com/gpu: 1` and adjust the ClusterQueue's resourceGroup accordingly.

## Further reading

- Kueue documentation (kueue.sigs.k8s.io).
- "Introducing Kueue" (Google Cloud blog).
- Kueue YAML examples in the kueue GitHub repo.

Next lesson: **Volcano.** The alternative batch scheduler. Gang scheduling, fair-share, queue management — overlaps Kueue but with different design choices.
