---
title: "Lesson 13 — Spot / Preemptible Scheduling"
date: "2026-06-04"
module: "cluster-orchestration"
order: 13
tags: ["spot", "preemptible", "checkpointing", "restart", "cost"]
author: "Sudipta Pathak"
prerequisites: ["12-multi-tenancy"]
---

# Lesson 13 — Spot / Preemptible Scheduling

## Why this lesson exists

Cloud providers offer "spot" or "preemptible" GPU instances at 50-90% discount vs on-demand prices. The catch: they can be reclaimed at any time, usually with 30 seconds to 2 minutes of warning.

For training workloads that can checkpoint and resume, this is a huge cost win. A 30-day training run on spot at 50% discount saves significant money — if you can recover from preemption without losing too much progress.

This lesson covers the patterns: checkpointing strategies, the orchestrator-level retry mechanisms, and the realistic recovery time per preemption.

The lesson is reading. The Hands-on configures a preemptible job in K8s.

## What spot instances are

Each cloud has its variant:
- **AWS Spot**: up to 90% discount. Reclaimed with 2-minute warning.
- **GCP Preemptible / Spot**: similar discount. Reclaimed with 30-second warning.
- **Azure Spot**: similar pricing; variable notice.

For GPU instances specifically:
- AWS p4d (A100) and p5 (H100) spot pricing fluctuates; sometimes 50%, sometimes 90% off.
- GCP A3 (H100) spot: typically 60-80% off.
- Azure NDv5 (H100) spot: similar.

The savings are real and substantial. A training run that costs $1M on-demand might cost $200-400K on spot.

The catch: preemption. If a 32-node training run loses 1 node mid-training, the whole run fails unless you've planned for it.

## The checkpointing strategy

The cornerstone: frequent checkpoints.

A typical recipe:
- **Save a checkpoint every 1000 steps** (or every 30-60 minutes, whichever is sooner).
- **Save to durable storage** (object store, parallel filesystem) not local disk.
- **On restart, resume from the latest checkpoint**.

The checkpoint includes:
- Model weights.
- Optimizer state (Adam moments, master FP32 weights).
- Learning rate scheduler state.
- RNG state (for deterministic resumption).
- The data loader's position (which batches have been seen).

In PyTorch:
```python
checkpoint = {
    'step': step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'rng': torch.get_rng_state(),
}
torch.save(checkpoint, '/checkpoints/step-' + str(step) + '.pt')
```

For FSDP / sharded models, checkpointing has additional complexity (each rank saves its shard; on resume, rebuild the full state). PyTorch's `torch.distributed.checkpoint` handles this.

## The cost of checkpointing

Checkpointing isn't free:
- I/O time to write the checkpoint.
- All ranks need to synchronize (FSDP).

For Llama 70B at BF16: checkpoint is ~140 GB (params + optimizer state). At a write speed of 5 GB/s to S3 (parallel uploads), ~28 seconds. Plus the gather operation to bring sharded state together: ~10-30 seconds.

Total checkpoint time: ~1 minute every checkpoint. For 1000-step checkpoint interval at 30 seconds per step: 1 minute of checkpoint per 8.3 hours of compute. ~0.2% overhead. Cheap.

## The orchestrator-level retry

Beyond checkpointing, the orchestrator needs to:
1. Detect when a pod is preempted.
2. Restart the pod (possibly on a different node).
3. Have the training script resume from checkpoint.

For Kubernetes:
- PyTorchJob's `restartPolicy: OnFailure` triggers retries.
- Cloud node pools (GKE preemptible, EKS spot) auto-replace preempted nodes; new pods schedule on the replacements.

For Slurm:
- `sbatch --requeue` re-queues the job on preemption.

The script handles "resume from latest checkpoint" logic itself; the orchestrator handles "restart the script."

## Time-to-recover

Realistic numbers for a preempted multi-node training run:

1. Preemption notice: 30-120 seconds.
2. Training script catches signal, finishes current step, saves a checkpoint: 1-2 minutes.
3. (Optional, if not done before signal) Wait for node replacement: 5-30 minutes for cloud spot.
4. New pod scheduled and starts: 30-60 seconds.
5. Training script loads checkpoint and resumes: 1-3 minutes.

Total: 5-35 minutes per preemption.

For a 30-day training run with one preemption per day (typical at moderate spot prices): ~150 minutes of "lost" time, less than 0.5% of total time.

If preemptions are very frequent (every hour or so at high-demand times), the math gets worse. Some shops use a hybrid: on-demand for the controller and a fraction of workers, spot for the rest.

## Elastic training (PyTorch elastic)

Elastic training (TorchElastic, baked into `torchrun`) allows the world size to change during training. If a worker is lost, the remaining workers re-form into a smaller world; new workers can join and the world grows.

This is more sophisticated than the simple "restart everything from checkpoint" pattern. Lost workers don't kill the whole job; just temporarily reduce throughput.

PyTorchJob supports elastic via `elasticPolicy`:

```yaml
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 4
    maxReplicas: 16
```

For spot workloads, elastic training shortens recovery time (don't have to wait for replacement nodes; continue with fewer workers until they arrive).

The catch: elastic training is more complex to write correctly. The training script must handle world-size changes mid-run (re-sharding optimizer state, etc.).

## Mixed on-demand + spot

A common pattern at scale:
- **20% on-demand**: the controller, master rank, and a fraction of workers. Stable.
- **80% spot**: the bulk of workers. Save money.

When spot workers are preempted, the on-demand portion ensures the job doesn't fully crash. Combined with elastic training, the job tolerates significant spot churn.

The cost optimum is workload-dependent. For very stable workloads (long pretraining), more spot. For latency-critical workloads (online fine-tuning), more on-demand.

## What you should believe after this lesson

Three sentences:

**1. Cloud spot/preemptible GPU instances offer 50-90% discount** at the cost of preemption (30s-2min warning). For training workloads that can checkpoint and resume, the cost savings are huge.

**2. The checkpoint cadence determines preemption cost**: frequent checkpoints (~ every 1000 steps or 30-60 minutes) limit progress loss to minutes per preemption. The checkpoint write overhead is <1% of training time.

**3. Elastic training (TorchElastic / PyTorchJob's elasticPolicy)** allows the world size to change mid-job; lost workers don't kill the job. Combined with mixed on-demand + spot (20%/80% typical), this enables aggressive spot usage with bounded risk.

## Hands-on (at home)

Configure a preemptible job in K8s (cloud-specific).

For GKE:
```yaml
# Use a preemptible node pool.
spec:
  nodeSelector:
    cloud.google.com/gke-preemptible: "true"
  tolerations:
  - key: cloud.google.com/gke-preemptible
    operator: Equal
    value: "true"
    effect: NoSchedule
```

For EKS with karpenter:
```yaml
# Provision spot instances.
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-spot
spec:
  template:
    spec:
      requirements:
      - key: karpenter.k8s.aws/instance-category
        operator: In
        values: ["p"]  # GPU instance families
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot"]
```

For training scripts, ensure checkpoint-on-shutdown logic:

```python
import signal
import sys

def handle_signal(sig, frame):
    print("Received preemption signal; saving checkpoint and exiting...")
    save_checkpoint(model, optimizer, step)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
```

When K8s preempts the pod, it sends SIGTERM with a grace period (default 30 seconds). The script saves and exits cleanly.

## Further reading

- "Spot instances for ML training" — cloud-specific blog posts (AWS, GCP, Azure).
- "Cost-Optimal ML Training on Spot Instances" tutorials.
- PyTorch Elastic / TorchElastic documentation.
- Karpenter (AWS) documentation.

Next lesson: **Storage for clusters.** Lustre, WekaFS, parallel S3 patterns. Why your training is IO-bound and you didn't notice.
