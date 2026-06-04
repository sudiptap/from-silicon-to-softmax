---
title: "Lesson 5 — Checkpoint Storage at Scale"
date: "2026-06-04"
module: "ml-platforms"
order: 5
tags: ["checkpoint", "s3", "multipart", "storage", "restore"]
author: "Sudipta Pathak"
prerequisites: ["04-data-infrastructure"]
---

# Lesson 5 — Checkpoint Storage at Scale

## Why this lesson exists

Module 9 Lesson 13 covered checkpointing as a fault-tolerance mechanism. This lesson goes deeper on the *storage* side: where checkpoints live, how to write them fast, how to restore them fast, and the storage costs that compound over a training campaign.

For frontier models, checkpoints are TB-scale. Writing a 1 TB checkpoint to S3 at the default 80 MB/s takes 3.5 hours. That's not viable; you'd lose the training during the write. The fix is multipart parallel uploads, multi-node distribution of write work, and the right storage tier.

The lesson is reading. The Hands-on benchmarks S3 multipart upload throughput.

## The checkpoint sizing

Per-rank checkpoint size = `params_per_rank × bytes_per_param + optimizer_state_per_rank`.

For Llama 405B with TP=8, PP=8 (so each rank holds 405B/64 = ~6.3B params), BF16:
- Params: 12.6 GB.
- Adam state (FP32 master + momentum + variance): 76 GB.
- Per rank: ~88 GB.
- Across 64 ranks: ~5.6 TB per checkpoint.

This is the upper end. For 70B models, ~1.4 TB. For 7B, ~120 GB. Substantial in all cases.

## Where checkpoints live

The storage tiers:

**Local NVMe**: temporary; fast (5 GB/s per drive). Write the checkpoint here first; flush to durable storage in the background.

**Parallel filesystem (Lustre, WekaFS)**: durable; fast (100 GB/s aggregate). Permanent storage for checkpoints during the training run.

**Object store (S3, GCS)**: durable; bandwidth varies. Long-term storage; usually after training completes; cheaper per-TB.

A typical recipe:
1. Each rank writes its checkpoint to local NVMe (fast).
2. A background job uploads from local NVMe to parallel FS (medium speed).
3. After training, the parallel FS checkpoint is copied to S3 for archival.

The cost optimization: parallel FS is more expensive than S3 per-TB. Don't keep every checkpoint indefinitely on parallel FS; archive to S3 and delete from parallel FS.

## Multipart upload

S3 (and equivalents) support multipart upload: split a large file into chunks (5MB+ each), upload chunks in parallel, S3 reassembles.

For a 100 GB checkpoint at 100 MB/s per connection:
- Single connection: 1000 seconds (~17 minutes).
- 32 parallel connections (multipart): 30 seconds.

The math is the same as RAID-style parallelism for storage. S3's bandwidth per bucket per second is high (multiple GB/s); the bottleneck is per-connection throughput, which multipart sidesteps.

The implementation: most cloud SDKs (boto3, gsutil, aws-cli) have multipart upload as standard. PyTorch's `torch.save` writes a single file; you wrap the save in a multipart-upload tool.

## Async checkpoint writing

The recipe that works for frontier training:

1. **Step N**: training continues.
2. **Step N+1**: training continues. Background thread starts writing checkpoint from N's state (which the framework kept).
3. **Step N+2 to N+M**: training continues; checkpoint flushes in background. Doesn't block training.
4. By the time N+M completes, the checkpoint is durable.

This requires holding two copies of the model state briefly (live + the snapshot being written). For large models, the extra memory cost is significant — but the alternative is blocking training for minutes per checkpoint.

PyTorch's `torch.distributed.checkpoint` API supports async checkpointing in recent versions.

## Restore speed

When a training job restarts (after preemption, after crash), it loads the latest checkpoint. Restore time matters: if it takes 30 minutes to restore, that's 30 minutes of wasted GPU per failure.

Optimizations:
- **Parallel reads**: each rank reads its shard in parallel.
- **Memory-mapped restore**: mmap the checkpoint file; the OS pages in on access.
- **From-fast-storage**: restore from local NVMe / parallel FS, not from S3.

For a 100 GB per-rank checkpoint at 5 GB/s local NVMe read: 20 seconds. Plus initialization overhead, ~1 minute total. Acceptable.

From S3 at 100 MB/s: 1000 seconds. Painful. Don't restore from S3 if you can avoid it; keep recent checkpoints on parallel FS or NVMe.

## Cost over a campaign

A long training run produces many checkpoints. Storage costs add up.

Example: a 30-day training run with 1 TB checkpoint every hour = 720 TB of checkpoint data.

On S3 standard at ~$23/TB/month: 720 TB × $23 / 30 days = ~$550 per day. Over the 30-day run: ~$16K. Substantial.

Mitigations:
- **Keep only the most recent N checkpoints** (typically 3-5). Older ones are deleted or archived to cheaper storage.
- **Use S3 Glacier or equivalent for archival**: 5-10× cheaper than S3 Standard. Slower to restore (hours), so only for end-of-run snapshots.
- **Compress**: BF16 checkpoints are already dense; compression typically gives only 5-10% reduction. Not worth it.

The right strategy depends on the org's "value of historical checkpoints" — research orgs that revisit old runs keep more; production orgs that only care about the final model keep less.

## Checkpoint format

Two main formats:
- **PyTorch native (`.pt`)**: standard. Each rank's shard is a separate file.
- **Safetensors (`.safetensors`)**: more recent. Faster to load (mmap-friendly); safer (no pickle deserialization vulnerabilities).

For new deployments, prefer safetensors. The HuggingFace ecosystem has standardized on it.

For sharded checkpoints (FSDP / ZeRO), each rank writes its own file. The naming convention typically encodes the rank: `model-r0.safetensors`, `model-r1.safetensors`, etc.

## What you should believe after this lesson

Three sentences:

**1. Frontier-scale checkpoints are TB-scale**; writing them requires multipart parallel uploads, multi-node distribution, and async write (training continues while the checkpoint flushes in background). Single-connection writes to S3 would block training for hours.

**2. The storage tier strategy is: local NVMe (temp) → parallel FS (durable, fast restore) → S3 (long-term archive).** Keep recent checkpoints on parallel FS for fast restore; archive older ones to S3 for cost savings.

**3. Storage costs over a long training campaign are substantial** ($10K+ for a 30-day run with frequent checkpoints). Mitigations: keep only recent N checkpoints; archive older ones to cheaper storage tiers; align cadence with actual restore needs.

## Hands-on (at home)

Benchmark S3 multipart upload throughput.

```python
# s3_multipart_demo.py
# pip install boto3
import boto3
import time
import os

# Create a test file (100 MB).
size_mb = 100
with open("/tmp/testfile", "wb") as f:
    f.write(os.urandom(size_mb * 1024 * 1024))

s3 = boto3.client('s3')
bucket = 'your-bucket'
key = 'multipart-test.bin'

# Single-part upload.
t0 = time.time()
s3.upload_file("/tmp/testfile", bucket, key)
t_single = time.time() - t0
print(f"Single-part: {size_mb} MB in {t_single:.1f}s = {size_mb/t_single:.0f} MB/s")

# Multipart upload (boto3 does multipart automatically for large files; tune the threshold).
from boto3.s3.transfer import TransferConfig
config = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,
    multipart_chunksize=5 * 1024 * 1024,
    max_concurrency=10,
)
t0 = time.time()
s3.upload_file("/tmp/testfile", bucket, key + "-multipart", Config=config)
t_multi = time.time() - t0
print(f"Multipart (concurrency=10): {size_mb} MB in {t_multi:.1f}s = {size_mb/t_multi:.0f} MB/s")
```

Without AWS credentials, you can simulate with a local minio container.

The pattern: more concurrency → faster upload, up to the per-bucket bandwidth limit.

## Further reading

- AWS S3 multipart upload documentation.
- PyTorch `torch.distributed.checkpoint` documentation.
- "Async Checkpointing in Production LLM Training" — various blog posts.
- "Lustre and WekaFS for ML Checkpointing" — vendor whitepapers.

Next lesson: **Training observability.** Prometheus + Grafana for GPU metrics, DCGM, NaN canaries, anomaly detection. Knowing when training is going wrong before the loss blows up.
