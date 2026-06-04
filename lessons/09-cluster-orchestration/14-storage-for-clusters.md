---
title: "Lesson 14 — Storage for Clusters"
date: "2026-06-04"
module: "cluster-orchestration"
order: 14
tags: ["storage", "lustre", "weka", "s3", "parallel-filesystem", "io-bound"]
author: "Sudipta Pathak"
prerequisites: ["13-spot-preemptible"]
---

# Lesson 14 — Storage for Clusters

## Why this lesson exists

ML training is often described as compute-bound. In reality, it's commonly IO-bound — the GPUs sit idle waiting for the next batch of training data to arrive from storage. For a 256-GPU job each step processes ~1 GB of input data; that has to be read from storage, decompressed, and shipped to each GPU. If your storage can't supply >256 GB/s sustained, the GPUs starve.

This lesson covers the storage tiers and patterns for ML clusters: shared filesystems (Lustre, WekaFS, BeeGFS), object stores (S3, GCS) with parallel access, and the right combination for a typical training run.

The lesson is reading. The Hands-on benchmarks IO throughput.

## The storage tiers

ML clusters typically have multiple storage tiers:

1. **Local SSDs / NVMe**: per-node storage. ~5 GB/s per drive. Use cases: scratch space, model weights cached for fast load, checkpoints during writing (before flushing to durable storage).

2. **Parallel shared filesystem**: across the cluster. ~100 GB/s to TB/s aggregate. Use cases: training data, checkpoints, model artifacts.

3. **Object store**: virtually unlimited capacity. ~per-bucket bandwidth varies (S3 can do GBs/s aggregate). Use cases: long-term storage, dataset archives, cross-region sharing.

The patterns:
- **Hot data**: in local NVMe (preloaded) or hot cache of the parallel FS.
- **Active data**: in the parallel FS.
- **Cold data**: in object storage.

## Parallel filesystems

The contenders:

**Lustre**: the HPC standard. Decades of HPC use; scales to PBs and 100 GB/s+. Open-source. Used by national labs and AWS FSx for Lustre.

**WekaFS** (Weka): commercial. Designed for ML. Combines NVMe storage with software that presents a parallel filesystem. Excellent for ML's mixed read/write patterns. Used by several large ML clusters.

**BeeGFS**: another HPC-origin option. Simpler than Lustre; popular in mid-scale HPC.

**GPFS / IBM Storage Scale**: enterprise option from IBM. Common in older HPC deployments.

**Vast Data**: another commercial ML-focused storage. Increasingly popular.

For training, the key metrics are:
- **Aggregate read throughput**: how fast can N nodes pull training data?
- **Metadata operations/second**: how many file opens/lookups per second?
- **Latency for small file reads**: data loaders often read many small files.

ML workloads often hit the metadata bottleneck — a directory with millions of files (e.g., one per image) is metadata-heavy. The parallel FS must handle metadata at scale.

## Object stores

S3 (AWS), GCS (Google), Azure Blob Storage. The common interface: HTTP-based key-value storage.

For ML, object stores serve as:
- The "source of truth" for datasets. Datasets live here long-term.
- Cross-region/cross-cluster sharing.
- Storage for trained models.

The challenge: object stores have higher per-request latency than filesystems (~10-100 ms per GET vs ~1 ms for a local read). For training that reads many small samples, this latency adds up.

Patterns:
- **Pre-staging**: copy data from S3 to the parallel FS or local NVMe before training. The training reads from the fast storage.
- **Streaming with parallel reads**: many concurrent S3 GETs (32-64 in parallel) to saturate bandwidth.
- **WebDataset format**: tar archives of training samples; read sequentially with high throughput. Avoids the per-sample S3 latency.

For long-running training on large datasets, pre-staging is the cleanest pattern. For evaluation or one-shot inference, streaming is fine.

## Data loading throughput

A 256-GPU training job at ~1000 samples/sec/GPU = 256K samples/sec total. At ~10 KB per sample (typical for text training): 2.5 GB/sec aggregate read.

This is well within parallel FS throughput (10-100 GB/s) but can exceed object-store sustained throughput without parallelism.

For larger samples (images at ~100 KB, video at ~1 MB), the throughput requirement scales up:
- Image classification at 256 GPUs: 25 GB/s.
- Video pretraining at 256 GPUs: 250 GB/s. Requires top-tier storage.

If your training is IO-bound (GPUs underutilized, storage saturated), throwing more GPUs at the problem doesn't help. Fix the storage.

## Checkpoint storage

Checkpoints are a different IO pattern: bursty, infrequent, large writes.

For Llama 70B BF16 + Adam: ~280 GB per checkpoint. At a 5 GB/s write to parallel FS: ~1 minute per checkpoint. Acceptable if you checkpoint every hour.

For very-large models (405B+), checkpoints are TB-scale. Write time becomes ~10 minutes per checkpoint; you check point less frequently. Hierarchical checkpointing (save partial state to local NVMe, then push to durable storage in the background) helps.

For sharded models (FSDP, ZeRO), each rank writes its shard. The parallel FS handles the concurrent writes; total write throughput is the bottleneck.

## CSI drivers and K8s integration

Kubernetes accesses storage via CSI (Container Storage Interface) drivers:
- **EBS CSI**: AWS Elastic Block Store. Per-pod block storage.
- **FSx CSI**: AWS FSx for Lustre.
- **Cloud-specific filesystem CSIs**: GCS Fuse, EFS, Azure Files.
- **Weka CSI**, **Lustre CSI**: for their specific systems.

The pattern: define a PersistentVolume backed by a CSI driver; pods claim it via PersistentVolumeClaim; the CSI handles mounting at pod startup.

For ML, a typical setup:
- Each training pod claims a shared PVC for the training data (read-only, mounted on all pods).
- Each pod has a separate PVC for its checkpoints (read-write).

Configuring CSI for parallel filesystems on K8s is a real engineering project. The cloud-managed options (FSx for Lustre, GCS) are much simpler than DIY.

## Storage for serving

Serving has different storage needs:
- Model weights need to be loaded quickly at startup (cold start).
- Possibly: persistent KV cache across requests.

Pattern: cache model weights on local NVMe per inference node; serve them from there. Avoid pulling 100GB from S3 every container restart.

The mmap pattern (Module 6 Lesson 7) interacts with this: mmap'd weights on local NVMe load on-demand; first inference is slow as pages fault in.

## What you should believe after this lesson

Three sentences:

**1. ML clusters need a multi-tier storage architecture**: local NVMe for scratch and hot cache; parallel filesystem (Lustre, WekaFS) for active data and checkpoints; object store for cold data. The parallel FS is the workhorse for training.

**2. Training is often IO-bound at scale** — 256-GPU jobs can need 25-250 GB/s of read throughput depending on sample size. If the storage can't keep up, GPUs sit idle; more GPUs doesn't help.

**3. Object stores (S3, GCS) need careful access patterns**: pre-staging to the parallel FS, parallel reads, or WebDataset-style tar archives. Direct random access at scale fights the per-request latency.

## Hands-on (at home)

Benchmark IO throughput on your machine.

```bash
# Sequential read bandwidth.
dd if=/dev/zero of=/tmp/test bs=1M count=1024  # write 1GB
dd if=/tmp/test of=/dev/null bs=1M  # read 1GB

# Random IOPS.
fio --name=randread --ioengine=libaio --iodepth=32 --rw=randread \
    --bs=4k --direct=1 --size=1G --runtime=10 --filename=/tmp/test

rm /tmp/test
```

For a real ML benchmark, time PyTorch DataLoader iteration:

```python
import time, torch
from torch.utils.data import DataLoader, TensorDataset

# Synthetic dataset on disk.
n_samples = 100000
data = torch.randn(n_samples, 224, 224, 3)
torch.save(data, '/tmp/data.pt')

# Load and iterate.
data = torch.load('/tmp/data.pt')
ds = TensorDataset(data, torch.zeros(n_samples))
dl = DataLoader(ds, batch_size=64, num_workers=4)
t0 = time.time()
for i, (x, y) in enumerate(dl):
    if i >= 100: break
t = time.time() - t0
print(f"100 batches in {t:.2f}s = {100 * 64 / t:.0f} samples/sec")
```

For a real cluster, multiply by the GPU count to see if your storage can keep up.

## Further reading

- "Lustre Filesystem" documentation.
- WekaFS architecture papers.
- "WebDataset" tutorials.
- AWS FSx for Lustre docs; GCP Filestore HSM docs.
- "Storage for AI/ML" — vendor whitepapers (Weka, Vast).

Next lesson: **Cluster networking topology + module wrap.** Fat trees, IB rails, RoCE; rack-aware scheduling; and the module wrap with a worked cluster design.
