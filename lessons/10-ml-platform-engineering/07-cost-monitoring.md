---
title: "Lesson 7 — Cost Monitoring"
date: "2026-06-04"
module: "ml-platforms"
order: 7
tags: ["cost", "attribution", "finops", "showback", "chargeback"]
author: "Sudipta Pathak"
prerequisites: ["06-training-observability"]
---

# Lesson 7 — Cost Monitoring

## Why this lesson exists

ML compute is expensive. A 1024-GPU training run at $5/GPU/hour for 30 days is $3.7M. Without per-job, per-team cost attribution, the bill arrives at the end of the month and nobody knows what drove it.

This lesson covers cost monitoring: tagging jobs with cost-centers, attributing cluster spend, the showback vs chargeback distinction, and the practical tools.

The lesson is reading. The Hands-on builds a simple cost attribution dashboard.

## The pieces of cost

For an ML cluster, total cost = sum of:

- **GPU compute hours**: the headline number. ~$2-5/hour per H100 on-demand cloud; ~$1-2 spot; ~$0.5-1 amortized on bare metal.
- **Storage**: TB/month for object store + parallel FS.
- **Networking**: cross-region or egress charges (can be substantial).
- **Other compute**: CPU nodes, control plane.

Each piece has different attribution challenges.

## Tagging

The foundation: every resource is tagged with attribution metadata.

For K8s pods:
```yaml
metadata:
  labels:
    team: research
    project: llama-finetune
    experiment-id: exp-1234
    user: alice@example.com
    cost-center: cc-100
```

For cloud resources (VMs, buckets):
- AWS: resource tags.
- GCP: labels.
- Azure: tags.

These tags propagate into billing reports. Without consistent tagging, cost attribution is impossible.

The discipline: enforce tagging via admission policies. Pods without required tags get rejected at submission time.

## Per-job cost computation

For a job running on K8s:
- Duration: from start_time to end_time.
- Resources: GPU count, CPU, memory.
- Hourly rate: from a price-list.

Job cost = duration × resources × hourly_rate.

For a 32-GPU job that ran for 5 hours at $4/GPU/hour: $640.

Tools:
- **Kubecost**: Kubernetes-native cost attribution. Aggregates from K8s metrics + cloud billing.
- **OpenCost** (Kubecost's open-source core): same idea.
- **Cloud provider tools**: AWS Cost Explorer, GCP Billing, Azure Cost Management.
- **Custom**: many large orgs build their own to integrate with internal accounting.

## Showback vs chargeback

Two operational modes:

**Showback**: report cost per team / project. Inform but don't enforce. Teams see "you spent $50K last month" but no internal money changes hands.

**Chargeback**: cross-charge cost between teams. The cluster runs on team-A's budget; if team-B uses it, team-A bills team-B.

Showback is easier and more common; introduces no internal politics. Chargeback is stricter; sometimes required by larger orgs' accounting practices.

Both require accurate cost attribution; the difference is what's done with the data.

## The unit economics question

ML cost monitoring often connects to product-level economics: "what's the cost per inference request?" "What's the cost per training run that produces a deployed model?" "What's the cost per A/B test winner?"

These need cost attribution + the business context. Tools like Kubecost can report at the namespace / pod level; the integration with business metrics is usually custom.

## Common cost surprises

Things that surprise people:

**Egress fees**: moving data out of a cloud region or out of the cloud entirely is expensive. A multi-region deployment can bleed money on egress.

**Idle resources**: dev clusters that nobody is using; persistent storage that nobody reads. These are pure waste.

**Over-provisioning**: requesting more GPUs than the job actually uses. The cluster reserves them; they're billed even if idle.

**Failed jobs**: a job that crashes after 4 hours still cost money. Aggregate over many failed runs is non-trivial.

**Forgotten resources**: PVCs left over from finished jobs; snapshot proliferation. Garbage collection is essential.

## The "experiment cost" question

A meaningful platform feature: each tracked experiment (Lesson 1) shows its cost.

Pattern:
1. Experiment tracking system stores run metadata.
2. Cost system computes cost per K8s pod / job.
3. Integration links: experiment_id → cost.

Result: each run in the W&B / MLflow UI shows its cost. Easier to see "this 100-run sweep cost $10K; was it worth it?"

Tools that integrate ML tracking with cost are emerging (Comet, MLflow + Kubecost integration); few do it well out of the box.

## What you should believe after this lesson

Three sentences:

**1. Cost monitoring requires consistent tagging** at the pod / resource level (team, project, experiment, user, cost-center). Without tags, the bill arrives and nobody knows what drove it; with tags, you can attribute spend to teams and projects.

**2. Per-job cost = duration × resources × hourly rate**; tools like Kubecost / OpenCost compute this automatically. Showback (report) vs chargeback (cross-charge) are the two operational modes; showback is more common and less politically fraught.

**3. Common cost surprises** are egress, idle resources, over-provisioning, failed jobs, forgotten resources. Regular cleanup, sane defaults, and dashboards that surface waste prevent the worst.

## Hands-on (at home)

Build a simple cost-per-job report.

```python
# cost_per_job.py
# Run on a K8s cluster with metrics-server installed.
import subprocess
import json
from datetime import datetime

# Get all pods with their labels and timings.
result = subprocess.run(['kubectl', 'get', 'pods', '-A', '-o', 'json'], capture_output=True, text=True)
pods = json.loads(result.stdout)['items']

# Toy hourly rates.
GPU_RATE = 4.0  # $/GPU/hour
CPU_RATE = 0.05  # $/CPU/hour
MEM_RATE = 0.01  # $/GB-RAM/hour

team_costs = {}
for pod in pods:
    if pod['status']['phase'] not in ['Succeeded', 'Running']:
        continue
    
    team = pod['metadata'].get('labels', {}).get('team', 'unassigned')
    
    # Compute runtime.
    start_str = pod['status'].get('startTime')
    if not start_str:
        continue
    start = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.utcnow()  # for running pods; for completed, would use finish time
    hours = (end - start).total_seconds() / 3600
    
    # Compute resource cost.
    cost = 0
    for container in pod['spec']['containers']:
        resources = container.get('resources', {}).get('limits', {})
        gpu = int(resources.get('nvidia.com/gpu', 0))
        cpu = float(resources.get('cpu', '0').replace('m', 'e-3') or 0)
        mem_str = resources.get('memory', '0')
        # Parse memory (simplified).
        if 'Gi' in mem_str:
            mem = float(mem_str.replace('Gi', ''))
        else:
            mem = 0
        cost += (gpu * GPU_RATE + cpu * CPU_RATE + mem * MEM_RATE) * hours
    
    team_costs[team] = team_costs.get(team, 0) + cost

print("Team costs (last hour):")
for team, cost in sorted(team_costs.items(), key=lambda x: -x[1]):
    print(f"  {team}: ${cost:.2f}")
```

This is conceptual; production tools like Kubecost handle the edge cases (spot vs on-demand pricing, networking costs, storage attribution) more rigorously.

## Further reading

- Kubecost / OpenCost documentation.
- AWS Cost Explorer documentation.
- FinOps Foundation materials.
- "Cloud FinOps" book (J. R. Storment & Mike Fuller).

Next lesson: **Workflow orchestration.** Argo Workflows, Flyte, Kubeflow Pipelines. Chaining train → eval → deploy as a coordinated workflow.
