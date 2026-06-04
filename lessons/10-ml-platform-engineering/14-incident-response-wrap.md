---
title: "Lesson 14 — Incident Response + Module Wrap"
date: "2026-06-04"
module: "ml-platforms"
order: 14
tags: ["incident-response", "oncall", "rollback", "module-wrap"]
author: "Sudipta Pathak"
prerequisites: ["13-compliance-governance"]
---

# Lesson 14 — Incident Response + Module Wrap

## Why this lesson exists

Production ML systems fail. Models regress; serving infrastructure breaks; data pipelines silently produce garbage. When they fail, someone must respond: diagnose, mitigate, fix, post-mortem.

This lesson covers ML incident response — the patterns from SRE / DevOps adapted for the specific failure modes of ML systems. Then the module wrap.

The lesson is reading. The Hands-on is a tabletop incident-response exercise.

## What "the model is down" can mean

For software services, "down" is unambiguous: requests return errors. For ML systems, "down" is fuzzy:

- **Hard down**: serving system returns errors. Standard service outage.
- **Slow down**: latency spike; requests timeout.
- **Quiet regression**: model's predictions silently get worse. Users notice; metrics drop; no alert fires for hours.
- **Data drift**: input distribution shifts; model performs worse on the new distribution.
- **Concept drift**: the right answer changes (e.g., user preferences shift); the model still predicts the old answer.
- **Adversarial**: someone is actively misusing or attacking the model.

Each requires different detection and response.

## The on-call setup

Like software on-call:
- Engineers rotate on-call for the ML platform.
- Alerts page on-call.
- Runbooks describe response procedures.
- Post-incident review for every meaningful incident.

ML-specific:
- Alerts on model metrics, not just service metrics.
- Runbooks include "how to rollback a model" alongside "how to rollback a deployment."
- Post-incident reviews often surface data or eval-gate gaps.

## The runbook

A runbook for an ML incident typically includes:

```
INCIDENT: Model X production accuracy dropped

LIKELY CAUSES:
- Recently deployed new version with bad eval gate.
- Data pipeline produced bad inputs.
- Feature store stale.
- Real-world distribution shifted.

DETECTION:
- Production accuracy metric below threshold for > 30 minutes.
- User complaint volume up.

DIAGNOSIS STEPS:
1. Check the model version currently serving.
2. Check when it was deployed; what changed.
3. Inspect recent input distribution; compare to training.
4. Check feature store freshness.
5. Check for upstream data pipeline issues.

MITIGATION:
1. If recent deploy: rollback to previous version.
2. If data issue: fail open or use cached/stale features.
3. If sustained drift: trigger retraining.

CONTACT:
- ML platform team for serving issues.
- Data engineering team for pipeline issues.
- Model owner for retraining decisions.
```

The runbook is updated after each incident with lessons learned.

## Rollback as the first reflex

For most production ML incidents, the first action is rollback to the last-known-good model. This:
- Stops the bleeding.
- Buys time to diagnose properly.
- Limits user impact.

Rollback requires:
- The previous model artifact still available (don't delete it immediately on deploy).
- A fast switch mechanism (canary in reverse).
- Confidence that rollback won't make things worse.

For LoRA hot-swap setups (Module 6 Lesson 10), rollback is just swapping the adapter — fast.

For larger architectural changes, rollback is slower; the previous infrastructure may need to be re-stood-up.

## Common ML incident causes

In rough order of frequency:

1. **Data pipeline broke**: training data is wrong; model trained on garbage; production looks wrong.
2. **Eval gate too lenient**: bad model passed the gate; deployed; production regression.
3. **Distribution shift**: real-world data changed; model trained on old distribution underperforms.
4. **Infrastructure issue**: serving fleet under-provisioned; latency spike.
5. **Adversarial**: someone exploiting the model.

Each has different mitigation. The post-incident review should identify the cause and the systemic fix (e.g., "we need stronger eval gates," "we need better drift detection").

## Module wrap

Fourteen lessons. The journey:

**Lessons 1-3 (tracking what you ran)**: experiment tracking, model registry, hyperparameter sweeps. The foundation — knowing what experiments you've done.

**Lessons 4-5 (feeding the beast)**: data infrastructure, checkpoint storage. The pipeline that produces training-ready data and the storage that holds the outputs.

**Lessons 6-7 (watching the run)**: training observability, cost monitoring. Knowing what's happening during the run and what it costs.

**Lessons 8-11 (shipping the run)**: workflow orchestration, CI/CD for models, feature stores, serving infrastructure. The path from "trained model" to "serving production traffic."

**Lessons 12-14 (living with the run)**: A/B testing, compliance, incident response. The operational side.

The synthesis: an ML platform is a coordinated set of services that turn an ML engineer's workflow ("I want to train this model") into a reliable, repeatable, observable process. Each piece is non-trivial individually; the integration is what makes the platform.

## Mental models to carry forward

Five sentences:

**1. Experiment tracking + model registry + lineage** is the bedrock. Without these, you can't reproduce, debug, or improve. Pick a stack (W&B + custom registry, or MLflow end-to-end) and standardize.

**2. The platform's job is to make the easy path the right path**: tagged jobs, eval-gated deploys, lineage tracked automatically. If engineers have to opt in to good behavior, they often won't.

**3. Treat datasets as products**: versioned, owned, documented, tested. Data quality dominates model quality; data infrastructure is half the platform.

**4. Production starts with deployment**, not ends with it: observability, A/B, on-call, incident response. The model is alive in production; treat it like any other service.

**5. Governance integrated into the platform**, not bolted on. Model cards from metadata; lineage at training time; audit logs automatically. Compliance is required for some industries; good engineering anyway.

## What we covered, what we skipped

Covered: the fourteen topics in the roadmap.

Skipped:
- **Specific vendor product reviews** (we mentioned tools; didn't compare in depth).
- **Cloud-managed ML platforms** (SageMaker, Vertex AI, Azure ML) in detail. They package the patterns we covered.
- **Org structure for ML platforms** (centralized vs decentralized; ML platform team's role).
- **Specific security configurations** (image signing, secrets management, network isolation).
- **Specific deep-learning experiment frameworks** beyond what we touched.

## What's next

**Module 11: Agents from Scratch.** The application layer. Tool use, planning, multi-step reasoning, the agent architectures. The other major topic in modern ML systems alongside training and serving.

## End of Module 10

Modules 1-9 built the ML systems stack from silicon up. Module 10 added the platform layer — the operational infrastructure that makes the stack usable day-to-day. With these ten modules, you have the full vertical for single-team ML systems work.

Module 11 picks up at the application layer: how the trained, served, monitored models compose into actual ML-powered products — agents, RAG systems, tool-using assistants. The bulk of ML application engineering in 2026.

## Hands-on (at home)

Tabletop incident-response exercise.

```
SCENARIO:
At 2 PM, your monitoring shows that "model_X" production accuracy has
dropped from 92% to 78% over the last hour. User complaints are starting.

YOUR ON-CALL ROTATION GOT THE PAGE. WHAT DO YOU DO?

Possible immediate actions:
- Roll back to the previous model version (last-known-good was v3.2; current is v3.3).
- Check the data pipeline for upstream issues.
- Check the feature store for staleness.
- Check the input distribution for shift.

WHO DO YOU INVOLVE?
- Model owner (the team that trained v3.3).
- Data engineering (if pipeline suspected).
- Customer support (to communicate to users).

WHAT'S YOUR FIRST 5 MINUTES?

WHAT'S YOUR FIRST 30 MINUTES?

WHEN DO YOU WRITE THE POST-INCIDENT REVIEW?
```

Write out your responses. Compare to a senior engineer's; the difference is often the discipline of "roll back first; diagnose after."

For practice in a real incident: read post-mortems from public sources (CloudFlare, Google SRE book, etc.) and ask "how would this go for an ML system?"

## Further reading

- "Site Reliability Engineering" (Google) — the textbook on SRE.
- "Incident Response for Machine Learning" — various blog posts.
- "Continuous Delivery for Machine Learning" (Sato et al, ThoughtWorks).
- Module 9 Lesson 13 — checkpoint and restart for compute-side recovery.
