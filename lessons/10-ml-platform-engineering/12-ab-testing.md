---
title: "Lesson 12 — A/B Testing Infrastructure"
date: "2026-06-04"
module: "ml-platforms"
order: 12
tags: ["ab-testing", "experimentation", "statistical-rigor", "attribution", "online-evaluation"]
author: "Sudipta Pathak"
prerequisites: ["11-model-serving-infra"]
---

# Lesson 12 — A/B Testing Infrastructure

## Why this lesson exists

Offline evaluation (Lesson 9) says "this model's accuracy on the eval set is X." That's necessary but not sufficient. The real question is "in production, does this model improve the business metric we care about?" — click-through rate, conversion, user retention, average session length, etc.

A/B testing answers this: route some users to the new model, some to the old, measure the business metric over time, conclude whether the new model is better.

This lesson covers the infrastructure: traffic splitting, statistical rigor, attribution, and the patterns that produce trustworthy results.

The lesson is reading. The Hands-on builds a simple A/B test simulator.

## The basic setup

For a new model variant:

1. **Hash users into buckets**: each user gets a stable bucket assignment based on a hash of their user_id. 50% in bucket A (control), 50% in bucket B (treatment).
2. **Route requests by bucket**: bucket A users go to model_old; bucket B to model_new.
3. **Measure business metric**: per-user, per-day, the metric you care about (e.g., clicks_per_session).
4. **Run for sufficient time**: enough data to be statistically significant.
5. **Compare**: is the bucket B average significantly different from bucket A?

The serving infrastructure (Lesson 11) supports the routing; the analytics infrastructure supports the metric tracking.

## The hashing trick

Why hash user_id (instead of random per-request)?
- Each user always sees the same model. Their experience is consistent.
- Measurement is per-user, not per-request. Avoids contamination.

Hash function: typically MD5 of (user_id + experiment_name). The first few bits determine bucket.

The experiment name is crucial — two simultaneous experiments shouldn't share bucket assignments (else they're confounded). Including the experiment name in the hash makes each experiment independent.

## Statistical rigor

The trap of A/B testing: you ran for a day, the new model looks 5% better, you ship it. Then over the next week the apparent improvement disappears.

The cause: noise. Day-to-day variance in user behavior is high; a 5% difference in one day could easily be noise.

Statistical rigor:
- **Pre-register**: decide the metric, the duration, and the success threshold *before* running.
- **Use a proper test**: t-test or non-parametric equivalent; compute p-value.
- **Account for multiple comparisons**: if you're testing many metrics, the false-positive rate goes up. Bonferroni or similar correction.
- **Run for enough time**: a week is typical; longer for low-traffic metrics.

For LLM-relevant business metrics (helpfulness ratings, retention, paid-conversion), the right test depends on the metric's distribution.

## Sequential testing (peeking problem)

"Peeking" at A/B test results before the test ends is dangerous. If you stop the test as soon as it looks significant, you've biased the result. (Classical statistics assumes fixed sample size; early stopping inflates false positives.)

Solutions:
- **Don't peek**. Commit to a duration up front.
- **Sequential testing**: use statistical methods that account for early stopping (e.g., always-valid p-values).

Some experimentation platforms (Eppo, Optimizely, internal tools at Meta/Google) implement sequential testing properly.

## Network effects and interference

Standard A/B testing assumes user A's experience doesn't affect user B's. This fails for:

- **Social networks**: if A and B are friends, A's behavior changes B's. A in treatment + B in control means B is partially treated.
- **Marketplaces**: changing one user's experience can affect supply for other users.
- **LLM chat with shared context**: less common; some agent setups.

For these, cluster-randomized A/B tests (assign groups to buckets, not individuals) are needed. More complex; lower statistical power.

## Power analysis

Before running a test, you should know: how much sample size do I need to detect a meaningful effect?

The math:
- Sample size depends on: baseline metric variance, effect size you want to detect, statistical power desired, significance threshold.
- A typical setup: 95% confidence, 80% power, detect a 1% relative change in a 5% baseline metric → need ~10K samples per arm.

For high-variance metrics, much more. For low-traffic apps, A/B tests can take weeks.

If the math says you can't get enough sample size in a reasonable time: do offline evaluation instead, or design a more sensitive metric.

## LLM-specific A/B challenges

For LLMs, A/B testing has extra wrinkles:

**Quality is hard to measure**: clicks and conversions don't capture "did the chatbot give a good answer." Need explicit user ratings or judge-model evaluation.

**Long-tail behavior matters**: the average might be the same, but the new model fails worse on rare cases. Aggregate metrics miss this.

**Outputs are conditional on prompt**: the same user might submit different prompts to A vs B (due to randomness in their behavior). This complicates attribution.

**Cost**: running two LLM versions in parallel doubles the inference cost. Some orgs limit A/B tests to small traffic fractions.

The right metric design for LLM A/B is an active area. User satisfaction scores, retention, conversion (for product LLMs) are the standards.

## Tools

Major options:

**Custom**: many large orgs build their own. The surface area is moderate; the customization is valuable.

**Statsig, Eppo, LaunchDarkly**: SaaS experimentation platforms. Provide bucketing, metric tracking, statistical analysis.

**Optimizely, Adobe Target**: marketing-experimentation tools; sometimes used for ML A/B.

**Google Optimize / Vertex AI Experiments**: Google's offerings.

For ML A/B specifically, Statsig and Eppo are increasingly popular; they're designed for product experimentation rather than just web experiments.

## What you should believe after this lesson

Three sentences:

**1. A/B testing is the production validation that offline eval can't replace** — measures actual business metrics on real user behavior. Standard setup: hash users into buckets, route requests, compare metrics over a pre-registered duration.

**2. Statistical rigor matters**: pre-register, run for enough time, don't peek, account for multiple comparisons. Sequential testing methods handle the peeking problem properly.

**3. LLM A/B is harder than classical ML A/B** — quality metrics are subjective; long-tail behavior matters; cost is high. The metric design (user ratings, retention, satisfaction) is more important than the statistical mechanics.

## Hands-on (at home)

Simulate an A/B test.

```python
# ab_test_sim.py
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate baseline (model A): conversion rate 5%.
n_per_arm = 5000
A = np.random.binomial(1, 0.05, n_per_arm)

# Simulate variant (model B): conversion rate 5.5% (a 10% relative improvement).
B = np.random.binomial(1, 0.055, n_per_arm)

print(f"Arm A: {A.sum()} conversions / {n_per_arm} = {A.mean()*100:.2f}%")
print(f"Arm B: {B.sum()} conversions / {n_per_arm} = {B.mean()*100:.2f}%")
print(f"Relative lift: {(B.mean() - A.mean()) / A.mean() * 100:.1f}%")

# Statistical test (z-test for proportions).
from scipy.stats import proportions_ztest
stat, pval = proportions_ztest([A.sum(), B.sum()], [n_per_arm, n_per_arm])
print(f"Z-statistic: {stat:.3f}, p-value: {pval:.4f}")
print(f"Significant at α=0.05: {pval < 0.05}")
```

Try with different sample sizes; you'll see that 5000 per arm is borderline for detecting a 0.5% absolute / 10% relative effect at 5% baseline. With 50000, the test would be highly significant; with 500, it'd be inconclusive.

For real production, you'd use Statsig / Eppo / your custom platform; the statistical mechanics is the same.

## Further reading

- "Trustworthy Online Controlled Experiments" (Kohavi, Tang, Xu) — the textbook on A/B testing.
- Statsig and Eppo documentation.
- "Sequential A/B Testing" (Johari et al, 2017).
- "Peeking at A/B Tests: Why it Matters, and What to Do About It" (Johari et al).

Next lesson: **Compliance & governance.** Model cards, lineage tracking, audit trails — the boring stuff that matters in regulated industries.
