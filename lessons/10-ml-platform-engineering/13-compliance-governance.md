---
title: "Lesson 13 — Compliance and Governance"
date: "2026-06-04"
module: "ml-platforms"
order: 13
tags: ["compliance", "governance", "model-cards", "audit", "regulated-industries"]
author: "Sudipta Pathak"
prerequisites: ["12-ab-testing"]
---

# Lesson 13 — Compliance and Governance

## Why this lesson exists

For ML in regulated industries (healthcare, finance, hiring, lending, education), compliance isn't optional: regulators require documentation, audit trails, fairness assessments, explainability.

Even outside regulation, internal governance — model cards, lineage tracking, approval workflows — is necessary at scale to know what's deployed, by whom, and why.

This lesson covers the standard governance patterns: model cards, lineage, audit trails, the EU AI Act and related regulations.

The lesson is reading. The Hands-on writes a model card.

## Model cards

A *model card* (Mitchell et al, 2018) is a structured document describing a model: its intended use, its limitations, its training data, its evaluation results, its known biases.

A typical model card includes:

- **Model details**: name, version, architecture, training framework.
- **Intended use**: what tasks the model is designed for; what tasks it shouldn't be used for.
- **Training data**: dataset description, sources, sizes, preprocessing.
- **Evaluation**: performance on various test sets and subgroups; known failure modes.
- **Limitations and bias**: known weaknesses, fairness assessments across subgroups.
- **Ethical considerations**: misuse risks, safety mitigations.
- **Contact**: who maintains this model.

The HuggingFace `README.md` template is the de-facto standard for open-source model cards.

For internal models, the org has its own template. Required fields differ by industry: healthcare requires patient-population descriptions; finance requires fairness across demographics; etc.

## Lineage tracking

Lineage: which data + code + hyperparameters produced this model.

For regulated deployments, lineage must be:
- **Complete**: every dependency captured.
- **Reproducible**: with the lineage, the model can be rebuilt.
- **Auditable**: trail is queryable post-hoc.

Tools:
- **MLflow lineage**: links models to runs to datasets.
- **DVC**: data version control for the dataset side.
- **Pachyderm**: data lineage at scale.
- **Custom**: many orgs have internal systems.

The bar: if a regulator asks "show me everything that went into this production model," you should produce it.

## Audit trails

Every model-related action gets logged:
- Who submitted this training run.
- Who approved this model for production.
- When was the model deployed.
- Who accessed the model's outputs.
- Any modifications to the deployment configuration.

The trail typically lives in an immutable log (append-only); accessible to compliance auditors; retained for years.

For high-stakes deployments (medical diagnosis), the audit trail might also include each individual prediction the model made (input → output → user action). Privacy considerations apply.

## Bias and fairness

Regulatory requirements often include fairness assessments:
- Performance metrics broken down by demographic subgroups (race, gender, age).
- Comparable false-positive and false-negative rates across groups.
- Mitigation strategies if disparities exist.

Tools:
- **Fairlearn**: open-source fairness toolkit.
- **Aequitas**: bias audit toolkit.
- **IBM AI Fairness 360**: comprehensive fairness library.

The metrics: demographic parity, equalized odds, predictive parity. Different metrics conflict (you can't achieve all at once); the choice depends on the use case.

For LLMs specifically, fairness assessments include:
- Performance across languages and dialects.
- Stereotyping in outputs (does the model associate professions with genders?).
- Safety across demographic groups (does the model refuse some queries from some groups more than others?).

## Explainability

For some regulated decisions, "the model said no" isn't acceptable; you need a human-interpretable reason.

Tools and techniques:
- **SHAP, LIME**: feature attribution for tree-based and tabular models.
- **Integrated Gradients**: attribution for neural networks.
- **Attention visualization**: for transformers; somewhat interpretive.
- **Counterfactual explanations**: "if X had been different, the decision would have flipped."

For LLMs, explainability is open research. "Why did the model say this?" is hard to answer mechanistically.

## The EU AI Act

The EU AI Act (effective 2025-2026) classifies AI systems by risk:

- **Unacceptable risk** (banned): social scoring, real-time biometric ID in public.
- **High risk**: hiring, lending, medical, education. Requires extensive documentation, human oversight, conformity assessment.
- **Limited risk**: chatbots, content generation. Transparency obligations (must disclose AI-generated content).
- **Minimal risk**: most other AI applications.

For high-risk systems, the act requires:
- Risk assessment and management.
- Data governance and training data documentation.
- Technical documentation including model cards.
- Logging and traceability.
- Human oversight.
- Accuracy, robustness, cybersecurity.

The compliance burden is significant. Organizations operating in EU markets need to plan for this; some are choosing to limit high-risk-system deployments to non-EU markets.

Similar regulations are emerging in other jurisdictions (NYC's AI hiring law, California, federal US frameworks, UK).

## US sector-specific regulations

- **HIPAA**: healthcare data privacy. Models trained on patient data have requirements around de-identification, access control, audit.
- **Fair Credit Reporting Act / Equal Credit Opportunity Act**: lending decisions. Models used in credit must demonstrate fairness.
- **EEOC guidance**: hiring AI. Adverse impact must be measured and mitigated.

Each industry has its own compliance landscape.

## Anti-patterns

**Treating governance as paperwork**: a model card is only useful if it's accurate and current. Treat it as part of the deployment pipeline; auto-generate from metadata; don't let it drift.

**Skipping fairness assessment**: even outside regulation, deploying a model with severe demographic disparities is a reputational and ethical risk.

**Lineage as an afterthought**: hard to reconstruct post-hoc. Capture at training time; store with the model.

**No human oversight for high-stakes decisions**: even with good models, automated decision-making for high-stakes (criminal sentencing, medical, hiring) should have human review.

## What you should believe after this lesson

Three sentences:

**1. Compliance and governance are required for ML in regulated industries** (healthcare, finance, hiring, EU AI Act). Standard practices: model cards, lineage tracking, audit trails, fairness assessments, explainability.

**2. The EU AI Act (2025-2026)** classifies AI by risk; high-risk systems require extensive documentation, human oversight, and conformity assessment. Similar regulations are emerging elsewhere. Plan for it if deploying to EU markets.

**3. Governance shouldn't be paperwork** — integrate into the platform. Auto-generate model cards from metadata; capture lineage at training time; log audit events automatically. Treat governance as code, not as a separate compliance silo.

## Hands-on (at home)

Write a model card for a small model.

```markdown
# Model Card: Iris Classifier v1.0

## Model Details

- **Developed by**: ML Platform team, Example Corp.
- **Model type**: Random Forest classifier.
- **Language**: N/A (tabular).
- **License**: Apache 2.0.
- **Architecture**: 100 decision trees.
- **Training framework**: scikit-learn 1.3.

## Intended Use

- **Primary use**: classify iris flowers by species based on sepal/petal measurements.
- **Intended users**: educational examples; not for production decisions.
- **Out-of-scope uses**: classifying other plant species; any production decision-making.

## Training Data

- **Dataset**: scikit-learn's built-in iris dataset (150 samples; Fisher 1936).
- **Features**: sepal length/width, petal length/width.
- **Classes**: setosa, versicolor, virginica.
- **Preprocessing**: none.

## Evaluation

- **Test set**: 30 samples held out.
- **Accuracy**: 0.97.
- **Per-class precision**: setosa 1.00, versicolor 0.94, virginica 0.97.

## Limitations

- **Small dataset**: 150 samples; results may not generalize to other iris varieties.
- **Limited classes**: only the 3 species in the original dataset.
- **No uncertainty estimates**: returns point predictions only.

## Ethical Considerations

- No personal data; no fairness considerations apply.
- This is an educational model; do not use for any consequential decision.

## Contact

- maintainer@example.com

## Lineage

- **Training run**: mlflow run ID `abc123`.
- **Code commit**: git SHA `def456`.
- **Dataset version**: sklearn version 1.3, internal dataset hash `xyz789`.
```

For larger models, the card is more substantial. HuggingFace's model card template gives you a starting point.

For real compliance, the model card is generated automatically from your tracking and registry systems.

## Further reading

- "Model Cards for Model Reporting" (Mitchell et al, 2018).
- HuggingFace Model Card guidelines.
- EU AI Act text.
- "Fairlearn" documentation.
- "Interpretable Machine Learning" (Christoph Molnar) — book on explainability.

Next lesson: **Incident response + module wrap.** What "the model is down" means; how to debug; who pages whom. Plus the module wrap.
