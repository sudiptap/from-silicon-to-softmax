---
title: "Lesson 23 — Tracing, Evaluation, and Observability"
date: "2026-06-04"
module: "agents"
order: 23
tags: ["tracing", "evaluation", "observability", "langfuse", "llm-as-judge"]
author: "Sudipta Pathak"
prerequisites: ["22-browser-computer-use"]
---

# Lesson 23 — Tracing, Evaluation, and Observability

## Why this lesson exists

Production agents fail in subtle ways: a tool call returns wrong data; a reasoning step goes off track; an output misses the user's intent. Without observability, you don't know what went wrong; without evaluation, you don't know if you're improving over time.

This lesson covers the production-grade observability for agents: distributed tracing (Langfuse, OpenTelemetry), evaluation strategies (LLM-as-judge, trajectory-level vs outcome-level), and the benchmarks.

The lesson is reading. The Hands-on integrates Langfuse with a simple agent.

## What to observe

For an agent, the observability surface is:

**Per-LLM-call**:
- Model used, prompt tokens, completion tokens.
- Latency.
- Cost.
- Full input + output for debugging.

**Per-tool-call**:
- Tool name, arguments.
- Result.
- Latency.
- Errors.

**Per-agent-run**:
- Total tokens / cost.
- Total steps.
- Success / failure.
- Trajectory (the sequence of LLM calls + tool calls).

**Aggregate metrics**:
- P50, P95, P99 latency.
- Success rate over time.
- Cost per task.
- Token usage trends.

## Distributed tracing

Agents call multiple LLMs + tools; each call is a "span" in the trace. The trace shows the structure:

```
Run: "Book a flight to Paris" [12 seconds, $0.32]
├─ LLM call: plan [0.8s, 1.2K tokens]
├─ Tool: search_flights [3.2s]
├─ LLM call: select flight [0.5s, 0.8K tokens]
├─ Tool: book_flight [2.1s]
├─ LLM call: confirm [0.3s, 0.4K tokens]
└─ Response to user [0.1s]
```

The hierarchy mirrors the agent's structure. Drill into any span to see details.

Tools:
- **Langfuse**: open-source LLM-focused tracing.
- **LangSmith**: hosted; integrated with LangChain.
- **Weights & Biases Traces**: integrated with W&B.
- **OpenTelemetry** (with custom instrumentation): standard distributed-tracing protocol; works with any backend.

For modern agents, Langfuse and LangSmith are dominant. They handle LLM-specific concerns (token counts, prompts, costs) out of the box.

## Integration

Adding Langfuse to an agent:

```python
# pip install langfuse
from langfuse import Langfuse
from langfuse.openai import openai  # langfuse wraps openai

# Initialize.
langfuse = Langfuse(public_key="...", secret_key="...", host="https://cloud.langfuse.com")

# Now use openai as normal; calls are auto-traced.
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"agent_run_id": "abc123"}  # for grouping
)
```

The decorator-based pattern auto-instruments. View traces in the Langfuse UI.

For OpenTelemetry, instrument manually:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent_run") as span:
    span.set_attribute("user_query", user_query)
    response = llm_call(...)
    span.set_attribute("tokens", response.usage.total_tokens)
```

More boilerplate but uses your existing observability stack.

## Evaluating agents

Two evaluation regimes:

**Trajectory eval**: assess the *steps* the agent took. Did it call the right tools? In the right order? Reason correctly?

**Outcome eval**: assess only the *final output*. Did it produce the right answer / complete the task?

For benchmarks (SWE-bench, GAIA, WebArena), outcome is what's measured. For debugging and improvement, trajectory matters too.

Different approaches:

**Human eval**: gold standard; expensive. Used for benchmark labeling and ongoing spot-checks.

**Programmatic eval**: deterministic checks. Tests pass / fail for coding; exact-match for QA. Cheap; precise.

**LLM-as-judge**: an LLM evaluates the output. Subjective metrics (helpfulness, faithfulness). Cheap, scales; biased.

**Replay eval**: re-run failed trajectories with debugging; understand what went wrong.

## LLM-as-judge

The most common automated evaluation for subjective tasks:

```python
def llm_judge(question, answer, criteria):
    prompt = f"""Evaluate this answer.

Question: {question}
Answer: {answer}

Criteria:
{criteria}

Rate on each criterion 1-10. Provide a brief justification."""
    
    return llm.complete(prompt)
```

Common criteria:
- **Helpfulness**: addresses the question.
- **Faithfulness**: claims are supported by context (for RAG).
- **Accuracy**: factually correct.
- **Conciseness**: not verbose.
- **Safety**: no harmful content.

The judge's verdict is averaged across many examples to compute aggregate quality.

Caveats (revisit from Lesson 14):
- Judge has the same biases as generator; use a different model for the judge.
- Judge's calibration is variable; standardize via rubrics.

## Benchmarks

For agents:

**SWE-bench**: real GitHub issues; the agent must produce a PR that fixes the issue and passes tests. Coding-agent benchmark.

**GAIA**: multi-step Q&A tasks requiring tool use and reasoning. General-agent benchmark.

**WebArena**: realistic web tasks.

**OSWorld**: realistic desktop tasks.

**MLR-Bench**: ML research tasks.

Each has its leaderboard; top systems are state-of-the-art agent products.

## RAG-specific evaluation

For RAG (Lessons 9-12), specific metrics:

**Context precision**: of the retrieved chunks, what fraction are relevant?

**Context recall**: of the relevant info in the corpus, what fraction was retrieved?

**Faithfulness**: do the answers actually use the retrieved context (vs hallucinating)?

**Answer relevance**: does the answer address the question?

Tools like RAGAs (Retrieval-Augmented Generation Assessment) automate these.

## A practical eval pipeline

For a production agent:

```
1. Collect a test set of representative queries (50-500).
2. Run the agent on each; record trajectories + outputs.
3. Run automated eval (programmatic + LLM-as-judge).
4. Spot-check by humans (random sample).
5. Aggregate metrics; track over time.
6. Alert on regressions.
```

The test set is a living artifact; expand it when failures occur in production.

The CI integration: run the eval on every meaningful change; gate deploys on eval not regressing.

## What you should believe after this lesson

Three sentences:

**1. Production agents need distributed tracing** — every LLM call and tool call is a span; the hierarchy mirrors agent structure. Langfuse, LangSmith, OpenTelemetry are the standard tools.

**2. Evaluation has two regimes** — trajectory (assess the steps) and outcome (assess the final result). For benchmarks, outcome is measured; for debugging, trajectory matters. LLM-as-judge automates subjective evaluation but has known biases.

**3. Production eval pipelines combine programmatic checks + LLM-as-judge + human spot-checks** on a representative test set. Run on every change; track over time; gate deploys on no regression.

## Hands-on (at home)

Integrate Langfuse with a simple agent.

```python
# langfuse_agent.py
# pip install langfuse openai

# Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in env.

from langfuse.openai import openai  # wraps standard openai
from langfuse.decorators import observe

@observe()
def my_agent(question):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

result = my_agent("What is the capital of France?")
print(result)

# Visit your Langfuse dashboard to see the trace.
```

Each call is traced with tokens, latency, cost. For a multi-step agent, the trace shows the full sequence.

For LLM-as-judge eval:

```python
def evaluate_answer(question, answer):
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this answer on a scale 1-10 for helpfulness:

Question: {question}
Answer: {answer}

Score (just the number):"""
        }]
    ).choices[0].message.content
```

Run on your agent's outputs; aggregate.

## Further reading

- Langfuse documentation.
- LangSmith documentation.
- "LLM as Judge" best practices.
- "RAGAs: Automated Evaluation of Retrieval Augmented Generation" (Es et al, 2023).
- SWE-bench leaderboard.

Next lesson: **Distributed agent scheduling + module wrap.** The final lesson: production scheduling, queue-backed workers, durable state. Then the wrap of Module 11 and the entire curriculum.
