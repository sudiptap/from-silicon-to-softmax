---
title: "Lesson 14 — Self-Reflection, Critic Loops, and Verifier Agents"
date: "2026-06-04"
module: "agents"
order: 14
tags: ["self-reflection", "critic", "verifier", "do-vs-check", "review"]
author: "Sudipta Pathak"
prerequisites: ["13-plan-execute-tot"]
---

# Lesson 14 — Self-Reflection, Critic Loops, and Verifier Agents

## Why this lesson exists

Generating an answer and reviewing it are different cognitive tasks. The model that generates code is often the same model that should review the code — but giving it both roles in one prompt produces worse results than giving each role separately.

This lesson covers the patterns: self-reflection (the same model critiquing its own output), critic loops (iterative critique-and-revise), verifier agents (a separate agent dedicated to checking).

The lesson is reading. The Hands-on implements a critic loop.

## The "do vs check" insight

When the model is told "write a Python function," it focuses on writing. When told "find bugs in this Python function," it focuses on checking. Asking it to both write AND check in one prompt diffuses focus; the output is worse than either alone.

The pattern: separate the prompts.

```
1. WRITE prompt: "Write a function to compute factorial."
2. CHECK prompt: "Review this function for bugs: [the function]. List any issues."
3. REVISE prompt: "Revise the function based on this feedback: [issues]."
```

Three LLM calls, but the per-call output is better focused. The combined result beats a single-shot generation.

## Self-reflection (Reflexion)

Reflexion (Shinn et al, 2023): the agent generates an attempt; reflects on it (criticizes itself); tries again with the reflection in context.

```python
def reflexion_agent(task, max_attempts=3):
    history = []
    for attempt in range(max_attempts):
        # Generate.
        prompt = f"Task: {task}\n\nPrevious attempts (with critiques):\n{history}\n\nAttempt:"
        attempt_output = llm(prompt)
        
        # Try it (run code, check answer, etc.).
        result = evaluate(attempt_output)
        if result.success:
            return attempt_output
        
        # Reflect.
        reflection_prompt = f"""Task: {task}
Attempt: {attempt_output}
Result: {result.error}

What went wrong? What should you do differently?"""
        reflection = llm(reflection_prompt)
        history.append({"attempt": attempt_output, "reflection": reflection})
    
    return attempt_output  # final attempt, even if not successful
```

The agent learns from each failure within a single task. Across attempts, the reflection helps the agent avoid repeating the same mistake.

Reflexion's gains: 5-15% on coding benchmarks, 10-30% on harder reasoning tasks.

## Critic loops

A more structured version: an explicit critic agent reviews the output.

```python
def critic_loop(task, max_iterations=3):
    output = generator_agent(task)
    
    for i in range(max_iterations):
        critique = critic_agent(task, output)
        if "approved" in critique.lower():
            return output
        output = revisor_agent(task, output, critique)
    
    return output
```

Three roles:
- **Generator**: produces the initial output.
- **Critic**: reviews; suggests improvements.
- **Revisor**: applies the improvements.

All three can be the same LLM with different prompts (cheaper) or different LLMs (more robust but more expensive).

## Verifier agents

For tasks with clear correctness criteria (code passes tests, math has a checkable answer), a *verifier* is a separate model dedicated to checking.

```python
def verified_generation(task, max_attempts=5):
    for _ in range(max_attempts):
        candidate = generator(task)
        if verifier(task, candidate).is_correct:
            return candidate
    return None  # failed
```

The verifier can be:
- **Deterministic**: a test suite for code; a math evaluator for arithmetic.
- **LLM-based**: another LLM that scores the output.
- **Hybrid**: deterministic where possible; LLM for fuzzy criteria.

For coding agents (Lesson 21), the verifier is often the test suite — generate code; run tests; if pass, done; if fail, revise.

## When critique helps, when it hurts

The patterns help when:
- **The output has clear correctness criteria** (tests, math, structured output).
- **The model can plausibly catch its mistakes** (the model knows what good output looks like).
- **The cost of multiple LLM calls is acceptable**.

They hurt or don't help when:
- **The output is subjective** (creative writing). The critic doesn't know what's "right."
- **The model's blind spots are systematic** (it can't catch its own mistakes). The critic might endorse wrong answers.
- **Latency is critical**. Multiple LLM calls add seconds.

## The "auto-eval" hazard

A specific concern: using LLM-as-judge to evaluate LLM-as-generator is contaminated. The judge has the same biases as the generator; they might agree on wrong answers.

Mitigations:
- **Different model**: use Claude as judge for GPT outputs (or vice versa).
- **Structured criteria**: instead of "is this good?", ask "does it satisfy criterion X?" Specific questions are easier to evaluate.
- **Ground truth where possible**: deterministic checks (tests, math) beat LLM critiques.

For research benchmarks, LLM-as-judge is widely used (despite the issues) because it scales. For production, prefer deterministic checks where possible.

## Constitutional AI

A specific application: a model self-critiques based on a list of principles (a "constitution"):

```
Constitution:
1. Don't generate harmful content.
2. Don't give financial advice.
3. Don't help with illegal activities.

Generate the response.
Critique: does the response violate any constitutional principle?
Revise: rewrite to comply.
```

This is part of Anthropic's training process for Claude; runtime variants (apply at inference) are also possible.

## Verifier-augmented inference

A more involved pattern: a separate small "verifier" model is trained to predict whether the generator's output is correct. At inference:
1. Generator produces N candidates.
2. Verifier scores each.
3. Return the highest-scoring.

For math (PRM = Process Reward Model), code, and other verifiable domains, verifier-augmented inference has shown strong results (OpenAI's PRM work).

The cost: extra training for the verifier; per-inference cost of N × generator + N × verifier.

## What you should believe after this lesson

Three sentences:

**1. Separating "do" from "check" improves quality** — generator focuses on producing; critic focuses on finding mistakes; revisor applies fixes. Costs more LLM calls; often worth it on complex tasks.

**2. Self-reflection (Reflexion)** is the lightweight in-loop pattern — agent attempts, reflects on failure, tries again. Critic loops are more structured (separate generator/critic/revisor). Verifier agents are the strongest pattern when correctness is checkable.

**3. The auto-eval hazard**: LLM-as-judge has the same biases as LLM-as-generator. Mitigate with different models, structured criteria, or ground-truth checks where possible.

## Hands-on (at home)

Implement a critic loop for code generation.

```python
# critic_loop.py
from openai import OpenAI
client = OpenAI()

def generator(task):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write Python code for: {task}\n\nOnly the code, no explanation."}]
    ).choices[0].message.content

def critic(task, code):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Task: {task}
Code:
{code}

Review this code. List any bugs, edge cases missed, or improvements. If the code is correct and clean, respond with "APPROVED"."""}]
    ).choices[0].message.content

def revisor(task, code, critique):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Task: {task}
Code:
{code}
Critique:
{critique}

Revise the code based on the critique. Only the revised code."""}]
    ).choices[0].message.content

def critic_loop(task, max_iterations=3):
    code = generator(task)
    print(f"Initial:\n{code}\n")
    
    for i in range(max_iterations):
        crit = critic(task, code)
        print(f"Critique {i}: {crit}\n")
        if "APPROVED" in crit:
            return code
        code = revisor(task, code, crit)
        print(f"Revised:\n{code}\n")
    
    return code

final = critic_loop("Write a function that returns the nth Fibonacci number.")
print(f"FINAL:\n{final}")
```

You should see the initial output, the critique catching any issues (off-by-one, missing edge case), the revision fixing them, and approval.

## Further reading

- "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al, 2023).
- "Constitutional AI" (Bai et al, 2022).
- "Let's Verify Step by Step" (Lightman et al, OpenAI, 2023) — process reward models.

End of Part 5. Next: Part 6 begins with **Context windows as a resource** — treating context like RAM you have to budget.
