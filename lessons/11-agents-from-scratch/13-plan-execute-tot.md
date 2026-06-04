---
title: "Lesson 13 — Plan-and-Execute and Tree of Thoughts"
date: "2026-06-04"
module: "agents"
order: 13
tags: ["plan-and-execute", "tree-of-thoughts", "tot", "planning", "lats"]
author: "Sudipta Pathak"
prerequisites: ["12-agentic-rag"]
---

# Lesson 13 — Plan-and-Execute and Tree of Thoughts

## Why this lesson exists

ReAct interleaves reasoning and action one step at a time. For complex multi-step tasks, this can be inefficient (the agent re-reasons everything every step) or wrong (the agent makes a locally-good but globally-bad choice).

This lesson covers two alternatives:

**Plan-and-execute**: separate planning from execution. First, generate a complete plan; then execute step by step.

**Tree of Thoughts (ToT)**: explore multiple reasoning paths in parallel; pick the best.

These are tools in the agent designer's kit. When the task is suitable, they dramatically improve quality.

The lesson is reading. The Hands-on implements a small ToT for a math problem.

## Plan-and-execute

Two phases:

**Plan phase**: the LLM generates a structured plan.
```
1. Search for X.
2. Read the top result.
3. Search for Y.
4. Compare X and Y.
5. Return the comparison.
```

**Execute phase**: an executor runs each step. The executor can be the same LLM (in worker mode) or a different one.

```python
def plan_and_execute(question):
    # Plan.
    plan_prompt = f"Generate a step-by-step plan to answer:\n{question}\n\nPlan:"
    plan_response = llm.complete(plan_prompt)
    steps = parse_steps(plan_response)
    
    # Execute.
    results = []
    for step in steps:
        result = execute_step(step, context=results)
        results.append(result)
    
    return synthesize(results)
```

Pros:
- One reasoning call upfront; cheap to execute.
- The plan is auditable; you can see the structure.
- Parallel execution where steps are independent.

Cons:
- Plan can be wrong; the agent doesn't adapt mid-execution.
- Mitigation: re-plan on failure (Lesson 3).

When to use: tasks with a clear structure that's predictable upfront.

## Tree of Thoughts

For hard reasoning tasks (math, puzzles, planning), ReAct's single-path reasoning makes one mistake and the whole thing goes wrong.

Tree of Thoughts (Yao et al, 2023): explore multiple reasoning branches; evaluate each; pick the best.

The algorithm:
1. Generate multiple candidate "thoughts" for the next step.
2. Evaluate each (LLM scores; or a verifier).
3. Keep the best k; expand each.
4. Continue until done or budget exhausted.
5. Pick the best terminal path.

```python
def tree_of_thoughts(question, depth=3, branches=3):
    # Each node is a partial reasoning path.
    root = {"path": [], "score": 0}
    frontier = [root]
    
    for d in range(depth):
        new_frontier = []
        for node in frontier:
            # Generate multiple next thoughts.
            candidates = llm_generate_n_thoughts(question, node["path"], n=branches)
            for thought in candidates:
                new_path = node["path"] + [thought]
                score = llm_evaluate(question, new_path)
                new_frontier.append({"path": new_path, "score": score})
        
        # Keep top k.
        frontier = sorted(new_frontier, key=lambda n: -n["score"])[:branches]
    
    # Pick the best.
    best = max(frontier, key=lambda n: n["score"])
    return best["path"]
```

The compute cost: O(depth × branches²) LLM calls. For depth=3, branches=3: 27 reasoning calls. Significant.

Compare to ReAct: 3 sequential reasoning steps × 1 call each = 3 calls. ToT is ~9× more expensive but explores 27 paths.

## When ToT pays off

The win:
- **Hard reasoning** where the single-path agent gets stuck. ToT explores; finds the right path.
- **Tasks with clear evaluation**: math problems (right/wrong answers), constraint satisfaction.

The loss:
- **Tasks with no clear evaluator**: ToT picks the path that *looks* best, not necessarily the actually-best.
- **Latency-sensitive applications**: ToT's 5-10× cost is real.

For most agent tasks (chat, customer service, simple Q&A): ReAct is sufficient. For math contests, code generation, planning: ToT shines.

## LATS (Language Agent Tree Search)

LATS (Zhou et al, 2023) extends ToT with:
- **MCTS-style exploration**: balance exploit (deep into promising branches) and explore (try new branches).
- **Better evaluation**: use a value function trained on past trajectories.

LATS is more sophisticated; performance gains are 5-20% over ToT on hard tasks. The implementation is more involved.

For production deployments where every percentage point matters (e.g., agent benchmarks), LATS is worth it. For most: ToT or plain ReAct.

## Self-reflection

A lighter-weight variant: after generating an answer, the agent reflects ("Is this correct? What might I have missed?"), then revises.

```python
def reflect_and_revise(question):
    initial = llm.complete(f"Answer: {question}")
    
    reflection_prompt = f"""
Question: {question}
Answer: {initial}

Critique the answer. Is it correct? What might be missing or wrong?
"""
    critique = llm.complete(reflection_prompt)
    
    revision_prompt = f"""
Question: {question}
Initial answer: {initial}
Critique: {critique}

Revise the answer.
"""
    revised = llm.complete(revision_prompt)
    return revised
```

Three LLM calls instead of one; quality improvement of 5-10% on many tasks. Cheaper than full ToT; still better than single-shot.

## Hybrid patterns

Production systems often combine:
- **ReAct** as the default (cheap, fast).
- **Plan-and-execute** when the task is structured.
- **Self-reflection** as a final pass for important outputs.
- **ToT or LATS** for hard reasoning subtasks within an agent.

The choice is per-task; the agent might use different strategies for different sub-tasks.

## What you should believe after this lesson

Three sentences:

**1. Plan-and-execute separates planning from execution** — generate the plan upfront; execute steps. Useful for structured tasks; cheap; less adaptive than ReAct.

**2. Tree of Thoughts (ToT) explores multiple reasoning branches in parallel; LATS adds MCTS-style search**. 5-10× more expensive than ReAct; significantly better for hard reasoning (math, code, planning). LATS gets another 5-20% over ToT on hard benchmarks.

**3. Self-reflection is the lightweight middle ground** — generate, critique, revise. 3 LLM calls instead of 1; 5-10% quality gain on many tasks. Use as a final pass for important outputs.

## Hands-on (at home)

Implement a small ToT for a math problem.

```python
# tot_demo.py
from openai import OpenAI
client = OpenAI()

def llm(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

question = "I have 23 marbles. I give away 7. Then I find 15 more. Then I divide what I have equally among 4 friends. How many does each friend get?"

def generate_thoughts(history, n=3):
    prompt = f"""Question: {question}
Reasoning so far:
{history}

Generate {n} different possible next thoughts (a sentence each).
Number them 1, 2, 3."""
    response = llm(prompt)
    # Parse into a list.
    thoughts = [line.split('.', 1)[1].strip() for line in response.split('\n') if line.strip() and line[0].isdigit()]
    return thoughts[:n]

def evaluate(history):
    prompt = f"""Question: {question}
Reasoning so far:
{history}

Rate this reasoning path's likelihood of leading to the correct answer, on a scale 1-10.
Respond with just the number."""
    response = llm(prompt).strip()
    try:
        return int(response.split()[0])
    except:
        return 5

def tree_of_thoughts(depth=3, branches=2):
    nodes = [{"history": "", "score": 5}]
    for d in range(depth):
        new_nodes = []
        for node in nodes:
            thoughts = generate_thoughts(node["history"], n=branches)
            for t in thoughts:
                new_history = node["history"] + "\n- " + t
                score = evaluate(new_history)
                new_nodes.append({"history": new_history, "score": score})
        nodes = sorted(new_nodes, key=lambda n: -n["score"])[:branches]
        print(f"Depth {d}: top score = {nodes[0]['score']}")
    
    best = nodes[0]
    print(f"\nBest path:\n{best['history']}")
    
    # Final answer from the best path.
    answer = llm(f"Based on this reasoning: {best['history']}\n\nWhat's the final answer?")
    print(f"\nFinal: {answer}")

tree_of_thoughts()
```

You should see the tree-search reasoning explore multiple paths and converge to the right answer. The cost: ~20 LLM calls for this simple problem vs 1-2 for direct prompting.

## Further reading

- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al, 2023).
- "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models" (Zhou et al, 2023).
- "Reflexion" (Shinn et al, 2023) — reflection patterns.

Next lesson: **Self-reflection, critic loops, and verifier agents.** Separating "do" from "check" — using a second LLM as a critic to validate the first's outputs.
