---
title: "Lesson 12 — Agentic RAG"
date: "2026-06-04"
module: "agents"
order: 12
tags: ["agentic-rag", "self-rag", "corrective-rag", "iterative-retrieval"]
author: "Sudipta Pathak"
prerequisites: ["11-hybrid-search-reranking"]
---

# Lesson 12 — Agentic RAG

## Why this lesson exists

Standard RAG (Lessons 9-11) is single-shot: one query → one retrieval → one generation. For complex questions, this fails:
- "What's the difference between X and Y?" needs information on both.
- "Has the policy on X changed since 2020?" needs multiple time-period sources.
- "Find all bugs in this codebase related to authentication" needs many iterations.

Agentic RAG: the agent decides *what to retrieve*, *when*, and *how*. Multiple retrievals; the LLM directs them based on what it discovers.

This lesson covers agentic RAG patterns: query decomposition, iterative retrieval, Self-RAG, Corrective RAG.

The lesson is reading. The Hands-on builds an iterative-retrieval agent.

## The basic agentic pattern

The agent has a `search` tool. It calls it as needed:

```python
# Agent reasoning:
THOUGHT: I need to know X to answer this question.
ACTION: search("query about X")
OBSERVATION: [results about X]
THOUGHT: I also need to know Y.
ACTION: search("query about Y")
OBSERVATION: [results about Y]
THOUGHT: Now I have both. The answer is...
```

The retrieval is no longer a preprocessing step — it's an action the agent takes during reasoning, multiple times if needed.

Module 11 Lessons 1-6 cover the agent loop and tool use that make this possible. Agentic RAG is just "RAG retrieval as a tool, used in an agent loop."

## Query decomposition

For complex queries, decompose into sub-queries first:

```python
def decompose_query(question):
    prompt = f"Decompose this question into 2-5 sub-questions:\n\nQuestion: {question}\n\nSub-questions:"
    response = llm.complete(prompt)
    return parse_sub_questions(response)

sub_queries = decompose_query("Compare GPT-4 and Claude in terms of safety and reasoning.")
# → ["What are GPT-4's safety features?", "What are Claude's safety features?",
#    "What's GPT-4's reasoning performance?", "What's Claude's reasoning performance?"]

# Retrieve for each sub-query.
contexts = [retrieve(sq) for sq in sub_queries]

# Generate the comparison.
prompt = f"Question: {question}\n\nContext: {contexts}\n\nAnswer:"
answer = llm.complete(prompt)
```

The decomposition is itself an LLM call; the retrievals are typically parallel; the final generation synthesizes.

## Self-RAG

Self-RAG (Asai et al, 2023): the LLM emits *reflection tokens* during generation that indicate whether to retrieve and whether retrieved context is relevant.

The flow:
1. LLM begins generating.
2. Special token `[Retrieve]` → trigger retrieval; insert results; continue.
3. After each retrieved chunk: `[Relevant]` / `[Irrelevant]` → score it.
4. `[Supported]` / `[Partially Supported]` / `[No Support]` → does the generation match the retrieved evidence?

This makes retrieval *dynamic*: the model retrieves only when it senses it needs to, and discards irrelevant retrievals.

Implementation requires fine-tuning the LLM on Self-RAG data; the base LLM doesn't emit reflection tokens. Off-the-shelf Self-RAG models exist (built on Llama, Mistral); using them requires loading a specific fine-tuned model.

## Corrective RAG (CRAG)

Corrective RAG (Yan et al, 2024): after retrieval, a *retrieval evaluator* assesses quality. If retrieval is bad, the agent corrects via:
- Query rewriting and re-retrieving.
- Web search as backup.
- Acknowledging uncertainty.

The retrieval evaluator is a separate small model (or LLM call). It classifies each retrieval as `correct`, `incorrect`, or `ambiguous`. Based on the verdict:
- `correct`: use the retrieved chunks.
- `incorrect`: throw them out; web-search instead.
- `ambiguous`: combine retrieved + web-search.

CRAG handles the failure mode where the local knowledge base doesn't have the answer.

## Iterative retrieval

For multi-hop questions ("Who succeeded the king who succeeded Henry VIII?"), single retrieval can't get the answer:
1. Retrieve about "who succeeded Henry VIII" → Edward VI.
2. Retrieve about "who succeeded Edward VI" → Mary I.

The agent does retrieval iteratively, using each result to inform the next query.

The pattern:
```python
while not done:
    query = generate_next_query(question, accumulated_context)
    chunks = retrieve(query)
    accumulated_context.extend(chunks)
    done = is_answer_findable(question, accumulated_context)

answer = generate_answer(question, accumulated_context)
```

The agent decides when to stop (typically when it has enough info to answer).

## Agentic RAG with multi-hop reasoning

A worked example:

```
QUESTION: What programming language was the creator of Linux trained in?

STEP 1:
  THOUGHT: I need to find who created Linux.
  ACTION: retrieve("creator of Linux")
  OBSERVATION: Linux was created by Linus Torvalds in 1991.

STEP 2:
  THOUGHT: Now I need to find what Linus Torvalds trained in.
  ACTION: retrieve("Linus Torvalds education background")
  OBSERVATION: Linus Torvalds studied computer science at the University of Helsinki.

STEP 3:
  THOUGHT: I need to find what programming languages were taught there.
  ACTION: retrieve("University of Helsinki computer science curriculum 1980s")
  OBSERVATION: The University of Helsinki's CS program in the 1980s emphasized Pascal and C.

STEP 4:
  THOUGHT: I have enough.
  FINAL ANSWER: Linus Torvalds was trained primarily in Pascal and C at the University of Helsinki.
```

Each retrieval depends on the previous; can't be parallelized. The agent's reasoning chains them.

## When agentic RAG is overkill

For simple Q&A ("What's the capital of France?"), single-shot RAG is fine. Agentic RAG adds LLM-call latency without benefit.

Use agentic RAG for:
- Multi-hop reasoning.
- Comparisons requiring multiple sources.
- Complex queries where retrieval may need to be refined.
- Knowledge-base coverage isn't perfect (need to fall back to other sources).

Use simple RAG for:
- Single-fact lookups.
- Latency-critical applications.
- When retrieval quality is already high.

## What you should believe after this lesson

Three sentences:

**1. Agentic RAG replaces single-shot retrieval with iterative agent-driven retrieval.** The agent decides what to query, when, and how to combine results. Patterns: query decomposition (for parallel sub-queries), iterative retrieval (for multi-hop), Self-RAG and CRAG (for adaptive retrieval and correction).

**2. Self-RAG requires a fine-tuned LLM** that emits reflection tokens; Corrective RAG uses a separate retrieval evaluator. Both improve robustness on hard queries; both have latency cost.

**3. Use agentic RAG for complex queries** (multi-hop, comparisons, ambiguous); use simple RAG for single-fact lookups where the extra LLM calls aren't worth the latency.

## Hands-on (at home)

Build an iterative-retrieval agent.

```python
# iterative_rag.py — sketch.
from openai import OpenAI

client = OpenAI()

def retrieve_tool(query: str) -> str:
    # Implementation from Lesson 9.
    ...

TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Retrieve information from the knowledge base. Use when you need information you don't have.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

def agentic_rag(question, max_steps=5):
    messages = [
        {"role": "system", "content": "Answer the user's question. Use retrieve() as needed. When done, just respond with the answer (no more tool calls)."},
        {"role": "user", "content": question}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        
        if not msg.tool_calls:
            return msg.content  # final answer
        
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = retrieve_tool(**args)
            print(f"Step {step}: retrieved for '{args['query']}'")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
    
    return "(max steps reached)"

answer = agentic_rag("Compare GPT-4 and Claude in terms of safety and reasoning.")
print(answer)
```

The agent decides how many retrievals to do based on the question's complexity. Watch the per-step retrieval queries.

## Further reading

- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al, 2023).
- "Corrective Retrieval Augmented Generation" (Yan et al, 2024).
- "Adaptive-RAG" (Jeong et al, 2024) — selects RAG strategy based on query complexity.

End of Part 4. Next: Part 5 begins with **Plan-and-execute and Tree of Thoughts** — multi-step planning patterns for when one step isn't enough.
