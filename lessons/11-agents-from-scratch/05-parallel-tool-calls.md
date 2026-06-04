---
title: "Lesson 5 — Parallel Tool Calls"
date: "2026-06-04"
module: "agents"
order: 5
tags: ["parallel-tools", "async", "fan-out", "fan-in", "tool-selection"]
author: "Sudipta Pathak"
prerequisites: ["04-native-function-calling"]
---

# Lesson 5 — Parallel Tool Calls

## Why this lesson exists

Many tasks are embarrassingly parallel from the agent's perspective: "look up X, Y, and Z" doesn't need to be three sequential steps. Native function-calling APIs support *parallel tool calls* — the model emits multiple `tool_calls` in one response; the agent executes them concurrently; results come back together for the next step.

This is a meaningful latency win: 3 sequential 1-second tool calls = 3 seconds. 3 parallel calls = 1 second.

This lesson covers parallel tool calling, the async patterns, and the related "tool selection at scale" problem (when you have hundreds of tools).

The lesson is reading. The Hands-on builds a parallel-tool-calling agent.

## How parallel tool calls work

The model's response can include multiple `tool_calls`:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {"id": "call_1", "function": {"name": "search", "arguments": "{\"query\": \"X\"}"}},
    {"id": "call_2", "function": {"name": "search", "arguments": "{\"query\": \"Y\"}"}},
    {"id": "call_3", "function": {"name": "search", "arguments": "{\"query\": \"Z\"}"}}
  ]
}
```

The agent executes all three in parallel; collects results; appends each as a separate `tool` message with matching `tool_call_id`:

```json
{"role": "tool", "tool_call_id": "call_1", "content": "Result for X"}
{"role": "tool", "tool_call_id": "call_2", "content": "Result for Y"}
{"role": "tool", "tool_call_id": "call_3", "content": "Result for Z"}
```

The model is trained to handle this pattern: see multiple results, synthesize.

## Async execution

In Python, execute tools concurrently with `asyncio`:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def search_async(query):
    # Real implementation would be async HTTP call.
    await asyncio.sleep(1)
    return f"Result for {query}"

TOOLS_ASYNC = {"search": search_async}

async def run_agent_async(question, max_steps=10):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        
        if not msg.tool_calls:
            return msg.content
        
        # Execute tool calls in parallel.
        async def run_tool(tc):
            args = json.loads(tc.function.arguments)
            tool_fn = TOOLS_ASYNC[tc.function.name]
            result = await tool_fn(**args)
            return tc.id, result
        
        results = await asyncio.gather(*[run_tool(tc) for tc in msg.tool_calls])
        
        for tc_id, result in results:
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(result)})
    
    return None

# Run.
asyncio.run(run_agent_async("Find the populations of Tokyo, Paris, and London."))
```

`asyncio.gather` runs the tools concurrently. For 3 tools at 1 second each: ~1 second total instead of 3.

## When parallelism helps

The win:
- **IO-bound tools** (HTTP calls, DB queries, file reads): big win. The latency is wait time.
- **Independent tools**: no dependency between them. Parallel is correct.

The non-win:
- **Sequential dependencies**: if tool 2's input is tool 1's output, you can't parallelize.
- **CPU-bound tools**: parallelism is via threads/processes, not async. More complex.

The model decides whether tools are parallelizable: if the response has multiple `tool_calls`, the model thinks they're independent. The agent trusts this and runs in parallel.

## Tool selection at scale

When you have a few tools (5-10), the model picks easily. When you have many tools (100+), tool selection becomes its own problem:
- Context bloat: each tool's schema in every call adds up.
- Selection accuracy: the model has more to choose from; mistakes more likely.

Patterns for many-tool selection:

**Tool retrieval**: at request time, retrieve the top-N relevant tools (via vector search over tool descriptions); only pass those to the model. Reduces context; improves accuracy.

**Tool hierarchies**: a "router" agent picks a category; a "worker" agent picks the specific tool within the category. Two-level selection.

**Tool taxonomy in the prompt**: a structured catalog of tools the model can navigate. The model picks the category, then the tool.

For 100+ tool deployments, tool retrieval is the standard pattern. Vector search over tool descriptions; pass top 10 to the model; the model picks.

## Mixed parallel + sequential

Sometimes a workflow has both parallel and sequential parts:

```
Search for X, Y, Z in parallel.
Pick the most relevant result.
Read that result in detail.
Summarize.
```

The model handles this implicitly: first response has 3 parallel `search` calls; second response has one `read` call based on the search results; third response is the summary.

The agent loop doesn't need special handling; the parallelism happens naturally where the model chooses.

## Error handling for parallel calls

When one of N parallel tool calls fails:
- Append an error result for that call: `{"role": "tool", "tool_call_id": ..., "content": "Error: ..."}`.
- Continue with the others.
- The model handles the partial-failure case in its next response.

Trying to fail the whole batch is usually wrong — the model can often work with partial results.

For critical tools (must succeed), retry within the tool implementation before reporting failure.

## What you should believe after this lesson

Three sentences:

**1. Parallel tool calls** — multiple `tool_calls` in one model response, executed concurrently — provide significant latency wins for IO-bound, independent tools. Supported natively by modern LLM APIs.

**2. Async execution** (Python `asyncio.gather`) is the natural fit; 3 tools at 1s each take 1s total instead of 3s. The model decides what's parallelizable; the agent loop trusts and dispatches.

**3. Tool selection at scale** (100+ tools) requires retrieval or hierarchies: don't put every tool's schema in every call. Vector-search the top relevant tools at request time; pass only those to the model.

## Hands-on (at home)

Build a parallel-tool-calling agent.

```python
# parallel_tools.py
import asyncio
import json
import time
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def slow_search(query: str) -> str:
    # Simulate a slow tool.
    await asyncio.sleep(2)
    return f"Search results for '{query}'"

TOOLS_ASYNC = {"search": slow_search}
TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }
}]

async def run(question):
    messages = [{"role": "user", "content": question}]
    t0 = time.time()
    
    for _ in range(5):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        
        if not msg.tool_calls:
            print(f"\nFinal response: {msg.content}")
            break
        
        print(f"Step: {len(msg.tool_calls)} parallel tool calls")
        
        async def run_tool(tc):
            args = json.loads(tc.function.arguments)
            return tc.id, await TOOLS_ASYNC[tc.function.name](**args)
        
        results = await asyncio.gather(*[run_tool(tc) for tc in msg.tool_calls])
        for tc_id, result in results:
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
    
    print(f"Total time: {time.time() - t0:.1f}s")

asyncio.run(run("Look up the populations of Tokyo, Paris, and London."))
```

You should see the model emit 3 parallel calls in one step; the agent runs them concurrently; total time is ~2s (one tool latency), not 6s.

For tool retrieval at scale, use sentence-transformers + FAISS over the tool descriptions; vector-search at request time.

## Further reading

- OpenAI's "Parallel function calling" documentation.
- "Tool Retrieval with Many Tools" — various 2024-2026 blog posts.
- Asyncio best practices.

Next lesson: **Designing tools well.** Schema design, side-effect taxonomy, reversibility, error recovery — what makes a tool good for an agent to use.
