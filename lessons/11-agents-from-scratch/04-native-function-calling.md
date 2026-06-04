---
title: "Lesson 4 — Native Function Calling"
date: "2026-06-04"
module: "agents"
order: 4
tags: ["function-calling", "tool-use", "json-schema", "structured-output"]
author: "Sudipta Pathak"
prerequisites: ["03-planner-vs-worker"]
---

# Lesson 4 — Native Function Calling

## Why this lesson exists

Lesson 1's ReAct loop parses tool calls from free text. The model emits `ACTION: search("query")`; the agent regexes it out. This works but is fragile.

Native function calling (introduced by OpenAI in mid-2023, now standard across APIs) provides a structured way: the API accepts a list of tool schemas; the model responds with a structured `tool_call` object containing the function name and parsed arguments. No regex; no parse errors.

This lesson covers what's in the function-calling API payload, the JSON-schema layer, constrained decoding integration, and the production patterns.

The lesson is reading. The Hands-on rebuilds the ReAct loop with native function calling.

## The API payload

A function-calling-enabled request to OpenAI / Anthropic / Gemini includes a `tools` array:

```json
{
  "model": "gpt-4o",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search",
        "description": "Search the web for information.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "The search query."}
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

The model, trained on similar formats, may respond with:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "search",
        "arguments": "{\"query\": \"latest news\"}"
      }
    }
  ]
}
```

The `tool_calls` is the structured representation. The agent loop:
1. Parse `tool_calls`.
2. For each call, execute the corresponding tool with the parsed `arguments`.
3. Append the tool's output to messages with `role: "tool"` and the matching `tool_call_id`.
4. Call the API again with updated messages; the model continues.

## JSON-schema for tool parameters

The tool's `parameters` field is a JSON schema. The model is trained / fine-tuned to emit arguments matching the schema.

Common schema patterns:

**Required fields**:
```json
{"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
```

**Enums** (restrict to specific values):
```json
{"properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}}}
```

**Nested objects**:
```json
{
  "properties": {
    "config": {
      "type": "object",
      "properties": {"alpha": {"type": "number"}}
    }
  }
}
```

**Arrays**:
```json
{"properties": {"items": {"type": "array", "items": {"type": "string"}}}}
```

The model uses the schema to know what arguments are valid. Good schemas (with `description` for each field) significantly improve the model's tool-call accuracy.

## Constrained decoding for guaranteed valid output

Module 7 Lesson 24 covered constrained decoding. For function calling, the constraint is "the output must match the schema."

Some APIs (OpenAI's `strict: true` mode; Anthropic's tool use) guarantee valid JSON matching the schema. Others rely on the model being well-trained and producing valid JSON most of the time, with retry on parse failure.

For production:
- Use `strict` mode when available.
- Otherwise, parse with error handling; if parse fails, ask the model to fix and retry.

## The "tool" role

The response from a tool comes back to the model with `role: "tool"`:

```json
{"role": "tool", "tool_call_id": "call_abc123", "content": "Search results: ..."}
```

The `tool_call_id` links the response to the original call (important for parallel tool calls; Lesson 5).

The content can be a string (the tool's output) or a more structured object. For long outputs (search results, API responses), you may need to truncate or summarize before passing back.

## A function-calling ReAct loop

The Lesson 1 ReAct rewritten with native function calling:

```python
import json
from openai import OpenAI

client = OpenAI()

def search(query: str) -> str:
    return f"Top result for '{query}': [...]"

def calculator(expression: str) -> str:
    return str(eval(expression))

TOOLS_IMPL = {"search": search, "calculator": calculator}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Compute a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

def run_agent(question, max_steps=10):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        
        if not msg.tool_calls:
            return msg.content
        
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_fn = TOOLS_IMPL[tc.function.name]
            result = tool_fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

print(run_agent("What is 17 * 24?"))
```

Compare to Lesson 1's version: no regex, no parse errors, clean integration. The tool schemas are explicit.

## Differences across providers

OpenAI, Anthropic, Gemini all support function calling with slightly different APIs:

- **OpenAI**: `tools` parameter; `tool_calls` in response; `tool` role for results.
- **Anthropic**: `tools` parameter; `tool_use` blocks in content; `tool_result` blocks for results.
- **Gemini**: `tools` parameter; function-call objects in candidates.

The semantics are equivalent; the field names and structures differ. Most agent frameworks (LangChain, LlamaIndex, our own) abstract over these.

For multi-provider deployments, an adapter layer is necessary. The patterns are the same; the wire format differs.

## What you should believe after this lesson

Three sentences:

**1. Native function calling provides structured tool invocation** — the model emits a `tool_call` object instead of free text. Cleaner than regex parsing; less fragile; supported by all major LLM APIs.

**2. The tool schema (JSON Schema)** tells the model what arguments are valid. Good schemas with descriptions improve tool-call accuracy; constrained decoding modes guarantee valid output.

**3. The agent loop with native function calling** mirrors the ReAct loop but with structured parsing: receive `tool_calls`, execute each, append results with `role: "tool"`, repeat.

## Hands-on (at home)

Rebuild the Lesson 1 ReAct loop using native function calling (the code above). Compare:
- Parse error rates (should be zero).
- Code complexity (cleaner).
- Latency (similar).
- Model behavior (similar; slightly more reliable).

For a real workload, try a more complex tool schema:

```json
{
  "name": "create_calendar_event",
  "parameters": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "start_time": {"type": "string", "format": "date-time"},
      "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 480},
      "attendees": {"type": "array", "items": {"type": "string", "format": "email"}}
    },
    "required": ["title", "start_time", "duration_minutes"]
  }
}
```

The model handles this; the schema constrains the output to valid calendar events.

## Further reading

- OpenAI function calling documentation.
- Anthropic tool use documentation.
- "Building Effective Agents" (Anthropic, 2024).
- JSON Schema spec.

Next lesson: **Parallel tool calls.** Many APIs let the model call multiple tools in one response. Fan-out / fan-in async patterns; how parallelism speeds up multi-step tasks.
