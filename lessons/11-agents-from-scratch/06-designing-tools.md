---
title: "Lesson 6 — Designing Tools Well"
date: "2026-06-04"
module: "agents"
order: 6
tags: ["tool-design", "schema", "side-effects", "reversibility", "idempotency"]
author: "Sudipta Pathak"
prerequisites: ["05-parallel-tool-calls"]
---

# Lesson 6 — Designing Tools Well

## Why this lesson exists

A well-designed tool makes the agent reliable; a poorly-designed tool makes the agent fail in confusing ways. Tool design is a real engineering discipline — schemas, error messages, side-effect taxonomy, reversibility, idempotency. The patterns aren't intuitive; the failure modes are subtle.

This lesson covers what makes a tool good for agents: the design principles, the anti-patterns, and the production patterns.

The lesson is reading. The Hands-on critiques and redesigns a poorly-designed tool.

## The dimensions of tool design

A tool is a function the agent can call. The design dimensions:

1. **Name**: what's it called? Should be clear.
2. **Description**: what does it do, when should the agent use it?
3. **Parameters**: what inputs does it accept? Schema; defaults; required vs optional.
4. **Return value**: what does it produce? Format; length; structure.
5. **Side effects**: what does it change in the world? Read-only vs writes; reversible vs not.
6. **Error behavior**: what happens when it fails? Error messages; partial success.

## Naming

Good tool names:
- Verb-based: `search_web`, `send_email`, `create_calendar_event`.
- Specific: `search_internal_wiki` better than `search` (when you have many search tools).
- Unambiguous: avoid `process_data` (vague).

Bad: `do_thing`, `helper`, `main`.

The name + description together let the model pick the right tool. Bad naming → wrong tool choices.

## Description

The description tells the model when to use the tool. Good descriptions include:
- What the tool does (one sentence).
- When to use it ("Use this when the user asks about ...").
- Examples (sometimes).
- What it returns (briefly).

A canonical good description:

```
"Search the web for current information.
Use when the user asks about recent events, facts that may have changed,
or information not in your training data.
Returns the top 5 search results with titles and snippets."
```

Bad description: `"Search."`. Tells the model nothing about when to use it vs alternatives.

For agents with many similar tools (search_web vs search_internal_docs vs search_codebase), descriptions become critical for disambiguation.

## Schema

The parameter schema (JSON Schema) describes inputs:

**Use specific types**: `"type": "integer"` better than `"type": "string"` for numeric values. The model is more likely to produce correct format.

**Add constraints**: `"minimum": 1, "maximum": 100` for ranges; `"enum": [...]` for limited choices. The model respects these.

**Mark required fields**: missing required → schema error; explicit is better.

**Avoid free-form strings when possible**: for "select a status" use `enum`; for "pick a date" use `format: "date"`.

**Add descriptions per field**: each `properties` entry should have a `description`. The model uses these to fill in correctly.

## Return value design

The tool's output is the agent's "observation." It needs to be:

**Concise**: long outputs bloat context. If the tool can return 10 KB of JSON, consider summarizing.

**Structured**: prefer JSON over free text when possible. The model parses structured data more reliably.

**Informative on failure**: errors should be specific. `"Error: API timeout after 30 seconds; retry suggested"` better than `"Error"`.

**Indicative of next steps**: hints about what to do next if the tool's response is incomplete.

For long-tail outputs (search results with potentially 100 hits), return a summary plus a way to get more detail. Don't overwhelm the context.

## Side-effect taxonomy

Tools have different impact on the world:

**Read-only**: queries that don't change state. `search`, `get_user_info`, `query_database`. Safe to retry.

**Write, idempotent**: creates or updates state, but the same call twice produces the same result. `set_user_email("new@example.com")`. Safe to retry.

**Write, non-idempotent**: each call adds something. `send_email`, `charge_credit_card`, `create_record`. NOT safe to retry without care.

**Destructive**: removes state. `delete_user`, `cancel_subscription`. Especially careful with retries.

The taxonomy matters for retry logic. The agent's retry policy should know:
- Read-only: retry freely.
- Idempotent write: retry freely.
- Non-idempotent write: retry only if you know the first call failed (not just timed out).
- Destructive: never retry without confirmation.

Some tools expose an `idempotency_key` for non-idempotent operations; retries with the same key are safe.

## Reversibility

Can the tool's effect be undone?

**Easily reversible**: a setting change with an explicit "unset." The agent can recover from a wrong decision.

**Hard reversible**: deleting data; sending an email. Once done, can't be undone.

For high-stakes irreversible actions, design the tool to require confirmation:

```python
def send_email(to: str, subject: str, body: str, confirm: bool = False):
    if not confirm:
        return "Confirmation required. Re-call with confirm=True."
    # ... actually send
```

The two-step pattern: first call shows the draft; second call (with `confirm=True`) sends. Reduces mistakes.

## Error recovery

Tool errors should help the agent recover:

**Specific errors**: `"User not found: ID 12345 does not exist"` not `"Error"`.

**Suggested actions**: `"Database timeout. Retry in 5 seconds, or use the cached version with get_user_cached()"`.

**Partial success**: if some part succeeded, report what.

The agent reads error messages and adapts. Good error messages make the agent more reliable; bad ones cause it to spin.

## The "tool surface" question

How many tools should an agent have?

**Few tools, broad scope**: simple to navigate; each tool does a lot.

**Many tools, narrow scope**: clearer purpose per tool; harder to navigate.

For agent reliability, fewer tools tend to work better. The model can hold the tool catalog in mind; tool selection is easier.

For complex domains (a full SaaS product), you may need many tools. Use tool hierarchies (Lesson 5) or tool retrieval.

A common pattern: a small set of "primary" tools (5-10) for common cases; many "specialized" tools accessible via retrieval. The model defaults to the primary set; the specialized set is available when needed.

## Anti-patterns

**Tool with hundred parameters**: hard for the model to fill correctly. Break into smaller tools.

**Tool that does too much**: "manage_user" with sub-commands for create / update / delete / list. Better to have separate tools.

**Tool that returns megabytes of data**: bloats context. Paginate or summarize.

**Tool that requires multiple correct calls**: a write tool that doesn't validate inputs and fails silently. Better to validate; clear error; the agent can recover.

**Tool with side effects on read calls**: e.g., a "search" that logs the user's query in a way the user wouldn't expect. Surprises break trust.

## What you should believe after this lesson

Three sentences:

**1. Tool design is a real engineering discipline**: clear names + descriptions + schemas + return values + error messages + side-effect classification. Bad tools cause confusing agent failures; good tools make agents reliable.

**2. The side-effect taxonomy** (read-only / idempotent / non-idempotent / destructive) determines retry policy. Read-only is safe to retry; destructive should never be retried without confirmation. Use idempotency keys for non-idempotent operations.

**3. The "few broad" vs "many narrow" tool surface tradeoff**: few is usually more reliable; many requires retrieval or hierarchies to navigate. Default to few; specialize as needed.

## Hands-on (at home)

Critique a poorly-designed tool and redesign it.

```python
# Bad tool.
{
    "name": "process_data",
    "description": "Processes data.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string"},
            "operation": {"type": "string"},
            "options": {"type": "string"}  # unstructured options!
        }
    }
}

# Redesigned.
{
    "name": "filter_data",
    "description": "Filter rows of a CSV file by a column value. Returns the filtered rows as JSON. Use when the user wants to subset data based on a specific column condition.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file."},
            "column": {"type": "string", "description": "The column to filter on."},
            "operator": {"type": "string", "enum": ["eq", "neq", "gt", "lt", "contains"], "description": "Comparison operator."},
            "value": {"type": "string", "description": "The value to compare against."},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100, "description": "Max rows to return."}
        },
        "required": ["file_path", "column", "operator", "value"]
    }
}
```

The redesigned version: specific name, clear description, structured parameters with constraints, sensible defaults, bounded outputs. The model will use this tool correctly far more often than the original.

## Further reading

- "Building Effective Agents" (Anthropic, 2024) — has a good tools section.
- OpenAI's function calling guide.
- "Tool Design for LLM Agents" — various blog posts.

End of Part 2. Next: Part 3 begins with **The agent memory hierarchy** — short-term (context), working (scratchpad), long-term (vector/key-value); when each matters.
