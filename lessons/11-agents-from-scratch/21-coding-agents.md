---
title: "Lesson 21 — Coding Agents"
date: "2026-06-04"
module: "agents"
order: 21
tags: ["coding-agents", "claude-code", "cursor", "aider", "code-editing"]
author: "Sudipta Pathak"
prerequisites: ["20-mcp-from-scratch"]
---

# Lesson 21 — Coding Agents

## Why this lesson exists

Coding agents (Claude Code, Cursor, Aider, GitHub Copilot Workspace, Devin, OpenHands) are the most mature vertical agent category. They've evolved fast in 2024-2026; the patterns are stabilizing.

This lesson covers what coding agents do, the architectural patterns, and the engineering decisions that distinguish good ones from poor ones.

The lesson is reading. The Hands-on builds a small file-editing agent.

## What coding agents do

The task: given a codebase + an instruction ("fix this bug," "add this feature," "refactor this module"), produce the right code edits.

The flow:
1. Understand the request.
2. Explore the codebase (find relevant files).
3. Read the relevant files.
4. Edit / add / delete code.
5. Test the changes.
6. Iterate until tests pass and the request is satisfied.

Each step requires tools: find_files, read_file, edit_file, run_command (for tests).

## The standard toolset

A coding agent typically has:

- **read_file(path)**: read a file.
- **write_file(path, content)**: write a file.
- **edit_file(path, old, new)**: replace `old` with `new` in the file (line-anchored or fuzzy).
- **list_files(directory)**: list files in a directory.
- **glob(pattern)**: find files matching a pattern.
- **grep(pattern, path)**: search for a regex in files.
- **run_command(cmd)**: execute a shell command (tests, build, etc.).
- **git** tools (status, diff, commit).

The toolset enables exploration + modification + verification.

## Edit formats

How does the agent specify edits? Several formats:

**Whole-file replacement**: the agent outputs the full new file. Simple; works for small files; expensive for large.

**Search-and-replace (Aider style)**: the agent outputs `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks. Compact for small changes.

**Unified diff (`diff --git` format)**: the agent outputs a diff; applied with `patch`. Compact; standard.

**Line-anchored edits**: the agent specifies "edit lines 50-55." Brittle if lines have shifted.

**Tool-based**: the agent calls `edit_file(path, old_chunk, new_chunk)`; the tool finds and replaces.

In 2026, search-and-replace and tool-based are the dominant formats. They're robust to small file changes and compact.

## Claude Code / Cursor patterns

Claude Code (Anthropic, 2024) and Cursor (a popular code editor with agent features) share patterns:

**Long-running session**: the agent works on a task for many minutes / hours; multiple file edits; multiple test runs.

**Tool-driven exploration**: the agent uses glob / grep to find relevant code; reads files; modifies; tests.

**Verifier-augmented**: tests serve as the verifier (Lesson 14). The agent generates code; runs tests; revises on failure.

**Subagent isolation**: for complex tasks (refactor a module), spawn a research subagent to map out the changes first.

**Context engineering**: keep the codebase context focused — only relevant files in context; summary of others.

## Aider's edit-file pattern

Aider (open-source coding assistant) uses a clever edit format:

```
filename.py
<<<<<<< SEARCH
def old_function():
    pass
=======
def new_function():
    return 42
>>>>>>> REPLACE
```

The agent generates this; aider parses; applies. Compact; the LLM only outputs what changes.

This format generalizes well; many other agents have adopted it.

## The "agentic" vs "completion" distinction

Two coding-agent styles:

**Completion-style** (GitHub Copilot, in-line suggestions): the agent suggests code at the cursor; the user accepts/rejects. Fast; per-keystroke; doesn't run autonomously.

**Agentic** (Claude Code, Cursor's agent mode, Devin): the agent operates on the whole codebase; runs autonomously; does multi-step work. Slower per task; handles complex changes.

In 2026, completion is mature; agentic is the active research / product area. Different tools serve different use cases.

## The "stuck" failure mode

Common agent failure: stuck in a loop trying to fix a test, making things worse with each iteration.

Causes:
- The agent's mental model of the code is wrong; edits don't fix the actual problem.
- Test failures are ambiguous; the agent fixes the wrong thing.
- The agent doesn't explore enough; misses the actual root cause.

Mitigations:
- **Test isolation**: run only the relevant tests; clearer signals.
- **Context refresh**: occasionally re-read the file to see the current state.
- **Explicit reasoning**: prompt the agent to articulate the root cause before editing.
- **Human escalation**: after N failed attempts, ask for help.

Cursor and Claude Code have refined these patterns; production coding agents handle most cases gracefully.

## Cost of coding agents

Coding agents are expensive:
- Long sessions (10K-100K tokens per task).
- Many tool calls (read file, edit, run tests, repeat).
- Multiple LLM calls per task (often 20-50).

A typical coding task on Claude: $0.50-$5 in API costs.

Optimizations:
- **Context trimming**: keep only relevant files in context.
- **Prompt caching**: stable parts (system prompt, codebase summary) cached.
- **Cheaper model for simple edits**: route simple changes to a cheaper model.
- **Streaming verification**: don't wait for full completion to check tests.

For production coding-agent products, cost engineering matters as much as quality.

## What you should believe after this lesson

Three sentences:

**1. Coding agents have a standard toolset** (read/write/edit/list/glob/grep/run_command + git) and patterns (long-running sessions, tool-driven exploration, verifier via tests, subagent isolation for complex tasks). Claude Code, Cursor, Aider all follow this template.

**2. Edit formats** vary: whole-file (simple), search-and-replace (compact; Aider's), unified diff (standard), line-anchored (brittle). Search-and-replace and tool-based are dominant in 2026.

**3. Coding agents are expensive** ($0.50-$5 per task) due to long sessions and many tool calls. Cost engineering (context trimming, caching, model routing) matters for production deployments.

## Hands-on (at home)

Build a small file-editing agent.

```python
# coding_agent.py — sketch.
import os
from openai import OpenAI
client = OpenAI()

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

def write_file(path: str, content: str) -> str:
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"

def list_files(directory: str) -> str:
    return "\n".join(os.listdir(directory))

def run_command(cmd: str) -> str:
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

# Tool schemas (omitted for brevity; same pattern as Lesson 4).

# Run an agent loop that uses these tools to complete the task:
# "Find the bug in /tmp/buggy.py and fix it."
```

For a complete demo:
1. Write a small buggy Python file.
2. Give the agent the path + the task.
3. Let it explore, read, edit, test.
4. Verify the fix.

For real production: Claude Code, Cursor, Aider — use them; read their open-source pieces; the patterns generalize.

## Further reading

- "Anthropic Claude Code documentation."
- Aider GitHub.
- "SWE-bench" — benchmark for coding agents.
- "Cursor architecture" blog posts (when published).

Next lesson: **Browser and computer-use agents.** Playwright loops, OSWorld, the screen-as-input pattern. The other major vertical agent category.
