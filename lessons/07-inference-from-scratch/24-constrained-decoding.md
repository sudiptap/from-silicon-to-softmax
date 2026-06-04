---
title: "Lesson 24 — Constrained Decoding: Grammars, JSON Schema, Regex"
date: "2026-06-04"
module: "inference-from-scratch"
order: 24
tags: ["constrained-decoding", "grammar", "json-schema", "regex", "structured-output"]
author: "Sudipta Pathak"
prerequisites: ["23-beam-search"]
---

# Lesson 24 — Constrained Decoding: Grammars, JSON Schema, Regex

## Why this lesson exists

When an LLM produces structured output (JSON, code, SQL, function calls), it sometimes gets the format wrong: missing closing brace, invalid field name, type mismatch. The naive fix is to sample, try to parse, retry on failure — expensive and brittle.

Constrained decoding is a cleaner solution: at each step, *mask out* tokens that would produce invalid output. The model can only generate valid sequences. The output is guaranteed to parse.

This lesson covers the mechanism, the major implementations (Outlines, Guidance, LMQL, vLLM's grammar mode, llama.cpp's GBNF, OpenAI's structured output), and the latency/quality tradeoffs.

The lesson is reading. The Hands-on uses llama.cpp's GBNF grammar to force JSON output.

## The basic mechanism

At each decode step:
1. Compute logits as usual.
2. Determine which tokens are valid *given the state of the constraint* (the JSON schema parser's current position, the regex's current state, etc.).
3. Set logits for invalid tokens to `-infinity`.
4. Sample from the masked distribution.

The constraint advances as tokens are emitted. After emitting `{"name":`, the constraint expects either `"` (start of string) or some other valid JSON value start — the constraint state encodes this.

## Three common constraint types

**1. Regex.** The output must match a specific regex. The constraint state is the regex's NFA / DFA state. Tokens are valid if they advance the state.

Example: phone number regex `\d{3}-\d{3}-\d{4}`. After emitting `555-`, only digit-tokens are valid for the next position.

**2. JSON schema.** The output must be valid JSON matching a given schema. The constraint state tracks position in the schema (which field, what type expected). Tokens are valid if they advance the parse.

Example: schema `{"type": "object", "properties": {"name": {"type": "string"}}}`. After emitting `{"name": `, only tokens that start a string (`"`) are valid.

**3. Context-free grammar (BNF / GBNF).** Most general. The output must be derivable from a grammar. The constraint state is the parser's stack. Tokens are valid if they advance the parse.

Example: SQL grammar. After emitting `SELECT name FROM`, only tokens that start a table name or alias are valid.

## llama.cpp's GBNF

llama.cpp uses a GBNF (Grammar in BNF) format for constraints. A JSON-only grammar:

```
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
string ::= "\"" [^"]* "\""
value ::= object | array | string | number | "true" | "false" | "null"
array ::= "[" value ("," value)* "]"
number ::= "-"? [0-9]+ ("." [0-9]+)? ("e" [+-]? [0-9]+)?
```

Apply at decode time:

```bash
./llama-cli -m model.gguf -p "Output a JSON object with a 'name' and 'age' field." \
    --grammar-file json.gbnf -n 100
```

The output is guaranteed to be valid JSON.

## Token-level vs character-level constraints

A subtle technical issue: constraints are typically defined in terms of *characters* (the JSON schema or regex operates on character strings), but generation is token-by-token. Tokens don't always align with character boundaries — the token `{"` is two characters; the next token might be `"n` which starts a string and adds an `n`.

The constraint engine needs to:
1. Maintain the character-level constraint state.
2. For each candidate token, "trial-emit" its characters and check if the constraint can still be satisfied.
3. Allow tokens whose characters are compatible with the constraint.

This per-step compatibility check is the main cost of constrained decoding. The number of vocabulary tokens (32K-128K) is large; checking each one against the constraint is expensive.

Modern implementations precompute lookup tables to make this fast: for each constraint state, the set of valid tokens is cached. The first time a state is encountered, you pay the lookup cost; subsequent visits are fast.

## The "guided generation" pattern

Beyond pure constraint-based generation, structured-output libraries support more flexible patterns:

**Field-by-field generation.** Generate one field of a JSON object at a time; constraints apply per field. Lets you mix free-form and constrained generation cleanly.

```python
# Pseudo-code:
result = {}
result["name"] = generate(prompt="What is the name?", max_tokens=20)  # free-form
result["age"] = generate(prompt="What is the age?", regex=r"\d+")     # constrained to number
result["bio"] = generate(prompt="Write a bio.", max_tokens=200)        # free-form
```

**Templates with holes.** Output a template; sample only at specific positions.

```python
# Output: "The person's name is _____, and they are _____ years old."
# Sample only at the underscored positions.
```

The Guidance library popularized this pattern. SGLang has a similar API.

## The latency cost

Constrained decoding has overhead:
- The constraint check per step: 0.1-1 ms depending on the constraint complexity.
- For long outputs, this is multiplied by the token count.

For most applications, the overhead is acceptable (5-15% throughput cost). For latency-critical applications, the constraint check is sometimes the bottleneck.

Modern implementations (xgrammar, Outlines with the FSM compiler) reduce this to near-zero by precomputing the constraint state machine.

## The accuracy cost

A subtle issue: constrained decoding forces the model to emit valid output, but doesn't make the *content* better. A model that's bad at JSON will produce syntactically-valid but semantically-wrong JSON. Constraints don't fix the underlying model capability.

For function calling and structured output, you still need a model trained for it. Constrained decoding makes the output parse; the model has to make the output correct.

Some quality concerns:
- Constraints can lead the model away from its natural high-probability distribution; outputs may be less coherent.
- For complex constraints (deep nested JSON), the model may "give up" — produce minimal valid output rather than reasoning correctly.

The right pattern: combine constrained decoding with a model trained on similar outputs. Llama 3.1+ are decent at JSON/function-call output without constraints; constraints make them robust.

## Production support

Most production runtimes support constrained decoding:
- **llama.cpp**: GBNF grammars.
- **vLLM**: Outlines integration (`extra_body={"guided_json": {...}}`).
- **SGLang**: built-in regex and JSON support.
- **TensorRT-LLM**: limited; constraints via post-processing.
- **OpenAI API**: native `response_format={"type": "json_object"}` and Structured Outputs feature.
- **Anthropic API**: native via tool use schemas.

In 2026, constrained decoding is a basic capability of any serving stack.

## What you should believe after this lesson

Three sentences:

**1. Constrained decoding masks out invalid tokens at each step**, so the output is guaranteed to match a regex, JSON schema, or context-free grammar. Outputs always parse; no retry loops needed.

**2. Token-vs-character boundary issues are the main implementation challenge** — modern libraries precompute lookup tables that make per-step constraint checking fast (0.1-1 ms). The throughput overhead is typically 5-15%.

**3. Constraints make outputs syntactically valid but don't make them semantically correct** — a model bad at JSON will produce syntactically-valid wrong JSON. Combine constraints with a model trained for the output format.

## Hands-on (at home)

Force JSON output with llama.cpp.

```bash
# Create a JSON grammar file.
cat > json.gbnf <<'EOF'
root ::= object
object ::= "{" ws pair ("," ws pair)* "}"
pair ::= string ws ":" ws value
string ::= "\"" [^"]* "\""
value ::= object | array | string | number | "true" | "false" | "null"
array ::= "[" ws value ("," ws value)* ws "]"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n]*
EOF

# Generate constrained JSON.
./llama-cli -m gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --grammar-file json.gbnf \
    -p "Generate a JSON object describing a fictional book with title, author, and year." \
    -n 100 --temp 0.3
```

Output should be valid JSON. Try without `--grammar-file`; observe that the model usually produces valid JSON anyway for this prompt, but may sometimes get the format wrong.

For more complex grammars (full SQL, Markdown subsets, etc.), see llama.cpp's grammars examples directory.

For the Outlines / Guidance Python approach:

```python
from outlines import models, generate
model = models.transformers("Qwen/Qwen2.5-1.5B-Instruct")
schema = '{"name": "...", "age": "...", "occupation": "..."}'
generator = generate.json(model, schema)
result = generator("Describe a fictional person.")
print(result)
```

## Further reading

- "Grammar-Aligned Decoding" (Park et al, 2024) — recent academic treatment.
- Outlines library docs (github.com/outlines-dev/outlines).
- Guidance library docs (github.com/guidance-ai/guidance).
- llama.cpp grammar documentation.
- "Efficient Guided Generation for Large Language Models" (Willard & Louf, 2023) — the FSM-based approach Outlines uses.

Next lesson: **Speculative decoding — draft + verify.** A different angle: instead of generating one token at a time, propose multiple at once via a draft model and verify in parallel. We covered the basics in Module 6 Lesson 9; here we derive the algorithm and implementation in detail.
