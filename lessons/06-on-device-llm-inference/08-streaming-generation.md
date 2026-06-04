---
title: "Lesson 8 — Streaming Generation Patterns"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 8
tags: ["streaming", "ux", "cancellation", "stop-tokens", "tokens-per-second"]
author: "Sudipta Pathak"
prerequisites: ["07-mmap-weight-streaming"]
---

# Lesson 8 — Streaming Generation Patterns

## Why this lesson exists

A 30 tok/s decode rate sounds slow until you compare it to "wait 5 seconds for the full response, then see it all." Streaming tokens to the UI as they're produced is the difference between an LLM that feels modern and one that feels like a 1990s batch system. It also exposes the model's pace to the user, which sets expectations around "thinking time" and lets them interrupt early if the generation goes the wrong direction.

Streaming sounds simple — just print tokens as they arrive — but the practical implementation has subtleties: token-by-token tokenization quirks, partial-multibyte-character handling, cancellation semantics, partial-rollback for "stop" interactions, and the integration with the app's UI thread. This lesson is the operational view.

The lesson is reading. The Hands-on builds a small streaming chat loop in Python that mirrors the production pattern.

## What streaming looks like at the API level

A non-streaming inference returns the whole completion at the end:

```python
response = model.generate(prompt, max_tokens=200)
print(response)  # all at once, after several seconds
```

A streaming inference yields tokens (or token-decoded strings) as they're produced:

```python
for token_text in model.generate_stream(prompt, max_tokens=200):
    print(token_text, end='', flush=True)
```

The user sees text appearing word by word, similar to a typewriter. Even at 15 tok/s — which would be a 13-second wait for a 200-token response — the streaming UX feels responsive because the user starts seeing output within ~100 ms (first token) and continues to see progress.

Behind the scenes:

- The runtime's generation loop produces one token at a time.
- After each token, a callback is invoked with that token (or the runtime yields it from a generator).
- The application code receives the token and updates the UI.

The token-to-UI path needs to be fast enough not to bottleneck generation. At 100 tok/s, you have 10 ms per token to do the UI update. A poorly-implemented UI update (rendering the whole conversation from scratch each token) can drop the effective throughput.

## Tokens vs displayed characters

A subtle gotcha: tokens are sub-word units in the model's tokenizer, not characters. The token "ing" comes out as a complete unit; the token "fox" comes out as a complete unit. Most tokens decode to ASCII characters or short multi-character substrings cleanly.

The tricky case: multi-byte UTF-8 characters that span multiple tokens. For example, an emoji might be encoded as 4 bytes in UTF-8, and the tokenizer may have split it across two or three tokens. Decoding token-by-token and printing each token's decoded bytes can produce invalid UTF-8 sequences mid-character.

The fix: maintain a small *buffer* of recently-produced bytes that haven't yet decoded to a complete UTF-8 character. When new bytes arrive, append to the buffer; try to decode; output the prefix that decodes cleanly; keep the trailing partial bytes for the next iteration.

```python
buffer = bytes()
for token_id in stream:
    new_bytes = tokenizer.decode([token_id]).encode('utf-8')
    buffer += new_bytes
    try:
        text = buffer.decode('utf-8')
        print(text, end='', flush=True)
        buffer = bytes()
    except UnicodeDecodeError:
        # Partial character; wait for more bytes.
        pass
```

The `transformers.TextStreamer` and similar standard streamers handle this. If you're rolling your own streaming UI, this is the bug you'll hit first.

## Cancellation

The user clicks "Stop" mid-generation. What happens?

The naive implementation: the generate-loop runs to completion; the UI just stops displaying. Bad — you waste compute and battery generating tokens nobody will see.

The right implementation: the generate loop checks a cancellation flag between iterations. When the flag is set, it exits early.

```python
def generate_stream(prompt, max_tokens, cancellation_token):
    state = init_state(prompt)
    for _ in range(max_tokens):
        if cancellation_token.is_cancelled():
            break
        token = step(state)
        yield token
        if token == eos_token:
            break
```

The application sets the cancellation token when the user clicks Stop. The runtime exits its loop, frees the KV cache for the (now-aborted) generation, and returns.

The user-facing semantics:

- Stop the generation immediately (don't wait for a "good" stopping point).
- The partial response stays in the conversation history.
- The user can retry with a different prompt or edit and continue.

For runtimes that don't expose a cancellation API natively (older versions of some libraries), you can hack it: run the generation in a thread; the cancel button kills the thread. This is uglier but works in a pinch.

## Stop tokens and stop sequences

A common pattern: stop generation when a specific token or sequence appears, even before `max_tokens`. Examples:

- The EOS token (always; this is built into every generation loop).
- A custom token like `<|im_end|>` for chat templates.
- A multi-token sequence like `\n\nHuman:` to prevent the model from continuing a fake conversation.

The single-token cases (EOS, `<|im_end|>`) are easy — the generation loop checks each new token against the stop set.

The multi-token sequences require *lookback*: keep a buffer of the last N tokens and check if the buffer ends with any stop sequence. The trade-off: longer stop sequences require larger lookback, and you may produce extra tokens before the stop sequence completes.

```python
stop_sequences = ["\n\nHuman:", "\n\nUser:"]
generated_buffer = ""
for token_text in stream:
    generated_buffer += token_text
    for stop in stop_sequences:
        if generated_buffer.endswith(stop):
            # Strip the stop sequence and emit, then stop.
            yield generated_buffer[:-len(stop)]
            return
    yield token_text  # streaming output
```

(This has a subtle issue: tokens emitted earlier in the loop have already been displayed, so you can't retroactively "remove" the stop sequence from the user's view if it's been partially shown. Either delay output by the length of the longest stop sequence — adding latency — or accept that the stop sequence appears briefly and then the generation stops. The "delay by N tokens" pattern is more common in production.)

## Partial rollback

A specific scenario: the user is mid-generation, types something into the input box, and clicks Send. The expected behavior: stop the current generation, accept the new input, treat the partial generation as the previous turn's incomplete response.

The implementation:
1. Cancel the current generation (Stop semantics).
2. Truncate the conversation state so the partial response is correctly recorded.
3. Append the new user input.
4. Start a new generation.

The KV cache implication: the cache contains the partial generation's K and V tensors. If the next inference's conversation history matches the prefix (which it does — the conversation up to the partial response), you can reuse the prefix's KV state. The "partial response" tokens at the end need to be evicted (or, equivalently, the cache's "current length" pointer reset to before they were added).

In llama.cpp this is done with the `llama_kv_cache_seq_rm` API; in MLX you re-construct the cache or use the relevant mlx-lm APIs.

The benefit of doing this right: the next generation's prefill skips the entire conversation history (it's already cached), and only processes the new user message. TTFT drops from "process the entire conversation again" to "process the new turn."

## Stream-aware UI patterns

A few UX considerations that emerge from streaming generation:

**1. Cursor / typing indicator.** While the model is generating, show a blinking cursor or a "thinking..." indicator. The user knows the system is working.

**2. Word-boundary smoothing.** Tokens don't align to word boundaries; the user sees text appearing in "Hello there, how" rather than "Hello there, how". The UI typically renders this fine, but layout-sensitive scenarios (markdown rendering, code blocks) may need to defer rendering until a token like newline appears.

**3. Markdown / code-block awareness.** Mid-generation, the model might be inside a code block. The UI should render it as formatted code, not as raw text. This usually means streaming into a "draft" text widget and re-rendering markdown periodically (every ~500 ms or every N tokens), not on every token.

**4. Cancellation feedback.** When the user clicks Stop, give visible feedback within ~50 ms. The actual cancellation might take ~100 ms (the generation loop has to check the flag and exit); the UI should acknowledge the click immediately.

**5. Error states.** The model produces nothing (immediate EOS); the model errors mid-generation; the model produces gibberish. The UI should handle each gracefully — display "(no response)" instead of an empty bubble, surface errors instead of silently failing.

## Throughput accounting

A note on the numbers you'll see: "tok/s" can mean two different things:

- **Prefill tok/s**: how fast the model processes the *input* prompt. Higher is better.
- **Decode tok/s**: how fast the model produces the *output*. Higher is better.

These are very different numbers — prefill is a single big matmul (high throughput, all-at-once), decode is many small matmuls (lower throughput, bandwidth-bound).

Typical ratios for a 3B model on M3 Pro:
- Prefill: ~2000 tok/s (a 1000-token prompt takes 500 ms).
- Decode: ~40 tok/s (100 tokens take 2.5 seconds).

TTFT (time-to-first-token) is `prefill_time + 1 decode token time`. For a 100-token prompt: ~50 ms + ~25 ms = ~75 ms. Comfortably under the 500 ms TTFT threshold.

Total time = `prefill_time + (output_length × decode_time)`. The decode rate is the user-perceived speed; the prefill rate matters mostly for long prompts.

When benchmarking, always report both. A model that prefills slow but decodes fast may be fine for chat (short prompts, long outputs) and bad for summarization (long prompts, short outputs).

## What you should believe after this lesson

Three sentences:

**1. Streaming tokens to the UI as they're generated is essential for modern LLM UX** — even at moderate decode rates (15 tok/s) the experience feels responsive because TTFT is small and the user sees progress. Token-to-UI latency must be small enough not to bottleneck the generation loop.

**2. The operational subtleties matter more than the algorithm**: partial multi-byte UTF-8 character buffering, cancellation tokens that the generation loop checks per iteration, stop-sequence lookback, and KV cache reuse on partial rollback. Each is a small thing; together they're the difference between a polished and a janky LLM experience.

**3. Throughput has two numbers** — prefill and decode tok/s — that mean very different things. Report both when benchmarking; the user-perceived speed is decode for chat-style workloads, prefill for long-prompt workloads.

## Hands-on (at home)

A minimal streaming chat loop.

```python
# streaming_chat.py
# pip install mlx mlx-lm
from mlx_lm import load, stream_generate
import time

model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

conversation = []  # list of (role, content) tuples

def chat_step(user_msg):
    conversation.append(("user", user_msg))
    messages = [{"role": role, "content": content} for role, content in conversation]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    full_response = ""
    first_token_time = None
    t0 = time.time()
    n_tokens = 0
    
    for r in stream_generate(model, tokenizer, prompt, max_tokens=200):
        token_text = r.text
        if first_token_time is None:
            first_token_time = time.time() - t0
        print(token_text, end='', flush=True)
        full_response += token_text
        n_tokens += 1
    
    total_time = time.time() - t0
    decode_time = total_time - first_token_time
    decode_tps = (n_tokens - 1) / decode_time if decode_time > 0 else 0
    print(f"\n  [TTFT: {first_token_time*1000:.0f} ms; decode: {decode_tps:.1f} tok/s]")
    
    conversation.append(("assistant", full_response))

while True:
    try:
        msg = input("\nYou: ")
        if not msg.strip():
            continue
        if msg.lower() in ("exit", "quit"):
            break
        print("Assistant: ", end='', flush=True)
        chat_step(msg)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        # Ideally: cancel the in-progress generation. mlx_lm.stream_generate
        # doesn't have a clean cancellation API as of mid-2026; you'd interrupt
        # the Python loop.
        break

print("Goodbye.")
```

Try it. Note the TTFT and decode tok/s after each turn; they should be in the expected range (50-200 ms TTFT for short prompts, 30-80 tok/s decode for a 1.5B model on M3 Pro).

To experiment with cancellation, modify the loop to add a signal-handler-based cancellation token. The proper implementation requires threading the cancellation through `stream_generate`; the mlx_lm package's API surface for this is improving across versions.

## Further reading

- The OpenAI Streaming API documentation — establishes the canonical streaming JSON format that every other API mimics.
- `transformers.TextStreamer` source — for a reference Python implementation of token-to-text streaming with UTF-8 handling.
- llama.cpp's `server` source — for a C++ reference implementation of the OpenAI-compatible streaming endpoint.
- "User-Perceived Latency in LLM Chat Applications" — various blog posts on the human factors side.

Next lesson: **Speculative decoding on-device.** A different angle on throughput: instead of making each token's generation faster, *propose* multiple tokens at once via a small draft model and verify them in parallel via the main model. We look at the throughput math, the memory cost, and when speculative decoding pays off on consumer hardware.
