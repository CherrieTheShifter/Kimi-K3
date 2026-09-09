# Kimi K3 Python SDK — proposed features

> Scope: the Python implementation in this repository (a Kimi K3 multimodal API client).
> Current state: `kimi_multimodal.py` implements image/video understanding and generation; `video_understanding.py` is the older video module (duplicating the same functionality).
> Priorities: P0 core gaps and fixes → P1 parity with the README → P2 engineering quality → P3 repository and release.

---

## Hard problems to fix first (not new features, but they will bite)

| # | Problem | Where | Notes |
|---|---------|-------|-------|
| B1 | **Inconsistent API base** | `kimi_multimodal.py` uses `api.moonshot.cn/v1`, `video_understanding.py` used `api.moonshot.ai` | Two different hosts for the same service. Pick one — the README points at `platform.kimi.ai`. |
| B2 | **Wrong reasoning_effort values** | Both modules wrote `"low"/"medium"/"high"` | The README specifies `"low"/"high"/"max"` (default `max`). |
| B3 | **Two duplicate clients** | `kimi_multimodal.py` vs `video_understanding.py` | The video features were written twice, almost identically. Double the maintenance, and the behaviour diverged. Merge into one client. |

---

## P0 — core capability gaps (key K3 usage patterns, currently missing entirely)

### 1. Multi-turn chat with preserved thinking history
K3 is trained in preserved-thinking-history mode. **For multi-turn conversations and tool calls, the complete assistant message returned by the API — including `reasoning_content` and `tool_calls` — must be passed back as-is.** Sending only `content` loses context.
- Current state: `analyze_image` / `analyze_video` are one-shot calls with no session state.
- Proposal: add a `ChatSession` class or `client.chat()` that maintains the `messages` list internally, preserves `reasoning_content` automatically, and offers `reset()`.

### 2. Basic text chat `chat()`
- Current state: the client is called "multimodal" but only does image/video analysis and generation — **there is no plain text chat entry point.**
- Proposal: `chat(prompt, stream=...)` via `chat.completions`, reusing the same reasoning/stream logic.

### 3. Separate streaming reasoning from content
- Current state: `_process_stream` yields `reasoning_content` and `content` mixed together.
- Proposal: support callbacks or separate capture, so a UI can show "thinking" and "final answer" apart (the README stresses that K3 always returns reasoning).

### 4. Merge the two clients, unify the base URL
- Remove or demote `video_understanding.py`; move all capability into `KimiClient` and unify the base URL (use the README's `platform.kimi.ai` and verify the host actually works).

---

## P1 — capabilities the README claims but the code lacks

### 5. Tool calling / function calling (`tool_calls`)
The README states K3 supports tool calling. Implement `client.chat_with_tools(tools, ...)` and handle the multi-turn tool loop.

### 6. Structured output / JSON mode
The README mentions structured output. Add `client.structured(prompt, schema)` returning a parsed object.

### 7. Context caching
The README mentions context caching (saves tokens on long context). Mark long system prompts and large video descriptions for caching.

### 8. Async client
Every call is currently a synchronous OpenAI client. Add `AsyncKimiClient` (`AsyncOpenAI`) for high-concurrency batch work.

---

## P2 — engineering and usability

### 9. A real CLI (replacing the current `input()` menus)
Use `argparse` to build a `kimi-k3` command:

```
kimi-k3 chat "hello"
kimi-k3 image ./cat.png --prompt "describe"
kimi-k3 gen-image "a ginger cat" -o out.png
```

Today `example_usage.py` and `run_tests.py` are `input()`-driven and cannot be scripted.

### 10. Batch / directory processing with concurrency
Promote the hand-written `for` loop in `example_usage.py` into `batch_analyze(pattern, max_concurrency=4)`, on top of the async client from item 8.

### 11. Large-file upload strategy (avoid full base64)
- Current state: the whole video is read into base64 and inlined as `data:video/mp4;base64,...`, which easily exceeds token and request-body limits.
- Proposal: support a file-upload endpoint or upload-then-reference-by-URL, and accept more video formats than mp4.

### 12. Retry, rate limiting, timeouts and backoff
- Current state: no retries; `requests.get(..., timeout=...)` runs once, and a 429 fails outright.
- Proposal: exponential backoff on 429/5xx, with configurable timeout and max retries.

### 13. Token usage and cost reporting
- Current state: the `usage` field in the response is never surfaced.
- Proposal: attach `prompt_tokens` / `completion_tokens` to results, with optional running totals.

---

## P3 — repository and release

### 14. Bring the SDK into the repository properly
Add an `sdk/` directory for the Python implementation, with its own README separate from the model documentation.

### 15. Packaging and publishing
Add `pyproject.toml` and `requirements.txt`, support `pip install kimi-k3-sdk`, and pin the `openai` / `requests` / `httpx` versions.

### 16. Non-interactive tests and CI
- Current state: the test scripts rely on `input()` menus and cannot run in CI.
- Proposal: move to pytest, support `pytest --api-key=...` or an environment variable, and add a non-interactive `--all` mode.

### 17. Model and capability discovery
Add `client.list_models()` or a config source, so future K3 sub-models do not require code changes.

---

## Suggested order of work

1. Fix B1/B2/B3 first (one client, correct parameters) — that is the foundation.
2. Then P0 items 1-3 (multi-turn chat, `chat()`, split streaming) to cover K3's most important usage patterns.
3. Then P1 (tool calling, structured output, caching, async).
4. Finally P2/P3 for usability and release quality.
