# Kimi K3 Python SDK

A small Python client for the Kimi K3 API — text chat, image and video understanding, image and video generation, and multi-turn conversations with preserved thinking history.

> This is the SDK documentation. For the model itself, see [README.md](README.md)  

---

## Install

```bash
pip install openai httpx requests
```

## Set your API key

```bash
export KIMI_API_KEY="your_api_key"
```

Get a key at [platform.kimi.ai](https://platform.kimi.ai).

## Quick start

```python
from kimi_multimodal import KimiClient

client = KimiClient()

print(client.chat("Explain mixture-of-experts in one paragraph."))
print(client.describe_image("./photo.png"))
print(client.describe_video("./orange-cat-agent.mp4"))
```

---

## Multi-turn chat

K3 is trained in preserved-thinking-history mode: the complete assistant message — including `reasoning_content` — has to be passed back on every turn, or context is lost. `ChatSession` handles that for you.

```python
session = client.create_session(system="You are a helpful assistant.")

session.chat("Tell me three random numbers.")
session.chat("What were the other two you had in mind?")   # answers from the preserved reasoning
```

| Method | Purpose |
|---|---|
| `chat(prompt, stream=False)` | Text turn |
| `send_image(image_source, prompt)` | Image + text turn |
| `send_video(video_source, prompt)` | Video + text turn |
| `reset()` | Clear history, keep the system message |
| `history()` | Full message list, reasoning included |

## Streaming

Streaming yields `(kind, text)` pairs, so thinking and the final answer can be shown separately:

```python
for kind, text in session.chat("Why is the sky blue?", stream=True):
    if kind == "reasoning":
        print(text, end="")      # the model's thinking
    else:
        print(text, end="")      # the final answer
```

---

## What KimiClient can do

**Images**

`analyze_image` · `describe_image` · `identify_objects` · `analyze_scene` · `extract_text_from_image` (OCR) · `answer_image_question` · `generate_image` · `generate_and_save_image` · `image_to_video`

**Video**

`analyze_video` · `describe_video` · `analyze_timeline` · `extract_key_frames` · `analyze_emotion` · `answer_video_question` · `generate_video` · `generate_and_save_video`

Image and video sources accept a local path, an `http(s)` URL, or a `base64:` prefix.

**Reasoning effort** — `"low"`, `"high"`, or `"max"` (default `"max"`), passed to any call.

---

## Files

| File | What it is |
|---|---|
| `kimi_multimodal.py` | The client. `KimiClient` and `ChatSession` live here. |
| `video_understanding.py` | Backwards-compatible re-export of the old `VideoUnderstanding` class. |
| `kimik3.py` | Minimal standalone example of preserved thinking history. |
| `example_usage.py` | Eight worked examples — basic, streaming, batch, custom analysers. |
| `quick_reference.py` | Cheat sheet of common calls. |
| `run_tests.py` | Quick non-interactive smoke test of image and video features. |
| `test_all_features.py` | Full interactive test menu, 13 tests. |
| `test_video_understanding.py` | Video-focused test menu, 7 tests. |
| `test_chat_session.py` | Offline unit tests — no API key or network needed. |
| `studio/` | Unsloth Studio Colab notebook. |
| `FEATURE_SUGGESTIONS.md` | Known gaps and a proposed roadmap. |

## Running the tests

```bash
python test_chat_session.py     # offline, no API key needed
python run_tests.py             # smoke test, needs KIMI_API_KEY
python test_all_features.py     # interactive menu
```

---

## Known issues

See [FEATURE_SUGGESTIONS.md](FEATURE_SUGGESTIONS.md). The main ones:

- **API base.** The client uses `api.moonshot.cn/v1`; the model README points at `platform.kimi.ai`. Verify which host works for you.
- **No tool calling or structured output** yet, though K3 supports both.
- **Video is inlined as base64**, so large files can exceed request limits.
- **No retries or backoff** — a 429 fails outright.
