"""
Kimi K3 multimodal AI client
Provides text chat, image recognition/generation, video understanding/generation, and image-to-video.

Notes:
- The video features from video_understanding.py have been merged into KimiClient here;
  video_understanding.py now only re-exports for backwards compatibility (see bottom of file).
- API base unified to the official OpenAI-compatible endpoint https://api.moonshot.cn/v1
  (the old video_understanding.py wrongly used api.moonshot.ai; fixed).
- reasoning_effort matches the official README: only "low" / "high" / "max" (default "max").
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional, Generator, List, Tuple, Dict, Any, Union

import httpx
import requests
from openai import OpenAI


# Unified API base: the official Kimi / Moonshot OpenAI-compatible endpoint.
# The old video_understanding.py wrongly used api.moonshot.ai; corrected to .cn here.
API_BASE = "https://api.moonshot.cn/v1"

# Matches the README: reasoning_effort accepts only these values, default "max".
VALID_REASONING_EFFORTS = ("low", "high", "max")
DEFAULT_REASONING_EFFORT = "max"


class KimiClient:
    """Kimi K3 multimodal AI client (single entry point)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "kimi-k3",
        api_base: str = API_BASE,
        bypass_proxy: bool = True,
    ):
        """
        Initialise the Kimi client

        Args:
            api_key: Kimi API key; read from the KIMI_API_KEY environment variable if omitted
            model: model name, default kimi-k3
            api_base: API base URL, defaults to the official OpenAI-compatible endpoint
            bypass_proxy: connect directly, bypassing the system proxy (default True)
        """
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Provide an API key or set the KIMI_API_KEY environment variable\n"
                "  export KIMI_API_KEY=\"your_api_key\"\n"
                "  Get an API key at https://platform.kimi.ai"
            )

        self.model = model
        self.api_base = api_base

        # Bypass the proxy and connect directly (avoids local proxy env vars interfering)
        http_client = httpx.Client(trust_env=not bypass_proxy)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client,
        )

    # ============================================================
    # Text chat
    # ============================================================

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """
        Single-shot text chat.

        Args:
            prompt: user input
            system: optional system prompt
            stream: whether to stream. When streaming, returns a generator of (kind, text) chunks;
                    kind is "reasoning" (thinking) or "content" (final answer)
            reasoning_effort: "low" / "high" / "max"

        Returns:
            Non-streaming returns the full text; streaming returns a (kind, text) generator
        """
        session = ChatSession(self, system=system, reasoning_effort=reasoning_effort)
        return session.chat(prompt, stream=stream)

    def create_session(
        self,
        system: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model: Optional[str] = None,
    ) -> "ChatSession":
        """Create a multi-turn session (thinking history preserved automatically)."""
        return ChatSession(self, system=system, reasoning_effort=reasoning_effort, model=model)

    # ============================================================
    # Image recognition
    # ============================================================

    def analyze_image(
        self,
        image_source: str,
        prompt: str = "Describe this image in detail.",
        stream: bool = False,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[Generator[str, None, None], str]:
        """
        Analyse image content

        Args:
            image_source: image source - local path, http(s) URL, or a base64: prefix
            prompt: analysis prompt
            stream: whether to stream output
            reasoning_effort: "low" / "high" / "max"

        Returns:
            stream=True returns a generator; otherwise the full response string
        """
        image_b64, mime_type = self._load_image(image_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            return self._process_stream(response)
        else:
            return response.choices[0].message.content

    def describe_image(self, image_source: str) -> str:
        """Briefly describe an image"""
        return self.analyze_image(
            image_source,
            prompt="Describe the main content of this image in 2-3 sentences.",
            stream=False,
        )

    def identify_objects(self, image_source: str) -> str:
        """Detect objects in an image"""
        return self.analyze_image(
            image_source,
            prompt="List every recognisable object in the image and briefly describe its position and features.",
            stream=False,
        )

    def analyze_scene(self, image_source: str) -> str:
        """Analyse the scene in an image"""
        return self.analyze_image(
            image_source,
            prompt="Analyse the scene: setting, mood, lighting, colour tone, and the likely time and place.",
            stream=False,
        )

    def extract_text_from_image(self, image_source: str) -> str:
        """Extract text from an image (OCR)"""
        return self.analyze_image(
            image_source,
            prompt="Recognise and extract all text in the image, preserving the original formatting.",
            stream=False,
        )

    def answer_image_question(self, image_source: str, question: str) -> str:
        """Answer a specific question about an image"""
        return self.analyze_image(
            image_source,
            prompt=f"Look at the image and answer this question: {question}",
            stream=False,
        )

    # ============================================================
    # Image generation
    # ============================================================

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        style: Optional[str] = None,
    ) -> List[str]:
        """
        Generate an image

        Args:
            prompt: image generation prompt
            size: image size - "1024x1024", "1792x1024", or "1024x1792"
            n: number of images to generate
            style: style description (optional)

        Returns:
            A list of generated image URLs
        """
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, style: {style}"

        response = self.client.images.generate(
            model="kimi-k3-image",
            prompt=full_prompt,
            size=size,
            n=n,
        )

        return [item.url for item in response.data]

    def generate_and_save_image(
        self,
        prompt: str,
        output_path: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
    ) -> str:
        """
        Generate an image and save it locally

        Returns:
            The saved file path
        """
        urls = self.generate_image(prompt, size=size, style=style)

        if not urls:
            raise RuntimeError("Image generation failed")

        url = urls[0]
        print("Downloading generated image...")

        resp = requests.get(url, stream=True, timeout=120, proxies={})
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Image saved: {output_path}")
        return output_path

    # ============================================================
    # Video understanding
    # ============================================================

    def analyze_video(
        self,
        video_source: str,
        prompt: str = "Watch this video carefully and describe it in detail: what is on screen, and what happens?",
        stream: bool = True,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Union[Generator[str, None, None], str]:
        """
        Analyse video content

        Args:
            video_source: video source - local path, http(s) URL, or a base64: prefix
            prompt: analysis prompt
            stream: whether to stream output
            reasoning_effort: "low" / "high" / "max"

        Returns:
            stream=True returns a generator; otherwise the full response string
        """
        video_b64 = self._load_video(video_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            return self._process_stream(response)
        else:
            return response.choices[0].message.content

    def describe_video(self, video_source: str) -> str:
        """Briefly describe a video"""
        return self.analyze_video(
            video_source,
            prompt="Describe the main content of this video in 2-3 sentences.",
            stream=False,
        )

    def analyze_timeline(self, video_source: str) -> str:
        """Analyse the video timeline"""
        return self.analyze_video(
            video_source,
            prompt="Describe the events in the video in chronological order, formatted as: [timestamp] event description",
            stream=False,
        )

    def extract_key_frames(self, video_source: str) -> str:
        """Extract key-frame descriptions"""
        return self.analyze_video(
            video_source,
            prompt="Identify the key frames/scenes in the video and describe each one's content and approximate timestamp.",
            stream=False,
        )

    def analyze_emotion(self, video_source: str) -> str:
        """Analyse the mood/atmosphere of a video"""
        return self.analyze_video(
            video_source,
            prompt="Analyse the emotional tone and atmosphere: colour grading, music, facial expressions, and so on.",
            stream=False,
        )

    def answer_video_question(self, video_source: str, question: str) -> str:
        """Answer a specific question about a video"""
        return self.analyze_video(
            video_source,
            prompt=f"Watch the video and answer this question: {question}",
            stream=False,
        )

    # ============================================================
    # Video generation
    # ============================================================

    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        size: str = "1280x720",
    ) -> str:
        """
        Generate a video

        Args:
            prompt: video generation prompt
            duration: video length in seconds, default 5
            size: video size, default "1280x720"

        Returns:
            The generated video URL
        """
        response = self.client.images.generate(
            model="kimi-k3-video",
            prompt=prompt,
            size=size,
            extra_body={"duration": duration},
        )

        if response.data:
            return response.data[0].url
        raise RuntimeError("Video generation failed")

    def generate_and_save_video(
        self,
        prompt: str,
        output_path: str,
        duration: int = 5,
        size: str = "1280x720",
    ) -> str:
        """
        Generate a video and save it locally

        Returns:
            The saved file path
        """
        url = self.generate_video(prompt, duration=duration, size=size)

        print("Downloading generated video...")

        resp = requests.get(url, stream=True, timeout=300, proxies={})
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  Download progress: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)",
                        end="",
                        flush=True,
                    )

        print(f"\nVideo saved: {output_path}")
        return output_path

    def image_to_video(self, image_source: str, prompt: str = "") -> str:
        """
        Convert an image into a video

        Returns:
            The generated video URL
        """
        image_b64, _ = self._load_image(image_source)

        full_prompt = f"Turn this image into a moving video. {prompt}" if prompt else "Turn this image into a moving video"

        response = self.client.images.generate(
            model="kimi-k3-video",
            prompt=full_prompt,
            image=f"data:image/png;base64,{image_b64}",
            extra_body={"type": "image_to_video"},
        )

        if response.data:
            return response.data[0].url
        raise RuntimeError("Image-to-video conversion failed")

    # ============================================================
    # Utility methods
    # ============================================================

    def _load_image(self, image_source: str) -> Tuple[str, str]:
        """Load an image and convert it to base64, returning (base64, mime_type)"""
        if image_source.startswith("base64:"):
            return image_source[7:], "image/png"

        if image_source.startswith(("http://", "https://")):
            return self._download_image(image_source)

        return self._read_local_image(image_source)

    def _download_image(self, url: str) -> Tuple[str, str]:
        """Download an image from a URL"""
        print(f"Downloading image: {url}")
        resp = requests.get(url, timeout=120, proxies={})
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/png")
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()

        image_b64 = base64.b64encode(resp.content).decode("utf-8")
        print(f"  Download complete, size: {len(resp.content) // 1024}KB")
        return image_b64, content_type

    def _read_local_image(self, file_path: str) -> Tuple[str, str]:
        """Read a local image file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(path.suffix.lower(), "image/png")

        print(f"Reading image: {file_path}")
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"  Read complete, size: {len(image_b64) * 3 // 4 // 1024}KB")
        return image_b64, mime_type

    def _load_video(self, video_source: str) -> str:
        """Load a video and convert it to base64"""
        if video_source.startswith("base64:"):
            return video_source[7:]

        if video_source.startswith(("http://", "https://")):
            return self._download_video(video_source)

        return self._read_local_file(video_source)

    def _download_video(self, url: str) -> str:
        """Download a video from a URL"""
        print(f"Downloading video: {url}")
        resp = requests.get(url, stream=True, timeout=300, proxies={})
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  Download progress: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)",
                        end="",
                        flush=True,
                    )

            tmp_path = tmp.name

        print(f"\n  Download complete: {tmp_path}")

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_path)
        return video_b64

    def _read_local_file(self, file_path: str) -> str:
        """Read a local video file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        print(f"Reading video: {file_path}")
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"  Read complete, size: {len(video_b64) * 3 // 4 // 1024}KB")
        return video_b64

    @staticmethod
    def _process_stream(response) -> Generator[str, None, None]:
        """
        Handle a streaming response (backwards compatible).

        Note: this method interleaves reasoning_content and content.
        To separate thinking from the final answer, use ChatSession's streaming interface.
        """
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield delta.reasoning_content
            if delta.content:
                yield delta.content


def create_client(api_key: Optional[str] = None, model: str = "kimi-k3") -> KimiClient:
    """Convenience function: create a Kimi client"""
    return KimiClient(api_key=api_key, model=model)


class ChatSession:
    """
    Multi-turn chat session.

    K3 is trained in preserved-thinking-history mode: for multi-turn chat and tool calls, the complete
    assistant message returned by the API (including reasoning_content and tool_calls) must be passed
    back as-is; sending only content loses context. This class maintains that full message list for you.

    Supports mixed text and multimodal (image/video) multi-turn conversations.
    """

    def __init__(
        self,
        client: KimiClient,
        system: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model: Optional[str] = None,
    ):
        self.client = client
        self.model = model or client.model
        self.reasoning_effort = reasoning_effort
        self.messages: List[Dict[str, Any]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    # ---------- Multi-turn entry points ----------

    def chat(
        self,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """Multi-turn text chat."""
        self.messages.append({"role": "user", "content": prompt})
        return self._run(stream)

    def send_image(
        self,
        image_source: str,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """Send an image plus text for multimodal multi-turn chat."""
        image_b64, mime = self.client._load_image(image_source)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run(stream)

    def send_video(
        self,
        video_source: str,
        prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator[Tuple[str, str], None, None]]:
        """Send a video plus text for multimodal multi-turn chat."""
        video_b64 = self.client._load_video(video_source)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run(stream)

    # ---------- Internal execution ----------

    def _run(self, stream: bool):
        if stream:
            return self._stream_response()
        return self._blocking_response()

    def _blocking_response(self) -> str:
        resp = self.client.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            reasoning_effort=self.reasoning_effort,
            stream=False,
        )
        msg = resp.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
        # Critical: preserve the full thinking history (reasoning_content / tool_calls)
        if getattr(msg, "reasoning_content", None):
            assistant_msg["reasoning_content"] = msg.reasoning_content
        if getattr(msg, "tool_calls", None):
            assistant_msg["tool_calls"] = msg.tool_calls
        self.messages.append(assistant_msg)
        return msg.content

    def _stream_response(self) -> Generator[Tuple[str, str], None, None]:
        """
        Streaming multi-turn. Yields (kind, text) chunks:
        - kind="reasoning": the thinking process
        - kind="content": the final answer
        When finished, the complete assistant message (with reasoning_content) is stored in history.
        """
        reasoning_parts: List[str] = []
        content_parts: List[str] = []

        stream = self.client.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            reasoning_effort=self.reasoning_effort,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
                yield ("reasoning", delta.reasoning_content)
            if delta.content:
                content_parts.append(delta.content)
                yield ("content", delta.content)

        assistant_msg = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            assistant_msg["reasoning_content"] = "".join(reasoning_parts)
        self.messages.append(assistant_msg)

    # ---------- Session management ----------

    def reset(self) -> None:
        """Clear the conversation history (keeping the system message, if any)."""
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    def history(self) -> List[Dict[str, Any]]:
        """Return the current full message list (including reasoning_content)."""
        return self.messages


# Backwards-compatible alias
VideoUnderstanding = KimiClient
