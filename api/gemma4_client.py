# ============================================================================
# Gemma 4 E4B - Python Client SDK
# Modified by ClaudeO
# Purpose: Convenient Python client for interacting with the Gemma 4 E4B API.
#          Supports chat, vision, audio, agent selection, and streaming.
# ============================================================================

import base64
import json
from pathlib import Path
from typing import Generator, Optional

import httpx


class Gemma4Client:
    """
    Python client for the Gemma 4 E4B Local API.
    Provides easy-to-use methods for text chat, vision analysis,
    audio transcription, and agent-based interactions.

    Usage:
        client = Gemma4Client()
        response = client.chat("Hello, Gemma!")
        print(response)
    """

    # Modified by ClaudeO: Initialize with configurable base URL and defaults
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_agent: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the Gemma 4 E4B client.

        Args:
            base_url: API server URL (default: http://localhost:8000)
            default_agent: Default agent_id to use for all requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.default_agent = default_agent
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._history: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the HTTP client connection."""
        self._client.close()

    # --- Modified by ClaudeO: Core chat method ---
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        agent_id: Optional[str] = None,
        images: Optional[list[str | Path]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        keep_history: bool = False,
    ) -> str:
        """
        Send a chat message and get a response.

        Args:
            message: The user's message text
            system_prompt: Optional system prompt (overrides agent default)
            agent_id: Agent to route to (uses default_agent if not set)
            images: List of image file paths or base64 strings for vision
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum response tokens
            enable_thinking: Enable Gemma 4 thinking/reasoning mode
            keep_history: Maintain conversation history across calls

        Returns:
            The assistant's response text
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if enabled
        if keep_history:
            messages.extend(self._history)

        # Process images for multimodal input
        encoded_images = None
        if images:
            encoded_images = []
            for img in images:
                if isinstance(img, Path) or (isinstance(img, str) and Path(img).exists()):
                    # Read and encode file
                    with open(img, "rb") as f:
                        encoded_images.append(base64.b64encode(f.read()).decode())
                else:
                    # Assume already base64
                    encoded_images.append(img)

        # Build user message
        user_msg = {"role": "user", "content": message}
        if encoded_images:
            user_msg["images"] = encoded_images
        messages.append(user_msg)

        # Send request
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "agent_id": agent_id or self.default_agent,
            "enable_thinking": enable_thinking,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        assistant_msg = data["choices"][0]["message"]["content"]

        # Update history if enabled
        if keep_history:
            self._history.append(user_msg)
            self._history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    # --- Modified by ClaudeO: Streaming chat method ---
    def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        agent_id: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Send a chat message and stream the response token by token.

        Yields:
            Individual response text chunks
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "agent_id": agent_id or self.default_agent,
        }

        with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield content

    # --- Modified by ClaudeO: Image analysis convenience method ---
    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str = "Describe this image in detail.",
        agent_id: Optional[str] = "vision",
    ) -> str:
        """
        Analyze an image file using Gemma 4 E4B vision capabilities.

        Args:
            image_path: Path to the image file
            prompt: Analysis instruction
            agent_id: Agent to use (default: vision)

        Returns:
            Analysis text
        """
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/jpeg")}
            data = {"prompt": prompt}
            if agent_id:
                data["agent_id"] = agent_id
            resp = self._client.post("/v1/vision/analyze", files=files, data=data)
        resp.raise_for_status()
        return resp.json()["analysis"]

    # --- Modified by ClaudeO: Audio transcription convenience method ---
    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: str = "auto",
        prompt: str = "Transcribe this audio accurately.",
    ) -> str:
        """
        Transcribe an audio file using Gemma 4 E4B native audio support.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC)
            language: Language code or 'auto'
            prompt: Transcription instruction

        Returns:
            Transcribed text
        """
        with open(audio_path, "rb") as f:
            files = {"audio": (Path(audio_path).name, f, "audio/wav")}
            data = {"language": language, "prompt": prompt}
            resp = self._client.post("/v1/audio/transcribe", files=files, data=data)
        resp.raise_for_status()
        return resp.json()["transcription"]

    # --- Modified by ClaudeO: Agent management methods ---
    def list_agents(self) -> list[dict]:
        """List all available agents."""
        resp = self._client.get("/v1/agents")
        resp.raise_for_status()
        return resp.json()["agents"]

    def register_agent(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        temperature: float = 0.7,
        description: str = "",
    ) -> dict:
        """Register a new agent with custom system prompt."""
        payload = {
            "agent_id": agent_id,
            "name": name,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "description": description,
        }
        resp = self._client.post("/v1/agents", json=payload)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Check API server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()


# ============================================================================
# Quick usage examples
# ============================================================================

if __name__ == "__main__":
    print("=== Gemma 4 E4B Python Client Demo ===\n")

    with Gemma4Client() as client:
        # Health check
        status = client.health()
        print(f"Server status: {status['status']}\n")

        # Basic chat
        print("--- Basic Chat ---")
        response = client.chat("What are 3 advantages of running AI locally?")
        print(f"Response: {response}\n")

        # Agent-based chat
        print("--- Code Agent ---")
        code = client.chat(
            "Write a Python function to calculate fibonacci numbers efficiently.",
            agent_id="coder",
        )
        print(f"Code: {code}\n")

        # Chat with custom system prompt
        print("--- Custom System Prompt ---")
        response = client.chat(
            "Translate 'Hello World' to 5 languages.",
            system_prompt="You are a concise translator. Respond in a table format.",
        )
        print(f"Translation: {response}\n")

        # Streaming example
        print("--- Streaming ---")
        print("Streaming: ", end="", flush=True)
        for chunk in client.chat_stream("Count from 1 to 10 in Japanese."):
            print(chunk, end="", flush=True)
        print("\n")

        # List agents
        print("--- Available Agents ---")
        agents = client.list_agents()
        for agent in agents:
            print(f"  [{agent['agent_id']}] {agent['name']}: {agent['description']}")
