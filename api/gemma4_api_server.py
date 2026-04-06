# ============================================================================
# Gemma 4 E4B - Unified API Server
# Modified by ClaudeO
# Purpose: FastAPI server that wraps Ollama's Gemma 4 E4B model, providing
#          OpenAI-compatible endpoints for Python/Node.js agents with
#          multimodal support (text, image, audio) and system prompt routing.
# ============================================================================

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- Modified by ClaudeO: Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gemma4-api")

# ============================================================================
# Configuration
# ============================================================================

# Modified by ClaudeO: Centralized config with environment variable overrides
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("GEMMA4_MODEL", "gemma4-e4b-opt")  # Use optimized variant
FALLBACK_MODEL = "gemma4:e4b"  # Fallback to base model
API_PORT = int(os.getenv("GEMMA4_API_PORT", "8000"))
MAX_CONTEXT_TOKENS = int(os.getenv("GEMMA4_MAX_CTX", "4096"))  # Memory-optimized default


# ============================================================================
# Pydantic Models for API Request/Response
# ============================================================================

# Modified by ClaudeO: OpenAI-compatible message format
class ChatMessage(BaseModel):
    """Single message in a conversation. Supports text and multimodal content."""
    role: str = Field(..., description="One of: system, user, assistant")
    content: str = Field(..., description="Text content of the message")
    images: Optional[list[str]] = Field(
        None, description="Base64-encoded images (for vision tasks)"
    )


class ChatRequest(BaseModel):
    """Chat completion request - OpenAI-compatible format."""
    model: Optional[str] = Field(None, description="Model name override")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(64, ge=1)
    max_tokens: Optional[int] = Field(None, description="Max response tokens")
    # Modified by ClaudeO: Agent-specific fields
    agent_id: Optional[str] = Field(None, description="Agent identifier for routing")
    enable_thinking: bool = Field(False, description="Enable Gemma 4 thinking mode")


class ChatChoice(BaseModel):
    """Single choice in a chat completion response."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Chat completion response - OpenAI-compatible format."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class AgentConfig(BaseModel):
    """Configuration for an AI agent persona."""
    agent_id: str
    name: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    description: str = ""


class AudioTranscribeRequest(BaseModel):
    """Request for audio transcription (E4B supports native audio)."""
    language: Optional[str] = Field("auto", description="Target language code")


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model: str
    ollama_url: str
    memory_mode: str
    uptime_seconds: float


# ============================================================================
# Agent Registry
# ============================================================================

# Modified by ClaudeO: Pre-configured agents with system prompts
AGENT_REGISTRY: dict[str, AgentConfig] = {}
_start_time = time.time()


def load_default_agents():
    """Load built-in agent configurations."""
    defaults = [
        AgentConfig(
            agent_id="general",
            name="General Assistant",
            system_prompt=(
                "You are Gemma 4 E4B, a helpful local AI assistant. "
                "Respond concisely and accurately. Use structured formatting "
                "when it aids clarity."
            ),
            description="Default general-purpose assistant",
        ),
        AgentConfig(
            agent_id="coder",
            name="Code Assistant",
            system_prompt=(
                "You are an expert programming assistant running locally via Gemma 4 E4B. "
                "Write clean, well-documented code. Always explain your approach briefly. "
                "Support Python, JavaScript, TypeScript, and shell scripting. "
                "When debugging, show the root cause first, then the fix."
            ),
            temperature=0.3,
            description="Coding and debugging specialist",
        ),
        AgentConfig(
            agent_id="analyst",
            name="Data Analyst",
            system_prompt=(
                "You are a data analysis agent powered by Gemma 4 E4B. "
                "Analyze data precisely. Present findings with clear structure. "
                "When given images of charts or tables, extract key insights. "
                "Always cite specific numbers from the data."
            ),
            temperature=0.4,
            description="Data analysis and visualization insights",
        ),
        AgentConfig(
            agent_id="translator",
            name="Multilingual Translator",
            system_prompt=(
                "You are a professional translator powered by Gemma 4 E4B with "
                "native support for 140+ languages. Translate accurately while "
                "preserving tone, idioms, and cultural nuances. Always specify "
                "the source and target languages in your response."
            ),
            temperature=0.3,
            description="Multi-language translation (140+ languages)",
        ),
        AgentConfig(
            agent_id="vision",
            name="Vision Analyst",
            system_prompt=(
                "You are a visual analysis agent running Gemma 4 E4B. "
                "Analyze images with precision: describe content, extract text (OCR), "
                "interpret charts, identify objects. For each image, provide structured "
                "analysis with: 1) Content summary, 2) Key details, 3) Extracted text if any."
            ),
            temperature=0.5,
            description="Image analysis, OCR, and visual reasoning",
        ),
        AgentConfig(
            agent_id="planner",
            name="Task Planner",
            system_prompt=(
                "You are an agentic task planner powered by Gemma 4 E4B. "
                "Break complex requests into actionable steps. For each step, "
                "specify: the action, required inputs, expected outputs, and any "
                "tool calls needed. Use function calling format when appropriate. "
                "Think step by step and verify your plan's completeness."
            ),
            temperature=0.5,
            description="Multi-step task planning and agentic workflows",
        ),
    ]
    for agent in defaults:
        AGENT_REGISTRY[agent.agent_id] = agent
    logger.info(f"Loaded {len(defaults)} default agents")


# ============================================================================
# Ollama Client
# ============================================================================

# Modified by ClaudeO: Async HTTP client for Ollama communication
class OllamaClient:
    """Async client for communicating with the local Ollama server."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    async def close(self):
        """Close the HTTP client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get("/")
            return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> dict:
        """Send a chat completion request to Ollama (non-streaming)."""
        client = await self._get_client()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            # Fallback to base model if optimized variant not found
            if e.response.status_code == 404 and model != FALLBACK_MODEL:
                logger.warning(f"Model '{model}' not found, falling back to '{FALLBACK_MODEL}'")
                payload["model"] = FALLBACK_MODEL
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        options: Optional[dict] = None,
    ):
        """Send a streaming chat completion request to Ollama."""
        client = await self._get_client()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {},
        }
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def list_models(self) -> list[dict]:
        """List all available Ollama models."""
        client = await self._get_client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])


# Global Ollama client instance
ollama_client = OllamaClient()


# ============================================================================
# FastAPI Application
# ============================================================================

# Modified by ClaudeO: Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    load_default_agents()
    # Verify Ollama connection
    if await ollama_client.health_check():
        logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL}")
    else:
        logger.warning(f"Ollama not reachable at {OLLAMA_BASE_URL}. Start it with: ollama serve")
    yield
    await ollama_client.close()
    logger.info("API server shut down cleanly")


app = FastAPI(
    title="Gemma 4 E4B Local API",
    description=(
        "Unified API server for Google Gemma 4 E4B running locally via Ollama. "
        "Supports text, vision, audio input with agent-based system prompts. "
        "OpenAI-compatible endpoints for easy integration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Modified by ClaudeO: CORS for local development (Python, Node.js, browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

# --- Modified by ClaudeO: Health check endpoint ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and Ollama server status."""
    ollama_ok = await ollama_client.health_check()
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        model=DEFAULT_MODEL,
        ollama_url=OLLAMA_BASE_URL,
        memory_mode=f"optimized (ctx={MAX_CONTEXT_TOKENS})",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# --- Modified by ClaudeO: Main chat completion endpoint (OpenAI-compatible) ---
@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["Chat"])
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Supports system prompts, vision (base64 images), and agent routing.
    """
    model = request.model or DEFAULT_MODEL
    messages = []

    # If an agent_id is specified, prepend its system prompt
    if request.agent_id and request.agent_id in AGENT_REGISTRY:
        agent = AGENT_REGISTRY[request.agent_id]
        # Check if a system message already exists
        has_system = any(m.role == "system" for m in request.messages)
        if not has_system:
            messages.append({"role": "system", "content": agent.system_prompt})
        # Override temperature if agent has custom setting
        if request.temperature == 0.7:  # Only override if user didn't customize
            request.temperature = agent.temperature

    # Modified by ClaudeO: Enable thinking mode via system prompt token
    if request.enable_thinking:
        # Find or create system message and prepend <|think|>
        system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"), -1
        )
        if system_idx >= 0:
            messages[system_idx]["content"] = (
                "<|think|>\n" + messages[system_idx]["content"]
            )
        else:
            messages.insert(0, {"role": "system", "content": "<|think|>"})

    # Convert request messages to Ollama format
    for msg in request.messages:
        ollama_msg = {"role": msg.role, "content": msg.content}
        if msg.images:
            ollama_msg["images"] = msg.images
        messages.append(ollama_msg)

    # Build Ollama options for memory optimization
    options = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "num_ctx": MAX_CONTEXT_TOKENS,
    }
    if request.max_tokens:
        options["num_predict"] = request.max_tokens

    # Handle streaming vs non-streaming
    if request.stream:
        return StreamingResponse(
            _stream_response(messages, model, options),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        result = await ollama_client.chat(messages, model=model, options=options)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    response_text = result.get("message", {}).get("content", "")
    return ChatResponse(
        model=model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_tokens=(
                result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            ),
        ),
    )


async def _stream_response(messages, model, options):
    """Generator for SSE streaming response."""
    try:
        async for chunk in ollama_client.chat_stream(messages, model, options):
            content = chunk.get("message", {}).get("content", "")
            if content:
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# --- Modified by ClaudeO: Vision endpoint for image analysis ---
@app.post("/v1/vision/analyze", tags=["Multimodal"])
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail."),
    agent_id: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image using Gemma 4 E4B's native vision capabilities.
    Supports JPEG, PNG, WebP formats.
    """
    # Read and encode the image
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
    if agent_id and agent_id in AGENT_REGISTRY:
        agent = AGENT_REGISTRY[agent_id]
        messages.append({"role": "system", "content": agent.system_prompt})

    messages.append({
        "role": "user",
        "content": prompt,
        "images": [image_b64],
    })

    try:
        result = await ollama_client.chat(messages, options={"num_ctx": MAX_CONTEXT_TOKENS})
        return {
            "analysis": result.get("message", {}).get("content", ""),
            "model": DEFAULT_MODEL,
            "image_size": len(image_bytes),
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Vision analysis failed: {str(e)}")


# --- Modified by ClaudeO: Audio transcription endpoint ---
@app.post("/v1/audio/transcribe", tags=["Multimodal"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    prompt: str = Form("Transcribe this audio accurately."),
):
    """
    Transcribe audio using Gemma 4 E4B's native audio input capability.
    Supports WAV, MP3, FLAC. Max 30 seconds.
    Note: Audio is sent as base64 to Ollama which forwards to the model.
    """
    audio_bytes = await audio.read()

    # Gemma 4 E4B supports native audio - encode as base64
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    transcribe_prompt = prompt
    if language != "auto":
        transcribe_prompt += f" The audio is in {language}."

    messages = [
        {"role": "system", "content": "You are an audio transcription specialist. Transcribe audio accurately, preserving punctuation and speaker intent."},
        {"role": "user", "content": transcribe_prompt, "images": [audio_b64]},
    ]

    try:
        result = await ollama_client.chat(messages, options={"num_ctx": MAX_CONTEXT_TOKENS})
        return {
            "transcription": result.get("message", {}).get("content", ""),
            "language": language,
            "audio_size": len(audio_bytes),
            "model": DEFAULT_MODEL,
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {str(e)}")


# --- Modified by ClaudeO: TTS placeholder endpoint ---
@app.post("/v1/audio/tts", tags=["Multimodal"])
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("default"),
    language: str = Form("en"),
):
    """
    Text-to-Speech endpoint.
    Note: Gemma 4 E4B is text-output only. TTS requires an external engine.
    This endpoint integrates with system TTS (Windows SAPI or piper-tts).
    For production, connect piper-tts or Coqui TTS as a sidecar service.
    """
    # Placeholder: In production, pipe through piper-tts or Windows SAPI
    return {
        "status": "tts_requires_external_engine",
        "text": text,
        "suggestion": (
            "Install piper-tts: pip install piper-tts\n"
            "Or use Windows SAPI via PowerShell:\n"
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('text')"
        ),
        "recommended_engines": [
            {"name": "piper-tts", "url": "https://github.com/rhasspy/piper", "quality": "high", "local": True},
            {"name": "Coqui TTS", "url": "https://github.com/coqui-ai/TTS", "quality": "high", "local": True},
            {"name": "Windows SAPI", "url": "built-in", "quality": "medium", "local": True},
        ],
    }


# --- Modified by ClaudeO: Agent management endpoints ---
@app.get("/v1/agents", tags=["Agents"])
async def list_agents():
    """List all registered AI agents with their configurations."""
    return {
        "agents": [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "temperature": a.temperature,
            }
            for a in AGENT_REGISTRY.values()
        ]
    }


@app.post("/v1/agents", tags=["Agents"])
async def register_agent(config: AgentConfig):
    """Register a new agent or update an existing one."""
    AGENT_REGISTRY[config.agent_id] = config
    logger.info(f"Agent registered: {config.agent_id} ({config.name})")
    return {"status": "registered", "agent_id": config.agent_id}


@app.delete("/v1/agents/{agent_id}", tags=["Agents"])
async def remove_agent(agent_id: str):
    """Remove a registered agent."""
    if agent_id in AGENT_REGISTRY:
        del AGENT_REGISTRY[agent_id]
        return {"status": "removed", "agent_id": agent_id}
    raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")


# --- Modified by ClaudeO: Model management ---
@app.get("/v1/models", tags=["Models"])
async def list_models():
    """List all available Ollama models (OpenAI-compatible)."""
    try:
        models = await ollama_client.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m.get("name", "unknown"),
                    "object": "model",
                    "owned_by": "local",
                    "size_bytes": m.get("size", 0),
                }
                for m in models
            ],
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {str(e)}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info(f"Starting Gemma 4 E4B API Server on port {API_PORT}")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Memory mode: optimized (ctx={MAX_CONTEXT_TOKENS})")
    uvicorn.run(
        "gemma4_api_server:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
        log_level="info",
    )
