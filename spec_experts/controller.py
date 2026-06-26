"""
Spec-Experts Controller Session
- Orchestrates monitor → chunk → route → tool session
- Exposes FastAPI on :8090
- Designed to wake every 2 hours via Windows Task Scheduler
- Shuts down cleanly after idle_timeout or explicit /shutdown
"""

import asyncio
import time
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from monitor.attention_entropy import EntropyMonitor
from monitor.perplexity_spike import PerplexitySpikeMonitor
from monitor.role_tag_parser import RoleTagParser
from monitor.signal_fusion import SignalFusion
from segmentor.chunk_classifier import classify_chunk, ChunkType
from spec_experts.tool_session import ToolSessionManager

CONTROLLER_PORT = 8090
IDLE_TIMEOUT_S  = 7200  # 2 hours — auto-shutdown if no request

_tool_mgr = ToolSessionManager()
_last_req  = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Controller] started on :{CONTROLLER_PORT}")
    asyncio.create_task(_idle_watchdog())
    yield
    print("[Controller] shutting down tool sessions")
    _tool_mgr.shutdown()


app = FastAPI(title="Spec-Experts Controller", lifespan=lifespan)


async def _idle_watchdog():
    while True:
        await asyncio.sleep(60)
        idle = time.time() - _last_req
        if idle > IDLE_TIMEOUT_S:
            print(f"[Controller] idle {idle:.0f}s > {IDLE_TIMEOUT_S}s — shutting down")
            _tool_mgr.shutdown()
            raise SystemExit(0)


# ── request models ────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt: str
    system: str = ""
    chunk_type: str = "auto"   # auto | code | reasoning | factual | creative | debug
    session_id: str = "default"


class ChunkRequest(BaseModel):
    text: str
    session_id: str = "default"


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_chunk_type(text: str) -> str:
    """Quick deterministic chunk type from role tags."""
    parser = RoleTagParser()
    bounds = parser.feed_text(text)
    if bounds:
        tag = bounds[0].tag
        if tag in ("<think>", "</think>"):
            return "reasoning"
        if tag == "```":
            return "code"
    # entropy-based: too expensive here, default to coding
    return "code"


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "idle_s": int(time.time() - _last_req)}


@app.post("/infer")
async def infer(req: InferRequest):
    global _last_req
    _last_req = time.time()

    chunk_type = req.chunk_type
    if chunk_type == "auto":
        chunk_type = _detect_chunk_type(req.prompt)

    result = await _tool_mgr.infer(chunk_type, req.prompt, req.system)
    if result is None:
        return JSONResponse({"error": "tool session failed"}, status_code=503)

    return {
        "chunk_type": chunk_type,
        "response": result,
        "model": _tool_mgr._current.model_key if _tool_mgr._current else "none",
    }


@app.post("/analyze_chunks")
async def analyze_chunks(req: ChunkRequest):
    """Detect chunk boundaries in text, classify each chunk."""
    global _last_req
    _last_req = time.time()

    parser  = RoleTagParser()
    fusion  = SignalFusion()
    ppl_mon = PerplexitySpikeMonitor()
    chunks  = []
    current_start = 0

    boundaries = parser.feed_text(req.text)
    for i, b in enumerate(boundaries):
        chunk_text = req.text[current_start:b.token_index * 4]  # approx char offset
        classified = classify_chunk(
            chunk_id=i,
            role_tag=b.tag,
            entropy_values=[1.5],  # placeholder without live logprobs
        )
        chunks.append({
            "chunk_id": i,
            "boundary_token": b.token_index,
            "boundary_type": b.boundary_type,
            "tag": b.tag,
            "chunk_type": classified.chunk_type,
            "route": classified.route,
        })
        current_start = b.token_index * 4

    return {"session_id": req.session_id, "chunks": chunks, "total": len(chunks)}


@app.post("/shutdown")
async def shutdown(background: BackgroundTasks):
    """Graceful shutdown — releases VRAM/CPU."""
    async def _do_shutdown():
        await asyncio.sleep(0.5)
        _tool_mgr.shutdown()
        raise SystemExit(0)
    background.add_task(_do_shutdown)
    return {"status": "shutting down"}


@app.get("/status")
async def status():
    current = _tool_mgr._current
    return {
        "active_model": current.model_key if current else None,
        "active_port":  current.port if current else None,
        "uptime_s":     int(time.time() - current.started_at) if current else 0,
        "idle_s":       int(time.time() - _last_req),
        "auto_shutdown_s": IDLE_TIMEOUT_S,
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "spec_experts.controller:app",
        host="127.0.0.1",
        port=CONTROLLER_PORT,
        log_level="info",
    )
