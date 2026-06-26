"""
Tool Session Manager
Manages a single llama-server.exe instance at a time (hot-swap by chunk_type).
GT 1030 850MB: can't run multiple servers simultaneously.
"""

import asyncio
import os
import subprocess
import time
import httpx
from dataclasses import dataclass
from pathlib import Path

LLAMA_SERVER = Path(__file__).parent.parent / "bin" / "llama-cpp" / "llama-server.exe"
MODEL_BASE    = Path("E:/LLMmodel")

# Per-model configs (path, ngl) at 850MB standard baseline
TOOL_MODELS = {
    "coding": {
        "path": MODEL_BASE / "Qwen2.5-Coder-1.5B" / "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
        "ngl": 28,
        "ctx": 4096,
        "port": 8081,
        "desc": "Qwen2.5-Coder-1.5B Q8 — 100% VRAM",
    },
    "reasoning": {
        "path": MODEL_BASE / "Qwen3-1.7B" / "qwen3-1.7b-q4_k_m.gguf",
        "ngl": 25,
        "ctx": 8192,
        "port": 8081,
        "desc": "Qwen3-1.7B Q4 — thinking mode, 89% VRAM",
    },
    "debug": {
        "path": MODEL_BASE / "DeepSeek-R1-1.5B" / "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        "ngl": 25,
        "ctx": 4096,
        "port": 8081,
        "desc": "DeepSeek-R1-1.5B Q8 — reasoning distill, 89% VRAM",
    },
    "heavy_coding": {
        "path": MODEL_BASE / "Qwen2.5-Coder-3B" / "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "ngl": 16,
        "ctx": 4096,
        "port": 8081,
        "desc": "Qwen2.5-Coder-3B Q4 — heavier coding, 44% VRAM",
    },
    "moe_expert": {
        "path": MODEL_BASE / "DeepSeek-Coder-V2-Lite" / "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "ngl": 13,
        "ctx": 4096,
        "port": 8081,
        "desc": "DeepSeek-Coder-V2-Lite Q4 — 16B MoE, 2.4B active, Spec-Experts target",
    },
}

# chunk_type → preferred tool model
CHUNK_ROUTE = {
    "code":      "coding",
    "reasoning": "reasoning",
    "factual":   "coding",     # coder models are good at factual too
    "creative":  "reasoning",
    "unknown":   "coding",
}


@dataclass
class ToolSession:
    model_key: str
    proc: subprocess.Popen
    port: int
    started_at: float


class ToolSessionManager:
    def __init__(self):
        self._current: ToolSession | None = None

    def _find_gguf(self, cfg: dict) -> Path | None:
        p = Path(cfg["path"])
        if p.exists():
            return p
        # glob fallback if filename differs slightly
        parent = p.parent
        if parent.exists():
            matches = sorted(parent.glob("*.gguf"))
            if matches:
                return matches[0]
        return None

    def _kill_current(self):
        if self._current is None:
            return
        print(f"[ToolSession] stopping {self._current.model_key}")
        try:
            self._current.proc.terminate()
            self._current.proc.wait(timeout=10)
        except Exception:
            self._current.proc.kill()
        self._current = None
        time.sleep(2)  # let VRAM drain

    async def _wait_ready(self, port: int, timeout: int = 60) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                try:
                    r = await client.get(url, timeout=2)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1)
        return False

    async def ensure(self, chunk_type: str) -> int | None:
        """Ensure correct model running for chunk_type. Returns port or None."""
        model_key = CHUNK_ROUTE.get(chunk_type, "coding")
        cfg = TOOL_MODELS[model_key]

        if self._current and self._current.model_key == model_key:
            return self._current.port  # already running

        self._kill_current()

        gguf = self._find_gguf(cfg)
        if not gguf:
            print(f"[ToolSession] model not found: {cfg['path']}")
            return None

        print(f"[ToolSession] starting {model_key}: {cfg['desc']}")
        cmd = [
            str(LLAMA_SERVER),
            "--model", str(gguf),
            "--n-gpu-layers", str(cfg["ngl"]),
            "--ctx-size", str(cfg["ctx"]),
            "--threads", "2",
            "--port", str(cfg["port"]),
            "--n-predict", "512",
            "--log-disable",  # reduce noise; remove to enable logprob monitor
            "--no-warmup",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current = ToolSession(
            model_key=model_key,
            proc=proc,
            port=cfg["port"],
            started_at=time.time(),
        )
        ready = await self._wait_ready(cfg["port"], timeout=90)
        if not ready:
            print(f"[ToolSession] server did not become ready in 90s")
            self._kill_current()
            return None

        print(f"[ToolSession] ready: {model_key} on :{cfg['port']}")
        return cfg["port"]

    async def infer(self, chunk_type: str, prompt: str, system: str = "") -> str | None:
        port = await self.ensure(chunk_type)
        if port is None:
            return None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"messages": messages, "max_tokens": 512, "stream": False},
                )
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[ToolSession] infer error: {e}")
                return None

    def shutdown(self):
        self._kill_current()
