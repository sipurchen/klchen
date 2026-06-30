"""
Phase 5 — Edge VLM Router
Visual Language Model deployment on edge devices.
Targets: GT 1030 (800MB), Jetson Nano (4GB), Android (1.5GB), ARM (1GB)

Routing strategy: vision chunk → VLM, text chunk → compact LLM
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import base64
import httpx


@dataclass
class VLMConfig:
    model_name: str
    gguf_path: str           # relative to LLMS_DIR
    backend: str             # "vulkan" | "metal" | "cuda" | "cpu"
    ngl: int
    ctx: int
    vram_mb: int
    fps_est: float           # estimated FPS for vision tasks
    device: str


# Edge VLM deployment targets
VLM_TARGETS = {
    "gt1030": VLMConfig(
        model_name="moondream-2B Q4_K_M",
        gguf_path="moondream2/moondream-2b-int4.gguf",
        backend="vulkan",
        ngl=8,
        ctx=2048,
        vram_mb=800,
        fps_est=1.5,
        device="GT 1030",
    ),
    "jetson_nano": VLMConfig(
        model_name="MobileVLM-1.7B Q4",
        gguf_path="MobileVLM/MobileVLM-1.7B-Q4_K_M.gguf",
        backend="cuda",
        ngl=32,
        ctx=2048,
        vram_mb=2000,
        fps_est=10.0,
        device="Jetson Nano 4GB",
    ),
    "android": VLMConfig(
        model_name="LLaVA-Phi-1.5 Q4",
        gguf_path="LLaVA-Phi/LLaVA-phi-1_5-Q4_K_M.gguf",
        backend="cpu",
        ngl=0,
        ctx=1024,
        vram_mb=0,
        fps_est=5.0,
        device="Android ARM",
    ),
    "arm_cam": VLMConfig(
        model_name="moondream-2B Q4 (ARM)",
        gguf_path="moondream2/moondream-2b-int4.gguf",
        backend="cpu",
        ngl=0,
        ctx=512,
        vram_mb=0,
        fps_est=2.0,
        device="ARM Surveillance",
    ),
}


class VLMRouter:
    """
    Routes chunks to VLM or text LLM based on content type.
    Handles base64 image encoding for llama.cpp mmproj API.
    """

    def __init__(
        self,
        target: str = "gt1030",
        llama_server_port: int = 8080,
        mmproj_path: Optional[str] = None,
    ):
        self.cfg = VLM_TARGETS.get(target)
        if self.cfg is None:
            raise ValueError(f"Unknown target: {target}. Known: {list(VLM_TARGETS)}")
        self.port = llama_server_port
        self.mmproj_path = mmproj_path

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for llama.cpp /completion with image."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def vision_infer(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
        max_tokens: int = 512,
        timeout: float = 120,
    ) -> str:
        """
        Run VLM inference. Uses llama.cpp /completion with image_data.
        Requires mmproj model loaded alongside text model.
        """
        if image_path and not image_b64:
            image_b64 = self.encode_image(image_path)

        payload: dict = {
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }
        if image_b64:
            payload["image_data"] = [{"data": image_b64, "id": 0}]
            payload["prompt"] = f"[INST] <image>\n{prompt} [/INST]"

        url = f"http://127.0.0.1:{self.port}/completion"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            data = r.json()
        return data.get("content", "")

    async def route_chunk(
        self,
        chunk_type: str,
        prompt: str,
        image_path: Optional[str] = None,
    ) -> dict:
        """
        Route: vision chunks → VLM, text chunks → fast text LLM.
        Returns {response, route, model}.
        """
        is_vision = chunk_type == "vision" or image_path is not None

        if is_vision:
            resp = await self.vision_infer(prompt, image_path=image_path)
            return {
                "response": resp,
                "route": "vlm",
                "model": self.cfg.model_name,
                "device": self.cfg.device,
            }
        else:
            # text-only: route to standard text completion
            url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, json=payload)
                data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "response": content,
                "route": "text",
                "model": f"{self.cfg.device}/text",
            }

    def startup_cmd(self, llms_dir: str) -> list[str]:
        """Generate llama-server launch command for this VLM target."""
        model_path = str(Path(llms_dir) / self.cfg.gguf_path)
        cmd = [
            "llama-server",
            "--model", model_path,
            "--n-gpu-layers", str(self.cfg.ngl),
            "--ctx-size", str(self.cfg.ctx),
            "--port", str(self.port),
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
        if self.cfg.backend == "vulkan":
            cmd += ["--gpu-layers", str(self.cfg.ngl)]
        return cmd

    def hardware_report(self) -> str:
        c = self.cfg
        return (
            f"VLM Target: {c.device}\n"
            f"  Model: {c.model_name}\n"
            f"  Backend: {c.backend} | ngl={c.ngl} | ctx={c.ctx}\n"
            f"  VRAM budget: {c.vram_mb} MB\n"
            f"  Estimated FPS: {c.fps_est}"
        )
