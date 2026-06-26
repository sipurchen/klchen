"""
Phase 3 — Directional Steering
Activation-space goal-directed behavior control via residual stream perturbation.
Ported from pi-ds4 concept; adapted for llama.cpp HTTP inference.

Since llama.cpp HTTP API does not expose activations directly, we implement
steering via:
  1. System prompt prepend (soft steering) — always available
  2. Temperature / logit_bias manipulation — via completion params
  3. Prefix injection (activation proxy) — via prompt engineering
  4. Future: ggml hook → direct residual addition (requires custom build)
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional
import httpx


@dataclass
class SteeringVector:
    """A direction in activation space associated with a concept."""
    concept: str
    direction_prompt: str     # positive exemplar prompt
    anti_prompt: str          # negative exemplar prompt (contrast)
    layer: int = 16           # target layer for injection (middle layers most effective)
    alpha: float = 15.0       # steering strength (additive scale)
    logit_bias: dict = field(default_factory=dict)  # token-level bias for proxy steering


# Predefined steering vectors for common chunk types
STEERING_LIBRARY: dict[str, SteeringVector] = {
    "concise": SteeringVector(
        concept="concise",
        direction_prompt="Answer in one sentence only. Be extremely brief.",
        anti_prompt="Give a very long detailed explanation with many examples.",
        alpha=10.0,
        logit_bias={},
    ),
    "reasoning": SteeringVector(
        concept="reasoning",
        direction_prompt="Think step by step. Show your reasoning process carefully.",
        anti_prompt="Just give the answer without explanation.",
        alpha=12.0,
    ),
    "code_quality": SteeringVector(
        concept="code_quality",
        direction_prompt="Write clean, well-structured, efficient Python code with type hints.",
        anti_prompt="Write messy undocumented code.",
        alpha=15.0,
    ),
    "factual": SteeringVector(
        concept="factual",
        direction_prompt="Give accurate, verifiable factual information. Cite sources.",
        anti_prompt="Speculate and make things up.",
        alpha=8.0,
    ),
    "creative": SteeringVector(
        concept="creative",
        direction_prompt="Be creative, original, and surprising in your response.",
        anti_prompt="Give a boring, predictable, conventional response.",
        alpha=10.0,
    ),
}


@dataclass
class SteeringConfig:
    """Active steering configuration for a chunk inference."""
    vectors: list[str]           # which concepts to steer toward
    strength: float = 1.0        # global strength multiplier
    mode: str = "system_prompt"  # "system_prompt" | "logit_bias" | "prefix"
    temperature_adjust: float = 0.0  # delta applied to base temperature


class DirectionalSteering:
    """
    Apply directional steering to llama-server inference.

    Without activation access, we use:
    - Composite system prompts that combine steering directions
    - Temperature adjustment per chunk type
    - Logit bias for token-level nudging
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url
        self._library = STEERING_LIBRARY.copy()

    def register_vector(self, sv: SteeringVector):
        self._library[sv.concept] = sv

    def compose_system_prompt(
        self,
        base_system: str,
        concepts: list[str],
        strength: float = 1.0,
    ) -> str:
        """Build composite system prompt from steering concepts."""
        if not concepts:
            return base_system

        steering_lines = []
        for c in concepts:
            sv = self._library.get(c)
            if sv:
                # Weight the direction by alpha * strength
                effective = sv.alpha * strength
                # High alpha → prepend as strong instruction
                if effective >= 10.0:
                    steering_lines.insert(0, sv.direction_prompt)
                else:
                    steering_lines.append(sv.direction_prompt)

        composed = "\n".join(steering_lines)
        if base_system:
            return f"{composed}\n\n{base_system}"
        return composed

    def compute_logit_bias(
        self,
        concepts: list[str],
        strength: float = 1.0,
    ) -> dict:
        """Merge logit_bias from all active steering vectors."""
        merged = {}
        for c in concepts:
            sv = self._library.get(c)
            if sv and sv.logit_bias:
                for token_id, bias in sv.logit_bias.items():
                    merged[token_id] = merged.get(token_id, 0) + bias * strength
        return merged

    def get_temperature(
        self,
        base_temp: float,
        concepts: list[str],
    ) -> float:
        """Adjust temperature based on chunk type."""
        # Reasoning/code: lower temp (more deterministic)
        # Creative: higher temp
        delta = 0.0
        if "reasoning" in concepts or "code_quality" in concepts:
            delta -= 0.2
        if "creative" in concepts:
            delta += 0.3
        if "concise" in concepts:
            delta -= 0.1
        return max(0.0, min(2.0, base_temp + delta))

    async def steer_infer(
        self,
        prompt: str,
        concepts: list[str],
        base_system: str = "/no_think",
        base_temperature: float = 0.7,
        max_tokens: int = 1024,
        port: int = 8080,
        timeout: float = 180,
    ) -> dict:
        """
        Run inference with directional steering applied.
        Returns {content, steering_info}.
        """
        system = self.compose_system_prompt(base_system, concepts)
        logit_bias = self.compute_logit_bias(concepts)
        temperature = self.get_temperature(base_temperature, concepts)

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if logit_bias:
            payload["logit_bias"] = logit_bias

        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            data = r.json()

        content = ""
        if "choices" in data and data["choices"]:
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""

        return {
            "content": content,
            "steering_info": {
                "concepts": concepts,
                "effective_temperature": temperature,
                "system_prompt_length": len(system),
                "logit_bias_tokens": len(logit_bias),
            },
        }

    def activation_proxy_prefix(self, concept: str) -> str:
        """
        Generate a prefix string that acts as a proxy for activation steering.
        Used when neither system prompt nor logit_bias is sufficient.
        Idea: prime the residual stream by front-loading concept-activating tokens.
        """
        sv = self._library.get(concept)
        if not sv:
            return ""
        # Abbreviated positive exemplar as context priming
        return f"[Approach: {sv.direction_prompt[:80]}]\n"

    def to_steer_params(self, chunk_type: str, strength: float = 1.0) -> SteeringConfig:
        """Map chunk_type to steering config."""
        mapping = {
            "reasoning": SteeringConfig(["reasoning"], strength, "system_prompt"),
            "code":      SteeringConfig(["code_quality", "concise"], strength, "system_prompt"),
            "factual":   SteeringConfig(["factual", "concise"], strength, "system_prompt"),
            "creative":  SteeringConfig(["creative"], strength, "system_prompt"),
        }
        return mapping.get(chunk_type, SteeringConfig([], strength))
