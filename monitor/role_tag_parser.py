"""
P1.3 Role Tag Parser
Deterministic boundary detection from structural markers in CoT output.
Highest priority signal — overrides entropy/perplexity candidates.
"""

import re
from dataclasses import dataclass

ROLE_TAG_PATTERNS = [
    re.compile(r"</think>"),
    re.compile(r"<think>"),
    re.compile(r"<role:(\w+)>"),
    re.compile(r"\[INST\]|\[/INST\]"),
    re.compile(r"#{1,3} "),           # markdown heading = section jump
    re.compile(r"```"),               # code block toggle
]


@dataclass
class RoleTagBoundary:
    token_index: int
    tag: str
    boundary_type: str = "role_tag"
    priority: int = 10  # highest


class RoleTagParser:
    def __init__(self):
        self._token_index = 0
        self._text_buffer = ""

    def feed_text(self, text: str) -> list[RoleTagBoundary]:
        """Feed decoded text tokens, return deterministic boundaries found."""
        boundaries: list[RoleTagBoundary] = []
        self._text_buffer += text

        for pattern in ROLE_TAG_PATTERNS:
            for m in pattern.finditer(self._text_buffer):
                # rough token index: assume ~4 chars/token
                approx_token = self._token_index + m.start() // 4
                boundaries.append(
                    RoleTagBoundary(token_index=approx_token, tag=m.group(0))
                )

        self._text_buffer = ""  # reset after scan
        self._token_index += len(text) // 4
        return sorted(boundaries, key=lambda b: b.token_index)
