"""LLM interpretation over the grounded IR document.

Implements the :class:`~fakebook.contracts.Interpreter` protocol. Calls the
Anthropic Messages API (lazy import; needs the ``llm`` extra + an API key), then
*enforces the guardrail after the fact*: any chosen chord name that is not in the
segment's enumerated ``chord_interpretations`` is rejected and the segment is
marked unsupported. The prompt asks for compliance; this pass guarantees it, so a
hallucinated chord can never leak into the output.
"""

from __future__ import annotations

import json
import os

from ..config import Config
from ..errors import InterpretationError
from .._optional import require
from ..schema import Interpretation, InterpretedSegment
from .prompt import SYSTEM_PROMPT, build_user_prompt


class AnthropicInterpreter:
    def __init__(self, config: Config | None = None, api_key: str | None = None):
        self.config = config or Config.default()
        self.model = self.config.get("interpret.model", "claude-opus-4-8")
        self.max_tokens = int(self.config.get("interpret.max_tokens", 4096))
        self.temperature = float(self.config.get("interpret.temperature", 0.2))
        self.enforce = bool(self.config.get("interpret.enforce_grounding", True))
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def interpret(self, document: dict) -> Interpretation:
        anthropic = require("anthropic", feature="LLM interpretation", extra="llm")
        if not self._api_key:
            raise InterpretationError(
                "ANTHROPIC_API_KEY not set (needed for the interpret stage)."
            )
        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(document)}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return self.parse_and_enforce(text, document, model=self.model)

    # -- pure, testable core ------------------------------------------------ #
    def parse_and_enforce(
        self, text: str, document: dict, model: str = ""
    ) -> Interpretation:
        """Parse the model's JSON and enforce grounding. No network here."""
        data = _extract_json(text)
        segments = document.get("segments", [])

        interp = Interpretation(
            summary=str(data.get("summary", "")),
            model=model,
            grounded=True,
            segments=[],
        )
        for raw in data.get("segments", []):
            idx = int(raw.get("segment_index", -1))
            valid_names = _valid_names(segments, idx)
            chosen = str(raw.get("chosen_interpretation", "") or "")
            competing = [c for c in raw.get("competing_hearings", []) if c in valid_names]
            unsupported = bool(raw.get("unsupported", False))

            if self.enforce and chosen and chosen not in valid_names:
                # Guardrail violation: drop the fabricated reading.
                chosen = ""
                unsupported = True
                interp.grounded = True  # document stays grounded; claim was scrubbed

            interp.segments.append(
                InterpretedSegment(
                    segment_index=idx,
                    chosen_interpretation=chosen,
                    rationale=str(raw.get("rationale", "")),
                    competing_hearings=competing,
                    idioms=[str(x) for x in raw.get("idioms", [])],
                    unsupported=unsupported,
                )
            )
        return interp


def _valid_names(segments: list, idx: int) -> set[str]:
    if idx < 0 or idx >= len(segments):
        return set()
    return {c.get("name", "") for c in segments[idx].get("chord_interpretations", [])}


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise InterpretationError(f"No JSON object found in model output: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InterpretationError(f"Model output was not valid JSON: {exc}") from exc
