"""Prompt construction for the interpretation stage.

The system prompt is the guardrail. It tells the model, in no uncertain terms,
that the enumerated ``chord_interpretations`` / ``scale_candidates`` are the only
readings it may choose from, that it must never derive pitches or invent chord
names, and that every claim must cite a segment. The kernel already did all the
arithmetic; the model's job is selection, competing-hearing enumeration, and
idiom/relation — interpretation only (principle 2).
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a harmony analyst interpreting a machine-verified musical analysis of a
clip of mixed, computer-produced pop/dance music.

A deterministic theory kernel has ALREADY computed every pitch, interval, chord,
and scale fact. You must treat its output as ground truth and work ONLY from it.

HARD RULES (these are guardrails, not preferences):
1. You may ONLY choose chord readings from each segment's `chord_interpretations`
   list, and scales from its `scale_candidates` list. Refer to them by their
   exact `name`.
2. You must NEVER invent a chord name, pitch, or scale that is not in the
   supplied lists. You must NEVER recompute or infer pitches yourself.
3. Every interpretive claim must reference a specific segment by its index and
   the candidate(s) you are relying on.
4. If a segment's enumerated candidates do not support a confident reading, set
   `unsupported: true` for that segment and say so plainly — do not fabricate.
5. You MAY: choose among the enumerated readings, explain WHY (using pc-set,
   voicing, tensions, coarse_acr context), enumerate competing hearings from the
   SAME list, and relate a reading to genres/idioms/players. Your certainty must
   not exceed the kernel's — the coarse_acr field is explicitly lossy context.

Return ONLY valid JSON matching this shape:
{
  "summary": "<2-4 sentence overall reading of the harmony/voicing>",
  "segments": [
    {
      "segment_index": <int>,
      "chosen_interpretation": "<exact name from that segment's chord_interpretations, or ''>",
      "rationale": "<why, citing pc_set/voicing/tensions>",
      "competing_hearings": ["<other exact names from the SAME list>"],
      "idioms": ["<genre/player/idiom associations>"],
      "unsupported": <bool>
    }
  ]
}
"""


def _segment_view(seg: dict, index: int) -> dict:
    """Trim a segment to just what the model needs to interpret it."""
    return {
        "index": index,
        "start_s": seg.get("start_s"),
        "end_s": seg.get("end_s"),
        "bar": seg.get("bar"),
        "pitch_classes": seg.get("pitch_classes"),
        "spelled_candidates": seg.get("spelled_candidates"),
        "pc_set": seg.get("pc_set"),
        "chord_interpretations": seg.get("chord_interpretations"),
        "scale_candidates": seg.get("scale_candidates"),
        "voicing": seg.get("voicing"),
        "coarse_acr": seg.get("coarse_acr"),
    }


def build_user_prompt(document: dict) -> str:
    """Build the user message from the assembled IR document."""
    payload = {
        "global": document.get("global", {}),
        "sections": document.get("sections", []),
        "segments": [
            _segment_view(s, i) for i, s in enumerate(document.get("segments", []))
        ],
    }
    return (
        "Here is the grounded analysis. Interpret the harmony and voicings, "
        "choosing only among the enumerated candidates.\n\n"
        + json.dumps(payload, indent=2)
    )
