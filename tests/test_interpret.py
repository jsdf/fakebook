"""Tests for the interpret stage's guardrail enforcement (no network)."""

import json

import pytest

from fakebook.errors import InterpretationError
from fakebook.interpret import build_user_prompt
from fakebook.interpret.interpreter import AnthropicInterpreter


def _document():
    return {
        "global": {"key": {"tonic": "C", "mode": "minor"}},
        "sections": [],
        "segments": [
            {
                "start_s": 0,
                "end_s": 2,
                "bar": 1,
                "pitch_classes": [0, 3, 7, 10],
                "spelled_candidates": ["C Eb G Bb"],
                "pc_set": {"forte": "4-26"},
                "chord_interpretations": [
                    {"name": "Cm7", "root": "C", "quality": "m7", "score": 0.9},
                    {"name": "Eb6", "root": "Eb", "quality": "6", "score": 0.9},
                ],
                "scale_candidates": [{"name": "C dorian", "fit": 1.0}],
                "voicing": {"stacking": "tertian"},
                "coarse_acr": "C:min",
            }
        ],
    }


def test_prompt_includes_only_needed_fields():
    prompt = build_user_prompt(_document())
    assert "chord_interpretations" in prompt
    assert "Cm7" in prompt


def test_valid_selection_passes_through():
    interp = AnthropicInterpreter()
    text = json.dumps(
        {
            "summary": "A minor-seventh vamp.",
            "segments": [
                {
                    "segment_index": 0,
                    "chosen_interpretation": "Cm7",
                    "rationale": "tertian, b3+b7",
                    "competing_hearings": ["Eb6"],
                    "idioms": ["neo-soul"],
                    "unsupported": False,
                }
            ],
        }
    )
    result = interp.parse_and_enforce(text, _document(), model="test")
    seg = result.segments[0]
    assert seg.chosen_interpretation == "Cm7"
    assert seg.competing_hearings == ["Eb6"]
    assert not seg.unsupported


def test_fabricated_chord_is_scrubbed():
    interp = AnthropicInterpreter()
    text = json.dumps(
        {
            "summary": "x",
            "segments": [
                {
                    "segment_index": 0,
                    "chosen_interpretation": "Cm7b5add13",  # not enumerated
                    "rationale": "hallucinated",
                    "competing_hearings": ["Gmaj7"],  # not enumerated either
                    "idioms": [],
                    "unsupported": False,
                }
            ],
        }
    )
    result = interp.parse_and_enforce(text, _document(), model="test")
    seg = result.segments[0]
    assert seg.chosen_interpretation == ""  # scrubbed
    assert seg.unsupported is True
    assert seg.competing_hearings == []  # non-enumerated competitor dropped


def test_json_fenced_output_is_parsed():
    interp = AnthropicInterpreter()
    text = "```json\n" + json.dumps(
        {"summary": "ok", "segments": []}
    ) + "\n```"
    result = interp.parse_and_enforce(text, _document(), model="test")
    assert result.summary == "ok"


def test_non_json_raises():
    interp = AnthropicInterpreter()
    with pytest.raises(InterpretationError):
        interp.parse_and_enforce("I cannot help with that.", _document())


def test_unsupported_flag_preserved():
    interp = AnthropicInterpreter()
    text = json.dumps(
        {
            "summary": "unclear",
            "segments": [
                {"segment_index": 0, "chosen_interpretation": "", "unsupported": True}
            ],
        }
    )
    result = interp.parse_and_enforce(text, _document(), model="test")
    assert result.segments[0].unsupported is True
