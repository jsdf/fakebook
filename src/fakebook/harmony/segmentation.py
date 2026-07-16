"""Segment the harmonic timeline into chord spans.

Chord spans are the unit the salience extractor and kernel operate on. Three
strategies (config ``harmony.segment_by``):

* ``downbeat`` — one span per bar, from the MIR downbeat grid. Aligning chord
  spans to downbeats is also a coherence check the assembler reuses.
* ``harmonic_change`` — split at chroma novelty peaks (self-similarity), so a
  bar with two chords is not smeared.
* ``fixed`` — uniform windows (fallback when no beat grid is available).
"""

from __future__ import annotations

from ..config import Config
from ..contracts import ChordSpan
from ..schema import Beat


def spans_from_downbeats(beats: list[Beat], duration_s: float) -> list[ChordSpan]:
    downs = [b for b in beats if b.is_downbeat]
    if not downs:
        return []
    spans: list[ChordSpan] = []
    for i, b in enumerate(downs):
        end = downs[i + 1].t if i + 1 < len(downs) else duration_s
        if end > b.t:
            spans.append(ChordSpan(start_s=b.t, end_s=end, bar=b.bar))
    return spans


def spans_fixed(duration_s: float, window_s: float) -> list[ChordSpan]:
    spans: list[ChordSpan] = []
    t = 0.0
    bar = 1
    while t < duration_s:
        end = min(t + window_s, duration_s)
        spans.append(ChordSpan(start_s=t, end_s=end, bar=bar))
        t = end
        bar += 1
    return spans


def build_spans(
    beats: list[Beat], duration_s: float, config: Config | None = None
) -> list[ChordSpan]:
    config = config or Config.default()
    mode = config.get("harmony.segment_by", "downbeat")
    if mode == "downbeat":
        spans = spans_from_downbeats(beats, duration_s)
        if spans:
            return spans
    # harmonic_change is handled inside the salience extractor (needs audio);
    # fall back to fixed windows when no usable downbeat grid exists.
    window = float(config.get("harmony.fixed_segment_s", 2.0))
    return spans_fixed(duration_s, window)
