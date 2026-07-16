"""Typed stage interfaces — the I/O contract between every stage.

Each stage is a small object implementing one of the Protocols below. The
orchestrator (:mod:`fakebook.pipeline`) depends only on these Protocols, so any
stage can be swapped (real model ↔ fallback ↔ stub) without touching the rest of
the pipeline. This is the "stable interface" the spec asks for — notably it is
what lets Branch B (``timbre``) be filled in later without ripples.

The dataclasses here carry the intermediate results that are *not* yet the final
IR JSON; :mod:`fakebook.assemble` folds them into an
:class:`~fakebook.schema.AnalysisDocument`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .audio import AudioClip, Stems
from .schema import (
    Beat,
    Global,
    Interpretation,
    Section,
    Segment,
    TimbreDescriptor,
)


# --------------------------------------------------------------------------- #
# Intermediate results                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MirResult:
    """Whole-mix / stem MIR features → populates ``global``/``beats``/``sections``."""

    global_: Global = field(default_factory=Global)
    beats: list[Beat] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    # Coarse ACR spans as (start_s, end_s, "C:min") — lossy context (principle 3).
    coarse_chords: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class ChordSpan:
    """A time span over which harmony is treated as stable (a chord segment)."""

    start_s: float
    end_s: float
    bar: int = 0


@dataclass
class PitchClassObservation:
    """Thresholded pitch-class collection for one chord span, pre-kernel."""

    span: ChordSpan
    pitch_classes: list[int]
    # per-pc relative salience/energy in [0, 1], parallel to ``pitch_classes``
    weights: list[float] = field(default_factory=list)
    coarse_acr: str = ""


# --------------------------------------------------------------------------- #
# Stage protocols                                                             #
# --------------------------------------------------------------------------- #
@runtime_checkable
class Ingestor(Protocol):
    def load(self, path: str) -> AudioClip: ...


@runtime_checkable
class Separator(Protocol):
    def separate(self, clip: AudioClip) -> Stems: ...


@runtime_checkable
class MirAnalyzer(Protocol):
    def analyze(self, clip: AudioClip, stems: Stems) -> MirResult: ...


@runtime_checkable
class SalienceExtractor(Protocol):
    """Harmonic stem + chord spans → per-span thresholded pitch classes."""

    def observe(
        self, harmonic: AudioClip, spans: list[ChordSpan]
    ) -> list[PitchClassObservation]: ...


@runtime_checkable
class HarmonyKernel(Protocol):
    """Deterministic: pitch-class observation → fully enumerated Segment.

    This is the hard guardrail boundary (principle 2). It performs *all* pitch /
    interval / chord / scale arithmetic. Nothing downstream computes pitches.
    """

    def enumerate_segment(self, obs: PitchClassObservation) -> Segment: ...


@runtime_checkable
class TimbreAnalyzer(Protocol):
    """Branch B. Harmonic-stem span → spectral descriptor."""

    def describe(self, harmonic: AudioClip, span: ChordSpan) -> TimbreDescriptor: ...


@runtime_checkable
class Interpreter(Protocol):
    """LLM over the grounded IR → interpretation. Never derives pitches."""

    def interpret(self, document: dict) -> Interpretation: ...


__all__ = [
    "MirResult",
    "ChordSpan",
    "PitchClassObservation",
    "Ingestor",
    "Separator",
    "MirAnalyzer",
    "SalienceExtractor",
    "HarmonyKernel",
    "TimbreAnalyzer",
    "Interpreter",
    "Segment",
    "Optional",
]
