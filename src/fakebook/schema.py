"""Intermediate representation (IR) — the contract between analysis and the LLM.

Every upstream stage populates a slice of this document; the ``interpret`` stage
consumes *only* this document. The models below mirror the schema in the build
spec 1:1 and are the single source of truth: the JSON Schema shipped in
``fakebook/schema/ir.schema.json`` is generated from these classes
(see :func:`export_json_schema`).

Design notes tied to the core principles:

* ``Segment.pitch_classes`` is the pc-set (collapses enharmonics) while
  ``spelled_candidates`` keeps spelled pitch — both are carried through, per
  principle 4 (♯9-vs-♭3 must survive).
* ``chord_interpretations`` / ``scale_candidates`` are *enumerated by the
  deterministic kernel*. They are the guardrail handed to the LLM (principle 2);
  the LLM selects among them and never derives new ones.
* ``coarse_acr`` is explicitly labelled known-lossy context (principle 3).
* ``interpretation`` is written only by the ``interpret`` stage.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Stacking = Literal["tertian", "quartal", "cluster", "unknown"]


class _Model(BaseModel):
    """Base with a strict-ish config shared by every IR node."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Global / meta                                                               #
# --------------------------------------------------------------------------- #
class Meta(_Model):
    duration_s: float = 0.0
    sample_rate: int = 0
    source: str = ""


class Key(_Model):
    tonic: str = ""
    mode: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Global(_Model):
    key: Key = Field(default_factory=Key)
    tempo_bpm: float = 0.0
    time_signature: str = ""
    tags: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Timeline                                                                     #
# --------------------------------------------------------------------------- #
class Beat(_Model):
    t: float = 0.0
    is_downbeat: bool = False
    bar: int = 0


class Section(_Model):
    label: str = ""
    start_s: float = 0.0
    end_s: float = 0.0


# --------------------------------------------------------------------------- #
# Harmony segment                                                             #
# --------------------------------------------------------------------------- #
class PcSet(_Model):
    """Pitch-class-set descriptors (music21 terminology)."""

    prime_form: str = ""
    forte: str = ""
    interval_vector: str = ""
    normal_order: str = ""


class ChordInterpretation(_Model):
    """One enumerated, deterministic reading. The guardrail unit."""

    name: str
    root: str = ""
    quality: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Which supplied pitch classes are chord tones vs. added tensions — lets the
    # LLM ground extension/alteration claims without recomputing anything.
    chord_tones: list[int] = Field(default_factory=list)
    tensions: list[int] = Field(default_factory=list)


class ScaleCandidate(_Model):
    name: str
    fit: float = Field(default=0.0, ge=0.0, le=1.0)
    # pitch classes present in the segment that fall outside this scale
    out_of_scale: list[int] = Field(default_factory=list)


class Voicing(_Model):
    stacking: Stacking = "unknown"
    upper_structure: Optional[str] = None
    omitted: list[str] = Field(default_factory=list)
    inversion: str = ""


class TimbreDescriptor(_Model):
    """Branch B (sound-design) descriptor, measured by the ``timbre`` stage.

    Each of the four things the spec calls "complex voicing" in the sound-design
    sense has a field: ``detune_cents`` (detuning), ``centroid_slope_hz_per_s``
    (filter movement), ``inharmonicity`` (FM/metallic), ``n_partials``
    (layering). ``tags`` are deterministic labels thresholded off those numbers —
    computed, never chosen by the LLM.
    """

    spectral_centroid_hz: Optional[float] = None  # brightness
    spectral_bandwidth_hz: Optional[float] = None  # spread about the centroid
    spectral_flatness: Optional[float] = None  # ~0 tonal, ->1 noise-like
    centroid_slope_hz_per_s: Optional[float] = None  # filter movement over the span
    inharmonicity: Optional[float] = None  # deviation from the dominant series
    harmonic_ratio: Optional[float] = None  # 1.0 = one harmonic source explains it
    detune_cents: Optional[float] = None  # unison spread; a lower bound
    n_partials: Optional[int] = None  # resolved partials (layering density)
    f0_hz: Optional[float] = None  # HPS estimate the deviations are measured against
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class Segment(_Model):
    start_s: float = 0.0
    end_s: float = 0.0
    bar: int = 0
    pitch_classes: list[int] = Field(default_factory=list)
    spelled_candidates: list[str] = Field(default_factory=list)
    pc_set: PcSet = Field(default_factory=PcSet)
    chord_interpretations: list[ChordInterpretation] = Field(default_factory=list)
    scale_candidates: list[ScaleCandidate] = Field(default_factory=list)
    voicing: Voicing = Field(default_factory=Voicing)
    coarse_acr: str = ""  # context only; known-lossy (principle 3)
    timbre: Optional[TimbreDescriptor] = None  # Branch B fills this


# --------------------------------------------------------------------------- #
# Interpretation (LLM output)                                                 #
# --------------------------------------------------------------------------- #
class InterpretedSegment(_Model):
    """One LLM reading, grounded in a segment + its enumerated candidates."""

    segment_index: int
    chosen_interpretation: str = ""
    rationale: str = ""
    competing_hearings: list[str] = Field(default_factory=list)
    idioms: list[str] = Field(default_factory=list)
    # True when the enumerated candidates did not support any confident reading;
    # the LLM must say so rather than fabricate (guardrail).
    unsupported: bool = False


class Interpretation(_Model):
    summary: str = ""
    segments: list[InterpretedSegment] = Field(default_factory=list)
    model: str = ""
    grounded: bool = True


# --------------------------------------------------------------------------- #
# Root document                                                               #
# --------------------------------------------------------------------------- #
class AnalysisDocument(_Model):
    """The full IR document. This is what ``assemble`` validates and emits."""

    meta: Meta = Field(default_factory=Meta)
    global_: Global = Field(default_factory=Global, alias="global")
    beats: list[Beat] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    interpretation: Optional[Interpretation] = None

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True
    )


def export_json_schema() -> dict:
    """Return the JSON Schema for :class:`AnalysisDocument`.

    ``global`` (a Python keyword) is exposed under its aliased name so the
    emitted JSON matches the spec.
    """

    return AnalysisDocument.model_json_schema(by_alias=True)
