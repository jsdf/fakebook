"""Assemble the IR document — the contract between analysis and the LLM.

Folds the MIR result, harmonic segments, meta, and (optionally) timbre into one
:class:`~fakebook.schema.AnalysisDocument`, validates it against the shipped JSON
Schema, and runs coherence checks (the spec's validation tier for material with
scarce ground truth):

* do chord segments align to the downbeat grid?
* are pitch-class sets stable within a segment (single dominant reading)?
* do sections/segments stay within the clip duration?

Coherence findings are advisory (attached to the returned report), not fatal —
they surface likely upstream problems without blocking the document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

from ..config import Config
from ..contracts import MirResult
from ..errors import SchemaValidationError
from ..schema import AnalysisDocument, Meta, Segment


@dataclass
class CoherenceReport:
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.ok = False
        self.warnings.append(msg)


def _load_schema() -> dict:
    text = resources.files("fakebook").joinpath("schema/ir.schema.json").read_text()
    return json.loads(text)


def coherence_checks(doc: AnalysisDocument, tol_s: float = 0.08) -> CoherenceReport:
    report = CoherenceReport()
    duration = doc.meta.duration_s or 0.0
    downbeats = [b.t for b in doc.beats if b.is_downbeat]

    # 1. segment/downbeat alignment
    if downbeats:
        for i, seg in enumerate(doc.segments):
            nearest = min(downbeats, key=lambda d: abs(d - seg.start_s))
            if abs(nearest - seg.start_s) > tol_s:
                report.warn(
                    f"segment {i} start {seg.start_s:.3f}s not aligned to a downbeat "
                    f"(nearest {nearest:.3f}s)"
                )

    # 2. within-duration bounds
    if duration:
        for i, seg in enumerate(doc.segments):
            if seg.end_s > duration + tol_s:
                report.warn(f"segment {i} ends at {seg.end_s:.3f}s beyond clip {duration:.3f}s")
        for i, sec in enumerate(doc.sections):
            if sec.end_s > duration + tol_s:
                report.warn(f"section {i} ends beyond clip duration")

    # 3. pc-set stability proxy: an empty pc collection on a non-trivial span is
    #    suspicious (salience found nothing where a chord was expected).
    for i, seg in enumerate(doc.segments):
        if (seg.end_s - seg.start_s) >= 0.5 and not seg.pitch_classes:
            report.warn(f"segment {i} has no pitch classes over a {seg.end_s - seg.start_s:.2f}s span")

    return report


class Assembler:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.validate = bool(self.config.get("assemble.validate", True))
        self.run_coherence = bool(self.config.get("assemble.coherence_checks", True))
        self._last_report: CoherenceReport | None = None

    @property
    def last_report(self) -> CoherenceReport | None:
        return self._last_report

    def assemble(
        self,
        meta: Meta,
        mir: MirResult,
        segments: list[Segment],
    ) -> AnalysisDocument:
        doc = AnalysisDocument(
            meta=meta,
            global_=mir.global_,
            beats=list(mir.beats),
            sections=list(mir.sections),
            segments=list(segments),
            interpretation=None,
        )
        # attach coarse ACR spans onto overlapping segments (context only).
        self._attach_coarse_acr(doc, mir)

        if self.run_coherence:
            self._last_report = coherence_checks(doc)
        if self.validate:
            self._validate(doc)
        return doc

    def _attach_coarse_acr(self, doc: AnalysisDocument, mir: MirResult) -> None:
        if not mir.coarse_chords:
            return
        for seg in doc.segments:
            if seg.coarse_acr:
                continue
            mid = (seg.start_s + seg.end_s) / 2
            for start, end, label in mir.coarse_chords:
                if start <= mid < end:
                    seg.coarse_acr = label
                    break

    def _validate(self, doc: AnalysisDocument) -> None:
        import jsonschema

        payload = doc.model_dump(by_alias=True, mode="json")
        try:
            jsonschema.validate(payload, _load_schema())
        except jsonschema.ValidationError as exc:  # pragma: no cover - defensive
            raise SchemaValidationError(str(exc)) from exc

    @staticmethod
    def to_json(doc: AnalysisDocument, indent: int = 2) -> str:
        return json.dumps(doc.model_dump(by_alias=True, mode="json"), indent=indent)
