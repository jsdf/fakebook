"""End-to-end pipeline smoke test using in-memory fake stages.

Exercises the whole wiring (separation → mir → segmentation → salience → kernel
→ assemble) with no audio/model dependencies, proving the contract seams hold
and the assembled document validates.
"""

import numpy as np

from fakebook.assemble import Assembler
from fakebook.audio import AudioClip, Stems
from fakebook.config import Config
from fakebook.contracts import (
    ChordSpan,
    MirResult,
    PitchClassObservation,
)
from fakebook.harmony import HarmonyKernel
from fakebook.pipeline import Pipeline
from fakebook.schema import Beat, Global, Key
from fakebook.timbre import TimbreStub


class FakeIngestor:
    def load(self, path):
        return AudioClip(np.zeros((1, 44100 * 4), dtype=np.float32), 44100, path)


class FakeSeparator:
    def separate(self, clip):
        return Stems(harmonic=clip, mix=clip)


class FakeMir:
    def analyze(self, clip, stems):
        return MirResult(
            global_=Global(key=Key(tonic="C", mode="minor", confidence=0.6), tempo_bpm=120.0,
                           time_signature="4/4"),
            beats=[
                Beat(t=0.0, is_downbeat=True, bar=1),
                Beat(t=2.0, is_downbeat=True, bar=2),
            ],
            sections=[],
            coarse_chords=[(0.0, 2.0, "C:min"), (2.0, 4.0, "F:min")],
        )


class FakeSalience:
    """Returns fixed pitch-class collections per span."""

    _sets = [[0, 3, 7, 10], [5, 8, 0, 3]]

    def observe(self, harmonic, spans):
        out = []
        for i, span in enumerate(spans):
            pcs = self._sets[i % len(self._sets)]
            out.append(PitchClassObservation(span=span, pitch_classes=pcs, weights=[]))
        return out


def _pipeline(config=None):
    config = config or Config.default()
    return Pipeline(
        config=config,
        ingestor=FakeIngestor(),
        separator=FakeSeparator(),
        mir=FakeMir(),
        salience=FakeSalience(),
        kernel=HarmonyKernel(config),
        assembler=Assembler(config),
        timbre=None,
    )


def test_pipeline_produces_valid_document():
    doc = _pipeline().analyze_file("fake.wav")
    assert len(doc.segments) == 2
    assert doc.segments[0].chord_interpretations[0].name == "Cm7"
    assert doc.meta.duration_s == 4.0
    # coarse ACR attached as context
    assert doc.segments[0].coarse_acr == "C:min"


def test_pipeline_segments_align_to_downbeats():
    pipe = _pipeline()
    doc = pipe.analyze_file("fake.wav")
    assert doc.segments[0].start_s == 0.0
    assert doc.segments[1].start_s == 2.0
    report = pipe.assembler.last_report
    assert report is not None and report.ok, report.warnings if report else None


def test_pipeline_with_timbre_stub_fills_slot():
    config = Config.default()
    pipe = _pipeline(config)
    pipe.timbre = TimbreStub(config)
    doc = pipe.analyze_file("fake.wav")
    assert doc.segments[0].timbre is not None
    assert "stub" in doc.segments[0].timbre.notes


def test_from_config_builds_without_heavy_deps():
    # Should construct with fallbacks even though audio/model extras are absent.
    pipe = Pipeline.from_config(Config.default())
    assert pipe.kernel is not None
    assert pipe.assembler is not None
