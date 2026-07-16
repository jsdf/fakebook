from fakebook.assemble import Assembler, coherence_checks
from fakebook.contracts import MirResult
from fakebook.schema import AnalysisDocument, Beat, Global, Key, Meta, Section, Segment


def _mir():
    return MirResult(
        global_=Global(key=Key(tonic="C", mode="minor", confidence=0.7), tempo_bpm=120.0,
                       time_signature="4/4"),
        beats=[
            Beat(t=0.0, is_downbeat=True, bar=1),
            Beat(t=0.5, is_downbeat=False, bar=1),
            Beat(t=2.0, is_downbeat=True, bar=2),
        ],
        sections=[Section(label="intro", start_s=0.0, end_s=4.0)],
        coarse_chords=[(0.0, 2.0, "C:min"), (2.0, 4.0, "F:min")],
    )


def _segments():
    return [
        Segment(start_s=0.0, end_s=2.0, bar=1, pitch_classes=[0, 3, 7, 10]),
        Segment(start_s=2.0, end_s=4.0, bar=2, pitch_classes=[5, 8, 0, 3]),
    ]


def test_assemble_produces_valid_document():
    asm = Assembler()
    doc = asm.assemble(Meta(duration_s=4.0, sample_rate=44100), _mir(), _segments())
    assert isinstance(doc, AnalysisDocument)
    assert len(doc.segments) == 2
    assert doc.global_.key.tonic == "C"


def test_coarse_acr_attached_from_spans():
    asm = Assembler()
    doc = asm.assemble(Meta(duration_s=4.0, sample_rate=44100), _mir(), _segments())
    assert doc.segments[0].coarse_acr == "C:min"
    assert doc.segments[1].coarse_acr == "F:min"


def test_coherence_flags_misaligned_segment():
    doc = AnalysisDocument(
        meta=Meta(duration_s=4.0, sample_rate=44100),
        beats=[Beat(t=0.0, is_downbeat=True, bar=1), Beat(t=2.0, is_downbeat=True, bar=2)],
        segments=[Segment(start_s=0.9, end_s=2.0, pitch_classes=[0, 4, 7])],
    )
    report = coherence_checks(doc)
    assert not report.ok
    assert any("not aligned" in w for w in report.warnings)


def test_coherence_flags_empty_pc_span():
    doc = AnalysisDocument(
        meta=Meta(duration_s=4.0, sample_rate=44100),
        beats=[Beat(t=0.0, is_downbeat=True, bar=1)],
        segments=[Segment(start_s=0.0, end_s=2.0, pitch_classes=[])],
    )
    report = coherence_checks(doc)
    assert any("no pitch classes" in w for w in report.warnings)


def test_coherence_clean_document_ok():
    asm = Assembler()
    doc = asm.assemble(Meta(duration_s=4.0, sample_rate=44100), _mir(), _segments())
    report = coherence_checks(doc)
    assert report.ok, report.warnings


def test_to_json_uses_alias():
    asm = Assembler()
    doc = asm.assemble(Meta(duration_s=4.0, sample_rate=44100), _mir(), _segments())
    text = asm.to_json(doc)
    assert '"global"' in text
    assert '"global_"' not in text
