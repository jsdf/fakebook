import json

import jsonschema
import pytest

from fakebook.schema import (
    AnalysisDocument,
    ChordInterpretation,
    Global,
    Key,
    Meta,
    Segment,
    export_json_schema,
)


def test_export_schema_has_global_alias():
    schema = export_json_schema()
    props = schema["properties"]
    assert "global" in props  # aliased, not "global_"
    assert "global_" not in props
    for key in ("meta", "beats", "sections", "segments", "interpretation"):
        assert key in props


def test_document_roundtrips_by_alias():
    doc = AnalysisDocument(
        meta=Meta(duration_s=12.0, sample_rate=44100, source="x.wav"),
        global_=Global(key=Key(tonic="C", mode="minor", confidence=0.8), tempo_bpm=120.0),
        segments=[Segment(start_s=0, end_s=2, pitch_classes=[0, 3, 7, 10])],
    )
    payload = doc.model_dump(by_alias=True, mode="json")
    assert "global" in payload
    reloaded = AnalysisDocument.model_validate(payload)
    assert reloaded.global_.key.tonic == "C"


def test_document_validates_against_json_schema():
    doc = AnalysisDocument(
        meta=Meta(duration_s=1.0, sample_rate=44100),
        segments=[
            Segment(
                start_s=0,
                end_s=1,
                pitch_classes=[0, 4, 7],
                chord_interpretations=[ChordInterpretation(name="Cmaj", root="C", quality="maj", score=0.9)],
            )
        ],
    )
    payload = doc.model_dump(by_alias=True, mode="json")
    jsonschema.validate(payload, export_json_schema())


def test_extra_fields_forbidden():
    with pytest.raises(Exception):
        Segment(start_s=0, end_s=1, bogus_field=1)  # type: ignore[call-arg]


def test_confidence_bounds_enforced():
    with pytest.raises(Exception):
        Key(tonic="C", mode="major", confidence=1.5)
