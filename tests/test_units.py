"""Unit tests for the smaller pure helpers."""

import numpy as np

from fakebook.audio import AudioClip, Stems
from fakebook.config import Config
from fakebook.contracts import ChordSpan
from fakebook.harmony.salience import threshold_chroma
from fakebook.harmony.segmentation import build_spans, spans_fixed, spans_from_downbeats
from fakebook.harmony.speller import spell
from fakebook.pitch import coarse_chord, estimate_key, pc_name
from fakebook.schema import Beat
from fakebook.timbre import TimbreStub


# -- speller ---------------------------------------------------------------- #
def test_speller_sharp_nine_vs_flat_three():
    # over C: interval 3 as a 9th -> D#, as a 3rd -> Eb
    assert spell(0, 3, 9) == "D#"
    assert spell(0, 3, 3) == "Eb"


def test_speller_altered_root():
    # over F#: minor third (interval 3) as degree 3 -> A natural
    assert spell(6, 3, 3) == "A"
    # over Eb: minor third -> Gb
    assert spell(3, 3, 3) == "Gb"


# -- pitch helpers ---------------------------------------------------------- #
def test_pc_name():
    assert pc_name(1) == "C#"
    assert pc_name(1, prefer_flats=True) == "Db"


def test_estimate_key_recovers_c_major():
    chroma = np.zeros(12)
    for pc in (0, 2, 4, 5, 7, 9, 11):  # C major scale mass
        chroma[pc] = 1.0
    tonic, mode, conf = estimate_key(chroma)
    assert tonic == "C"
    assert mode == "major"
    assert conf > 0.5


def test_coarse_chord_labels_minor():
    chroma = np.zeros(12)
    for pc in (0, 3, 7):
        chroma[pc] = 1.0
    assert coarse_chord(chroma) == "C:min"


def test_estimate_key_empty_is_safe():
    assert estimate_key(np.zeros(12)) == ("", "", 0.0)


# -- salience thresholding -------------------------------------------------- #
def test_threshold_chroma_keeps_strong_classes():
    v = np.zeros(12)
    v[0], v[4], v[7] = 1.0, 0.9, 0.8
    v[1] = 0.01  # noise floor
    pcs, weights = threshold_chroma(v, rel_threshold=0.15, min_energy=0.05, max_pcs=7)
    assert set(pcs) == {0, 4, 7}
    assert len(weights) == 3


def test_threshold_chroma_respects_max_pcs():
    v = np.ones(12)
    pcs, _ = threshold_chroma(v, rel_threshold=0.0, min_energy=0.0, max_pcs=4)
    assert len(pcs) == 4


def test_threshold_empty():
    pcs, weights = threshold_chroma(np.zeros(12), 0.1, 0.1, 7)
    assert pcs == [] and weights == []


# -- segmentation ----------------------------------------------------------- #
def test_spans_from_downbeats():
    beats = [
        Beat(t=0.0, is_downbeat=True, bar=1),
        Beat(t=0.5, is_downbeat=False, bar=1),
        Beat(t=2.0, is_downbeat=True, bar=2),
    ]
    spans = spans_from_downbeats(beats, duration_s=4.0)
    assert len(spans) == 2
    assert spans[0].start_s == 0.0 and spans[0].end_s == 2.0
    assert spans[1].end_s == 4.0


def test_spans_fixed():
    spans = spans_fixed(5.0, 2.0)
    assert [round(s.end_s, 1) for s in spans] == [2.0, 4.0, 5.0]


def test_build_spans_falls_back_to_fixed_without_downbeats():
    config = Config.default()
    spans = build_spans([], duration_s=4.0, config=config)
    assert len(spans) >= 1
    assert isinstance(spans[0], ChordSpan)


# -- audio containers ------------------------------------------------------- #
def test_audio_clip_shapes_and_window():
    clip = AudioClip(np.zeros(44100 * 3, dtype=np.float32), 44100, "x")
    assert clip.n_channels == 1
    assert abs(clip.duration_s - 3.0) < 1e-6
    windows = list(clip.window(1.0))
    assert len(windows) == 3


def test_audio_clip_to_mono():
    stereo = AudioClip(np.ones((2, 100), dtype=np.float32), 44100, "x")
    mono = stereo.to_mono()
    assert mono.n_channels == 1


def test_stems_named():
    clip = AudioClip(np.zeros((1, 10), dtype=np.float32), 44100, "x")
    stems = Stems(harmonic=clip, mix=clip)
    named = stems.named()
    assert "harmonic" in named and "mix" in named


# -- timbre stub ------------------------------------------------------------ #
def test_timbre_stub_returns_shape():
    clip = AudioClip(np.zeros((1, 100), dtype=np.float32), 44100, "x")
    desc = TimbreStub().describe(clip, ChordSpan(0.0, 2.0, 1))
    assert desc.inharmonicity is None
    assert "stub" in desc.notes
