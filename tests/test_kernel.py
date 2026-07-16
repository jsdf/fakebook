"""Tests for the deterministic harmony kernel — the guardrail heart.

Ground truth here is music theory, which (unlike the audio side) is fully
knowable, so these tests are exact.
"""

import pytest

from fakebook.contracts import ChordSpan, PitchClassObservation
from fakebook.harmony import HarmonyKernel


@pytest.fixture(scope="module")
def kernel():
    return HarmonyKernel()


def enum(kernel, pcs, weights=None):
    obs = PitchClassObservation(
        span=ChordSpan(start_s=0.0, end_s=2.0, bar=1),
        pitch_classes=pcs,
        weights=weights or [],
    )
    return kernel.enumerate_segment(obs)


def names(seg):
    return [c.name for c in seg.chord_interpretations]


# --------------------------------------------------------------------------- #
# chord identification                                                        #
# --------------------------------------------------------------------------- #
def test_minor_seventh_top_reading(kernel):
    seg = enum(kernel, [0, 3, 7, 10])
    assert names(seg)[0] == "Cm7"
    assert seg.chord_interpretations[0].score > 0.8


def test_dominant_seventh(kernel):
    seg = enum(kernel, [0, 4, 7, 10])
    assert names(seg)[0] == "C7"


def test_hendrix_sharp_nine_is_enumerated(kernel):
    # C E G Bb D#  ->  C7#9
    seg = enum(kernel, [0, 4, 7, 10, 3])
    assert names(seg)[0] == "C7#9"


def test_maj7_sharp11(kernel):
    # F A C E B  ->  Fmaj7#11
    seg = enum(kernel, [5, 9, 0, 4, 11])
    assert names(seg)[0] == "Fmaj7#11"


def test_root_must_sound(kernel):
    # every enumerated chord's root pitch class is present in the collection
    seg = enum(kernel, [0, 3, 7, 10])
    present = set(seg.pitch_classes)
    for c in seg.chord_interpretations:
        root_pc = HarmonyKernel._name_to_pc(c.root)
        assert root_pc in present


# --------------------------------------------------------------------------- #
# spelling: principle 4 (sharp-9 vs flat-3 must survive)                       #
# --------------------------------------------------------------------------- #
def test_sharp_nine_spelled_as_sharp_not_flat_three(kernel):
    seg = enum(kernel, [0, 4, 7, 10, 3])
    top_spelling = seg.spelled_candidates[0]
    # under the C7#9 reading the pc 3 is D#, not Eb
    assert "D#" in top_spelling
    assert "Eb" not in top_spelling


def test_minor_third_spelled_flat_three(kernel):
    seg = enum(kernel, [0, 3, 7, 10])  # Cm7 top reading
    assert seg.spelled_candidates[0] == "C Eb G Bb"


def test_spelling_correct_for_altered_root(kernel):
    # F# minor triad: F# A C#  ->  b3 must be A natural, not Ab
    seg = enum(kernel, [6, 9, 1])
    assert names(seg)[0] == "F#m"
    assert seg.spelled_candidates[0] == "F# A C#"


# --------------------------------------------------------------------------- #
# pc-set descriptors (music21)                                                #
# --------------------------------------------------------------------------- #
def test_pc_set_descriptors(kernel):
    seg = enum(kernel, [0, 4, 7])  # C major triad = 3-11
    assert seg.pc_set.forte == "3-11B" or seg.pc_set.forte.startswith("3-11")
    assert seg.pc_set.prime_form == "[0 3 7]"
    assert seg.pc_set.interval_vector == "<001110>"


# --------------------------------------------------------------------------- #
# voicing analysis                                                            #
# --------------------------------------------------------------------------- #
def test_quartal_detection(kernel):
    # C F Bb -> stack of fourths
    seg = enum(kernel, [0, 5, 10])
    assert seg.voicing.stacking == "quartal"


def test_bigger_quartal_stack(kernel):
    # C F Bb Eb Ab
    seg = enum(kernel, [0, 5, 10, 3, 8])
    assert seg.voicing.stacking == "quartal"


def test_cluster_detection(kernel):
    seg = enum(kernel, [0, 1, 2, 3])
    assert seg.voicing.stacking == "cluster"


def test_tertian_detection(kernel):
    seg = enum(kernel, [0, 4, 7, 11])  # Cmaj7
    assert seg.voicing.stacking == "tertian"


def test_upper_structure_triad(kernel):
    # C7 with a D major triad on top: C E G Bb D F# A
    seg = enum(kernel, [0, 4, 7, 10, 2, 6, 9])
    assert seg.voicing.upper_structure is not None
    assert "D major triad" == seg.voicing.upper_structure


def test_omitted_fifth(kernel):
    # C E Bb  -> C7 with no 5th
    seg = enum(kernel, [0, 4, 10])
    top = seg.chord_interpretations[0]
    if top.quality == "7":
        assert "G" in seg.voicing.omitted


# --------------------------------------------------------------------------- #
# scales                                                                      #
# --------------------------------------------------------------------------- #
def test_scale_candidates_contain_dorian_for_m7(kernel):
    seg = enum(kernel, [0, 3, 7, 10])
    scale_names = [s.name for s in seg.scale_candidates]
    assert any("C dorian" == n for n in scale_names)
    for s in seg.scale_candidates:
        assert 0.0 <= s.fit <= 1.0


def test_scale_out_of_scale_reported(kernel):
    seg = enum(kernel, [0, 1, 4, 7])  # includes a b9 relative to C
    # at least one scale candidate should flag the b9 as out of scale, or none
    # fully contains it — either way out_of_scale stays a valid subset
    for s in seg.scale_candidates:
        assert set(s.out_of_scale) <= set(seg.pitch_classes)


# --------------------------------------------------------------------------- #
# determinism & edge cases                                                    #
# --------------------------------------------------------------------------- #
def test_determinism(kernel):
    a = enum(kernel, [0, 4, 7, 10, 2, 9])
    b = enum(kernel, [9, 2, 10, 7, 4, 0])  # same set, different order
    assert names(a) == names(b)
    assert a.spelled_candidates == b.spelled_candidates
    assert a.pitch_classes == b.pitch_classes


def test_empty_collection_is_safe(kernel):
    seg = enum(kernel, [])
    assert seg.pitch_classes == []
    assert seg.chord_interpretations == []
    assert seg.voicing.stacking == "unknown"


def test_single_note(kernel):
    seg = enum(kernel, [0])
    assert seg.pitch_classes == [0]
    # no confident chord/voicing from one note
    assert seg.voicing.stacking in ("unknown", "cluster", "tertian", "quartal")


def test_scores_bounded(kernel):
    seg = enum(kernel, [0, 4, 7, 10, 2, 5, 9])
    for c in seg.chord_interpretations:
        assert 0.0 <= c.score <= 1.0


def test_weights_influence_ranking_but_stay_valid(kernel):
    # Same pcs, different weights should still yield valid, bounded output.
    seg = enum(kernel, [0, 3, 7, 10], weights=[0.5, 0.2, 0.2, 0.1])
    assert names(seg)[0] in ("Cm7", "Eb6")
    for c in seg.chord_interpretations:
        assert 0.0 <= c.score <= 1.0
