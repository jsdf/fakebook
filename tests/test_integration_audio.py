"""End-to-end integration tests over *real synthesized audio*.

Unlike ``test_pipeline.py`` (which injects fake stages), these render known
chords to a signal and drive the real direct-from-audio path — librosa CQT
chroma salience → the deterministic kernel — plus the librosa MIR key head and a
full ``analyze_file`` run from a written WAV. This is the coverage the spec's
validation notes call for: does the audio front-end actually recover the pitch
content, and does it stay coherent through assembly?

Requires the ``audio`` extra; skipped cleanly otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")

from fakebook.audio import AudioClip  # noqa: E402
from fakebook.config import Config  # noqa: E402
from fakebook.contracts import ChordSpan  # noqa: E402
from fakebook.harmony import HarmonyKernel  # noqa: E402
from fakebook.harmony.salience import ChromaSalience  # noqa: E402
from fakebook.mir import LibrosaMirAnalyzer  # noqa: E402
from fakebook.separation import PassthroughSeparator  # noqa: E402

SR = 22050

pytestmark = pytest.mark.audio


# --------------------------------------------------------------------------- #
# synthesis helpers                                                           #
# --------------------------------------------------------------------------- #
def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def render_chord(
    midis, dur_s: float, sr: int = SR, noise: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Sum of sawtooth-ish tones (fundamental + harmonics), optionally + noise."""
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    sig = np.zeros_like(t)
    for m in midis:
        f = _midi_to_hz(m)
        for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
            sig += amp * np.sin(2 * np.pi * f * h * t)
    sig /= np.max(np.abs(sig)) + 1e-9
    if noise:
        sig = sig + noise * np.random.default_rng(seed).standard_normal(len(sig))
    return sig.astype(np.float32)


def render_progression(progression, dur_each: float, sr: int = SR) -> np.ndarray:
    return np.concatenate([render_chord(m, dur_each, sr) for m in progression])


# Voiced around C4/F4 so the CQT has real low-mid content.
CHORDS = {
    "Cm7": [60, 63, 67, 70],           # C Eb G Bb
    "Fm7": [65, 68, 72, 75],           # F Ab C Eb
    "Fmaj7#11": [65, 69, 72, 76, 71],  # F A C E B
    "C7#9": [60, 64, 67, 70, 63],      # C E G Bb Eb(=D#)
}


def _observe_single_chord(midis, dur=2.0, noise=0.0):
    sig = render_chord(midis, dur, noise=noise)
    clip = AudioClip(sig, SR, "synth")
    sal = ChromaSalience(Config.default())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = sal.observe(clip, [ChordSpan(0.0, dur, 1)])
    return obs[0]


# --------------------------------------------------------------------------- #
# salience recovers the pitch content                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CHORDS))
def test_chroma_salience_recovers_pitch_classes(name):
    expected = sorted({m % 12 for m in CHORDS[name]})
    obs = _observe_single_chord(CHORDS[name])
    assert obs.pitch_classes == expected, (name, obs.pitch_classes, expected)


@pytest.mark.parametrize("name", list(CHORDS))
def test_kernel_names_chord_from_real_audio(name):
    obs = _observe_single_chord(CHORDS[name])
    seg = HarmonyKernel().enumerate_segment(obs)
    top_names = [c.name for c in seg.chord_interpretations]
    assert seg.chord_interpretations[0].name == name, (name, top_names)


def test_sharp_nine_spelled_from_real_audio():
    obs = _observe_single_chord(CHORDS["C7#9"])
    seg = HarmonyKernel().enumerate_segment(obs)
    assert "D#" in seg.spelled_candidates[0]
    assert "Eb" not in seg.spelled_candidates[0]


# --------------------------------------------------------------------------- #
# robustness & documented limitations                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("noise", [0.1, 0.25, 0.5, 1.0])
def test_salience_robust_to_broadband_noise(noise):
    # Averaging CQT chroma over a 2s span rejects broadband noise; even at
    # ~0 dB SNR the pitch content survives.
    obs = _observe_single_chord(CHORDS["Cm7"], noise=noise)
    assert obs.pitch_classes == [0, 3, 7, 10], (noise, obs.pitch_classes)


def test_wide_voicing_with_low_root_recovered():
    # C2 E4 G4 B4 — a wide, open Cmaj7 voicing.
    obs = _observe_single_chord([48, 64, 67, 71])
    seg = HarmonyKernel().enumerate_segment(obs)
    assert seg.chord_interpretations[0].name == "Cmaj7"


def test_single_bright_tone_leaks_fifth_KNOWN_LIMITATION():
    # A single note's 3rd harmonic is a perfect fifth, so for a bright,
    # harmonic-rich tone chroma cannot tell a lone note from a bare fifth: a
    # sawtooth C reads as {C, G}. This is an inherent property of chroma, not a
    # kernel bug — pinned here so the behaviour is documented and a change to it
    # is noticed. (A gentler tone keeps the leak below the energy floor; see
    # test_kernel single-note handling.)
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    f = _midi_to_hz(60)
    sig = sum((1.0 / h) * np.sin(2 * np.pi * f * h * t) for h in range(1, 12))
    sig = (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = ChromaSalience(Config.default()).observe(
            AudioClip(sig, SR, "bright"), [ChordSpan(0.0, 2.0, 1)]
        )[0]
    assert obs.pitch_classes == [0, 7]  # fundamental + leaked fifth
    assert 4 not in obs.pitch_classes  # major-third (5th harmonic) stays below floor


# --------------------------------------------------------------------------- #
# per-span recovery across a progression                                       #
# --------------------------------------------------------------------------- #
def test_progression_recovered_per_span():
    prog = [CHORDS["Cm7"], CHORDS["Fm7"]]
    sig = render_progression(prog, 2.0)
    clip = AudioClip(sig, SR, "prog")
    spans = [ChordSpan(0.0, 2.0, 1), ChordSpan(2.0, 4.0, 2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observations = ChromaSalience(Config.default()).observe(clip, spans)
    kernel = HarmonyKernel()
    names = [kernel.enumerate_segment(o).chord_interpretations[0].name for o in observations]
    assert names == ["Cm7", "Fm7"], names


# --------------------------------------------------------------------------- #
# MIR key head                                                                  #
# --------------------------------------------------------------------------- #
def test_mir_key_head_recovers_c_minor_ish():
    # A C-minor triad bed → key estimate should be C (major/minor ambiguity is
    # fine for a bare triad; we assert the tonic).
    sig = render_progression([CHORDS["Cm7"], CHORDS["Fm7"], CHORDS["Cm7"]], 2.0)
    clip = AudioClip(sig, SR, "keytest")
    stems = PassthroughSeparator(Config.default()).separate(clip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LibrosaMirAnalyzer(Config.default()).analyze(clip, stems)
    assert result.global_.key.tonic in ("C", "Eb")  # relative major of Cm is Eb
    assert result.global_.tempo_bpm > 0


# --------------------------------------------------------------------------- #
# full pipeline from a written WAV                                              #
# --------------------------------------------------------------------------- #
def test_full_pipeline_from_wav(tmp_path, monkeypatch):
    import soundfile as sf

    # Fixed-window segmentation so spans land exactly on the 2s chord changes,
    # independent of beat-tracking synthetic tones.
    monkeypatch.setenv("FAKEBOOK_HARMONY__SEGMENT_BY", "fixed")
    monkeypatch.setenv("FAKEBOOK_HARMONY__FIXED_SEGMENT_S", "2.0")
    monkeypatch.setenv("FAKEBOOK_SEPARATION__BACKEND", "passthrough")

    sig = render_progression([CHORDS["Cm7"], CHORDS["Fm7"]], 2.0)
    wav = tmp_path / "prog.wav"
    sf.write(wav, sig, SR)

    from fakebook.pipeline import Pipeline

    config = Config.load()
    pipe = Pipeline.from_config(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = pipe.analyze_file(str(wav))

    assert len(doc.segments) == 2
    recovered = [s.chord_interpretations[0].name for s in doc.segments if s.chord_interpretations]
    assert "Cm7" in recovered and "Fm7" in recovered, recovered
    # document already passed JSON-Schema validation inside assemble; sanity-check meta
    assert abs(doc.meta.duration_s - 4.0) < 0.05
