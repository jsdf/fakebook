"""Branch B (sound-design) tests, over synthesized signals with known answers.

Every case is a patch whose ground truth we control — a plain saw is exactly
harmonic, a 7-voice unison is detuned by exactly N cents, an FM tone with an
irrational ratio is genuinely inharmonic — so the descriptors can be checked
against the number they are supposed to recover, and (just as important) against
each other: the point of a descriptor is that it *separates* patches.

Numpy only: Branch B needs no optional extras, so these run in the core suite.
"""

from __future__ import annotations

import numpy as np

from fakebook.audio import AudioClip
from fakebook.config import Config
from fakebook.contracts import ChordSpan
from fakebook.schema import TimbreDescriptor
from fakebook.timbre import SpectralTimbre, TimbreStub, build_timbre, describe_tags
from fakebook.timbre.spectral import (
    cents,
    detune_cents,
    estimate_f0,
    find_peaks,
    harmonic_energy_ratio,
    inharmonicity,
    refine_f0,
    spectral_centroid,
    spectral_flatness,
    stft_magnitude,
    trend_per_second,
)

SR = 22050


# --------------------------------------------------------------------------- #
# synthesis helpers                                                           #
# --------------------------------------------------------------------------- #
def saw(f: float, dur_s: float = 2.0, n_harmonics: int = 24, sr: int = SR) -> np.ndarray:
    """Additive sawtooth: an exactly harmonic series, amplitudes 1/k."""
    t = np.arange(int(dur_s * sr)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if k * f >= sr / 2:
            break
        y += np.sin(2 * np.pi * k * f * t) / k
    return y


def supersaw(f: float, detune: float = 25.0, voices: int = 7, dur_s: float = 2.0) -> np.ndarray:
    """``voices`` saws spread evenly over +-``detune`` cents (total span 2x)."""
    y = np.zeros(int(dur_s * SR))
    for c in np.linspace(-detune, detune, voices):
        y += saw(f * 2 ** (c / 1200), dur_s)
    return y / voices


def fm_tone(
    carrier: float = 440.0, ratio: float = np.sqrt(2), index: float = 6.0, dur_s: float = 2.0
) -> np.ndarray:
    """FM with an irrational c:m ratio — sidebands fall off any harmonic series."""
    t = np.arange(int(dur_s * SR)) / SR
    return np.sin(2 * np.pi * carrier * t + index * np.sin(2 * np.pi * carrier * ratio * t))


def filtered_saw(
    f: float = 110.0, cutoff_from: float = 300.0, cutoff_to: float = 4000.0, dur_s: float = 2.0
) -> np.ndarray:
    """Saw under a lowpass whose cutoff sweeps linearly across the span."""
    t = np.arange(int(dur_s * SR)) / SR
    cutoff = cutoff_from + (cutoff_to - cutoff_from) * t / dur_s
    y = np.zeros_like(t)
    for k in range(1, 41):
        fk = k * f
        if fk >= SR / 2:
            break
        y += np.sin(2 * np.pi * fk * t) / (k * (1.0 + (fk / cutoff) ** 4))
    return y


def timbre_config(**overrides) -> Config:
    """Default config with the ``timbre`` section patched."""
    data = Config.default().as_dict()
    data["timbre"].update(overrides)
    return Config(data)


def describe(y: np.ndarray, config: Config | None = None) -> TimbreDescriptor:
    clip = AudioClip(np.asarray(y, dtype=np.float32), SR)
    analyzer = SpectralTimbre(config)
    return analyzer.describe(clip, ChordSpan(0.0, clip.duration_s, 1))


# --------------------------------------------------------------------------- #
# spectral primitives                                                         #
# --------------------------------------------------------------------------- #
def test_centroid_tracks_a_known_sine():
    freqs, mag, _ = stft_magnitude(np.sin(2 * np.pi * 1000 * np.arange(SR) / SR), SR)
    assert abs(spectral_centroid(freqs, mag).mean() - 1000.0) < 20.0


def test_flatness_separates_tone_from_noise():
    tone = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    noise = np.random.default_rng(0).normal(0, 0.2, SR)
    _, tone_mag, _ = stft_magnitude(tone, SR)
    _, noise_mag, _ = stft_magnitude(noise, SR)
    assert spectral_flatness(tone_mag).mean() < 0.01
    assert spectral_flatness(noise_mag).mean() > 0.2


def test_peaks_are_interpolated_off_bin_centres():
    # 443 Hz sits between 8192-point bins (2.69 Hz apart); the parabolic fit
    # should land far closer than half a bin.
    freqs, mag, _ = stft_magnitude(np.sin(2 * np.pi * 443 * np.arange(SR) / SR), SR)
    peak_f, _ = find_peaks(freqs, mag.mean(axis=1))
    assert len(peak_f) == 1
    assert abs(peak_f[0] - 443.0) < 0.5


def test_trend_screens_out_non_linear_jitter():
    times = np.linspace(0, 2, 20)
    rising = 500 + 300 * times
    assert abs(trend_per_second(rising, times) - 300.0) < 1.0
    jitter = 500 + 400 * np.sin(np.arange(20) * 2.7)  # big swings, no trend
    assert trend_per_second(jitter, times) == 0.0


def test_cents_is_signed_and_symmetric():
    assert abs(cents(880.0, 440.0) - 1200.0) < 1e-9
    assert abs(cents(440.0, 880.0) + 1200.0) < 1e-9


# --------------------------------------------------------------------------- #
# f0                                                                          #
# --------------------------------------------------------------------------- #
def test_f0_survives_the_saw_octave_trap():
    # Raw HPS picks 880 Hz here (strong even harmonics + fractional bins); the
    # envelope + subharmonic check must return the true 440.
    freqs, mag, _ = stft_magnitude(saw(440.0), SR)
    assert abs(estimate_f0(freqs, mag.mean(axis=1)) - 440.0) < 5.0


def test_f0_falls_back_to_the_lone_partial_of_a_sine():
    # A single sine has no series for HPS to lock onto; refinement must still
    # return the partial that is actually there, not an empty low bin.
    freqs, mag, _ = stft_magnitude(np.sin(2 * np.pi * 200 * np.arange(2 * SR) / SR), SR)
    avg = mag.mean(axis=1)
    peak_f, peak_m = find_peaks(freqs, avg)
    assert abs(refine_f0(estimate_f0(freqs, avg), peak_f, peak_m) - 200.0) < 1.0


def test_f0_refinement_ignores_sub_f0_min_rumble():
    peaks = np.array([7.0, 220.0, 440.0])
    mags = np.array([1.0, 1.0, 0.5])
    assert refine_f0(0.0, peaks, mags, f_min=40.0) == 220.0


# --------------------------------------------------------------------------- #
# inharmonicity                                                               #
# --------------------------------------------------------------------------- #
def test_ideal_series_reads_as_harmonic():
    f0 = 100.0
    peaks = np.arange(1, 11) * f0
    assert inharmonicity(peaks, np.ones(10), f0) == 0.0


def test_stretched_series_reads_as_inharmonic():
    f0 = 100.0
    peaks = np.arange(1, 11) * f0 * 1.03  # 3% stretch
    assert inharmonicity(peaks, np.ones(10), f0) > 0.02


def test_fm_is_inharmonic_and_a_saw_is_not():
    saw_desc = describe(saw(440.0))
    fm_desc = describe(fm_tone())
    assert saw_desc.inharmonicity < 0.001
    assert fm_desc.inharmonicity > 0.02
    assert "off-series partials" in fm_desc.tags
    assert "off-series partials" not in saw_desc.tags


def test_harmonic_ratio_separates_one_source_from_a_stack():
    single = describe(saw(440.0))
    chord = describe(
        sum(supersaw(440.0 * 2 ** ((m - 69) / 12), detune=20.0) for m in (48, 51, 55, 58, 65)) / 5
    )
    assert single.harmonic_ratio > 0.95  # one series explains the whole spectrum
    assert chord.harmonic_ratio < 0.6  # four other notes' partials are off it


def test_polyphony_reads_as_off_series_partials_not_as_an_fm_claim():
    """Pinned limitation: inharmonicity is a *single source* measurement.

    A chord of perfectly harmonic oscillators still deviates from the dominant
    note's series, so the number goes up. The descriptor must not turn that into
    a timbral claim — the tag stays neutral, and ``harmonic_ratio`` is what says
    a single series did not explain the span. Disambiguating polyphony from
    genuine FM inharmonicity is the interpret stage's job, using Branch A's
    pitch-class count for the same segment.
    """
    chord = describe(
        sum(saw(440.0 * 2 ** ((m - 69) / 12)) for m in (48, 51, 55, 58, 65)) / 5
    )
    assert chord.inharmonicity > 0.02
    assert chord.harmonic_ratio < 0.6
    assert "off-series partials" in chord.tags
    assert not any("FM" in tag or "metallic" in tag for tag in chord.tags)


def test_harmonic_energy_ratio_counts_only_on_series_peaks():
    f0 = 100.0
    on_series = np.array([100.0, 200.0, 300.0])
    off_series = np.array([137.0, 271.0])
    freqs = np.concatenate([on_series, off_series])
    mags = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(harmonic_energy_ratio(freqs, mags, f0) - 0.6) < 1e-9
    assert harmonic_energy_ratio(on_series, np.ones(3), f0) == 1.0


# --------------------------------------------------------------------------- #
# detuning                                                                    #
# --------------------------------------------------------------------------- #
def test_detune_recovers_the_unison_spread():
    # +-25 cents over 7 voices = a 50-cent span; the estimate is a lower bound
    # (voices are unresolved at the low partials), so allow it to under-read.
    desc = describe(supersaw(440.0, detune=25.0))
    assert 35.0 <= desc.detune_cents <= 55.0
    assert "detuned unison (supersaw-like)" in desc.tags


def test_detune_scales_with_the_spread_and_is_zero_for_one_oscillator():
    wide = describe(supersaw(440.0, detune=25.0)).detune_cents
    narrow = describe(supersaw(440.0, detune=10.0)).detune_cents
    single = describe(saw(440.0)).detune_cents
    assert wide > narrow > single == 0.0


def test_detune_needs_most_partials_split_not_one_coincidence():
    # Two partials that happen to fall close together are not a unison.
    freqs = np.array([100.0, 200.0, 300.0, 400.0, 402.0, 500.0, 600.0])
    mags = np.ones(len(freqs))
    assert detune_cents(freqs, mags) == 0.0
    # Every partial split into a pair: a unison.
    split = np.array([100.0, 101.0, 200.0, 202.0, 300.0, 303.0])
    assert detune_cents(split, np.ones(len(split))) > 0.0


def test_inharmonic_fm_is_not_labelled_a_detuned_unison():
    desc = describe(fm_tone(440.0, ratio=1.618, index=5.0))
    assert desc.detune_cents == 0.0
    assert "detuned unison (supersaw-like)" not in desc.tags


# --------------------------------------------------------------------------- #
# filter movement / brightness / noise                                        #
# --------------------------------------------------------------------------- #
def test_filter_sweep_direction_is_recovered():
    up = describe(filtered_saw())
    down = describe(filtered_saw()[::-1].copy())
    assert up.centroid_slope_hz_per_s > 200.0
    assert down.centroid_slope_hz_per_s < -200.0
    assert "filter opening" in up.tags
    assert "filter closing" in down.tags


def test_static_patch_has_no_filter_movement():
    assert describe(saw(220.0)).centroid_slope_hz_per_s == 0.0


def test_bright_and_dark_are_opposite_ends_of_the_centroid():
    dark = describe(np.sin(2 * np.pi * 200 * np.arange(2 * SR) / SR))
    bright = describe(supersaw(440.0, detune=6.0))
    assert "dark" in dark.tags
    assert "bright" in bright.tags
    assert bright.spectral_centroid_hz > dark.spectral_centroid_hz


def test_noise_is_noisy_and_not_a_unison():
    desc = describe(np.random.default_rng(0).normal(0, 0.2, 2 * SR))
    assert "noisy" in desc.tags
    # Noise peaks cluster too; the detune tag must not follow them.
    assert "detuned unison (supersaw-like)" not in desc.tags


def test_partial_count_separates_a_sine_from_a_stack():
    assert describe(np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)).n_partials == 1
    assert describe(supersaw(440.0)).n_partials > 20


def test_swept_supersaw_matches_the_documented_example():
    """The worked example in the README, pinned so the numbers cannot drift."""
    t = np.arange(2 * SR) / SR
    cutoff = 800.0 + (6000.0 - 800.0) * t / 2.0
    y = np.zeros_like(t)
    for c in np.linspace(-25.0, 25.0, 7):
        f = 440.0 * 2 ** (c / 1200)
        for k in range(1, 25):
            if k * f >= SR / 2:
                break
            y += np.sin(2 * np.pi * k * f * t) / (k * (1.0 + (k * f / cutoff) ** 4))
    desc = describe(y / 7)
    assert desc.tags == [
        "detuned unison (supersaw-like)",
        "filter opening",
        "dense/layered",
    ]
    assert abs(desc.detune_cents - 48.4) < 2.0
    assert abs(desc.centroid_slope_hz_per_s - 830.0) < 50.0
    assert abs(desc.harmonic_ratio - 0.64) < 0.05


# --------------------------------------------------------------------------- #
# stage plumbing                                                              #
# --------------------------------------------------------------------------- #
def test_silent_span_is_reported_not_guessed():
    desc = describe(np.zeros(SR))
    assert desc.spectral_centroid_hz is None
    assert desc.tags == []
    assert "silent" in desc.notes


def test_describe_uses_only_the_requested_span():
    # dark first half, bright second half — each span must see only its own half
    y = np.concatenate([saw(110.0, dur_s=2.0, n_harmonics=3), supersaw(440.0, dur_s=2.0)])
    clip = AudioClip(np.asarray(y, dtype=np.float32), SR)
    analyzer = SpectralTimbre()
    first = analyzer.describe(clip, ChordSpan(0.0, 2.0, 1))
    second = analyzer.describe(clip, ChordSpan(2.0, 4.0, 2))
    assert second.spectral_centroid_hz > 3 * first.spectral_centroid_hz


def test_tags_are_derived_from_the_descriptor_alone():
    desc = TimbreDescriptor(
        spectral_centroid_hz=4200.0,
        detune_cents=30.0,
        inharmonicity=0.0,
        spectral_flatness=0.01,
        centroid_slope_hz_per_s=-900.0,
        n_partials=40,
    )
    assert describe_tags(desc) == [
        "bright",
        "detuned unison (supersaw-like)",
        "filter closing",
        "dense/layered",
    ]


def test_build_timbre_honours_config():
    assert isinstance(build_timbre(Config.default()), SpectralTimbre)
    assert isinstance(build_timbre(timbre_config(backend="stub")), TimbreStub)
    assert build_timbre(timbre_config(enabled=False)) is None
