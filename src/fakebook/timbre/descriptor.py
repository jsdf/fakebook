"""Branch B: harmonic-stem span → spectral/sound-design descriptor.

Branch B treats a "complex voicing" as a *sound-design object* rather than a
chord: detuning, filter movement, FM inharmonicity, layering. Those are the four
things the spec names, and they map onto the descriptor fields as:

===================  ==============================================
detuning             ``detune_cents``       (split-partial cluster width)
filter movement      ``centroid_slope_hz_per_s`` (centroid trend)
FM inharmonicity     ``inharmonicity``      (deviation from an ideal series)
layering             ``n_partials``         (resolved partial count)
===================  ==============================================

with ``spectral_centroid_hz`` / ``spectral_bandwidth_hz`` / ``spectral_flatness``
as the general brightness-spread-noisiness frame, and ``tags`` as deterministic
labels thresholded off those numbers.

One honest limit is worth stating up front: ``inharmonicity`` is a *single
source* measurement — deviation of the partials from one ideal harmonic series —
so a polyphonic span reads high simply because the other chord tones' partials
do not belong to the dominant note's series. It is reported with
``harmonic_ratio`` (how much of the partial energy that one series explains), and
the tag it drives is the neutral ``off-series partials`` rather than a claim of
FM/metallic timbre. Which of the two it is falls out at interpretation time, from
the *other* branch: off-series partials over a five-pitch-class segment is
polyphony; the same tag over one or two pitch classes is genuine inharmonicity.

Two implementations behind the :class:`~fakebook.contracts.TimbreAnalyzer`
protocol:

* :class:`SpectralTimbre` — the real extractor (:mod:`fakebook.timbre.spectral`,
  numpy only, so it runs on a core install).
* :class:`TimbreStub` — the original no-op, kept as the explicit "not measured"
  path so a caller can disable Branch B without losing the descriptor shape.

The same guardrail as Branch A applies at the other end: these are *measured
facts* handed to the LLM, which may describe them but must not turn them into
harmony. ``tags`` are computed here, by fixed thresholds, not by the model.
"""

from __future__ import annotations

import numpy as np

from ..audio import AudioClip
from ..config import Config
from ..contracts import ChordSpan
from ..schema import TimbreDescriptor
from .spectral import (
    detune_cents,
    estimate_f0,
    find_peaks,
    frame_energy,
    harmonic_energy_ratio,
    inharmonicity,
    refine_f0,
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    stft_magnitude,
    trend_per_second,
    weighted_mean,
)

# Tag thresholds. Fixed constants rather than config, so a descriptor's tags mean
# the same thing across runs and can be compared between tracks.
_BRIGHT_HZ = 3000.0
_DARK_HZ = 800.0
_DETUNED_CENTS = 8.0  # below this, a "unison" is a single oscillator
_INHARMONIC = 0.02  # a single saw/square reads ~1e-5; FM bells three orders up
_NOISY_FLATNESS = 0.2
_SWEEP_HZ_PER_S = 200.0
_DENSE_PARTIALS = 24


class SpectralTimbre:
    """Measure Branch B descriptors over a chord span of the harmonic stem."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.n_fft = int(self.config.get("timbre.n_fft", 8192))
        self.hop_length = int(self.config.get("timbre.hop_length", 2048))
        self.peak_threshold = float(self.config.get("timbre.peak_threshold", 0.05))
        self.cluster_cents = float(self.config.get("timbre.cluster_cents", 70.0))
        self.min_split_ratio = float(self.config.get("timbre.min_split_ratio", 0.5))
        self.f0_min = float(self.config.get("timbre.f0_min", 40.0))
        self.f0_max = float(self.config.get("timbre.f0_max", 1200.0))
        self.peak_fmax = float(self.config.get("timbre.peak_fmax", 8000.0))
        self.max_peaks = int(self.config.get("timbre.max_peaks", 64))

    def describe(self, harmonic: AudioClip, span: ChordSpan) -> TimbreDescriptor:
        clip = harmonic.slice_s(span.start_s, span.end_s).to_mono()
        y = clip.samples[0]
        if y.size == 0 or not np.any(np.abs(y) > 0):
            return TimbreDescriptor(notes="silent span — no timbre measured")

        freqs, mag, times = stft_magnitude(y, clip.sample_rate, self.n_fft, self.hop_length)
        energy = frame_energy(mag)

        centroid = spectral_centroid(freqs, mag)
        bandwidth = spectral_bandwidth(freqs, mag, centroid)
        flatness = spectral_flatness(mag)

        # Partials come from the time-averaged spectrum: averaging first keeps a
        # steady partial and suppresses per-frame noise, which is what the
        # detune/inharmonicity estimates need.
        avg = mag.mean(axis=1)
        peak_f, peak_m = find_peaks(
            freqs, avg, self.peak_threshold, self.max_peaks, self.peak_fmax
        )
        f0 = refine_f0(
            estimate_f0(freqs, avg, self.f0_min, self.f0_max),
            peak_f,
            peak_m,
            f_min=self.f0_min,
        )

        desc = TimbreDescriptor(
            spectral_centroid_hz=round(weighted_mean(centroid, energy), 2),
            spectral_bandwidth_hz=round(weighted_mean(bandwidth, energy), 2),
            spectral_flatness=round(weighted_mean(flatness, energy), 5),
            centroid_slope_hz_per_s=round(trend_per_second(centroid, times), 2),
            inharmonicity=round(inharmonicity(peak_f, peak_m, f0), 5),
            harmonic_ratio=round(harmonic_energy_ratio(peak_f, peak_m, f0), 4),
            detune_cents=round(
                detune_cents(peak_f, peak_m, self.cluster_cents, self.min_split_ratio), 2
            ),
            n_partials=int(len(peak_f)),
            f0_hz=round(f0, 2) if f0 > 0 else None,
        )
        desc.tags = describe_tags(desc)
        desc.notes = (
            f"spectral descriptors over {span.end_s - span.start_s:.2f}s "
            f"(n_fft={self.n_fft}); detune is a lower bound at this resolution"
        )
        return desc


def describe_tags(desc: TimbreDescriptor) -> list[str]:
    """Deterministic sound-design labels from the measured numbers.

    Thresholded here rather than in the interpret stage on purpose: the LLM is
    allowed to *explain* a tag, never to assign one (principle 2, applied to
    Branch B).
    """
    tags: list[str] = []
    centroid = desc.spectral_centroid_hz or 0.0
    if centroid >= _BRIGHT_HZ:
        tags.append("bright")
    elif 0 < centroid <= _DARK_HZ:
        tags.append("dark")
    noisy = (desc.spectral_flatness or 0.0) >= _NOISY_FLATNESS
    # Only tag detuning on tonal material: dense noise peaks cluster too, and a
    # cluster width measured through noise is not a unison spread.
    if (desc.detune_cents or 0.0) >= _DETUNED_CENTS and not noisy:
        tags.append("detuned unison (supersaw-like)")
    if (desc.inharmonicity or 0.0) >= _INHARMONIC:
        tags.append("off-series partials")
    if noisy:
        tags.append("noisy")
    slope = desc.centroid_slope_hz_per_s or 0.0
    if slope >= _SWEEP_HZ_PER_S:
        tags.append("filter opening")
    elif slope <= -_SWEEP_HZ_PER_S:
        tags.append("filter closing")
    if (desc.n_partials or 0) >= _DENSE_PARTIALS:
        tags.append("dense/layered")
    return tags


class TimbreStub:
    """No-op timbre analyzer. Returns the descriptor shape, not real features."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()

    def describe(self, harmonic: AudioClip, span: ChordSpan) -> TimbreDescriptor:
        return TimbreDescriptor(notes="timbre stub — Branch B not measured")


def build_timbre(config: Config | None = None):
    """Choose a timbre analyzer from config (``timbre.backend``).

    Returns ``None`` when Branch B is disabled, so the pipeline can skip the
    stage entirely rather than fill every segment with an empty descriptor.
    """
    config = config or Config.default()
    if not config.get("timbre.enabled", True):
        return None
    if config.get("timbre.backend", "spectral") == "stub":
        return TimbreStub(config)
    return SpectralTimbre(config)
