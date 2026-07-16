"""Salience / multipitch → per-span pitch-class collections.

For voicing detail we go chroma / multipitch-salience → theory kernel, *not*
audio chord recognition (principle 3). Two extractors behind the
:class:`~fakebook.contracts.SalienceExtractor` protocol:

* :class:`ChromaSalience` — NNLS-style CQT chroma (librosa). Lighter start.
* :class:`BasicPitchSalience` — the basic-pitch salience/multipitch layer,
  folded to 12 pitch classes. Better at separating simultaneous partials.

Both aggregate energy over a chord span, then threshold to a pitch-class set +
per-pc weights (relative salience), which the kernel consumes.
"""

from __future__ import annotations

import numpy as np

from .._optional import is_available, require
from ..audio import AudioClip
from ..config import Config
from ..contracts import ChordSpan, PitchClassObservation


def threshold_chroma(
    chroma_mean: np.ndarray, rel_threshold: float, min_energy: float, max_pcs: int
) -> tuple[list[int], list[float]]:
    """Turn a 12-vector of pc energy into (pitch_classes, weights).

    ``rel_threshold`` is relative to the per-span max; ``min_energy`` is a share
    of total energy. Keeps at most ``max_pcs`` strongest classes.
    """
    v = np.asarray(chroma_mean, dtype=float)
    if v.sum() <= 0:
        return [], []
    peak = v.max()
    total = v.sum()
    keep = [
        pc
        for pc in range(12)
        if v[pc] >= rel_threshold * peak and (v[pc] / total) >= min_energy
    ]
    keep.sort(key=lambda pc: -v[pc])
    keep = keep[:max_pcs]
    keep.sort()
    weights = [float(v[pc]) for pc in keep]
    return keep, weights


class _BaseSalience:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.rel_threshold = float(self.config.get("harmony.salience_threshold", 0.15))
        self.min_energy = float(self.config.get("harmony.min_pc_energy", 0.08))
        self.max_pcs = int(self.config.get("harmony.max_pitch_classes", 7))
        self.sr = int(self.config.get("harmony.salience_sample_rate", 22050))

    def _observe_from_chromagram(
        self, chroma: np.ndarray, times: np.ndarray, spans: list[ChordSpan]
    ) -> list[PitchClassObservation]:
        out: list[PitchClassObservation] = []
        for span in spans:
            mask = (times >= span.start_s) & (times < span.end_s)
            if not mask.any():
                # nearest frame fallback so every span yields an observation
                idx = int(np.argmin(np.abs(times - (span.start_s + span.end_s) / 2)))
                frame = chroma[:, idx]
            else:
                frame = chroma[:, mask].mean(axis=1)
            pcs, weights = threshold_chroma(
                frame, self.rel_threshold, self.min_energy, self.max_pcs
            )
            out.append(PitchClassObservation(span=span, pitch_classes=pcs, weights=weights))
        return out


class ChromaSalience(_BaseSalience):
    """CQT chroma (librosa) → pitch-class observations."""

    def observe(
        self, harmonic: AudioClip, spans: list[ChordSpan]
    ) -> list[PitchClassObservation]:
        librosa = require("librosa", feature="chroma salience", extra="audio")
        mono = harmonic.to_mono()
        y = mono.samples[0]
        sr = mono.sample_rate
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
        return self._observe_from_chromagram(chroma, frame_times, spans)


class BasicPitchSalience(_BaseSalience):
    """basic-pitch multipitch salience folded to pitch classes."""

    def observe(
        self, harmonic: AudioClip, spans: list[ChordSpan]
    ) -> list[PitchClassObservation]:
        bp = require(
            "basic_pitch.inference", feature="basic-pitch salience", extra="harmony",
            package="basic-pitch",
        )
        librosa = require("librosa", feature="resampling", extra="audio")
        mono = harmonic.to_mono()
        y = mono.samples[0]
        if mono.sample_rate != 22050:
            y = librosa.resample(y, orig_sr=mono.sample_rate, target_sr=22050)
        # basic-pitch returns (model_output, midi, note_events); the contours
        # ("contour" salience) are in model_output. Fold the frequency axis to 12.
        model_output, _, _ = bp.predict(y)  # type: ignore[attr-defined]
        contour = np.asarray(model_output["contour"]).T  # (freq_bins, frames)
        chroma = self._fold_to_chroma(contour)
        # basic-pitch contour frames are ~ 172 Hz; build a time axis.
        frame_times = np.arange(chroma.shape[1]) * (len(y) / 22050) / max(chroma.shape[1], 1)
        return self._observe_from_chromagram(chroma, frame_times, spans)

    @staticmethod
    def _fold_to_chroma(contour: np.ndarray) -> np.ndarray:
        # basic-pitch contour has 3 bins/semitone starting at MIDI 21 (A0).
        n_bins = contour.shape[0]
        semitones = n_bins // 3
        chroma = np.zeros((12, contour.shape[1]))
        for s in range(semitones):
            midi = 21 + s
            pc = midi % 12
            chroma[pc] += contour[s * 3 : s * 3 + 3].sum(axis=0)
        return chroma


def build_salience(config: Config | None = None):
    config = config or Config.default()
    which = config.get("harmony.salience", "basic_pitch")
    if which == "basic_pitch" and is_available("basic_pitch"):
        return BasicPitchSalience(config)
    return ChromaSalience(config)
