"""MIR analyzers that populate ``global`` / ``beats`` / ``sections``.

``LibrosaMirAnalyzer`` is the practical workhorse (spec's per-task fallback tier):

* **tempo + beats** — ``librosa.beat.beat_track`` on the full mix.
* **downbeats** — beats grouped into bars (4/4 assumed unless config overrides);
  a real deployment swaps in madmom's downbeat RNN behind the same interface.
* **key** — Krumhansl-Schmuckler over mean chroma of the harmonic stem.
* **sections** — recurrence-matrix segmentation (``librosa.segment``), labelled
  generically (structure model / msaf can replace this behind the interface).
* **coarse chords** — per-beat maj/min template match. Lossy context only
  (principle 3): it never feeds the voicing kernel.

``NullMirAnalyzer`` returns a valid, empty-ish result so the pipeline runs and is
testable with no audio deps.
"""

from __future__ import annotations

import numpy as np

from .._optional import is_available, require
from ..audio import AudioClip, Stems
from ..config import Config
from ..contracts import MirResult
from ..pitch import coarse_chord, estimate_key
from ..schema import Beat, Global, Key, Section


class NullMirAnalyzer:
    """Fallback: minimal valid MIR result, no audio analysis performed."""

    def analyze(self, clip: AudioClip, stems: Stems) -> MirResult:
        g = Global(key=Key(), tempo_bpm=0.0, time_signature="")
        return MirResult(global_=g, beats=[], sections=[], coarse_chords=[])


class LibrosaMirAnalyzer:
    """librosa-based MIR heads."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        beats_per_bar = self.config.get("mir.beats_per_bar", 4)
        self.beats_per_bar = int(beats_per_bar)
        self.compute_acr = bool(self.config.get("mir.compute_coarse_acr", True))

    def analyze(self, clip: AudioClip, stems: Stems) -> MirResult:
        librosa = require("librosa", feature="MIR analysis", extra="audio")
        mix = (stems.mix or clip).to_mono()
        harmonic = stems.harmonic.to_mono()
        sr = mix.sample_rate
        y = mix.samples[0]

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beats = self._beats(beat_times)

        key = self._key(librosa, harmonic)
        sections = self._sections(librosa, y, sr)
        coarse = (
            self._coarse_chords(librosa, harmonic, beat_times)
            if self.compute_acr
            else []
        )

        g = Global(
            key=key,
            tempo_bpm=float(np.atleast_1d(tempo)[0]),
            time_signature=f"{self.beats_per_bar}/4",
        )
        return MirResult(global_=g, beats=beats, sections=sections, coarse_chords=coarse)

    # -- heads -------------------------------------------------------------- #
    def _beats(self, beat_times: np.ndarray) -> list[Beat]:
        beats: list[Beat] = []
        for i, t in enumerate(beat_times):
            is_down = (i % self.beats_per_bar) == 0
            bar = i // self.beats_per_bar + 1
            beats.append(Beat(t=float(t), is_downbeat=is_down, bar=bar))
        return beats

    def _key(self, librosa, harmonic: AudioClip) -> Key:
        chroma = librosa.feature.chroma_cqt(y=harmonic.samples[0], sr=harmonic.sample_rate)
        tonic, mode, conf = estimate_key(chroma.mean(axis=1))
        return Key(tonic=tonic, mode=mode, confidence=conf)

    def _sections(self, librosa, y: np.ndarray, sr: int) -> list[Section]:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            bounds = librosa.segment.agglomerative(chroma, 8)
            times = librosa.frames_to_time(bounds, sr=sr)
        except Exception:
            return []
        total = len(y) / sr
        edges = [0.0, *[float(t) for t in times], total]
        edges = sorted(set(round(e, 3) for e in edges))
        sections: list[Section] = []
        for i in range(len(edges) - 1):
            sections.append(Section(label=f"seg{i + 1}", start_s=edges[i], end_s=edges[i + 1]))
        return sections

    def _coarse_chords(self, librosa, harmonic: AudioClip, beat_times: np.ndarray):
        y, sr = harmonic.samples[0], harmonic.sample_rate
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        spans: list[tuple[float, float, str]] = []
        edges = [*beat_times, len(y) / sr]
        for i in range(len(edges) - 1):
            a, b = float(edges[i]), float(edges[i + 1])
            fa = librosa.time_to_frames(a, sr=sr)
            fb = max(fa + 1, librosa.time_to_frames(b, sr=sr))
            frame = chroma[:, fa:fb].mean(axis=1)
            spans.append((a, b, coarse_chord(frame)))
        return spans


def build_mir_analyzer(config: Config | None = None):
    """Pick a MIR analyzer from config / available deps.

    Note: even with ``backbone: mert`` selected, per-task *predictions* use the
    librosa heads until probe weights exist (see :mod:`.backbone`). The backbone
    is used to enrich features, not to fabricate predictions it cannot make.
    """
    config = config or Config.default()
    if is_available("librosa"):
        return LibrosaMirAnalyzer(config)
    return NullMirAnalyzer()
