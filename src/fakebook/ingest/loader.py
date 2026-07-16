"""Load audio from disk, resample to the working rate, optionally window.

Implements the :class:`~fakebook.contracts.Ingestor` protocol. Uses ``soundfile``
to read and ``librosa`` to resample (both in the ``audio`` extra), imported
lazily so this module still imports on a minimal install.

The clip windowing is deliberately loose here — the spec wants windows *aligned
to the later downbeat grid*, which is only known after the MIR stage. So the
ingest window is a coarse pre-cut for very long inputs; the harmony stage does
the beat-aligned segmentation.
"""

from __future__ import annotations

import numpy as np

from .._optional import require
from ..audio import AudioClip
from ..config import Config


class AudioLoader:
    """Ingestor: file path → resampled :class:`AudioClip`."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.target_sr = int(self.config.get("ingest.target_sample_rate", 44100))
        self.force_mono = bool(self.config.get("ingest.mono", False))

    def load(self, path: str) -> AudioClip:
        sf = require("soundfile", feature="audio loading", extra="audio")
        data, sr = sf.read(path, always_2d=True, dtype="float32")  # (n, channels)
        samples = data.T  # → (channels, n)
        clip = AudioClip(samples=samples, sample_rate=sr, source=path)
        if self.force_mono:
            clip = clip.to_mono()
        if sr != self.target_sr:
            clip = self.resample(clip, self.target_sr)
        return clip

    def from_array(
        self, samples: np.ndarray, sample_rate: int, source: str = "<array>"
    ) -> AudioClip:
        """Build a clip from an in-memory array (used by tests / synthesis)."""
        clip = AudioClip(samples=np.asarray(samples, dtype=np.float32), sample_rate=sample_rate, source=source)
        if self.force_mono:
            clip = clip.to_mono()
        if sample_rate != self.target_sr:
            clip = self.resample(clip, self.target_sr)
        return clip

    @staticmethod
    def resample(clip: AudioClip, target_sr: int) -> AudioClip:
        if clip.sample_rate == target_sr:
            return clip
        librosa = require("librosa", feature="resampling", extra="audio")
        out = librosa.resample(
            clip.samples, orig_sr=clip.sample_rate, target_sr=target_sr, axis=1
        )
        return AudioClip(out.astype(np.float32), target_sr, clip.source)
