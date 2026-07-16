"""In-memory audio containers passed between the early stages.

These are *not* part of the serialized IR (that is :mod:`fakebook.schema`).
They are the runtime hand-off objects: an :class:`AudioClip` flows
ingest → separation, and :class:`Stems` flows separation → mir/harmony/timbre.

Only numpy is required here (a core dependency), so these containers import on a
minimal install even though the code that *produces* them needs the audio extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


@dataclass
class AudioClip:
    """A block of PCM audio.

    ``samples`` is shape ``(channels, n)`` (channels-first). Mono is ``(1, n)``.
    """

    samples: np.ndarray
    sample_rate: int
    source: str = ""

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[np.newaxis, :]
        if self.samples.ndim != 2:
            raise ValueError(f"samples must be 1-D or 2-D, got {self.samples.ndim}-D")

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0

    def to_mono(self) -> "AudioClip":
        if self.n_channels == 1:
            return self
        mono = self.samples.mean(axis=0, keepdims=True)
        return AudioClip(mono, self.sample_rate, self.source)

    def window(self, max_s: float) -> Iterator["AudioClip"]:
        """Yield non-overlapping windows of at most ``max_s`` seconds."""
        if max_s <= 0:
            yield self
            return
        step = int(max_s * self.sample_rate)
        for start in range(0, self.n_samples, step):
            chunk = self.samples[:, start : start + step]
            yield AudioClip(chunk, self.sample_rate, self.source)

    def slice_s(self, start_s: float, end_s: float) -> "AudioClip":
        a = max(0, int(start_s * self.sample_rate))
        b = min(self.n_samples, int(end_s * self.sample_rate))
        return AudioClip(self.samples[:, a:b], self.sample_rate, self.source)


@dataclass
class Stems:
    """Separated stems. ``harmonic`` is the stem the harmony stage consumes."""

    harmonic: AudioClip
    bass: Optional[AudioClip] = None
    drums: Optional[AudioClip] = None
    vocal: Optional[AudioClip] = None
    # The full mix, kept for whole-mix MIR (key/tempo/tags often prefer the mix).
    mix: Optional[AudioClip] = None
    extra: dict[str, AudioClip] = field(default_factory=dict)

    def named(self) -> dict[str, AudioClip]:
        out = {"harmonic": self.harmonic}
        for name in ("bass", "drums", "vocal", "mix"):
            clip = getattr(self, name)
            if clip is not None:
                out[name] = clip
        out.update(self.extra)
        return out
