"""Branch B stub: harmonic-stem span → spectral descriptor.

Deliberately minimal. It implements the :class:`~fakebook.contracts.TimbreAnalyzer`
protocol and returns an empty :class:`~fakebook.schema.TimbreDescriptor` with the
intended fields present but unfilled, so the assembly/interpret stages already
have a stable shape to consume. The real feature extraction is left as clearly
scoped TODOs.
"""

from __future__ import annotations

from ..audio import AudioClip
from ..config import Config
from ..contracts import ChordSpan
from ..schema import TimbreDescriptor


class TimbreStub:
    """No-op timbre analyzer. Returns the descriptor shape, not real features."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()

    def describe(self, harmonic: AudioClip, span: ChordSpan) -> TimbreDescriptor:
        # TODO(branch-b): spectral centroid & bandwidth over the span (librosa).
        # TODO(branch-b): inharmonicity via partial-frequency deviation from an
        #                 ideal harmonic series (sinusoidal/partial tracking).
        # TODO(branch-b): detuning estimate from super-saw/unison partial spread.
        # TODO(branch-b): partial count / spectral-peak tracking for layering.
        return TimbreDescriptor(
            spectral_centroid_hz=None,
            spectral_bandwidth_hz=None,
            inharmonicity=None,
            detune_cents=None,
            n_partials=None,
            notes="timbre stub — Branch B not yet implemented",
        )
