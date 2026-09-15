"""Timbre stage (Branch B — sound-design).

Branch B treats a "complex voicing" as a spectral/timbre object — detuning,
filter movement, FM inharmonicity, layering — rather than a chord. It runs
beside the harmony branch over the same chord spans and fills the ``timbre``
slot the IR carries on every segment.

:class:`SpectralTimbre` is the real extractor (numpy only — no extras needed);
:class:`TimbreStub` is the explicit "not measured" path.
"""

from .descriptor import SpectralTimbre, TimbreStub, build_timbre, describe_tags

__all__ = ["SpectralTimbre", "TimbreStub", "build_timbre", "describe_tags"]
