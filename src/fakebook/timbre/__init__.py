"""Timbre stage (Branch B — sound-design). Stub behind a stable interface.

Branch B treats a "complex voicing" as a spectral/timbre object (detuning, filter
movement, FM inharmonicity, layering) rather than a chord. Per the spec this is
implemented as a stub with the interface only, so it can be filled in without
touching the rest of the pipeline. The kernel/assembler already carry a
``timbre`` slot on every segment.
"""

from .descriptor import TimbreStub

__all__ = ["TimbreStub"]
