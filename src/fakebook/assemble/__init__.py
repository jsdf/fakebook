"""Assemble stage: merge all stage outputs into one validated IR document."""

from .assembler import Assembler, CoherenceReport, coherence_checks

__all__ = ["Assembler", "CoherenceReport", "coherence_checks"]
