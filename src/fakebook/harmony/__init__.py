"""Harmony stage (Branch A): salience → pitch classes → music21 kernel → Segment.

The kernel is the deterministic heart; it enumerates the valid readings that the
LLM later selects among.
"""

from .kernel import HarmonyKernel
from .salience import BasicPitchSalience, ChromaSalience, build_salience, threshold_chroma
from .segmentation import build_spans, spans_fixed, spans_from_downbeats

__all__ = [
    "HarmonyKernel",
    "ChromaSalience",
    "BasicPitchSalience",
    "build_salience",
    "threshold_chroma",
    "build_spans",
    "spans_from_downbeats",
    "spans_fixed",
]
