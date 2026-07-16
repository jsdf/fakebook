"""fakebook — musical analysis for mixed, computer-produced pop/dance music.

A pipeline that goes from an audio clip to a structured musical description plus
an interpretive layer. Two hard rules shape the design:

* the deterministic **theory kernel** (music21) enumerates every pitch / chord /
  scale fact; the **LLM interprets** only — it never computes pitches;
* everything the LLM sees is the validated IR document (:mod:`fakebook.schema`).

Only the light contract + kernel layers import eagerly here. Audio / model / LLM
stages are imported lazily by the pipeline so ``import fakebook`` works on a
minimal install.
"""

from __future__ import annotations

from .config import Config
from .schema import (
    AnalysisDocument,
    ChordInterpretation,
    Global,
    Interpretation,
    ScaleCandidate,
    Segment,
    export_json_schema,
)

__all__ = [
    "Config",
    "AnalysisDocument",
    "Segment",
    "ChordInterpretation",
    "ScaleCandidate",
    "Global",
    "Interpretation",
    "export_json_schema",
    "__version__",
]

__version__ = "0.1.0"
