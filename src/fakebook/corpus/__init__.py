"""Reference-corpus evaluation (the validation tier the spec calls for).

Ground truth is scarce for mixed, computer-produced pop/dance, so the corpora
here are chosen for genre match first: Beatport/GiantSteps EDM previews with
expert key annotations. Loading is delegated to ``mirdata``; this package adds
the registry of what is worth evaluating, MIREX-comparable metrics, and a runner
that scores the real pipeline track by track.

Nothing here is imported by the pipeline itself — it is a validation harness, and
needs the ``corpus`` extra.
"""

from .metrics import beat_f_measure, best_key_score, key_score, normalize_key, tempo_accuracy
from .registry import REGISTRY, CorpusSpec, CorpusTrack, default_data_home, load_corpus, spec
from .runner import EvaluationReport, TrackResult, evaluate_corpus

__all__ = [
    "REGISTRY",
    "CorpusSpec",
    "CorpusTrack",
    "EvaluationReport",
    "TrackResult",
    "beat_f_measure",
    "best_key_score",
    "default_data_home",
    "evaluate_corpus",
    "key_score",
    "load_corpus",
    "normalize_key",
    "spec",
    "tempo_accuracy",
]
