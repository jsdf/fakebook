"""Evaluation metrics for corpus runs.

Thin, explicit wrappers over :mod:`mir_eval` so a report says the same thing a
MIREX-style evaluation would, plus the two places we deliberately do our own
thing:

* **Key** uses ``mir_eval.key.weighted_score``, but the report also keeps the
  *relationship* (correct / fifth / relative / parallel / other). A scalar mean
  hides the interesting failure: confusing a key with its relative minor is a
  different bug from confusing it with a tritone away.
* **Tempo** does not use ``mir_eval.tempo.detection``, which is built for
  perceptual tempo *pairs* (two reference tempi and a weight). Corpus tempo
  ground truth here is a single value, so Accuracy1/Accuracy2 are computed
  directly from their definitions rather than by faking a second reference.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .._optional import require

KEY_UNKNOWN = "X"

# mir_eval.key.weighted_score's score -> relationship, inverted so a report can
# name the failure instead of only averaging it.
_KEY_RELATIONSHIP = {
    1.0: "correct",
    0.5: "fifth",
    0.3: "relative",
    0.2: "parallel",
    0.0: "other",
}

# mir_eval's key vocabulary (mir_eval.key.KEY_TO_SEMITONE minus 'x'), inlined so
# normalization is pure: an out-of-vocabulary tonic like "Cb" becomes "X" here
# instead of raising from inside mir_eval later.
_TONICS = frozenset(
    "C C# Db D D# Eb E F F# Gb G G# Ab A A# Bb B".split()
)

_MODES = {
    "major": "major", "maj": "major", "M": "major", "ionian": "major",
    "minor": "minor", "min": "minor", "m": "minor", "aeolian": "minor",
}


def normalize_key(label: Optional[str]) -> str:
    """Normalize a key label to mir_eval's ``'<tonic> <mode>'`` form.

    Accepts the spellings the corpora and the pipeline actually produce —
    ``"Eb minor"``, ``"D:min"``, ``"f MINOR"``, ``"C# Major"`` — and returns
    ``"X"`` (mir_eval's uncategorized) for anything empty, unknown, or without a
    mode. Returning ``X`` rather than guessing a mode keeps an unparsable label
    from silently scoring as a major-key hit.
    """
    if not label:
        return KEY_UNKNOWN
    text = str(label).strip().replace(":", " ").replace("-", " ")
    if not text or text.upper() == KEY_UNKNOWN:
        return KEY_UNKNOWN
    parts = text.split()
    tonic = parts[0]
    tonic = tonic[0].upper() + tonic[1:].lower().replace("s", "#")
    mode = _MODES.get(parts[1].lower()) if len(parts) > 1 else None
    if mode is None or tonic not in _TONICS:
        return KEY_UNKNOWN
    return f"{tonic} {mode}"


def key_score(reference: Optional[str], estimate: Optional[str]) -> tuple[float, str]:
    """Return ``(weighted_score, relationship)`` for one key estimate.

    An unparsable or missing label on either side scores 0.0 as ``"unknown"``,
    kept distinct from ``"other"`` so a report separates "we got the key wrong"
    from "there was nothing to compare".
    """
    ref, est = normalize_key(reference), normalize_key(estimate)
    if ref == KEY_UNKNOWN or est == KEY_UNKNOWN:
        return 0.0, "unknown"
    mir_eval = require("mir_eval", feature="key evaluation", extra="corpus")
    score = float(mir_eval.key.weighted_score(ref, est))
    return score, _KEY_RELATIONSHIP.get(round(score, 1), "other")


def best_key_score(
    references: Sequence[Optional[str]], estimate: Optional[str]
) -> tuple[float, str, str]:
    """Score against several reference keys, keeping the best.

    Some corpora (beatport_key) carry more than one valid key per track. MIREX
    practice is to credit the best match; the winning reference is returned so
    the report can show which one was credited.
    """
    best = (0.0, "unknown", KEY_UNKNOWN)
    for ref in references or []:
        score, relationship = key_score(ref, estimate)
        if score >= best[0] and (score > best[0] or best[2] == KEY_UNKNOWN):
            best = (score, relationship, normalize_key(ref))
    return best


def tempo_accuracy(
    reference_bpm: Optional[float], estimate_bpm: Optional[float], tol: float = 0.04
) -> tuple[bool, bool]:
    """Return ``(accuracy1, accuracy2)`` for one tempo estimate.

    Accuracy1 is a hit within ``tol`` (4% by convention). Accuracy2 also credits
    the octave errors a beat tracker habitually makes — half, double, a third,
    and triple the reference — which on this material is usually a metrical
    reading, not a wrong tempo.
    """
    if not reference_bpm or not estimate_bpm or reference_bpm <= 0 or estimate_bpm <= 0:
        return False, False
    ref, est = float(reference_bpm), float(estimate_bpm)
    acc1 = abs(est - ref) <= tol * ref
    acc2 = any(abs(est - ref * f) <= tol * ref * f for f in (1 / 3, 0.5, 1.0, 2.0, 3.0))
    return bool(acc1), bool(acc2)


def beat_f_measure(
    reference_times: Iterable[float], estimate_times: Iterable[float], tol: float = 0.07
) -> float:
    """Beat-tracking F-measure (mir_eval, 70 ms window by default).

    Works for downbeats too — pass only the downbeat times on both sides.
    """
    mir_eval = require("mir_eval", feature="beat evaluation", extra="corpus")
    ref = np.asarray(sorted(float(t) for t in reference_times), dtype=float)
    est = np.asarray(sorted(float(t) for t in estimate_times), dtype=float)
    if ref.size == 0 or est.size == 0:
        return 0.0
    # mir_eval discards beats in the first 5 s; keep its convention rather than
    # inventing our own, so numbers are comparable with published ones.
    return float(mir_eval.beat.f_measure(ref, est, f_measure_threshold=tol))
