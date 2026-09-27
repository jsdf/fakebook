"""Run the pipeline over a reference corpus and score it.

The report is per-track first and aggregate second, deliberately: with 604 tracks
the interesting question is never "what is the mean" but "which tracks, and do
they have something in common" (one genre, one tempo range, one key). Rows carry
the genre label so that question can be asked of the output.

A corpus run must survive a bad file, so a track that raises is recorded as an
error row and the run continues.
"""

from __future__ import annotations

import collections
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

from ..config import Config
from .metrics import best_key_score, tempo_accuracy
from .registry import CorpusTrack


@dataclass
class TrackResult:
    track_id: str
    genre: str = ""
    ref_key: str = ""
    est_key: str = ""
    key_score: float = 0.0
    key_relationship: str = "unknown"
    ref_tempo: Optional[float] = None
    est_tempo: Optional[float] = None
    tempo_acc1: bool = False
    tempo_acc2: bool = False
    error: str = ""


@dataclass
class EvaluationReport:
    corpus: str
    n_tracks: int = 0  # tracks actually analyzed
    n_missing_audio: int = 0
    n_errors: int = 0
    key_mean_score: float = 0.0
    key_accuracy: float = 0.0  # share scored exactly correct
    key_relationships: dict[str, int] = field(default_factory=dict)
    tempo_acc1: float = 0.0
    tempo_acc2: float = 0.0
    n_key_scored: int = 0
    n_tempo_scored: int = 0
    rows: list[TrackResult] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=indent)

    def summary(self) -> str:
        """Human-readable digest — what a run prints when it finishes."""
        lines = [
            f"corpus: {self.corpus}",
            f"analyzed: {self.n_tracks}  missing audio: {self.n_missing_audio}  errors: {self.n_errors}",
        ]
        if self.n_key_scored:
            rel = ", ".join(f"{k} {v}" for k, v in sorted(self.key_relationships.items()))
            lines += [
                f"key: exact {self.key_accuracy:.1%}, MIREX weighted {self.key_mean_score:.3f} "
                f"(n={self.n_key_scored})",
                f"     relationships: {rel}",
            ]
        if self.n_tempo_scored:
            lines.append(
                f"tempo: acc1 {self.tempo_acc1:.1%}, acc2 {self.tempo_acc2:.1%} "
                f"(n={self.n_tempo_scored})"
            )
        return "\n".join(lines)


def evaluate_corpus(
    tracks: Iterable[CorpusTrack],
    corpus_name: str = "",
    pipeline=None,
    config: Optional[Config] = None,
    on_track: Optional[Callable[[TrackResult], None]] = None,
) -> EvaluationReport:
    """Analyze every available track and score key and tempo against reference.

    ``pipeline`` defaults to one built from ``config``; pass an explicit pipeline
    (or a stub with ``analyze_file``) to evaluate a variant configuration.
    """
    if pipeline is None:
        from ..pipeline import Pipeline

        pipeline = Pipeline.from_config(config or Config.load())

    report = EvaluationReport(corpus=corpus_name)
    relationships: collections.Counter = collections.Counter()
    key_scores: list[float] = []
    tempo_hits = [0, 0]

    for track in tracks:
        if not track.is_available:
            report.n_missing_audio += 1
            continue

        row = TrackResult(
            track_id=track.track_id,
            genre=track.genre,
            ref_key=track.ref_keys[0] if track.ref_keys else "",
            ref_tempo=track.ref_tempo,
        )
        try:
            doc = pipeline.analyze_file(track.audio_path)
        except Exception as exc:  # one bad file must not end the run
            row.error = f"{type(exc).__name__}: {exc}"
            report.n_errors += 1
            report.rows.append(row)
            if on_track:
                on_track(row)
            continue

        report.n_tracks += 1
        key = doc.global_.key
        row.est_key = f"{key.tonic} {key.mode}".strip()
        row.est_tempo = doc.global_.tempo_bpm or None

        if track.ref_keys:
            score, relationship, credited = best_key_score(track.ref_keys, row.est_key)
            row.key_score, row.key_relationship, row.ref_key = score, relationship, credited
            if relationship != "unknown":
                key_scores.append(score)
                relationships[relationship] += 1
        if track.ref_tempo:
            row.tempo_acc1, row.tempo_acc2 = tempo_accuracy(track.ref_tempo, row.est_tempo)
            report.n_tempo_scored += 1
            tempo_hits[0] += int(row.tempo_acc1)
            tempo_hits[1] += int(row.tempo_acc2)

        report.rows.append(row)
        if on_track:
            on_track(row)

    report.n_key_scored = len(key_scores)
    if key_scores:
        report.key_mean_score = round(sum(key_scores) / len(key_scores), 4)
        report.key_accuracy = round(relationships["correct"] / len(key_scores), 4)
    report.key_relationships = dict(relationships)
    if report.n_tempo_scored:
        report.tempo_acc1 = round(tempo_hits[0] / report.n_tempo_scored, 4)
        report.tempo_acc2 = round(tempo_hits[1] / report.n_tempo_scored, 4)
    return report
