"""Corpus registry: which reference corpora we evaluate against, and how to load.

Ground truth for this material is scarce, and what exists is scattered across
hosts with their own licences and download scripts. Rather than reimplement that,
loading goes through `mirdata <https://mirdata.readthedocs.io>`_, which owns the
index, the checksums, and the archive layout. This module is the thin adapter:
it says which corpora are useful here, what reference data each one actually
carries, and which hosts a download needs — so a blocked network reports the
host to allow instead of a traceback from inside a zip reader.

Audio is never vendored into the repository. It lands in a cache directory
(``FAKEBOOK_CORPUS_DIR``, else ``~/.cache/fakebook/corpus``) and stays there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .._optional import require
from ..errors import CorpusUnavailableError


@dataclass(frozen=True)
class CorpusSpec:
    """One reference corpus and what it can validate."""

    name: str
    mirdata_id: str
    provides: frozenset[str]  # subset of {"key", "tempo", "genre"}
    hosts: tuple[str, ...]  # hosts a download must reach
    material: str  # what the audio actually is
    notes: str = ""


# Corpora worth the bandwidth for *this* pipeline: the target is mixed,
# computer-produced pop/dance, so genre-matched EDM with key ground truth beats a
# larger corpus of rock or classical.
REGISTRY: dict[str, CorpusSpec] = {
    "giantsteps_key": CorpusSpec(
        name="giantsteps_key",
        mirdata_id="giantsteps_key",
        provides=frozenset({"key", "tempo", "genre"}),
        hosts=("zenodo.org",),
        material="604 two-minute Beatport previews (EDM), key annotated by experts",
        notes=(
            "Genre-matched and the primary key benchmark. Tempo and genre ride "
            "along from the original Beatport metadata — weaker ground truth "
            "than the annotated giantsteps-tempo set (that one is a different "
            "track list, only 43 tracks overlap), so treat tempo here as "
            "indicative and key as the real benchmark."
        ),
    ),
    "beatport_key": CorpusSpec(
        name="beatport_key",
        mirdata_id="beatport_key",
        provides=frozenset({"key", "tempo", "genre"}),
        hosts=("zenodo.org",),
        material="1486 two-minute Beatport previews (EDM), multiple key annotators",
        notes=(
            "Larger superset-style sibling of giantsteps_key. Tracks may carry "
            "several valid keys; scoring credits the best match."
        ),
    ),
}


def default_data_home() -> Path:
    """Where corpora are cached. Never inside the repository."""
    env = os.environ.get("FAKEBOOK_CORPUS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "fakebook" / "corpus"


def spec(name: str) -> CorpusSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown corpus '{name}'; known corpora: {known}") from None


@dataclass
class CorpusTrack:
    """One evaluable track: audio on disk plus whatever reference data exists."""

    track_id: str
    audio_path: Optional[str]
    ref_keys: tuple[str, ...] = ()
    ref_tempo: Optional[float] = None
    genre: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        path = self.audio_path
        if not path:
            return False
        return Path(path).exists()


def load_corpus(
    name: str,
    data_home: Optional[str | Path] = None,
    download: bool = False,
    limit: Optional[int] = None,
) -> list[CorpusTrack]:
    """Load a corpus's track list, optionally downloading it first.

    Tracks whose audio is missing are still returned (with ``audio_path`` set but
    absent on disk) so a caller can report coverage honestly rather than silently
    evaluating a subset. Pass ``download=True`` to fetch; a download that cannot
    reach its hosts raises :class:`CorpusUnavailableError` naming them.
    """
    corpus = spec(name)
    mirdata = require("mirdata", feature=f"the {name} corpus", extra="corpus")
    home = Path(data_home) if data_home else default_data_home() / name
    dataset = mirdata.initialize(corpus.mirdata_id, data_home=str(home))

    if download:
        try:
            dataset.download()
        except Exception as exc:  # network, policy, checksum — all actionable
            raise CorpusUnavailableError(corpus.name, corpus.hosts, str(exc)) from exc

    try:
        track_ids = dataset.track_ids
    except Exception as exc:
        raise CorpusUnavailableError(
            corpus.name, corpus.hosts, f"index not available locally ({exc})"
        ) from exc

    tracks: list[CorpusTrack] = []
    for track_id in track_ids[:limit] if limit else track_ids:
        tracks.append(_to_track(dataset.track(track_id)))
    return tracks


def _to_track(track) -> CorpusTrack:
    """Adapt a mirdata track to our record, tolerating per-field absence.

    mirdata raises when an annotation file is missing rather than returning None,
    and a partially downloaded corpus is normal, so each field is read defensively.
    """

    def attr(name):
        try:
            return getattr(track, name)
        except Exception:
            return None

    key = attr("key")
    keys: tuple[str, ...]
    if isinstance(key, str):
        keys = (key,)
    elif isinstance(key, (list, tuple)):
        keys = tuple(str(k) for k in key if k)
    else:
        keys = ()

    tempo = attr("tempo")
    genres = attr("genres") or attr("genre") or {}
    if isinstance(genres, dict):
        flat = [str(v) for values in genres.values() for v in (values if isinstance(values, (list, tuple)) else [values])]
        genre = ", ".join(flat)
    else:
        genre = str(genres or "")

    return CorpusTrack(
        track_id=str(track.track_id),
        audio_path=attr("audio_path"),
        ref_keys=keys,
        ref_tempo=float(tempo) if isinstance(tempo, (int, float)) and tempo else None,
        genre=genre,
    )
