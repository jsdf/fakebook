"""Tests for the reference-corpus evaluation harness.

The harness exists to score the pipeline against annotated corpora, but none of
it may *require* a corpus to be tested: a download is gigabytes and a network
away. So the metrics are checked against hand-worked cases, and the runner is
driven by a stub pipeline over fabricated tracks — which also pins the behaviour
that matters most on a real run, that one unreadable file does not end it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fakebook.corpus import (
    REGISTRY,
    CorpusTrack,
    EvaluationReport,
    default_data_home,
    evaluate_corpus,
    normalize_key,
    spec,
    tempo_accuracy,
)
from fakebook.errors import CorpusUnavailableError
from fakebook.schema import AnalysisDocument, Global, Key, Meta


# --------------------------------------------------------------------------- #
# key normalization (pure — no mir_eval needed)                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Eb minor", "Eb minor"),
        ("D:min", "D minor"),  # Harte style, as POP909/ChoCo write it
        ("f MINOR", "F minor"),
        ("C# Major", "C# major"),
        ("A  maj", "A major"),
        ("Cs min", "C# minor"),  # 's' for sharp
        ("", "X"),
        (None, "X"),
        ("X", "X"),
        ("C", "X"),  # no mode: refuse rather than assume major
        ("Cb minor", "X"),  # outside mir_eval's vocabulary
        ("H dur", "X"),  # German spelling is not supported; say so
        ("C lydian", "X"),  # mode we cannot score
    ],
)
def test_normalize_key(raw, expected):
    assert normalize_key(raw) == expected


# --------------------------------------------------------------------------- #
# key scoring (MIREX weighting)                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ref,est,score,relationship",
    [
        ("C major", "C major", 1.0, "correct"),
        ("C major", "G major", 0.5, "fifth"),
        ("C major", "A minor", 0.3, "relative"),
        ("C major", "C minor", 0.2, "parallel"),
        ("C major", "D major", 0.0, "other"),
        ("Eb minor", "Eb minor", 1.0, "correct"),
        ("", "C major", 0.0, "unknown"),
        ("C major", "", 0.0, "unknown"),
    ],
)
def test_key_score_relationships(ref, est, score, relationship):
    pytest.importorskip("mir_eval")
    from fakebook.corpus import key_score

    assert key_score(ref, est) == (score, relationship)


def test_best_key_score_credits_the_matching_reference():
    pytest.importorskip("mir_eval")
    from fakebook.corpus import best_key_score

    # beatport_key tracks can carry several valid keys; the best one counts.
    score, relationship, credited = best_key_score(("A minor", "C major"), "C major")
    assert (score, relationship, credited) == (1.0, "correct", "C major")


def test_best_key_score_with_no_reference_is_unknown():
    pytest.importorskip("mir_eval")
    from fakebook.corpus import best_key_score

    assert best_key_score((), "C major") == (0.0, "unknown", "X")


# --------------------------------------------------------------------------- #
# tempo                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ref,est,acc1,acc2",
    [
        (128.0, 128.0, True, True),
        (128.0, 132.0, True, True),  # 3.1% — inside the 4% window
        (128.0, 136.0, False, False),  # 6.3% — outside it
        (128.0, 64.0, False, True),  # half: a metrical reading, not a wrong tempo
        (128.0, 256.0, False, True),  # double
        (128.0, 128.0 / 3, False, True),  # a third
        (174.0, 87.0, False, True),  # d&b heard at half time
        (128.0, None, False, False),
        (None, 128.0, False, False),
        (128.0, 0.0, False, False),
    ],
)
def test_tempo_accuracy(ref, est, acc1, acc2):
    assert tempo_accuracy(ref, est) == (acc1, acc2)


# --------------------------------------------------------------------------- #
# beats                                                                       #
# --------------------------------------------------------------------------- #
def test_beat_f_measure_rewards_alignment_and_punishes_drift():
    pytest.importorskip("mir_eval")
    from fakebook.corpus import beat_f_measure

    ref = np.arange(6.0, 20.0, 0.5)  # past mir_eval's 5 s trim
    assert beat_f_measure(ref, ref) == 1.0
    assert beat_f_measure(ref, ref + 0.02) == 1.0  # inside the 70 ms window
    assert beat_f_measure(ref, ref + 0.3) == 0.0  # outside it
    assert 0.6 < beat_f_measure(ref, ref[::2]) < 0.7  # half the beats found
    assert beat_f_measure(ref, []) == 0.0


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
def test_registry_specs_name_their_hosts_and_reference_data():
    assert set(REGISTRY) >= {"giantsteps_key", "beatport_key"}
    gs = spec("giantsteps_key")
    assert gs.hosts == ("zenodo.org",)
    assert "key" in gs.provides
    assert gs.notes  # every spec says what its ground truth is worth


def test_unknown_corpus_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        spec("nope")
    assert "giantsteps_key" in str(excinfo.value)


def test_data_home_is_outside_the_repo_and_env_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKEBOOK_CORPUS_DIR", str(tmp_path / "corpora"))
    assert default_data_home() == tmp_path / "corpora"
    monkeypatch.delenv("FAKEBOOK_CORPUS_DIR")
    assert "fakebook" in str(default_data_home())  # a cache dir, never the checkout


def test_corpus_unavailable_error_names_the_host_to_allow():
    err = CorpusUnavailableError("giantsteps_key", ("zenodo.org",), "403 Forbidden")
    text = str(err)
    assert "zenodo.org" in text and "giantsteps_key" in text and "403" in text


def test_track_availability_follows_the_file(tmp_path):
    missing = CorpusTrack("1", str(tmp_path / "gone.mp3"))
    assert not missing.is_available
    present = tmp_path / "there.mp3"
    present.write_bytes(b"\x00")
    assert CorpusTrack("2", str(present)).is_available
    assert not CorpusTrack("3", None).is_available


# --------------------------------------------------------------------------- #
# runner                                                                      #
# --------------------------------------------------------------------------- #
class StubPipeline:
    """Returns a canned analysis per path, or raises for paths marked bad."""

    def __init__(self, answers):
        self.answers = answers
        self.seen: list[str] = []

    def analyze_file(self, path, interpret=False):
        self.seen.append(path)
        answer = self.answers[path]
        if isinstance(answer, Exception):
            raise answer
        tonic, mode, tempo = answer
        return AnalysisDocument(
            meta=Meta(duration_s=120.0, sample_rate=44100, source=path),
            global_=Global(key=Key(tonic=tonic, mode=mode, confidence=0.9), tempo_bpm=tempo),
        )


def _tracks(tmp_path, n):
    made = []
    for i in range(n):
        p = tmp_path / f"{i}.mp3"
        p.write_bytes(b"\x00")
        made.append(str(p))
    return made


def test_runner_scores_key_and_tempo_and_aggregates(tmp_path):
    pytest.importorskip("mir_eval")
    paths = _tracks(tmp_path, 3)
    pipeline = StubPipeline(
        {
            paths[0]: ("F", "minor", 128.0),  # exact key, exact tempo
            paths[1]: ("C", "major", 64.0),  # relative key, half tempo
            paths[2]: ("D", "major", 174.0),  # wrong key, exact tempo
        }
    )
    tracks = [
        CorpusTrack("t0", paths[0], ("F minor",), 128.0, genre="techno"),
        CorpusTrack("t1", paths[1], ("A minor",), 128.0, genre="trance"),
        CorpusTrack("t2", paths[2], ("F minor",), 174.0, genre="drum-and-bass"),
    ]
    report = evaluate_corpus(tracks, corpus_name="fake", pipeline=pipeline)

    assert isinstance(report, EvaluationReport)
    assert report.n_tracks == 3 and report.n_errors == 0 and report.n_missing_audio == 0
    assert report.key_relationships == {"correct": 1, "relative": 1, "other": 1}
    assert report.key_accuracy == pytest.approx(1 / 3, abs=1e-3)
    assert report.key_mean_score == pytest.approx((1.0 + 0.3 + 0.0) / 3, abs=1e-3)
    assert report.tempo_acc1 == pytest.approx(2 / 3, abs=1e-3)  # the half-tempo misses
    assert report.tempo_acc2 == 1.0  # ...but counts as a metrical reading
    assert [r.genre for r in report.rows] == ["techno", "trance", "drum-and-bass"]


def test_runner_survives_a_bad_file_and_records_it(tmp_path):
    pytest.importorskip("mir_eval")
    paths = _tracks(tmp_path, 3)
    pipeline = StubPipeline(
        {
            paths[0]: ("F", "minor", 128.0),
            paths[1]: RuntimeError("corrupt mp3 frame"),
            paths[2]: ("G", "minor", 130.0),
        }
    )
    tracks = [CorpusTrack(f"t{i}", p, ("F minor",), 128.0) for i, p in enumerate(paths)]
    report = evaluate_corpus(tracks, corpus_name="fake", pipeline=pipeline)

    assert report.n_tracks == 2 and report.n_errors == 1
    assert pipeline.seen == paths  # the run continued past the failure
    bad = [r for r in report.rows if r.error]
    assert len(bad) == 1 and "corrupt mp3 frame" in bad[0].error


def test_runner_counts_missing_audio_instead_of_skipping_silently(tmp_path):
    pytest.importorskip("mir_eval")
    paths = _tracks(tmp_path, 1)
    pipeline = StubPipeline({paths[0]: ("F", "minor", 128.0)})
    tracks = [
        CorpusTrack("present", paths[0], ("F minor",), 128.0),
        CorpusTrack("absent", str(tmp_path / "not-downloaded.mp3"), ("A minor",), 120.0),
    ]
    report = evaluate_corpus(tracks, corpus_name="fake", pipeline=pipeline)
    assert report.n_tracks == 1 and report.n_missing_audio == 1


def test_report_json_and_summary_are_self_describing(tmp_path):
    pytest.importorskip("mir_eval")
    paths = _tracks(tmp_path, 1)
    pipeline = StubPipeline({paths[0]: ("F", "minor", 128.0)})
    report = evaluate_corpus(
        [CorpusTrack("t0", paths[0], ("F minor",), 128.0, genre="techno")],
        corpus_name="giantsteps_key",
        pipeline=pipeline,
    )
    import json

    payload = json.loads(report.to_json())
    assert payload["corpus"] == "giantsteps_key"
    assert payload["rows"][0]["track_id"] == "t0"
    text = report.summary()
    assert "giantsteps_key" in text and "key:" in text and "tempo:" in text


def test_cli_reports_a_blocked_corpus_as_a_message_not_a_traceback(monkeypatch, capsys):
    # The most likely failure on a fresh machine is a denied download host. The
    # CLI must say which host, and exit non-zero, without a traceback.
    from fakebook import cli
    from fakebook.corpus import registry

    def blocked(*args, **kwargs):
        raise CorpusUnavailableError("giantsteps_key", ("zenodo.org",), "403 Forbidden")

    monkeypatch.setattr(registry, "load_corpus", blocked)
    monkeypatch.setattr("fakebook.corpus.load_corpus", blocked)
    assert cli.main(["evaluate", "giantsteps_key"]) == 1
    err = capsys.readouterr().err
    assert "zenodo.org" in err and "Traceback" not in err


def test_cli_lists_registered_corpora(capsys):
    from fakebook import cli

    assert cli.main(["evaluate", "list"]) == 0
    out = capsys.readouterr().out
    assert "giantsteps_key" in out and "zenodo.org" in out
