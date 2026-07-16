"""Pitch-class name tables and key-profile templates (no music21 needed).

Used by the MIR stage for chroma-based key estimation and coarse ACR. The
harmony *kernel* uses music21 directly (it needs spelled-pitch reasoning); this
module is the light, numpy-only theory helper for the audio-feature side.
"""

from __future__ import annotations

import numpy as np

# Pitch-class index → sharp spelling (chroma bins are enharmonically neutral).
PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Schmuckler key profiles (major / minor), correlated against a chroma
# vector rotated to every tonic to estimate key.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Binary maj/min triad templates for coarse ACR (root, 3rd, 5th).
_MAJ_TRIAD = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_MIN_TRIAD = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)


def pc_name(pc: int, prefer_flats: bool = False) -> str:
    table = PC_FLAT if prefer_flats else PC_SHARP
    return table[pc % 12]


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl-Schmuckler key estimation from a mean chroma vector.

    Returns ``(tonic, mode, confidence)`` where confidence is the best
    correlation rescaled to [0, 1].
    """
    v = np.asarray(chroma_mean, dtype=float)
    if v.sum() <= 0:
        return "", "", 0.0
    v = v - v.mean()

    best = (-2.0, 0, "major")
    for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        p = profile - profile.mean()
        denom = np.linalg.norm(p) or 1.0
        for tonic in range(12):
            rotated = np.roll(v, -tonic)
            num = float(np.dot(rotated, p))
            corr = num / ((np.linalg.norm(rotated) or 1.0) * denom)
            if corr > best[0]:
                best = (corr, tonic, mode)

    corr, tonic, mode = best
    prefer_flats = mode == "minor"
    confidence = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    return pc_name(tonic, prefer_flats), mode, confidence


def coarse_chord(chroma_mean: np.ndarray) -> str:
    """Best maj/min triad label for a chroma frame, ``"root:quality"``.

    Deliberately a ~maj/min vocabulary — this is the lossy ACR context of
    principle 3, never used for voicing detail.
    """
    v = np.asarray(chroma_mean, dtype=float)
    if v.sum() <= 0:
        return "N"
    v = v / (np.linalg.norm(v) or 1.0)
    best = (-1.0, "N")
    for tonic in range(12):
        for tmpl, qual in ((_MAJ_TRIAD, "maj"), (_MIN_TRIAD, "min")):
            rolled = np.roll(tmpl, tonic)
            score = float(np.dot(v, rolled / np.linalg.norm(rolled)))
            if score > best[0]:
                best = (score, f"{pc_name(tonic)}:{qual}")
    return best[1]
