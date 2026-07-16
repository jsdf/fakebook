"""Chord-quality and scale vocabulary for the deterministic kernel.

Each chord quality maps *interval-from-root* (semitones, mod 12) → *degree
number* (1,2,3,4,5,6,7,9,11,13). The degree number selects the letter name so
the same pitch-class can be spelled ♯9 or ♭3 depending on the reading — this is
principle 4 made concrete. Accidentals are then computed from the actual pitch
(see :mod:`fakebook.harmony.speller`), so spellings stay correct for any root.

The vocabulary is intentionally rich (upper structures, altered dominants,
quartal) because off-the-shelf ACR would flatten these (principle 3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quality:
    name: str  # suffix appended to the root, e.g. "m7", "7#9", "maj7"
    degrees: dict[int, int]  # interval(semitones) -> degree number

    @property
    def intervals(self) -> frozenset[int]:
        return frozenset(self.degrees)

    @property
    def size(self) -> int:
        return len(self.degrees)


# Ordered roughly simple → complex; enumeration keeps the highest-scoring few.
QUALITIES: tuple[Quality, ...] = (
    # triads
    Quality("maj", {0: 1, 4: 3, 7: 5}),
    Quality("m", {0: 1, 3: 3, 7: 5}),
    Quality("dim", {0: 1, 3: 3, 6: 5}),
    Quality("aug", {0: 1, 4: 3, 8: 5}),
    Quality("sus4", {0: 1, 5: 4, 7: 5}),
    Quality("sus2", {0: 1, 2: 2, 7: 5}),
    # sixths
    Quality("6", {0: 1, 4: 3, 7: 5, 9: 6}),
    Quality("m6", {0: 1, 3: 3, 7: 5, 9: 6}),
    # sevenths
    Quality("maj7", {0: 1, 4: 3, 7: 5, 11: 7}),
    Quality("7", {0: 1, 4: 3, 7: 5, 10: 7}),
    Quality("m7", {0: 1, 3: 3, 7: 5, 10: 7}),
    Quality("m7b5", {0: 1, 3: 3, 6: 5, 10: 7}),
    Quality("dim7", {0: 1, 3: 3, 6: 5, 9: 6}),
    Quality("mMaj7", {0: 1, 3: 3, 7: 5, 11: 7}),
    Quality("maj7#5", {0: 1, 4: 3, 8: 5, 11: 7}),
    Quality("7#5", {0: 1, 4: 3, 8: 5, 10: 7}),
    Quality("7b5", {0: 1, 4: 3, 6: 5, 10: 7}),
    # ninths
    Quality("maj9", {0: 1, 4: 3, 7: 5, 11: 7, 2: 9}),
    Quality("9", {0: 1, 4: 3, 7: 5, 10: 7, 2: 9}),
    Quality("m9", {0: 1, 3: 3, 7: 5, 10: 7, 2: 9}),
    Quality("7b9", {0: 1, 4: 3, 7: 5, 10: 7, 1: 9}),
    Quality("7#9", {0: 1, 4: 3, 7: 5, 10: 7, 3: 9}),
    Quality("m11", {0: 1, 3: 3, 7: 5, 10: 7, 2: 9, 5: 11}),
    # extended dominants
    Quality("7#11", {0: 1, 4: 3, 7: 5, 10: 7, 6: 11}),
    Quality("13", {0: 1, 4: 3, 7: 5, 10: 7, 9: 13}),
    Quality("7#9#5", {0: 1, 4: 3, 8: 5, 10: 7, 3: 9}),
    Quality("maj7#11", {0: 1, 4: 3, 7: 5, 11: 7, 6: 11}),
)

# Fallback interval → degree, used to spell tones that fall outside the matched
# quality (i.e. added tensions). Common jazz convention.
TENSION_DEGREE: dict[int, int] = {
    0: 1,
    1: 9,   # b9
    2: 9,   # 9
    3: 9,   # #9
    4: 3,
    5: 11,  # 11
    6: 11,  # #11
    7: 5,
    8: 13,  # b13
    9: 13,  # 13
    10: 7,  # b7
    11: 7,  # maj7
}


@dataclass(frozen=True)
class ScaleType:
    name: str
    intervals: frozenset[int]


# Chord-scale candidates. Modes of major, melodic minor, plus symmetric scales.
SCALES: tuple[ScaleType, ...] = (
    ScaleType("ionian", frozenset({0, 2, 4, 5, 7, 9, 11})),
    ScaleType("dorian", frozenset({0, 2, 3, 5, 7, 9, 10})),
    ScaleType("phrygian", frozenset({0, 1, 3, 5, 7, 8, 10})),
    ScaleType("lydian", frozenset({0, 2, 4, 6, 7, 9, 11})),
    ScaleType("mixolydian", frozenset({0, 2, 4, 5, 7, 9, 10})),
    ScaleType("aeolian", frozenset({0, 2, 3, 5, 7, 8, 10})),
    ScaleType("locrian", frozenset({0, 1, 3, 5, 6, 8, 10})),
    ScaleType("melodic minor", frozenset({0, 2, 3, 5, 7, 9, 11})),
    ScaleType("lydian dominant", frozenset({0, 2, 4, 6, 7, 9, 10})),
    ScaleType("altered", frozenset({0, 1, 3, 4, 6, 8, 10})),
    ScaleType("harmonic minor", frozenset({0, 2, 3, 5, 7, 8, 11})),
    ScaleType("phrygian dominant", frozenset({0, 1, 4, 5, 7, 8, 10})),
    ScaleType("whole tone", frozenset({0, 2, 4, 6, 8, 10})),
    ScaleType("diminished (h-w)", frozenset({0, 1, 3, 4, 6, 7, 9, 10})),
    ScaleType("major pentatonic", frozenset({0, 2, 4, 7, 9})),
    ScaleType("minor pentatonic", frozenset({0, 3, 5, 7, 10})),
)

# "Commonness" prior so ties break toward everyday scales (higher = preferred).
_SCALE_PRIOR = {s.name: 1.0 for s in SCALES}
for _n in ("ionian", "aeolian", "dorian", "mixolydian"):
    _SCALE_PRIOR[_n] = 1.15


def scale_prior(name: str) -> float:
    return _SCALE_PRIOR.get(name, 1.0)
