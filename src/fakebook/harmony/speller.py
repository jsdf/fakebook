"""Interpretation-aware pitch spelling.

Given a root and a degree number for each note, choose the *letter* from the
degree and compute the accidental from the actual pitch class. This keeps the
♯9-vs-♭3 distinction (principle 4): the same pc is spelled differently under
different chord readings, and stays correct for any root (including altered
roots like F♯, where a ♭3 is A natural, not A♭).
"""

from __future__ import annotations

# Preferred (flat-leaning, pop-idiomatic) root spellings per pitch class.
_ROOT_NAME = {
    0: ("C", 0), 1: ("D", -1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0),
    5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0),
    10: ("B", -1), 11: ("B", 0),
}

_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Degree number → how many letter-steps above the root letter.
_DEGREE_STEP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 1, 11: 3, 13: 5}


def _accidental_symbol(n: int) -> str:
    if n == 0:
        return ""
    return ("#" * n) if n > 0 else ("b" * -n)


def root_spelling(root_pc: int) -> tuple[str, int]:
    """Return (letter, accidental) for a root pitch class."""
    return _ROOT_NAME[root_pc % 12]


def spell(root_pc: int, interval: int, degree: int) -> str:
    """Spell the note ``interval`` semitones above ``root_pc`` as ``degree``.

    The degree fixes the letter; the accidental is derived so the spelled pitch
    equals ``(root_pc + interval) % 12`` exactly.
    """
    root_letter, _ = root_spelling(root_pc)
    step = _DEGREE_STEP[degree]
    target_letter = _LETTERS[(_LETTERS.index(root_letter) + step) % 7]
    actual_pc = (root_pc + interval) % 12
    natural_pc = _LETTER_PC[target_letter]
    # signed accidental in [-6, 5]; in practice small.
    acc = ((actual_pc - natural_pc + 6) % 12) - 6
    return f"{target_letter}{_accidental_symbol(acc)}"


def spell_pitch_classes(pcs: list[int], prefer_flats: bool = True) -> list[str]:
    """Neutral spelling of a bare pc list (no chord context)."""
    from ..pitch import pc_name

    return [pc_name(pc, prefer_flats=prefer_flats) for pc in pcs]
