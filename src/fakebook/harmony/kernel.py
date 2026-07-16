"""The deterministic theory kernel — the hard guardrail (principle 2).

Input: a thresholded pitch-class collection for one chord span
(:class:`~fakebook.contracts.PitchClassObservation`).
Output: a fully enumerated :class:`~fakebook.schema.Segment` — pc-set descriptors,
the ranked space of chord readings, chord-scale candidates, voicing analysis, and
spelled pitches under each reading.

Every pitch/interval/chord/scale fact downstream originates here. The LLM stage
receives this enumeration and may only *select and explain* among it.

music21 is used for the pc-set descriptors (prime/normal form, Forte class,
interval vector). The chord/scale enumeration uses the local vocabulary so the
rich voicing vocabulary (upper structures, altered dominants, quartal) is under
our control rather than an ACR maj/min vocabulary.
"""

from __future__ import annotations

from ..config import Config
from ..contracts import PitchClassObservation
from ..schema import (
    ChordInterpretation,
    PcSet,
    ScaleCandidate,
    Segment,
    Voicing,
)
from . import speller
from .vocabulary import QUALITIES, SCALES, TENSION_DEGREE, scale_prior


class HarmonyKernel:
    """Enumerate every deterministic reading of a pitch-class collection."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.max_chords = int(self.config.get("harmony.max_chord_interpretations", 5))
        self.max_scales = int(self.config.get("harmony.max_scale_candidates", 5))
        self.detect_quartal = bool(self.config.get("harmony.detect_quartal", True))
        self.detect_ust = bool(self.config.get("harmony.detect_upper_structure", True))

    # -- public ------------------------------------------------------------- #
    def enumerate_segment(self, obs: PitchClassObservation) -> Segment:
        pcs = sorted({p % 12 for p in obs.pitch_classes})
        weights = self._normalize_weights(pcs, obs)

        seg = Segment(
            start_s=obs.span.start_s,
            end_s=obs.span.end_s,
            bar=obs.span.bar,
            pitch_classes=pcs,
            coarse_acr=obs.coarse_acr,
        )
        if not pcs:
            return seg

        seg.pc_set = self._pc_set(pcs)
        seg.chord_interpretations = self._enumerate_chords(pcs, weights)
        seg.scale_candidates = self._enumerate_scales(pcs, seg.chord_interpretations)
        seg.voicing = self._voicing(pcs, seg.chord_interpretations)
        seg.spelled_candidates = self._spelled(pcs, seg.chord_interpretations)
        return seg

    # -- pc-set (music21) --------------------------------------------------- #
    def _pc_set(self, pcs: list[int]) -> PcSet:
        from music21 import chord as m21chord

        c = m21chord.Chord(pcs)
        try:
            forte = c.forteClass
        except Exception:  # pragma: no cover - defensive
            forte = ""
        iv = "".join(str(x) if x < 10 else f"({x})" for x in c.intervalVector)
        return PcSet(
            prime_form="[" + " ".join(str(p) for p in c.primeForm) + "]",
            forte=forte or "",
            interval_vector="<" + iv + ">",
            normal_order="[" + " ".join(str(p) for p in c.normalOrder) + "]",
        )

    # -- chord enumeration -------------------------------------------------- #
    def _enumerate_chords(
        self, pcs: list[int], weights: dict[int, float]
    ) -> list[ChordInterpretation]:
        present = set(pcs)
        scored: list[tuple[float, ChordInterpretation]] = []

        for root in range(12):
            if root not in present:
                continue  # a chord root should sound in the collection
            intervals_present = {(pc - root) % 12 for pc in present}
            for q in QUALITIES:
                q_iv = set(q.intervals)
                inter = q_iv & intervals_present
                union = q_iv | intervals_present
                jaccard = len(inter) / len(union)
                # weighted coverage of the quality's own tones (rewards a match
                # whose chord tones actually carry salience).
                covered_w = sum(
                    weights.get((root + iv) % 12, 0.0) for iv in inter
                )
                if jaccard < 0.34:
                    continue
                root_w = weights.get(root, 0.0)
                score = 0.72 * jaccard + 0.18 * covered_w + 0.10 * root_w

                chord_tone_pcs = sorted((root + iv) % 12 for iv in inter)
                tension_pcs = sorted(
                    pc for pc in present if (pc - root) % 12 not in q_iv
                )
                ci = ChordInterpretation(
                    name=f"{speller.spell(root, 0, 1)}{q.name}",
                    root=speller.spell(root, 0, 1),
                    quality=q.name,
                    score=round(min(score, 1.0), 4),
                    chord_tones=chord_tone_pcs,
                    tensions=tension_pcs,
                )
                scored.append((score, ci))

        scored.sort(key=lambda t: (-t[0], t[1].name))
        # de-dup by chord name, keep best
        seen: set[str] = set()
        out: list[ChordInterpretation] = []
        for _, ci in scored:
            if ci.name in seen:
                continue
            seen.add(ci.name)
            out.append(ci)
            if len(out) >= self.max_chords:
                break
        return out

    # -- scale enumeration -------------------------------------------------- #
    def _enumerate_scales(
        self, pcs: list[int], chords: list[ChordInterpretation]
    ) -> list[ScaleCandidate]:
        present = set(pcs)
        # Prefer scale roots that match a top chord root.
        chord_root_pcs = {self._name_to_pc(c.root) for c in chords[:3]}

        scored: list[tuple[float, ScaleCandidate]] = []
        for tonic in range(12):
            for st in SCALES:
                scale_pcs = {(tonic + iv) % 12 for iv in st.intervals}
                inside = present & scale_pcs
                fit = len(inside) / len(present) if present else 0.0
                if fit < 0.6:
                    continue
                out_of = sorted(present - scale_pcs)
                prior = scale_prior(st.name)
                root_bonus = 0.1 if tonic in chord_root_pcs else 0.0
                score = fit * prior + root_bonus
                name = f"{speller.spell(tonic, 0, 1)} {st.name}"
                scored.append(
                    (score, ScaleCandidate(name=name, fit=round(fit, 4), out_of_scale=out_of))
                )

        scored.sort(key=lambda t: (-t[0], t[1].name))
        seen: set[str] = set()
        out: list[ScaleCandidate] = []
        for _, sc in scored:
            if sc.name in seen:
                continue
            seen.add(sc.name)
            out.append(sc)
            if len(out) >= self.max_scales:
                break
        return out

    # -- voicing ------------------------------------------------------------ #
    def _voicing(self, pcs: list[int], chords: list[ChordInterpretation]) -> Voicing:
        stacking = self._stacking(pcs, chords)
        upper = self._upper_structure(pcs, chords) if self.detect_ust else None
        omitted = self._omitted(pcs, chords)
        return Voicing(stacking=stacking, upper_structure=upper, omitted=omitted, inversion="")

    def _stacking(self, pcs: list[int], chords: list[ChordInterpretation]) -> str:
        if self.detect_quartal and self._is_quartal(pcs):
            return "quartal"
        # A dense chromatic cluster wins over a weak tertian match: several
        # semitone adjacencies is unambiguous voicing evidence.
        if self._is_cluster(pcs):
            return "cluster"
        if chords and chords[0].score >= 0.5 and len(pcs) >= 3:
            return "tertian"
        return "unknown"

    @staticmethod
    def _is_quartal(pcs: list[int]) -> bool:
        n = len(pcs)
        if n < 3:
            return False
        target = set(pcs)
        for start in pcs:
            chain = {(start + 5 * k) % 12 for k in range(n)}
            if chain == target:
                return True
        return False

    @staticmethod
    def _is_cluster(pcs: list[int]) -> bool:
        if len(pcs) < 3:
            return False
        s = sorted(pcs)
        semitone_adjacencies = sum(
            1 for i in range(len(s)) if ((s[(i + 1) % len(s)] - s[i]) % 12) == 1
        )
        return semitone_adjacencies >= max(2, len(pcs) // 2)

    def _upper_structure(
        self, pcs: list[int], chords: list[ChordInterpretation]
    ) -> str | None:
        if not chords:
            return None
        present = set(pcs)
        primary_root = self._name_to_pc(chords[0].root)
        best: tuple[int, str] | None = None  # (tension_rank, label)
        for troot in present:
            if troot == primary_root:
                continue
            for iv, qual in (({0, 4, 7}, "major"), ({0, 3, 7}, "minor")):
                triad = {(troot + x) % 12 for x in iv}
                if triad <= present:
                    interval = (troot - primary_root) % 12
                    # Prefer more "colorful" upper structures (tritone, m2, etc.).
                    rank = {6: 5, 2: 4, 8: 4, 1: 3, 9: 3}.get(interval, 1)
                    label = f"{speller.spell(troot, 0, 1)} {qual} triad"
                    if best is None or rank > best[0]:
                        best = (rank, label)
        return best[1] if best else None

    def _omitted(self, pcs: list[int], chords: list[ChordInterpretation]) -> list[str]:
        if not chords:
            return []
        top = chords[0]
        root = self._name_to_pc(top.root)
        present = set(pcs)
        q = next((x for x in QUALITIES if x.name == top.quality), None)
        if q is None:
            return []
        missing = []
        for iv, deg in q.degrees.items():
            if (root + iv) % 12 not in present:
                missing.append(speller.spell(root, iv, deg))
        return missing

    # -- spelling ----------------------------------------------------------- #
    def _spelled(self, pcs: list[int], chords: list[ChordInterpretation]) -> list[str]:
        """One spelled-pitch string per top chord reading (+ a neutral one)."""
        out: list[str] = []
        for ci in chords[:3]:
            root = self._name_to_pc(ci.root)
            q = next((x for x in QUALITIES if x.name == ci.quality), None)
            degrees = dict(q.degrees) if q else {}
            names: list[str] = []
            # Spell from the root upward (root, then ascending intervals) so the
            # string reads like a chord rather than a raw pc list.
            for pc in sorted(pcs, key=lambda p: (p - root) % 12):
                iv = (pc - root) % 12
                degree = degrees.get(iv, TENSION_DEGREE[iv])
                names.append(speller.spell(root, iv, degree))
            out.append(" ".join(names))
        if not out:
            out.append(" ".join(speller.spell_pitch_classes(pcs)))
        # de-dup preserving order
        seen: set[str] = set()
        uniq = []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _name_to_pc(name: str) -> int:
        base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        pc = base[name[0]]
        for ch in name[1:]:
            if ch == "#":
                pc += 1
            elif ch == "b":
                pc -= 1
        return pc % 12

    @staticmethod
    def _normalize_weights(pcs: list[int], obs: PitchClassObservation) -> dict[int, float]:
        if obs.weights and len(obs.weights) == len(obs.pitch_classes):
            raw = {p % 12: w for p, w in zip(obs.pitch_classes, obs.weights)}
        else:
            raw = {p: 1.0 for p in pcs}
        total = sum(raw.values()) or 1.0
        return {p: w / total for p, w in raw.items()}
