# fakebook

A musical-analysis pipeline for **mixed, computer-produced pop/dance music that
does not necessarily use classic instruments**. It goes from an audio clip to a
structured musical description (key, tempo, structure, and — the point — rich
chord *voicings*) plus an interpretive layer that relates those voicings to
chords, scales, and idioms.

The name is deliberate: a *fake book* is a collection of chord charts, and rich
voicing detail is exactly what a coarse chart flattens away.

---

## Why it is built this way

Five design principles drive the architecture. Several are counterintuitive and
were chosen on purpose:

1. **No note-level transcription as the backbone.** For synth/sample music the
   "note" abstraction breaks (808 glides, supersaws, evolving pads). We use
   direct-from-audio representations (chroma / multipitch salience), not MIDI.
2. **The deterministic kernel enumerates; the LLM interprets.** All pitch /
   interval / chord / scale arithmetic is computed by a rule engine
   ([`music21`](https://web.mit.edu/music21/) + a local vocabulary). The LLM
   receives verified facts and the *enumerated space of valid readings* and
   produces interpretation only — it never computes pitches. This is a hard
   guardrail, enforced both in the prompt and by a post-hoc scrub.
3. **Coarse chord recognition is not enough for voicings.** Off-the-shelf ACR
   uses a ~maj/min vocabulary and flattens a rich voicing to "F minor". For
   voicing detail we go **chroma / salience → theory kernel**, not ACR. ACR is
   kept only as clearly-labelled lossy context.
4. **Spelled pitch is carried alongside pitch-class sets.** pc-sets collapse
   enharmonics; we need the ♯9-vs-♭3 distinction, so both representations flow
   through the harmony stage.
5. **Stems are separated before harmonic analysis.** Dense, sidechained mixes
   smear pitch content, so the harmonic/synth stem is isolated first.

### The fork

"Complex voicing" can mean two things in this genre:

- **Branch A — Harmonic** (extensions, chord-scale color, reharmonization),
  served by the chroma/salience → theory-kernel path. **Built as primary.**
- **Branch B — Sound-design** (detuning, filter movement, FM inharmonicity,
  layering), a spectral/timbre object rather than a chord. **Implemented as a
  stub behind a stable `timbre/` interface** so it can be filled in without
  touching the rest of the pipeline. If your real target is B, flip the
  emphasis — the scaffolding and assembly stages are shared.

---

## Architecture

```
audio clip
  └─ ingest/        load, resample, segment
  └─ separation/    stems: harmonic, bass, drums, vocal
        ├─ mir/       key · tempo · beat/downbeat · structure · tags/mood · coarse chords
        ├─ harmony/   [Branch A] salience/chroma → pitch-class collection
        │              → music21 kernel: pc-set + spelled pitch, chord/scale
        │                enumeration, quartal / upper-structure detection
        └─ timbre/    [Branch B — stub] spectral descriptor (interface only)
  └─ assemble/      merge into one validated JSON document (the IR)
  └─ interpret/     LLM over the grounded IR → interpretation, competing hearings, idioms
```

Every stage implements a typed Protocol in [`contracts.py`](src/fakebook/contracts.py),
so any stage can be swapped (real model ↔ per-task fallback ↔ stub) without
touching the orchestrator. The connective tissue is the **intermediate
representation** in [`schema.py`](src/fakebook/schema.py); its JSON Schema is
shipped at [`schema/ir.schema.json`](schema/ir.schema.json) and is the contract
the LLM consumes.

---

## Install

The core install runs the deterministic kernel, schema/assembly, and the LLM
guardrail logic with **no GPU and no model weights**:

```bash
pip install -e .
```

Heavy capabilities are optional extras, imported lazily so the core always works:

| Extra      | Enables                                             | Key packages                     |
|------------|-----------------------------------------------------|----------------------------------|
| `audio`    | file loading, resampling, source separation, chroma | librosa, soundfile, demucs       |
| `mir`      | MERT/foundation-model backbone, madmom beats        | torch, transformers, madmom      |
| `harmony`  | basic-pitch multipitch salience                     | basic-pitch                      |
| `llm`      | the interpret stage                                 | anthropic                        |
| `all`      | everything above                                    |                                  |

```bash
pip install -e '.[all]'      # full audio → interpretation pipeline
pip install -e '.[dev]'      # tests + linters
```

If an extra is missing, the affected stage raises a clear
`MissingDependencyError` telling you which extra to install; where a graceful
fallback exists (e.g. passthrough separation, null MIR) the pipeline degrades
instead of failing.

---

## Usage

### Inspect the kernel directly (no audio deps)

The deterministic core is usable on its own — hand it pitch classes:

```bash
$ fakebook kernel 0 4 7 10 3        # C E G Bb D#  — the "Hendrix" chord
```

```jsonc
{
  "pitch_classes": [0, 3, 4, 7, 10],
  "pc_set": { "prime_form": "[0 1 4 6 9]", "forte": "5-32B", "interval_vector": "<113221>", ... },
  "chord_interpretations": [
    { "name": "C7#9", "root": "C", "quality": "7#9", "score": 0.92, "chord_tones": [...], "tensions": [...] },
    { "name": "C7",   ... }, { "name": "Cm7", ... }
  ],
  "scale_candidates": [ { "name": "C diminished (h-w)", "fit": 1.0 }, ... ],
  "voicing": { "stacking": "tertian", "upper_structure": "Eb major triad", "omitted": [] },
  "spelled_candidates": ["C D# E G Bb", "C Eb E G Bb"]   // ♯9 vs ♭3 both offered
}
```

### Full pipeline

```bash
fakebook analyze track.wav -o track.analysis.json      # needs [audio]
fakebook analyze track.wav -o track.analysis.json --interpret   # + [llm], ANTHROPIC_API_KEY
fakebook schema                                        # print the IR JSON Schema
```

### From Python

```python
from fakebook import Config
from fakebook.pipeline import Pipeline

pipe = Pipeline.from_config(Config.default())
doc = pipe.analyze_file("track.wav", interpret=False)
print(doc.segments[0].chord_interpretations[0].name)
```

Configuration is layered: packaged defaults → an optional YAML
(`--config my.yaml`) → `FAKEBOOK_SECTION__KEY` environment variables. See
[`data/default.yaml`](src/fakebook/data/default.yaml).

---

## The LLM guardrail

The interpret stage never derives musical facts. Its system prompt restricts it
to *selecting and explaining* among the enumerated `chord_interpretations` /
`scale_candidates`, requires every claim to cite a segment, and requires it to
flag `unsupported: true` rather than invent a reading. After the model responds,
[`interpreter.py`](src/fakebook/interpret/interpreter.py) **re-checks** every
chosen chord name against that segment's enumerated list and scrubs anything not
present — so a hallucinated chord cannot reach the output even if the prompt were
ignored.

---

## Validation

Ground truth is scarce for this material, so validation is layered:

- **Kernel correctness** — music theory is exactly knowable, so the kernel is
  tested against known voicings, spellings (♯9 vs ♭3), pc-set descriptors,
  quartal/cluster/upper-structure detection, and determinism.
- **Coherence checks** (in `assemble/`) — do chord segments align to the
  downbeat grid? Are spans within the clip? Are pc collections non-empty where a
  chord is expected? Findings are advisory warnings attached to the document.
- **Audio heads** — intended checks are key/tempo against known tracks and
  beat-tracking F-measure on annotated clips (require the `audio`/`mir` extras).

The LLM layer's output is treated as interpretation, not fact; it inherits, and
must not exceed, the certainty of the kernel.

```bash
pytest          # 60+ tests, no heavy deps required
```

---

## Build status

| Phase | Scope                                             | Status |
|-------|---------------------------------------------------|--------|
| 0     | scaffold, config, IR schema, typed contracts      | ✅ |
| 1     | `ingest/` + `separation/`                         | ✅ (demucs + passthrough fallback) |
| 2     | `mir/` heads → global/beats/sections              | ✅ (librosa heads + MERT backbone seam) |
| 3     | `harmony/` kernel (Branch A)                      | ✅ (fully tested) |
| 4     | `assemble/` → validated JSON                      | ✅ |
| 5     | `interpret/` LLM guardrail layer                  | ✅ |
| 6     | `timbre/` (Branch B)                              | ◻️ stub behind stable interface |

Stages that wrap heavy models (demucs separation, MERT probing, basic-pitch
salience) are written against their real APIs behind the contract interfaces;
running them end-to-end needs the corresponding extra + weights. The MERT
backbone exposes embedding extraction and a `probe()` seam but defers per-task
*predictions* to the librosa heads rather than fabricating outputs it has no
trained head for.
