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
  layering), a spectral/timbre object rather than a chord. **Built**, over the
  same chord spans, into the `timbre` slot every segment carries. It needs no
  optional extras — the whole descriptor is `np.fft` over audio the pipeline
  already has in memory.

Both branches run side by side on the same spans, so a segment reads as
*"Cm11, supersaw-wide unison, filter opening"* rather than forcing a choice
between the two senses of "complex voicing".

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
        └─ timbre/    [Branch B] STFT → centroid/bandwidth/flatness, partial
                       tracking → detuning · inharmonicity · layering · filter
                       movement, plus thresholded sound-design tags
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
| `corpus`   | reference-corpus evaluation (validation only)       | mirdata, mir_eval                |
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

### Branch B: what a segment's `timbre` block looks like

Every segment carries a measured sound-design descriptor beside its harmonic
reading (a 7-voice supersaw, detuned ±25 cents, under an opening filter):

```jsonc
{
  "spectral_centroid_hz": 1517.9,   // brightness
  "spectral_bandwidth_hz": 1242.7,  // spread about the centroid
  "spectral_flatness": 0.0,         // ~0 tonal, ->1 noise-like
  "centroid_slope_hz_per_s": 829.9, // filter movement: the cutoff opening
  "inharmonicity": 0.0129,          // deviation from the dominant harmonic series
  "harmonic_ratio": 0.64,           // how much of the energy that one series explains
  "detune_cents": 48.4,             // unison spread (true spread: 50)
  "n_partials": 47,                 // resolved partials — layering density
  "f0_hz": 445.6,
  "tags": ["detuned unison (supersaw-like)", "filter opening", "dense/layered"]
}
```

`tags` are assigned here, by fixed thresholds — the LLM may explain a tag, never
assign one. `detune_cents` is evidence of a unison rather than a calibrated
figure: unresolved voices make it under-read, and on a dense chord another note's
partial can widen — or break up — a cluster. The limit that matters most is
stated rather than papered over:
`inharmonicity` is a *single-source* number, so a dense chord of perfectly
harmonic oscillators raises it too. That is why the tag is the neutral
`off-series partials` and why `harmonic_ratio` ships beside it — and why the
disambiguation happens at interpretation time, where the *other* branch is
visible: off-series partials over five pitch classes is polyphony, the same tag
over one or two is a genuinely inharmonic patch.

### Score the pipeline against a reference corpus

```bash
fakebook evaluate list                                   # what is registered
fakebook evaluate giantsteps_key --download --limit 50 -o report.json
```

The report shape (values shown as placeholders — **no corpus has been scored
yet**, see *Validation* below):

```
corpus: giantsteps_key
analyzed: N  missing audio: N  errors: N
key: exact NN.N%, MIREX weighted N.NNN (n=N)
     relationships: correct N, fifth N, parallel N, relative N, other N
tempo: acc1 NN.N%, acc2 NN.N% (n=N)
```

Registered corpora are chosen for **genre match** first — the target is mixed,
computer-produced pop/dance, so 604 annotated Beatport/EDM previews beat a larger
corpus of rock or classical:

| Corpus           | Material                                        | Reference data      |
|------------------|-------------------------------------------------|---------------------|
| `giantsteps_key` | 604 × 2-min Beatport previews (EDM)             | key (expert), tempo + genre (store metadata) |
| `beatport_key`   | 1486 × 2-min Beatport previews (EDM)            | key (multiple annotators), tempo + genre |

Loading, checksums and archive layout are delegated to
[`mirdata`](https://mirdata.readthedocs.io); [`corpus/`](src/fakebook/corpus)
adds the registry, MIREX-comparable metrics, and the runner. Audio is **never
vendored** — it caches under `FAKEBOOK_CORPUS_DIR` (default
`~/.cache/fakebook/corpus`). Reports are per-track first and aggregate second,
with the genre label on every row, because the useful question is never the mean
but *which* tracks failed and what they share.

Two deliberate choices in the metrics: key keeps the **relationship**
(correct / fifth / relative / parallel / other), since mistaking a key for its
relative minor is a different bug from a tritone error; and tempo reports
Accuracy1/Accuracy2 computed from their definitions rather than through
`mir_eval.tempo.detection`, which expects perceptual tempo *pairs* that this
ground truth does not have.

A corpus download needs network access to its hosts (Zenodo, for both of the
above). If that is denied, the harness raises `CorpusUnavailableError` naming the
host to allow rather than failing deep inside a zip reader.

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
- **Not yet measured** — no corpus run has happened. The harness below is built
  and tested, but the environment it was written in denies the download host, so
  there are no key/tempo numbers for this pipeline yet. They are one
  `fakebook evaluate giantsteps_key --download` away on a machine that can reach
  Zenodo; until then, treat the audio heads as unquantified.

- **Reference corpora** — `fakebook evaluate` scores the real pipeline against
  annotated EDM corpora (see above): MIREX key score with the error
  *relationship* broken out, tempo Accuracy1/Accuracy2, and beat F-measure where
  beat annotations exist. The harness itself is tested without any download — a
  stub pipeline over fabricated tracks pins the aggregation, the per-track error
  containment (one corrupt file must not end a 604-track run), and the
  missing-audio accounting.

- **Branch B descriptors** — `tests/test_timbre.py` synthesizes patches whose
  ground truth is known by construction (an additive saw is exactly harmonic; a
  7-voice unison is detuned by exactly N cents; FM on an irrational ratio is
  genuinely inharmonic) and checks both the recovered value and the separation
  between patches: the detune estimate lands within a few cents of the true
  spread and orders correctly with it, a saw reads harmonic to 1e-5 while FM
  reads three orders higher, sweep direction is recovered, and the false
  positives that cost the most are pinned shut — noise is not a unison, FM
  sidebands are not a unison, unison beating is not a filter sweep, and
  polyphony is not an FM claim.

- **Audio integration** — `tests/test_integration_audio.py` renders known chords
  to a signal and drives the *real* front-end (librosa CQT chroma → kernel, the
  MIR key head, and a full `analyze_file` from a written WAV), checking that the
  pitch content is actually recovered, that it survives heavy broadband noise,
  and pinning known chroma limitations (a bright single tone leaks its fifth).
  These need the `audio` extra and skip cleanly without it.

The LLM layer's output is treated as interpretation, not fact; it inherits, and
must not exceed, the certainty of the kernel.

```bash
pytest                      # core suite — no heavy deps required
pip install -e '.[audio]' && pytest   # + real-audio integration tests
```

---

## Build status

| Phase | Scope                                             | Status |
|-------|---------------------------------------------------|--------|
| 0     | scaffold, config, IR schema, typed contracts      | ✅ |
| 1     | `ingest/` + `separation/`                         | ✅ (demucs; HPSS + passthrough fallbacks) |
| 2     | `mir/` heads → global/beats/sections              | ✅ (librosa heads + MERT backbone seam) |
| 3     | `harmony/` kernel (Branch A)                      | ✅ (fully tested) |
| 4     | `assemble/` → validated JSON                      | ✅ |
| 5     | `interpret/` LLM guardrail layer                  | ✅ |
| 6     | `timbre/` (Branch B)                              | ✅ (numpy-only spectral descriptors) |

Stages that wrap heavy models (demucs separation, MERT probing, basic-pitch
salience) are written against their real APIs behind the contract interfaces;
running them end-to-end needs the corresponding extra + weights. The MERT
backbone exposes embedding extraction and a `probe()` seam but defers per-task
*predictions* to the librosa heads rather than fabricating outputs it has no
trained head for.
