"""Pipeline orchestrator.

Wires the stages together through the :mod:`fakebook.contracts` Protocols, so any
stage can be swapped (real ↔ fallback ↔ stub) without changing this file. The
data flow mirrors the architecture in the spec:

    ingest → separation → mir ─┬─ harmony (salience → kernel) → assemble → interpret
                               └─ timbre (Branch B stub)

Two entry points:

* :meth:`analyze_file` — full audio path (needs the relevant extras).
* :meth:`analyze_clip` — start from an already-loaded :class:`AudioClip`.

Stages are injected in the constructor; :meth:`from_config` builds the default
set with graceful fallbacks. The harmony kernel and assembler always run; audio
stages degrade to fallbacks when their deps are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .audio import AudioClip
from .config import Config
from .contracts import (
    HarmonyKernel as HarmonyKernelProto,
)
from .contracts import (
    Ingestor,
    Interpreter,
    MirAnalyzer,
    SalienceExtractor,
    Separator,
    TimbreAnalyzer,
)
from .harmony.segmentation import build_spans
from .schema import AnalysisDocument, Meta


@dataclass
class Pipeline:
    config: Config
    ingestor: Ingestor
    separator: Separator
    mir: MirAnalyzer
    salience: SalienceExtractor
    kernel: HarmonyKernelProto
    assembler: "object"  # fakebook.assemble.Assembler
    timbre: Optional[TimbreAnalyzer] = None
    interpreter: Optional[Interpreter] = None

    # -- factory ------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Config | None = None) -> "Pipeline":
        config = config or Config.default()
        from .assemble import Assembler
        from .harmony import HarmonyKernel, build_salience
        from .ingest import AudioLoader
        from .mir import build_mir_analyzer
        from .separation import build_separator
        from .timbre import TimbreStub

        timbre = TimbreStub(config) if config.get("timbre.enabled", False) else None
        interpreter = None
        return cls(
            config=config,
            ingestor=AudioLoader(config),
            separator=build_separator(config),
            mir=build_mir_analyzer(config),
            salience=build_salience(config),
            kernel=HarmonyKernel(config),
            assembler=Assembler(config),
            timbre=timbre,
            interpreter=interpreter,
        )

    # -- entry points ------------------------------------------------------- #
    def analyze_file(self, path: str, interpret: bool = False) -> AnalysisDocument:
        clip = self.ingestor.load(path)
        return self.analyze_clip(clip, interpret=interpret)

    def analyze_clip(self, clip: AudioClip, interpret: bool = False) -> AnalysisDocument:
        stems = self.separator.separate(clip)
        mir_result = self.mir.analyze(clip, stems)

        spans = build_spans(mir_result.beats, clip.duration_s, self.config)
        observations = self.salience.observe(stems.harmonic, spans)

        segments = []
        for obs in observations:
            seg = self.kernel.enumerate_segment(obs)
            if self.timbre is not None:
                seg.timbre = self.timbre.describe(stems.harmonic, obs.span)
            segments.append(seg)

        meta = Meta(
            duration_s=round(clip.duration_s, 3),
            sample_rate=clip.sample_rate,
            source=clip.source,
        )
        doc = self.assembler.assemble(meta, mir_result, segments)

        if interpret:
            doc.interpretation = self._interpret(doc)
        return doc

    def _interpret(self, doc: AnalysisDocument):
        interpreter = self.interpreter
        if interpreter is None:
            from .interpret import AnthropicInterpreter

            interpreter = AnthropicInterpreter(self.config)
        payload = doc.model_dump(by_alias=True, mode="json")
        return interpreter.interpret(payload)
