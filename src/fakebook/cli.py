"""Command-line interface.

Subcommands:

* ``fakebook analyze AUDIO [-o out.json] [--interpret]`` — run the full pipeline.
* ``fakebook kernel PC [PC ...]`` — run just the deterministic kernel on a
  pitch-class list (e.g. ``fakebook kernel 0 4 7 10 3`` for a Hendrix chord).
  Needs no audio deps — handy for inspecting the guardrail enumeration.
* ``fakebook schema`` — print the IR JSON Schema.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .assemble import Assembler
    from .pipeline import Pipeline

    # Config.load handles a None path: packaged defaults + FAKEBOOK_* env overrides.
    config = Config.load(args.config)
    pipeline = Pipeline.from_config(config)
    doc = pipeline.analyze_file(args.audio, interpret=args.interpret)

    out = Assembler.to_json(doc)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out)
        report = pipeline.assembler.last_report  # type: ignore[attr-defined]
        print(f"wrote {args.output} ({len(doc.segments)} segments)", file=sys.stderr)
        if report and report.warnings:
            print(f"coherence warnings: {len(report.warnings)}", file=sys.stderr)
            for w in report.warnings:
                print(f"  - {w}", file=sys.stderr)
    else:
        print(out)
    return 0


def _cmd_kernel(args: argparse.Namespace) -> int:
    from .contracts import ChordSpan, PitchClassObservation
    from .harmony import HarmonyKernel

    # Config.load handles a None path: packaged defaults + FAKEBOOK_* env overrides.
    config = Config.load(args.config)
    kernel = HarmonyKernel(config)
    obs = PitchClassObservation(
        span=ChordSpan(start_s=0.0, end_s=float(args.duration), bar=1),
        pitch_classes=[int(p) for p in args.pitch_classes],
    )
    seg = kernel.enumerate_segment(obs)
    print(json.dumps(seg.model_dump(by_alias=True, mode="json"), indent=2))
    return 0


def _cmd_schema(_args: argparse.Namespace) -> int:
    from .schema import export_json_schema

    print(json.dumps(export_json_schema(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fakebook", description=__doc__)
    p.add_argument("--config", help="path to a YAML config override")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="run the full pipeline on an audio file")
    a.add_argument("audio", help="path to an audio file")
    a.add_argument("-o", "--output", help="write JSON here instead of stdout")
    a.add_argument("--interpret", action="store_true", help="run the LLM interpret stage")
    a.set_defaults(func=_cmd_analyze)

    k = sub.add_parser("kernel", help="run the deterministic kernel on pitch classes")
    k.add_argument("pitch_classes", nargs="+", help="pitch classes 0-11, e.g. 0 4 7 10 3")
    k.add_argument("--duration", default=2.0, type=float, help="span duration seconds")
    k.set_defaults(func=_cmd_kernel)

    s = sub.add_parser("schema", help="print the IR JSON Schema")
    s.set_defaults(func=_cmd_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
