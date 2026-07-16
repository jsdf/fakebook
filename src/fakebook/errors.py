"""Exception hierarchy for the pipeline."""

from __future__ import annotations


class FakebookError(Exception):
    """Base class for all fakebook errors."""


class MissingDependencyError(FakebookError):
    """A stage needs an optional extra that is not installed.

    Raised lazily (at call time, not import time) so the deterministic kernel and
    the schema/contract layer always import cleanly on a light install.
    """

    def __init__(self, feature: str, extra: str, package: str):
        self.feature = feature
        self.extra = extra
        self.package = package
        super().__init__(
            f"{feature} requires the optional '{package}' package. "
            f"Install it with:  pip install 'fakebook[{extra}]'"
        )


class SchemaValidationError(FakebookError):
    """The assembled IR document failed JSON-Schema validation."""


class InterpretationError(FakebookError):
    """The LLM interpretation stage failed or violated a guardrail."""
