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


class CorpusUnavailableError(FakebookError):
    """A reference corpus could not be fetched or read.

    Carries the hosts a download needs, because the usual cause in a sandboxed
    environment is an egress policy denying one of them — which is fixable, but
    only if the message says which host to allow.
    """

    def __init__(self, corpus: str, hosts: tuple[str, ...], reason: str = ""):
        self.corpus = corpus
        self.hosts = hosts
        self.reason = reason
        host_list = ", ".join(hosts) if hosts else "its download hosts"
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"corpus '{corpus}' is not available{detail}. "
            f"It downloads from: {host_list}. If the network denies those hosts, "
            f"allow them for this environment; if it is already downloaded, point "
            f"FAKEBOOK_CORPUS_DIR at the cache directory."
        )
