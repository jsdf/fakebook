"""Helper for lazy optional imports.

Heavy dependencies (torch, librosa, demucs, anthropic, ...) are imported through
:func:`require` at call time. This keeps ``import fakebook`` — and the whole
deterministic kernel — working on a minimal install, and turns a missing extra
into a clear, actionable :class:`MissingDependencyError` instead of an
``ImportError`` deep in a stage.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import MissingDependencyError


def require(module: str, *, feature: str, extra: str, package: str | None = None) -> ModuleType:
    """Import ``module`` or raise a helpful :class:`MissingDependencyError`.

    Parameters
    ----------
    module:  the importable module name, e.g. ``"librosa"`` or ``"torch"``.
    feature: human-readable name of the capability, for the error message.
    extra:   the pip extra that provides it, e.g. ``"audio"``.
    package: pip package name if it differs from ``module`` (defaults to the
             top-level of ``module``).
    """

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only w/o extras
        pkg = package or module.split(".")[0]
        raise MissingDependencyError(feature=feature, extra=extra, package=pkg) from exc


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported."""
    return importlib.util.find_spec(module) is not None
