"""Configuration loading.

A single :class:`Config` wraps a nested dict of settings. Precedence, lowest to
highest:

1. the packaged ``config/default.yaml``
2. an optional user YAML passed to :meth:`Config.load`
3. ``FAKEBOOK_<SECTION>__<KEY>`` environment variables (double underscore nests)

The object is deliberately a thin dotted-access dict rather than a rigid model:
stage configs evolve independently and each stage reads only the subtree it
cares about.
"""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import yaml

_ENV_PREFIX = "FAKEBOOK_"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for env-var overrides."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _env_overrides() -> dict:
    """Turn ``FAKEBOOK_SECTION__KEY=val`` env vars into a nested dict."""
    out: dict = {}
    for name, raw in os.environ.items():
        if not name.startswith(_ENV_PREFIX):
            continue
        path = name[len(_ENV_PREFIX) :].lower().split("__")
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw)
    return out


class Config:
    """Dotted-path access over the merged configuration dict."""

    def __init__(self, data: dict):
        self._data = data

    # -- construction ------------------------------------------------------- #
    @classmethod
    def default(cls) -> "Config":
        text = resources.files("fakebook").joinpath("data/default.yaml").read_text()
        return cls(yaml.safe_load(text) or {})

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        data = cls.default()._data
        if path is not None:
            user = yaml.safe_load(Path(path).read_text()) or {}
            data = _deep_merge(data, user)
        env = _env_overrides()
        if env:
            data = _deep_merge(data, env)
        return cls(data)

    # -- access ------------------------------------------------------------- #
    def get(self, dotted: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def section(self, name: str) -> dict:
        value = self._data.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config(sections={sorted(self._data)})"
