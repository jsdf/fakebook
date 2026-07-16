"""Ingest stage: load, resample, segment audio into an :class:`AudioClip`."""

from .loader import AudioLoader

__all__ = ["AudioLoader"]
