"""Source separation → stems.

Principle 5: separate stems *before* harmonic analysis, because dense sidechained
mixes smear pitch content. The harmony stage consumes a dedicated harmonic
("other") stem.

Two implementations behind the :class:`~fakebook.contracts.Separator` protocol:

* :class:`DemucsSeparator` — HTDemucs FT for vocals/drums/bass/other, with an
  optional BS-Roformer vocal model (via ``audio-separator`` / UVR weights) to
  improve the vocal split and, by subtraction, the instrumental. Needs the
  ``audio`` extra + model weights.
* :class:`PassthroughSeparator` — no separation: the whole mix is treated as the
  harmonic stem. Lets the rest of the pipeline run (and be tested) with no heavy
  deps. Clearly lossy; flagged as such.

``build_separator`` picks one from config, falling back to passthrough when the
requested backend's dependencies are unavailable.
"""

from __future__ import annotations

import numpy as np

from .._optional import is_available, require
from ..audio import AudioClip, Stems
from ..config import Config

# HTDemucs FT stem order.
_DEMUCS_STEMS = ("drums", "bass", "other", "vocals")


class PassthroughSeparator:
    """No-op separator: mix → harmonic=mix. Lossy fallback for light installs."""

    lossy = True

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()

    def separate(self, clip: AudioClip) -> Stems:
        return Stems(harmonic=clip, mix=clip)


class DemucsSeparator:
    """HTDemucs FT (+ optional BS-Roformer vocal) separator."""

    lossy = False

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.model_name = self.config.get("separation.demucs_model", "htdemucs_ft")
        self.harmonic_stem = self.config.get("separation.harmonic_stem", "other")
        self.device = self.config.get("separation.device", "auto")
        self.use_roformer_vocal = self.config.get("separation.vocal_model") == "bs_roformer"
        self.ensemble = bool(self.config.get("separation.ensemble", False))

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch = require("torch", feature="separation", extra="audio")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def separate(self, clip: AudioClip) -> Stems:
        torch = require("torch", feature="source separation", extra="audio")
        from demucs.apply import apply_model  # noqa: E402  (lazy)
        from demucs.pretrained import get_model  # noqa: E402

        device = self._resolve_device()
        model = get_model(self.model_name).to(device)
        model.eval()

        # demucs expects (batch, channels, samples) at model.samplerate.
        clip = self._match_rate(clip, model.samplerate)
        wav = torch.from_numpy(clip.samples).unsqueeze(0).to(device)
        if wav.shape[1] == 1:  # demucs wants stereo
            wav = wav.repeat(1, 2, 1)

        with torch.no_grad():
            est = apply_model(model, wav, split=True, overlap=0.25)[0]  # (stems, ch, n)
        sources = {name: est[i].cpu().numpy() for i, name in enumerate(model.sources)}

        harmonic = sources.get(self.harmonic_stem, sources.get("other"))
        stems = Stems(
            harmonic=AudioClip(harmonic, model.samplerate, clip.source),
            bass=self._maybe(sources, "bass", model.samplerate, clip.source),
            drums=self._maybe(sources, "drums", model.samplerate, clip.source),
            vocal=self._maybe(sources, "vocals", model.samplerate, clip.source),
            mix=clip,
        )
        if self.use_roformer_vocal and is_available("audio_separator"):
            stems.vocal = self._roformer_vocal(clip) or stems.vocal
        return stems

    def _roformer_vocal(self, clip: AudioClip) -> AudioClip | None:
        """Refine the vocal split with a BS-Roformer model via audio-separator.

        Returns None if weights/model are unavailable; the demucs vocal is kept.
        """
        try:  # pragma: no cover - requires weights
            from audio_separator.separator import Separator as UVR

            uvr = UVR()
            uvr.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            # audio-separator is file-oriented; a real impl would round-trip via a
            # temp WAV. Left as a targeted TODO to avoid disk churn in the core path.
            return None
        except Exception:
            return None

    @staticmethod
    def _maybe(sources: dict, key: str, sr: int, src: str) -> AudioClip | None:
        arr = sources.get(key)
        return AudioClip(arr, sr, src) if arr is not None else None

    @staticmethod
    def _match_rate(clip: AudioClip, sr: int) -> AudioClip:
        if clip.sample_rate == sr:
            return clip
        librosa = require("librosa", feature="resampling", extra="audio")
        out = librosa.resample(clip.samples, orig_sr=clip.sample_rate, target_sr=sr, axis=1)
        return AudioClip(out.astype(np.float32), sr, clip.source)


def build_separator(config: Config | None = None):
    """Choose a separator from config, falling back to passthrough when needed."""
    config = config or Config.default()
    backend = config.get("separation.backend", "demucs")
    if backend == "passthrough":
        return PassthroughSeparator(config)
    if backend == "demucs" and is_available("demucs") and is_available("torch"):
        return DemucsSeparator(config)
    # Requested a real backend but deps are missing → degrade, don't crash.
    return PassthroughSeparator(config)
