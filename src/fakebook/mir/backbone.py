"""Self-supervised music foundation-model backbone (MERT and friends).

The spec prefers an audio-native SSL backbone (MERT / MusicFM / MuQ / OMAR-RQ)
probed with light heads, because these are genre-robust (trained on electronic
music too) where hand-built DSP features are brittle.

What is honest to implement here without shipping trained probe weights:

* **embedding extraction** — run the backbone and return per-layer hidden states.
  These are the inputs a linear probe head would consume for key / tempo /
  structure. Real heads are ``nn.Linear`` layers fit on labelled data; we expose
  the embeddings and a clean seam (:meth:`probe`) to attach them.

Per-task *predictions* (key/tempo/beat) are left to the librosa fallback in
:mod:`.analyzer` until probe weights exist, so the pipeline never fabricates a
backbone prediction it cannot actually make. Tag/mood zero-shot is likewise left
as a documented seam rather than a fake output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._optional import require
from ..audio import AudioClip
from ..config import Config


@dataclass
class BackboneEmbedding:
    """Per-layer hidden states from the backbone, mean-pooled over time.

    ``layers`` is shape ``(n_layers, hidden)``. A weighted sum over layers (the
    usual MERT probing recipe) is the input to a downstream head.
    """

    layers: np.ndarray
    frame_rate_hz: float
    model_name: str


class MertBackbone:
    """Wraps a HuggingFace MERT-style model for embedding extraction."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.default()
        self.model_name = self.config.get("mir.backbone_model", "m-a-p/MERT-v1-95M")
        self.target_sr = int(self.config.get("mir.backbone_sample_rate", 24000))
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        transformers = require("transformers", feature="MERT backbone", extra="mir")
        torch = require("torch", feature="MERT backbone", extra="mir")
        self._torch = torch
        self._model = transformers.AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).eval()
        self._processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

    def embed(self, clip: AudioClip) -> BackboneEmbedding:
        """Extract mean-pooled per-layer embeddings for a clip."""
        self._lazy_load()
        torch = self._torch
        mono = clip.to_mono()
        wav = mono.samples[0]
        if clip.sample_rate != self.target_sr:
            librosa = require("librosa", feature="resampling", extra="audio")
            wav = librosa.resample(wav, orig_sr=clip.sample_rate, target_sr=self.target_sr)

        inputs = self._processor(wav, sampling_rate=self.target_sr, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # (n_layers, time, hidden) → mean over time → (n_layers, hidden)
        hidden = torch.stack(out.hidden_states).squeeze(1)
        pooled = hidden.mean(dim=1).cpu().numpy()
        # MERT-v1-95M runs at ~75 Hz frame rate.
        return BackboneEmbedding(pooled, frame_rate_hz=75.0, model_name=self.model_name)

    def probe(self, embedding: BackboneEmbedding, head) -> np.ndarray:
        """Apply a trained probe ``head`` (callable) to a layer-weighted embedding.

        Seam for attaching fitted heads. ``head`` maps ``(hidden,) -> np.ndarray``.
        """
        # Uniform layer weighting by default; a fitted recipe learns these too.
        pooled = embedding.layers.mean(axis=0)
        return head(pooled)
