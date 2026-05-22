"""Transcoder adapter — wraps sae_lens Transcoder as an ArtifactAdapter."""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class TranscoderAdapter(ArtifactAdapter):
    artifact_type = "transcoder"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._transcoder = None

    def load(self) -> None:
        from sae_lens import SAE

        sae, _, _ = SAE.from_pretrained(
            release=self.manifest.construction.get("release", ""),
            sae_id=self.manifest.construction.get("sae_id", ""),
        )
        self._transcoder = sae

    def directions(self, layer: int | None = None):
        if self._transcoder is None:
            self.load()
        return self._transcoder.W_dec.detach()

    def activations(self, model, tokens, hook_name: str):
        if self._transcoder is None:
            self.load()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            return self._transcoder.encode(acts)

    def ablate(self, model, tokens, units: list[int], site: str):
        if self._transcoder is None:
            self.load()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            acts = cache[site]
            feature_acts = self._transcoder.encode(acts)
            feature_acts[:, :, units] = 0.0
            return self._transcoder.decode(feature_acts)

    @property
    def d_out(self) -> int | None:
        if self._transcoder is None:
            return None
        return self._transcoder.cfg.d_out

    def metadata(self) -> dict:
        base = super().metadata()
        if self._transcoder is not None:
            base["d_out"] = self._transcoder.cfg.d_out
        return base

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str = "", hook_point: str = "", d_out: int | None = None,
    ) -> TranscoderAdapter:
        manifest = ArtifactManifest(
            artifact_type="transcoder",
            target_model="gpt2",
            hook_point=hook_point,
            d_in=768,
            construction={"release": release, "sae_id": sae_id},
        )
        adapter = cls(manifest)
        return adapter
