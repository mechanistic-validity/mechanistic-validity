"""FF Key-Value baseline adapter — raw W_in rows as a feature dictionary.

Ye et al. (NeurIPS 2025) showed that raw feed-forward key-value memories
are nearly as interpretable as SAE features on standard benchmarks. This
adapter wraps W_in rows as directions (no training involved), providing
a stronger null hypothesis than random directions.
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactManifest, WeightArtifactAdapter


class FFKeyValueAdapter(WeightArtifactAdapter):
    artifact_type = "ff_key_value"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._W_in = None  # (d_mlp, d_model)

    def _get_directions(self) -> torch.Tensor:
        if self._W_in is None:
            self.load()
        return self._W_in

    def load(self) -> None:
        path = self.manifest.construction.get("path", "")
        if not path:
            raise ValueError("FFKeyValueAdapter requires a 'path' in construction config")
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._W_in = data["W_in"]

    def metadata(self) -> dict:
        base = super().metadata()
        if self._W_in is not None:
            base["d_mlp"] = self._W_in.shape[0]
            base["d_model"] = self._W_in.shape[1]
        return base

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        layer: int = 0,
        hook_point: str = "",
    ) -> FFKeyValueAdapter:
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(model_name)
        W_in = model.blocks[layer].mlp.W_in.detach().T  # (d_mlp, d_model)
        if not hook_point:
            hook_point = f"blocks.{layer}.hook_mlp_out"
        return cls.from_weights(W_in, target_model=model_name, hook_point=hook_point)

    @classmethod
    def from_weights(
        cls,
        W_in: torch.Tensor,
        target_model: str = "gpt2",
        hook_point: str = "",
    ) -> FFKeyValueAdapter:
        d_mlp, d_model = W_in.shape
        manifest = ArtifactManifest(
            artifact_type="ff_key_value",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_model,
            d_sae=d_mlp,
        )
        adapter = cls(manifest)
        adapter._W_in = W_in
        return adapter

    @classmethod
    def from_model(
        cls,
        model,
        layer: int,
        hook_point: str = "",
    ) -> FFKeyValueAdapter:
        W_in = model.blocks[layer].mlp.W_in.detach().T  # (d_mlp, d_model)
        model_name = getattr(model.cfg, "model_name", "unknown")
        if not hook_point:
            hook_point = f"blocks.{layer}.hook_mlp_out"
        return cls.from_weights(W_in, target_model=model_name, hook_point=hook_point)
