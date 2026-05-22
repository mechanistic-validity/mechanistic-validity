"""Crosscoder adapter — wraps a multi-head crosscoder as an ArtifactAdapter.

A crosscoder reads activations from multiple hook points simultaneously,
encodes them into a shared sparse feature space, and decodes back to each
hook point. This adapter handles the multi-head bookkeeping so downstream
metrics see a standard directions/activations/ablate interface per head.
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class CrosscoderAdapter(ArtifactAdapter):
    artifact_type = "crosscoder"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._crosscoder = None
        self._hook_points: list[str] = manifest.construction.get("hook_points", [])

    def load(self) -> None:
        raise NotImplementedError(
            "CrosscoderAdapter.load() requires specifying how to load the "
            "crosscoder. Use CrosscoderAdapter.from_module(module, hook_points) "
            "or implement a loader for your crosscoder format."
        )

    @property
    def hook_points(self) -> list[str]:
        return self._hook_points

    @property
    def n_heads(self) -> int:
        return len(self._hook_points)

    def directions(self, layer: int | None = None):
        """Return decoder directions. Shape: (n_heads, d_sae, d_model) or (d_sae, d_model) for a single head."""
        if self._crosscoder is None:
            self.load()
        W_D = self._crosscoder.W_D.detach()
        if layer is not None and W_D.ndim == 3:
            head_idx = self._head_index_for_layer(layer)
            return W_D[head_idx]
        return W_D

    def activations(self, model, tokens, hook_name: str):
        """Encode activations from all hook points into shared feature space."""
        if self._crosscoder is None:
            self.load()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=self._hook_points)
            stacked = torch.stack([cache[hp] for hp in self._hook_points], dim=-2)
            return self._crosscoder.encode(stacked)

    def ablate(self, model, tokens, units: list[int], site: str):
        """Ablate units and return reconstructed activations for the given site."""
        if self._crosscoder is None:
            self.load()
        head_idx = self._hook_points.index(site) if site in self._hook_points else 0
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=self._hook_points)
            stacked = torch.stack([cache[hp] for hp in self._hook_points], dim=-2)
            feature_acts = self._crosscoder.encode(stacked)
            feature_acts[..., units] = 0.0
            reconstructed = self._crosscoder.decode(feature_acts)
            if reconstructed.ndim >= 3:
                return reconstructed[..., head_idx, :]
            return reconstructed

    def metadata(self) -> dict:
        base = super().metadata()
        base["hook_points"] = self._hook_points
        base["n_heads"] = self.n_heads
        return base

    def _head_index_for_layer(self, layer: int) -> int:
        for i, hp in enumerate(self._hook_points):
            if f".{layer}." in hp or hp.endswith(f".{layer}"):
                return i
        return 0

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        hook_points: list[str],
        target_model: str = "gpt2",
        d_in: int = 768,
    ) -> CrosscoderAdapter:
        manifest = ArtifactManifest(
            artifact_type="crosscoder",
            target_model=target_model,
            hook_point=hook_points[0],
            d_in=d_in,
            construction={"hook_points": hook_points},
        )
        adapter = cls(manifest)
        adapter._crosscoder = module
        adapter._hook_points = hook_points
        return adapter
