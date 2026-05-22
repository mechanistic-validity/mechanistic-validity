"""ReFT/LoReFT adapter — wraps learned low-rank intervention directions.

ReFT (Representation Finetuning, Wu & Arora et al. NeurIPS 2024) learns a
task-specific low-rank subspace intervention:
  h <- h + W^T (W h + b)
The learned W matrix rows are interpretable feature directions. This is a
weight-only decomposition — no learned encoder — so we inherit from
WeightArtifactAdapter and project activations onto W rows post-hoc.
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactManifest, WeightArtifactAdapter


class ReFTAdapter(WeightArtifactAdapter):
    artifact_type = "reft"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._W: torch.Tensor | None = None  # (rank, d_model)
        self._b: torch.Tensor | None = None  # (rank,)
        self._rank: int | None = None

    def load(self) -> None:
        path = self.manifest.construction.get("path", "")
        if not path:
            raise ValueError("ReFTAdapter requires a 'path' in construction config")
        data = torch.load(path, map_location="cpu", weights_only=True)
        # Support both direct W/b keys and pyreft checkpoint layout
        if "learned_source" in data:
            # pyreft saves the intervention as learned_source (W) and bias
            self._W = data["learned_source"]
            self._b = data.get("bias", torch.zeros(self._W.shape[0]))
        else:
            self._W = data["W"]
            self._b = data.get("b", torch.zeros(self._W.shape[0]))
        self._rank = self._W.shape[0]

    def _get_directions(self) -> torch.Tensor:
        if self._W is None:
            self.load()
        return self._W

    def metadata(self) -> dict:
        base = super().metadata()
        if self._W is not None:
            base["rank"] = self._W.shape[0]
            base["d_model"] = self._W.shape[1]
        if self._b is not None:
            base["has_bias"] = bool((self._b.abs() > 0).any())
        task = self.manifest.construction.get("task")
        if task is not None:
            base["task"] = task
        return base

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        target_model: str = "gpt2",
        hook_point: str = "",
        d_in: int = 768,
        task: str | None = None,
    ) -> ReFTAdapter:
        construction: dict = {"path": path}
        if task is not None:
            construction["task"] = task
        manifest = ArtifactManifest(
            artifact_type="reft",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            construction=construction,
        )
        return cls(manifest)

    @classmethod
    def from_weights(
        cls,
        W: torch.Tensor,
        b: torch.Tensor | None = None,
        target_model: str = "gpt2",
        hook_point: str = "",
        task: str | None = None,
    ) -> ReFTAdapter:
        rank, d_model = W.shape
        construction: dict = {}
        if task is not None:
            construction["task"] = task
        manifest = ArtifactManifest(
            artifact_type="reft",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_model,
            construction=construction,
        )
        adapter = cls(manifest)
        adapter._W = W
        adapter._b = b if b is not None else torch.zeros(rank)
        adapter._rank = rank
        return adapter
