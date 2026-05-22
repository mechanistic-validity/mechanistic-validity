"""Bilinear MLP adapter — wraps eigendecomposed bilinear MLP weights.

Bilinear MLPs replace the activation function with a bilinear form:
  out = (x @ W1) * (x @ W2) @ W_out
The interaction tensor W1^T @ diag @ W2 can be eigendecomposed to yield
interpretable directions without a learned dictionary. No encode/decode —
directions are eigenvectors, activations are projections onto them.
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactManifest, WeightArtifactAdapter


class BilinearMLPAdapter(WeightArtifactAdapter):
    artifact_type = "bilinear_mlp"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._eigenvectors = None  # (n_components, d_model)
        self._eigenvalues = None  # (n_components,)
        self._W1 = None  # (d_model, d_mlp)
        self._W2 = None  # (d_model, d_mlp)
        self._W_out = None  # (d_mlp, d_model)

    def load(self) -> None:
        path = self.manifest.construction.get("path", "")
        if not path:
            raise ValueError("BilinearMLPAdapter requires a 'path' in construction config")
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._eigenvectors = data["eigenvectors"]
        self._eigenvalues = data["eigenvalues"]
        if "W1" in data:
            self._W1 = data["W1"]
            self._W2 = data["W2"]
            self._W_out = data["W_out"]

    @property
    def n_components(self) -> int | None:
        if self._eigenvectors is None:
            return None
        return self._eigenvectors.shape[0]

    def _get_directions(self) -> torch.Tensor:
        if self._eigenvectors is None:
            self.load()
        return self._eigenvectors

    def metadata(self) -> dict:
        base = super().metadata()
        if self._eigenvectors is not None:
            base["n_components"] = self._eigenvectors.shape[0]
            base["d_model"] = self._eigenvectors.shape[1]
        if self._eigenvalues is not None:
            base["top_eigenvalue"] = float(self._eigenvalues[0])
            base["eigenvalue_sum"] = float(self._eigenvalues.sum())
        return base

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        target_model: str = "gpt2",
        hook_point: str = "",
        d_in: int = 768,
    ) -> BilinearMLPAdapter:
        manifest = ArtifactManifest(
            artifact_type="bilinear_mlp",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            construction={"path": path},
        )
        return cls(manifest)

    @classmethod
    def from_eigendecomposition(
        cls,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        W1: torch.Tensor | None = None,
        W2: torch.Tensor | None = None,
        W_out: torch.Tensor | None = None,
        target_model: str = "gpt2",
        hook_point: str = "",
    ) -> BilinearMLPAdapter:
        d_model = eigenvectors.shape[1]
        manifest = ArtifactManifest(
            artifact_type="bilinear_mlp",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_model,
        )
        adapter = cls(manifest)
        adapter._eigenvectors = eigenvectors
        adapter._eigenvalues = eigenvalues
        adapter._W1 = W1
        adapter._W2 = W2
        adapter._W_out = W_out
        return adapter

    @classmethod
    def from_weights(
        cls,
        W1: torch.Tensor,
        W2: torch.Tensor,
        W_out: torch.Tensor,
        n_components: int | None = None,
        target_model: str = "gpt2",
        hook_point: str = "",
    ) -> BilinearMLPAdapter:
        d_model = W1.shape[0]
        interaction = W1 @ torch.diag_embed(torch.ones(W1.shape[1])) @ W2.T
        sym = (interaction + interaction.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)
        idx = eigenvalues.abs().argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].T

        if n_components is not None:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:n_components]

        manifest = ArtifactManifest(
            artifact_type="bilinear_mlp",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_model,
        )
        adapter = cls(manifest)
        adapter._eigenvectors = eigenvectors
        adapter._eigenvalues = eigenvalues
        adapter._W1 = W1
        adapter._W2 = W2
        adapter._W_out = W_out
        return adapter
