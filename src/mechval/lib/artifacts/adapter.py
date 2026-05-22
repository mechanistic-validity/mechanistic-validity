"""ArtifactAdapter — unified interface for learned representation artifacts.

The framework evaluates artifacts; it does not train them. An adapter wraps
an external artifact (SAE, factor bank, transcoder, crosscoder) and exposes
a standard interface for metrics that need feature directions, activations,
or ablation capabilities.

Two base classes:
  ArtifactAdapter       — learned dictionaries with encode/decode (SAE, CLT, transcoder)
  WeightArtifactAdapter — weight-only decompositions (bilinear MLP, SVD);
                          activations() does post-hoc projection, not encode()
"""
from __future__ import annotations

import torch
from pydantic import BaseModel, Field


class ArtifactManifest(BaseModel):
    artifact_type: str
    target_model: str
    hook_point: str
    d_in: int
    d_sae: int | None = None
    construction: dict = Field(default_factory=dict)


class ArtifactAdapter:
    artifact_type: str = ""

    def __init__(self, manifest: ArtifactManifest) -> None:
        self.manifest = manifest

    def load(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement load()")

    def directions(self, layer: int | None = None):
        raise NotImplementedError(f"{type(self).__name__} must implement directions()")

    def activations(self, model, tokens, hook_name: str):
        raise NotImplementedError(f"{type(self).__name__} must implement activations()")

    def ablate(self, model, tokens, units: list[int], site: str):
        raise NotImplementedError(f"{type(self).__name__} must implement ablate()")

    def metadata(self) -> dict:
        return self.manifest.model_dump()


class WeightArtifactAdapter(ArtifactAdapter):
    """Base for weight-only decompositions (no learned encoder).

    Subclasses store directions (eigenvectors, SVD components, etc.) derived
    purely from model weights. activations() projects the residual stream
    onto those directions post-hoc. ablate() zeros projections and
    reconstructs via the pseudoinverse.
    """

    def _get_directions(self) -> torch.Tensor:
        raise NotImplementedError

    def directions(self, layer: int | None = None):
        return self._get_directions().detach()

    def activations(self, model, tokens, hook_name: str):
        dirs = self._get_directions()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            dirs = dirs.to(acts.device)
            return acts @ dirs.T

    def ablate(self, model, tokens, units: list[int], site: str):
        dirs = self._get_directions()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            acts = cache[site]
            dirs = dirs.to(acts.device)
            projections = acts @ dirs.T
            projections[..., units] = 0.0
            return projections @ dirs
