"""ArtifactAdapter — unified interface for learned representation artifacts.

The framework evaluates artifacts; it does not train them. An adapter wraps
an external artifact (SAE, factor bank, transcoder, crosscoder) and exposes
a standard interface for metrics that need feature directions, activations,
or ablation capabilities.
"""
from __future__ import annotations

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
