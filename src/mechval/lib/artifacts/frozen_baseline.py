"""Frozen baseline adapter — randomizes components of any adapter for null hypothesis.

Implements the "sanity checks" pattern from Korznikov et al. (2026): wraps any
ArtifactAdapter and replaces either the encoder, decoder, or both with random
fixed directions. This produces a null-hypothesis comparison — if a fully-trained
artifact scores no better than a frozen-random one on a metric, that metric is
not measuring anything meaningful about the learned representations.

Three modes:
  frozen_decoder  — random fixed directions, real encode() activations
  frozen_encoder  — random detection, real decode() directions
  frozen_both     — fully random baseline
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class FrozenBaselineAdapter(ArtifactAdapter):
    artifact_type = "frozen_baseline"

    def __init__(self, wrapped: ArtifactAdapter, mode: str = "frozen_decoder",
                 seed: int = 0) -> None:
        super().__init__(wrapped.manifest)
        self._wrapped = wrapped
        self._mode = mode
        self._seed = seed
        self._random_directions = None

    def _ensure_random_directions(self):
        if self._random_directions is not None:
            return
        real_dirs = self._wrapped.directions()
        gen = torch.Generator()
        gen.manual_seed(self._seed)
        rand = torch.randn(real_dirs.shape, generator=gen)
        rand = rand / rand.norm(dim=-1, keepdim=True)
        self._random_directions = rand

    def load(self) -> None:
        self._wrapped.load()

    def directions(self, layer: int | None = None):
        if self._mode in ("frozen_decoder", "frozen_both"):
            self._ensure_random_directions()
            return self._random_directions.detach()
        return self._wrapped.directions(layer)

    def activations(self, model, tokens, hook_name: str):
        if self._mode in ("frozen_encoder", "frozen_both"):
            self._ensure_random_directions()
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                acts = cache[hook_name]
                dirs = self._random_directions.to(acts.device)
                return acts @ dirs.T
        return self._wrapped.activations(model, tokens, hook_name)

    def ablate(self, model, tokens, units: list[int], site: str):
        dirs = self.directions()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            acts = cache[site]
            dirs = dirs.to(acts.device)
            feature_acts = acts @ dirs.T
            feature_acts[..., units] = 0.0
            return feature_acts @ dirs

    def metadata(self) -> dict:
        base = self._wrapped.metadata()
        base["frozen_baseline_mode"] = self._mode
        base["frozen_baseline_seed"] = self._seed
        base["artifact_type"] = "frozen_baseline"
        return base

    @classmethod
    def from_adapter(cls, adapter: ArtifactAdapter, mode: str = "frozen_decoder",
                     seed: int = 0) -> FrozenBaselineAdapter:
        return cls(adapter, mode=mode, seed=seed)
