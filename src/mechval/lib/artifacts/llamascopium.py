"""Llamascopium adapter — wraps any SparseDictionary subclass as an ArtifactAdapter.

Covers all Llamascopium model types (SAE, crosscoder, LORSA, CLT, MOLT)
through their shared SparseDictionary interface. Also handles SAELens objects
that have been converted via SparseDictionary.from_saelens().
"""
from __future__ import annotations

import torch

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class LlamascopiumAdapter(ArtifactAdapter):
    artifact_type = "llamascopium"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._model = None

    def load(self) -> None:
        from llamascopium.models.sparse_dictionary import SparseDictionary

        path = self.manifest.construction.get("path", "")
        self._model = SparseDictionary.from_pretrained(path)

    @property
    def sae_type(self) -> str | None:
        if self._model is None:
            return None
        return self._model.cfg.sae_type

    @property
    def hook_points(self) -> list[str]:
        if self._model is None:
            return []
        return self._model.cfg.associated_hook_points

    def directions(self, layer: int | None = None):
        if self._model is None:
            self.load()
        if hasattr(self._model, "W_D"):
            W_D = self._model.W_D.detach()
            if W_D.ndim == 3 and layer is not None:
                return W_D[layer]
            return W_D
        if hasattr(self._model, "W_dec"):
            return self._model.W_dec.detach()
        raise AttributeError(f"{type(self._model).__name__} has no decoder weight")

    def activations(self, model, tokens, hook_name: str):
        if self._model is None:
            self.load()
        with torch.no_grad():
            hooks_in = self._model.cfg.hooks_in
            _, cache = model.run_with_cache(tokens, names_filter=hooks_in)
            if len(hooks_in) == 1:
                x = cache[hooks_in[0]]
            else:
                x = torch.stack([cache[hp] for hp in hooks_in], dim=-2)
            self._model.to(x.device)
            return self._model.encode(x)

    def ablate(self, model, tokens, units: list[int], site: str):
        if self._model is None:
            self.load()
        with torch.no_grad():
            hooks_in = self._model.cfg.hooks_in
            _, cache = model.run_with_cache(tokens, names_filter=hooks_in)
            if len(hooks_in) == 1:
                x = cache[hooks_in[0]]
            else:
                x = torch.stack([cache[hp] for hp in hooks_in], dim=-2)
            self._model.to(x.device)
            feature_acts = self._model.encode(x)
            feature_acts[..., units] = 0.0
            reconstructed = self._model.decode(feature_acts)
            if reconstructed.ndim > x.ndim and site in hooks_in:
                head_idx = hooks_in.index(site)
                return reconstructed[..., head_idx, :]
            return reconstructed

    def metadata(self) -> dict:
        base = super().metadata()
        if self._model is not None:
            base["sae_type"] = self._model.cfg.sae_type
            base["d_model"] = self._model.cfg.d_model
            base["d_sae"] = self._model.cfg.d_sae
            base["hooks_in"] = self._model.cfg.hooks_in
            base["hooks_out"] = self._model.cfg.hooks_out
        return base

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        target_model: str = "gpt2",
        hook_point: str = "",
        d_in: int = 768,
    ) -> LlamascopiumAdapter:
        manifest = ArtifactManifest(
            artifact_type="llamascopium",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            construction={"path": path},
        )
        return cls(manifest)

    @classmethod
    def from_module(
        cls,
        module,
        target_model: str = "gpt2",
        d_in: int = 768,
    ) -> LlamascopiumAdapter:
        hook_point = ""
        if hasattr(module, "cfg") and hasattr(module.cfg, "hooks_in"):
            hooks = module.cfg.hooks_in
            if hooks:
                hook_point = hooks[0]
        manifest = ArtifactManifest(
            artifact_type="llamascopium",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            construction={},
        )
        adapter = cls(manifest)
        adapter._model = module
        return adapter
