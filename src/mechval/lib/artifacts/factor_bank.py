"""Factor bank adapter — wraps FactorBankSAE as an ArtifactAdapter."""
from __future__ import annotations

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class FactorBankAdapter(ArtifactAdapter):
    artifact_type = "factor_bank"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._bank = None

    def load(self) -> None:
        raise NotImplementedError(
            "FactorBankAdapter.load() requires a checkpoint path. "
            "Use FactorBankAdapter.from_checkpoint(path) instead."
        )

    def directions(self, layer: int | None = None):
        if self._bank is None:
            raise RuntimeError("Factor bank not loaded. Call load() or from_checkpoint() first.")
        return self._bank.factors.detach()

    def activations(self, model, tokens, hook_name: str):
        if self._bank is None:
            raise RuntimeError("Factor bank not loaded.")
        import torch

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            return acts @ self._bank.factors.T

    def ablate(self, model, tokens, units: list[int], site: str):
        if self._bank is None:
            raise RuntimeError("Factor bank not loaded.")
        import torch

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            acts = cache[site]
            factor_acts = acts @ self._bank.factors.T
            factor_acts[:, :, units] = 0.0
            reconstructed = factor_acts @ self._bank.factors
            return reconstructed
