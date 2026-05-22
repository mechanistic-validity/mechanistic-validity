"""SAE adapter — wraps sae_lens.SAE as an ArtifactAdapter."""
from __future__ import annotations

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class SAEAdapter(ArtifactAdapter):
    artifact_type = "sae"

    def __init__(self, manifest: ArtifactManifest) -> None:
        super().__init__(manifest)
        self._sae = None

    def load(self, device: str | None = None) -> None:
        from sae_lens import SAE

        kwargs = dict(
            release=self.manifest.construction.get("release", ""),
            sae_id=self.manifest.construction.get("sae_id", ""),
        )
        if device is not None:
            kwargs["device"] = device
        self._sae = SAE.from_pretrained(**kwargs)

    def directions(self, layer: int | None = None):
        if self._sae is None:
            self.load()
        return self._sae.W_dec.detach()

    def activations(self, model, tokens, hook_name: str):
        if self._sae is None:
            self.load()
        import torch

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            self._sae.to(acts.device)
            return self._sae.encode(acts)

    def ablate(self, model, tokens, units: list[int], site: str):
        if self._sae is None:
            self.load()
        import torch

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            acts = cache[site]
            self._sae.to(acts.device)
            feature_acts = self._sae.encode(acts)
            feature_acts[:, :, units] = 0.0
            reconstructed = self._sae.decode(feature_acts)
            return reconstructed

    @classmethod
    def from_pretrained(cls, release: str, sae_id: str = "", hook_point: str = "") -> SAEAdapter:
        manifest = ArtifactManifest(
            artifact_type="sae",
            target_model="gpt2",
            hook_point=hook_point,
            d_in=768,
            construction={"release": release, "sae_id": sae_id},
        )
        adapter = cls(manifest)
        return adapter
