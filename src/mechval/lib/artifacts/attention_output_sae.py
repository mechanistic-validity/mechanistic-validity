"""Attention Output SAE adapter (Kissane et al., ICML 2024).

These SAEs are trained on hook_z (concatenated mixed values -- attention
outputs before the W_O projection). The key property: since
hook_z = concat(z_0, z_1, ..., z_h), each feature's decoder direction
decomposes into per-head contributions, enabling head-level attribution.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest


class AttentionOutputSAEAdapter(ArtifactAdapter):
    artifact_type = "attention_output_sae"

    def __init__(
        self,
        manifest: ArtifactManifest,
        n_heads: int,
        d_head: int,
    ) -> None:
        super().__init__(manifest)
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_enc: torch.Tensor | None = None
        self.W_dec: torch.Tensor | None = None
        self.b_enc: torch.Tensor | None = None
        self.b_dec: torch.Tensor | None = None

    def _check_loaded(self) -> None:
        if self.W_enc is None:
            raise RuntimeError("Weights not loaded. Call load(), from_pretrained(), or from_weights() first.")

    def load(self, device: str | None = None) -> None:
        path = self.manifest.construction.get("path")
        if path is None:
            raise ValueError("No 'path' in manifest construction; use from_pretrained() or from_weights().")
        state = torch.load(path, map_location=device or "cpu", weights_only=True)
        self.W_enc = state["W_enc"]
        self.W_dec = state["W_dec"]
        self.b_enc = state["b_enc"]
        self.b_dec = state["b_dec"]

    def directions(self, layer: int | None = None) -> torch.Tensor:
        self._check_loaded()
        return self.W_dec.detach()

    def activations(self, model, tokens, hook_name: str) -> torch.Tensor:
        self._check_loaded()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            z = cache[hook_name]  # (batch, seq, n_heads, d_head)
            batch, seq = z.shape[0], z.shape[1]
            z_flat = z.reshape(batch, seq, -1)  # (batch, seq, n_heads * d_head)
            W_enc = self.W_enc.to(z_flat.device)
            b_enc = self.b_enc.to(z_flat.device)
            pre_acts = z_flat @ W_enc + b_enc  # (batch, seq, d_sae)
            return torch.relu(pre_acts)

    def ablate(self, model, tokens, units: list[int], site: str) -> torch.Tensor:
        self._check_loaded()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[site])
            z = cache[site]  # (batch, seq, n_heads, d_head)
            batch, seq = z.shape[0], z.shape[1]
            z_flat = z.reshape(batch, seq, -1)
            device = z_flat.device
            W_enc = self.W_enc.to(device)
            W_dec = self.W_dec.to(device)
            b_enc = self.b_enc.to(device)
            b_dec = self.b_dec.to(device)
            pre_acts = z_flat @ W_enc + b_enc
            feature_acts = torch.relu(pre_acts)
            feature_acts[..., units] = 0.0
            reconstructed = feature_acts @ W_dec + b_dec
            return reconstructed.reshape(batch, seq, self.n_heads, self.d_head)

    def head_attribution(self, feature_idx: int) -> torch.Tensor:
        """Per-head contribution to a feature via decoder norm decomposition.

        Returns a tensor of shape (n_heads,) where each entry is the L2 norm
        of that head's slice of the decoder direction for the given feature.
        """
        self._check_loaded()
        dec_row = self.W_dec[feature_idx]  # (n_heads * d_head,)
        per_head = dec_row.reshape(self.n_heads, self.d_head)  # (n_heads, d_head)
        return per_head.norm(dim=1)  # (n_heads,)

    def per_head_activations(
        self, model, tokens, hook_name: str
    ) -> dict[int, torch.Tensor]:
        """Feature activations from each head's z slice independently.

        Returns a dict mapping head_idx -> (batch, seq, d_sae) tensor of
        feature activations computed using only that head's contribution.
        """
        self._check_loaded()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            z = cache[hook_name]  # (batch, seq, n_heads, d_head)
            device = z.device
            W_enc = self.W_enc.to(device)
            b_enc = self.b_enc.to(device)

            result = {}
            for h in range(self.n_heads):
                z_h = torch.zeros_like(z)
                z_h[:, :, h, :] = z[:, :, h, :]
                batch, seq = z_h.shape[0], z_h.shape[1]
                z_h_flat = z_h.reshape(batch, seq, -1)
                pre_acts = z_h_flat @ W_enc + b_enc
                result[h] = torch.relu(pre_acts)
            return result

    def metadata(self) -> dict:
        base = super().metadata()
        base["n_heads"] = self.n_heads
        base["d_head"] = self.d_head
        return base

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        hook_point: str = "",
        target_model: str = "gpt2",
        n_heads: int = 12,
        d_head: int = 64,
        d_in: int | None = None,
        d_sae: int | None = None,
        device: str | None = None,
    ) -> AttentionOutputSAEAdapter:
        if d_in is None:
            d_in = n_heads * d_head
        manifest = ArtifactManifest(
            artifact_type="attention_output_sae",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            d_sae=d_sae,
            construction={"path": str(path)},
        )
        adapter = cls(manifest, n_heads=n_heads, d_head=d_head)
        adapter.load(device=device)
        return adapter

    @classmethod
    def from_weights(
        cls,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        n_heads: int,
        d_head: int,
        hook_point: str = "",
        target_model: str = "gpt2",
    ) -> AttentionOutputSAEAdapter:
        d_in = n_heads * d_head
        d_sae = W_dec.shape[0]
        manifest = ArtifactManifest(
            artifact_type="attention_output_sae",
            target_model=target_model,
            hook_point=hook_point,
            d_in=d_in,
            d_sae=d_sae,
        )
        adapter = cls(manifest, n_heads=n_heads, d_head=d_head)
        adapter.W_enc = W_enc
        adapter.W_dec = W_dec
        adapter.b_enc = b_enc
        adapter.b_dec = b_dec
        return adapter
