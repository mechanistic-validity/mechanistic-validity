"""Tests for artifact adapters (SAE, transcoder, crosscoder, factor bank)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from mechval.lib.artifacts.adapter import ArtifactAdapter, ArtifactManifest, WeightArtifactAdapter
from mechval.lib.artifacts.bilinear_mlp import BilinearMLPAdapter
from mechval.lib.artifacts.frozen_baseline import FrozenBaselineAdapter
from mechval.lib.artifacts.reft import ReFTAdapter
from mechval.lib.artifacts.transcoder import TranscoderAdapter
from mechval.lib.artifacts.crosscoder import CrosscoderAdapter
from mechval.lib.artifacts.llamascopium import LlamascopiumAdapter


class FakeTranscoder(nn.Module):
    """Minimal transcoder-like module for testing."""

    def __init__(self, d_in: int, d_sae: int, d_out: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        self.cfg = type("Cfg", (), {"d_out": d_out})()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec


class FakeCrosscoder(nn.Module):
    """Minimal crosscoder-like module for testing."""

    def __init__(self, n_heads: int, d_model: int, d_sae: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(n_heads, d_model, d_sae))
        self.b_E = nn.Parameter(torch.zeros(n_heads, d_sae))
        self.W_D = nn.Parameter(torch.randn(n_heads, d_sae, d_model))
        self.b_D = nn.Parameter(torch.zeros(n_heads, d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_heads, d_model) -> (batch, d_sae)
        pre = torch.einsum("...hm,hms->...hs", x, self.W_E) + self.b_E
        summed = pre.sum(dim=-2)
        return torch.relu(summed)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, d_sae) -> (batch, n_heads, d_model)
        n_heads = self.W_D.shape[0]
        h_expanded = h.unsqueeze(-2).expand(*h.shape[:-1], n_heads, h.shape[-1])
        return torch.einsum("...hs,hsm->...hm", h_expanded, self.W_D) + self.b_D


class FakeModel:
    """Minimal model mock that returns cached activations."""

    def __init__(self, cache: dict[str, torch.Tensor]):
        self._cache = cache

    def run_with_cache(self, tokens, names_filter=None):
        filtered = {k: v for k, v in self._cache.items() if names_filter is None or k in names_filter}
        return None, filtered


def test_transcoder_adapter_directions():
    tc = FakeTranscoder(d_in=16, d_sae=32, d_out=16)
    manifest = ArtifactManifest(
        artifact_type="transcoder", target_model="test", hook_point="hook", d_in=16,
    )
    adapter = TranscoderAdapter(manifest)
    adapter._transcoder = tc
    dirs = adapter.directions()
    assert dirs.shape == (32, 16)


def test_transcoder_adapter_activations():
    tc = FakeTranscoder(d_in=16, d_sae=32, d_out=16)
    acts = torch.randn(2, 5, 16)
    model = FakeModel({"blocks.0.hook_resid_pre": acts})

    manifest = ArtifactManifest(
        artifact_type="transcoder", target_model="test", hook_point="blocks.0.hook_resid_pre", d_in=16,
    )
    adapter = TranscoderAdapter(manifest)
    adapter._transcoder = tc
    result = adapter.activations(model, None, "blocks.0.hook_resid_pre")
    assert result.shape == (2, 5, 32)
    expected = tc.encode(acts)
    assert torch.allclose(result, expected)


def test_transcoder_adapter_ablate_zeros_units():
    d_in, d_sae, d_out = 16, 32, 16
    tc = FakeTranscoder(d_in=d_in, d_sae=d_sae, d_out=d_out)
    acts = torch.randn(2, 5, d_in)
    model = FakeModel({"hook": acts})

    manifest = ArtifactManifest(
        artifact_type="transcoder", target_model="test", hook_point="hook", d_in=d_in,
    )
    adapter = TranscoderAdapter(manifest)
    adapter._transcoder = tc

    units_to_ablate = [0, 5, 10]
    result = adapter.ablate(model, None, units_to_ablate, "hook")
    assert result.shape == (2, 5, d_out)

    feature_acts = tc.encode(acts)
    feature_acts[:, :, units_to_ablate] = 0.0
    expected = tc.decode(feature_acts)
    assert torch.allclose(result, expected)


def test_transcoder_metadata_includes_d_out():
    tc = FakeTranscoder(d_in=16, d_sae=32, d_out=24)
    manifest = ArtifactManifest(
        artifact_type="transcoder", target_model="test", hook_point="hook", d_in=16,
    )
    adapter = TranscoderAdapter(manifest)
    adapter._transcoder = tc
    meta = adapter.metadata()
    assert meta["d_out"] == 24
    assert meta["artifact_type"] == "transcoder"


def test_crosscoder_adapter_directions_per_head():
    n_heads, d_model, d_sae = 3, 16, 32
    cc = FakeCrosscoder(n_heads=n_heads, d_model=d_model, d_sae=d_sae)
    hook_points = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre"]

    adapter = CrosscoderAdapter.from_module(cc, hook_points, d_in=d_model)
    dirs = adapter.directions()
    assert dirs.shape == (n_heads, d_sae, d_model)

    dirs_layer1 = adapter.directions(layer=1)
    assert dirs_layer1.shape == (d_sae, d_model)
    assert torch.allclose(dirs_layer1, cc.W_D[1].detach())


def test_crosscoder_adapter_activations():
    n_heads, d_model, d_sae = 2, 16, 32
    cc = FakeCrosscoder(n_heads=n_heads, d_model=d_model, d_sae=d_sae)
    hook_points = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"]

    acts0 = torch.randn(2, 5, d_model)
    acts1 = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_pre": acts0, "blocks.1.hook_resid_pre": acts1})

    adapter = CrosscoderAdapter.from_module(cc, hook_points, d_in=d_model)
    result = adapter.activations(model, None, "blocks.0.hook_resid_pre")

    stacked = torch.stack([acts0, acts1], dim=-2)
    expected = cc.encode(stacked)
    assert torch.allclose(result, expected)


def test_crosscoder_adapter_ablate():
    n_heads, d_model, d_sae = 2, 16, 32
    cc = FakeCrosscoder(n_heads=n_heads, d_model=d_model, d_sae=d_sae)
    hook_points = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"]

    acts0 = torch.randn(2, 5, d_model)
    acts1 = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_pre": acts0, "blocks.1.hook_resid_pre": acts1})

    adapter = CrosscoderAdapter.from_module(cc, hook_points, d_in=d_model)
    units_to_ablate = [0, 3, 7]
    result = adapter.ablate(model, None, units_to_ablate, "blocks.1.hook_resid_pre")
    assert result.shape == (2, 5, d_model)

    stacked = torch.stack([acts0, acts1], dim=-2)
    feature_acts = cc.encode(stacked)
    feature_acts[..., units_to_ablate] = 0.0
    reconstructed = cc.decode(feature_acts)
    expected = reconstructed[..., 1, :]
    assert torch.allclose(result, expected)


def test_crosscoder_metadata():
    cc = FakeCrosscoder(n_heads=3, d_model=16, d_sae=32)
    hook_points = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre"]
    adapter = CrosscoderAdapter.from_module(cc, hook_points, d_in=16)
    meta = adapter.metadata()
    assert meta["n_heads"] == 3
    assert meta["hook_points"] == hook_points
    assert meta["artifact_type"] == "crosscoder"


class FakeLlamascopiumConfig:
    """Minimal SparseDictionaryConfig-like object."""

    def __init__(self, sae_type: str, d_model: int, d_sae: int, hooks_in: list[str], hooks_out: list[str]):
        self.sae_type = sae_type
        self.d_model = d_model
        self.d_sae = d_sae
        self._hooks_in = hooks_in
        self._hooks_out = hooks_out

    @property
    def hooks_in(self) -> list[str]:
        return self._hooks_in

    @property
    def hooks_out(self) -> list[str]:
        return self._hooks_out

    @property
    def associated_hook_points(self) -> list[str]:
        return self._hooks_in + self._hooks_out


class FakeLlamascopiumSAE(nn.Module):
    """Minimal SparseDictionary-like SAE (single hook point, W_E/W_D naming)."""

    def __init__(self, d_model: int, d_sae: int, hook_point: str):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_sae))
        self.b_E = nn.Parameter(torch.zeros(d_sae))
        self.W_D = nn.Parameter(torch.randn(d_sae, d_model))
        self.b_D = nn.Parameter(torch.zeros(d_model))
        self.cfg = FakeLlamascopiumConfig(
            sae_type="sae", d_model=d_model, d_sae=d_sae,
            hooks_in=[hook_point], hooks_out=[hook_point],
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_E + self.b_E)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_D + self.b_D


class FakeLlamascopiumCLT(nn.Module):
    """Minimal CLT-like module (multi-layer, W_D has layer dimension)."""

    def __init__(self, n_layers: int, d_model: int, d_sae: int, hook_points: list[str]):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(n_layers, d_model, d_sae))
        self.b_E = nn.Parameter(torch.zeros(n_layers, d_sae))
        self.W_D = nn.Parameter(torch.randn(n_layers, d_sae, d_model))
        self.b_D = nn.Parameter(torch.zeros(n_layers, d_model))
        self.cfg = FakeLlamascopiumConfig(
            sae_type="clt", d_model=d_model, d_sae=d_sae,
            hooks_in=hook_points, hooks_out=hook_points,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_layers, d_model)
        pre = torch.einsum("...lm,lms->...ls", x, self.W_E) + self.b_E
        return torch.relu(pre.sum(dim=-2))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        n_layers = self.W_D.shape[0]
        h_exp = h.unsqueeze(-2).expand(*h.shape[:-1], n_layers, h.shape[-1])
        return torch.einsum("...ls,lsm->...lm", h_exp, self.W_D) + self.b_D


def test_llamascopium_sae_directions():
    d_model, d_sae = 16, 32
    sae = FakeLlamascopiumSAE(d_model, d_sae, "blocks.0.hook_resid_pre")
    adapter = LlamascopiumAdapter.from_module(sae, d_in=d_model)
    dirs = adapter.directions()
    assert dirs.shape == (d_sae, d_model)
    assert torch.allclose(dirs, sae.W_D.detach())


def test_llamascopium_sae_activations():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    acts = torch.randn(2, 5, d_model)
    model = FakeModel({hp: acts})

    adapter = LlamascopiumAdapter.from_module(sae, d_in=d_model)
    result = adapter.activations(model, None, hp)
    expected = sae.encode(acts)
    assert torch.allclose(result, expected)


def test_llamascopium_sae_ablate():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    acts = torch.randn(2, 5, d_model)
    model = FakeModel({hp: acts})

    adapter = LlamascopiumAdapter.from_module(sae, d_in=d_model)
    units = [0, 5, 10]
    result = adapter.ablate(model, None, units, hp)
    assert result.shape == (2, 5, d_model)

    feature_acts = sae.encode(acts)
    feature_acts[..., units] = 0.0
    expected = sae.decode(feature_acts)
    assert torch.allclose(result, expected)


def test_llamascopium_clt_directions_per_layer():
    n_layers, d_model, d_sae = 3, 16, 32
    hook_points = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    clt = FakeLlamascopiumCLT(n_layers, d_model, d_sae, hook_points)

    adapter = LlamascopiumAdapter.from_module(clt, d_in=d_model)
    dirs = adapter.directions()
    assert dirs.shape == (n_layers, d_sae, d_model)

    dirs_l1 = adapter.directions(layer=1)
    assert dirs_l1.shape == (d_sae, d_model)
    assert torch.allclose(dirs_l1, clt.W_D[1].detach())


def test_llamascopium_metadata():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    adapter = LlamascopiumAdapter.from_module(sae, d_in=d_model)
    meta = adapter.metadata()
    assert meta["sae_type"] == "sae"
    assert meta["d_model"] == d_model
    assert meta["d_sae"] == d_sae
    assert meta["hooks_in"] == [hp]
    assert meta["artifact_type"] == "llamascopium"


def test_bilinear_mlp_from_eigendecomposition():
    d_model, n_components = 16, 8
    eigenvectors = torch.randn(n_components, d_model)
    eigenvectors = eigenvectors / eigenvectors.norm(dim=1, keepdim=True)
    eigenvalues = torch.sort(torch.rand(n_components), descending=True).values

    adapter = BilinearMLPAdapter.from_eigendecomposition(
        eigenvectors, eigenvalues, hook_point="blocks.0.hook_resid_mid",
    )
    dirs = adapter.directions()
    assert dirs.shape == (n_components, d_model)
    assert torch.allclose(dirs, eigenvectors)


def test_bilinear_mlp_activations():
    d_model, n_components = 16, 8
    eigenvectors = torch.randn(n_components, d_model)
    eigenvalues = torch.rand(n_components)

    acts = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_mid": acts})

    adapter = BilinearMLPAdapter.from_eigendecomposition(
        eigenvectors, eigenvalues, hook_point="blocks.0.hook_resid_mid",
    )
    result = adapter.activations(model, None, "blocks.0.hook_resid_mid")
    expected = acts @ eigenvectors.T
    assert result.shape == (2, 5, n_components)
    assert torch.allclose(result, expected)


def test_bilinear_mlp_ablate():
    d_model, n_components = 16, 8
    eigenvectors = torch.randn(n_components, d_model)
    eigenvalues = torch.rand(n_components)

    acts = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_mid": acts})

    adapter = BilinearMLPAdapter.from_eigendecomposition(
        eigenvectors, eigenvalues, hook_point="blocks.0.hook_resid_mid",
    )
    units = [0, 3, 5]
    result = adapter.ablate(model, None, units, "blocks.0.hook_resid_mid")
    assert result.shape == (2, 5, d_model)

    projections = acts @ eigenvectors.T
    projections[..., units] = 0.0
    expected = projections @ eigenvectors
    assert torch.allclose(result, expected)


def test_bilinear_mlp_from_weights():
    d_model, d_mlp = 16, 32
    W1 = torch.randn(d_model, d_mlp)
    W2 = torch.randn(d_model, d_mlp)
    W_out = torch.randn(d_mlp, d_model)

    adapter = BilinearMLPAdapter.from_weights(
        W1, W2, W_out, n_components=8, hook_point="blocks.0.hook_resid_mid",
    )
    dirs = adapter.directions()
    assert dirs.shape == (8, d_model)
    assert adapter.n_components == 8

    meta = adapter.metadata()
    assert meta["artifact_type"] == "bilinear_mlp"
    assert meta["n_components"] == 8
    assert "top_eigenvalue" in meta


def test_bilinear_mlp_metadata():
    d_model, n_components = 16, 8
    eigenvalues = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05])
    eigenvectors = torch.randn(n_components, d_model)

    adapter = BilinearMLPAdapter.from_eigendecomposition(
        eigenvectors, eigenvalues, hook_point="blocks.0.hook_resid_mid",
    )
    meta = adapter.metadata()
    assert meta["n_components"] == 8
    assert meta["d_model"] == 16
    assert meta["top_eigenvalue"] == pytest.approx(5.0)
    assert meta["eigenvalue_sum"] == pytest.approx(11.95)
    assert meta["artifact_type"] == "bilinear_mlp"


def test_bilinear_mlp_inherits_weight_artifact_adapter():
    adapter = BilinearMLPAdapter.from_eigendecomposition(
        torch.randn(4, 8), torch.rand(4),
    )
    assert isinstance(adapter, WeightArtifactAdapter)
    assert isinstance(adapter, ArtifactAdapter)


def test_frozen_baseline_decoder_randomizes_directions():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    inner = LlamascopiumAdapter.from_module(sae, d_in=d_model)

    frozen = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_decoder", seed=42)
    real_dirs = inner.directions()
    frozen_dirs = frozen.directions()

    assert frozen_dirs.shape == real_dirs.shape
    cosines = torch.nn.functional.cosine_similarity(real_dirs, frozen_dirs, dim=-1)
    assert cosines.abs().mean() < 0.5


def test_frozen_baseline_encoder_randomizes_activations():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    inner = LlamascopiumAdapter.from_module(sae, d_in=d_model)

    acts = torch.randn(2, 5, d_model)
    model = FakeModel({hp: acts})

    frozen = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_encoder", seed=42)
    real_result = inner.activations(model, None, hp)
    frozen_result = frozen.activations(model, None, hp)

    assert frozen_result.shape == real_result.shape
    assert not torch.allclose(frozen_result, real_result, atol=1e-3)


def test_frozen_baseline_preserves_real_directions_in_encoder_mode():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    inner = LlamascopiumAdapter.from_module(sae, d_in=d_model)

    frozen = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_encoder", seed=42)
    assert torch.allclose(frozen.directions(), inner.directions())


def test_frozen_baseline_metadata():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    inner = LlamascopiumAdapter.from_module(sae, d_in=d_model)

    frozen = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_decoder", seed=99)
    meta = frozen.metadata()
    assert meta["frozen_baseline_mode"] == "frozen_decoder"
    assert meta["frozen_baseline_seed"] == 99
    assert meta["artifact_type"] == "frozen_baseline"


def test_frozen_baseline_seed_deterministic():
    d_model, d_sae = 16, 32
    hp = "blocks.0.hook_resid_pre"
    sae = FakeLlamascopiumSAE(d_model, d_sae, hp)
    inner = LlamascopiumAdapter.from_module(sae, d_in=d_model)

    f1 = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_decoder", seed=42)
    f2 = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_decoder", seed=42)
    assert torch.allclose(f1.directions(), f2.directions())

    f3 = FrozenBaselineAdapter.from_adapter(inner, mode="frozen_decoder", seed=99)
    assert not torch.allclose(f1.directions(), f3.directions())


def test_reft_directions_from_weights():
    rank, d_model = 4, 16
    W = torch.randn(rank, d_model)
    adapter = ReFTAdapter.from_weights(W, hook_point="blocks.0.hook_resid_pre")
    dirs = adapter.directions()
    assert dirs.shape == (rank, d_model)
    assert torch.allclose(dirs, W)


def test_reft_activations():
    rank, d_model = 4, 16
    W = torch.randn(rank, d_model)
    b = torch.randn(rank)
    acts = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_pre": acts})

    adapter = ReFTAdapter.from_weights(W, b, hook_point="blocks.0.hook_resid_pre")
    result = adapter.activations(model, None, "blocks.0.hook_resid_pre")
    expected = acts @ W.T
    assert result.shape == (2, 5, rank)
    assert torch.allclose(result, expected)


def test_reft_ablate():
    rank, d_model = 4, 16
    W = torch.randn(rank, d_model)
    acts = torch.randn(2, 5, d_model)
    model = FakeModel({"blocks.0.hook_resid_pre": acts})

    adapter = ReFTAdapter.from_weights(W, hook_point="blocks.0.hook_resid_pre")
    units = [0, 2]
    result = adapter.ablate(model, None, units, "blocks.0.hook_resid_pre")
    assert result.shape == (2, 5, d_model)

    projections = acts @ W.T
    projections[..., units] = 0.0
    expected = projections @ W
    assert torch.allclose(result, expected)


def test_reft_metadata():
    rank, d_model = 4, 16
    W = torch.randn(rank, d_model)
    b = torch.randn(rank)
    adapter = ReFTAdapter.from_weights(
        W, b, hook_point="blocks.0.hook_resid_pre", task="sentiment",
    )
    meta = adapter.metadata()
    assert meta["rank"] == rank
    assert meta["d_model"] == d_model
    assert meta["has_bias"] is True
    assert meta["task"] == "sentiment"
    assert meta["artifact_type"] == "reft"


def test_all_adapters_share_interface():
    """All adapter types implement the same base interface."""
    from mechval.lib.artifacts import SAEAdapter, FactorBankAdapter

    for cls in [SAEAdapter, TranscoderAdapter, CrosscoderAdapter, FactorBankAdapter, LlamascopiumAdapter, BilinearMLPAdapter, ReFTAdapter]:
        assert hasattr(cls, "artifact_type")
        assert hasattr(cls, "load")
        assert hasattr(cls, "directions")
        assert hasattr(cls, "activations")
        assert hasattr(cls, "ablate")
        assert hasattr(cls, "metadata")
        assert issubclass(cls, ArtifactAdapter)
