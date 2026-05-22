"""Tests for the Sparse Feature Circuits metric."""
from __future__ import annotations

import numpy as np
import pytest

import importlib
sfc_mod = importlib.import_module("mechval.metrics.mechanistic_interpretability.methods.discovery.sparse_feature_circuits.92_sfc")


def test_hook_to_layer_parses_standard_names():
    assert sfc_mod._hook_to_layer("blocks.0.hook_resid_pre") == 0
    assert sfc_mod._hook_to_layer("blocks.5.attn.hook_z") == 5
    assert sfc_mod._hook_to_layer("blocks.11.hook_resid_mid") == 11


def test_hook_to_layer_returns_none_for_bad_names():
    assert sfc_mod._hook_to_layer("embed") is None
    assert sfc_mod._hook_to_layer("ln_final.hook_normalized") is None


def test_sfc_auroc_perfect_separation():
    n_features = 120
    n_heads = 12
    n_per_head = n_features // n_heads

    feature_attr = np.zeros(n_features)
    circuit_heads = {(3, 1), (3, 7)}
    for h in [1, 7]:
        start = h * n_per_head
        feature_attr[start:start + n_per_head] = 10.0

    auroc, concentration, stats = sfc_mod.compute_sfc_auroc(
        feature_attr, circuit_heads, "blocks.3.hook_resid_pre",
        n_layers=12, n_heads=12,
    )
    assert auroc == pytest.approx(1.0)
    assert concentration == pytest.approx(1.0)
    assert stats["n_circuit_heads_at_layer"] == 2


def test_sfc_auroc_random_features():
    rng = np.random.default_rng(42)
    n_features = 120
    feature_attr = rng.uniform(0, 1, n_features)
    circuit_heads = {(3, 0), (3, 5)}

    auroc, concentration, stats = sfc_mod.compute_sfc_auroc(
        feature_attr, circuit_heads, "blocks.3.hook_resid_pre",
        n_layers=12, n_heads=12,
    )
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= concentration <= 1.0


def test_sfc_auroc_no_circuit_heads_at_layer():
    n_features = 120
    feature_attr = np.ones(n_features)
    circuit_heads = {(5, 0)}

    auroc, concentration, stats = sfc_mod.compute_sfc_auroc(
        feature_attr, circuit_heads, "blocks.3.hook_resid_pre",
        n_layers=12, n_heads=12,
    )
    assert stats["n_circuit_heads_at_layer"] == 0


def test_sfc_auroc_empty_features():
    auroc, concentration, stats = sfc_mod.compute_sfc_auroc(
        np.array([]), {(3, 0)}, "blocks.3.hook_resid_pre",
        n_layers=12, n_heads=12,
    )
    assert "error" in stats
