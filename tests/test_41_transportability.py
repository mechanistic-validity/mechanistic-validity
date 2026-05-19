import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "transportability" / "41_transportability.py"
)
_spec = importlib.util.spec_from_file_location("transportability_41", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["transportability_41"] = _mod
_spec.loader.exec_module(_mod)

normalize_layer_positions = _mod.normalize_layer_positions
layer_profile = _mod.layer_profile
compute_structural_transportability = _mod.compute_structural_transportability

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_normalize_layer_positions_basic():
    heads = {(0, 0), (6, 1), (11, 2)}
    result = normalize_layer_positions(heads, n_layers=12)
    assert len(result) == 3
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_normalize_layer_positions_single():
    heads = {(5, 0)}
    result = normalize_layer_positions(heads, n_layers=12)
    assert len(result) == 1
    assert result[0] == pytest.approx(5.0 / 11.0)


def test_layer_profile_normalized():
    heads = {(0, 0), (0, 1), (3, 0)}
    profile = layer_profile(heads, n_layers=5)
    assert profile.shape == (5,)
    assert profile.sum() == pytest.approx(1.0)


def test_layer_profile_empty():
    profile = layer_profile(set(), n_layers=4)
    assert profile.sum() == pytest.approx(0.0)


def test_compute_structural_transportability_same_model():
    heads = {(0, 0), (5, 1), (11, 2)}
    result = compute_structural_transportability(heads, 12, heads, 12)
    assert result["thirds_cosine_similarity"] == pytest.approx(1.0)
    assert result["source_layer_fractions"] == result["target_layer_fractions"]


def test_compute_structural_transportability_different_sizes():
    source_heads = {(0, 0), (5, 1), (11, 2)}
    target_heads = {(0, 0), (12, 1), (23, 2)}
    result = compute_structural_transportability(source_heads, 12, target_heads, 24)
    assert "thirds_cosine_similarity" in result
    assert 0.0 <= result["thirds_cosine_similarity"] <= 1.0


def test_compute_structural_transportability_keys():
    heads = {(0, 0)}
    result = compute_structural_transportability(heads, 4, heads, 4)
    expected_keys = {"source_layer_fractions", "target_layer_fractions",
                     "source_thirds_distribution", "target_thirds_distribution",
                     "thirds_cosine_similarity"}
    assert set(result.keys()) == expected_keys


def test_compute_structural_transportability_thirds_sum_to_one():
    heads = {(0, 0), (3, 1), (8, 2), (11, 3)}
    result = compute_structural_transportability(heads, 12, heads, 12)
    assert sum(result["source_thirds_distribution"]) == pytest.approx(1.0)
    assert sum(result["target_thirds_distribution"]) == pytest.approx(1.0)
