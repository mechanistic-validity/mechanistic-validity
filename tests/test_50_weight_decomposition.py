import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "ica_nmf" / "50_weight_decomposition.py"
)
_spec = importlib.util.spec_from_file_location("_weight_decomp_50", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

decompose_and_measure = _mod.decompose_and_measure
components_for_variance = _mod.components_for_variance
get_ov_matrices = _mod.get_ov_matrices

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_decompose_and_measure_returns_float():
    stacked = np.abs(np.random.randn(5, 100)).astype(np.float64) + 0.01
    error = decompose_and_measure(stacked, n_components=2)
    assert isinstance(error, float)
    assert 0.0 <= error <= 1.0


def test_decompose_and_measure_more_components_lower_error():
    stacked = np.abs(np.random.randn(10, 50)).astype(np.float64) + 0.01
    err_2 = decompose_and_measure(stacked, n_components=2)
    err_5 = decompose_and_measure(stacked, n_components=5)
    assert err_5 <= err_2 + 0.01


def test_decompose_and_measure_rank_one_matrix():
    row = np.abs(np.random.randn(1, 100)) + 0.01
    stacked = np.repeat(row, 5, axis=0)
    error = decompose_and_measure(stacked, n_components=1)
    assert error == pytest.approx(0.0, abs=0.05)


def test_components_for_variance_returns_positive_int():
    stacked = np.random.randn(8, 50).astype(np.float64)
    n_comp = components_for_variance(stacked, threshold=0.90)
    assert isinstance(n_comp, int)
    assert 1 <= n_comp <= 8


def test_components_for_variance_higher_threshold_needs_more():
    stacked = np.random.randn(10, 50).astype(np.float64)
    n_comp_50 = components_for_variance(stacked, threshold=0.50)
    n_comp_99 = components_for_variance(stacked, threshold=0.99)
    assert n_comp_99 >= n_comp_50


def test_get_ov_matrices_shape(gpt2_model):
    ov_flat = get_ov_matrices(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(ov_flat) == n_layers * n_heads
    for (L, H), vec in ov_flat.items():
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert (vec >= 0).all()


def test_get_ov_matrices_keys_cover_all_heads(gpt2_model):
    ov_flat = get_ov_matrices(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    for L in range(n_layers):
        for H in range(n_heads):
            assert (L, H) in ov_flat
