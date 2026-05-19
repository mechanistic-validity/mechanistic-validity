import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import (
    EvalResult, load_model, get_circuit_heads, get_circuit, get_all_edges,
)

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "ov_qk_analysis" / "49_ov_qk_composition.py"
)
_spec = importlib.util.spec_from_file_location("_ov_qk_49", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_composition_matrix = _mod.compute_composition_matrix

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def comp_matrix(gpt2_model):
    return compute_composition_matrix(gpt2_model)


def test_compute_composition_matrix_shape(gpt2_model, comp_matrix):
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert comp_matrix.shape == (n_layers, n_heads, n_layers, n_heads)


def test_compute_composition_matrix_nonnegative(comp_matrix):
    assert (comp_matrix >= 0).all()


def test_compute_composition_matrix_bounded_by_one(comp_matrix):
    assert comp_matrix.max() <= 1.0 + 1e-6


def test_compute_composition_matrix_zero_same_layer(gpt2_model, comp_matrix):
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    for L in range(n_layers):
        for H1 in range(n_heads):
            for H2 in range(n_heads):
                assert comp_matrix[L, H1, L, H2] == pytest.approx(0.0)


def test_compute_composition_matrix_zero_backward(gpt2_model, comp_matrix):
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    for L1 in range(n_layers):
        for H1 in range(n_heads):
            for L2 in range(L1):
                for H2 in range(n_heads):
                    assert comp_matrix[L1, H1, L2, H2] == pytest.approx(0.0)


def test_circuit_edges_have_scores(gpt2_model, comp_matrix):
    circuit = get_circuit(TASK)
    circuit_edges = get_all_edges(circuit)
    for L1, H1, L2, H2 in circuit_edges:
        score = comp_matrix[L1, H1, L2, H2]
        assert isinstance(float(score), float)
        assert score >= 0.0
