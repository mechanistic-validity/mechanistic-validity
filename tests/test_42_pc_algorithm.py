import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "causal_discovery" / "42_pc_algorithm.py"
)
_spec = importlib.util.spec_from_file_location("pc_algorithm_42", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pc_algorithm_42"] = _mod
_spec.loader.exec_module(_mod)

pc_algorithm = _mod.pc_algorithm
partial_correlation = _mod.partial_correlation
compare_to_known_circuit = _mod.compare_to_known_circuit

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_partial_correlation_perfect():
    rng = np.random.RandomState(42)
    n = 100
    X = np.column_stack([rng.randn(n), rng.randn(n)])
    X[:, 1] = X[:, 0] + rng.randn(n) * 0.01
    r, p = partial_correlation(X, 0, 1, [])
    assert abs(r) > 0.9
    assert p < 0.05


def test_partial_correlation_independent():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 2)
    r, p = partial_correlation(X, 0, 1, [])
    assert abs(r) < 0.3


def test_partial_correlation_conditional():
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    x = z + rng.randn(n) * 0.1
    y = z + rng.randn(n) * 0.1
    w = rng.randn(n)
    X = np.column_stack([x, y, z, w])
    r_uncond, _ = partial_correlation(X, 0, 1, [])
    r_cond, _ = partial_correlation(X, 0, 1, [2])
    assert abs(r_cond) < abs(r_uncond)


def test_pc_algorithm_identity_matrix():
    rng = np.random.RandomState(42)
    n, d = 100, 3
    X = rng.randn(n, d)
    adj = pc_algorithm(X, alpha=0.05, max_cond_size=1)
    assert adj.shape == (d, d)
    for i in range(d):
        assert adj[i, i] == 0


def test_pc_algorithm_symmetric():
    rng = np.random.RandomState(42)
    n = 100
    x = rng.randn(n)
    y = x + rng.randn(n) * 0.1
    z = rng.randn(n)
    X = np.column_stack([x, y, z])
    adj = pc_algorithm(X, alpha=0.05, max_cond_size=1)
    for i in range(3):
        for j in range(3):
            assert adj[i, j] == adj[j, i]


def test_pc_algorithm_with_node_subset():
    rng = np.random.RandomState(42)
    n, d = 100, 6
    X = rng.randn(n, d)
    subset = [0, 2, 4]
    adj = pc_algorithm(X, alpha=0.05, max_cond_size=1, node_subset=subset)
    assert adj.shape == (3, 3)


def test_compare_to_known_circuit_perfect():
    n_heads = 2
    node_indices = [0, 1, 2, 3]
    adj = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    known_edges = {(0, 0, 1, 0)}
    result = compare_to_known_circuit(adj, node_indices, known_edges, n_heads)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["structural_hamming_distance"] == 0


def test_compare_to_known_circuit_no_overlap():
    n_heads = 2
    node_indices = [0, 1, 2, 3]
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = adj[1, 0] = 1
    known_edges = {(1, 0, 1, 1)}
    result = compare_to_known_circuit(adj, node_indices, known_edges, n_heads)
    assert result["recall"] == pytest.approx(0.0)
