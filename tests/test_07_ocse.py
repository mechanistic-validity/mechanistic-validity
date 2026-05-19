import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "granger_te" / "07_ocse.py"
)
_spec = importlib.util.spec_from_file_location("ocse_07", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ocse_07"] = _mod
_spec.loader.exec_module(_mod)

run_ocse = _mod.run_ocse
stability_selection = _mod.stability_selection
gaussian_cmi = _mod.gaussian_cmi
compute_ocse_parents = _mod.compute_ocse_parents
_compute_f1 = _mod._compute_f1

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_gaussian_cmi_perfect_correlation():
    rng = np.random.RandomState(42)
    x = rng.randn(100)
    y = x + rng.randn(100) * 0.01
    cmi = gaussian_cmi(x, y)
    assert cmi > 1.0


def test_gaussian_cmi_independent():
    rng = np.random.RandomState(42)
    x = rng.randn(100)
    y = rng.randn(100)
    cmi = gaussian_cmi(x, y)
    assert cmi < 0.5


def test_gaussian_cmi_conditional():
    rng = np.random.RandomState(42)
    z = rng.randn(100)
    x = z + rng.randn(100) * 0.1
    y = z + rng.randn(100) * 0.1
    cmi_marginal = gaussian_cmi(x, y)
    cmi_conditional = gaussian_cmi(x, y, z)
    assert cmi_conditional < cmi_marginal


def test_gaussian_cmi_too_few_samples():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    assert gaussian_cmi(x, y) == pytest.approx(0.0)


def test_compute_f1_perfect():
    circuit_heads = {(0, 0), (1, 1)}
    discovered = {0, 5}  # 0*4+0=0, 1*4+1=5
    f1, p, r = _compute_f1(circuit_heads, discovered, n_heads=4)
    assert f1 == pytest.approx(1.0)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_compute_f1_no_overlap():
    circuit_heads = {(0, 0)}
    discovered = {3}  # L0H3
    f1, p, r = _compute_f1(circuit_heads, discovered, n_heads=4)
    assert f1 == pytest.approx(0.0)
    assert p == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_compute_f1_partial():
    circuit_heads = {(0, 0), (1, 0)}
    discovered = {0, 2}  # L0H0 correct, L0H2 wrong
    f1, p, r = _compute_f1(circuit_heads, discovered, n_heads=4)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_stability_selection_returns_list():
    rng = np.random.RandomState(42)
    n, d = 50, 5
    X = rng.randn(n, d)
    y = X[:, 0] * 2 + X[:, 2] * 3 + rng.randn(n) * 0.1
    result = stability_selection(X, y, n_bootstrap=5, threshold=0.3)
    assert isinstance(result, list)
    for idx, freq, coef in result:
        assert 0 <= idx < d
        assert 0.0 <= freq <= 1.0


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_ocse(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C7.ocse"


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "precision", "recall", "stability_selection",
                "ocse", "combined", "n_circuit_heads", "n_discovered"}
    assert set(meta.keys()) == expected


def test_f1_in_range(circuit_results):
    assert 0.0 <= circuit_results[0].value <= 1.0


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
