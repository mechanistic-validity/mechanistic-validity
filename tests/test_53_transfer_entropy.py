import importlib
import math

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_mod = importlib.import_module(
    "mechval.metrics.information.transfer_entropy.53_transfer_entropy"
)
partial_correlation = _mod.partial_correlation
compute_transfer_entropy_proxy = _mod.compute_transfer_entropy_proxy
run_transfer_entropy = _mod.run_transfer_entropy

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_transfer_entropy(gpt2_model, [TASK], n_prompts=3)


def test_partial_correlation_perfect_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = x * 2.0 + 1.0
    r, p = partial_correlation(x, y, None)
    assert r == pytest.approx(1.0, abs=1e-6)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_partial_correlation_no_controls():
    rng = np.random.RandomState(42)
    x = rng.randn(100)
    y = rng.randn(100)
    r, p = partial_correlation(x, y, None)
    assert abs(r) < 0.3


def test_partial_correlation_controlling_for_confounder():
    rng = np.random.RandomState(42)
    z = rng.randn(100)
    x = z + rng.randn(100) * 0.1
    y = z + rng.randn(100) * 0.1
    r_raw, _ = partial_correlation(x, y, None)
    r_cond, _ = partial_correlation(x, y, z.reshape(-1, 1))
    assert abs(r_cond) < abs(r_raw)


def test_partial_correlation_constant_residuals():
    x = np.ones(20)
    y = np.arange(20, dtype=float)
    r, p = partial_correlation(x, y, None)
    assert r == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_compute_transfer_entropy_proxy_known_edge():
    n_prompts = 50
    rng = np.random.RandomState(42)
    n_layers, n_heads = 3, 2
    dla = rng.randn(n_prompts, n_layers, n_heads)
    dla[:, 1, 0] = dla[:, 0, 0] * 0.8 + rng.randn(n_prompts) * 0.2

    circuit_heads = {(0, 0), (1, 0), (2, 0)}
    circuit_edges = {(0, 0, 1, 0)}

    circuit_te, non_circuit_te, details = compute_transfer_entropy_proxy(
        dla, circuit_heads, circuit_edges, n_layers, n_heads
    )

    assert len(circuit_te) == 1
    assert len(non_circuit_te) >= 1
    assert circuit_te[0] > 0.0
    for d in details:
        assert "sender" in d
        assert "receiver" in d
        assert "te_proxy" in d
        assert "is_circuit_edge" in d


def test_compute_transfer_entropy_proxy_independent_heads():
    n_prompts = 50
    rng = np.random.RandomState(42)
    n_layers, n_heads = 3, 1
    dla = rng.randn(n_prompts, n_layers, n_heads)

    circuit_heads = {(0, 0), (1, 0), (2, 0)}
    circuit_edges = {(0, 0, 1, 0)}

    circuit_te, non_circuit_te, details = compute_transfer_entropy_proxy(
        dla, circuit_heads, circuit_edges, n_layers, n_heads
    )

    for te_val in circuit_te + non_circuit_te:
        assert te_val < 0.3


def test_run_transfer_entropy_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_transfer_entropy_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C1.transfer_entropy"


def test_run_transfer_entropy_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_transfer_entropy_metadata_keys(circuit_results):
    expected = {"task", "mean_circuit_te", "mean_non_circuit_te", "ratio",
                "n_circuit_edges", "n_non_circuit_edges", "top_edges"}
    assert expected <= set(circuit_results[0].metadata.keys())


def test_run_transfer_entropy_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_transfer_entropy_n_samples(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 3
