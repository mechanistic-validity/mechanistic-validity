import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "causal"
    / "atp_star"
    / "G2b_atp_star.py"
)
_spec = importlib.util.spec_from_file_location("atp_star_g2b", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["atp_star_g2b"] = _mod
_spec.loader.exec_module(_mod)

compute_atp_star_scores = _mod.compute_atp_star_scores
score_circuit_edges = _mod.score_circuit_edges
run_atp_star = _mod.run_atp_star

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_score_circuit_edges_perfect_discrimination():
    n_layers = 3
    n_heads = 2
    n_total = n_layers * n_heads
    atp_star_scores = np.zeros((n_total, n_total), dtype=np.float64)

    circuit_edges = {(0, 0, 1, 0), (0, 1, 2, 0)}
    for Ls, Hs, Lr, Hr in circuit_edges:
        s_idx = Ls * n_heads + Hs
        r_idx = Lr * n_heads + Hr
        atp_star_scores[s_idx, r_idx] = 100.0

    fraction, stats = score_circuit_edges(
        atp_star_scores, circuit_edges, n_layers, n_heads
    )

    assert fraction == pytest.approx(1.0)
    assert stats["n_circuit_edges"] == 2
    assert stats["n_non_circuit_edges"] > 0
    assert stats["mean_circuit_score"] == pytest.approx(100.0)
    assert stats["mean_non_circuit_score"] == pytest.approx(0.0)


def test_score_circuit_edges_no_discrimination():
    n_layers = 4
    n_heads = 2
    n_total = n_layers * n_heads
    atp_star_scores = np.ones((n_total, n_total), dtype=np.float64) * 5.0

    circuit_edges = {(0, 0, 1, 0), (1, 1, 3, 0)}

    fraction, stats = score_circuit_edges(
        atp_star_scores, circuit_edges, n_layers, n_heads
    )

    assert fraction == pytest.approx(0.0)
    assert stats["n_circuit_edges"] == 2
    assert stats["mean_circuit_score"] == pytest.approx(5.0)
    assert stats["mean_non_circuit_score"] == pytest.approx(5.0)


def test_score_circuit_edges_empty():
    n_layers = 2
    n_heads = 2
    n_total = n_layers * n_heads
    atp_star_scores = np.ones((n_total, n_total), dtype=np.float64)

    fraction, stats = score_circuit_edges(
        atp_star_scores, set(), n_layers, n_heads
    )

    assert fraction == pytest.approx(0.0)


def test_atp_star_scores_always_nonnegative():
    n_layers = 3
    n_heads = 2
    n_total = n_layers * n_heads

    for _ in range(10):
        scores = np.random.randn(n_total, n_total)
        # AtP* scores are sums of absolute values, so must be >= 0
        abs_scores = np.abs(scores)
        circuit_edges = {(0, 0, 1, 0)}
        _, stats = score_circuit_edges(abs_scores, circuit_edges, n_layers, n_heads)
        assert stats["mean_circuit_score"] >= 0.0
        assert stats["mean_non_circuit_score"] >= 0.0


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_atp_star(gpt2_model, tasks=[TASK], n_prompts=3)


def test_run_atp_star_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_atp_star_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "G2b.atp_star"


def test_run_atp_star_value_in_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0


def test_run_atp_star_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_atp_star_metadata_has_per_edge(circuit_results):
    for r in circuit_results:
        assert "per_edge" in r.metadata
        assert isinstance(r.metadata["per_edge"], list)
        for edge in r.metadata["per_edge"]:
            assert "edge" in edge
            assert "atp" in edge
            assert "atp_star" in edge
            assert edge["atp_star"] >= 0.0


def test_run_atp_star_metadata_has_cancellation_ratio(circuit_results):
    for r in circuit_results:
        assert "cancellation_ratio" in r.metadata
        assert r.metadata["cancellation_ratio"] >= 1.0


def test_run_atp_star_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0
