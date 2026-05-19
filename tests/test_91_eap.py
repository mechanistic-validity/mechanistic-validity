import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "causal"
    / "eap"
    / "91_eap.py"
)
_spec = importlib.util.spec_from_file_location("eap_91", _EAP_PATH)
_eap_mod = importlib.util.module_from_spec(_spec)
sys.modules["eap_91"] = _eap_mod
_spec.loader.exec_module(_eap_mod)

compute_auroc = _eap_mod.compute_auroc
compute_eap_scores = _eap_mod.compute_eap_scores
run_eap = _eap_mod.run_eap

from mechval.metrics.common import EvalResult, load_model


def test_compute_auroc_perfect_discrimination():
    n_layers = 3
    n_heads = 2
    n_total = n_layers * n_heads

    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)

    circuit_edges = {(0, 0, 1, 0), (0, 1, 2, 0)}

    for Ls, Hs, Lr, Hr in circuit_edges:
        s_idx = Ls * n_heads + Hs
        r_idx = Lr * n_heads + Hr
        edge_scores[s_idx, r_idx] = 10.0

    auroc, stats = compute_auroc(edge_scores, circuit_edges, n_layers, n_heads)

    assert auroc == pytest.approx(1.0)
    assert stats["n_circuit_edges"] == 2
    assert stats["n_non_circuit_edges"] > 0
    assert stats["mean_circuit_score"] == pytest.approx(10.0)
    assert stats["mean_non_circuit_score"] == pytest.approx(0.0)


def test_compute_auroc_no_discrimination():
    n_layers = 4
    n_heads = 2
    n_total = n_layers * n_heads

    edge_scores = np.ones((n_total, n_total), dtype=np.float64) * 5.0

    circuit_edges = {(0, 0, 1, 0), (1, 1, 3, 0)}

    auroc, stats = compute_auroc(edge_scores, circuit_edges, n_layers, n_heads)

    assert auroc == pytest.approx(0.5)
    assert stats["n_circuit_edges"] == 2
    assert stats["mean_circuit_score"] == pytest.approx(5.0)
    assert stats["mean_non_circuit_score"] == pytest.approx(5.0)


def test_compute_auroc_uses_absolute_scores():
    n_layers = 3
    n_heads = 2
    n_total = n_layers * n_heads

    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)

    circuit_edges = {(0, 0, 1, 0)}
    s_idx = 0 * n_heads + 0
    r_idx = 1 * n_heads + 0
    edge_scores[s_idx, r_idx] = -10.0

    auroc, stats = compute_auroc(edge_scores, circuit_edges, n_layers, n_heads)

    assert auroc == pytest.approx(1.0)
    assert stats["mean_circuit_score"] == pytest.approx(10.0)


def test_compute_auroc_only_forward_edges():
    n_layers = 2
    n_heads = 2
    n_total = n_layers * n_heads

    edge_scores = np.zeros((n_total, n_total), dtype=np.float64)
    circuit_edges = {(0, 0, 1, 0), (0, 1, 1, 1)}

    for Ls, Hs, Lr, Hr in circuit_edges:
        s_idx = Ls * n_heads + Hs
        r_idx = Lr * n_heads + Hr
        edge_scores[s_idx, r_idx] = 5.0

    auroc, stats = compute_auroc(edge_scores, circuit_edges, n_layers, n_heads)

    total_forward_edges = 0
    for Ls in range(n_layers):
        for Hs in range(n_heads):
            for Lr in range(Ls + 1, n_layers):
                for Hr in range(n_heads):
                    total_forward_edges += 1

    assert stats["n_circuit_edges"] + stats["n_non_circuit_edges"] == total_forward_edges


def test_compute_auroc_degenerate_all_positive():
    n_layers = 2
    n_heads = 1
    n_total = n_layers * n_heads

    edge_scores = np.ones((n_total, n_total), dtype=np.float64)

    circuit_edges = {(0, 0, 1, 0)}

    auroc, stats = compute_auroc(edge_scores, circuit_edges, n_layers, n_heads)

    assert stats["n_circuit"] == 1
    assert stats["n_total"] == 1
    assert auroc == pytest.approx(0.0)


def test_edge_scores_shape():
    model = load_model("gpt2", "cpu")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    from mechval.metrics.common import generate_prompts, get_token_ids

    prompts = generate_prompts(TASK, model.tokenizer, n_prompts=2)
    correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)

    edge_scores = compute_eap_scores(model, prompts, correct_ids, incorrect_ids)

    assert edge_scores.shape == (n_total, n_total)
    assert edge_scores.dtype == np.float64


TASK = "ioi"


def test_run_eap_returns_valid_result():
    model = load_model("gpt2", "cpu")

    results = run_eap(model, tasks=[TASK], n_prompts=5)

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C7.eap_auroc"
    assert 0.0 <= r.value <= 1.0
    assert r.n_samples == 5

    meta = r.metadata
    assert meta["task"] == TASK
    assert "auroc" in meta
    assert meta["auroc"] == pytest.approx(r.value)
    assert "passed" in meta
    assert isinstance(meta["passed"], bool)
    assert meta["threshold"] == pytest.approx(0.70)
    assert "n_circuit_edges" in meta
    assert "n_non_circuit_edges" in meta
    assert "mean_circuit_score" in meta
    assert "mean_non_circuit_score" in meta
    assert "median_circuit_score" in meta
    assert "median_non_circuit_score" in meta
    assert "top_edges" in meta
    assert isinstance(meta["top_edges"], list)
