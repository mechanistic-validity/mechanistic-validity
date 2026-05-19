import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "metrics"
    / "causal"
    / "counterfactual_das"
    / "DAS_iia_task.py"
)
_spec = importlib.util.spec_from_file_location("das_iia_task", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["das_iia_task"] = _mod
_spec.loader.exec_module(_mod)

make_counterfactual_pairs = _mod.make_counterfactual_pairs
compute_pca_directions = _mod.compute_pca_directions
compute_iia = _mod.compute_iia
run_das_iia = _mod.run_das_iia

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_make_counterfactual_pairs_correct_length():
    n = 10
    correct_ids = list(range(n))
    incorrect_ids = list(range(n, 2 * n))
    prompts = [None] * n
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids)
    assert len(pairs) == n


def test_make_counterfactual_pairs_no_self_pairing():
    n = 8
    correct_ids = [0, 1, 0, 1, 0, 1, 0, 1]
    incorrect_ids = list(range(n))
    prompts = [None] * n
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids)
    for i, j in pairs:
        assert i != j


def test_make_counterfactual_pairs_prefers_different_targets():
    n = 6
    correct_ids = [0, 0, 0, 1, 1, 1]
    incorrect_ids = list(range(n))
    prompts = [None] * n
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids)
    for i, j in pairs:
        assert correct_ids[j] != correct_ids[i]


def test_make_counterfactual_pairs_all_same_still_works():
    n = 5
    correct_ids = [42] * n
    incorrect_ids = list(range(n))
    prompts = [None] * n
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids)
    assert len(pairs) == n
    for i, j in pairs:
        assert i != j


def test_compute_pca_directions_shape():
    d_head = 16
    n = 10
    base_acts = [torch.randn(d_head) for _ in range(n)]
    cf_acts = [torch.randn(d_head) for _ in range(n)]
    pairs = [(i, (i + 1) % n) for i in range(n)]

    directions = compute_pca_directions(base_acts, cf_acts, pairs, k_components=3)
    assert directions.shape == (d_head, 3)


def test_compute_pca_directions_orthonormal_columns():
    d_head = 16
    n = 20
    base_acts = [torch.randn(d_head) for _ in range(n)]
    cf_acts = [torch.randn(d_head) for _ in range(n)]
    pairs = [(i, (i + 1) % n) for i in range(n)]

    directions = compute_pca_directions(base_acts, cf_acts, pairs, k_components=3)
    DtD = directions.T @ directions
    assert DtD == pytest.approx(torch.eye(3), abs=1e-5)


def test_compute_pca_directions_k_larger_than_n():
    d_head = 16
    n = 2
    base_acts = [torch.randn(d_head) for _ in range(n)]
    cf_acts = [torch.randn(d_head) for _ in range(n)]
    pairs = [(0, 1)]

    directions = compute_pca_directions(base_acts, cf_acts, pairs, k_components=10)
    assert directions.shape[0] == d_head
    assert directions.shape[1] <= min(10, 1, d_head)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_das_iia(gpt2_model, tasks=[TASK], n_prompts=6)


def test_run_das_iia_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_das_iia_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "DAS.iia"


def test_run_das_iia_value_in_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0


def test_run_das_iia_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_das_iia_metadata_has_layer_info(circuit_results):
    for r in circuit_results:
        assert "layer_intervened" in r.metadata
        assert isinstance(r.metadata["layer_intervened"], int)
        assert "head_intervened" in r.metadata
        assert isinstance(r.metadata["head_intervened"], int)


def test_run_das_iia_metadata_has_accuracies(circuit_results):
    for r in circuit_results:
        assert "base_accuracy" in r.metadata
        assert 0.0 <= r.metadata["base_accuracy"] <= 1.0
        assert "intervention_accuracy" in r.metadata
        assert 0.0 <= r.metadata["intervention_accuracy"] <= 1.0


def test_run_das_iia_metadata_has_n_components(circuit_results):
    for r in circuit_results:
        assert "n_components_used" in r.metadata
        assert r.metadata["n_components_used"] >= 1


def test_run_das_iia_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0
