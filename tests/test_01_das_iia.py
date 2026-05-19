import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "counterfactual_das" / "01_das_iia.py"
)
_spec = importlib.util.spec_from_file_location("das_iia_01", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["das_iia_01"] = _mod
_spec.loader.exec_module(_mod)

make_counterfactual_pairs = _mod.make_counterfactual_pairs
compute_iia_with_rotation = _mod.compute_iia_with_rotation
train_rotation = _mod.train_rotation
random_rotation = _mod.random_rotation
run_das_iia = _mod.run_das_iia

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


def test_make_counterfactual_pairs_returns_correct_length():
    n = 10
    correct_ids = list(range(n))
    incorrect_ids = list(range(n, 2 * n))
    prompts = [None] * n
    rng = np.random.RandomState(42)
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng)
    assert len(pairs) == n


def test_make_counterfactual_pairs_no_self_pairing():
    n = 8
    correct_ids = [0, 1, 0, 1, 0, 1, 0, 1]
    incorrect_ids = list(range(n))
    prompts = [None] * n
    rng = np.random.RandomState(42)
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng)
    for i, j in pairs:
        assert i != j


def test_make_counterfactual_pairs_prefers_different_targets():
    n = 6
    correct_ids = [0, 0, 0, 1, 1, 1]
    incorrect_ids = list(range(n))
    prompts = [None] * n
    rng = np.random.RandomState(42)
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng)
    for i, j in pairs:
        assert correct_ids[j] != correct_ids[i]


def test_make_counterfactual_pairs_all_same_correct_still_works():
    n = 5
    correct_ids = [42] * n
    incorrect_ids = list(range(n))
    prompts = [None] * n
    rng = np.random.RandomState(42)
    pairs = make_counterfactual_pairs(prompts, correct_ids, incorrect_ids, rng)
    assert len(pairs) == n
    for i, j in pairs:
        assert i != j


def test_random_rotation_is_orthogonal():
    d_head = 16
    d_sub = 4
    Q = random_rotation(d_head, d_sub, "cpu")
    assert Q.shape == (d_head, d_sub)
    QtQ = Q.T @ Q
    assert QtQ == pytest.approx(torch.eye(d_sub), abs=1e-5)


def test_random_rotation_different_seeds_give_different_results():
    d_head = 16
    d_sub = 4
    Q1 = random_rotation(d_head, d_sub, "cpu")
    Q2 = random_rotation(d_head, d_sub, "cpu")
    assert not torch.allclose(Q1, Q2)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_das_iia(gpt2_model, tasks=[TASK], n_prompts=4, subspace_dims=[1])


def test_run_das_iia_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_das_iia_metric_id_format(circuit_results):
    for r in circuit_results:
        assert r.metric_id.startswith("C1.das_iia_k")


def test_run_das_iia_value_in_valid_range(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.value <= 1.0


def test_run_das_iia_metadata_has_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_das_iia_metadata_has_per_head_iia(circuit_results):
    for r in circuit_results:
        assert "per_head_iia" in r.metadata
        assert isinstance(r.metadata["per_head_iia"], dict)


def test_run_das_iia_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0
