import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "subnetwork_probe" / "48_subnetwork_probe.py"
)
_spec = importlib.util.spec_from_file_location("subnetwork_probe_48", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["subnetwork_probe_48"] = _mod
_spec.loader.exec_module(_mod)

logistic_probe_accuracy = _mod.logistic_probe_accuracy
run_subnetwork_probe = _mod.run_subnetwork_probe

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_logistic_probe_accuracy_perfectly_separable():
    features = np.array([
        [10.0, 0.0], [9.0, 0.0], [8.0, 0.0],
        [-10.0, 0.0], [-9.0, 0.0], [-8.0, 0.0],
    ])
    labels = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    acc = logistic_probe_accuracy(features, labels)
    assert acc == pytest.approx(1.0)


def test_logistic_probe_accuracy_random_features():
    rng = np.random.RandomState(42)
    features = rng.randn(50, 10)
    labels = rng.randint(0, 2, 50).astype(float)
    acc = logistic_probe_accuracy(features, labels)
    assert 0.0 <= acc <= 1.0


def test_logistic_probe_accuracy_too_few_samples():
    features = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([1.0, 0.0, 1.0])
    acc = logistic_probe_accuracy(features, labels)
    assert 0.0 <= acc <= 1.0


def test_logistic_probe_accuracy_degenerate():
    features = np.array([[1.0], [1.0]])
    labels = np.array([0.0, 1.0])
    acc = logistic_probe_accuracy(features, labels)
    assert acc == pytest.approx(0.5)


def test_logistic_probe_accuracy_zero_dim():
    features = np.zeros((5, 0))
    labels = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    acc = logistic_probe_accuracy(features, labels)
    assert acc == pytest.approx(0.5)


def test_logistic_probe_accuracy_all_same_features():
    features = np.ones((10, 5))
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    acc = logistic_probe_accuracy(features, labels)
    assert 0.0 <= acc <= 1.0


def test_run_subnetwork_probe_returns_results(gpt2_model):
    results = run_subnetwork_probe(gpt2_model, [TASK], n_prompts=5, n_random=2)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D09.subnetwork_probe"
    assert r.n_samples >= 1


def test_run_subnetwork_probe_metadata_keys(gpt2_model):
    results = run_subnetwork_probe(gpt2_model, [TASK], n_prompts=5, n_random=2)
    r = results[0]
    expected_keys = {
        "task", "circuit_probe_accuracy", "random_mean_accuracy",
        "random_std_accuracy", "advantage", "z_score",
        "n_circuit_heads", "n_random_baselines", "n_prompts",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_subnetwork_probe_accuracy_bounded(gpt2_model):
    results = run_subnetwork_probe(gpt2_model, [TASK], n_prompts=5, n_random=2)
    r = results[0]
    assert 0.0 <= r.value <= 1.0
    assert 0.0 <= r.baseline_random <= 1.0


def test_run_subnetwork_probe_value_matches_circuit_accuracy(gpt2_model):
    results = run_subnetwork_probe(gpt2_model, [TASK], n_prompts=5, n_random=2)
    r = results[0]
    assert r.value == pytest.approx(r.metadata["circuit_probe_accuracy"])


def test_run_subnetwork_probe_unknown_task_returns_empty(gpt2_model):
    results = run_subnetwork_probe(gpt2_model, ["nonexistent_task_xyz"], n_prompts=5, n_random=2)
    assert results == []
