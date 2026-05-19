import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "calibrations" / "bootstrap_stability" / "30_seed_variance.py"
)
_spec = importlib.util.spec_from_file_location("_seed_var_30", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_subsample_prompts = _mod._subsample_prompts
run_seed_variance = _mod.run_seed_variance

TASK = "ioi"


def test_subsample_fewer_than_n_returns_all():
    prompts = ["a", "b", "c"]
    correct = [1, 2, 3]
    incorrect = [4, 5, 6]
    sp, sc, si = _subsample_prompts(prompts, correct, incorrect, n=10, seed=42)
    assert sp == prompts
    assert sc == correct
    assert si == incorrect


def test_subsample_returns_correct_count():
    prompts = list(range(20))
    correct = list(range(20))
    incorrect = list(range(20))
    sp, sc, si = _subsample_prompts(prompts, correct, incorrect, n=5, seed=42)
    assert len(sp) == 5
    assert len(sc) == 5
    assert len(si) == 5


def test_subsample_different_seeds_give_different_results():
    prompts = list(range(100))
    correct = list(range(100))
    incorrect = list(range(100))
    sp1, _, _ = _subsample_prompts(prompts, correct, incorrect, n=10, seed=42)
    sp2, _, _ = _subsample_prompts(prompts, correct, incorrect, n=10, seed=123)
    assert sp1 != sp2


def test_subsample_indices_are_sorted():
    prompts = list(range(50))
    correct = list(range(50))
    incorrect = list(range(50))
    sp, _, _ = _subsample_prompts(prompts, correct, incorrect, n=10, seed=42)
    assert sp == sorted(sp)


def test_subsample_preserves_alignment():
    prompts = [f"p{i}" for i in range(30)]
    correct = [i * 10 for i in range(30)]
    incorrect = [i * 100 for i in range(30)]
    sp, sc, si = _subsample_prompts(prompts, correct, incorrect, n=5, seed=99)
    for p, c, inc in zip(sp, sc, si):
        idx = int(p[1:])
        assert c == idx * 10
        assert inc == idx * 100


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_seed_variance(gpt2_model, [TASK], n_prompts=3)


def test_run_seed_variance_returns_eval_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_seed_variance_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C30.seed_variance"


def test_run_seed_variance_metadata_keys(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert "cv" in r.metadata
        assert "mean_faithfulness" in r.metadata
        assert "std_faithfulness" in r.metadata
        assert "seed_scores" in r.metadata
        assert "range" in r.metadata


def test_run_seed_variance_cv_is_nonnegative(circuit_results):
    for r in circuit_results:
        assert r.value >= 0.0


def test_run_seed_variance_n_samples_equals_n_seeds(circuit_results):
    for r in circuit_results:
        assert r.n_samples == len(_mod.SEEDS)
