import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "woodward" / "03_sigma_ablation.py"
)
_spec = importlib.util.spec_from_file_location("sigma_ablation_03", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sigma_ablation_03"] = _mod
_spec.loader.exec_module(_mod)

run_sigma_ablation = _mod.run_sigma_ablation
faithfulness_with_ablation = _mod.faithfulness_with_ablation
cache_all_z = _mod.cache_all_z
ABLATION_METHODS = _mod.ABLATION_METHODS

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_sigma_ablation(gpt2_model, tasks=[TASK], n_prompts=3)


def test_ablation_methods_count():
    assert len(ABLATION_METHODS) == 8


def test_ablation_methods_contains_zero():
    assert "zero" in ABLATION_METHODS


def test_ablation_methods_contains_mean():
    assert "mean" in ABLATION_METHODS


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C3.sigma_ablation"


def test_value_is_cv(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    expected_cv = meta["std_faithfulness"] / abs(meta["mean_faithfulness"]) if abs(meta["mean_faithfulness"]) > 1e-8 else float("inf")
    assert r.value == pytest.approx(expected_cv)


def test_metadata_has_method_scores(circuit_results):
    meta = circuit_results[0].metadata
    assert "method_scores" in meta
    assert len(meta["method_scores"]) == 8


def test_metadata_task(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_cv_is_non_negative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_method_scores_all_finite(circuit_results):
    for method, score in circuit_results[0].metadata["method_scores"].items():
        assert np.isfinite(score), f"{method} has non-finite score"
