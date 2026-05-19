import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "mdl_slt" / "29_hyperparam_sensitivity.py"
)
_spec = importlib.util.spec_from_file_location("hyperparam_29", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hyperparam_29"] = _mod
_spec.loader.exec_module(_mod)

run_hyperparam_sensitivity = _mod.run_hyperparam_sensitivity
faithfulness_with_ablation_type = _mod.faithfulness_with_ablation_type
N_PROMPTS_SETTINGS = _mod.N_PROMPTS_SETTINGS
ABLATION_SETTINGS = _mod.ABLATION_SETTINGS

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_n_prompts_settings():
    assert N_PROMPTS_SETTINGS == [20, 40, 80]


def test_ablation_settings():
    assert ABLATION_SETTINGS == ["zero", "mean", "mean_last"]


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_hyperparam_sensitivity(gpt2_model, tasks=[TASK], n_prompts=5)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C29.hyperparam_sensitivity"


def test_value_is_cv(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["cv"])


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "setting_scores", "mean_faithfulness",
                "std_faithfulness", "cv", "n_circuit_heads",
                "n_settings_tested"}
    assert set(meta.keys()) == expected


def test_setting_scores_has_entries(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["setting_scores"]) >= 3


def test_cv_non_negative(circuit_results):
    assert circuit_results[0].value >= 0.0


def test_n_samples_equals_settings_count(circuit_results):
    r = circuit_results[0]
    assert r.n_samples == r.metadata["n_settings_tested"]


def test_setting_scores_all_finite(circuit_results):
    for key, val in circuit_results[0].metadata["setting_scores"].items():
        assert np.isfinite(val), f"Setting {key} has non-finite value"
