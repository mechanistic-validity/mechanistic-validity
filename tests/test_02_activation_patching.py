import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "scm_pearl" / "02_activation_patching.py"
)
_spec = importlib.util.spec_from_file_location("activation_patching_02", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["activation_patching_02"] = _mod
_spec.loader.exec_module(_mod)

run_activation_patching = _mod.run_activation_patching
patch_head_effect = _mod.patch_head_effect

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_activation_patching(gpt2_model, tasks=[TASK], n_prompts=3, n_random_baselines=10)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C2.activation_patching"


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "per_head_effects", "top5_non_circuit",
                "random_std", "n_circuit_heads", "circuit_heads"}
    assert set(meta.keys()) == expected


def test_per_head_effects_has_circuit_heads(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["per_head_effects"]) == meta["n_circuit_heads"]


def test_top5_non_circuit_max_5(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["top5_non_circuit"]) <= 5


def test_baseline_random_exists(circuit_results):
    assert circuit_results[0].baseline_random is not None


def test_value_is_circuit_sum(circuit_results):
    r = circuit_results[0]
    expected_sum = sum(r.metadata["per_head_effects"].values())
    assert r.value == pytest.approx(expected_sum)


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_circuit_score_exceeds_random_baseline(circuit_results):
    r = circuit_results[0]
    assert r.value > r.baseline_random


def test_task_metadata(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK
