import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "causal" / "rubin_cate" / "25_intervention_specificity.py"
)
_spec = importlib.util.spec_from_file_location("intervention_specificity_25", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["intervention_specificity_25"] = _mod
_spec.loader.exec_module(_mod)

run_intervention_specificity = _mod.run_intervention_specificity
compute_ablation_effect = _mod.compute_ablation_effect

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_intervention_specificity(gpt2_model, tasks=[TASK, "greater_than"], n_prompts=4)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C25.intervention_specificity"


def test_specificity_positive(circuit_results):
    for r in circuit_results:
        assert r.value > 0.0


def test_metadata_keys(circuit_results):
    for r in circuit_results:
        meta = r.metadata
        expected = {"task", "target_effect", "mean_nontarget_effect",
                    "nontarget_effects", "specificity", "n_circuit_heads",
                    "n_nontarget_tasks"}
        assert set(meta.keys()) == expected


def test_specificity_equals_ratio(circuit_results):
    for r in circuit_results:
        meta = r.metadata
        if meta["mean_nontarget_effect"] > 1e-8:
            expected = meta["target_effect"] / meta["mean_nontarget_effect"]
            assert r.value == pytest.approx(expected)


def test_nontarget_effects_excludes_self(circuit_results):
    for r in circuit_results:
        meta = r.metadata
        assert meta["task"] not in meta["nontarget_effects"]


def test_n_samples_positive(circuit_results):
    for r in circuit_results:
        assert r.n_samples > 0


def test_needs_at_least_two_tasks(gpt2_model):
    results = run_intervention_specificity(gpt2_model, tasks=[TASK], n_prompts=3)
    assert len(results) == 0
