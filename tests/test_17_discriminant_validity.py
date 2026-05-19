import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "calibrations" / "discriminant_validity" / "17_discriminant_validity.py"
)
_spec = importlib.util.spec_from_file_location("_dv_17", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_patching_effects = _mod.compute_patching_effects
run_discriminant_validity = _mod.run_discriminant_validity

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_discriminant_validity(
        gpt2_model, ["ioi", "greater_than"], n_prompts=3, n_random_baselines=2
    )


def test_run_discriminant_validity_needs_two_tasks(gpt2_model):
    results = run_discriminant_validity(gpt2_model, [TASK], n_prompts=3,
                                       n_random_baselines=2)
    assert results == []


def test_run_discriminant_validity_returns_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_discriminant_validity_has_matrix_result(circuit_results):
    matrix_results = [r for r in circuit_results
                      if r.metric_id == "C17.discriminant_matrix"]
    assert len(matrix_results) == 1
    mat = matrix_results[0]
    assert "effect_matrix" in mat.metadata
    assert "random_baselines" in mat.metadata


def test_discriminant_validity_per_task_results(circuit_results):
    per_task = [r for r in circuit_results
                if r.metric_id == "C17.discriminant_validity"]
    assert len(per_task) == 2
    tasks_seen = {r.metadata["task"] for r in per_task}
    assert tasks_seen == {"ioi", "greater_than"}


def test_discriminant_ratio_metadata(circuit_results):
    per_task = [r for r in circuit_results
                if r.metric_id == "C17.discriminant_validity"]
    for r in per_task:
        assert "diagonal_effect" in r.metadata
        assert "off_diagonal_mean" in r.metadata
        assert "effect_row" in r.metadata
        assert r.metadata["n_circuit_heads"] > 0


def test_discriminant_matrix_is_square(circuit_results):
    mat_result = [r for r in circuit_results
                  if r.metric_id == "C17.discriminant_matrix"][0]
    effect_matrix = mat_result.metadata["effect_matrix"]
    tasks = mat_result.metadata["tasks"]
    for src in tasks:
        assert set(effect_matrix[src].keys()) == set(tasks)
