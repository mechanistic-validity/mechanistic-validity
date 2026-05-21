import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "causal" / "scm_pearl" / "role_ablation.py"
)
_spec = importlib.util.spec_from_file_location("role_ablation", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["role_ablation"] = _mod
_spec.loader.exec_module(_mod)

run_role_ablation = _mod.run_role_ablation

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def results(gpt2_model):
    return run_role_ablation(gpt2_model, [TASK], n_prompts=5)


def test_returns_results(results):
    assert len(results) >= 1


def test_result_type(results):
    for r in results:
        assert isinstance(r, EvalResult)


def test_metric_id(results):
    for r in results:
        assert r.metric_id == "role_ablation"


def test_value_is_finite(results):
    import math
    for r in results:
        assert math.isfinite(r.value)


def test_metadata_has_task(results):
    for r in results:
        assert r.metadata["task"] == TASK


def test_n_samples_positive(results):
    for r in results:
        assert r.n_samples > 0


def test_full_scan_produces_multiple_roles(results):
    assert len(results) >= 2


def test_metadata_has_clean_and_ablated(results):
    for r in results:
        assert "clean_mean" in r.metadata
        assert "ablated_mean" in r.metadata


def test_ablation_effect_nonzero_for_some_role(results):
    effects = [abs(r.value) for r in results]
    assert max(effects) > 0.01


def test_targeted_ablation(gpt2_model):
    results = run_role_ablation(
        gpt2_model, [TASK], n_prompts=5,
        intervention_target="name_mover",
        measurement_target="output",
    )
    if results:
        r = results[0]
        assert r.metric_id == "role_ablation"
        assert r.metadata["intervention_target"] == "name_mover"
        assert r.metadata["measurement_target"] == "output"


def test_unknown_task_returns_empty(gpt2_model):
    results = run_role_ablation(gpt2_model, ["nonexistent_xyz"], n_prompts=5)
    assert results == []
