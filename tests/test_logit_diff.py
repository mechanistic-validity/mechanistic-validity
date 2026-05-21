import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "causal" / "scm_pearl" / "logit_diff.py"
)
_spec = importlib.util.spec_from_file_location("logit_diff", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["logit_diff"] = _mod
_spec.loader.exec_module(_mod)

run_logit_diff = _mod.run_logit_diff

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def results(gpt2_model):
    return run_logit_diff(gpt2_model, [TASK], n_prompts=5)


def test_returns_results(results):
    assert len(results) >= 1


def test_result_type(results):
    assert isinstance(results[0], EvalResult)


def test_metric_id(results):
    assert results[0].metric_id == "logit_diff"


def test_value_is_finite(results):
    import math
    assert math.isfinite(results[0].value)


def test_metadata_has_task(results):
    assert results[0].metadata["task"] == TASK


def test_n_samples_positive(results):
    assert results[0].n_samples > 0


def test_metadata_has_expected_keys(results):
    expected = {"task", "per_prompt_diffs", "min", "max"}
    assert expected.issubset(results[0].metadata.keys())


def test_per_prompt_diffs_length_matches_n_samples(results):
    diffs = results[0].metadata["per_prompt_diffs"]
    assert len(diffs) == results[0].n_samples


def test_min_max_consistent_with_diffs(results):
    diffs = results[0].metadata["per_prompt_diffs"]
    assert results[0].metadata["min"] == pytest.approx(min(diffs))
    assert results[0].metadata["max"] == pytest.approx(max(diffs))


def test_value_is_mean_of_diffs(results):
    diffs = results[0].metadata["per_prompt_diffs"]
    expected_mean = sum(diffs) / len(diffs)
    assert results[0].value == pytest.approx(expected_mean)


def test_ioi_logit_diff_positive(results):
    assert results[0].value > 0


def test_unknown_task_returns_empty(gpt2_model):
    results = run_logit_diff(gpt2_model, ["nonexistent_xyz"], n_prompts=5)
    assert results == []
