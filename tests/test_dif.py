import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "measurement" / "psychometrics" / "EX2_dif.py"
)
_spec = importlib.util.spec_from_file_location("EX2_dif", _SRC_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["EX2_dif"] = _mod
_spec.loader.exec_module(_mod)

run_dif = _mod.run_dif

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def dif_results(gpt2_model):
    return run_dif(gpt2_model, tasks=[TASK], n_prompts=5)


def test_returns_eval_result_list(dif_results):
    assert len(dif_results) == 1
    assert isinstance(dif_results[0], EvalResult)


def test_metric_id(dif_results):
    assert dif_results[0].metric_id == "EX2.dif"


def test_value_is_finite(dif_results):
    assert math.isfinite(dif_results[0].value)


def test_value_in_zero_one(dif_results):
    assert 0.0 <= dif_results[0].value <= 1.0


def test_n_samples_positive(dif_results):
    assert dif_results[0].n_samples > 0


def test_metadata_has_expected_keys(dif_results):
    meta = dif_results[0].metadata
    expected = {
        "task", "n_heads", "name_groups", "per_group_stats",
        "pairwise_dif", "max_dif_cohens_d", "passed", "threshold",
    }
    assert expected.issubset(set(meta.keys()))


def test_name_groups_has_three_groups(dif_results):
    groups = dif_results[0].metadata["name_groups"]
    assert len(groups) == 3
    assert "common" in groups
    assert "uncommon" in groups
    assert "diverse_origin" in groups


def test_per_group_stats_has_all_groups(dif_results):
    meta = dif_results[0].metadata
    for g in meta["name_groups"]:
        assert g in meta["per_group_stats"]


def test_passed_is_bool(dif_results):
    assert isinstance(dif_results[0].metadata["passed"], bool)


def test_max_dif_is_non_negative(dif_results):
    assert dif_results[0].metadata["max_dif_cohens_d"] >= 0.0
