import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "measurement" / "signal_detection" / "EX1_dprime.py"
)
_spec = importlib.util.spec_from_file_location("EX1_dprime", _SRC_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["EX1_dprime"] = _mod
_spec.loader.exec_module(_mod)

run_dprime = _mod.run_dprime

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def dprime_results(gpt2_model):
    return run_dprime(gpt2_model, tasks=[TASK], n_prompts=5)


def test_returns_eval_result_list(dprime_results):
    assert len(dprime_results) == 1
    assert isinstance(dprime_results[0], EvalResult)


def test_metric_id(dprime_results):
    assert dprime_results[0].metric_id == "EX1.dprime"


def test_value_is_finite(dprime_results):
    assert math.isfinite(dprime_results[0].value)


def test_n_samples_positive(dprime_results):
    assert dprime_results[0].n_samples > 0
    assert dprime_results[0].n_samples <= 5


def test_metadata_has_expected_keys(dprime_results):
    meta = dprime_results[0].metadata
    expected = {
        "task", "n_heads", "hit_rate", "false_alarm_rate",
        "dprime", "criterion", "auc",
        "signal_mean_ld", "signal_std_ld",
        "noise_mean_ld", "noise_std_ld",
        "passed", "dprime_threshold", "auc_threshold",
    }
    assert expected.issubset(set(meta.keys()))


def test_hit_rate_in_range(dprime_results):
    hr = dprime_results[0].metadata["hit_rate"]
    assert 0.0 <= hr <= 1.0


def test_false_alarm_rate_in_range(dprime_results):
    fa = dprime_results[0].metadata["false_alarm_rate"]
    assert 0.0 <= fa <= 1.0


def test_auc_in_range(dprime_results):
    auc = dprime_results[0].metadata["auc"]
    assert 0.0 <= auc <= 1.0


def test_value_equals_dprime_in_metadata(dprime_results):
    assert dprime_results[0].value == pytest.approx(
        dprime_results[0].metadata["dprime"])


def test_passed_is_bool(dprime_results):
    assert isinstance(dprime_results[0].metadata["passed"], bool)
