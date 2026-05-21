import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "measurement" / "psychophysics" / "EX11_weber_fechner.py"
)
_spec = importlib.util.spec_from_file_location("EX11_weber_fechner", _SRC_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["EX11_weber_fechner"] = _mod
_spec.loader.exec_module(_mod)

run_weber_fechner = _mod.run_weber_fechner

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def weber_results(gpt2_model):
    return run_weber_fechner(gpt2_model, tasks=[TASK], n_prompts=5)


def test_returns_eval_result_list(weber_results):
    assert len(weber_results) == 1
    assert isinstance(weber_results[0], EvalResult)


def test_metric_id(weber_results):
    assert weber_results[0].metric_id == "EX11.weber_fechner"


def test_value_is_finite(weber_results):
    assert math.isfinite(weber_results[0].value)


def test_n_samples_positive(weber_results):
    assert weber_results[0].n_samples > 0
    assert weber_results[0].n_samples <= 5


def test_metadata_has_expected_keys(weber_results):
    meta = weber_results[0].metadata
    expected = {
        "task", "weber_consistency", "all_heads_detectable",
        "n_detectable_heads", "n_circuit_heads",
        "mean_baseline_ld", "mean_reduced_ld",
        "reduced_baseline_scale", "scale_factors",
        "jnd_detection_threshold", "per_head_weber",
        "per_head_curves_full", "passed",
    }
    assert expected.issubset(set(meta.keys()))


def test_value_equals_weber_consistency(weber_results):
    assert weber_results[0].value == pytest.approx(
        weber_results[0].metadata["weber_consistency"])


def test_per_head_weber_not_empty(weber_results):
    assert len(weber_results[0].metadata["per_head_weber"]) > 0


def test_scale_factors_match_module(weber_results):
    from_meta = weber_results[0].metadata["scale_factors"]
    assert from_meta == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]


def test_passed_is_bool(weber_results):
    assert isinstance(weber_results[0].metadata["passed"], bool)


def test_n_detectable_heads_bounded(weber_results):
    meta = weber_results[0].metadata
    assert meta["n_detectable_heads"] <= meta["n_circuit_heads"]
    assert meta["n_detectable_heads"] >= 0
