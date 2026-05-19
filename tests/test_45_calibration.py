import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "behavioral" / "calibration" / "45_calibration.py"
)
_spec = importlib.util.spec_from_file_location("calibration_45", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["calibration_45"] = _mod
_spec.loader.exec_module(_mod)

compute_ece = _mod.compute_ece
run_calibration = _mod.run_calibration

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_ece_perfectly_calibrated():
    probs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    correct = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ece, bins = compute_ece(probs, correct)
    assert ece == pytest.approx(0.0, abs=0.15)


def test_compute_ece_worst_case():
    probs = [0.95, 0.95, 0.95, 0.95, 0.95]
    correct = [0, 0, 0, 0, 0]
    ece, bins = compute_ece(probs, correct)
    assert ece == pytest.approx(0.95, abs=0.01)


def test_compute_ece_all_correct_high_confidence():
    probs = [0.95, 0.92, 0.98, 0.91, 0.96]
    correct = [1, 1, 1, 1, 1]
    ece, bins = compute_ece(probs, correct)
    assert ece < 0.1


def test_compute_ece_returns_bin_details():
    probs = [0.1, 0.2, 0.8, 0.9]
    correct = [0, 0, 1, 1]
    ece, bins = compute_ece(probs, correct)
    assert len(bins) == 10
    for b in bins:
        assert "bin" in b
        assert "n" in b
        assert "mean_conf" in b
        assert "accuracy" in b


def test_compute_ece_empty_input():
    ece, bins = compute_ece([], [])
    assert ece == pytest.approx(0.0)


def test_compute_ece_single_sample():
    ece, bins = compute_ece([0.7], [1])
    assert isinstance(ece, float)
    assert ece >= 0.0


def test_compute_ece_is_nonnegative():
    rng = np.random.RandomState(42)
    probs = rng.uniform(0, 1, 100).tolist()
    correct = rng.randint(0, 2, 100).tolist()
    ece, _ = compute_ece(probs, correct)
    assert ece >= 0.0


def test_run_calibration_returns_results(gpt2_model):
    results = run_calibration(gpt2_model, [TASK], n_prompts=3)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "D06.calibration_ece"
    assert r.n_samples >= 1


def test_run_calibration_metadata_keys(gpt2_model):
    results = run_calibration(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    expected_keys = {
        "task", "ece_full", "ece_circuit_only", "ece_ratio",
        "accuracy_full", "accuracy_circuit_only",
        "mean_prob_full", "mean_prob_circuit",
        "n_circuit_heads", "bins_full", "bins_circuit",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_calibration_ece_nonnegative(gpt2_model):
    results = run_calibration(gpt2_model, [TASK], n_prompts=3)
    r = results[0]
    assert r.value >= 0.0
    assert r.metadata["ece_full"] >= 0.0


def test_run_calibration_unknown_task_returns_empty(gpt2_model):
    results = run_calibration(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3)
    assert results == []
