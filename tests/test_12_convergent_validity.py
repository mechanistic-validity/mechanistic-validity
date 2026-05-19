import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

from mechval.metrics.common import EvalResult

_CV_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechval"
    / "calibrations"
    / "convergent_validity"
    / "12_convergent_validity.py"
)
_spec = importlib.util.spec_from_file_location("_cv12", _CV_PATH)
_cv_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cv_mod
_spec.loader.exec_module(_cv_mod)

compute_convergent_validity = _cv_mod.compute_convergent_validity
load_metric_scores = _cv_mod.load_metric_scores
run_convergent_validity = _cv_mod.run_convergent_validity

TASK = "ioi"


def test_compute_convergent_validity_with_mocked_scores(monkeypatch):
    scores_a = {"L0H0": 1.0, "L1H1": 2.0, "L2H2": 3.0, "L3H3": 4.0}
    scores_b = {"L0H0": 1.5, "L1H1": 2.5, "L2H2": 3.5, "L3H3": 4.5}
    scores_c = {"L0H0": 0.5, "L1H1": 1.5, "L2H2": 2.5, "L3H3": 3.5}

    call_count = {"n": 0}
    fake_scores = {"02": scores_a, "07": scores_b, "10": scores_c}

    def mock_load(metric_file, task):
        for key, fname in _cv_mod.METRIC_FILES.items():
            if metric_file == fname and key in fake_scores:
                return fake_scores[key]
        return None

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07", "10"])
    assert result is not None
    assert isinstance(result, EvalResult)
    assert result.metric_id == "C12.convergent_validity"
    assert result.n_samples == 4
    assert result.metadata["task"] == TASK
    assert len(result.metadata["spearman_pairs"]) == 3
    assert result.value == pytest.approx(
        np.mean([v["rho"] for v in result.metadata["spearman_pairs"].values()])
    )


def test_compute_convergent_validity_perfect_agreement(monkeypatch):
    scores = {"L0H0": 1.0, "L1H1": 2.0, "L2H2": 3.0, "L3H3": 4.0}

    def mock_load(metric_file, task):
        return scores.copy()

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07"])
    assert result is not None
    assert result.value == pytest.approx(1.0)


def test_compute_convergent_validity_anti_correlated(monkeypatch):
    scores_a = {"L0H0": 1.0, "L1H1": 2.0, "L2H2": 3.0, "L3H3": 4.0}
    scores_b = {"L0H0": 4.0, "L1H1": 3.0, "L2H2": 2.0, "L3H3": 1.0}

    toggle = {"call": 0}

    def mock_load(metric_file, task):
        toggle["call"] += 1
        if toggle["call"] % 2 == 1:
            return scores_a.copy()
        return scores_b.copy()

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07"])
    assert result is not None
    assert result.value == pytest.approx(-1.0)


def test_compute_convergent_validity_insufficient_metrics(monkeypatch):
    def mock_load(metric_file, task):
        return None

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07"])
    assert result is None


def test_compute_convergent_validity_one_metric_only(monkeypatch):
    def mock_load(metric_file, task):
        if "02" in metric_file:
            return {"L0H0": 1.0, "L1H1": 2.0}
        return None

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07"])
    assert result is None


def test_compute_convergent_validity_disjoint_heads(monkeypatch):
    scores_a = {"L0H0": 1.0, "L1H1": 2.0}
    scores_b = {"L2H2": 3.0, "L3H3": 4.0}

    def mock_load(metric_file, task):
        if "02" in metric_file:
            return scores_a
        if "07" in metric_file:
            return scores_b
        return None

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    result = compute_convergent_validity(TASK, metrics=["02", "07"])
    assert result is not None
    assert result.n_samples == 4


def test_run_convergent_validity_returns_list(monkeypatch):
    scores = {"L0H0": 1.0, "L1H1": 2.0, "L2H2": 3.0}

    def mock_load(metric_file, task):
        return scores.copy()

    monkeypatch.setattr(_cv_mod, "load_metric_scores", mock_load)

    results = run_convergent_validity([TASK], metrics=["02", "07"])
    assert len(results) == 1
    assert results[0].metadata["task"] == TASK
