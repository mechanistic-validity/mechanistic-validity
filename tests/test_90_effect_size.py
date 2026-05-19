import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ES_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "effect_size" / "90_effect_size.py"
)
_spec = importlib.util.spec_from_file_location("effect_size_90", _ES_PATH)
_es_mod = importlib.util.module_from_spec(_spec)
sys.modules["effect_size_90"] = _es_mod
_spec.loader.exec_module(_es_mod)

cohens_d = _es_mod.cohens_d
glass_delta = _es_mod.glass_delta
hedges_g = _es_mod.hedges_g
run_effect_size = _es_mod.run_effect_size

from mechanistic_validity.instruments.common import EvalResult, load_model


# ── Pure math: Cohen's d ─────────────────────────────────────────────

def test_cohens_d_known_values():
    g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    g2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    d = cohens_d(g1, g2)
    mean_diff = np.mean(g1) - np.mean(g2)
    pooled_var = ((4 * np.var(g1, ddof=1) + 4 * np.var(g2, ddof=1)) / 8)
    expected = mean_diff / math.sqrt(pooled_var)
    assert d == pytest.approx(expected)


def test_cohens_d_identical_distributions_is_zero():
    g = np.array([5.0, 10.0, 15.0, 20.0])
    assert cohens_d(g, g.copy()) == pytest.approx(0.0)


def test_cohens_d_sign_flips_with_group_order():
    g1 = np.array([10.0, 20.0, 30.0])
    g2 = np.array([1.0, 2.0, 3.0])
    assert cohens_d(g1, g2) == pytest.approx(-cohens_d(g2, g1))


def test_cohens_d_returns_zero_for_single_element_groups():
    g1 = np.array([5.0])
    g2 = np.array([10.0])
    assert cohens_d(g1, g2) == pytest.approx(0.0)


def test_cohens_d_returns_zero_for_zero_variance():
    g1 = np.array([7.0, 7.0, 7.0])
    g2 = np.array([7.0, 7.0, 7.0])
    assert cohens_d(g1, g2) == pytest.approx(0.0)


def test_cohens_d_constant_groups_different_means_returns_zero():
    g1 = np.array([10.0, 10.0, 10.0])
    g2 = np.array([5.0, 5.0, 5.0])
    assert cohens_d(g1, g2) == pytest.approx(0.0)


def test_cohens_d_textbook_large_effect():
    g1 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = cohens_d(g1, g2)
    assert d > 0.8


# ── Pure math: Glass's delta ─────────────────────────────────────────

def test_glass_delta_known_values():
    g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    g2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    delta = glass_delta(g1, g2)
    expected = (np.mean(g1) - np.mean(g2)) / float(np.std(g2, ddof=1))
    assert delta == pytest.approx(expected)


def test_glass_delta_uses_control_group_std():
    g1 = np.array([100.0, 200.0, 300.0])
    g2 = np.array([10.0, 20.0, 30.0])
    delta = glass_delta(g1, g2)
    expected = (np.mean(g1) - np.mean(g2)) / float(np.std(g2, ddof=1))
    assert delta == pytest.approx(expected)


def test_glass_delta_differs_from_cohens_d_with_unequal_variances():
    g1 = np.array([10.0, 11.0, 12.0])
    g2 = np.array([1.0, 5.0, 9.0])
    d = cohens_d(g1, g2)
    delta = glass_delta(g1, g2)
    assert d != pytest.approx(delta)


def test_glass_delta_returns_zero_for_single_element_control():
    g1 = np.array([10.0, 20.0, 30.0])
    g2 = np.array([5.0])
    assert glass_delta(g1, g2) == pytest.approx(0.0)


def test_glass_delta_zero_control_std_returns_zero():
    g1 = np.array([10.0, 20.0, 30.0])
    g2 = np.array([5.0, 5.0, 5.0])
    assert glass_delta(g1, g2) == pytest.approx(0.0)


def test_glass_delta_identical_distributions_is_zero():
    g = np.array([3.0, 6.0, 9.0, 12.0])
    assert glass_delta(g, g.copy()) == pytest.approx(0.0)


# ── Pure math: Hedges' g ─────────────────────────────────────────────

def test_hedges_g_equals_d_times_correction():
    g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    g2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    d = cohens_d(g1, g2)
    n = len(g1) + len(g2)
    correction = 1.0 - 3.0 / (4.0 * (n - 2) - 1.0)
    expected = d * correction
    assert hedges_g(g1, g2) == pytest.approx(expected)


def test_hedges_g_smaller_than_cohens_d_in_magnitude():
    g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    g2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    d = cohens_d(g1, g2)
    g = hedges_g(g1, g2)
    assert abs(g) < abs(d)


def test_hedges_g_equals_d_for_very_small_samples():
    g1 = np.array([5.0])
    g2 = np.array([10.0, 15.0])
    d = cohens_d(g1, g2)
    g = hedges_g(g1, g2)
    assert g == pytest.approx(d)


def test_hedges_g_correction_approaches_one_for_large_n():
    n = 10000
    correction = 1.0 - 3.0 / (4.0 * (n - 2) - 1.0)
    assert correction == pytest.approx(1.0, abs=1e-3)


def test_hedges_g_identical_distributions_is_zero():
    g = np.array([5.0, 10.0, 15.0, 20.0])
    assert hedges_g(g, g.copy()) == pytest.approx(0.0)


# ── Integration: run_effect_size ─────────────────────────────────────

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_result(gpt2_model):
    return run_effect_size(gpt2_model, TASK, n_prompts=5)


def test_run_effect_size_returns_eval_result(circuit_result):
    assert circuit_result is not None
    assert isinstance(circuit_result, EvalResult)


def test_run_effect_size_has_correct_metric_id(circuit_result):
    assert circuit_result.metric_id == "M90.effect_size"


def test_run_effect_size_metadata_keys(circuit_result):
    expected_keys = {
        "task", "cohens_d", "glass_delta", "hedges_g",
        "circuit_mean", "circuit_std", "non_circuit_mean", "non_circuit_std",
        "n_circuit_heads", "n_non_circuit_heads", "passed", "threshold",
    }
    assert set(circuit_result.metadata.keys()) == expected_keys


def test_run_effect_size_value_equals_cohens_d(circuit_result):
    assert circuit_result.value == pytest.approx(circuit_result.metadata["cohens_d"])


def test_run_effect_size_n_samples_matches_prompts(circuit_result):
    assert circuit_result.n_samples <= 5
    assert circuit_result.n_samples > 0


def test_run_effect_size_head_counts_are_positive(circuit_result):
    assert circuit_result.metadata["n_circuit_heads"] > 0
    assert circuit_result.metadata["n_non_circuit_heads"] > 0


def test_run_effect_size_circuit_mean_higher_than_non_circuit(circuit_result):
    assert circuit_result.metadata["circuit_mean"] > circuit_result.metadata["non_circuit_mean"]


def test_run_effect_size_hedges_g_smaller_than_cohens_d(circuit_result):
    d = circuit_result.metadata["cohens_d"]
    g = circuit_result.metadata["hedges_g"]
    if abs(d) > 1e-8:
        assert abs(g) <= abs(d) + 1e-10


def test_run_effect_size_unknown_task_returns_none(gpt2_model):
    result = run_effect_size(gpt2_model, "nonexistent_task_xyz", n_prompts=5)
    assert result is None
