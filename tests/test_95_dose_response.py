import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DR_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "dose_response" / "95_dose_response.py"
)
_spec = importlib.util.spec_from_file_location("dose_response_95", _DR_PATH)
_dr_mod = importlib.util.module_from_spec(_spec)
sys.modules["dose_response_95"] = _dr_mod
_spec.loader.exec_module(_dr_mod)

compute_monotonicity = _dr_mod.compute_monotonicity
compute_slope = _dr_mod.compute_slope
make_dose_hooks = _dr_mod.make_dose_hooks
sweep_dose_response = _dr_mod.sweep_dose_response
run_dose_response = _dr_mod.run_dose_response
ABLATION_FRACTIONS = _dr_mod.ABLATION_FRACTIONS

from mechanistic_validity.instruments.common import EvalResult, load_model


# -- Pure math: compute_monotonicity ------------------------------------------

def test_monotonicity_perfectly_decreasing():
    assert compute_monotonicity([10.0, 8.0, 6.0, 4.0, 2.0]) == pytest.approx(1.0)


def test_monotonicity_perfectly_increasing():
    assert compute_monotonicity([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.0)


def test_monotonicity_flat():
    assert compute_monotonicity([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_monotonicity_single_element():
    assert compute_monotonicity([42.0]) == pytest.approx(1.0)


def test_monotonicity_two_elements_decreasing():
    assert compute_monotonicity([10.0, 5.0]) == pytest.approx(1.0)


def test_monotonicity_two_elements_increasing():
    assert compute_monotonicity([5.0, 10.0]) == pytest.approx(0.0)


def test_monotonicity_partial():
    assert compute_monotonicity([10.0, 8.0, 9.0, 6.0]) == pytest.approx(2.0 / 3.0)


def test_monotonicity_range_bounded():
    values = [10.0, 12.0, 8.0, 7.0, 11.0]
    m = compute_monotonicity(values)
    assert 0.0 <= m <= 1.0


# -- Pure math: compute_slope ------------------------------------------------

def test_slope_known_linear():
    values = [10.0, 7.5, 5.0, 2.5, 0.0]
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert compute_slope(values, fractions) == pytest.approx(10.0)


def test_slope_flat():
    values = [5.0, 5.0, 5.0]
    fractions = [0.0, 0.5, 1.0]
    assert compute_slope(values, fractions) == pytest.approx(0.0)


def test_slope_single_element():
    assert compute_slope([5.0], [0.0]) == pytest.approx(0.0)


def test_slope_negative_when_increasing():
    values = [0.0, 5.0, 10.0]
    fractions = [0.0, 0.5, 1.0]
    assert compute_slope(values, fractions) == pytest.approx(-10.0)


def test_slope_uses_only_endpoints():
    values = [10.0, 100.0, 0.0]
    fractions = [0.0, 0.5, 1.0]
    assert compute_slope(values, fractions) == pytest.approx(10.0)


# -- Integration: run_dose_response on IOI -----------------------------------

@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_run_dose_response_returns_eval_results(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert len(results) == 1
    assert isinstance(results[0], EvalResult)


def test_run_dose_response_metric_id(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert results[0].metric_id == "B95.dose_response"


def test_run_dose_response_metadata_keys(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    meta = results[0].metadata
    expected_keys = {
        "task", "fractions", "circuit_curve", "monotonicity",
        "circuit_slope", "mean_random_slope", "selectivity",
        "n_circuit_heads", "circuit_heads", "n_random_baselines",
        "passed", "threshold_monotonicity", "threshold_selectivity",
    }
    assert set(meta.keys()) == expected_keys


def test_run_dose_response_monotonicity_in_range(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    mono = results[0].metadata["monotonicity"]
    assert 0.0 <= mono <= 1.0


def test_run_dose_response_selectivity_positive(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    sel = results[0].metadata["selectivity"]
    assert sel > 0.0


def test_run_dose_response_curve_length_matches_fractions(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    curve = results[0].metadata["circuit_curve"]
    fracs = results[0].metadata["fractions"]
    assert len(curve) == len(fracs)
    assert len(fracs) == 5
    assert fracs == ABLATION_FRACTIONS


def test_run_dose_response_value_equals_monotonicity(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert results[0].value == pytest.approx(results[0].metadata["monotonicity"])


def test_run_dose_response_n_samples_positive(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert results[0].n_samples > 0
    assert results[0].n_samples <= 5


def test_run_dose_response_circuit_heads_positive(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert results[0].metadata["n_circuit_heads"] > 0


def test_run_dose_response_baseline_random_set(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    assert results[0].baseline_random is not None


def test_run_dose_response_clean_curve_first_point_nonzero(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["ioi"], n_prompts=5, n_random_baselines=3,
    )
    curve = results[0].metadata["circuit_curve"]
    assert abs(curve[0]) > 1e-6


def test_run_dose_response_unknown_task_returns_empty(gpt2_model):
    results = run_dose_response(
        gpt2_model, tasks=["nonexistent_task_xyz"], n_prompts=5, n_random_baselines=3,
    )
    assert results == []
