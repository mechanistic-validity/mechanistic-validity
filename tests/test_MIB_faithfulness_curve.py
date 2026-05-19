import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "mib_faithfulness" / "MIB_faithfulness_curve.py"
)
_spec = importlib.util.spec_from_file_location("mib_faith", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mib_faith"] = _mod
_spec.loader.exec_module(_mod)

edges_to_heads = _mod.edges_to_heads
compute_cpr = _mod.compute_cpr
compute_cmd = _mod.compute_cmd
run_mib_faithfulness = _mod.run_mib_faithfulness
MIB_THRESHOLDS = _mod.MIB_THRESHOLDS

from mechanistic_validity.instruments.common import EvalResult, get_circuit_info, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_mib_faithfulness(gpt2_model, [TASK], n_prompts=5)


def test_returns_eval_result_list(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "MIB.faithfulness_curve"


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_value_is_cpr(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.cpr)


def test_cpr_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].cpr)


def test_cmd_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].cmd)


def test_faithfulness_curve_present(circuit_results):
    r = circuit_results[0]
    assert r.faithfulness_curve is not None
    assert len(r.faithfulness_curve) == len(MIB_THRESHOLDS)


def test_faithfulness_curve_values_are_finite(circuit_results):
    for t, f in circuit_results[0].faithfulness_curve.items():
        assert math.isfinite(f), f"Non-finite faithfulness at t={t}: {f}"


def test_metadata_has_expected_keys(circuit_results):
    expected_keys = {
        "task", "n_heads", "n_edges", "thresholds",
        "per_threshold_faithfulness", "passed",
    }
    assert set(circuit_results[0].metadata.keys()) == expected_keys


def test_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_passed_consistent_with_cpr(circuit_results):
    r = circuit_results[0]
    assert r.metadata["passed"] == (r.cpr > 0.5)


def test_edges_to_heads_basic():
    edges = {(0, 1, 1, 2), (1, 0, 2, 3)}
    heads = edges_to_heads(edges)
    assert heads == {(0, 1), (1, 2), (1, 0), (2, 3)}


def test_edges_to_heads_deduplicates():
    edges = {(0, 1, 1, 2), (0, 1, 2, 3)}
    heads = edges_to_heads(edges)
    assert (0, 1) in heads
    assert len(heads) == 3


def test_compute_cpr_perfect_curve():
    thresholds = [0.0, 0.5, 1.0]
    faithfulness = [1.0, 1.0, 1.0]
    cpr = compute_cpr(thresholds, faithfulness)
    assert cpr == pytest.approx(1.0)


def test_compute_cpr_zero_curve():
    thresholds = [0.0, 0.5, 1.0]
    faithfulness = [0.0, 0.0, 0.0]
    cpr = compute_cpr(thresholds, faithfulness)
    assert cpr == pytest.approx(0.0)


def test_compute_cpr_linear_curve():
    thresholds = [0.0, 1.0]
    faithfulness = [0.0, 1.0]
    cpr = compute_cpr(thresholds, faithfulness)
    assert cpr == pytest.approx(0.5)


def test_compute_cmd_perfect_curve():
    thresholds = [0.0, 0.5, 1.0]
    faithfulness = [1.0, 1.0, 1.0]
    cmd = compute_cmd(thresholds, faithfulness)
    assert cmd == pytest.approx(0.0)


def test_compute_cmd_zero_curve():
    thresholds = [0.0, 0.5, 1.0]
    faithfulness = [0.0, 0.0, 0.0]
    cmd = compute_cmd(thresholds, faithfulness)
    assert cmd == pytest.approx(1.0)


def test_cpr_plus_cmd_equals_total_area():
    thresholds = [0.0, 0.5, 1.0]
    faithfulness = [0.3, 0.7, 0.9]
    cpr = compute_cpr(thresholds, faithfulness)
    cmd = compute_cmd(thresholds, faithfulness)
    # Total area under y=1 from x=0 to x=1 is 1.0
    assert cpr + cmd == pytest.approx(1.0)


def test_per_threshold_faithfulness_count(circuit_results):
    per_t = circuit_results[0].metadata["per_threshold_faithfulness"]
    assert len(per_t) == len(MIB_THRESHOLDS)


def test_n_edges_matches_circuit(circuit_results):
    _, _, edges = get_circuit_info(TASK)
    assert circuit_results[0].metadata["n_edges"] == len(edges)
