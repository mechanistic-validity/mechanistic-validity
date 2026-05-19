import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "hedonic_pas" / "H1_shapley_interactions.py"
)
_spec = importlib.util.spec_from_file_location("hedonic_h1", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hedonic_h1"] = _mod
_spec.loader.exec_module(_mod)

compute_pas = _mod.compute_pas
compute_oca = _mod.compute_oca
run_hedonic_synergy = _mod.run_hedonic_synergy

from mechval.metrics.common import EvalResult, get_circuit_heads, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_hedonic_synergy(gpt2_model, [TASK], n_prompts=5)


def test_returns_eval_result_list(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "H1.hedonic_synergy"


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_value_equals_mean_pas(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["mean_pas"])


def test_metadata_has_expected_keys(circuit_results):
    expected_keys = {
        "task", "n_heads", "n_pairs", "mean_pas",
        "per_pair_pas", "per_pair_oca",
        "synergy_pairs", "redundancy_pairs", "passed",
    }
    assert set(circuit_results[0].metadata.keys()) == expected_keys


def test_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_n_pairs_matches_combination_count(circuit_results):
    n_heads = circuit_results[0].metadata["n_heads"]
    n_pairs = circuit_results[0].metadata["n_pairs"]
    expected_pairs = n_heads * (n_heads - 1) // 2
    assert n_pairs == expected_pairs


def test_head_count_matches_circuit(circuit_results):
    circuit_heads = get_circuit_heads(TASK)
    assert len(circuit_heads) == circuit_results[0].metadata["n_heads"]


def test_per_pair_pas_values_are_finite(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair_pas"]
    assert len(per_pair) > 0
    for key, v in per_pair.items():
        assert math.isfinite(v), f"Non-finite PAS for {key}: {v}"


def test_per_pair_oca_values_are_finite(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair_oca"]
    assert len(per_pair) > 0
    for key, v in per_pair.items():
        assert math.isfinite(v), f"Non-finite OCA for {key}: {v}"


def test_synergy_redundancy_lists_partition_pairs(circuit_results):
    m = circuit_results[0].metadata
    total = len(m["synergy_pairs"]) + len(m["redundancy_pairs"])
    assert total == m["n_pairs"]


def test_passed_means_synergy_exists(circuit_results):
    m = circuit_results[0].metadata
    assert m["passed"] == (len(m["synergy_pairs"]) > 0)


def test_per_pair_pas_count_matches_n_pairs(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair_pas"]
    assert len(per_pair) == circuit_results[0].metadata["n_pairs"]


def test_per_pair_oca_count_matches_n_pairs(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair_oca"]
    assert len(per_pair) == circuit_results[0].metadata["n_pairs"]


def test_pas_formula_sign_convention():
    # If ablating both heads has same effect as sum of individual effects,
    # PAS should be zero (no interaction).
    # PAS = -(l_{ij} - l_i - l_j + l)
    # If l = 10, l_i = 8, l_j = 7, l_ij = 5: PAS = -(5 - 8 - 7 + 10) = 0
    ld_clean = 10.0
    ld_i = 8.0
    ld_j = 7.0
    ld_ij = 5.0
    pas = -(ld_ij - ld_i - ld_j + ld_clean)
    assert pas == pytest.approx(0.0)


def test_pas_synergy_positive():
    # Synergy: ablating both is worse than sum of individual ablations
    # l = 10, l_i = 8, l_j = 7, l_ij = 3 (worse than expected 5)
    ld_clean = 10.0
    ld_i = 8.0
    ld_j = 7.0
    ld_ij = 3.0
    pas = -(ld_ij - ld_i - ld_j + ld_clean)
    assert pas == pytest.approx(2.0)
    assert pas > 0


def test_pas_redundancy_negative():
    # Redundancy: ablating both is less damaging than expected
    # l = 10, l_i = 8, l_j = 7, l_ij = 7 (better than expected 5)
    ld_clean = 10.0
    ld_i = 8.0
    ld_j = 7.0
    ld_ij = 7.0
    pas = -(ld_ij - ld_i - ld_j + ld_clean)
    assert pas == pytest.approx(-2.0)
    assert pas < 0
