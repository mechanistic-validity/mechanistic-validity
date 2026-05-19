import importlib
import math

import pytest

from mechanistic_validity.instruments.common import EvalResult, get_circuit_heads, load_model

_mod = importlib.import_module(
    "mechanistic_validity.instruments.causal.hedonic_pas.94_pairwise_synergy"
)
compute_pairwise_synergy = _mod.compute_pairwise_synergy
run_pairwise_synergy = _mod.run_pairwise_synergy


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


TASK = "epistemic_framing"


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_pairwise_synergy(gpt2_model, [TASK], n_prompts=5)


def test_run_pairwise_synergy_returns_eval_result_list(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_run_pairwise_synergy_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C10.pairwise_ablation_synergy"


def test_run_pairwise_synergy_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0
    assert circuit_results[0].n_samples <= 5


def test_run_pairwise_synergy_value_is_finite(circuit_results):
    assert math.isfinite(circuit_results[0].value)


def test_run_pairwise_synergy_value_equals_mean_abs_pas(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["mean_abs_pas"])


def test_run_pairwise_synergy_metadata_has_expected_keys(circuit_results):
    expected_keys = {
        "task", "n_heads", "n_pairs", "mean_abs_pas", "max_abs_pas",
        "n_synergistic", "n_redundant", "n_independent", "per_pair",
        "passed", "threshold",
    }
    assert set(circuit_results[0].metadata.keys()) == expected_keys


def test_run_pairwise_synergy_task_matches(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_run_pairwise_synergy_n_pairs_matches_combination_count(circuit_results):
    n_heads = circuit_results[0].metadata["n_heads"]
    n_pairs = circuit_results[0].metadata["n_pairs"]
    expected_pairs = n_heads * (n_heads - 1) // 2
    assert n_pairs == expected_pairs


def test_run_pairwise_synergy_head_count_matches_circuit(circuit_results):
    circuit_heads = get_circuit_heads(TASK)
    assert len(circuit_heads) == circuit_results[0].metadata["n_heads"]


def test_run_pairwise_synergy_pair_count_matches_formula(circuit_results):
    n = circuit_results[0].metadata["n_heads"]
    assert circuit_results[0].metadata["n_pairs"] == n * (n - 1) // 2


def test_run_pairwise_synergy_per_pair_values_are_finite(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair"]
    assert len(per_pair) > 0
    for key, info in per_pair.items():
        assert math.isfinite(info["pas"]), f"Non-finite PAS for {key}: {info['pas']}"


def test_run_pairwise_synergy_per_pair_labels_valid(circuit_results):
    valid_labels = {"synergy", "redundancy", "independent"}
    per_pair = circuit_results[0].metadata["per_pair"]
    for key, info in per_pair.items():
        assert info["label"] in valid_labels, f"Invalid label for {key}: {info['label']}"


def test_run_pairwise_synergy_label_counts_sum_to_n_pairs(circuit_results):
    m = circuit_results[0].metadata
    total = m["n_synergistic"] + m["n_redundant"] + m["n_independent"]
    assert total == m["n_pairs"]


def test_run_pairwise_synergy_max_abs_pas_ge_mean(circuit_results):
    m = circuit_results[0].metadata
    assert m["max_abs_pas"] >= m["mean_abs_pas"] - 1e-10


def test_run_pairwise_synergy_passed_consistent_with_threshold(circuit_results):
    m = circuit_results[0].metadata
    assert m["passed"] == (m["mean_abs_pas"] > m["threshold"])


def test_run_pairwise_synergy_per_pair_count_matches_n_pairs(circuit_results):
    per_pair = circuit_results[0].metadata["per_pair"]
    assert len(per_pair) == circuit_results[0].metadata["n_pairs"]
