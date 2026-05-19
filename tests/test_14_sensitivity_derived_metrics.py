import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult

_SDM_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "calibrations"
    / "sensitivity"
    / "14_derived_metrics.py"
)
_spec = importlib.util.spec_from_file_location("_sdm14", _SDM_PATH)
_sdm_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _sdm_mod
_spec.loader.exec_module(_sdm_mod)

_d_prime = _sdm_mod._d_prime
_compute_sparsity = _sdm_mod._compute_sparsity
_compute_node_overlap = _sdm_mod._compute_node_overlap
_compute_weight_classifier_sdt = _sdm_mod._compute_weight_classifier_sdt
GPT2_TOTAL_HEADS = _sdm_mod.GPT2_TOTAL_HEADS
PUBLISHED_CIRCUIT_SIZES = _sdm_mod.PUBLISHED_CIRCUIT_SIZES
WEIGHT_CLASSIFIER_RESULTS = _sdm_mod.WEIGHT_CLASSIFIER_RESULTS

TASK = "ioi"


def test_d_prime_large_separation():
    d = _d_prime(5.0, 0.0, 1.0, 1.0)
    assert d == pytest.approx(5.0)


def test_d_prime_zero_separation():
    d = _d_prime(3.0, 3.0, 1.0, 1.0)
    assert d == pytest.approx(0.0)


def test_d_prime_negative_separation():
    d = _d_prime(0.0, 5.0, 1.0, 1.0)
    assert d == pytest.approx(-5.0)


def test_d_prime_zero_std():
    d = _d_prime(5.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(0.0)


def test_d_prime_unequal_stds():
    d = _d_prime(4.0, 0.0, 2.0, 0.0)
    pooled = math.sqrt((4.0 + 0.0) / 2.0)
    assert d == pytest.approx(4.0 / pooled)


def test_d_prime_formula():
    mean_s, mean_n, std_s, std_n = 10.0, 5.0, 2.0, 3.0
    pooled = math.sqrt((std_s**2 + std_n**2) / 2.0)
    expected = (mean_s - mean_n) / pooled
    assert _d_prime(mean_s, mean_n, std_s, std_n) == pytest.approx(expected)


def test_compute_sparsity_ioi():
    results = _compute_sparsity([TASK])
    assert len(results) >= 1
    r = results[0]
    assert r.metric_id == "C14.sparsity"
    assert r.metadata["task"] == TASK
    assert r.metadata["n_total_heads"] == GPT2_TOTAL_HEADS
    n_heads = r.metadata["n_circuit_heads"]
    assert r.value == pytest.approx(n_heads / GPT2_TOTAL_HEADS)
    assert 0 < r.value < 1


def test_compute_sparsity_unknown_task():
    results = _compute_sparsity(["nonexistent_xyz"])
    assert results == []


def test_compute_node_overlap_ioi():
    results = _compute_node_overlap([TASK])
    assert len(results) >= 1
    r = results[0]
    assert r.metric_id == "C14.node_overlap_jaccard"
    our = r.metadata["our_circuit_size"]
    pub = r.metadata["published_circuit_size"]
    assert pub == PUBLISHED_CIRCUIT_SIZES[TASK]
    assert r.value == pytest.approx(min(our, pub) / max(our, pub))
    assert 0 < r.value <= 1.0


def test_compute_node_overlap_no_published_size():
    results = _compute_node_overlap(["rti"])
    assert results == []


def test_compute_weight_classifier_sdt_ioi():
    results = _compute_weight_classifier_sdt([TASK])
    hit_results = [r for r in results if r.metric_id == "C14.hit_rate"]
    fa_results = [r for r in results if r.metric_id == "C14.false_alarm_rate"]
    assert len(hit_results) == 1
    assert len(fa_results) == 1
    wc = WEIGHT_CLASSIFIER_RESULTS[TASK]
    assert hit_results[0].value == pytest.approx(wc["recall"])
    assert fa_results[0].value == pytest.approx(1.0 - wc["precision"])


def test_compute_weight_classifier_sdt_all_tasks():
    tasks = list(WEIGHT_CLASSIFIER_RESULTS.keys())
    results = _compute_weight_classifier_sdt(tasks)
    hit_count = sum(1 for r in results if r.metric_id == "C14.hit_rate")
    fa_count = sum(1 for r in results if r.metric_id == "C14.false_alarm_rate")
    assert hit_count == len(tasks)
    assert fa_count == len(tasks)


def test_compute_weight_classifier_sdt_unknown_task():
    results = _compute_weight_classifier_sdt(["nonexistent_xyz"])
    assert results == []


def test_compute_weight_classifier_false_alarm_bounded():
    for task, wc in WEIGHT_CLASSIFIER_RESULTS.items():
        fa = 1.0 - wc["precision"]
        assert 0.0 <= fa <= 1.0
        assert 0.0 <= wc["recall"] <= 1.0
