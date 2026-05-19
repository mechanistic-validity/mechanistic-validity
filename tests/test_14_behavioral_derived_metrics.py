import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult

_BDM_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "metrics"
    / "behavioral"
    / "logit_diff_recovery"
    / "14_derived_metrics.py"
)
_spec = importlib.util.spec_from_file_location("_bdm14", _BDM_PATH)
_bdm_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bdm_mod
_spec.loader.exec_module(_bdm_mod)

_d_prime = _bdm_mod._d_prime
_compute_sparsity = _bdm_mod._compute_sparsity
_compute_node_overlap = _bdm_mod._compute_node_overlap
_compute_weight_classifier_sdt = _bdm_mod._compute_weight_classifier_sdt
GPT2_TOTAL_HEADS = _bdm_mod.GPT2_TOTAL_HEADS
PUBLISHED_CIRCUIT_SIZES = _bdm_mod.PUBLISHED_CIRCUIT_SIZES
WEIGHT_CLASSIFIER_RESULTS = _bdm_mod.WEIGHT_CLASSIFIER_RESULTS

TASK = "ioi"


def test_d_prime_basic():
    d = _d_prime(10.0, 0.0, 1.0, 1.0)
    assert d == pytest.approx(10.0)


def test_d_prime_zero_difference():
    d = _d_prime(5.0, 5.0, 2.0, 2.0)
    assert d == pytest.approx(0.0)


def test_d_prime_zero_variance():
    d = _d_prime(5.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(0.0)


def test_d_prime_asymmetric_std():
    d = _d_prime(6.0, 2.0, 1.0, 3.0)
    pooled = math.sqrt((1.0 + 9.0) / 2.0)
    assert d == pytest.approx((6.0 - 2.0) / pooled)


def test_compute_sparsity_returns_fraction():
    results = _compute_sparsity([TASK])
    assert len(results) >= 1
    r = results[0]
    assert r.metric_id == "C14.sparsity"
    assert 0 < r.value < 1
    assert r.n_samples == GPT2_TOTAL_HEADS


def test_compute_sparsity_multiple_tasks():
    tasks = ["ioi", "sva", "induction"]
    results = _compute_sparsity(tasks)
    assert len(results) == len(tasks)
    task_set = {r.metadata["task"] for r in results}
    for t in tasks:
        assert t in task_set


def test_compute_sparsity_nonexistent():
    results = _compute_sparsity(["nonexistent_xyz"])
    assert results == []


def test_compute_node_overlap_returns_ratio():
    results = _compute_node_overlap([TASK])
    assert len(results) == 1
    r = results[0]
    assert r.metric_id == "C14.node_overlap_jaccard"
    assert 0 < r.value <= 1.0
    pub = PUBLISHED_CIRCUIT_SIZES[TASK]
    our = r.metadata["our_circuit_size"]
    expected = min(our, pub) / max(our, pub)
    assert r.value == pytest.approx(expected)


def test_compute_node_overlap_task_without_published():
    results = _compute_node_overlap(["acronym"])
    assert results == []


def test_weight_classifier_sdt_produces_two_per_task():
    results = _compute_weight_classifier_sdt([TASK])
    assert len(results) == 2
    metric_ids = {r.metric_id for r in results}
    assert "C14.hit_rate" in metric_ids
    assert "C14.false_alarm_rate" in metric_ids


def test_weight_classifier_sdt_values_match_constants():
    for task, wc in WEIGHT_CLASSIFIER_RESULTS.items():
        results = _compute_weight_classifier_sdt([task])
        hit = [r for r in results if r.metric_id == "C14.hit_rate"][0]
        fa = [r for r in results if r.metric_id == "C14.false_alarm_rate"][0]
        assert hit.value == pytest.approx(wc["recall"])
        assert fa.value == pytest.approx(1.0 - wc["precision"])


def test_weight_classifier_sdt_nonexistent_task():
    results = _compute_weight_classifier_sdt(["nonexistent_xyz"])
    assert results == []


def test_published_circuit_sizes_all_positive():
    for task, size in PUBLISHED_CIRCUIT_SIZES.items():
        assert size > 0
