import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.instruments.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "measurement" / "convergent_validity" / "36_incremental_validity.py"
)
_spec = importlib.util.spec_from_file_location("_iv_36", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

get_topk_heads_by_magnitude = _mod.get_topk_heads_by_magnitude
compute_head_activation_magnitudes = _mod.compute_head_activation_magnitudes
run_incremental_validity = _mod.run_incremental_validity

TASK = "ioi"


def test_get_topk_heads_selects_highest():
    magnitudes = np.array([
        [1.0, 5.0, 3.0],
        [4.0, 2.0, 6.0],
    ])
    top2 = get_topk_heads_by_magnitude(magnitudes, k=2)
    assert (1, 2) in top2
    assert (0, 1) in top2
    assert len(top2) == 2


def test_get_topk_heads_k_equals_total():
    magnitudes = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    top4 = get_topk_heads_by_magnitude(magnitudes, k=4)
    assert len(top4) == 4


def test_get_topk_heads_k_one():
    magnitudes = np.array([
        [0.1, 0.2],
        [0.3, 10.0],
    ])
    top1 = get_topk_heads_by_magnitude(magnitudes, k=1)
    assert top1 == {(1, 1)}


def test_get_topk_heads_zero_magnitudes():
    magnitudes = np.zeros((3, 4))
    top2 = get_topk_heads_by_magnitude(magnitudes, k=2)
    assert len(top2) == 2


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_incremental_validity(gpt2_model, [TASK], n_prompts=3)


def test_run_incremental_validity_returns_results(circuit_results):
    assert len(circuit_results) > 0
    assert all(isinstance(r, EvalResult) for r in circuit_results)


def test_run_incremental_validity_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C36.incremental_validity"


def test_run_incremental_validity_delta_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_incremental_validity_metadata_keys(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert "our_faithfulness" in r.metadata
        assert "topk_faithfulness" in r.metadata
        assert "delta" in r.metadata
        assert "k" in r.metadata
        assert "our_heads" in r.metadata
        assert "topk_heads" in r.metadata
        assert "overlap_count" in r.metadata
        assert "jaccard" in r.metadata


def test_run_incremental_validity_delta_matches_difference(circuit_results):
    for r in circuit_results:
        expected_delta = r.metadata["our_faithfulness"] - r.metadata["topk_faithfulness"]
        assert r.value == pytest.approx(expected_delta)
        assert r.metadata["delta"] == pytest.approx(expected_delta)


def test_run_incremental_validity_jaccard_bounded(circuit_results):
    for r in circuit_results:
        assert 0.0 <= r.metadata["jaccard"] <= 1.0


def test_run_incremental_validity_k_matches_circuit_size(circuit_results):
    for r in circuit_results:
        assert r.metadata["k"] == len(r.metadata["our_heads"])


def test_run_incremental_validity_skips_unknown_task(gpt2_model):
    results = run_incremental_validity(gpt2_model, ["nonexistent"], n_prompts=3)
    assert results == []
