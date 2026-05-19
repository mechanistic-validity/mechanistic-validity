import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "mdc_glennan" / "18_weight_extended.py"
)
_spec = importlib.util.spec_from_file_location("weight_extended_18", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["weight_extended_18"] = _mod
_spec.loader.exec_module(_mod)

effective_rank = _mod.effective_rank
compute_wqk_effective_rank = _mod.compute_wqk_effective_rank
compute_wov_top_directions = _mod.compute_wov_top_directions
compute_cosine_alignment = _mod.compute_cosine_alignment
compute_spectral_norms = _mod.compute_spectral_norms
run_weight_extended = _mod.run_weight_extended

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


def test_effective_rank_uniform_singular_values():
    sv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rank = effective_rank(sv)
    assert rank == pytest.approx(4.0)


def test_effective_rank_single_dominant():
    sv = torch.tensor([100.0, 0.001, 0.001, 0.001])
    rank = effective_rank(sv)
    assert rank == pytest.approx(1.0, abs=0.1)


def test_effective_rank_empty_returns_zero():
    sv = torch.tensor([0.0, 0.0])
    rank = effective_rank(sv)
    assert rank == pytest.approx(0.0)


def test_effective_rank_two_equal():
    sv = torch.tensor([5.0, 5.0, 0.0, 0.0])
    rank = effective_rank(sv)
    assert rank == pytest.approx(2.0)


def test_effective_rank_monotone_in_entropy():
    sv_low = torch.tensor([10.0, 0.01, 0.01])
    sv_high = torch.tensor([3.0, 3.0, 3.0])
    assert effective_rank(sv_low) < effective_rank(sv_high)


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_wqk_effective_rank_shape(gpt2_model):
    ranks = compute_wqk_effective_rank(gpt2_model)
    assert ranks.shape == (gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads)
    assert np.all(ranks >= 0)


def test_compute_spectral_norms_shape(gpt2_model):
    norms = compute_spectral_norms(gpt2_model)
    assert norms.shape == (gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads)
    assert np.all(norms >= 0)


def test_compute_wov_top_directions_returns_dict(gpt2_model):
    directions = compute_wov_top_directions(gpt2_model, k=2)
    assert isinstance(directions, dict)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert len(directions) == n_total
    for (L, H), tensor in directions.items():
        assert tensor.shape[0] == 2
        assert tensor.shape[1] == gpt2_model.cfg.d_vocab


def test_compute_cosine_alignment_returns_values():
    d_vocab = 100
    directions = {
        (0, 0): torch.randn(3, d_vocab),
        (0, 1): torch.randn(3, d_vocab),
        (1, 0): torch.randn(3, d_vocab),
        (1, 1): torch.randn(3, d_vocab),
    }
    for k in directions:
        norms = directions[k].norm(dim=-1, keepdim=True).clamp(min=1e-10)
        directions[k] = directions[k] / norms

    circuit_heads = {(0, 0), (1, 0)}
    result = compute_cosine_alignment(directions, circuit_heads)
    assert isinstance(result, dict)
    assert len(result) == 2
    for key, val in result.items():
        assert 0.0 <= val <= 1.0


def test_compute_cosine_alignment_empty_non_circuit():
    directions = {(0, 0): torch.randn(3, 50)}
    norms = directions[(0, 0)].norm(dim=-1, keepdim=True).clamp(min=1e-10)
    directions[(0, 0)] = directions[(0, 0)] / norms
    circuit_heads = {(0, 0)}
    result = compute_cosine_alignment(directions, circuit_heads)
    assert result == {}


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_weight_extended(gpt2_model, tasks=[TASK])


def test_run_weight_extended_returns_three_metrics(circuit_results):
    metric_ids = {r.metric_id for r in circuit_results}
    assert "C18.wqk_effective_rank" in metric_ids
    assert "C18.cosine_alignment" in metric_ids
    assert "C18.spectral_norm_ratio" in metric_ids


def test_run_weight_extended_all_eval_results(circuit_results):
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_weight_extended_values_are_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_weight_extended_spectral_ratio_positive(circuit_results):
    for r in circuit_results:
        if r.metric_id == "C18.spectral_norm_ratio":
            assert r.value > 0


def test_run_weight_extended_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
