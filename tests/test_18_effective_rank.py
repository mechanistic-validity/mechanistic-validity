import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.metrics.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "structural" / "effective_rank" / "18_weight_extended.py"
)
_spec = importlib.util.spec_from_file_location("_eff_rank_18", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

effective_rank = _mod.effective_rank
compute_wqk_effective_rank = _mod.compute_wqk_effective_rank
compute_wov_top_directions = _mod.compute_wov_top_directions
compute_cosine_alignment = _mod.compute_cosine_alignment
compute_spectral_norms = _mod.compute_spectral_norms
run_weight_extended = _mod.run_weight_extended

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_weight_extended(gpt2_model, [TASK])


def test_effective_rank_uniform_singular_values():
    sv = torch.ones(10)
    er = effective_rank(sv)
    assert er == pytest.approx(10.0, abs=1e-4)


def test_effective_rank_single_dominant():
    sv = torch.tensor([100.0, 0.0, 0.0, 0.0])
    er = effective_rank(sv)
    assert er == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_empty():
    sv = torch.tensor([0.0, 0.0])
    er = effective_rank(sv)
    assert er == pytest.approx(0.0)


def test_effective_rank_two_equal():
    sv = torch.tensor([5.0, 5.0, 0.0])
    er = effective_rank(sv)
    assert er == pytest.approx(2.0, abs=1e-4)


def test_compute_wqk_effective_rank_shape(gpt2_model):
    ranks = compute_wqk_effective_rank(gpt2_model)
    assert ranks.shape == (gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads)
    assert (ranks >= 0).all()


def test_compute_spectral_norms_shape(gpt2_model):
    norms = compute_spectral_norms(gpt2_model)
    assert norms.shape == (gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads)
    assert (norms >= 0).all()


def test_compute_wov_top_directions_shape(gpt2_model):
    directions = compute_wov_top_directions(gpt2_model, k=3)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(directions) == n_layers * n_heads
    for (L, H), tensor in directions.items():
        assert tensor.shape[0] == 3


def test_compute_cosine_alignment_returns_dict(gpt2_model):
    directions = compute_wov_top_directions(gpt2_model, k=3)
    circuit_heads = get_circuit_heads(TASK)
    alignment = compute_cosine_alignment(directions, circuit_heads)
    assert isinstance(alignment, dict)
    assert len(alignment) == len(circuit_heads)
    for key, val in alignment.items():
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_run_weight_extended_returns_three_results_per_task(circuit_results):
    assert len(circuit_results) == 3
    metric_ids = [r.metric_id for r in circuit_results]
    assert "C18.wqk_effective_rank" in metric_ids
    assert "C18.cosine_alignment" in metric_ids
    assert "C18.spectral_norm_ratio" in metric_ids


def test_run_weight_extended_effective_rank_result(circuit_results):
    r = [x for x in circuit_results if x.metric_id == "C18.wqk_effective_rank"][0]
    assert isinstance(r, EvalResult)
    assert r.value > 0
    assert r.baseline_random is not None
    assert r.baseline_random > 0
    assert r.metadata["task"] == TASK
    assert "per_head" in r.metadata
    assert "n_circuit_heads" in r.metadata


def test_run_weight_extended_spectral_ratio_result(circuit_results):
    r = [x for x in circuit_results if x.metric_id == "C18.spectral_norm_ratio"][0]
    assert r.value > 0
    assert r.baseline_random == pytest.approx(1.0)
    assert "per_head_norms" in r.metadata
    assert "circuit_mean_norm" in r.metadata
    assert "non_circuit_mean_norm" in r.metadata
