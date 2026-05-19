import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechanistic_validity.instruments.common import EvalResult, load_model, get_circuit_heads

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "structural" / "spectral_svd" / "18_weight_extended.py"
)
_spec = importlib.util.spec_from_file_location("_spectral_svd_18", _MOD_PATH)
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


def test_effective_rank_monotone_with_spread():
    sv_concentrated = torch.tensor([100.0, 1.0, 0.01])
    sv_spread = torch.tensor([10.0, 10.0, 10.0])
    er_conc = effective_rank(sv_concentrated)
    er_spread = effective_rank(sv_spread)
    assert er_spread > er_conc


def test_effective_rank_bounded_by_count():
    n = 8
    sv = torch.rand(n) + 0.1
    er = effective_rank(sv)
    assert 1.0 <= er <= n


def test_compute_wqk_effective_rank_positive(gpt2_model):
    ranks = compute_wqk_effective_rank(gpt2_model)
    assert (ranks > 0).all()


def test_compute_wov_top_directions_normalized(gpt2_model):
    directions = compute_wov_top_directions(gpt2_model, k=2)
    for key, tensor in directions.items():
        norms = tensor.norm(dim=-1)
        for n in norms:
            assert n.item() == pytest.approx(1.0, abs=1e-4)


def test_compute_cosine_alignment_empty_when_all_circuit():
    n_layers, n_heads = 2, 2
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    dummy_dirs = {h: torch.randn(3, 10) for h in all_heads}
    for h in dummy_dirs:
        norms = dummy_dirs[h].norm(dim=-1, keepdim=True).clamp(min=1e-10)
        dummy_dirs[h] = dummy_dirs[h] / norms
    alignment = compute_cosine_alignment(dummy_dirs, all_heads)
    assert alignment == {}


def test_run_weight_extended_produces_results(circuit_results):
    assert len(circuit_results) == 3


def test_run_weight_extended_cosine_alignment_result(circuit_results):
    r = [x for x in circuit_results if x.metric_id == "C18.cosine_alignment"][0]
    assert isinstance(r, EvalResult)
    assert 0.0 <= r.value <= 1.0
    assert r.metadata["task"] == TASK
    assert "per_head" in r.metadata
    assert r.metadata["interpretation"] == "low=specialized, high=generic"


def test_run_weight_extended_all_tasks_match(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
        assert r.n_samples > 0
