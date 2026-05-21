import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "structural" / "effective_rank" / "B13_capacity_utilization.py"
)
_spec = importlib.util.spec_from_file_location("_b13_capacity", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

effective_rank = _mod.effective_rank
sv_concentration = _mod.sv_concentration
compute_head_capacity = _mod.compute_head_capacity
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run(model=gpt2_model, tasks=[TASK], resume=False)


def test_effective_rank_spread_higher_than_concentrated():
    sv_concentrated = torch.tensor([100.0, 1.0, 0.01])
    sv_spread = torch.tensor([10.0, 10.0, 10.0])
    assert effective_rank(sv_spread) > effective_rank(sv_concentrated)


def test_effective_rank_bounded_by_count():
    n = 8
    sv = torch.rand(n) + 0.1
    er = effective_rank(sv)
    assert 1.0 <= er <= n


def test_effective_rank_single_sv_equals_one():
    sv = torch.tensor([5.0])
    assert effective_rank(sv) == pytest.approx(1.0)


def test_effective_rank_equal_svs_equals_count():
    n = 6
    sv = torch.ones(n) * 3.0
    assert effective_rank(sv) == pytest.approx(float(n), abs=1e-5)


def test_effective_rank_zeros_return_zero():
    sv = torch.tensor([0.0, 0.0, 0.0])
    assert effective_rank(sv) == pytest.approx(0.0)


def test_sv_concentration_dominated_by_top():
    sv = torch.tensor([100.0, 0.001, 0.001, 0.001])
    assert sv_concentration(sv) == pytest.approx(1.0, abs=0.01)


def test_sv_concentration_equal_svs_low():
    n = 10
    sv = torch.ones(n) * 5.0
    assert sv_concentration(sv) == pytest.approx(1.0 / n, abs=1e-5)


def test_sv_concentration_zeros_return_zero():
    sv = torch.tensor([0.0, 0.0])
    assert sv_concentration(sv) == pytest.approx(0.0)


def test_compute_head_capacity_keys_and_subkeys(gpt2_model):
    cap = compute_head_capacity(gpt2_model)
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    assert len(cap) == n_layers * n_heads
    for (L, H), v in cap.items():
        assert 0 <= L < n_layers
        assert 0 <= H < n_heads
        for key in ["qk_effective_rank", "ov_effective_rank",
                     "qk_concentration", "ov_concentration",
                     "qk_spectral_norm", "ov_spectral_norm"]:
            assert key in v


def test_compute_head_capacity_values_positive(gpt2_model):
    cap = compute_head_capacity(gpt2_model)
    for v in cap.values():
        assert v["qk_effective_rank"] > 0
        assert v["ov_effective_rank"] > 0
        assert 0.0 < v["qk_concentration"] <= 1.0
        assert 0.0 < v["ov_concentration"] <= 1.0
        assert v["qk_spectral_norm"] > 0
        assert v["ov_spectral_norm"] > 0


def test_run_returns_eval_result(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "B13.capacity_utilization"


def test_run_value_positive(circuit_results):
    r = circuit_results[0]
    assert r.value > 0


def test_run_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert meta["n_circuit_heads"] >= 1
    for key in ["circuit_qk_effective_rank", "circuit_ov_effective_rank",
                "circuit_qk_concentration", "circuit_ov_concentration",
                "background_qk_effective_rank", "background_ov_effective_rank",
                "qk_rank_ratio", "ov_rank_ratio",
                "qk_concentration_ratio", "ov_concentration_ratio"]:
        assert key in meta
