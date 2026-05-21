import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "structural" / "k_composition" / "B10_k_composition.py"
)
_spec = importlib.util.spec_from_file_location("_b10_k_composition", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

compute_k_composition_matrix = _mod.compute_k_composition_matrix
head_idx = _mod.head_idx
longest_strong_chain = _mod.longest_strong_chain
run = _mod.run

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("b10")
    return run(model=gpt2_model, tasks=[TASK], device="cpu", save=False, resume=False,
               output_dir=str(out))


def test_head_idx_basic():
    assert head_idx(0, 0, 12) == 0
    assert head_idx(0, 5, 12) == 5
    assert head_idx(2, 3, 12) == 27
    assert head_idx(1, 0, 12) == 12


def test_compute_k_composition_matrix_shape(gpt2_model):
    kcomp = compute_k_composition_matrix(gpt2_model)
    n_total = gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert kcomp.shape == (n_total, n_total)


def test_compute_k_composition_matrix_diagonal_zero(gpt2_model):
    kcomp = compute_k_composition_matrix(gpt2_model)
    diag = np.diag(kcomp)
    np.testing.assert_array_equal(diag, 0.0)


def test_compute_k_composition_matrix_off_diagonal_nonnegative(gpt2_model):
    kcomp = compute_k_composition_matrix(gpt2_model)
    assert (kcomp >= 0).all()


def test_longest_strong_chain_linear():
    n_heads = 4
    n_total = 3 * n_heads
    kcomp = np.zeros((n_total, n_total))
    heads = {(0, 0), (1, 1), (2, 2)}
    kcomp[head_idx(0, 0, n_heads), head_idx(1, 1, n_heads)] = 10.0
    kcomp[head_idx(1, 1, n_heads), head_idx(2, 2, n_heads)] = 10.0
    chain = longest_strong_chain(heads, kcomp, n_heads, threshold=1.0)
    assert chain == 3


def test_longest_strong_chain_no_edges():
    n_heads = 4
    n_total = 2 * n_heads
    kcomp = np.zeros((n_total, n_total))
    heads = {(0, 0), (1, 1)}
    chain = longest_strong_chain(heads, kcomp, n_heads, threshold=1.0)
    assert chain == 1


def test_longest_strong_chain_single_head():
    n_heads = 4
    n_total = 2 * n_heads
    kcomp = np.zeros((n_total, n_total))
    heads = {(0, 2)}
    chain = longest_strong_chain(heads, kcomp, n_heads, threshold=1.0)
    assert chain == 1


def test_run_returns_eval_result(circuit_results):
    assert len(circuit_results) >= 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "B10.k_composition_mean"


def test_run_value_positive(circuit_results):
    r = circuit_results[0]
    assert r.value > 0


def test_run_metadata_fields(circuit_results):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert "background_mean" in meta
    assert "background_std" in meta
    assert "n_strong_edges" in meta
    assert "hierarchy_depth" in meta
    assert "top_10_edges" in meta
    assert isinstance(meta["top_10_edges"], dict)


def test_run_n_samples_positive(circuit_results):
    r = circuit_results[0]
    assert r.n_samples >= 1
