import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "counterfactual_das" / "20_corrupt_restore.py"
)
_spec = importlib.util.spec_from_file_location("corrupt_restore_20", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["corrupt_restore_20"] = _mod
_spec.loader.exec_module(_mod)

cache_clean_z = _mod.cache_clean_z
make_corrupt_then_restore_hooks = _mod.make_corrupt_then_restore_hooks
make_full_corrupt_hooks = _mod.make_full_corrupt_hooks
run_corrupt_restore = _mod.run_corrupt_restore

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_cache_clean_z_returns_all_layers(gpt2_model):
    from mechanistic_validity.metrics.common import generate_prompts

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=1)
    tokens = gpt2_model.to_tokens(prompts[0].text)
    clean_z = cache_clean_z(gpt2_model, tokens)
    assert len(clean_z) == gpt2_model.cfg.n_layers
    for L in range(gpt2_model.cfg.n_layers):
        assert L in clean_z
        assert clean_z[L].shape[2] == gpt2_model.cfg.n_heads


def test_make_full_corrupt_hooks_returns_hooks_per_layer():
    n_layers = 4
    mean_z = torch.randn(n_layers, 2, 8)
    hooks = make_full_corrupt_hooks(n_layers, mean_z)
    assert len(hooks) == n_layers
    for hook_name, hook_fn in hooks:
        assert "hook_z" in hook_name


def test_make_corrupt_then_restore_hooks_returns_hooks_per_layer():
    n_layers = 3
    n_heads = 2
    mean_z = torch.randn(n_layers, n_heads, 8)
    clean_z = {L: torch.randn(1, 5, n_heads, 8) for L in range(n_layers)}
    restore_heads = {(0, 0), (2, 1)}
    hooks = make_corrupt_then_restore_hooks(n_layers, n_heads, restore_heads, mean_z, clean_z)
    assert len(hooks) == n_layers


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_corrupt_restore(gpt2_model, tasks=[TASK], n_prompts=3, n_random_baselines=2)


def test_run_corrupt_restore_returns_eval_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_corrupt_restore_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C20.corrupt_restore"


def test_run_corrupt_restore_has_restoration_rate(circuit_results):
    for r in circuit_results:
        assert "restoration_rate" in r.metadata
        assert isinstance(r.metadata["restoration_rate"], float)


def test_run_corrupt_restore_has_baselines(circuit_results):
    for r in circuit_results:
        assert r.baseline_random is not None


def test_run_corrupt_restore_has_per_head_data(circuit_results):
    for r in circuit_results:
        assert "per_head_restoration" in r.metadata
        assert isinstance(r.metadata["per_head_restoration"], dict)


def test_run_corrupt_restore_clean_greater_than_corrupt(circuit_results):
    for r in circuit_results:
        assert r.metadata["mean_clean_ld"] != r.metadata["mean_corrupt_ld"]


def test_run_corrupt_restore_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK
