import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "instruments"
    / "behavioral" / "logit_diff_recovery" / "20_corrupt_restore.py"
)
_spec = importlib.util.spec_from_file_location("corrupt_restore_20", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["corrupt_restore_20"] = _mod
_spec.loader.exec_module(_mod)

cache_clean_z = _mod.cache_clean_z
make_corrupt_then_restore_hooks = _mod.make_corrupt_then_restore_hooks
make_full_corrupt_hooks = _mod.make_full_corrupt_hooks
run_corrupt_restore = _mod.run_corrupt_restore

from mechanistic_validity.instruments.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_cache_clean_z_returns_all_layers(gpt2_model):
    tokens = gpt2_model.to_tokens("The cat sat on the mat")
    clean_z = cache_clean_z(gpt2_model, tokens)
    assert len(clean_z) == gpt2_model.cfg.n_layers
    for layer_idx, tensor in clean_z.items():
        assert tensor.shape[2] == gpt2_model.cfg.n_heads
        assert tensor.shape[3] == gpt2_model.cfg.d_head


def test_make_full_corrupt_hooks_returns_hooks_for_all_layers(gpt2_model):
    n_layers = gpt2_model.cfg.n_layers
    mean_z = torch.zeros(n_layers, gpt2_model.cfg.n_heads, gpt2_model.cfg.d_head)
    hooks = make_full_corrupt_hooks(n_layers, mean_z)
    assert len(hooks) == n_layers
    for name, fn in hooks:
        assert "hook_z" in name
        assert callable(fn)


def test_make_corrupt_then_restore_hooks_length(gpt2_model):
    n_layers = gpt2_model.cfg.n_layers
    n_heads = gpt2_model.cfg.n_heads
    d_head = gpt2_model.cfg.d_head
    mean_z = torch.zeros(n_layers, n_heads, d_head)
    clean_z = {L: torch.zeros(1, 10, n_heads, d_head) for L in range(n_layers)}
    restore_heads = {(0, 0), (1, 1)}
    hooks = make_corrupt_then_restore_hooks(n_layers, n_heads, restore_heads, mean_z, clean_z)
    assert len(hooks) == n_layers


def test_full_corrupt_changes_output(gpt2_model):
    tokens = gpt2_model.to_tokens("When Mary and John went to the store, John gave a drink to")
    with torch.no_grad():
        clean_logits = gpt2_model(tokens)
        n_layers = gpt2_model.cfg.n_layers
        mean_z = torch.zeros(n_layers, gpt2_model.cfg.n_heads, gpt2_model.cfg.d_head)
        hooks = make_full_corrupt_hooks(n_layers, mean_z)
        corrupt_logits = gpt2_model.run_with_hooks(tokens, fwd_hooks=hooks)
    diff = (clean_logits[0, -1] - corrupt_logits[0, -1]).abs().max().item()
    assert diff > 0.1


def test_run_corrupt_restore_returns_results(gpt2_model):
    results = run_corrupt_restore(gpt2_model, [TASK], n_prompts=3, n_random_baselines=2)
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "C20.corrupt_restore"
    assert r.n_samples >= 1


def test_run_corrupt_restore_metadata_keys(gpt2_model):
    results = run_corrupt_restore(gpt2_model, [TASK], n_prompts=3, n_random_baselines=2)
    r = results[0]
    expected_keys = {
        "task", "mean_clean_ld", "mean_corrupt_ld",
        "mean_restored_ld", "restoration_rate",
        "per_head_restoration", "random_baseline_std",
        "n_circuit_heads", "circuit_heads",
    }
    assert expected_keys.issubset(r.metadata.keys())


def test_run_corrupt_restore_restoration_rate_is_value(gpt2_model):
    results = run_corrupt_restore(gpt2_model, [TASK], n_prompts=3, n_random_baselines=2)
    r = results[0]
    assert r.value == pytest.approx(r.metadata["restoration_rate"])


def test_run_corrupt_restore_has_baseline(gpt2_model):
    results = run_corrupt_restore(gpt2_model, [TASK], n_prompts=3, n_random_baselines=2)
    r = results[0]
    assert r.baseline_random is not None
    assert isinstance(r.baseline_random, float)


def test_run_corrupt_restore_per_head_restoration_keys(gpt2_model):
    results = run_corrupt_restore(gpt2_model, [TASK], n_prompts=3, n_random_baselines=2)
    r = results[0]
    per_head = r.metadata["per_head_restoration"]
    assert isinstance(per_head, dict)
    assert len(per_head) == r.metadata["n_circuit_heads"]


def test_run_corrupt_restore_unknown_task_returns_empty(gpt2_model):
    results = run_corrupt_restore(gpt2_model, ["nonexistent_task_xyz"], n_prompts=3, n_random_baselines=2)
    assert results == []
