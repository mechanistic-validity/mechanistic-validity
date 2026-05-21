import importlib.util
import sys
from pathlib import Path

import pytest

from mechval.metrics.common import EvalResult, load_model

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "core" / "behavioral" / "logit_diff_recovery" / "20_corrupt_restore.py"
)
_spec = importlib.util.spec_from_file_location("corrupt_restore_behavioral", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["corrupt_restore_behavioral"] = _mod
_spec.loader.exec_module(_mod)

run_corrupt_restore = _mod.run_corrupt_restore
cache_clean_z = _mod.cache_clean_z
make_full_corrupt_hooks = _mod.make_full_corrupt_hooks
make_corrupt_then_restore_hooks = _mod.make_corrupt_then_restore_hooks

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def results(gpt2_model):
    return run_corrupt_restore(gpt2_model, [TASK], n_prompts=5, n_random_baselines=3)


def test_returns_results(results):
    assert len(results) >= 1


def test_result_type(results):
    assert isinstance(results[0], EvalResult)


def test_metric_id(results):
    assert results[0].metric_id == "C20.corrupt_restore"


def test_value_is_finite(results):
    import math
    assert math.isfinite(results[0].value)


def test_metadata_has_task(results):
    assert results[0].metadata["task"] == TASK


def test_n_samples_positive(results):
    assert results[0].n_samples > 0


def test_metadata_has_expected_keys(results):
    expected = {"task", "mean_clean_ld", "mean_corrupt_ld",
                "mean_restored_ld", "restoration_rate",
                "per_head_restoration", "random_baseline_std",
                "n_circuit_heads", "circuit_heads"}
    assert expected.issubset(results[0].metadata.keys())


def test_restoration_rate_matches_value(results):
    assert results[0].value == pytest.approx(results[0].metadata["restoration_rate"])


def test_has_baseline_random(results):
    assert results[0].baseline_random is not None
    assert isinstance(results[0].baseline_random, float)


def test_clean_ld_greater_than_corrupt_ld(results):
    meta = results[0].metadata
    assert meta["mean_clean_ld"] > meta["mean_corrupt_ld"]


def test_per_head_restoration_matches_circuit_size(results):
    meta = results[0].metadata
    assert len(meta["per_head_restoration"]) == meta["n_circuit_heads"]


def test_cache_clean_z_captures_all_layers(gpt2_model):
    import torch
    tokens = gpt2_model.to_tokens("When Mary and John went to the store")
    with torch.no_grad():
        clean_z = cache_clean_z(gpt2_model, tokens)
    assert len(clean_z) == gpt2_model.cfg.n_layers


def test_unknown_task_returns_empty(gpt2_model):
    results = run_corrupt_restore(gpt2_model, ["nonexistent_xyz"], n_prompts=5, n_random_baselines=2)
    assert results == []
