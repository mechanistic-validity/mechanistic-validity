import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "causal" / "woodward" / "35_resample_complement.py"
)
_spec = importlib.util.spec_from_file_location("resample_complement_35", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["resample_complement_35"] = _mod
_spec.loader.exec_module(_mod)

run_resample_complement = _mod.run_resample_complement
compute_resample_faithfulness = _mod.compute_resample_faithfulness
cache_all_z = _mod.cache_all_z
make_resample_hooks = _mod.make_resample_hooks

from mechval.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_resample_complement(gpt2_model, tasks=[TASK], n_prompts=4, n_resample_trials=2)


def test_run_returns_eval_results(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "C35.resample_complement"


def test_value_equals_resample_faithfulness(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["resample_faithfulness"])


def test_metadata_keys(circuit_results):
    meta = circuit_results[0].metadata
    expected = {"task", "resample_faithfulness", "resample_std",
                "mean_ablation_faithfulness", "delta", "per_trial",
                "n_resample_trials", "n_circuit_heads", "circuit_heads"}
    assert set(meta.keys()) == expected


def test_delta_is_difference(circuit_results):
    meta = circuit_results[0].metadata
    expected_delta = meta["resample_faithfulness"] - meta["mean_ablation_faithfulness"]
    assert meta["delta"] == pytest.approx(expected_delta)


def test_per_trial_length(circuit_results):
    meta = circuit_results[0].metadata
    assert len(meta["per_trial"]) == meta["n_resample_trials"]


def test_n_samples_positive(circuit_results):
    assert circuit_results[0].n_samples > 0


def test_make_resample_hooks_returns_list():
    dummy_z = {0: torch.randn(5, 2, 16)}
    hooks = make_resample_hooks({0: [0, 1]}, dummy_z)
    assert isinstance(hooks, list)
    assert len(hooks) == 1
    assert hooks[0][0] == "blocks.0.attn.hook_z"


def test_task_metadata(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK
