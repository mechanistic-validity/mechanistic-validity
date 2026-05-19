import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechanistic_validity" / "metrics"
    / "causal" / "counterfactual_das" / "33_path_patching.py"
)
_spec = importlib.util.spec_from_file_location("path_patching_33", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["path_patching_33"] = _mod
_spec.loader.exec_module(_mod)

path_patch_edge = _mod.path_patch_edge
run_path_patching = _mod.run_path_patching

from mechanistic_validity.metrics.common import EvalResult, load_model

TASK = "ioi"


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_path_patching(gpt2_model, tasks=[TASK], n_prompts=3, n_random_baselines=2)


def test_run_path_patching_returns_results(circuit_results):
    assert len(circuit_results) >= 1
    for r in circuit_results:
        assert isinstance(r, EvalResult)


def test_run_path_patching_metric_id(circuit_results):
    for r in circuit_results:
        assert r.metric_id == "C33.path_patching"


def test_run_path_patching_value_is_finite(circuit_results):
    for r in circuit_results:
        assert np.isfinite(r.value)


def test_run_path_patching_has_edge_effects(circuit_results):
    for r in circuit_results:
        assert "edge_effects" in r.metadata
        assert isinstance(r.metadata["edge_effects"], dict)


def test_run_path_patching_has_top_edges(circuit_results):
    for r in circuit_results:
        assert "top_edges" in r.metadata
        assert isinstance(r.metadata["top_edges"], list)


def test_run_path_patching_has_baseline(circuit_results):
    for r in circuit_results:
        assert r.baseline_random is not None


def test_run_path_patching_metadata_task(circuit_results):
    for r in circuit_results:
        assert r.metadata["task"] == TASK


def test_run_path_patching_n_edges_positive(circuit_results):
    for r in circuit_results:
        assert r.metadata["n_edges"] > 0


def test_path_patch_edge_returns_finite(gpt2_model):
    from mechanistic_validity.metrics.common import (
        calibrate_mean_z,
        generate_prompts,
        get_token_ids,
    )

    prompts = generate_prompts(TASK, gpt2_model.tokenizer, n_prompts=2)
    correct_ids, incorrect_ids = get_token_ids(prompts, gpt2_model.tokenizer)
    mean_z = calibrate_mean_z(gpt2_model, prompts, n_calibration=2)

    clean_tokens = gpt2_model.to_tokens(prompts[0].text)
    corrupt_tokens = gpt2_model.to_tokens(prompts[1].text)

    effect = path_patch_edge(
        gpt2_model, clean_tokens, corrupt_tokens,
        correct_ids[0], incorrect_ids[0],
        src_layer=0, src_head=0, dst_layer=1, dst_head=0,
        mean_z=mean_z,
    )
    assert np.isfinite(effect)
