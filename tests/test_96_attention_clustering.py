import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult, load_model

_CLUSTERING_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "metrics"
    / "structural"
    / "attention_clustering"
    / "96_attention_clustering.py"
)
_spec = importlib.util.spec_from_file_location("_attn_clustering", _CLUSTERING_PATH)
_attn_clustering_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _attn_clustering_mod
_spec.loader.exec_module(_attn_clustering_mod)

run_attention_clustering = _attn_clustering_mod.run_attention_clustering
compute_cluster_purity = _attn_clustering_mod.compute_cluster_purity


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model("gpt2", "cpu")


def test_compute_cluster_purity_all_in_one_cluster():
    labels = np.array([0, 0, 0, 1, 1])
    circuit_mask = np.array([True, True, True, False, False])
    assert compute_cluster_purity(labels, circuit_mask) == pytest.approx(1.0)


def test_compute_cluster_purity_split_across_clusters():
    labels = np.array([0, 1, 0, 1, 2])
    circuit_mask = np.array([True, True, True, False, False])
    assert compute_cluster_purity(labels, circuit_mask) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_compute_cluster_purity_no_circuit_heads():
    labels = np.array([0, 1, 2])
    circuit_mask = np.array([False, False, False])
    assert compute_cluster_purity(labels, circuit_mask) == pytest.approx(0.0)


TASK = "ioi"


@pytest.fixture(scope="module")
def circuit_results(gpt2_model):
    return run_attention_clustering(gpt2_model, [TASK], n_prompts=5)


def test_run_attention_clustering_returns_result(circuit_results):
    assert len(circuit_results) == 1
    r = circuit_results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "S96.attention_clustering"
    assert r.n_samples == 5

    sil = r.metadata["silhouette_score"]
    assert -1.0 <= sil <= 1.0

    purity = r.metadata["cluster_purity"]
    assert 0.0 <= purity <= 1.0


def test_run_attention_clustering_value_equals_silhouette(circuit_results):
    r = circuit_results[0]
    assert r.value == pytest.approx(r.metadata["silhouette_score"])


def test_run_attention_clustering_metadata_fields(circuit_results, gpt2_model):
    r = circuit_results[0]
    meta = r.metadata
    assert meta["task"] == TASK
    assert meta["n_circuit_heads"] >= 2
    assert meta["n_total_heads"] == gpt2_model.cfg.n_layers * gpt2_model.cfg.n_heads
    assert meta["k_clusters"] == meta["n_circuit_heads"]
    assert isinstance(meta["passed"], bool)
    assert meta["threshold"] == pytest.approx(0.1)
    assert isinstance(meta["circuit_heads"], list)
    assert len(meta["circuit_heads"]) == meta["n_circuit_heads"]
