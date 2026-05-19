import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic_validity.metrics.common import EvalResult

_MOTIF_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mechanistic_validity"
    / "metrics"
    / "structural"
    / "network_motifs"
    / "97_network_motifs.py"
)
_spec = importlib.util.spec_from_file_location("_network_motifs", _MOTIF_PATH)
_motif_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _motif_mod
_spec.loader.exec_module(_motif_mod)

_edges_to_adjacency = _motif_mod._edges_to_adjacency
count_triads = _motif_mod.count_triads
count_bifan = _motif_mod.count_bifan
count_all_motifs = _motif_mod.count_all_motifs
generate_random_graph_edges = _motif_mod.generate_random_graph_edges
run_network_motifs = _motif_mod.run_network_motifs


# ── Adjacency building ──────────────────────────────────────────────

def test_edges_to_adjacency_simple():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C")}
    adj = _edges_to_adjacency(nodes, edges)
    assert adj["A"] == {"B"}
    assert adj["B"] == {"C"}
    assert adj["C"] == set()


def test_edges_to_adjacency_ignores_unknown_sources():
    nodes = ["A", "B"]
    edges = {("X", "B"), ("A", "B")}
    adj = _edges_to_adjacency(nodes, edges)
    assert adj["A"] == {"B"}
    assert adj["B"] == set()


# ── Feed-forward chain: A->B->C ─────────────────────────────────────

def test_chain_simple():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C")}
    counts = count_all_motifs(nodes, edges)
    assert counts["feed_forward_chain"] == 1


def test_chain_with_shortcut_is_not_chain():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C"), ("A", "C")}
    counts = count_all_motifs(nodes, edges)
    assert counts["feed_forward_chain"] == 0


def test_chain_two_paths():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("B", "C"), ("C", "D")}
    counts = count_all_motifs(nodes, edges)
    # A->B->C is a chain (A not connected to C)
    # B->C->D is a chain (B not connected to D)
    # A->B->C->D also yields: A->B not connected to D doesn't form direct chain of length 2
    assert counts["feed_forward_chain"] == 2


# ── Fan-out: A->B, A->C ─────────────────────────────────────────────

def test_fan_out_simple():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("A", "C")}
    counts = count_all_motifs(nodes, edges)
    assert counts["fan_out"] == 1


def test_fan_out_three_targets():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("A", "C"), ("A", "D")}
    counts = count_all_motifs(nodes, edges)
    # C(3,2) = 3 pairs of targets
    assert counts["fan_out"] == 3


def test_no_fan_out_single_target():
    nodes = ["A", "B"]
    edges = {("A", "B")}
    counts = count_all_motifs(nodes, edges)
    assert counts["fan_out"] == 0


# ── Fan-in: A->C, B->C ──────────────────────────────────────────────

def test_fan_in_simple():
    nodes = ["A", "B", "C"]
    edges = {("A", "C"), ("B", "C")}
    counts = count_all_motifs(nodes, edges)
    assert counts["fan_in"] == 1


def test_fan_in_three_sources():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "D"), ("B", "D"), ("C", "D")}
    counts = count_all_motifs(nodes, edges)
    # C(3,2) = 3 pairs of sources
    assert counts["fan_in"] == 3


# ── Bi-fan: A->C, A->D, B->C, B->D ─────────────────────────────────

def test_bifan_simple():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
    adj = _edges_to_adjacency(nodes, edges)
    assert count_bifan(nodes, adj) == 1


def test_bifan_three_shared_targets():
    nodes = ["A", "B", "C", "D", "E"]
    edges = {("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")}
    adj = _edges_to_adjacency(nodes, edges)
    # A and B share 3 targets -> C(3,2) = 3 bi-fan instances
    assert count_bifan(nodes, adj) == 3


def test_bifan_no_shared_targets():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "C"), ("B", "D")}
    adj = _edges_to_adjacency(nodes, edges)
    assert count_bifan(nodes, adj) == 0


# ── Combined motif counting ─────────────────────────────────────────

def test_count_all_motifs_empty_graph():
    nodes = ["A", "B", "C"]
    edges = set()
    counts = count_all_motifs(nodes, edges)
    assert counts["feed_forward_chain"] == 0
    assert counts["fan_in"] == 0
    assert counts["fan_out"] == 0
    assert counts["bi_fan"] == 0


def test_count_all_motifs_complete_triangle():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C"), ("A", "C"), ("B", "A"), ("C", "A"), ("C", "B")}
    counts = count_all_motifs(nodes, edges)
    # Every node has 2 targets -> fan_out = C(2,2)*3 = 3
    assert counts["fan_out"] == 3
    # Every node has 2 sources -> fan_in = C(2,2)*3 = 3
    assert counts["fan_in"] == 3
    # For chain A->B->C, need A not connected to C; but A IS connected to C, so 0
    assert counts["feed_forward_chain"] == 0


# ── Random graph generation ─────────────────────────────────────────

def test_random_graph_correct_edge_count():
    nodes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(123)
    for target_edges in [0, 3, 7, 20]:
        max_possible = len(nodes) * (len(nodes) - 1)  # 20
        expected = min(target_edges, max_possible)
        edges = generate_random_graph_edges(nodes, target_edges, rng)
        assert len(edges) == expected


def test_random_graph_no_self_loops():
    nodes = list(range(10))
    rng = np.random.default_rng(456)
    edges = generate_random_graph_edges(nodes, 30, rng)
    for a, b in edges:
        assert a != b


def test_random_graph_too_few_nodes():
    nodes = ["A"]
    rng = np.random.default_rng(789)
    edges = generate_random_graph_edges(nodes, 5, rng)
    assert len(edges) == 0


TASK = "ioi"


# ── Full run ─────────────────────────────────────────────────────────

def test_run_network_motifs_returns_result():
    results = run_network_motifs([TASK], n_random=50)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.metric_id == "G5.network_motif_enrichment"
    assert r.n_samples == 50
    assert r.metadata["task"] == TASK
    assert r.metadata["n_nodes"] >= 3
    assert r.metadata["n_edges"] >= 1
    assert "motifs" in r.metadata
    motifs = r.metadata["motifs"]
    for name in ["feed_forward_chain", "fan_in", "fan_out", "bi_fan"]:
        assert name in motifs
        assert "z_score" in motifs[name]
        assert "observed" in motifs[name]


def test_run_network_motifs_z_scores_are_finite():
    results = run_network_motifs([TASK], n_random=50)
    r = results[0]
    motifs = r.metadata["motifs"]
    for name, data in motifs.items():
        z = data["z_score"]
        assert math.isfinite(z), f"z-score for {name} is not finite: {z}"
    assert math.isfinite(r.value)


def test_run_network_motifs_observed_counts_nonneg():
    results = run_network_motifs([TASK], n_random=50)
    r = results[0]
    motifs = r.metadata["motifs"]
    for name, data in motifs.items():
        assert data["observed"] >= 0


def test_run_network_motifs_value_is_best_z():
    results = run_network_motifs([TASK], n_random=50)
    r = results[0]
    motifs = r.metadata["motifs"]
    best_z = max(m["z_score"] for m in motifs.values())
    assert r.value == pytest.approx(best_z)


def test_run_network_motifs_unknown_task():
    results = run_network_motifs(["nonexistent_task_xyz"])
    assert results == []
