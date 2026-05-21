import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mechval.metrics.common import EvalResult

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "mechval" / "metrics"
    / "structural" / "network_motifs" / "G7_motif_enrichment.py"
)
_spec = importlib.util.spec_from_file_location("G7_motif_enrichment", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["G7_motif_enrichment"] = _mod
_spec.loader.exec_module(_mod)

triad_census = _mod.triad_census
count_bifan = _mod.count_bifan
degree_preserving_rewire = _mod.degree_preserving_rewire
run_motif_enrichment = _mod.run_motif_enrichment
TRIAD_PATTERNS = _mod.TRIAD_PATTERNS

TASK = "ioi"


# -- Triad census --

def test_triad_census_empty_graph():
    nodes = ["A", "B", "C"]
    counts = triad_census(nodes, set())
    assert counts["003"] == 1
    total_non_003 = sum(v for k, v in counts.items() if k != "003")
    assert total_non_003 == 0


def test_triad_census_single_edge():
    nodes = ["A", "B", "C"]
    edges = {("A", "B")}
    counts = triad_census(nodes, edges)
    assert counts["012"] == 1
    assert counts["003"] == 0


def test_triad_census_mutual_edge():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "A")}
    counts = triad_census(nodes, edges)
    assert counts["102"] == 1


def test_triad_census_fan_out():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("A", "C")}
    counts = triad_census(nodes, edges)
    assert counts["021D"] == 1


def test_triad_census_fan_in():
    nodes = ["A", "B", "C"]
    edges = {("B", "A"), ("C", "A")}
    counts = triad_census(nodes, edges)
    assert counts["021U"] == 1


def test_triad_census_chain():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C")}
    counts = triad_census(nodes, edges)
    # A->B, B->C is a 021C (chain) pattern
    assert counts["021C"] == 1


def test_triad_census_cycle():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C"), ("C", "A")}
    counts = triad_census(nodes, edges)
    assert counts["030C"] == 1


def test_triad_census_transitive():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("A", "C"), ("B", "C")}
    counts = triad_census(nodes, edges)
    assert counts["030T"] == 1


def test_triad_census_complete_bidirectional():
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")}
    counts = triad_census(nodes, edges)
    assert counts["300"] == 1


def test_triad_census_total_is_c_n_3():
    nodes = list(range(6))
    rng = np.random.default_rng(42)
    edges = set()
    for i in range(6):
        for j in range(6):
            if i != j and rng.random() < 0.4:
                edges.add((i, j))
    counts = triad_census(nodes, edges)
    total = sum(counts.values())
    expected = 20  # C(6,3) = 20
    assert total == expected


def test_all_16_pattern_names_present():
    assert len(TRIAD_PATTERNS) == 16


# -- Bi-fan --

def test_bifan_simple():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
    assert count_bifan(nodes, edges) == 1


def test_bifan_three_shared():
    nodes = ["A", "B", "C", "D", "E"]
    edges = {("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")}
    assert count_bifan(nodes, edges) == 3


def test_bifan_no_shared():
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "C"), ("B", "D")}
    assert count_bifan(nodes, edges) == 0


# -- Degree-preserving rewire --

def test_rewire_preserves_edge_count():
    nodes = list(range(8))
    edges = {(i, j) for i in range(4) for j in range(4, 8)}
    rng = np.random.default_rng(42)
    rewired = degree_preserving_rewire(nodes, edges, 200, rng)
    assert len(rewired) == len(edges)


def test_rewire_no_self_loops():
    nodes = list(range(6))
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    rng = np.random.default_rng(99)
    rewired = degree_preserving_rewire(nodes, edges, 100, rng)
    for a, b in rewired:
        assert a != b


def test_rewire_few_edges():
    nodes = [0, 1]
    edges = {(0, 1)}
    rng = np.random.default_rng(7)
    rewired = degree_preserving_rewire(nodes, edges, 50, rng)
    assert len(rewired) == 1


# -- Full run --

@pytest.fixture(scope="module")
def circuit_results():
    return run_motif_enrichment(tasks=[TASK], n_random=30)


def test_run_returns_result(circuit_results):
    assert len(circuit_results) == 1
    assert isinstance(circuit_results[0], EvalResult)


def test_metric_id(circuit_results):
    assert circuit_results[0].metric_id == "G7.motif_enrichment"


def test_n_samples(circuit_results):
    assert circuit_results[0].n_samples == 30


def test_metadata_task(circuit_results):
    assert circuit_results[0].metadata["task"] == TASK


def test_z_profile_present(circuit_results):
    assert "z_profile" in circuit_results[0].metadata
    profile = circuit_results[0].metadata["z_profile"]
    assert len(profile) > 0


def test_motif_results_have_p_values(circuit_results):
    motifs = circuit_results[0].metadata["motifs"]
    for name, data in motifs.items():
        assert "p_value" in data
        assert 0.0 <= data["p_value"] <= 1.0


def test_z_scores_are_finite(circuit_results):
    motifs = circuit_results[0].metadata["motifs"]
    for name, data in motifs.items():
        assert math.isfinite(data["z_score"])


def test_value_is_best_z(circuit_results):
    r = circuit_results[0]
    motifs = r.metadata["motifs"]
    best_z = max(m["z_score"] for m in motifs.values())
    assert r.value == pytest.approx(best_z)


def test_unknown_task():
    results = run_motif_enrichment(tasks=["nonexistent_task_xyz"])
    assert results == []
