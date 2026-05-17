"""OV/QK Composition Scores (Elhage et al. 2021)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     B03 — OV/QK Decomposition
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit edges are structurally privileged in OV/QK composition space
Requires:       CPU, model weights only
Doc:            /instruments_v2/structural/b03-ov-qk-analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures how strongly one head's OV output aligns with another head's QK
input, quantifying composition between head pairs. For each pair (h1 in
layer L1, h2 in layer L2, L1 < L2), computes:

    score = ||W_OV_h1 @ W_QK_h2||_F / (||W_OV_h1||_F * ||W_QK_h2||_F)

Compares composition scores for circuit edges (sender->receiver pathway
pairs) vs non-circuit edges. Higher ratio means the circuit's wiring is
structurally privileged in weight space.

Usage:
    uv run python 49_ov_qk_composition.py --tasks ioi greater_than
    uv run python 49_ov_qk_composition.py --device cpu
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "causal"))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_all_edges,
    get_circuit,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_composition_matrix(model) -> np.ndarray:
    """Compute pairwise OV->QK composition scores for all head pairs.

    Returns array of shape (n_layers, n_heads, n_layers, n_heads) where
    entry [L1, H1, L2, H2] is the composition score from h1=(L1,H1) to
    h2=(L2,H2). Only valid when L1 < L2.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Precompute OV and QK matrices
    W_OV = {}  # (L, H) -> (d_model, d_model)
    W_QK = {}  # (L, H) -> (d_model, d_model)

    for L in range(n_layers):
        W_V = model.blocks[L].attn.W_V  # (n_heads, d_model, d_head)
        W_O = model.blocks[L].attn.W_O  # (n_heads, d_head, d_model)
        W_Q = model.blocks[L].attn.W_Q  # (n_heads, d_model, d_head)
        W_K = model.blocks[L].attn.W_K  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            W_OV[(L, H)] = (W_V[H] @ W_O[H]).float()  # (d_model, d_model)
            W_QK[(L, H)] = (W_Q[H] @ W_K[H].T).float()  # (d_model, d_model)

    # Precompute Frobenius norms
    ov_norms = {k: v.norm().item() for k, v in W_OV.items()}
    qk_norms = {k: v.norm().item() for k, v in W_QK.items()}

    scores = np.zeros((n_layers, n_heads, n_layers, n_heads))
    for L1 in range(n_layers):
        for H1 in range(n_heads):
            ov = W_OV[(L1, H1)]
            ov_n = ov_norms[(L1, H1)]
            if ov_n < 1e-10:
                continue
            for L2 in range(L1 + 1, n_layers):
                for H2 in range(n_heads):
                    qk = W_QK[(L2, H2)]
                    qk_n = qk_norms[(L2, H2)]
                    if qk_n < 1e-10:
                        continue
                    product = ov @ qk
                    scores[L1, H1, L2, H2] = product.norm().item() / (ov_n * qk_n)

    return scores


@torch.no_grad()
def main():
    parser = parse_common_args("B49: OV/QK Composition Scores")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B49: OV/QK COMPOSITION SCORES")
    log("=" * 60)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    log("Computing pairwise composition matrix...")
    comp_matrix = compute_composition_matrix(model)

    results = []
    for task in tasks:
        circuit = get_circuit(task)
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        circuit_edges = get_all_edges(circuit)
        log(f"  {task}: {len(circuit_heads)} heads, {len(circuit_edges)} edges")

        # Composition scores for circuit edges
        circuit_scores = []
        for L1, H1, L2, H2 in circuit_edges:
            circuit_scores.append(comp_matrix[L1, H1, L2, H2])

        # Composition scores for non-circuit edges (all L1<L2 pairs not in circuit)
        non_circuit_scores = []
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
        for i, (L1, H1) in enumerate(all_heads):
            for L2, H2 in all_heads:
                if L2 <= L1:
                    continue
                if (L1, H1, L2, H2) not in circuit_edges:
                    non_circuit_scores.append(comp_matrix[L1, H1, L2, H2])

        mean_circuit = float(np.mean(circuit_scores)) if circuit_scores else 0.0
        mean_non_circuit = float(np.mean(non_circuit_scores)) if non_circuit_scores else 0.0
        ratio = mean_circuit / (mean_non_circuit + 1e-10)

        log(f"    Circuit edges: {mean_circuit:.4f}  Non-circuit: {mean_non_circuit:.4f}  "
            f"Ratio: {ratio:.3f}")

        results.append(EvalResult(
            metric_id="B49.ov_qk_composition",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(circuit_scores),
            metadata={
                "task": task,
                "circuit_mean": mean_circuit,
                "non_circuit_mean": mean_non_circuit,
                "ratio": ratio,
                "n_circuit_edges": len(circuit_edges),
                "circuit_scores_std": float(np.std(circuit_scores)) if circuit_scores else 0.0,
            },
        ))

    out = args.out or "49_ov_qk_composition.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
