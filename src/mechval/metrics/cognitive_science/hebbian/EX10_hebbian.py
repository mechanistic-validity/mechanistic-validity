"""Hebbian Correlation — Fire Together, Wire Together
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX10 — Hebbian Correlation
Categories:     structural, hebbian
Evidence family: structural
Validity layer: Construct

Tests whether circuit head pairs in adjacent layers show Hebbian-like
structure: heads that are strongly "wired" together (high composition
scores) should also have correlated functional roles (similar output
directions in logit space).

Background:
    Hebb's rule (Hebb 1949) states that neurons that fire together
    wire together — synaptic connections strengthen between
    co-activated neurons. Caporale & Dan (2008) review modern evidence
    for spike-timing-dependent plasticity (STDP), the biological
    mechanism underlying Hebb's postulate.

    Applied to circuits: the "wiring" between two attention heads is
    their composition score (how much one head reads from the other's
    output). The "functional similarity" is the cosine similarity of
    their output directions projected through the unembedding matrix.
    If the circuit's wiring follows Hebbian principles, strongly wired
    pairs should have similar functional roles.

Method (weight-space only, no forward pass):
    1. For each pair of circuit heads (A in layer L1, B in layer L2,
       L1 < L2):
       - OV composition: ||W_V[B] @ W_O[A]||_F / (||W_V[B]||_F * ||W_O[A]||_F)
       - QK composition: ||W_Q[B] @ W_K[A].T||_F / (||W_Q[B]||_F * ||W_K[A]||_F)
    2. Wiring strength W_ij = OV_comp(i, j) + QK_comp(i, j)
    3. Functional similarity: cosine(W_O[A].mean_over_heads @ W_U,
                                     W_O[B].mean_over_heads @ W_U)
       where W_U is the unembedding matrix.
    4. Hebbian score = Pearson correlation(wiring_strength,
                                           functional_similarity)
    5. Pass: hebbian_score > 0.3

Refs: Hebb 1949; Caporale & Dan 2008

Usage:
    uv run python EX10_hebbian.py --tasks ioi --device cpu
    uv run python EX10_hebbian.py --tasks ioi greater_than
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Hebbian Correlation (Fire Together, Wire Together)",
    paper_ref="Hebb 1949; Caporale & Dan 2008",
    paper_cite="Hebb 1949, The Organization of Behavior; Caporale & Dan 2008, Spike timing-dependent plasticity: a Hebbian learning rule, Annu Rev Neurosci 31",
    description="Tests whether strongly wired circuit head pairs also have similar functional roles in logit space, following Hebbian associative principles",
    category="structural",
    tier="cogsci",
    origin="established",
)

HEBBIAN_THRESHOLD = 0.3


@torch.no_grad()
def compute_ov_composition(model, head_a: tuple[int, int],
                           head_b: tuple[int, int]) -> float:
    """OV composition score: how much head B reads from head A's output.

    score = ||W_V[B] @ W_O[A]||_F / (||W_V[B]||_F * ||W_O[A]||_F)
    """
    la, ha = head_a
    lb, hb = head_b
    W_O_a = model.blocks[la].attn.W_O[ha].float()  # (d_head, d_model)
    W_V_b = model.blocks[lb].attn.W_V[hb].float()  # (d_model, d_head)

    norm_o = W_O_a.norm()
    norm_v = W_V_b.norm()
    if norm_o < 1e-10 or norm_v < 1e-10:
        return 0.0

    product = W_V_b @ W_O_a  # (d_head, d_head) — via (d_model, d_head) @ (d_head, d_model)
    return (product.norm() / (norm_v * norm_o)).item()


@torch.no_grad()
def compute_qk_composition(model, head_a: tuple[int, int],
                           head_b: tuple[int, int]) -> float:
    """QK composition score: how much head B attends to positions head A attended to.

    score = ||W_Q[B] @ W_K[A].T||_F / (||W_Q[B]||_F * ||W_K[A]||_F)
    """
    la, ha = head_a
    lb, hb = head_b
    W_K_a = model.blocks[la].attn.W_K[ha].float()  # (d_model, d_head)
    W_Q_b = model.blocks[lb].attn.W_Q[hb].float()  # (d_model, d_head)

    norm_k = W_K_a.norm()
    norm_q = W_Q_b.norm()
    if norm_k < 1e-10 or norm_q < 1e-10:
        return 0.0

    product = W_Q_b.T @ W_K_a  # (d_head, d_head) — via (d_head, d_model) @ (d_model, d_head)
    return (product.norm() / (norm_q * norm_k)).item()


@torch.no_grad()
def compute_functional_similarity(model, head_a: tuple[int, int],
                                  head_b: tuple[int, int]) -> float:
    """Functional similarity via unembedding projection.

    For each head, compute the mean output direction projected through W_U,
    then return cosine similarity between the two.
    """
    la, ha = head_a
    lb, hb = head_b
    W_O_a = model.blocks[la].attn.W_O[ha].float()  # (d_head, d_model)
    W_O_b = model.blocks[lb].attn.W_O[hb].float()  # (d_head, d_model)
    W_U = model.W_U.float()  # (d_model, d_vocab)

    # Project each head's output through unembedding: mean over d_head dimension
    # W_O is (d_head, d_model), so W_O @ W_U gives (d_head, d_vocab)
    # We take the mean over d_head to get a single (d_vocab,) direction
    proj_a = (W_O_a @ W_U).mean(dim=0)  # (d_vocab,)
    proj_b = (W_O_b @ W_U).mean(dim=0)  # (d_vocab,)

    norm_a = proj_a.norm()
    norm_b = proj_b.norm()
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return (torch.dot(proj_a, proj_b) / (norm_a * norm_b)).item()


@torch.no_grad()
def run_hebbian(model, tasks: list[str]) -> list[EvalResult]:
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        sorted_heads = sorted(circuit_heads)
        log(f"  {task}: {len(circuit_heads)} heads")

        # Build all cross-layer pairs (A in earlier layer, B in later layer)
        pairs = []
        for i, head_a in enumerate(sorted_heads):
            for head_b in sorted_heads[i + 1:]:
                if head_a[0] < head_b[0]:
                    pairs.append((head_a, head_b))

        if len(pairs) < 3:
            log(f"  {task}: only {len(pairs)} cross-layer pairs, need >= 3 for correlation, skipping")
            continue

        wiring_strengths = []
        functional_similarities = []
        pair_details = []

        for head_a, head_b in pairs:
            ov_comp = compute_ov_composition(model, head_a, head_b)
            qk_comp = compute_qk_composition(model, head_a, head_b)
            wiring = ov_comp + qk_comp

            func_sim = compute_functional_similarity(model, head_a, head_b)

            wiring_strengths.append(wiring)
            functional_similarities.append(func_sim)
            pair_details.append({
                "sender": f"L{head_a[0]}H{head_a[1]}",
                "receiver": f"L{head_b[0]}H{head_b[1]}",
                "ov_composition": ov_comp,
                "qk_composition": qk_comp,
                "wiring_strength": wiring,
                "functional_similarity": func_sim,
            })

        wiring_arr = np.array(wiring_strengths)
        func_arr = np.array(functional_similarities)

        # Pearson correlation between wiring strength and functional similarity
        if np.std(wiring_arr) < 1e-10 or np.std(func_arr) < 1e-10:
            hebbian_score = 0.0
        else:
            hebbian_score = float(np.corrcoef(wiring_arr, func_arr)[0, 1])

        passed = hebbian_score > HEBBIAN_THRESHOLD

        log(f"    {len(pairs)} cross-layer pairs")
        log(f"    mean_wiring={float(np.mean(wiring_arr)):.4f}  "
            f"mean_func_sim={float(np.mean(func_arr)):.4f}")
        log(f"    hebbian_score={hebbian_score:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        # Top wired pairs
        sorted_pairs = sorted(pair_details, key=lambda x: x["wiring_strength"], reverse=True)
        top_wired = sorted_pairs[:5]

        results.append(EvalResult(
            metric_id="EX10.hebbian",
            value=hebbian_score,
            n_samples=len(pairs),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "hebbian_score": hebbian_score,
                "n_pairs": len(pairs),
                "n_circuit_heads": len(circuit_heads),
                "mean_wiring_strength": float(np.mean(wiring_arr)),
                "mean_functional_similarity": float(np.mean(func_arr)),
                "std_wiring_strength": float(np.std(wiring_arr)),
                "std_functional_similarity": float(np.std(func_arr)),
                "top_wired_pairs": top_wired,
                "passed": passed,
                "threshold": HEBBIAN_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX10: Hebbian Correlation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX10: HEBBIAN CORRELATION (FIRE TOGETHER, WIRE TOGETHER)")
    log("=" * 60)

    out = args.out or "EX10_hebbian.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_hebbian(model, [task])
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
