"""K-Alignment with Embedding Directions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Categories:     structural
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads' W_K subspace aligns with token embedding
                directions more than non-circuit heads
Requires:       CPU, model weights only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each head, compute the top-k right singular vectors of W_QK, then
measure alignment (max cosine similarity) with the mean embedding
direction of the model's vocabulary.

Circuit heads that attend to specific token types should show higher
alignment between their QK subspace and the embedding space, since
the keys they match against are functions of token embeddings.

Also reports OV alignment: top SVD directions of W_OV projected onto
the unembedding matrix W_U.

Usage:
    uv run python B14_k_alignment.py --tasks ioi sva
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_completed_tasks,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

JSONL_FILE = "B14_k_alignment.jsonl"


@torch.no_grad()
def compute_alignment_scores(model, k_dirs: int = 3) -> dict[tuple[int, int], dict]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_E = model.W_E.float()  # (d_vocab, d_model)
    W_U = model.W_U.float()  # (d_model, d_vocab)

    embed_mean = W_E.mean(dim=0)
    embed_mean = embed_mean / (embed_mean.norm() + 1e-10)

    results = {}
    for L in range(n_layers):
        W_Q = model.W_Q[L].float()
        W_K = model.W_K[L].float()
        W_V = model.W_V[L].float()
        W_O = model.W_O[L].float()

        for H in range(n_heads):
            W_QK = W_Q[H] @ W_K[H].T  # (d_model, d_model)
            _, _, Vt_qk = torch.linalg.svd(W_QK, full_matrices=False)
            top_k_dirs = Vt_qk[:k_dirs]  # (k, d_model) — right singular vectors

            qk_align = float(torch.abs(top_k_dirs @ embed_mean).max())

            W_OV = W_V[H] @ W_O[H]  # (d_model, d_model)
            U_ov, _, _ = torch.linalg.svd(W_OV, full_matrices=False)
            top_ov = U_ov[:, :k_dirs]  # (d_model, k) — left singular vectors

            ov_proj = W_U.T @ top_ov  # (d_vocab, k)
            ov_norms = ov_proj.norm(dim=0)  # (k,)
            ov_align = float(ov_norms.max() / (W_U.norm() + 1e-10))

            results[(L, H)] = {
                "qk_embed_alignment": qk_align,
                "ov_unembed_alignment": ov_align,
            }

    return results


def run(model=None, tasks: list[str] | None = None, resume: bool = True) -> list[EvalResult]:
    if tasks is None:
        tasks = CIRCUIT_TASKS

    completed, prior = load_completed_tasks(JSONL_FILE) if resume else (set(), [])
    results = list(prior)

    scores = compute_alignment_scores(model)
    all_heads = set(scores.keys())

    for task in tasks:
        if task in completed:
            continue

        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        circuit_set = set(circuit_heads)
        non_circuit = all_heads - circuit_set

        circ_scores = [scores[h] for h in circuit_set if h in scores]
        bg_scores = [scores[h] for h in non_circuit if h in scores]

        if not circ_scores or not bg_scores:
            continue

        circ_qk = float(np.mean([s["qk_embed_alignment"] for s in circ_scores]))
        bg_qk = float(np.mean([s["qk_embed_alignment"] for s in bg_scores]))
        circ_ov = float(np.mean([s["ov_unembed_alignment"] for s in circ_scores]))
        bg_ov = float(np.mean([s["ov_unembed_alignment"] for s in bg_scores]))

        log(f"  {task}: {len(circuit_heads)} circuit heads")
        log(f"    QK-embed align: circuit={circ_qk:.4f}  bg={bg_qk:.4f}  ratio={circ_qk/bg_qk:.3f}")
        log(f"    OV-unembed align: circuit={circ_ov:.6f}  bg={bg_ov:.6f}  ratio={circ_ov/bg_ov:.3f}")

        r = EvalResult(
            metric_id="B14.k_alignment",
            value=circ_qk / bg_qk if bg_qk > 0 else 0.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "n_circuit_heads": len(circuit_heads),
                "circuit_qk_embed_alignment": circ_qk,
                "background_qk_embed_alignment": bg_qk,
                "qk_alignment_ratio": circ_qk / bg_qk if bg_qk > 0 else 0.0,
                "circuit_ov_unembed_alignment": circ_ov,
                "background_ov_unembed_alignment": bg_ov,
                "ov_alignment_ratio": circ_ov / bg_ov if bg_ov > 0 else 0.0,
            },
        )
        results.append(r)
        save_incremental(r, JSONL_FILE)

    return results


def main():
    parser = parse_common_args("B14: K-Alignment with Embedding Directions")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("B14: K-ALIGNMENT WITH EMBEDDING DIRECTIONS")
    log("=" * 60)

    results = run(model=model, tasks=tasks)
    save_results(results, args.out or "B14_k_alignment.json", args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
