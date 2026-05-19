"""Circuit Subspace Alignment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E05 — Subspace Alignment
Categories:     representational, structural
Validity layer: Construct
Criteria:       C2/C5
Establishes:    Circuit heads share output subspaces aligned with the answer direction
Requires:       CPU, model weights only
Doc:            /instruments_v2/representational/e05-subspace-alignment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each circuit head, computes the subspace spanned by its OV output
across prompts (top-k singular vectors of the head's output matrix).
Then measures:

  (1) Pairwise alignment between circuit heads' output subspaces via
      principal angles and Grassmann distance.
  (2) Alignment of each head's subspace with the correct-incorrect
      unembedding direction -- heads more aligned with the answer
      direction contribute more directly to the behavioral output.

Heads sharing output subspace may be functionally redundant.

Usage:
    uv run python 64_subspace_alignment.py --tasks ioi sva --device cpu
    uv run python 64_subspace_alignment.py --device cuda --n-prompts 40
"""
from itertools import combinations

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.

    U, V: (d, k) orthonormal basis matrices.
    Returns array of cos(theta_i) values.
    """
    M = U.T @ V
    singular_values = np.linalg.svd(M, compute_uv=False)
    return np.clip(singular_values, 0.0, 1.0)


def grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Geodesic distance on the Grassmann manifold.

    d_G = sqrt(sum(theta_i^2)) where theta_i are principal angles.
    """
    cos_angles = principal_angles(U, V)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    return float(np.sqrt(np.sum(angles ** 2)))


def subspace_alignment_score(U: np.ndarray, V: np.ndarray) -> float:
    """Mean cos^2 of principal angles -- 1 = identical, 0 = orthogonal."""
    cos_angles = principal_angles(U, V)
    return float(np.mean(cos_angles ** 2))


@torch.no_grad()
def collect_head_outputs(model, prompts, layer: int, head: int) -> np.ndarray:
    """Collect OV output of a specific head at last token. Returns (n_prompts, d_model)."""
    outputs = []
    hook_name = f"blocks.{layer}.attn.hook_result"
    W_O = model.W_O[layer, head].float()  # (d_head, d_model)

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        # hook_result is already post-OV: (batch, seq, n_heads, d_head)
        head_out = cache[hook_name][0, -1, head]  # (d_head,)
        # Project through W_O to get contribution to residual stream
        out = (head_out.float() @ W_O).cpu().numpy()  # (d_model,)
        outputs.append(out)

    return np.stack(outputs, axis=0)


def get_head_subspace(head_outputs: np.ndarray, k: int = 5) -> np.ndarray:
    """Get top-k principal directions of head outputs in d_model space.

    Returns (d_model, k) orthonormal columns.
    """
    centered = head_outputs - head_outputs.mean(axis=0, keepdims=True)
    k = min(k, centered.shape[0], centered.shape[1])
    # SVD of (n_prompts, d_model): V^T rows are principal directions in d_model
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:k].T  # (d_model, k)


@torch.no_grad()
def main():
    parser = parse_common_args("E64: Circuit Subspace Alignment")
    parser.add_argument("--subspace-k", type=int, default=5,
                        help="Number of singular vectors per head subspace")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    tokenizer = model.tokenizer
    k = args.subspace_k

    log("=" * 60)
    log("E64: CIRCUIT SUBSPACE ALIGNMENT")
    log("=" * 60)

    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 2:
            log(f"  {task}: need >= 2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, args.n_prompts)
        if not prompts or len(prompts) < k + 1:
            log(f"  {task}: insufficient prompts")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)

        log(f"  {task}: {len(circuit_heads)} circuit heads, {len(prompts)} prompts")

        # Compute answer direction (mean across prompts)
        W_U = model.W_U.float()  # (d_model, d_vocab)
        answer_dirs = []
        for i in range(min(len(correct_ids), len(prompts))):
            d = W_U[:, correct_ids[i]] - W_U[:, incorrect_ids[i]]
            answer_dirs.append(d.cpu().numpy())
        mean_answer_dir = np.mean(answer_dirs, axis=0)
        mean_answer_dir = mean_answer_dir / (np.linalg.norm(mean_answer_dir) + 1e-10)

        # Collect subspaces for each circuit head
        head_subspaces = {}
        head_answer_alignment = {}

        sorted_heads = sorted(circuit_heads)
        for L, H in sorted_heads:
            outputs = collect_head_outputs(model, prompts, L, H)
            subspace = get_head_subspace(outputs, k=k)
            head_subspaces[(L, H)] = subspace

            # Alignment with answer direction: project answer_dir onto subspace
            proj = subspace @ (subspace.T @ mean_answer_dir)
            alignment = float(np.linalg.norm(proj))  # fraction of answer dir in subspace
            head_answer_alignment[(L, H)] = alignment

        # Pairwise alignment between circuit heads
        pairwise_scores = []
        pairwise_grassmann = []

        for (L1, H1), (L2, H2) in combinations(sorted_heads, 2):
            U1 = head_subspaces[(L1, H1)]
            U2 = head_subspaces[(L2, H2)]
            score = subspace_alignment_score(U1, U2)
            g_dist = grassmann_distance(U1, U2)
            pairwise_scores.append(score)
            pairwise_grassmann.append(g_dist)

        mean_alignment = float(np.mean(pairwise_scores))
        mean_grassmann = float(np.mean(pairwise_grassmann))
        mean_answer_align = float(np.mean(list(head_answer_alignment.values())))

        log(f"    mean pairwise alignment={mean_alignment:.3f}")
        log(f"    mean Grassmann distance={mean_grassmann:.3f}")
        log(f"    mean answer alignment={mean_answer_align:.3f}")

        results.append(EvalResult(
            metric_id="E64.pairwise_subspace_alignment",
            value=mean_alignment,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_pairs": len(pairwise_scores),
                "max_alignment": float(np.max(pairwise_scores)),
                "min_alignment": float(np.min(pairwise_scores)),
                "subspace_k": k,
            },
        ))
        results.append(EvalResult(
            metric_id="E64.mean_grassmann_distance",
            value=mean_grassmann,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_pairs": len(pairwise_grassmann),
                "max_grassmann": float(np.max(pairwise_grassmann)),
                "min_grassmann": float(np.min(pairwise_grassmann)),
            },
        ))
        results.append(EvalResult(
            metric_id="E64.answer_direction_alignment",
            value=mean_answer_align,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "per_head_alignment": {
                    f"L{L}H{H}": v for (L, H), v in head_answer_alignment.items()
                },
                "max_alignment": float(np.max(list(head_answer_alignment.values()))),
                "most_aligned_head": max(head_answer_alignment, key=head_answer_alignment.get),
            },
        ))

    out = args.out or "64_subspace_alignment.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} metrics.")


if __name__ == "__main__":
    main()
