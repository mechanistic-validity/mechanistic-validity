"""Extended Weight-Space Metrics (W_QK Rank, Cosine Alignment, Spectral Norm Ratio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads have distinct weight-space signatures (rank, alignment, spectral norm)
Requires:       CPU, model
Doc:            /instruments_v2/causal/a05-mdc-glennan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CPU-only metrics computed from model weights — no forward passes needed.

Metric #65 — Effective rank of W_QK:
    For each head, compute W_Q @ W_K.T, SVD decompose, compute
    effective_rank = exp(entropy of normalized singular values).

Metric #63 — Cosine alignment:
    For each circuit head, compute top-3 SVD directions of W_OV,
    project through W_U (unembedding), and compute max cosine similarity
    with any non-circuit head's top-3 directions.
    Low cosine = specialized; high cosine = generic.

Metric #68 — Spectral norm ratio:
    spectral_norm(circuit heads) / spectral_norm(non-circuit heads).

Usage:
    uv run python 18_weight_extended.py --tasks ioi sva greater_than
    uv run python 18_weight_extended.py --tasks ioi --device cpu
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


def effective_rank(singular_values: torch.Tensor) -> float:
    """exp(entropy of normalized singular values)."""
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum().item()
    return float(np.exp(entropy))


@torch.no_grad()
def compute_wqk_effective_rank(model) -> np.ndarray:
    """Compute effective rank of W_Q @ W_K.T for each (layer, head).

    Returns array of shape (n_layers, n_heads).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ranks = np.zeros((n_layers, n_heads))

    for L in range(n_layers):
        W_Q = model.W_Q[L]  # (n_heads, d_model, d_head)
        W_K = model.W_K[L]  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            # W_QK = W_Q[H].T @ W_K[H] -> (d_head, d_head)
            # But the full QK circuit is (d_model, d_model): W_Q[H] @ W_K[H].T
            W_QK = W_Q[H] @ W_K[H].T  # (d_model, d_model)
            sv = torch.linalg.svdvals(W_QK.float())
            ranks[L, H] = effective_rank(sv)

    return ranks


@torch.no_grad()
def compute_wov_top_directions(model, k: int = 3) -> dict[tuple[int, int], torch.Tensor]:
    """Compute top-k right singular vectors of W_OV projected through W_U.

    Returns {(layer, head): (k, d_vocab)} tensor of projected directions.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U  # (d_model, d_vocab)

    directions = {}
    for L in range(n_layers):
        W_O = model.W_O[L]  # (n_heads, d_head, d_model)
        W_V = model.W_V[L]  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            W_OV = W_V[H] @ W_O[H]  # (d_model, d_model)
            _, _, Vt = torch.linalg.svd(W_OV.float(), full_matrices=False)
            top_dirs = Vt[:k]  # (k, d_model)
            projected = top_dirs @ W_U.float()  # (k, d_vocab)
            # Normalize each direction
            norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            directions[(L, H)] = projected / norms

    return directions


@torch.no_grad()
def compute_cosine_alignment(directions: dict[tuple[int, int], torch.Tensor],
                             circuit_heads: set[tuple[int, int]]) -> dict[str, float]:
    """Max cosine similarity between each circuit head's directions and non-circuit heads'.

    Returns {\"L{l}H{h}\": max_cosine}.
    """
    non_circuit = {h for h in directions if h not in circuit_heads}
    if not non_circuit:
        return {}

    # Stack non-circuit directions: (n_non_circuit * k, d_vocab)
    non_circuit_dirs = torch.cat([directions[h] for h in sorted(non_circuit)], dim=0)

    result = {}
    for L, H in sorted(circuit_heads):
        if (L, H) not in directions:
            continue
        circ_dirs = directions[(L, H)]  # (k, d_vocab)
        # Cosine similarities: (k, n_non_circuit*k)
        cos_sim = circ_dirs @ non_circuit_dirs.T
        max_cos = cos_sim.abs().max().item()
        result[f"L{L}H{H}"] = max_cos

    return result


@torch.no_grad()
def compute_spectral_norms(model) -> np.ndarray:
    """Spectral norm of W_OV for each (layer, head).

    Returns array of shape (n_layers, n_heads).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    norms = np.zeros((n_layers, n_heads))

    for L in range(n_layers):
        W_O = model.W_O[L]  # (n_heads, d_head, d_model)
        W_V = model.W_V[L]  # (n_heads, d_model, d_head)
        for H in range(n_heads):
            W_OV = W_V[H] @ W_O[H]  # (d_model, d_model)
            sv = torch.linalg.svdvals(W_OV.float())
            norms[L, H] = sv[0].item()

    return norms


def run_weight_extended(model, tasks: list[str],
                        n_prompts: int = 40) -> list[EvalResult]:
    """Compute all weight-space metrics for each task's circuit."""
    log("  Computing W_QK effective ranks...")
    wqk_ranks = compute_wqk_effective_rank(model)

    log("  Computing W_OV top SVD directions...")
    wov_directions = compute_wov_top_directions(model, k=3)

    log("  Computing spectral norms...")
    spectral_norms = compute_spectral_norms(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    results = []
    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        non_circuit = all_heads - circuit_heads
        log(f"  {task} ({len(circuit_heads)} circuit heads)...")

        # Metric #65: W_QK effective rank
        circuit_qk_ranks = {f"L{L}H{H}": wqk_ranks[L, H] for L, H in circuit_heads}
        non_circuit_qk_ranks = [wqk_ranks[L, H] for L, H in non_circuit]
        mean_circuit_qk = float(np.mean(list(circuit_qk_ranks.values())))
        mean_non_circuit_qk = float(np.mean(non_circuit_qk_ranks)) if non_circuit_qk_ranks else 0.0

        log(f"    W_QK eff_rank: circuit={mean_circuit_qk:.2f}  non_circuit={mean_non_circuit_qk:.2f}")

        results.append(EvalResult(
            metric_id="C18.wqk_effective_rank",
            value=mean_circuit_qk,
            baseline_random=mean_non_circuit_qk,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head": circuit_qk_ranks,
                "non_circuit_mean": mean_non_circuit_qk,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

        # Metric #63: Cosine alignment
        cosine_alignment = compute_cosine_alignment(wov_directions, circuit_heads)
        mean_cosine = float(np.mean(list(cosine_alignment.values()))) if cosine_alignment else 0.0

        log(f"    Cosine alignment: mean_max_cos={mean_cosine:.4f}")

        results.append(EvalResult(
            metric_id="C18.cosine_alignment",
            value=mean_cosine,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head": cosine_alignment,
                "n_circuit_heads": len(circuit_heads),
                "interpretation": "low=specialized, high=generic",
            },
        ))

        # Metric #68: Spectral norm ratio
        circuit_norms = [spectral_norms[L, H] for L, H in circuit_heads]
        non_circuit_norms = [spectral_norms[L, H] for L, H in non_circuit]
        mean_circuit_norm = float(np.mean(circuit_norms))
        mean_non_circuit_norm = float(np.mean(non_circuit_norms)) if non_circuit_norms else 1e-8
        spectral_ratio = mean_circuit_norm / (mean_non_circuit_norm + 1e-8)

        log(f"    Spectral norm ratio: {spectral_ratio:.3f} "
            f"(circuit={mean_circuit_norm:.3f}, non_circuit={mean_non_circuit_norm:.3f})")

        per_head_norms = {f"L{L}H{H}": spectral_norms[L, H] for L, H in circuit_heads}
        results.append(EvalResult(
            metric_id="C18.spectral_norm_ratio",
            value=spectral_ratio,
            baseline_random=1.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head_norms": per_head_norms,
                "circuit_mean_norm": mean_circuit_norm,
                "non_circuit_mean_norm": mean_non_circuit_norm,
                "n_circuit_heads": len(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C18: Extended Weight-Space Metrics")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C18: EXTENDED WEIGHT-SPACE METRICS")
    log("=" * 60)

    results = run_weight_extended(model, tasks, args.n_prompts)

    out = args.out or "18_weight_extended.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across "
        f"{len(set(r.metadata['task'] for r in results))} tasks.")


if __name__ == "__main__":
    main()
