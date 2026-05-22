"""TDA Factors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E09b — TDA
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads form a low-rank system controlled by few latent factors
Requires:       GPU, model
Doc:            /instruments_v2/representational/e09b-tda-factors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyzes whether circuit heads behave as a low-rank system by computing SVD
of the (n_prompts x n_circuit_heads) DLA (direct logit attribution) matrix.
Reports: (1) effective rank (singular values needed for 90% variance),
(2) top singular vectors showing which heads co-vary, (3) comparison to
random head subsets. A low effective rank means the circuit's behavior is
controlled by a few latent factors rather than independent head contributions.

Usage:
    uv run python 69_tda_factors.py --tasks ioi greater_than
    uv run python 69_tda_factors.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechval.metrics.common import (
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


def effective_rank_90(singular_values: np.ndarray) -> int:
    """Number of singular values needed to capture 90% of total variance."""
    if len(singular_values) == 0:
        return 0
    variance = singular_values ** 2
    total = variance.sum()
    if total < 1e-12:
        return len(singular_values)
    cumulative = np.cumsum(variance) / total
    rank = int(np.searchsorted(cumulative, 0.9)) + 1
    return min(rank, len(singular_values))


def compute_dla_matrix(model, prompts, correct_ids, incorrect_ids,
                       heads: list[tuple[int, int]], device: str) -> np.ndarray:
    """Compute DLA (direct logit attribution) for each head on each prompt.

    DLA for head (L,H) = (head_output @ W_U[:, correct] - head_output @ W_U[:, incorrect])
    at the last sequence position.

    Returns: (n_prompts, n_heads) array.
    """
    W_U = model.W_U.detach().cpu().numpy()  # (d_model, d_vocab)
    n_prompts = min(len(prompts), len(correct_ids))
    n_heads = len(heads)
    dla_matrix = np.zeros((n_prompts, n_heads))

    for i in range(n_prompts):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "attn.hook_result" in n
        )

        logit_dir = W_U[:, correct_ids[i]] - W_U[:, incorrect_ids[i]]  # (d_model,)

        for j, (L, H) in enumerate(heads):
            # Head output at last position: (d_model,) after O projection
            # hook_result is post-O: (batch, seq, n_heads, d_model) -- actually (batch, seq, n_heads, d_head)
            # We need hook_result which is (batch, seq, n_heads, d_head), then project through W_O
            head_out = cache[f"blocks.{L}.attn.hook_result"][0, -1, H].cpu().numpy()  # (d_head,)
            # Project through W_O to get contribution to residual stream
            W_O = model.W_O[L, H].detach().cpu().numpy()  # (d_head, d_model)
            contribution = head_out @ W_O  # (d_model,)
            dla_matrix[i, j] = float(contribution @ logit_dir)

    return dla_matrix


@torch.no_grad()
def main():
    parser = parse_common_args("E09: Factored Structure (SVD of DLA)")
    parser.add_argument("--n-random-subsets", type=int, default=20,
                        help="Random head subsets for baseline comparison")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    results = []

    log("=" * 60)
    log("E09: TDA FACTORS (SVD OF CIRCUIT DLA)")
    log("=" * 60)

    all_heads = [(L, H) for L in range(model.cfg.n_layers)
                 for H in range(model.cfg.n_heads)]

    for task in tasks:
        log(f"\n--- Task: {task} ---")
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  No circuit heads for {task}, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, args.n_prompts)
        if not prompts:
            log(f"  No prompts for {task}, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if len(correct_ids) < 10:
            log(f"  Too few valid prompts for {task}, skipping")
            continue

        circuit_head_list = sorted(circuit_heads)
        n_circuit = len(circuit_head_list)

        log(f"  Computing DLA for {n_circuit} circuit heads across {len(correct_ids)} prompts...")
        dla = compute_dla_matrix(model, prompts, correct_ids, incorrect_ids,
                                 circuit_head_list, args.device)

        # Center the DLA matrix
        dla_centered = dla - dla.mean(axis=0, keepdims=True)

        # SVD of circuit DLA
        U, S, Vt = np.linalg.svd(dla_centered, full_matrices=False)
        eff_rank = effective_rank_90(S)
        variance_explained = (S ** 2) / (S ** 2).sum() if (S ** 2).sum() > 1e-12 else S * 0

        log(f"  Effective rank (90% variance): {eff_rank} / {n_circuit}")
        log(f"  Top-3 singular value fractions: {variance_explained[:3].tolist()}")

        # Factor loadings: columns of V^T give how heads load on each factor
        factor_loadings = {}
        for k in range(min(3, Vt.shape[0])):
            loadings = Vt[k]  # (n_circuit_heads,)
            factor_loadings[f"factor_{k}"] = {
                f"L{L}H{H}": float(loadings[j])
                for j, (L, H) in enumerate(circuit_head_list)
            }

        # Random baseline: same-size subsets of heads
        random_ranks = []
        rng = np.random.default_rng(42)
        non_circuit = [h for h in all_heads if h not in circuit_heads]

        for _ in range(args.n_random_subsets):
            if len(non_circuit) < n_circuit:
                break
            subset = [non_circuit[i] for i in
                      rng.choice(len(non_circuit), size=n_circuit, replace=False)]
            rand_dla = compute_dla_matrix(model, prompts, correct_ids, incorrect_ids,
                                          subset, args.device)
            rand_centered = rand_dla - rand_dla.mean(axis=0, keepdims=True)
            _, S_rand, _ = np.linalg.svd(rand_centered, full_matrices=False)
            random_ranks.append(effective_rank_90(S_rand))

        mean_random_rank = float(np.mean(random_ranks)) if random_ranks else float(n_circuit)

        log(f"  Random subset effective rank: {mean_random_rank:.1f}")
        log(f"  Rank ratio (circuit/random): {eff_rank / mean_random_rank:.3f}"
            if mean_random_rank > 0 else "  Rank ratio: N/A")

        results.append(EvalResult(
            metric_id="E09.tda_factors",
            value=float(eff_rank),
            baseline_random=mean_random_rank,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "effective_rank_90": eff_rank,
                "n_circuit_heads": n_circuit,
                "singular_values": S.tolist(),
                "variance_explained": variance_explained.tolist(),
                "factor_loadings_top3": factor_loadings,
                "random_baseline_ranks": random_ranks,
                "mean_random_rank": mean_random_rank,
                "rank_ratio": eff_rank / mean_random_rank if mean_random_rank > 0 else 0.0,
            },
        ))

    out = args.out or "69_tda_factors.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
