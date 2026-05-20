"""Participation Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     E08 — Participation Ratio
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility
Establishes:    Circuit heads project onto focused subspaces with low effective dimensionality
Requires:       CPU/GPU, model
Doc:            /instruments_v2/representational/e08-participation-ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Measures the effective dimensionality of each attention head's output
subspace using the participation ratio: PR = (sum lambda_i)^2 / sum(lambda_i^2).
PR=1 means one dimension dominates; PR=d_head means all dimensions contribute
equally. Circuit heads with low PR project onto focused subspaces, suggesting
they encode specific features rather than distributed information.

Usage:
    uv run python 65_participation_ratio.py --tasks ioi sva
    uv run python 65_participation_ratio.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2). Clamp negatives to 0."""
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0
    return float(total ** 2 / (eigenvalues ** 2).sum())


@torch.no_grad()
def collect_head_outputs(model, prompts, device: str) -> dict[tuple[int, int], np.ndarray]:
    """Collect last-position head outputs across all prompts.

    Returns dict mapping (layer, head) -> array of shape (n_prompts, d_head).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    outputs: dict[tuple[int, int], list[np.ndarray]] = {
        (L, H): [] for L in range(n_layers) for H in range(n_heads)
    }

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "attn.hook_result" in n
        )
        for L in range(n_layers):
            result = cache[f"blocks.{L}.attn.hook_result"]  # (1, seq, n_heads, d_head)
            last_pos = result[0, -1].cpu().numpy()  # (n_heads, d_head)
            for H in range(n_heads):
                outputs[(L, H)].append(last_pos[H])

    return {k: np.stack(v) for k, v in outputs.items()}


@torch.no_grad()
def main():
    parser = parse_common_args("E02: Participation Ratio of Head Output Covariance")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)
    results = []

    log("=" * 60)
    log("E02: PARTICIPATION RATIO")
    log("=" * 60)

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

        head_outputs = collect_head_outputs(model, prompts, args.device)

        pr_circuit = []
        pr_non_circuit = []
        per_head_pr = {}

        for (L, H), acts in head_outputs.items():
            if acts.shape[0] < 2:
                continue
            cov = np.cov(acts, rowvar=False)  # (d_head, d_head)
            eigenvalues = np.linalg.eigvalsh(cov)
            pr = compute_participation_ratio(eigenvalues)
            per_head_pr[(L, H)] = pr

            if (L, H) in circuit_heads:
                pr_circuit.append(pr)
            else:
                pr_non_circuit.append(pr)

        if not pr_circuit:
            log(f"  No valid circuit head data for {task}")
            continue

        mean_circuit = float(np.mean(pr_circuit))
        mean_non_circuit = float(np.mean(pr_non_circuit)) if pr_non_circuit else 0.0
        ratio = mean_circuit / mean_non_circuit if mean_non_circuit > 0 else 0.0

        log(f"  Circuit PR: {mean_circuit:.2f} (n={len(pr_circuit)})")
        log(f"  Non-circuit PR: {mean_non_circuit:.2f} (n={len(pr_non_circuit)})")
        log(f"  Ratio (circuit/non-circuit): {ratio:.3f}")

        results.append(EvalResult(
            metric_id="E02.participation_ratio",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "mean_pr_circuit": mean_circuit,
                "mean_pr_non_circuit": mean_non_circuit,
                "ratio_circuit_over_noncircuit": ratio,
                "n_circuit_heads": len(pr_circuit),
                "n_non_circuit_heads": len(pr_non_circuit),
                "circuit_head_prs": {f"L{L}H{H}": per_head_pr[(L, H)]
                                     for (L, H) in sorted(circuit_heads)
                                     if (L, H) in per_head_pr},
                "d_head": model.cfg.d_head,
            },
        ))

    out = args.out or "65_participation_ratio.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
