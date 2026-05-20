"""Binding Theory / Coreference -- Head Role Consistency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX5 -- Binding Theory
Categories:     behavioral, linguistics
Evidence family: behavioral
Description mode: implementational-functional

Tests whether each circuit head plays a consistent functional role
across different prompts, by measuring the stability of per-head
contribution rankings.

Background:
    Binding Theory (Chomsky 1981, "Lectures on Government and
    Binding") governs how noun phrases relate to each other within
    sentences. Principle A (anaphors must be locally bound),
    Principle B (pronouns must be locally free), and Principle C
    (R-expressions must be free) define coreference constraints.

    Applied to circuits: if a circuit head truly implements a specific
    syntactic role (e.g., name mover, backup name mover, S-inhibition),
    its contribution should be consistent across prompts. A head that
    ranks as the top contributor on one prompt subset but irrelevant on
    another is not implementing a stable binding-like role -- its
    apparent function is prompt-dependent.

    Method: rather than constructing binding-specific stimuli, we
    measure the consistency of head roles directly:
    - Split prompts into two random halves
    - Compute per-head contribution (via mean ablation) on each half
    - Measure rank correlation (Spearman rho) between halves
    - Repeat with multiple random splits for robustness

    High rho means heads play consistent roles regardless of which
    prompts they see -- analogous to consistent binding behavior.

    Connections:
    - Chomsky (1981) "Lectures on Government and Binding", Foris
    - Reinhart (1983) "Anaphora and Semantic Interpretation", Croom Helm
    - Wang et al. (2022) "Interpretability in the Wild" -- IOI circuit
      roles assumed to be stable across prompts

Method:
    1. Generate N prompts for the task, compute full-model logit-diffs.
    2. For each circuit head, compute its contribution via mean ablation:
       contribution(h) = mean_ld_full - mean_ld_ablate_h
    3. Repeat on K random half-splits of the prompts.
    4. For each split pair, compute Spearman rank correlation of per-head
       contributions between the two halves.
    5. Binding consistency = mean rho across all K splits.
    6. Pass: mean rho > 0.7 (consistent head roles).

Usage:
    mechval.run("binding_theory", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Binding Theory (Head Role Consistency)",
    paper_ref="Chomsky 1981",
    paper_cite="Chomsky 1981, Lectures on Government and Binding",
    description="Tests whether circuit head contributions are rank-stable across prompt subsets, indicating consistent functional roles",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

BINDING_THRESHOLD = 0.7
N_SPLITS = 10


def _compute_head_contributions(
    model,
    prompts: list,
    correct_ids: list[int],
    incorrect_ids: list[int],
    circuit_heads: set[tuple[int, int]],
    mean_z: torch.Tensor,
    indices: list[int],
) -> dict[tuple[int, int], float]:
    """Compute each circuit head's contribution on a subset of prompts.

    Contribution of head h = mean_ld_full - mean_ld_with_h_ablated.
    Positive contribution means the head helps the task.
    """
    n_valid = len(indices)
    if n_valid == 0:
        return {h: 0.0 for h in circuit_heads}

    # Full model logit-diffs on this subset
    full_lds = []
    for i in indices:
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
        full_lds.append(ld)
    mean_full_ld = np.mean(full_lds)

    # Per-head ablation
    contributions = {}
    for layer, head in sorted(circuit_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"

        def _ablate_hook(z, hook, _H=head, _L=layer):
            z[0, :, _H, :] = mean_z[_L, _H].to(z.device)
            return z

        ablated_lds = []
        for i in indices:
            tokens = model.to_tokens(prompts[i].text)
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, _ablate_hook)]
            )
            ld = logit_diff_from_logits(
                ablated_logits, correct_ids[i], incorrect_ids[i]
            )
            ablated_lds.append(ld)

        mean_ablated_ld = np.mean(ablated_lds)
        contributions[(layer, head)] = mean_full_ld - mean_ablated_ld

    return contributions


@torch.no_grad()
def run_binding_theory(
    model,
    tasks: list[str],
    n_prompts: int = 40,
    n_splits: int = N_SPLITS,
) -> list[EvalResult]:
    results = []
    for task in tasks:
        r = _run_binding_theory_single(model, task, n_prompts, n_splits)
        if r is not None:
            results.append(r)
    return results


@torch.no_grad()
def _run_binding_theory_single(
    model,
    task: str,
    n_prompts: int = 40,
    n_splits: int = N_SPLITS,
) -> EvalResult | None:
    tokenizer = model.tokenizer

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        log(f"  {task}: no circuit heads, skipping")
        return None

    if len(circuit_heads) < 3:
        log(f"  {task}: need at least 3 circuit heads for ranking, skipping")
        return None

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if len(prompts) < 6:
        log(f"  {task}: need at least 6 prompts for split-half, skipping")
        return None

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    n_valid = min(len(prompts), len(correct_ids))
    if n_valid < 6:
        log(f"  {task}: fewer than 6 valid prompts, skipping")
        return None

    log(f"  {task}: {len(circuit_heads)} circuit heads, {n_valid} prompts, {n_splits} splits")

    mean_z = calibrate_mean_z(model, prompts[:n_valid])

    # Sorted head list for consistent ordering across splits
    sorted_heads = sorted(circuit_heads)
    rhos: list[float] = []
    rng = np.random.RandomState(0)

    for split_idx in range(n_splits):
        # Random split into two halves
        all_indices = list(range(n_valid))
        rng.shuffle(all_indices)
        mid = len(all_indices) // 2
        half_a = all_indices[:mid]
        half_b = all_indices[mid:]

        if len(half_a) < 2 or len(half_b) < 2:
            continue

        contrib_a = _compute_head_contributions(
            model, prompts, correct_ids, incorrect_ids,
            circuit_heads, mean_z, half_a,
        )
        contrib_b = _compute_head_contributions(
            model, prompts, correct_ids, incorrect_ids,
            circuit_heads, mean_z, half_b,
        )

        # Extract contribution vectors in consistent order
        vec_a = [contrib_a[h] for h in sorted_heads]
        vec_b = [contrib_b[h] for h in sorted_heads]

        rho, _ = spearmanr(vec_a, vec_b)
        if not np.isnan(rho):
            rhos.append(float(rho))
            log(f"    split {split_idx}: rho={rho:.4f}")

    if not rhos:
        log(f"  {task}: no valid splits, skipping")
        return None

    rhos_arr = np.array(rhos)
    mean_rho = float(rhos_arr.mean())
    std_rho = float(rhos_arr.std()) if len(rhos_arr) > 1 else 0.0
    passed = mean_rho > BINDING_THRESHOLD

    log(f"    mean_rho={mean_rho:.4f}  std={std_rho:.4f}  "
        f"[{'PASS (consistent roles)' if passed else 'FAIL (unstable roles)'}]")

    # Also compute overall head contributions for reporting
    all_indices = list(range(n_valid))
    overall_contrib = _compute_head_contributions(
        model, prompts, correct_ids, incorrect_ids,
        circuit_heads, mean_z, all_indices,
    )
    per_head_contributions = {
        f"L{layer}H{head}": float(overall_contrib[(layer, head)])
        for layer, head in sorted_heads
    }

    return EvalResult(
        metric_id="EX5.binding_theory",
        value=mean_rho,
        n_samples=n_valid,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": task,
            "n_heads": len(circuit_heads),
            "n_splits": len(rhos),
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "min_rho": float(rhos_arr.min()),
            "max_rho": float(rhos_arr.max()),
            "per_head_contributions": per_head_contributions,
            "passed": passed,
            "threshold": BINDING_THRESHOLD,
        },
    )


def main():
    parser = parse_common_args("EX5: Binding Theory (Head Role Consistency)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX5: BINDING THEORY (HEAD ROLE CONSISTENCY)")
    log("=" * 60)

    out = args.out or "EX5_binding_theory.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = _run_binding_theory_single(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
