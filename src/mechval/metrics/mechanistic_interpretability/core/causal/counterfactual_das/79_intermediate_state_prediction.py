"""Intermediate State Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal
Validity layer: Internal
Criteria:       A4 Intermediate State Prediction (proposed)
Establishes:    Whether we can predict intermediate component activations from the algorithm spec
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each pathway (sender->receiver):
1. Compute sender head's logit attribution per prompt (scalar: z @ W_O @ W_U diff).
2. Compute receiver head's logit attribution per prompt.
3. Compute Spearman rank correlation between sender and receiver attributions.
4. Baseline: correlate receiver with random non-circuit heads.
5. Pass: mean pathway correlation > mean baseline + 0.15 AND pathway corr > 0.3.

Using scalar logit attributions instead of raw hook_z avoids the overfitting
problem of high-dimensional linear prediction with few samples.

Usage:
    uv run python 79_intermediate_state_prediction.py --tasks ioi sva
    uv run python 79_intermediate_state_prediction.py --device cpu --n-prompts 60
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

PATHWAY_CORR_THRESHOLD = 0.3
UPLIFT_THRESHOLD = 0.15


@torch.no_grad()
def collect_logit_attributions(model, prompts, heads: set[tuple[int, int]],
                               correct_ids: list[int],
                               incorrect_ids: list[int]) -> dict[tuple[int, int], np.ndarray]:
    """For each head, compute per-prompt logit attribution: z @ W_O @ (W_U[correct] - W_U[incorrect]).

    Returns dict mapping (layer, head) -> (n_prompts,) array of scalar attributions.
    """
    layers_needed = {L for L, _ in heads}
    names_filter = lambda n: any(
        n == f"blocks.{L}.attn.hook_z" for L in layers_needed
    )

    W_U = model.W_U.cpu().float()

    per_head = {h: [] for h in heads}
    n = min(len(prompts), len(correct_ids), len(incorrect_ids))

    for i in range(n):
        tokens = model.to_tokens(prompts[i].text)
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        logit_dir = W_U[:, correct_ids[i]] - W_U[:, incorrect_ids[i]]

        for L, H in heads:
            z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H].cpu().float()
            W_O = model.W_O[L, H].cpu().float()
            contrib = z @ W_O
            attr = float(contrib @ logit_dir)
            per_head[(L, H)].append(attr)

    return {h: np.array(vs) for h, vs in per_head.items() if len(vs) >= 4}


def run_intermediate_state_prediction(model, tasks: list[str],
                                      n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []
    rng = np.random.RandomState(42)

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if len(correct_ids) < 6:
            log(f"  {task}: too few usable prompts, skipping")
            continue

        log(f"  {task} ({len(all_heads)} heads, {len(correct_ids)} prompts, "
            f"{len(all_edges)} edges)...")

        heads_to_collect = set(all_heads)
        non_circuit = [(L, H) for L in range(n_layers) for H in range(n_heads)
                       if (L, H) not in all_heads]
        n_baseline_heads = min(len(non_circuit), max(10, len(all_heads)))
        rng.shuffle(non_circuit)
        baseline_heads = non_circuit[:n_baseline_heads]
        heads_to_collect.update(baseline_heads)

        attrs = collect_logit_attributions(
            model, prompts, heads_to_collect, correct_ids, incorrect_ids
        )

        edge_results = {}
        for src_l, src_h, dst_l, dst_h in sorted(all_edges):
            src_key = (src_l, src_h)
            dst_key = (dst_l, dst_h)
            if src_key not in attrs or dst_key not in attrs:
                continue

            rho, pval = spearmanr(attrs[src_key], attrs[dst_key])
            if np.isnan(rho):
                rho = 0.0
            edge_name = f"L{src_l}H{src_h}->L{dst_l}H{dst_h}"
            edge_results[edge_name] = {"rho": float(rho), "pval": float(pval)}
            log(f"    {edge_name}: rho={rho:.3f} (p={pval:.3f})")

        baseline_rhos = []
        receiver_heads = {(dl, dh) for _, _, dl, dh in all_edges}
        for dst_l, dst_h in receiver_heads:
            dst_key = (dst_l, dst_h)
            if dst_key not in attrs:
                continue
            for bh in baseline_heads:
                if bh[0] >= dst_l or bh not in attrs:
                    continue
                rho, _ = spearmanr(attrs[bh], attrs[dst_key])
                if not np.isnan(rho):
                    baseline_rhos.append(float(rho))

        mean_pathway = float(np.mean(
            [e["rho"] for e in edge_results.values()]
        )) if edge_results else 0.0
        mean_baseline = float(np.mean(baseline_rhos)) if baseline_rhos else 0.0
        uplift = mean_pathway - mean_baseline

        passed = mean_pathway > PATHWAY_CORR_THRESHOLD and uplift > UPLIFT_THRESHOLD

        log(f"    mean_pathway_rho={mean_pathway:.3f}  "
            f"mean_baseline_rho={mean_baseline:.3f}  "
            f"uplift={uplift:+.3f}")

        results.append(EvalResult(
            metric_id="A4.intermediate_state_prediction",
            value=mean_pathway,
            baseline_random=mean_baseline,
            n_samples=len(correct_ids),
            metadata={
                "task": task,
                "edge_results": edge_results,
                "mean_pathway_rho": mean_pathway,
                "mean_baseline_rho": mean_baseline,
                "uplift": uplift,
                "n_edges": len(edge_results),
                "n_baseline_pairs": len(baseline_rhos),
                "passed": passed,
                "threshold_rho": PATHWAY_CORR_THRESHOLD,
                "threshold_uplift": UPLIFT_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("A4: Intermediate State Prediction")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("A4: INTERMEDIATE STATE PREDICTION")
    log("=" * 60)

    results = run_intermediate_state_prediction(
        model, tasks, args.n_prompts,
    )

    out = args.out or "79_intermediate_state_prediction.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: pathway_rho={r.value:.3f}  baseline={r.baseline_random:.3f}  "
            f"uplift={r.metadata['uplift']:+.3f}  [{p}]")


if __name__ == "__main__":
    main()
