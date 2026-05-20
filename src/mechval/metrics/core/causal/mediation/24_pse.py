"""Path-Specific Effect (Metric #4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A06 — Mediation Analysis
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Each circuit head carries identifiable path-specific causal effect on output
Requires:       GPU, model
Doc:            /instruments_v2/causal/a06-mediation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extended mediation analysis. For each circuit head, compute the effect
that flows ONLY through that specific head (path-specific) vs the total
effect. Activation patching: patch ONE head at a time from
corrupt->clean while leaving others at clean values.

PSE = LD_with_one_patched - LD_clean.
Reports per-head PSE and sum-of-PSE vs total-effect ratio.

Usage:
    uv run python 24_pse.py --tasks ioi sva
    uv run python 24_pse.py --device cuda --n-prompts 60
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_pse(model, prompts, correct_ids, incorrect_ids,
                circuit_heads: set[tuple[int, int]], mean_z: torch.Tensor) -> dict:
    """Compute path-specific effect for each circuit head.

    For each head h in the circuit:
      1. Run clean forward, get LD_clean
      2. Patch head h from corrupt (mean-ablate) while keeping all
         others at their clean values
      3. PSE_h = LD_clean - LD_with_h_patched
         (positive = head contributes positively to the task)
    """
    per_head_pse = {f"L{L}H{H}": [] for L, H in sorted(circuit_heads)}
    clean_lds = []

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break

        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        clean_lds.append(clean_ld)

        # For each circuit head, patch it to mean (corrupt) while
        # keeping everything else clean
        for L, H in sorted(circuit_heads):
            def _hook(z, hook, _layer=L, _head=H):
                z[0, :, _head, :] = mean_z[_layer, _head].to(z.device)
                return z

            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{L}.attn.hook_z", _hook)],
            )
            patched_ld = logit_diff_from_logits(patched_logits, correct_ids[i], incorrect_ids[i])
            pse = clean_ld - patched_ld
            per_head_pse[f"L{L}H{H}"].append(pse)

    return per_head_pse, clean_lds


@torch.no_grad()
def run_pse(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        per_head_pse, clean_lds = compute_pse(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )

        # Aggregate
        mean_clean_ld = float(np.mean(clean_lds))
        per_head_mean_pse = {k: float(np.mean(v)) for k, v in per_head_pse.items() if v}
        sum_pse = sum(per_head_mean_pse.values())
        total_effect = abs(mean_clean_ld)
        pse_ratio = sum_pse / total_effect if total_effect > 1e-8 else 0.0

        # Sort by magnitude
        sorted_heads = sorted(per_head_mean_pse.items(), key=lambda x: abs(x[1]), reverse=True)

        log(f"    sum_PSE={sum_pse:.4f}  total_effect={total_effect:.4f}  "
            f"ratio={pse_ratio:.4f}")
        for hk, hv in sorted_heads[:5]:
            log(f"      {hk}: PSE={hv:.4f}")

        results.append(EvalResult(
            metric_id="C24.pse",
            value=pse_ratio,
            n_samples=len(clean_lds),
            metadata={
                "task": task,
                "sum_pse": sum_pse,
                "total_effect": total_effect,
                "pse_ratio": pse_ratio,
                "mean_clean_ld": mean_clean_ld,
                "per_head_pse": per_head_mean_pse,
                "n_circuit_heads": len(circuit_heads),
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C24: Path-Specific Effect")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C24: PATH-SPECIFIC EFFECT")
    log("=" * 60)

    results = run_pse(model, tasks, args.n_prompts)

    out = args.out or "24_pse.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: PSE_ratio={r.value:.4f}  sum_PSE={r.metadata['sum_pse']:.4f}")


if __name__ == "__main__":
    main()
