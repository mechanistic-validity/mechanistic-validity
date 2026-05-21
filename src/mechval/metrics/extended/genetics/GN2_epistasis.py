"""Epistasis: Pairwise Interaction Effects
Tests whether ablating circuit heads A and B together produces a
different effect than the sum of ablating each individually.
Non-additive effects indicate functional interaction (epistasis).

Pass: mean |epistasis| > 0.05
Ref: Fisher 1918, TRSEM 52:399-433

Usage:
    uv run python GN2_epistasis.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Epistasis",
    paper_ref="Fisher 1918, TRSEM 52:399-433",
    paper_cite="Fisher 1918, The Correlation Between Relatives on the Supposition of Mendelian Inheritance",
    description="Tests pairwise interaction effects: epistasis = effect(A+B) - effect(A) - effect(B)",
    category="extended",
    tier="extended",
    origin="established",
)

EPISTASIS_THRESHOLD = 0.05
MAX_PAIRS = 20


@torch.no_grad()
def run_epistasis(model, tasks: list[str],
                  n_prompts: int = 40) -> list[EvalResult]:
    """Measure epistatic interactions between pairs of circuit heads."""
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if len(circuit_heads) < 2:
            log(f"  {task}: need >= 2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_valid = min(len(prompts), len(correct_ids))
        log(f"  {task}: {len(circuit_heads)} heads, {n_valid} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, n_valid))

        # Compute clean baseline logit diff per prompt
        clean_lds = []
        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            logits = model(tokens)
            clean_lds.append(logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx]))

        mean_clean_ld = float(np.mean(clean_lds))
        if abs(mean_clean_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        head_list = sorted(circuit_heads)

        # Compute single-head ablation effects
        single_effects: dict[tuple[int, int], float] = {}
        for (L, H) in head_list:
            hooks = make_ablation_hook({L: [H]}, mean_z, "mean")
            ld_sum = 0.0
            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld_sum += logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
            # Effect = drop in LD when ablating this head
            single_effects[(L, H)] = mean_clean_ld - ld_sum / n_valid

        # Compute pairwise ablation effects and epistasis
        pairs = []
        for i in range(len(head_list)):
            for j in range(i + 1, len(head_list)):
                pairs.append((head_list[i], head_list[j]))

        # Limit pairs to avoid combinatorial explosion
        if len(pairs) > MAX_PAIRS:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pairs), MAX_PAIRS, replace=False)
            pairs = [pairs[k] for k in sorted(indices)]

        epistasis_values = []
        pair_details = []

        for (la, ha), (lb, hb) in pairs:
            by_layer: dict[int, list[int]] = {}
            by_layer.setdefault(la, []).append(ha)
            by_layer.setdefault(lb, []).append(hb)
            hooks = make_ablation_hook(by_layer, mean_z, "mean")

            ld_sum = 0.0
            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld_sum += logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])

            joint_effect = mean_clean_ld - ld_sum / n_valid
            epistasis = joint_effect - single_effects[(la, ha)] - single_effects[(lb, hb)]
            epistasis_values.append(epistasis)

            pair_details.append({
                "head_a": f"L{la}H{ha}",
                "head_b": f"L{lb}H{hb}",
                "effect_a": float(single_effects[(la, ha)]),
                "effect_b": float(single_effects[(lb, hb)]),
                "effect_joint": float(joint_effect),
                "epistasis": float(epistasis),
            })

        mean_abs_epistasis = float(np.mean(np.abs(epistasis_values)))
        passed = mean_abs_epistasis > EPISTASIS_THRESHOLD

        log(f"    mean|epistasis|={mean_abs_epistasis:.4f}  "
            f"({len(pairs)} pairs) [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="GN2.epistasis",
            value=mean_abs_epistasis,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "n_pairs": len(pairs),
                "mean_abs_epistasis": mean_abs_epistasis,
                "max_abs_epistasis": float(np.max(np.abs(epistasis_values))),
                "mean_epistasis": float(np.mean(epistasis_values)),
                "fraction_synergistic": float(np.mean(np.array(epistasis_values) > 0)),
                "baseline_ld": mean_clean_ld,
                "passed": passed,
                "threshold": EPISTASIS_THRESHOLD,
                "pair_details": sorted(pair_details,
                                       key=lambda d: abs(d["epistasis"]),
                                       reverse=True)[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("GN2: Epistasis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GN2: EPISTASIS")
    log("=" * 60)

    out = args.out or "GN2_epistasis.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_epistasis(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
