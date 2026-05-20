"""Critical Period — Layer-Wise Circuit Crystallization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX9 — Critical Period
Categories:     behavioral, developmental
Evidence family: behavioral
Validity layer: Construct

Tests whether the circuit's computation crystallizes at a single layer
depth or is distributed across multiple layers, using layer-wise
cumulative contribution analysis.

Background:
    Critical periods (Hubel & Wiesel 1970) are developmental windows
    during which neural circuits are especially plastic and susceptible
    to environmental input. After the critical period closes, the
    circuit's wiring becomes relatively fixed (Hensch 2005). This
    concept maps to whether a circuit's functional contribution
    "crystallizes" at a particular layer depth.

    Applied to circuits: if ablating all circuit heads above layer L
    leaves performance nearly intact, then the circuit's computation
    is complete by layer L — it has "crystallized." A healthy, well-
    distributed circuit should not crystallize too early (all work done
    by one layer), nor should a single layer provide the majority of
    the total contribution.

Method (no checkpoints needed — uses layer-wise analysis):
    1. Compute full-circuit logit-diff (baseline, no ablation).
    2. For each layer cutoff l (from 0 to max_circuit_layer):
       - Ablate all circuit heads in layers > l.
       - Measure logit-diff with only early circuit heads active.
    3. Cumulative contribution at layer l = ld_up_to_l / ld_full.
    4. Find the critical layer: the layer with the largest single-step
       increase in cumulative contribution.
    5. Crystallization ratio = max_single_step / total_contribution.
    6. Pass: crystallization_ratio < 0.5 (no single layer dominates;
       circuit is distributed).

Refs: Hubel & Wiesel 1970; Hensch 2005

Usage:
    uv run python EX9_critical_period.py --tasks ioi --n-prompts 40
    uv run python EX9_critical_period.py --device cpu
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
    name="Critical Period (Layer-Wise Crystallization)",
    paper_ref="Hubel & Wiesel 1970; Hensch 2005",
    paper_cite="Hubel & Wiesel 1970, The period of susceptibility to the physiological effects of unilateral eye closure in kittens, J Physiol 206; Hensch 2005, Critical period plasticity in local cortical circuits, Nat Rev Neurosci 6",
    description="Tests whether the circuit crystallizes at a single layer or is distributed, by measuring cumulative layer-wise contribution",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

CRYSTALLIZATION_THRESHOLD = 0.5


@torch.no_grad()
def run_critical_period(
    model, tasks: list[str], n_prompts: int = 40,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token pairs, skipping")
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Identify which layers have circuit heads
        circuit_layers = sorted({L for L, _ in circuit_heads})
        if len(circuit_layers) < 2:
            log(f"  {task}: circuit spans only {len(circuit_layers)} layer(s), "
                f"cannot measure crystallization, skipping")
            continue

        n_valid = min(len(prompts), len(correct_ids))

        # Step 1: Full-circuit logit-diff (no ablation)
        full_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            full_lds.append(ld)
        mean_full_ld = float(np.mean(full_lds))

        if abs(mean_full_ld) < 1e-8:
            log(f"  {task}: near-zero full logit-diff, skipping")
            continue

        # Step 2: For each cutoff layer, ablate circuit heads above it
        cumulative_contributions = {}

        for cutoff in circuit_layers:
            # Ablate circuit heads in layers strictly above cutoff
            heads_to_ablate = {(L, H) for L, H in circuit_heads if L > cutoff}

            if not heads_to_ablate:
                # No heads above cutoff — full circuit active
                cumulative_contributions[cutoff] = 1.0
                continue

            ablate_by_layer = heads_to_layer_dict(heads_to_ablate)
            hooks = make_ablation_hook(ablate_by_layer, mean_z, "mean")

            cutoff_lds = []
            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
                cutoff_lds.append(ld)

            mean_cutoff_ld = float(np.mean(cutoff_lds))
            cumulative_contributions[cutoff] = mean_cutoff_ld / mean_full_ld

        # Step 3: Also compute the contribution with ALL circuit heads ablated
        # (baseline with no circuit)
        all_circuit_by_layer = heads_to_layer_dict(circuit_heads)
        no_circuit_hooks = make_ablation_hook(all_circuit_by_layer, mean_z, "mean")
        no_circuit_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model.run_with_hooks(tokens, fwd_hooks=no_circuit_hooks)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            no_circuit_lds.append(ld)
        no_circuit_fraction = float(np.mean(no_circuit_lds)) / mean_full_ld

        # Step 4: Compute per-layer incremental contribution
        sorted_layers = sorted(cumulative_contributions.keys())
        increments = {}
        prev = no_circuit_fraction
        for layer in sorted_layers:
            current = cumulative_contributions[layer]
            increments[layer] = current - prev
            prev = current

        # Step 5: Find critical layer and crystallization ratio
        total_contribution = 1.0 - no_circuit_fraction
        if abs(total_contribution) < 1e-8:
            log(f"  {task}: circuit has negligible total contribution, skipping")
            continue

        max_increment_layer = max(increments, key=lambda k: increments[k])
        max_increment = increments[max_increment_layer]
        crystallization_ratio = max_increment / total_contribution

        passed = crystallization_ratio < CRYSTALLIZATION_THRESHOLD

        log(f"    layers: {sorted_layers}")
        log(f"    cumulative: {[(l, f'{v:.3f}') for l, v in sorted(cumulative_contributions.items())]}")
        log(f"    increments: {[(l, f'{v:.3f}') for l, v in sorted(increments.items())]}")
        log(f"    critical_layer=L{max_increment_layer}  "
            f"crystallization_ratio={crystallization_ratio:.4f}  "
            f"[{'PASS (distributed)' if passed else 'FAIL (concentrated)'}]")

        results.append(EvalResult(
            metric_id="EX9.critical_period",
            value=crystallization_ratio,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "crystallization_ratio": crystallization_ratio,
                "critical_layer": max_increment_layer,
                "max_layer_increment": max_increment,
                "total_contribution": total_contribution,
                "no_circuit_fraction": no_circuit_fraction,
                "mean_full_ld": mean_full_ld,
                "cumulative_contributions": {
                    str(l): v for l, v in cumulative_contributions.items()
                },
                "per_layer_increments": {
                    str(l): v for l, v in increments.items()
                },
                "circuit_layers": circuit_layers,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
                "threshold": CRYSTALLIZATION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX9: Critical Period (Layer-Wise Crystallization)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX9: CRITICAL PERIOD (LAYER-WISE CRYSTALLIZATION)")
    log("=" * 60)

    out = args.out or "EX9_critical_period.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_critical_period(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
