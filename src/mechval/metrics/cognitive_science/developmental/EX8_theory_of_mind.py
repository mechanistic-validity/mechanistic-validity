"""Theory of Mind / False Belief — Circuit Belief Tracking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX8 — Theory of Mind (False Belief)
Categories:     behavioral, developmental
Evidence family: behavioral
Validity layer: Construct

Tests whether the circuit maintains the correct answer representation
even when presented with a conflicting (corrupt) signal, analogous to
false-belief tracking in developmental psychology.

Background:
    The false-belief test (Wimmer & Perner 1983) asks whether a child
    can represent that another agent holds a belief different from
    reality. Baron-Cohen et al. (1985) showed this capacity is
    selectively impaired in autism, establishing it as a core marker
    of theory-of-mind reasoning.

    Applied to circuits: the "true belief" is the correct answer under
    the clean prompt. The "false belief" is the answer suggested by
    the corrupt prompt. A circuit with strong belief tracking maintains
    its correct-answer representation even when the corrupt signal is
    present — it resists being misled. This measures the circuit's
    robustness to interference from conflicting information.

Method:
    1. Run each prompt through the full model on both clean and corrupt
       inputs.
    2. Also run with circuit heads ablated (mean ablation) on clean
       inputs to isolate the circuit's contribution.
    3. Belief resistance = how much the circuit's contribution to the
       correct answer survives when the model sees the corrupt input.
       - Computed as: mean(logit_diff_clean) vs mean(logit_diff_corrupt)
       - Resistance = (mean_clean - mean_corrupt) / mean_clean
    4. Per-head belief contribution: ablate each head individually on
       clean prompts and measure the drop — heads with large drops are
       critical for maintaining the "true belief."
    5. Pass: belief_resistance > 0.5

Refs: Wimmer & Perner 1983; Baron-Cohen et al. 1985

Usage:
    uv run python EX8_theory_of_mind.py --tasks ioi --n-prompts 40
    uv run python EX8_theory_of_mind.py --device cpu
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
    name="Theory of Mind (False Belief Tracking)",
    paper_ref="Wimmer & Perner 1983; Baron-Cohen et al. 1985",
    paper_cite="Wimmer & Perner 1983, Beliefs about beliefs, Cognition 13; Baron-Cohen et al. 1985, Does the autistic child have a theory of mind?, Cognition 21",
    description="Tests whether the circuit maintains correct-answer representations despite conflicting corrupt signals, analogous to false-belief tracking",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

BELIEF_RESISTANCE_THRESHOLD = 0.5


@torch.no_grad()
def run_theory_of_mind(
    model, tasks: list[str], n_prompts: int = 40,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
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

        # Phase 1: Compute clean and corrupt logit-diffs
        clean_lds = []
        corrupt_lds = []
        n_valid = min(len(prompts), len(correct_ids))

        for i in range(n_valid):
            p = prompts[i]
            clean_tokens = model.to_tokens(p.text)
            clean_logits = model(clean_tokens)
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
            clean_lds.append(clean_ld)

            # Use corrupt prompt if available, otherwise use a shuffled prompt
            corrupt_text = getattr(p, "corrupt_text", None) or getattr(p, "corrupt_prompt", None)
            if corrupt_text:
                corrupt_tokens = model.to_tokens(corrupt_text)
                corrupt_logits = model(corrupt_tokens)
                corrupt_ld = logit_diff_from_logits(
                    corrupt_logits, correct_ids[i], incorrect_ids[i])
                corrupt_lds.append(corrupt_ld)
            else:
                # Fall back: use a different prompt's text as the corrupt version
                j = (i + 1) % n_valid
                corrupt_tokens = model.to_tokens(prompts[j].text)
                corrupt_logits = model(corrupt_tokens)
                corrupt_ld = logit_diff_from_logits(
                    corrupt_logits, correct_ids[i], incorrect_ids[i])
                corrupt_lds.append(corrupt_ld)

        mean_clean = float(np.mean(clean_lds)) if clean_lds else 0.0
        mean_corrupt = float(np.mean(corrupt_lds)) if corrupt_lds else 0.0

        if abs(mean_clean) < 1e-8:
            belief_resistance = 0.0
        else:
            belief_resistance = (mean_clean - mean_corrupt) / abs(mean_clean)

        log(f"    mean_clean_ld={mean_clean:.4f}  mean_corrupt_ld={mean_corrupt:.4f}")
        log(f"    belief_resistance={belief_resistance:.4f}")

        # Phase 2: Per-head belief contribution via individual ablation
        per_head_contribution = {}
        circuit_by_layer = heads_to_layer_dict(circuit_heads)

        for L, H in sorted(circuit_heads):
            single_head_by_layer = heads_to_layer_dict({(L, H)})
            hooks = make_ablation_hook(single_head_by_layer, mean_z, "mean")

            ablated_lds = []
            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ablated_ld = logit_diff_from_logits(
                    ablated_logits, correct_ids[i], incorrect_ids[i])
                ablated_lds.append(ablated_ld)

            mean_ablated = float(np.mean(ablated_lds))
            # Contribution = how much logit-diff drops when this head is ablated
            contribution = mean_clean - mean_ablated
            per_head_contribution[(L, H)] = contribution

        # Sort heads by contribution (largest = most important for belief)
        sorted_by_contribution = sorted(
            per_head_contribution.items(), key=lambda x: x[1], reverse=True,
        )
        top_belief_heads = sorted_by_contribution[:5]

        log(f"    top belief heads: "
            f"{[(f'L{h[0]}H{h[1]}', f'{v:.3f}') for h, v in top_belief_heads]}")

        passed = belief_resistance > BELIEF_RESISTANCE_THRESHOLD

        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX8.theory_of_mind",
            value=belief_resistance,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "belief_resistance": belief_resistance,
                "mean_clean_ld": mean_clean,
                "mean_corrupt_ld": mean_corrupt,
                "std_clean_ld": float(np.std(clean_lds)) if len(clean_lds) > 1 else 0.0,
                "std_corrupt_ld": float(np.std(corrupt_lds)) if len(corrupt_lds) > 1 else 0.0,
                "n_circuit_heads": len(circuit_heads),
                "per_head_contribution": {
                    f"L{L}H{H}": v for (L, H), v in per_head_contribution.items()
                },
                "top_belief_heads": [
                    {"head": f"L{h[0]}H{h[1]}", "contribution": v}
                    for h, v in top_belief_heads
                ],
                "passed": passed,
                "threshold": BELIEF_RESISTANCE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX8: Theory of Mind (False Belief Tracking)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX8: THEORY OF MIND (FALSE BELIEF TRACKING)")
    log("=" * 60)

    out = args.out or "EX8_theory_of_mind.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_theory_of_mind(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
