"""Phase Transition Detection in Circuit Behavior
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         PH1 — Phase Transitions
Categories:     cross_discipline, physics
Tier:           cross_discipline
Origin:         established

Tests whether scaling circuit head outputs reveals discontinuous jumps
in logit difference, analogous to phase transitions in statistical
mechanics. Sweeps a multiplicative scale factor alpha from 0 to 2 on
all circuit heads, computes the second derivative of the LD(alpha)
curve, and checks for sharp peaks.

Pass: max |d^2 LD / d alpha^2| > threshold.
Ref: Goldenfeld 1992, Lectures on Phase Transitions and the
     Renormalization Group.

Usage:
    uv run python PH1_phase_transitions.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
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
    name="Phase Transitions",
    paper_ref="Goldenfeld 1992",
    paper_cite="Goldenfeld 1992, Lectures on Phase Transitions and the Renormalization Group",
    description="Detects discontinuous jumps in circuit behavior under continuous scaling of head outputs",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

N_STEPS = 20
ALPHA_MIN = 0.0
ALPHA_MAX = 2.0
SECOND_DERIV_THRESHOLD = 0.5


@torch.no_grad()
def run_phase_transitions(model, tasks: list[str],
                          n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, N_STEPS)

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        heads_by_layer: dict[int, list[int]] = {}
        for L, H in circuit_heads:
            heads_by_layer.setdefault(L, []).append(H)

        mean_ld_per_alpha = np.zeros(N_STEPS)

        for ai, alpha in enumerate(alphas):
            hooks = []
            for layer, head_list in heads_by_layer.items():
                def _hook(z, hook, _heads=head_list, _alpha=alpha):
                    for H in _heads:
                        z[0, :, H, :] *= _alpha
                    return z
                hooks.append((f"blocks.{layer}.attn.hook_z", _hook))

            ld_sum = 0.0
            n_valid = 0
            for idx in range(min(len(prompts), len(correct_ids))):
                tokens = model.to_tokens(prompts[idx].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
                ld_sum += ld
                n_valid += 1

            mean_ld_per_alpha[ai] = ld_sum / max(n_valid, 1)

        # Compute second derivative via finite differences
        d_alpha = alphas[1] - alphas[0]
        first_deriv = np.gradient(mean_ld_per_alpha, d_alpha)
        second_deriv = np.gradient(first_deriv, d_alpha)

        max_second_deriv = float(np.max(np.abs(second_deriv)))
        peak_alpha_idx = int(np.argmax(np.abs(second_deriv)))
        peak_alpha = float(alphas[peak_alpha_idx])
        passed = max_second_deriv > SECOND_DERIV_THRESHOLD

        log(f"    max|d2LD/da2|={max_second_deriv:.4f} at alpha={peak_alpha:.2f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="PH1.phase_transitions",
            value=max_second_deriv,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "alphas": alphas.tolist(),
                "mean_ld_curve": mean_ld_per_alpha.tolist(),
                "second_derivative": second_deriv.tolist(),
                "max_second_deriv": max_second_deriv,
                "peak_alpha": peak_alpha,
                "threshold": SECOND_DERIV_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("PH1: Phase Transitions")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("PH1: PHASE TRANSITIONS")
    log("=" * 60)

    out = args.out or "PH1_phase_transitions.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_phase_transitions(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
