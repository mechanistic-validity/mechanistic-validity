"""Metric: Adversarial Ablation Verification --- false necessity detection via worst-case ablation

Paper: Sharkey et al. (2026). "Interpreting Language Model Parameters."
Goodfire / Apollo Research. goodfire.ai/research/interpreting-lm-parameters

Tests whether circuit heads that appear necessary under standard
(mean) ablation remain necessary under adversarial ablation of other
heads.  Standard ablation replaces non-circuit heads with their mean
activations, which is benign.  Adversarial ablation instead replaces
each non-circuit head's output with a value designed to maximally
disrupt the remaining circuit head's contribution.  A large gap
between standard and adversarial recovery reveals *false necessity*:
heads that appear necessary only because benign ablation happens to
preserve favourable interactions.

Adversarial Ablation Verification (Evaluation EX32)
=============================================
Instrument:     EX32 --- Adversarial Ablation Gap
Categories:     evaluation
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Whether circuit heads identified as necessary under
                standard ablation remain necessary under adversarial
                ablation, detecting false necessity
Requires:       CPU or GPU, model
=============================================

Core logic:
1. For each circuit head h:
   a. Standard: mean-ablate all other circuit heads, measure
      logit-diff recovery of h alone.
   b. Adversarial: for each non-circuit head, replace its output with
      the negative of its normal output (most disruptive simple
      perturbation), measure logit-diff recovery of h.
2. Report gap = standard_recovery - adversarial_recovery.
3. High gap = false necessity concern.

Pass condition: mean adversarial_gap < 0.3

Usage:
    uv run python 127_adversarial_ablation.py --model gpt2 --device cpu
    uv run python 127_adversarial_ablation.py --tasks ioi --n-prompts 40
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
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Adversarial Ablation Verification",
    paper_ref="Sharkey et al. Goodfire/Apollo (May 2026)",
    paper_cite=(
        "Sharkey et al. 2026, "
        "Interpreting Language Model Parameters "
        "(Goodfire / Apollo Research, goodfire.ai/research/interpreting-lm-parameters)"
    ),
    description=(
        "Compares circuit head recovery under standard (mean) ablation "
        "vs adversarial ablation of non-circuit heads. A large gap "
        "indicates false necessity -- heads that appear necessary only "
        "under benign ablation conditions."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

GAP_THRESHOLD = 0.3


def _heads_to_layer_dict(heads: set[tuple[int, int]]) -> dict[int, list[int]]:
    d: dict[int, list[int]] = {}
    for L, H in heads:
        d.setdefault(L, []).append(H)
    return d


@torch.no_grad()
def _single_head_recovery_standard(
    model,
    head: tuple[int, int],
    other_circuit_heads: set[tuple[int, int]],
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> float:
    """Recovery when keeping only *head* and mean-ablating all other circuit heads."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    ablate_heads = all_heads - {head}
    ablate_by_layer = _heads_to_layer_dict(ablate_heads)

    hooks = []
    for layer, head_list in ablate_by_layer.items():
        def _hook(z, hook, _layer=layer, _heads=head_list):
            for H in _heads:
                z[0, :, H, :] = mean_z[_layer, H].to(z.device)
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))

    recov_num, recov_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_ld = logit_diff_from_logits(model(tokens), correct_ids[i], incorrect_ids[i])
        abl_ld = logit_diff_from_logits(
            model.run_with_hooks(tokens, fwd_hooks=hooks), correct_ids[i], incorrect_ids[i]
        )
        recov_num += abl_ld
        recov_den += clean_ld

    if abs(recov_den) < 1e-8:
        return 0.0
    return recov_num / recov_den


@torch.no_grad()
def _single_head_recovery_adversarial(
    model,
    head: tuple[int, int],
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    mean_z: torch.Tensor,
) -> float:
    """Recovery when keeping *head* but adversarially perturbing all others.

    Adversarial perturbation: replace each non-target head's output with
    the negative of its clean activation. This is the simplest worst-case
    perturbation that flips the head's contribution direction.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    target_L, target_H = head

    hooks = []
    for layer in range(n_layers):
        heads_in_layer = list(range(n_heads))
        def _hook(z, hook, _layer=layer, _heads=heads_in_layer):
            for H in _heads:
                if (_layer, H) == (target_L, target_H):
                    continue
                z[0, :, H, :] = -z[0, :, H, :]
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))

    recov_num, recov_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_ld = logit_diff_from_logits(model(tokens), correct_ids[i], incorrect_ids[i])
        adv_ld = logit_diff_from_logits(
            model.run_with_hooks(tokens, fwd_hooks=hooks), correct_ids[i], incorrect_ids[i]
        )
        recov_num += adv_ld
        recov_den += clean_ld

    if abs(recov_den) < 1e-8:
        return 0.0
    return recov_num / recov_den


def run_adversarial_ablation_verification(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Run adversarial ablation verification across tasks.

    For each task's circuit, computes standard vs adversarial recovery
    for each circuit head and reports the gap.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: all circuit tasks).
        n_prompts: prompts per task.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    results = []
    all_gaps = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            continue

        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            continue

        mean_z = calibrate_mean_z(model, prompts)

        head_gaps = []
        per_head = {}

        for head in sorted(circuit_heads):
            other_circuit = circuit_heads - {head}

            std_recovery = _single_head_recovery_standard(
                model, head, other_circuit, prompts,
                correct_ids, incorrect_ids, mean_z,
            )
            adv_recovery = _single_head_recovery_adversarial(
                model, head, prompts,
                correct_ids, incorrect_ids, mean_z,
            )
            gap = std_recovery - adv_recovery
            head_gaps.append(gap)
            per_head[f"L{head[0]}H{head[1]}"] = {
                "standard_recovery": std_recovery,
                "adversarial_recovery": adv_recovery,
                "gap": gap,
            }
            log(f"    {task} L{head[0]}H{head[1]}: std={std_recovery:.4f} "
                f"adv={adv_recovery:.4f} gap={gap:.4f}")

        mean_gap = float(np.mean(head_gaps))
        all_gaps.append(mean_gap)
        passed = mean_gap < GAP_THRESHOLD

        log(f"  {task}: mean_gap={mean_gap:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.adversarial_ablation_gap",
            value=mean_gap,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "mean_gap": mean_gap,
                "per_head": per_head,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
                "threshold": GAP_THRESHOLD,
            },
        ))

    # Aggregate
    if all_gaps:
        agg_gap = float(np.mean(all_gaps))
        agg_passed = agg_gap < GAP_THRESHOLD
        log(f"  Aggregate: mean_gap={agg_gap:.4f} ({'PASS' if agg_passed else 'FAIL'})")
        results.append(EvalResult(
            metric_id="EX32.adversarial_ablation_gap",
            value=agg_gap,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_gap": agg_gap,
                "n_tasks": len(all_gaps),
                "passed": agg_passed,
                "threshold": GAP_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX32: Adversarial Ablation Verification")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX32: ADVERSARIAL ABLATION VERIFICATION")
    log("=" * 60)

    results = run_adversarial_ablation_verification(
        model,
        tasks=args.tasks or CIRCUIT_TASKS,
        n_prompts=args.n_prompts,
    )

    out = args.out or "127_adversarial_ablation.json"
    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
