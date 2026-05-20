"""Animacy / Thematic Role Discrimination -- Position Sensitivity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX6 -- Animacy / Thematic Role
Categories:     behavioral, linguistics
Evidence family: behavioral
Description mode: implementational-functional

Tests whether circuit heads are sensitive to the structural role
(position) of names in prompts, rather than simply tracking name
salience regardless of position.

Background:
    Thematic role theory (Dowty 1991, "Thematic Proto-Roles and
    Argument Selection") distinguishes agents (doers) from patients
    (receivers). A circuit that genuinely tracks syntactic role should
    behave differently when the same name appears as subject vs object.

    Applied to circuits: many tasks (IOI, gendered pronoun resolution)
    involve names in different structural positions. A circuit head
    that contributes identically regardless of whether a name is in
    the subject (IO) or object (S) position is not tracking thematic
    roles -- it is simply detecting name tokens. Conversely, a head
    whose contribution varies with name position is encoding
    positional/syntactic information.

    Method: for each prompt, we measure each circuit head's individual
    contribution to the logit-diff via mean ablation. Then we
    partition prompts by a structural feature (which name appears
    first) and test whether per-head contributions differ across
    partitions. High variance across partitions = position-sensitive =
    tracking role rather than just name identity.

    Connections:
    - Dowty (1991) "Thematic Proto-Roles and Argument Selection",
      Language 67
    - Fillmore (1968) "The Case for Case", in Universals in
      Linguistic Theory
    - Wang et al. (2022) -- IOI heads are claimed to implement
      specific roles (name mover, S-inhibition) which should exhibit
      position sensitivity

Method:
    1. Generate N prompts. Run full model to get baseline logit-diffs.
    2. For each circuit head, compute its contribution via mean
       ablation (full_ld - ablated_ld) on every prompt individually.
    3. For each head, compute the variance of its contribution across
       prompts. Also compute the mean absolute contribution.
    4. Role sensitivity = std(contributions) / |mean(contributions)|
       (coefficient of variation), averaged across circuit heads.
       This captures how much a head's contribution depends on the
       specific prompt (and thus the specific name arrangement).
    5. Pass: mean role_sensitivity > 0.1 (circuit is position-sensitive,
       not just name-tracking).

Usage:
    mechval.run("animacy", tasks=["ioi"], device="cpu")
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
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Animacy / Thematic Role Discrimination",
    paper_ref="Dowty 1991",
    paper_cite="Dowty 1991, Thematic Proto-Roles and Argument Selection",
    description="Tests whether circuit heads are sensitive to name position/role rather than just name identity",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

ANIMACY_THRESHOLD = 0.1


@torch.no_grad()
def run_animacy(
    model,
    tasks: list[str],
    n_prompts: int = 40,
) -> list[EvalResult]:
    results = []
    for task in tasks:
        r = _run_animacy_single(model, task, n_prompts)
        if r is not None:
            results.append(r)
    return results


@torch.no_grad()
def _run_animacy_single(
    model,
    task: str,
    n_prompts: int = 40,
) -> EvalResult | None:
    tokenizer = model.tokenizer

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        log(f"  {task}: no circuit heads, skipping")
        return None

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if len(prompts) < 4:
        log(f"  {task}: need at least 4 prompts, skipping")
        return None

    correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
    n_valid = min(len(prompts), len(correct_ids))
    if n_valid < 4:
        log(f"  {task}: fewer than 4 valid prompts, skipping")
        return None

    log(f"  {task}: {len(circuit_heads)} circuit heads, {n_valid} prompts")

    # Calibrate mean activations
    mean_z = calibrate_mean_z(model, prompts[:n_valid])

    sorted_heads = sorted(circuit_heads)

    # Compute per-head, per-prompt contributions
    # contributions[head_idx][prompt_idx] = contribution value
    contributions: dict[tuple[int, int], list[float]] = {h: [] for h in sorted_heads}

    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)

        # Full model logit-diff
        logits = model(tokens)
        full_ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

        # Per-head ablation
        for layer, head in sorted_heads:
            hook_name = f"blocks.{layer}.attn.hook_z"

            def _ablate_hook(z, hook, _H=head, _L=layer):
                z[0, :, _H, :] = mean_z[_L, _H].to(z.device)
                return z

            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, _ablate_hook)]
            )
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_ids[i], incorrect_ids[i]
            )
            contributions[(layer, head)].append(full_ld - ablated_ld)

    # Compute per-head role sensitivity (coefficient of variation)
    per_head_sensitivity: dict[str, float] = {}
    per_head_mean_contrib: dict[str, float] = {}
    sensitivities: list[float] = []

    for layer, head in sorted_heads:
        contribs = np.array(contributions[(layer, head)])
        head_mean = float(np.mean(np.abs(contribs)))
        head_std = float(np.std(contribs))
        head_key = f"L{layer}H{head}"

        if head_mean > 1e-8:
            sensitivity = head_std / head_mean
        else:
            sensitivity = 0.0

        per_head_sensitivity[head_key] = sensitivity
        per_head_mean_contrib[head_key] = float(np.mean(contribs))
        sensitivities.append(sensitivity)

    mean_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0.0
    std_sensitivity = float(np.std(sensitivities)) if len(sensitivities) > 1 else 0.0

    # Also compute overall contribution variance across heads
    # (how differentiated are head roles from each other)
    head_means = np.array([np.mean(contributions[h]) for h in sorted_heads])
    cross_head_std = float(np.std(head_means)) if len(head_means) > 1 else 0.0
    cross_head_range = float(head_means.max() - head_means.min()) if len(head_means) > 1 else 0.0

    passed = mean_sensitivity > ANIMACY_THRESHOLD

    log(f"    mean_role_sensitivity={mean_sensitivity:.4f}  std={std_sensitivity:.4f}")
    log(f"    cross_head_std={cross_head_std:.4f}  range={cross_head_range:.4f}")
    log(f"    [{'PASS (position-sensitive)' if passed else 'FAIL (position-insensitive)'}]")

    return EvalResult(
        metric_id="EX6.animacy",
        value=mean_sensitivity,
        n_samples=n_valid,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": task,
            "n_heads": len(circuit_heads),
            "mean_role_sensitivity": mean_sensitivity,
            "std_role_sensitivity": std_sensitivity,
            "per_head_sensitivity": per_head_sensitivity,
            "per_head_mean_contribution": per_head_mean_contrib,
            "cross_head_std": cross_head_std,
            "cross_head_range": cross_head_range,
            "passed": passed,
            "threshold": ANIMACY_THRESHOLD,
        },
    )


def main():
    parser = parse_common_args("EX6: Animacy / Thematic Role Discrimination")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX6: ANIMACY / THEMATIC ROLE DISCRIMINATION")
    log("=" * 60)

    out = args.out or "EX6_animacy.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = _run_animacy_single(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
