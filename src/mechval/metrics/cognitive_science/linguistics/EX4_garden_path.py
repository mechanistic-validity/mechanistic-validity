"""Garden Path Disambiguation -- Syntactic Revision Sensitivity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX4 -- Garden Path Disambiguation
Categories:     behavioral, linguistics
Evidence family: behavioral
Description mode: implementational-functional

Tests whether the circuit handles syntactically difficult prompts
(those requiring reanalysis) as well as syntactically simple ones.

Background:
    Garden path sentences (Frazier & Rayner 1982, "Making and
    Correcting Errors during Sentence Comprehension") force readers
    to revise their initial syntactic parse. The classic example:
    "The horse raced past the barn fell" initially parses "raced" as
    active voice, then requires reanalysis as a reduced relative.

    Applied to circuits: if a circuit claims to perform a syntactic
    task (e.g., IOI, SVA), its performance should degrade gracefully
    on harder prompts rather than catastrophically failing. A circuit
    that only works on easy prompts is measuring surface heuristics,
    not genuine syntactic computation.

    Method: rather than constructing explicit garden path stimuli
    (which would require task-specific templates), we use a
    difficulty-stratified approach:
    - Run the full model on all prompts, compute logit-diff for each
    - Split prompts into "easy" (high baseline ld) and "hard" (low
      baseline ld) halves
    - For each half, ablate non-circuit heads and measure recovery
    - Garden path penalty = how much worse the circuit does on hard
      prompts relative to easy ones

    This captures the same construct: circuit robustness to input
    difficulty.

    Connections:
    - Frazier & Rayner (1982) "Making and Correcting Errors during
      Sentence Comprehension: Eye Movements in the Analysis of
      Structurally Ambiguous Sentences", Cognitive Psychology 14
    - Frazier (1987) "Sentence Processing: A Tutorial Review", in
      Attention and Performance XII

Method:
    1. Generate N prompts, compute baseline (full-model) logit-diff
       for each.
    2. Rank prompts by baseline ld. Split into easy (top half) and
       hard (bottom half).
    3. For each half, ablate all non-circuit heads (mean ablation) and
       compute circuit-only logit-diff.
    4. Recovery_easy = circuit_ld_easy / baseline_ld_easy
       Recovery_hard = circuit_ld_hard / baseline_ld_hard
    5. Garden path penalty = (recovery_easy - recovery_hard) /
       recovery_easy. Clipped to [0, 1].
    6. Pass: penalty < 0.3 (circuit handles hard prompts adequately).

Usage:
    mechval.run("garden_path", tasks=["ioi"], device="cpu")
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
    name="Garden Path Disambiguation",
    paper_ref="Frazier & Rayner 1982",
    paper_cite="Frazier & Rayner 1982, Making and Correcting Errors during Sentence Comprehension",
    description="Tests whether the circuit degrades gracefully on syntactically difficult prompts vs easy ones",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

GARDEN_PATH_THRESHOLD = 0.3


@torch.no_grad()
def run_garden_path(
    model,
    tasks: list[str],
    n_prompts: int = 40,
) -> list[EvalResult]:
    results = []
    for task in tasks:
        r = _run_garden_path_single(model, task, n_prompts)
        if r is not None:
            results.append(r)
    return results


@torch.no_grad()
def _run_garden_path_single(
    model,
    task: str,
    n_prompts: int = 40,
) -> EvalResult | None:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

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

    # Step 1: compute baseline logit-diffs
    baseline_lds = []
    for i in range(n_valid):
        tokens = model.to_tokens(prompts[i].text)
        logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
        baseline_lds.append(ld)

    # Step 2: rank by difficulty, split into easy/hard halves
    indexed = sorted(enumerate(baseline_lds), key=lambda x: x[1], reverse=True)
    mid = len(indexed) // 2
    easy_indices = [idx for idx, _ in indexed[:mid]]
    hard_indices = [idx for idx, _ in indexed[mid:]]

    log(f"    easy (n={len(easy_indices)}): mean_ld={np.mean([baseline_lds[i] for i in easy_indices]):.4f}")
    log(f"    hard (n={len(hard_indices)}): mean_ld={np.mean([baseline_lds[i] for i in hard_indices]):.4f}")

    # Step 3: calibrate mean activations and set up ablation hooks
    mean_z = calibrate_mean_z(model, prompts[:n_valid])

    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    non_circuit = all_heads - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    # Step 4: compute circuit-only logit-diffs for each half
    def _compute_recovery(indices: list[int]) -> tuple[float, float]:
        """Return (mean_baseline_ld, mean_circuit_ld) for the given prompt indices."""
        sum_baseline = 0.0
        sum_circuit = 0.0
        count = 0
        for i in indices:
            tokens = model.to_tokens(prompts[i].text)
            bl = baseline_lds[i]
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            circuit_ld = logit_diff_from_logits(
                ablated_logits, correct_ids[i], incorrect_ids[i]
            )
            sum_baseline += bl
            sum_circuit += circuit_ld
            count += 1
        if count == 0:
            return 0.0, 0.0
        return sum_baseline / count, sum_circuit / count

    mean_bl_easy, mean_circ_easy = _compute_recovery(easy_indices)
    mean_bl_hard, mean_circ_hard = _compute_recovery(hard_indices)

    # Step 5: compute recovery rates and garden path penalty
    recovery_easy = mean_circ_easy / mean_bl_easy if abs(mean_bl_easy) > 1e-8 else 0.0
    recovery_hard = mean_circ_hard / mean_bl_hard if abs(mean_bl_hard) > 1e-8 else 0.0

    if abs(recovery_easy) > 1e-8:
        penalty = max(0.0, min(1.0, (recovery_easy - recovery_hard) / recovery_easy))
    else:
        penalty = 0.0

    passed = penalty < GARDEN_PATH_THRESHOLD

    log(f"    recovery_easy={recovery_easy:.4f}  recovery_hard={recovery_hard:.4f}")
    log(f"    garden_path_penalty={penalty:.4f}  "
        f"[{'PASS (robust)' if passed else 'FAIL (fragile)'}]")

    return EvalResult(
        metric_id="EX4.garden_path",
        value=1.0 - penalty,
        n_samples=n_valid,
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": task,
            "n_heads": len(circuit_heads),
            "n_easy": len(easy_indices),
            "n_hard": len(hard_indices),
            "mean_baseline_ld_easy": mean_bl_easy,
            "mean_baseline_ld_hard": mean_bl_hard,
            "mean_circuit_ld_easy": mean_circ_easy,
            "mean_circuit_ld_hard": mean_circ_hard,
            "recovery_easy": recovery_easy,
            "recovery_hard": recovery_hard,
            "garden_path_penalty": penalty,
            "passed": passed,
            "threshold": GARDEN_PATH_THRESHOLD,
        },
    )


def main():
    parser = parse_common_args("EX4: Garden Path Disambiguation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX4: GARDEN PATH DISAMBIGUATION")
    log("=" * 60)

    out = args.out or "EX4_garden_path.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = _run_garden_path_single(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
