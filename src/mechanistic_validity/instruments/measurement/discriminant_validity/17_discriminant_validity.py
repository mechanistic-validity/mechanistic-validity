"""Discriminant Validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     F03 — Discriminant Validity
Categories:     measurement
Validity layer: Measurement
Criteria:       M3 Baseline separation
Establishes:    Circuits are task-specific and do not contaminate unrelated tasks
Requires:       GPU, model
Doc:            /instruments_v2/measurement/f03-discriminant-validity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether a circuit identified for task A shows high causal importance
on task B.  It SHOULD NOT -- a good circuit is task-specific.

For each ordered pair (A, B) with A != B:
  1. Get circuit heads for task A
  2. Generate prompts for task B
  3. Compute per-head activation patching effect of task-A's circuit heads
     on task-B's prompts
  4. Compare to:
     (a) task-B's own circuit heads' effect on task-B prompts (convergent)
     (b) random same-size head sets' effect on task-B prompts

Output: a discriminant validity matrix where rows = circuit source task,
columns = test task, cells = mean patching effect.  Diagonal should be
high, off-diagonal should be low.

Usage:
    uv run python 17_discriminant_validity.py --tasks ioi sva greater_than
    uv run python 17_discriminant_validity.py --device cuda --n-prompts 40
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
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
def compute_patching_effects(model, prompts, correct_ids, incorrect_ids,
                             heads: set[tuple[int, int]], rng) -> dict[str, float]:
    """Compute mean per-head patching effect for *heads* on *prompts*.

    Returns {\"L{l}H{h}\": mean_normalized_effect}.
    """
    n_valid = min(len(prompts), len(correct_ids))
    accum = {f"L{L}H{H}": 0.0 for L, H in heads}
    count = 0

    corrupt_indices = list(rng.permutation(n_valid))
    for i in range(n_valid):
        if corrupt_indices[i] == i:
            corrupt_indices[i] = (i + 1) % n_valid

    for i in range(n_valid):
        ci = corrupt_indices[i]
        clean_tokens = model.to_tokens(prompts[i].text)
        corrupt_tokens = model.to_tokens(prompts[ci].text)

        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: "hook_z" in n)

        clean_ld = logit_diff_from_logits(
            model(clean_tokens), correct_ids[i], incorrect_ids[i])
        corrupt_ld = logit_diff_from_logits(
            model(corrupt_tokens), correct_ids[i], incorrect_ids[i])
        gap = clean_ld - corrupt_ld
        if abs(gap) < 1e-8:
            continue

        for L, H in heads:
            hook_name = f"blocks.{L}.attn.hook_z"
            clean_z = clean_cache[hook_name]

            def patch_hook(z, hook, _H=H, _cz=clean_z):
                seq_len = min(z.shape[1], _cz.shape[1])
                z[0, :seq_len, _H, :] = _cz[0, :seq_len, _H, :]
                return z

            patched_logits = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(hook_name, patch_hook)])
            patched_ld = logit_diff_from_logits(
                patched_logits, correct_ids[i], incorrect_ids[i])
            accum[f"L{L}H{H}"] += (patched_ld - corrupt_ld) / gap

        count += 1

    if count > 0:
        for key in accum:
            accum[key] /= count
    return accum


def run_discriminant_validity(model, tasks: list[str], n_prompts: int = 40,
                              n_random_baselines: int = 50) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_model_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    rng = np.random.RandomState(42)

    # Pre-compute per-task circuit heads and prompts
    task_data = {}
    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            log(f"  {task}: no circuit, skipping")
            continue
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue
        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue
        task_data[task] = {
            "heads": heads,
            "prompts": prompts,
            "correct_ids": correct_ids,
            "incorrect_ids": incorrect_ids,
        }

    valid_tasks = sorted(task_data.keys())
    if len(valid_tasks) < 2:
        log("Need >= 2 tasks with circuits for discriminant validity")
        return []

    log(f"  Computing {len(valid_tasks)}x{len(valid_tasks)} discriminant matrix...")

    # Build the matrix: effect[source_task][test_task] = mean patching score
    effect_matrix = {}
    for source_task in valid_tasks:
        source_heads = task_data[source_task]["heads"]
        effect_matrix[source_task] = {}

        for test_task in valid_tasks:
            td = task_data[test_task]
            per_head = compute_patching_effects(
                model, td["prompts"], td["correct_ids"], td["incorrect_ids"],
                source_heads, rng)
            mean_effect = float(np.mean(list(per_head.values()))) if per_head else 0.0
            effect_matrix[source_task][test_task] = mean_effect

            label = "DIAG" if source_task == test_task else "off "
            log(f"    [{label}] circuit({source_task}) on prompts({test_task}): "
                f"mean_effect={mean_effect:.4f}")

    # Random baseline: random head sets of same size tested on each task
    random_baselines = {}
    for test_task in valid_tasks:
        td = task_data[test_task]
        k = len(task_data[test_task]["heads"])
        rand_scores = []
        for _ in range(n_random_baselines):
            rand_heads_idx = rng.choice(len(all_model_heads), size=min(k, len(all_model_heads)),
                                        replace=False)
            rand_heads = {all_model_heads[j] for j in rand_heads_idx}
            per_head = compute_patching_effects(
                model, td["prompts"], td["correct_ids"], td["incorrect_ids"],
                rand_heads, rng)
            rand_scores.append(float(np.mean(list(per_head.values()))) if per_head else 0.0)
        random_baselines[test_task] = {
            "mean": float(np.mean(rand_scores)),
            "std": float(np.std(rand_scores)),
        }

    # Compute discriminant ratio: diagonal / mean(off-diagonal) per source task
    results = []
    for source_task in valid_tasks:
        diagonal = effect_matrix[source_task][source_task]
        off_diag = [effect_matrix[source_task][t] for t in valid_tasks if t != source_task]
        mean_off_diag = float(np.mean(off_diag)) if off_diag else 0.0
        discriminant_ratio = diagonal / (abs(mean_off_diag) + 1e-8)

        log(f"  {source_task}: diag={diagonal:.4f}  off_diag_mean={mean_off_diag:.4f}  "
            f"ratio={discriminant_ratio:.2f}")

        results.append(EvalResult(
            metric_id="C17.discriminant_validity",
            value=discriminant_ratio,
            baseline_random=random_baselines.get(source_task, {}).get("mean"),
            n_samples=len(task_data[source_task]["prompts"]),
            metadata={
                "task": source_task,
                "diagonal_effect": diagonal,
                "off_diagonal_mean": mean_off_diag,
                "effect_row": effect_matrix[source_task],
                "n_circuit_heads": len(task_data[source_task]["heads"]),
                "random_baseline": random_baselines.get(source_task),
            },
        ))

    # Store the full matrix as an additional result
    results.append(EvalResult(
        metric_id="C17.discriminant_matrix",
        value=float(np.mean([r.value for r in results])),
        n_samples=len(valid_tasks),
        metadata={
            "tasks": valid_tasks,
            "effect_matrix": effect_matrix,
            "random_baselines": random_baselines,
        },
    ))

    return results


def main():
    parser = parse_common_args("C17: Discriminant Validity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C17: DISCRIMINANT VALIDITY (Cross-Task Circuit Contamination)")
    log("=" * 60)

    results = run_discriminant_validity(model, tasks, args.n_prompts,
                                       args.n_random_baselines)

    out = args.out or "17_discriminant_validity.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
