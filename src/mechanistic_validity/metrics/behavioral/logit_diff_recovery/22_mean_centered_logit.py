"""Mean-Centered Logit Diff
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D01 — Logit Diff Recovery
Categories:     behavioral
Validity layer: Internal
Criteria:       I2 Sufficiency
Establishes:    Logit diff metric is robust to mean-centering choice
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d01-logit-diff-recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trivial robustness check: compute logit_diff using mean-centered logits
(subtract mean of logit vector before taking diff) and compare to the
standard logit_diff. Reports ratio and absolute difference to show
that the metric is robust to the centering choice.

Usage:
    uv run python 22_mean_centered_logit.py --tasks ioi sva
    uv run python 22_mean_centered_logit.py --device cuda --n-prompts 60
"""

import numpy as np
import torch

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def logit_diff_standard(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    return (last[correct_id] - last[incorrect_id]).item()


@torch.no_grad()
def logit_diff_mean_centered(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    last = logits[0, -1]
    centered = last - last.mean()
    return (centered[correct_id] - centered[incorrect_id]).item()


@torch.no_grad()
def run_mean_centered_logit(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
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

        log(f"  {task} ({len(prompts)} prompts)...")

        standard_diffs = []
        centered_diffs = []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)

            std_ld = logit_diff_standard(logits, correct_ids[i], incorrect_ids[i])
            ctr_ld = logit_diff_mean_centered(logits, correct_ids[i], incorrect_ids[i])

            standard_diffs.append(std_ld)
            centered_diffs.append(ctr_ld)

        mean_std = float(np.mean(standard_diffs))
        mean_ctr = float(np.mean(centered_diffs))
        abs_diff = abs(mean_std - mean_ctr)
        ratio = mean_ctr / mean_std if abs(mean_std) > 1e-8 else float("inf")

        # Per-prompt ratio
        per_prompt_ratios = []
        for s, c in zip(standard_diffs, centered_diffs):
            if abs(s) > 1e-8:
                per_prompt_ratios.append(c / s)

        mean_ratio = float(np.mean(per_prompt_ratios)) if per_prompt_ratios else float("inf")
        std_ratio = float(np.std(per_prompt_ratios)) if per_prompt_ratios else 0.0

        log(f"    standard={mean_std:.4f}  centered={mean_ctr:.4f}  "
            f"ratio={ratio:.6f}  abs_diff={abs_diff:.6f}")

        results.append(EvalResult(
            metric_id="C22.mean_centered_logit",
            value=ratio,
            n_samples=len(standard_diffs),
            metadata={
                "task": task,
                "mean_standard_ld": mean_std,
                "mean_centered_ld": mean_ctr,
                "abs_diff": abs_diff,
                "ratio": ratio,
                "per_prompt_mean_ratio": mean_ratio,
                "per_prompt_std_ratio": std_ratio,
                "note": "ratio should be ~1.0 since centering cancels in diff",
            },
        ))

    return results


def main():
    parser = parse_common_args("C22: Mean-Centered Logit Diff")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C22: MEAN-CENTERED LOGIT DIFF")
    log("=" * 60)

    results = run_mean_centered_logit(model, tasks, args.n_prompts)

    out = args.out or "22_mean_centered_logit.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: ratio={r.value:.6f}  abs_diff={r.metadata['abs_diff']:.6f}")


if __name__ == "__main__":
    main()
