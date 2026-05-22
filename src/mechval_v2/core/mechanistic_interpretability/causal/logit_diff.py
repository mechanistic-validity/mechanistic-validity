"""Logit Difference — baseline task performance metric.

Measures the mean logit difference between correct and incorrect tokens
across prompts. This is the standard IOI metric from Wang et al. 2023.

Usage:
    mv run logit_diff --tasks ioi sva
"""

import torch

from mechval.metrics.common import (
    EvalResult,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
)


@torch.no_grad()
def run_logit_diff(model, tasks: list[str], n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        diffs = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)
            diffs.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

        mean_diff = sum(diffs) / len(diffs)
        log(f"  {task}: mean_logit_diff={mean_diff:.4f} (n={len(diffs)})")

        results.append(EvalResult(
            metric_id="logit_diff",
            value=mean_diff,
            n_samples=len(diffs),
            metadata={
                "task": task,
                "per_prompt_diffs": diffs,
                "min": min(diffs),
                "max": max(diffs),
            },
        ))

    return results
