"""CLT Attribution Graph Cross-Prompt Consistency (Evaluation EX30)
Paper: Ameisen, Lindsey et al. (Anthropic, 2025). "Circuit Tracing:
Revealing Computational Graphs in Language Models."
transformer-circuits.pub
=============================================
Instrument:     EX30 --- CLT Attribution Graph Cross-Prompt Consistency
Categories:     evaluation
Validity layer: Measurement
Criteria:       M1 Reliability, M2 Invariance
Establishes:    Whether the same circuit (set of important heads and
                features) is identified across semantically equivalent
                prompts, testing measurement reliability
Requires:       CPU or GPU, model
=============================================

Tests M1 reliability of circuit identification. For each task:

1. Generate groups of semantically equivalent prompts (paraphrases
   targeting the same answer).
2. For each prompt, identify the top-k most causally important heads
   via activation patching (mean-ablation logit-diff delta).
3. Compute pairwise Jaccard similarity of top-k head sets across
   prompts within each group.
4. Average across groups to get the cross-prompt consistency score.

High consistency means circuit identification is reliable: the same
mechanism is identified regardless of exact phrasing.

Pass condition: cross_prompt_consistency > 0.4

Usage:
    uv run python EX30_clt_cross_prompt_consistency.py --n-prompts 30
    uv run python EX30_clt_cross_prompt_consistency.py --model gpt2 --device cpu --top-k 10
"""

import itertools

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="CLT Attribution Graph Cross-Prompt Consistency",
    paper_ref="Ameisen, Lindsey et al. (Anthropic, 2025)",
    paper_cite=(
        "Ameisen, Lindsey et al. 2025, "
        "Circuit Tracing: Revealing Computational Graphs in Language Models "
        "(Anthropic, transformer-circuits.pub)"
    ),
    description=(
        "Tests M1 reliability by generating semantically equivalent "
        "prompts (same task, same answer), identifying top-k causally "
        "important heads for each via activation patching, and measuring "
        "pairwise Jaccard overlap of the identified head sets."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

CONSISTENCY_THRESHOLD = 0.4


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


@torch.no_grad()
def _get_top_heads(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
    top_k: int,
) -> set[tuple[int, int]]:
    """Identify top-k heads by causal importance (activation patching)."""
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores: dict[tuple[int, int], float] = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            hooks = make_ablation_hook(
                {layer: [head]}, mean_z, ablation_type="mean"
            )
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_id, incorrect_id
            )
            scores[(layer, head)] = abs(baseline_ld - ablated_ld)

    sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {h for h, _ in sorted_heads[:top_k]}


def run_clt_cross_prompt_consistency(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
    top_k: int = 10,
    group_size: int = 5,
) -> list[EvalResult]:
    """Compute cross-prompt consistency of circuit identification.

    For each task, partitions prompts into groups (by shared answer token
    where possible, or sequentially), identifies top-k heads per prompt,
    and computes pairwise Jaccard within groups.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        top_k: number of top heads to compare.
        group_size: minimum prompts per group for pairwise comparison.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    log(f"  Cross-Prompt Consistency: top_k={top_k}, n_prompts={n_prompts}, "
        f"group_size={group_size}")

    results = []
    all_consistency = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if len(correct_ids) < 2:
            log(f"    {task}: need at least 2 prompts, skipping")
            continue

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(20, len(prompts)))

        # Group prompts by correct answer token (semantically equivalent)
        groups: dict[int, list[int]] = {}
        for i, cid in enumerate(correct_ids):
            groups.setdefault(cid, []).append(i)

        # Only keep groups with enough prompts for comparison
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

        # If no groups with shared answers, use sequential grouping
        if not valid_groups:
            n_valid = min(len(correct_ids), n_prompts)
            seq_groups: dict[int, list[int]] = {}
            for idx in range(0, n_valid, group_size):
                chunk = list(range(idx, min(idx + group_size, n_valid)))
                if len(chunk) >= 2:
                    seq_groups[idx] = chunk
            valid_groups = seq_groups

        if not valid_groups:
            log(f"    {task}: no valid groups, skipping")
            continue

        # For each prompt, find top-k heads
        prompt_heads: dict[int, set[tuple[int, int]]] = {}
        for indices in valid_groups.values():
            for i in indices:
                if i in prompt_heads:
                    continue
                if i >= len(prompts):
                    continue
                tokens = model.to_tokens(prompts[i].text)
                try:
                    heads = _get_top_heads(
                        model, tokens, correct_ids[i], incorrect_ids[i],
                        mean_z, top_k,
                    )
                    prompt_heads[i] = heads
                except Exception as e:
                    log(f"    {task} prompt {i}: error {e}")

        # Compute pairwise Jaccard within each group
        group_jaccards = []
        group_details = []
        for group_id, indices in valid_groups.items():
            available = [i for i in indices if i in prompt_heads]
            if len(available) < 2:
                continue

            pairwise = []
            for a, b in itertools.combinations(available, 2):
                j = _jaccard(prompt_heads[a], prompt_heads[b])
                pairwise.append(j)

            if pairwise:
                group_mean = float(np.mean(pairwise))
                group_jaccards.append(group_mean)
                group_details.append({
                    "group_id": int(group_id),
                    "n_prompts": len(available),
                    "n_pairs": len(pairwise),
                    "mean_jaccard": group_mean,
                })

        if not group_jaccards:
            log(f"    {task}: no valid pairwise comparisons")
            continue

        mean_consistency = float(np.mean(group_jaccards))
        std_consistency = float(np.std(group_jaccards))
        passed = mean_consistency > CONSISTENCY_THRESHOLD
        all_consistency.append(mean_consistency)

        log(f"    {task}: consistency={mean_consistency:.4f} "
            f"+/- {std_consistency:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX30.clt_cross_prompt_consistency",
            value=mean_consistency,
            n_samples=len(group_jaccards),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "cross_prompt_consistency": mean_consistency,
                "cross_prompt_consistency_std": std_consistency,
                "n_groups": len(group_jaccards),
                "top_k": top_k,
                "passed": passed,
                "threshold": CONSISTENCY_THRESHOLD,
                "groups": group_details,
            },
        ))

    # Aggregate result
    if all_consistency:
        agg_mean = float(np.mean(all_consistency))
        agg_std = float(np.std(all_consistency))
        agg_passed = agg_mean > CONSISTENCY_THRESHOLD
        log(f"  Aggregate: consistency={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX30.clt_cross_prompt_consistency",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "cross_prompt_consistency": agg_mean,
                "cross_prompt_consistency_std": agg_std,
                "n_tasks_evaluated": len(all_consistency),
                "per_task_consistency": {
                    r.metadata["task"]: r.metadata["cross_prompt_consistency"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "top_k": top_k,
                "passed": agg_passed,
                "threshold": CONSISTENCY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX30: CLT Cross-Prompt Consistency")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top heads to compare (default: 10)")
    parser.add_argument("--group-size", type=int, default=5,
                        help="Minimum prompts per group for fallback grouping (default: 5)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX30: CLT ATTRIBUTION GRAPH CROSS-PROMPT CONSISTENCY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_clt_cross_prompt_consistency(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        top_k=args.top_k,
        group_size=args.group_size,
    )

    out = args.out or "EX30_clt_cross_prompt_consistency.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
