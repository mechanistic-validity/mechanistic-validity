"""CLT Attribution Graph Faithfulness (Evaluation EX29)
Paper: Ameisen, Lindsey et al. (Anthropic, 2025). "Circuit Tracing:
Revealing Computational Graphs in Language Models."
transformer-circuits.pub
=============================================
Instrument:     EX29 --- CLT Attribution Graph Faithfulness
Categories:     evaluation
Validity layer: Internal
Criteria:       I2 Compositional Sufficiency, C2 Structural Plausibility
Establishes:    Whether a pruned attribution graph (keeping only the
                highest-weight edges) preserves the model's behavior,
                validating that the graph captures the causally relevant
                computation
Requires:       CPU or GPU, model
=============================================

Measures how well a pruned attribution graph preserves model behavior.
For each task prompt:

1. Run the model forward, caching residual-stream activations at every
   layer boundary ("feature activations").
2. Use activation patching (mean-ablation) to score each head's causal
   contribution to the correct-token logit difference at the final
   position. These scores form a simplified attribution graph.
3. Retain only the top-k heads by attribution score (prune the rest).
4. Re-run the model with all pruned heads mean-ablated.
5. Measure the logit difference of the pruned model vs the full model.
   graph_faithfulness = pruned_logit_diff / full_logit_diff (clipped to
   [0, 1]).

High faithfulness means the pruned graph (top-k heads) is sufficient to
reproduce the model's output, validating the attribution graph.

Pass condition: graph_faithfulness > 0.8

Usage:
    uv run python EX29_clt_graph_faithfulness.py --n-prompts 30
    uv run python EX29_clt_graph_faithfulness.py --model gpt2 --device cpu --top-k 10
"""

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
    name="CLT Attribution Graph Faithfulness",
    paper_ref="Ameisen, Lindsey et al. (Anthropic, 2025)",
    paper_cite=(
        "Ameisen, Lindsey et al. 2025, "
        "Circuit Tracing: Revealing Computational Graphs in Language Models "
        "(Anthropic, transformer-circuits.pub)"
    ),
    description=(
        "Builds a simplified attribution graph via activation patching, "
        "prunes low-attribution edges (heads), re-runs the model with "
        "pruned heads mean-ablated, and measures logit-diff recovery. "
        "High faithfulness means the pruned graph is sufficient."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

FAITHFULNESS_THRESHOLD = 0.8


@torch.no_grad()
def _score_heads_by_patching(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
) -> dict[tuple[int, int], float]:
    """Score every (layer, head) by logit-diff change under mean-ablation.

    Returns dict mapping (layer, head) -> absolute logit-diff change.
    """
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

    return scores


@torch.no_grad()
def _compute_faithfulness(
    model,
    tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    mean_z: torch.Tensor,
    head_scores: dict[tuple[int, int], float],
    top_k: int,
) -> float:
    """Compute graph faithfulness for a single prompt.

    Keeps top_k heads (by attribution score), mean-ablates the rest,
    and returns pruned_logit_diff / full_logit_diff clipped to [0, 1].
    """
    baseline_logits = model(tokens)
    baseline_ld = logit_diff_from_logits(baseline_logits, correct_id, incorrect_id)

    if abs(baseline_ld) < 1e-6:
        return 1.0  # no signal to preserve

    # Sort heads by score descending, keep top_k
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    kept = {h for h, _ in sorted_heads[:top_k]}

    # Ablate all heads NOT in kept set
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ablate_by_layer: dict[int, list[int]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            if (layer, head) not in kept:
                ablate_by_layer.setdefault(layer, []).append(head)

    if not ablate_by_layer:
        return 1.0

    hooks = make_ablation_hook(ablate_by_layer, mean_z, ablation_type="mean")
    pruned_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    pruned_ld = logit_diff_from_logits(pruned_logits, correct_id, incorrect_id)

    # Faithfulness: fraction of logit diff preserved
    faithfulness = pruned_ld / baseline_ld
    return float(np.clip(faithfulness, 0.0, 1.0))


def run_clt_graph_faithfulness(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 30,
    top_k: int = 10,
) -> list[EvalResult]:
    """Compute CLT attribution graph faithfulness across tasks.

    For each task, scores heads via activation patching, prunes to
    top_k, and measures logit-diff recovery.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        top_k: number of top-attribution heads to retain in the pruned graph.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    log(f"  CLT Graph Faithfulness: top_k={top_k}, n_prompts={n_prompts}")

    results = []
    all_faithfulness = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(20, len(prompts)))

        task_faithfulness = []
        per_prompt_details = []

        for i, prompt in enumerate(prompts):
            if i >= len(correct_ids):
                break

            tokens = model.to_tokens(prompt.text)

            try:
                head_scores = _score_heads_by_patching(
                    model, tokens, correct_ids[i], incorrect_ids[i], mean_z
                )
                faith = _compute_faithfulness(
                    model, tokens, correct_ids[i], incorrect_ids[i],
                    mean_z, head_scores, top_k,
                )
            except Exception as e:
                log(f"    {task} prompt {i}: error {e}")
                continue

            task_faithfulness.append(faith)
            top_heads = sorted(
                head_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            per_prompt_details.append({
                "prompt_index": i,
                "faithfulness": faith,
                "top_heads": [
                    {"layer": h[0], "head": h[1], "score": s}
                    for (h, s) in top_heads
                ],
            })

        if not task_faithfulness:
            log(f"    {task}: no valid results")
            continue

        mean_faith = float(np.mean(task_faithfulness))
        std_faith = float(np.std(task_faithfulness))
        passed = mean_faith > FAITHFULNESS_THRESHOLD
        all_faithfulness.append(mean_faith)

        log(f"    {task}: faithfulness={mean_faith:.4f} "
            f"+/- {std_faith:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX29.clt_graph_faithfulness",
            value=mean_faith,
            n_samples=len(task_faithfulness),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "graph_faithfulness": mean_faith,
                "graph_faithfulness_std": std_faith,
                "n_prompts_evaluated": len(task_faithfulness),
                "top_k": top_k,
                "passed": passed,
                "threshold": FAITHFULNESS_THRESHOLD,
                "per_prompt": per_prompt_details,
            },
        ))

    # Aggregate result
    if all_faithfulness:
        agg_mean = float(np.mean(all_faithfulness))
        agg_std = float(np.std(all_faithfulness))
        agg_passed = agg_mean > FAITHFULNESS_THRESHOLD
        log(f"  Aggregate: faithfulness={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX29.clt_graph_faithfulness",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "graph_faithfulness": agg_mean,
                "graph_faithfulness_std": agg_std,
                "n_tasks_evaluated": len(all_faithfulness),
                "per_task_faithfulness": {
                    r.metadata["task"]: r.metadata["graph_faithfulness"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "top_k": top_k,
                "passed": agg_passed,
                "threshold": FAITHFULNESS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX29: CLT Attribution Graph Faithfulness")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top-attribution heads to keep (default: 10)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX29: CLT ATTRIBUTION GRAPH FAITHFULNESS")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_clt_graph_faithfulness(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        top_k=args.top_k,
    )

    out = args.out or "EX29_clt_graph_faithfulness.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
