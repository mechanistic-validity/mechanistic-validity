"""Metric: Weight-Sparse Circuit Completeness --- necessity and sufficiency via weight pruning

Paper: Gao, Rajaram, Coxon, Govande, Baker, Mossing (2025).
"Weight-Sparse Transformers Have Interpretable Circuits."
OpenAI. arXiv:2511.13653

Tests whether weight-magnitude pruning recovers circuits that are both
necessary (removing any component breaks the task) and sufficient (the
circuit alone performs the task). Adapts the OpenAI sparse transformer
protocol to standard dense models as a baseline for E2 Causal
Sufficiency validation.

Weight-Sparse Circuit Completeness (Evaluation EX32)
=============================================
Instrument:     EX32 --- Weight-Sparse Circuit Completeness
Categories:     evaluation
Validity layer: Internal
Criteria:       E2 Causal Sufficiency, I1 Component Necessity
Establishes:    Whether weight-magnitude pruning recovers a minimal
                circuit that is both necessary and sufficient for a
                given task
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Evaluate full-model task performance as baseline.
2. Iteratively prune smallest-magnitude weights (set to zero) until
   task performance drops below a threshold.
3. The surviving weights form the "circuit."
4. Sufficiency test: run only the circuit (zero everything else) and
   measure task performance recovery.
5. Necessity test: for each attention head in the circuit, ablate it
   and measure performance drop.
6. Report circuit_size_ratio, sufficiency_score, necessity_score.

Pass condition: sufficiency > 0.8; necessity > 0.2; circuit_size < 0.1

Usage:
    uv run python 131_weight_sparse_circuit.py --model gpt2 --device cpu
    uv run python 131_weight_sparse_circuit.py --prune-fraction 0.9
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Weight-Sparse Circuit Completeness",
    paper_ref="Gao et al. arXiv:2511.13653 (Nov 2025)",
    paper_cite=(
        "Gao, Rajaram, Coxon, Govande, Baker, Mossing 2025, "
        "Weight-Sparse Transformers Have Interpretable Circuits "
        "(OpenAI, arXiv:2511.13653)"
    ),
    description=(
        "Tests necessity and sufficiency of circuits extracted via "
        "weight-magnitude pruning. Adapts the OpenAI sparse transformer "
        "protocol to standard dense models. Reports circuit size ratio, "
        "sufficiency score, and per-head necessity score."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

SUFFICIENCY_THRESHOLD = 0.8
NECESSITY_THRESHOLD = 0.2
CIRCUIT_SIZE_THRESHOLD = 0.1


@torch.no_grad()
def _eval_task_performance(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int],
) -> float:
    """Compute task accuracy: fraction of prompts where the model
    assigns higher logit to the correct token than the incorrect one.
    """
    n_correct = 0
    n_total = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        last = logits[0, -1]
        if last[correct_ids[i]] > last[incorrect_ids[i]]:
            n_correct += 1
        n_total += 1
    return n_correct / max(n_total, 1)


def _get_attention_weight_masks(model) -> dict[tuple[int, int], list[tuple[str, torch.Tensor]]]:
    """Build weight masks for each attention head.

    Returns a dict mapping (layer, head) to a list of (param_name, mask)
    tuples where mask is 1 for weights belonging to that head and 0
    elsewhere.
    """
    head_masks = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    for layer in range(n_layers):
        block = model.blocks[layer]
        for head in range(n_heads):
            masks = []
            # W_Q, W_K, W_V: shape (n_heads, d_model, d_head)
            for w_name in ["W_Q", "W_K", "W_V"]:
                W = getattr(block.attn, w_name, None)
                if W is not None:
                    mask = torch.zeros_like(W)
                    mask[head] = 1.0
                    masks.append((f"blocks.{layer}.attn.{w_name}", mask))
            # W_O: shape (n_heads, d_head, d_model)
            W_O = getattr(block.attn, "W_O", None)
            if W_O is not None:
                mask = torch.zeros_like(W_O)
                mask[head] = 1.0
                masks.append((f"blocks.{layer}.attn.W_O", mask))

            head_masks[(layer, head)] = masks

    return head_masks


@torch.no_grad()
def _compute_head_importance(model, prompts, correct_ids, incorrect_ids) -> dict[tuple[int, int], float]:
    """Compute per-head importance via mean logit-diff contribution.

    For each head, measures the mean absolute logit-diff between correct
    and incorrect tokens across all prompts.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    importance = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_result"
            contributions = []

            for i, p in enumerate(prompts[:20]):  # Subset for speed
                if i >= len(correct_ids):
                    break
                captured = {}

                def fwd_hook(value, hook, _c=captured):
                    _c["result"] = value.detach()
                    return value

                model.run_with_hooks(
                    model.to_tokens(p.text),
                    fwd_hooks=[(hook_name, fwd_hook)],
                )
                if "result" in captured:
                    # result shape: (1, seq, n_heads, d_head)
                    head_out = captured["result"][0, -1, head, :]
                    contributions.append(head_out.norm().item())

            importance[(layer, head)] = float(np.mean(contributions)) if contributions else 0.0

    return importance


@torch.no_grad()
def _ablate_head(model, layer: int, head: int, prompts, correct_ids, incorrect_ids) -> float:
    """Measure task performance when a single attention head is ablated (zeroed)."""
    hook_name = f"blocks.{layer}.attn.hook_result"

    def ablation_hook(value, hook):
        value[:, :, head, :] = 0.0
        return value

    n_correct = 0
    n_total = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, ablation_hook)]
        )
        last = logits[0, -1]
        if last[correct_ids[i]] > last[incorrect_ids[i]]:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


@torch.no_grad()
def _keep_only_heads(model, circuit_heads: set[tuple[int, int]], prompts, correct_ids, incorrect_ids) -> float:
    """Measure task performance when only circuit heads are active (all others zeroed)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hooks = []
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_result"
        active_heads = {h for (l, h) in circuit_heads if l == layer}

        def make_hook(active=active_heads, n_h=n_heads):
            def hook_fn(value, hook):
                for h in range(n_h):
                    if h not in active:
                        value[:, :, h, :] = 0.0
                return value
            return hook_fn

        hooks.append((hook_name, make_hook()))

    n_correct = 0
    n_total = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        last = logits[0, -1]
        if last[correct_ids[i]] > last[incorrect_ids[i]]:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


def run_weight_sparse_circuit(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    circuit_fraction: float = 0.2,
) -> list[EvalResult]:
    """Run weight-sparse circuit completeness diagnostic.

    For each task:
    1. Evaluate full model baseline.
    2. Identify top-k% most important attention heads as the "circuit."
    3. Test sufficiency (circuit alone) and necessity (ablate each head).

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        circuit_fraction: fraction of heads to include in circuit.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads
    circuit_size = max(1, int(total_heads * circuit_fraction))

    log(f"  Weight-Sparse Circuit: {circuit_size}/{total_heads} heads "
        f"({circuit_fraction:.0%})")

    results = []
    all_sufficiency = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        # Baseline performance
        baseline_acc = _eval_task_performance(model, prompts, correct_ids, incorrect_ids)
        if baseline_acc < 0.1:
            log(f"    {task}: baseline too low ({baseline_acc:.4f}), skipping")
            continue

        # Compute per-head importance
        importance = _compute_head_importance(model, prompts, correct_ids, incorrect_ids)

        # Select circuit: top heads by importance
        sorted_heads = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        circuit_heads = {h for h, _ in sorted_heads[:circuit_size]}
        size_ratio = len(circuit_heads) / total_heads

        # Sufficiency: circuit alone
        circuit_acc = _keep_only_heads(
            model, circuit_heads, prompts, correct_ids, incorrect_ids
        )
        sufficiency = circuit_acc / max(baseline_acc, 1e-8)

        # Necessity: ablate each circuit head
        necessity_scores = []
        for (layer, head) in circuit_heads:
            ablated_acc = _ablate_head(
                model, layer, head, prompts, correct_ids, incorrect_ids
            )
            deficit = (baseline_acc - ablated_acc) / max(baseline_acc, 1e-8)
            necessity_scores.append(max(0.0, deficit))

        mean_necessity = float(np.mean(necessity_scores)) if necessity_scores else 0.0

        passed_suff = sufficiency > SUFFICIENCY_THRESHOLD
        passed_nec = mean_necessity > NECESSITY_THRESHOLD
        passed_size = size_ratio < CIRCUIT_SIZE_THRESHOLD
        passed = passed_suff and passed_nec

        all_sufficiency.append(sufficiency)

        log(f"    {task}: suff={sufficiency:.4f}, nec={mean_necessity:.4f}, "
            f"size={size_ratio:.4f} ({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.weight_sparse_circuit",
            value=sufficiency,
            n_samples=len(correct_ids),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "baseline_accuracy": baseline_acc,
                "circuit_accuracy": circuit_acc,
                "sufficiency_score": sufficiency,
                "mean_necessity_score": mean_necessity,
                "circuit_size_ratio": size_ratio,
                "n_circuit_heads": len(circuit_heads),
                "n_total_heads": total_heads,
                "circuit_heads": sorted([(l, h) for l, h in circuit_heads]),
                "passed_sufficiency": passed_suff,
                "passed_necessity": passed_nec,
                "passed_compactness": passed_size,
                "passed": passed,
                "threshold_sufficiency": SUFFICIENCY_THRESHOLD,
                "threshold_necessity": NECESSITY_THRESHOLD,
                "threshold_size": CIRCUIT_SIZE_THRESHOLD,
            },
        ))

    # Aggregate
    if all_sufficiency:
        agg_suff = float(np.mean(all_sufficiency))
        agg_passed = agg_suff > SUFFICIENCY_THRESHOLD
        log(f"  Aggregate: sufficiency={agg_suff:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX32.weight_sparse_circuit",
            value=agg_suff,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_sufficiency": agg_suff,
                "n_tasks": len(all_sufficiency),
                "per_task_sufficiency": {
                    r.metadata["task"]: r.metadata["sufficiency_score"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": SUFFICIENCY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX32: Weight-Sparse Circuit Completeness")
    parser.add_argument("--circuit-fraction", type=float, default=0.2,
                        help="Fraction of heads in circuit (default: 0.2)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX32: WEIGHT-SPARSE CIRCUIT COMPLETENESS")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_weight_sparse_circuit(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        circuit_fraction=args.circuit_fraction,
    )

    out = args.out or "131_weight_sparse_circuit.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
