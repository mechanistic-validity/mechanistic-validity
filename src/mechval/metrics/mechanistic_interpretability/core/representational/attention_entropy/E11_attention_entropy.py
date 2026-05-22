"""Attention Entropy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         E11 — Attention Entropy
Categories:     representational
Validity layer: Construct
Criteria:       C2 Structural plausibility, C3 Task specificity
Establishes:    Attention focus patterns differ between circuit and non-circuit heads
Requires:       GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes attention entropy per head on standard prompts. Low entropy =
focused attention (PTH heads have entropy ~0.02, attending almost
exclusively to position t-1). High entropy = diffuse attention.

Per-circuit outputs:
  - mean/min/max/std of attention entropy across circuit heads
  - ratio vs non-circuit heads
  - per-head entropy values

Usage:
    uv run python E11_attention_entropy.py --tasks ioi sva greater_than
    uv run python E11_attention_entropy.py --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_completed_tasks,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

JSONL_FILE = "E11_attention_entropy.jsonl"


@torch.no_grad()
def compute_attention_entropy(model, prompts, n_prompts: int = 40):
    """Compute mean attention entropy per head across prompts."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    entropy_sums = np.zeros((n_layers, n_heads))
    count = 0

    for prompt in prompts[:n_prompts]:
        tokens = model.to_tokens(prompt.text).to(model.cfg.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "attn.hook_pattern" in n)

        for L in range(n_layers):
            pattern = cache[f"blocks.{L}.attn.hook_pattern"]  # (batch, n_heads, seq, seq)
            for H in range(n_heads):
                attn = pattern[0, H]  # (seq, seq)
                attn_clamped = attn.clamp(min=1e-10)
                ent = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (seq,)
                entropy_sums[L, H] += float(ent.mean().item())

        count += 1

    if count == 0:
        return {}

    entropy_means = entropy_sums / count

    metrics = {}
    for L in range(n_layers):
        for H in range(n_heads):
            metrics[(L, H)] = float(entropy_means[L, H])

    return metrics


@torch.no_grad()
def run(model=None, tasks=None, device="cpu", model_name="gpt2", n_prompts=40,
        save=True, output_dir=None, resume=True):
    if model is None:
        model = load_model(model_name, device)
    tasks = tasks or CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}

    if output_dir is not None:
        from mechval.metrics.common import set_data_dir
        set_data_dir(output_dir)

    done_tasks, _ = load_completed_tasks(JSONL_FILE) if resume else (set(), [])

    log("=" * 60)
    log("E11: ATTENTION ENTROPY")
    log("=" * 60)

    results = []
    for task in tasks:
        if task in done_tasks:
            log(f"  {task}: already done, skipping")
            continue

        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        non_circuit = all_heads - circuit_heads
        log(f"  {task}: {len(circuit_heads)} circuit heads")

        prompts = generate_prompts(task, model.tokenizer, n_prompts)
        log(f"    Computing attention entropy on {len(prompts)} prompts...")
        entropy = compute_attention_entropy(model, prompts, n_prompts)

        if not entropy:
            log(f"    No entropy computed, skipping")
            continue

        circuit_vals = [entropy[(L, H)] for L, H in circuit_heads if (L, H) in entropy]
        non_circuit_vals = [entropy[(L, H)] for L, H in non_circuit if (L, H) in entropy]

        mean_circuit = float(np.mean(circuit_vals))
        mean_non_circuit = float(np.mean(non_circuit_vals))
        min_circuit = float(np.min(circuit_vals))
        max_circuit = float(np.max(circuit_vals))
        ratio = mean_circuit / (mean_non_circuit + 1e-10)

        log(f"    circuit_mean={mean_circuit:.3f}  non_circuit={mean_non_circuit:.3f}  "
            f"min={min_circuit:.3f}  ratio={ratio:.3f}")

        per_head = {f"L{L}H{H}": round(entropy[(L, H)], 4)
                    for L, H in sorted(circuit_heads) if (L, H) in entropy}

        result = EvalResult(
            metric_id="E11.attention_entropy",
            value=mean_circuit,
            baseline_random=mean_non_circuit,
            n_samples=len(circuit_vals),
            metadata={
                "task": task,
                "metric": "attention_entropy",
                "circuit_mean": mean_circuit,
                "circuit_min": min_circuit,
                "circuit_max": max_circuit,
                "circuit_std": float(np.std(circuit_vals)),
                "non_circuit_mean": mean_non_circuit,
                "ratio": ratio,
                "per_head": per_head,
                "interpretation": "low entropy = focused attention (PTH-like)",
            },
        )
        results.append(result)
        if save:
            save_incremental(result, JSONL_FILE)

    if save and results:
        save_results(results, "E11_attention_entropy.json")
    log(f"\nDone. {len(results)} new results across {len(tasks)} tasks.")
    return results


@torch.no_grad()
def main():
    parser = parse_common_args("E11: Attention Entropy")
    args = parser.parse_args()
    return run(
        model=None, tasks=args.tasks, device=args.device, model_name=args.model,
        n_prompts=args.n_prompts, save=True,
        output_dir=getattr(args, "data_dir", None),
    )


if __name__ == "__main__":
    main()
