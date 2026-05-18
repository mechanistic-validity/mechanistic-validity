"""Operation Specification (Component Function Characterization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Internal
Criteria:       F1 Operation Specification (proposed)
Establishes:    Whether input-output function of each circuit component can be specified quantitatively
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two complementary metrics for how well a head's function can be specified:

1. **Output consistency** (cross-prompt): compute the head's output (hook_z)
   across many prompts. Take the first principal component's variance ratio.
   High ratio = the head's output lives in a low-dimensional subspace =
   well-specified function. Low ratio = context-dependent, harder to characterize.

2. **Attention-weighted OV prediction**: for each prompt, compute the actual
   attention pattern, apply it to get the true input (attn @ resid), then
   predict output via W_V. This accounts for attention and measures how well
   the OV matrix characterizes the *attended* input → output mapping.

Circuit heads should have HIGHER specification scores than random heads,
because the mechanistic account claims to describe what each component does.

Pass: circuit heads' mean specification > non-circuit baseline mean.

Usage:
    uv run python 70_operation_specification.py --tasks ioi sva
    uv run python 70_operation_specification.py --device cpu --n-prompts 60
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
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
def collect_head_outputs(model, prompts, layer: int, head: int):
    """Collect hook_z and attention-weighted OV prediction for a head.

    Returns:
        actual_z: (n_prompts, d_head)
        attn_ov_pred: (n_prompts, d_head) — prediction using actual attention pattern
        consistency: float — first PC variance ratio of actual_z
    """
    actual_zs = []
    attn_ov_preds = []

    W_V = model.W_V[layer, head].cpu().float()

    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n, _L=layer: (
                n == f"blocks.{_L}.hook_resid_pre"
                or n == f"blocks.{_L}.attn.hook_z"
                or n == f"blocks.{_L}.attn.hook_pattern"
            ),
        )
        z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head].cpu().float()
        actual_zs.append(z)

        resid = cache[f"blocks.{layer}.hook_resid_pre"][0].cpu().float()
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, -1].cpu().float()
        attended_input = attn @ resid
        pred_z = attended_input @ W_V
        attn_ov_preds.append(pred_z)

    actual_z = torch.stack(actual_zs)
    attn_ov_pred = torch.stack(attn_ov_preds)

    centered = actual_z - actual_z.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        consistency = 0.0
    else:
        S = torch.linalg.svdvals(centered)
        total_var = (S ** 2).sum().item()
        consistency = float((S[0] ** 2).item() / total_var) if total_var > 1e-12 else 0.0

    return actual_z, attn_ov_pred, consistency


def r_squared(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    pred_flat = predicted.reshape(predicted.shape[0], -1).float()
    act_flat = actual.reshape(actual.shape[0], -1).float()
    ss_res = ((act_flat - pred_flat) ** 2).sum().item()
    mean_act = act_flat.mean(dim=0, keepdim=True)
    ss_tot = ((act_flat - mean_act) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def run_operation_specification(model, tasks: list[str],
                                n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
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

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        non_circuit = all_heads - circuit_heads

        circuit_scores = {}
        for L, H in sorted(circuit_heads):
            actual_z, attn_ov_pred, consistency = collect_head_outputs(
                model, prompts, L, H
            )
            attn_ov_r2 = r_squared(attn_ov_pred, actual_z)
            circuit_scores[f"L{L}H{H}"] = {
                "consistency": consistency,
                "attn_ov_r2": attn_ov_r2,
                "combined": (consistency + max(attn_ov_r2, 0)) / 2,
            }
            log(f"    L{L}H{H}: consistency={consistency:.4f}  "
                f"attn_ov_R2={attn_ov_r2:.4f}")

        rng = np.random.RandomState(42)
        n_baseline = min(len(non_circuit), max(len(circuit_heads), 10))
        baseline_list = sorted(non_circuit)
        rng.shuffle(baseline_list)
        baseline_list = baseline_list[:n_baseline]

        baseline_scores = []
        for L, H in baseline_list:
            actual_z, attn_ov_pred, consistency = collect_head_outputs(
                model, prompts, L, H
            )
            attn_ov_r2 = r_squared(attn_ov_pred, actual_z)
            baseline_scores.append((consistency + max(attn_ov_r2, 0)) / 2)

        circuit_combined = [s["combined"] for s in circuit_scores.values()]
        mean_circuit = float(np.mean(circuit_combined))
        mean_baseline = float(np.mean(baseline_scores)) if baseline_scores else 0.0
        passed = mean_circuit > mean_baseline

        log(f"    circuit mean={mean_circuit:.4f}  baseline={mean_baseline:.4f}  "
            f"{'PASS' if passed else 'FAIL'}")

        results.append(EvalResult(
            metric_id="F1.operation_specification",
            value=mean_circuit,
            baseline_random=mean_baseline,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head": circuit_scores,
                "baseline_mean": mean_baseline,
                "n_baseline": n_baseline,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("F1: Operation Specification")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("F1: OPERATION SPECIFICATION")
    log("=" * 60)

    results = run_operation_specification(model, tasks, args.n_prompts)

    out = args.out or "70_operation_specification.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across "
        f"{len(set(r.metadata['task'] for r in results))} tasks.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: combined={r.value:.4f} (baseline={r.baseline_random:.4f})  [{p}]")


if __name__ == "__main__":
    main()
