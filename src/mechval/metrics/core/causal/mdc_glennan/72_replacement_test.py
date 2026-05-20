"""Replacement Test (Functional Substitution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Internal
Criteria:       F3 Replacement Test (proposed)
Establishes:    Whether substituting the claimed function preserves behavior
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each circuit head, test whether a simplified functional substitute
preserves the model's task behavior:

Replacement #1 — Constant (mean activation):
    Replace the head's hook_z with its mean activation across calibration prompts.
    Recovery = logit_diff(replaced) / logit_diff(original).

Replacement #2 — Linear OV (ignore attention pattern):
    Replace the head's output with resid_pre @ W_V @ W_O (as if attention were
    uniform over the last position only). This tests whether the OV circuit alone,
    without the learned attention pattern, suffices.

High recovery for the constant replacement means the head contributes a
near-constant signal (e.g., a bias). High recovery for the linear OV
replacement means the attention pattern is not critical — the head's function
is dominated by its OV weights.

Usage:
    uv run python 72_replacement_test.py --tasks ioi sva
    uv run python 72_replacement_test.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
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
def recovery_constant_replacement(model, prompts, correct_ids, incorrect_ids,
                                  head: tuple[int, int],
                                  mean_z: torch.Tensor) -> float:
    """Replace a single head's hook_z with its mean activation.

    Returns recovery = sum(replaced_ld) / sum(clean_ld).
    """
    L, H = head
    numer, denom = 0.0, 0.0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        def hook_fn(z, hook, _L=L, _H=H):
            z[0, -1, _H, :] = mean_z[_L, _H].to(z.device)
            return z

        replaced_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)],
        )
        replaced_ld = logit_diff_from_logits(replaced_logits, correct_ids[i], incorrect_ids[i])

        numer += replaced_ld
        denom += clean_ld

    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


@torch.no_grad()
def recovery_linear_ov_replacement(model, prompts, correct_ids, incorrect_ids,
                                   head: tuple[int, int]) -> float:
    """Replace a head's hook_z with the linear OV prediction from resid_pre.

    For each prompt: z_replaced[head] = resid_pre[last] @ W_V[head].
    This ignores the attention pattern entirely — the head output is just
    the OV transform of the last-position residual stream.

    Returns recovery = sum(replaced_ld) / sum(clean_ld).
    """
    L, H = head
    W_V = model.W_V[L, H]  # (d_model, d_head)
    numer, denom = 0.0, 0.0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)

        # Get resid_pre for computing the linear replacement
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n, _L=L: n == f"blocks.{_L}.hook_resid_pre",
        )
        resid_pre_last = cache[f"blocks.{L}.hook_resid_pre"][0, -1]  # (d_model,)
        linear_z = resid_pre_last @ W_V  # (d_head,)

        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        def hook_fn(z, hook, _H=H, _linear_z=linear_z):
            z[0, -1, _H, :] = _linear_z.to(z.device)
            return z

        replaced_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)],
        )
        replaced_ld = logit_diff_from_logits(replaced_logits, correct_ids[i], incorrect_ids[i])

        numer += replaced_ld
        denom += clean_ld

    if abs(denom) < 1e-8:
        return 0.0
    return numer / denom


def run_replacement_test(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
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

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        constant_recoveries = {}
        linear_ov_recoveries = {}

        for L, H in sorted(circuit_heads):
            rec_const = recovery_constant_replacement(
                model, prompts, correct_ids, incorrect_ids, (L, H), mean_z,
            )
            rec_ov = recovery_linear_ov_replacement(
                model, prompts, correct_ids, incorrect_ids, (L, H),
            )
            constant_recoveries[f"L{L}H{H}"] = rec_const
            linear_ov_recoveries[f"L{L}H{H}"] = rec_ov
            log(f"    L{L}H{H}: constant={rec_const:.4f}  linear_ov={rec_ov:.4f}")

        mean_const = float(np.mean(list(constant_recoveries.values())))
        mean_ov = float(np.mean(list(linear_ov_recoveries.values())))

        log(f"    mean constant recovery={mean_const:.4f}")
        log(f"    mean linear OV recovery={mean_ov:.4f}")

        results.append(EvalResult(
            metric_id="F3.replacement_constant",
            value=mean_const,
            baseline_random=1.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head_recovery": constant_recoveries,
                "replacement_type": "constant_mean_activation",
                "n_circuit_heads": len(circuit_heads),
                "interpretation": "recovery when replacing head with mean activation (1.0 = perfect, 0.0 = head is critical)",
            },
        ))

        results.append(EvalResult(
            metric_id="F3.replacement_linear_ov",
            value=mean_ov,
            baseline_random=1.0,
            n_samples=len(circuit_heads),
            metadata={
                "task": task,
                "per_head_recovery": linear_ov_recoveries,
                "replacement_type": "linear_ov_no_attention",
                "n_circuit_heads": len(circuit_heads),
                "interpretation": "recovery when replacing head with resid_pre @ W_V (1.0 = attention pattern irrelevant)",
            },
        ))

    return results


def main():
    parser = parse_common_args("F3: Replacement Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("F3: REPLACEMENT TEST (FUNCTIONAL SUBSTITUTION)")
    log("=" * 60)

    results = run_replacement_test(model, tasks, args.n_prompts)

    out = args.out or "72_replacement_test.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results across "
        f"{len(set(r.metadata['task'] for r in results))} tasks.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}/{r.metric_id}: recovery={r.value:.4f}")


if __name__ == "__main__":
    main()
