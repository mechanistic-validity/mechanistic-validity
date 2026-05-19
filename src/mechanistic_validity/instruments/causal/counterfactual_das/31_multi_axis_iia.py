"""Multi-Axis IIA (Metrics #10, #11)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal, representational
Validity layer: Internal + Representational
Criteria:       I2 Sufficiency
Establishes:    Joint and per-axis DAS rotations achieve IIA for multi-variable tasks
Requires:       GPU, model
Doc:            /instruments_v2/causal/a02-counterfactual-das
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For tasks with multiple causal variables (e.g. IOI: gender + name),
train DAS rotations for each axis separately and jointly, then measure
multi-axis IIA.

Metric #10 = multi-axis IIA on causal axes (joint intervention).
Metric #11 = control (zero-out one axis, measure residual IIA).

For tasks with only one causal variable, reports single-axis IIA.

Usage:
    uv run python 31_multi_axis_iia.py --tasks ioi sva --n-prompts 40
    uv run python 31_multi_axis_iia.py --device cuda
"""
import importlib
import random

import numpy as np
import torch
import torch.nn.functional as F

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

das_module = importlib.import_module("01_das_iia")
make_counterfactual_pairs = das_module.make_counterfactual_pairs
train_rotation = das_module.train_rotation
compute_iia_with_rotation = das_module.compute_iia_with_rotation

# Tasks that have two independent causal variables.
# For IOI: axis 0 = name identity, axis 1 = gender (IO vs S).
# For gendered_pronoun: axis 0 = subject, axis 1 = gender.
MULTI_AXIS_TASKS = {"ioi", "gendered_pronoun"}


def split_prompts_by_axis(prompts, correct_ids, incorrect_ids, task: str, axis: int):
    """Partition counterfactual pairs so they differ on exactly one axis.

    For IOI axis 0 (name): pairs share gender but differ in name.
    For IOI axis 1 (gender): pairs share name-structure but differ in gender.

    Falls back to standard pairing if metadata is unavailable.
    """
    pairs = []
    n = len(prompts)
    for i in range(n):
        candidates = []
        for j in range(n):
            if j == i:
                continue
            meta_i = getattr(prompts[i], "metadata", {}) or {}
            meta_j = getattr(prompts[j], "metadata", {}) or {}
            if task == "ioi" and axis == 0:
                # Same gender, different name (different correct token)
                if (meta_i.get("gender") == meta_j.get("gender")
                        and correct_ids[i] != correct_ids[j]):
                    candidates.append(j)
            elif task == "ioi" and axis == 1:
                # Same name structure, different gender
                if (meta_i.get("gender") != meta_j.get("gender")
                        and correct_ids[i] != correct_ids[j]):
                    candidates.append(j)
            else:
                if correct_ids[i] != correct_ids[j]:
                    candidates.append(j)
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        pairs.append((i, random.Random(42 + i).choice(candidates)))
    return pairs


@torch.no_grad()
def compute_joint_iia(model, prompts, correct_ids, incorrect_ids,
                      pairs, layer: int, head: int,
                      rotations: list[torch.Tensor], device: str) -> float:
    """IIA with joint intervention across multiple rotation axes."""
    n_correct = 0
    n_total = 0
    hook_name = f"blocks.{layer}.attn.hook_z"

    projs = []
    for R in rotations:
        R_dev = R.to(device)
        projs.append(R_dev @ R_dev.t())

    for base_idx, source_idx in pairs:
        base_tokens = model.to_tokens(prompts[base_idx].text).to(device)
        source_tokens = model.to_tokens(prompts[source_idx].text).to(device)

        _, base_cache = model.run_with_cache(base_tokens, names_filter=lambda n: n == hook_name)
        _, source_cache = model.run_with_cache(source_tokens, names_filter=lambda n: n == hook_name)

        base_z = base_cache[hook_name][0, -1, head]
        source_z = source_cache[hook_name][0, -1, head]

        # Apply all rotations jointly
        intervened_z = base_z.clone()
        for proj in projs:
            intervened_z = intervened_z - base_z @ proj + source_z @ proj

        def _hook(z, hook, _head=head, _z=intervened_z):
            z[0, -1, _head] = _z
            return z

        logits = model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, _hook)])
        source_correct_id = correct_ids[source_idx]
        source_incorrect_id = incorrect_ids[source_idx]
        ld = logit_diff_from_logits(logits, source_correct_id, source_incorrect_id)

        if ld > 0:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


@torch.no_grad()
def compute_residual_iia(model, prompts, correct_ids, incorrect_ids,
                         pairs, layer: int, head: int,
                         rotation_to_zero: torch.Tensor,
                         rotation_to_test: torch.Tensor,
                         device: str) -> float:
    """Zero out one axis, measure residual IIA on the other axis."""
    n_correct = 0
    n_total = 0
    hook_name = f"blocks.{layer}.attn.hook_z"

    R_zero = rotation_to_zero.to(device)
    proj_zero = R_zero @ R_zero.t()
    R_test = rotation_to_test.to(device)
    proj_test = R_test @ R_test.t()

    for base_idx, source_idx in pairs:
        base_tokens = model.to_tokens(prompts[base_idx].text).to(device)
        source_tokens = model.to_tokens(prompts[source_idx].text).to(device)

        _, base_cache = model.run_with_cache(base_tokens, names_filter=lambda n: n == hook_name)
        _, source_cache = model.run_with_cache(source_tokens, names_filter=lambda n: n == hook_name)

        base_z = base_cache[hook_name][0, -1, head]
        source_z = source_cache[hook_name][0, -1, head]

        # Zero out one axis, intervene on the other
        zeroed_z = base_z - base_z @ proj_zero  # remove axis
        intervened_z = zeroed_z - base_z @ proj_test + source_z @ proj_test

        def _hook(z, hook, _head=head, _z=intervened_z):
            z[0, -1, _head] = _z
            return z

        logits = model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, _hook)])
        source_correct_id = correct_ids[source_idx]
        source_incorrect_id = incorrect_ids[source_idx]
        ld = logit_diff_from_logits(logits, source_correct_id, source_incorrect_id)

        if ld > 0:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


def run_multi_axis_iia(model, tasks: list[str], n_prompts: int = 40,
                       d_sub: int = 2) -> list[EvalResult]:
    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    results = []
    rng = np.random.RandomState(42)

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

        n_train = len(prompts) // 2
        train_prompts = prompts[:n_train]
        test_prompts = prompts[n_train:]
        train_correct = correct_ids[:n_train]
        train_incorrect = incorrect_ids[:n_train]
        test_correct = correct_ids[n_train:]
        test_incorrect = incorrect_ids[n_train:]

        is_multi = task in MULTI_AXIS_TASKS
        n_axes = 2 if is_multi else 1

        log(f"  {task} ({len(circuit_heads)} heads, {n_axes} axes)...")

        best_joint_iia = 0.0
        best_head = None
        best_rotations = []
        per_head_results = {}

        for L, H in sorted(circuit_heads):
            axis_rotations = []
            axis_iias = []

            for axis in range(n_axes):
                if is_multi:
                    train_pairs = split_prompts_by_axis(
                        train_prompts, train_correct, train_incorrect, task, axis)
                else:
                    train_pairs = make_counterfactual_pairs(
                        train_prompts, train_correct, train_incorrect, rng)

                R = train_rotation(model, train_prompts, train_correct, train_incorrect,
                                   train_pairs, L, H, d_sub, device, n_steps=150)
                axis_rotations.append(R)

                test_pairs = make_counterfactual_pairs(
                    test_prompts, test_correct, test_incorrect, rng)
                axis_iia = compute_iia_with_rotation(
                    model, test_prompts, test_correct, test_incorrect,
                    test_pairs, L, H, R, device)
                axis_iias.append(axis_iia)

            # Joint IIA
            test_pairs = make_counterfactual_pairs(
                test_prompts, test_correct, test_incorrect, rng)
            if n_axes > 1:
                joint_iia = compute_joint_iia(
                    model, test_prompts, test_correct, test_incorrect,
                    test_pairs, L, H, axis_rotations, device)
            else:
                joint_iia = axis_iias[0]

            per_head_results[f"L{L}H{H}"] = {
                "per_axis_iia": axis_iias,
                "joint_iia": joint_iia,
            }

            if joint_iia > best_joint_iia:
                best_joint_iia = joint_iia
                best_head = (L, H)
                best_rotations = axis_rotations

        # Metric #10: multi-axis IIA
        log(f"    metric10 joint_IIA={best_joint_iia:.3f} (head={best_head})")
        results.append(EvalResult(
            metric_id="C31.multi_axis_iia",
            value=best_joint_iia,
            n_samples=len(test_prompts),
            metadata={
                "task": task,
                "n_axes": n_axes,
                "best_head": list(best_head) if best_head else None,
                "per_head": per_head_results,
                "subspace_dim": d_sub,
            },
        ))

        # Metric #11: control — zero one axis, measure residual
        if is_multi and best_head is not None and len(best_rotations) == 2:
            L_best, H_best = best_head
            test_pairs = make_counterfactual_pairs(
                test_prompts, test_correct, test_incorrect, rng)

            residual_axis0 = compute_residual_iia(
                model, test_prompts, test_correct, test_incorrect,
                test_pairs, L_best, H_best,
                best_rotations[1], best_rotations[0], device)

            residual_axis1 = compute_residual_iia(
                model, test_prompts, test_correct, test_incorrect,
                test_pairs, L_best, H_best,
                best_rotations[0], best_rotations[1], device)

            control_score = (residual_axis0 + residual_axis1) / 2.0
            log(f"    metric11 control: axis0_residual={residual_axis0:.3f} "
                f"axis1_residual={residual_axis1:.3f}")

            results.append(EvalResult(
                metric_id="C31.multi_axis_control",
                value=control_score,
                n_samples=len(test_prompts),
                metadata={
                    "task": task,
                    "residual_axis0": residual_axis0,
                    "residual_axis1": residual_axis1,
                    "best_head": list(best_head),
                },
            ))
        elif not is_multi:
            # Single-axis: control is N/A, report single-axis IIA
            log(f"    metric11 control: single-axis task, N/A")
            results.append(EvalResult(
                metric_id="C31.multi_axis_control",
                value=best_joint_iia,
                n_samples=len(test_prompts),
                metadata={
                    "task": task,
                    "note": "single_axis_task",
                    "best_head": list(best_head) if best_head else None,
                },
            ))

    return results


def main():
    parser = parse_common_args("C31: Multi-Axis IIA")
    parser.add_argument("--subspace-dim", type=int, default=2)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C31: MULTI-AXIS IIA (Metrics #10, #11)")
    log("=" * 60)

    results = run_multi_axis_iia(model, tasks, args.n_prompts, args.subspace_dim)

    out = args.out or "31_multi_axis_iia.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: {r.metric_id}={r.value:.3f}")


if __name__ == "__main__":
    main()
