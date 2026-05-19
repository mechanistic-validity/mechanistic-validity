"""Cross-Task IIA Transfer (Metric #16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A07 — Granger/Transfer Entropy
Categories:     causal, information
Validity layer: Internal
Criteria:       I4 Consistency
Establishes:    DAS rotations are task-specific (high diagonal, low off-diagonal in transfer matrix)
Requires:       GPU, model
Doc:            /instruments_v2/causal/a07-granger-te
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Train a DAS rotation on task A, then test IIA on task B. For each pair
(A, B): train rotation using A's prompts and circuit heads, then evaluate
IIA using B's prompts with A's rotation.

Report transfer matrix (rows=train task, cols=test task). High diagonal +
low off-diagonal = task-specific representations.

Usage:
    uv run python 32_cross_task_iia_transfer.py --tasks ioi sva --n-prompts 40
    uv run python 32_cross_task_iia_transfer.py --device cuda
"""
import importlib

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
    parse_common_args,
    save_results,
)

das_module = importlib.import_module("01_das_iia")
make_counterfactual_pairs = das_module.make_counterfactual_pairs
train_rotation = das_module.train_rotation
compute_iia_with_rotation = das_module.compute_iia_with_rotation


def run_cross_task_transfer(model, tasks: list[str], n_prompts: int = 40,
                            d_sub: int = 2) -> list[EvalResult]:
    tokenizer = model.tokenizer
    device = str(model.cfg.device)
    rng = np.random.RandomState(42)

    # Pre-compute per-task data: prompts, ids, pairs, best rotation + head
    task_data = {}
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

        train_pairs = make_counterfactual_pairs(train_prompts, train_correct, train_incorrect, rng)
        test_pairs = make_counterfactual_pairs(test_prompts, test_correct, test_incorrect, rng)

        # Train rotation on best head
        best_iia = 0.0
        best_head = None
        best_rotation = None

        for L, H in sorted(circuit_heads):
            R = train_rotation(model, train_prompts, train_correct, train_incorrect,
                               train_pairs, L, H, d_sub, device, n_steps=150)
            iia = compute_iia_with_rotation(model, test_prompts, test_correct, test_incorrect,
                                            test_pairs, L, H, R, device)
            if iia > best_iia:
                best_iia = iia
                best_head = (L, H)
                best_rotation = R

        if best_head is None:
            continue

        task_data[task] = {
            "circuit_heads": circuit_heads,
            "test_prompts": test_prompts,
            "test_correct": test_correct,
            "test_incorrect": test_incorrect,
            "test_pairs": test_pairs,
            "best_head": best_head,
            "best_rotation": best_rotation,
            "self_iia": best_iia,
        }
        log(f"  {task}: trained rotation on L{best_head[0]}H{best_head[1]}, self-IIA={best_iia:.3f}")

    # Build transfer matrix
    valid_tasks = sorted(task_data.keys())
    n_tasks = len(valid_tasks)
    transfer_matrix = np.zeros((n_tasks, n_tasks))
    transfer_details = {}

    for i, train_task in enumerate(valid_tasks):
        train_info = task_data[train_task]
        train_L, train_H = train_info["best_head"]
        train_R = train_info["best_rotation"]

        for j, test_task in enumerate(valid_tasks):
            test_info = task_data[test_task]

            if train_task == test_task:
                # Diagonal: use self-IIA
                transfer_matrix[i, j] = train_info["self_iia"]
            else:
                # Off-diagonal: apply train_task's rotation to test_task's data
                # Use the training task's head location on test task's prompts
                test_pairs = test_info["test_pairs"]
                iia = compute_iia_with_rotation(
                    model, test_info["test_prompts"],
                    test_info["test_correct"], test_info["test_incorrect"],
                    test_pairs, train_L, train_H, train_R, device)
                transfer_matrix[i, j] = iia

            transfer_details[f"{train_task}->{test_task}"] = float(transfer_matrix[i, j])

    # Compute specificity: diagonal mean vs off-diagonal mean
    if n_tasks >= 2:
        diag_mean = float(np.mean(np.diag(transfer_matrix)))
        mask = ~np.eye(n_tasks, dtype=bool)
        offdiag_mean = float(np.mean(transfer_matrix[mask]))
        specificity = diag_mean - offdiag_mean
    else:
        diag_mean = float(transfer_matrix[0, 0]) if n_tasks == 1 else 0.0
        offdiag_mean = 0.0
        specificity = diag_mean

    log(f"\n  Transfer specificity: diag={diag_mean:.3f} offdiag={offdiag_mean:.3f} "
        f"delta={specificity:.3f}")

    results = [EvalResult(
        metric_id="C32.cross_task_iia_transfer",
        value=specificity,
        n_samples=n_tasks * n_tasks,
        metadata={
            "tasks": valid_tasks,
            "transfer_matrix": transfer_matrix.tolist(),
            "transfer_details": transfer_details,
            "diagonal_mean": diag_mean,
            "offdiagonal_mean": offdiag_mean,
            "specificity": specificity,
            "subspace_dim": d_sub,
        },
    )]

    # Per-task self-IIA for reference
    for task in valid_tasks:
        results.append(EvalResult(
            metric_id="C32.self_iia",
            value=task_data[task]["self_iia"],
            n_samples=len(task_data[task]["test_prompts"]),
            metadata={
                "task": task,
                "best_head": list(task_data[task]["best_head"]),
            },
        ))

    return results


def main():
    parser = parse_common_args("C32: Cross-Task IIA Transfer")
    parser.add_argument("--subspace-dim", type=int, default=2)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C32: CROSS-TASK IIA TRANSFER (Metric #16)")
    log("=" * 60)

    results = run_cross_task_transfer(model, tasks, args.n_prompts, args.subspace_dim)

    out = args.out or "32_cross_task_iia_transfer.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
