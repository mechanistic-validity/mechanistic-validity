"""Metric: ModCirc Cross-Task Modularity --- circuit reuse across tasks

Paper: He, Zheng, Dong, Zhu, Chen, Li (2025). "Towards Global-Level
Mechanistic Interpretability: A Perspective of Modular Circuits of
Large Language Models." ICML 2025, PMLR 267:22865-22880.
arXiv: see proceedings.mlr.press/v267/he25x.html

Measures whether circuit components discovered on one task reuse across
other tasks, operationalizing ModCirc's five criteria (task-agnosticity,
composability, interpretability, faithfulness, coverage). The primary
diagnostic is cross-task faithfulness: whether a circuit discovered on
task A maintains non-trivial faithfulness when evaluated on task B.
High cross-task faithfulness indicates the circuit captures a reusable
computational primitive rather than a task-specific artifact.

ModCirc Cross-Task Modularity (Evaluation EX20)
=============================================
Instrument:     EX20 --- ModCirc Cross-Task Modularity
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity (cross-task), E2 Causal Sufficiency
Establishes:    Whether circuits are reusable across tasks (model property)
                or task-specific artifacts
Requires:       CPU or GPU, model, circuits for multiple tasks
=============================================

Core logic:
1. For each pair of tasks (A, B) that have defined circuits:
   a. Get circuit_A (heads identified for task A).
   b. Evaluate circuit_A's faithfulness on task B's prompts.
   c. Compute node overlap between circuit_A and circuit_B.
2. Cross-task faithfulness = faithfulness of circuit_A on task_B prompts.
3. Node overlap = Jaccard similarity between circuit head sets.
4. Aggregate: mean cross-task faithfulness across all valid pairs.

Pass condition: mean_cross_task_faithfulness > 0.4

Usage:
    uv run python 122_modcirc.py --model gpt2 --device cpu
    uv run python 122_modcirc.py --n-prompts 40 --tasks ioi greater_than
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="ModCirc Cross-Task Modularity",
    paper_ref="He et al. ICML 2025",
    paper_cite=(
        "He, Zheng, Dong, Zhu, Chen, Li 2025, "
        "Towards Global-Level Mechanistic Interpretability: "
        "A Perspective of Modular Circuits of Large Language Models "
        "(ICML 2025, PMLR 267:22865)"
    ),
    description=(
        "Measures whether circuit components discovered on one task "
        "maintain faithfulness on other tasks (cross-task modularity). "
        "The cross-task faithfulness score and node overlap Jaccard "
        "index are the primary diagnostics."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

CROSS_TASK_FAITHFULNESS_THRESHOLD = 0.4


def _node_overlap_jaccard(
    heads_a: set[tuple[int, int]], heads_b: set[tuple[int, int]]
) -> float:
    """Jaccard similarity between two circuit head sets."""
    if not heads_a and not heads_b:
        return 0.0
    intersection = len(heads_a & heads_b)
    union = len(heads_a | heads_b)
    return intersection / union if union > 0 else 0.0


def run_modcirc_modularity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Measure cross-task circuit modularity.

    For each pair of tasks with defined circuits, evaluates whether
    circuit_A maintains faithfulness on task_B's prompts.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names with circuits.
        n_prompts: prompts per task for evaluation.

    Returns:
        List of EvalResult, one per task pair plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    # Filter to tasks with circuits
    task_circuits: dict[str, set[tuple[int, int]]] = {}
    task_prompts = {}
    task_correct = {}
    task_incorrect = {}

    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            continue
        correct, incorrect = get_token_ids(prompts, model.tokenizer)
        if not correct:
            continue
        task_circuits[task] = heads
        task_prompts[task] = prompts
        task_correct[task] = correct
        task_incorrect[task] = incorrect

    valid_tasks = list(task_circuits.keys())
    log(f"  Tasks with circuits: {len(valid_tasks)}")

    if len(valid_tasks) < 2:
        log("  Need at least 2 tasks with circuits for cross-task analysis")
        return []

    # Calibrate mean activations for ablation
    all_prompts = []
    for t in valid_tasks:
        all_prompts.extend(task_prompts[t][:20])
    mean_z = calibrate_mean_z(model, all_prompts, n_calibration=min(100, len(all_prompts)))

    results = []
    all_cross_faiths = []
    all_overlaps = []

    for task_a in valid_tasks:
        circuit_a = task_circuits[task_a]

        for task_b in valid_tasks:
            if task_a == task_b:
                continue

            # Evaluate circuit_A's faithfulness on task_B
            cross_faith = compute_faithfulness(
                model,
                task_prompts[task_b],
                task_correct[task_b],
                task_incorrect[task_b],
                circuit_a,
                mean_z,
            )

            # Node overlap
            circuit_b = task_circuits[task_b]
            overlap = _node_overlap_jaccard(circuit_a, circuit_b)

            all_cross_faiths.append(cross_faith)
            all_overlaps.append(overlap)

            passed = cross_faith > CROSS_TASK_FAITHFULNESS_THRESHOLD

            log(f"    {task_a} -> {task_b}: faith={cross_faith:.4f}, "
                f"overlap={overlap:.4f} ({'PASS' if passed else 'FAIL'})")

            results.append(EvalResult(
                metric_id="EX20.modcirc_modularity",
                value=cross_faith,
                n_samples=len(task_correct[task_b]),
                instrument_info=INSTRUMENT_INFO,
                metadata={
                    "task": f"{task_a}->{task_b}",
                    "source_task": task_a,
                    "target_task": task_b,
                    "cross_task_faithfulness": cross_faith,
                    "node_overlap_jaccard": overlap,
                    "n_source_heads": len(circuit_a),
                    "n_target_heads": len(circuit_b),
                    "n_shared_heads": len(circuit_a & circuit_b),
                    "passed": passed,
                    "threshold": CROSS_TASK_FAITHFULNESS_THRESHOLD,
                },
            ))

    # Aggregate
    if all_cross_faiths:
        agg_faith = float(np.mean(all_cross_faiths))
        agg_faith_std = float(np.std(all_cross_faiths))
        agg_overlap = float(np.mean(all_overlaps))
        agg_passed = agg_faith > CROSS_TASK_FAITHFULNESS_THRESHOLD

        log(f"  Aggregate: cross_faith={agg_faith:.4f} +/- {agg_faith_std:.4f}, "
            f"overlap={agg_overlap:.4f} ({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX20.modcirc_modularity",
            value=agg_faith,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_cross_task_faithfulness": agg_faith,
                "std_cross_task_faithfulness": agg_faith_std,
                "mean_node_overlap": agg_overlap,
                "n_pairs": len(all_cross_faiths),
                "n_tasks": len(valid_tasks),
                "passed": agg_passed,
                "threshold": CROSS_TASK_FAITHFULNESS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX20: ModCirc Cross-Task Modularity")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX20: MODCIRC CROSS-TASK MODULARITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_modcirc_modularity(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
    )

    out = args.out or "122_modcirc.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
