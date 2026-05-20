"""Convergent Evolution: Cross-Task Structural Similarity
Tests whether circuits for related tasks converge on similar
architectural structure (layer distribution, relative head positions),
analogous to convergent evolution producing similar solutions
independently.

Pass: structural_overlap > 0.5
Ref: Conway Morris 2003, Life's Solution, Cambridge University Press

Usage:
    uv run python GN4_convergent_evolution.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Convergent Evolution",
    paper_ref="Conway Morris 2003, Cambridge University Press",
    paper_cite="Conway Morris 2003, Life's Solution: Inevitable Humans in a Lonely Universe",
    description="Tests whether different tasks produce structurally similar circuits (layer/position overlap)",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

OVERLAP_THRESHOLD = 0.5


@torch.no_grad()
def run_convergent_evolution(model, tasks: list[str],
                             n_prompts: int = 40) -> list[EvalResult]:
    """Compare structural features of circuits across task pairs."""
    results = []

    # Collect circuit structure per task
    task_info: dict[str, dict] = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for task in tasks:
        heads = get_circuit_heads(task)
        if not heads:
            continue

        layers_used = sorted({L for L, H in heads})
        # Normalized layer positions (0 to 1)
        norm_positions = [L / max(n_layers - 1, 1) for L, H in heads]
        # Layer density: fraction of heads used per layer
        layer_density = np.zeros(n_layers)
        for L, H in heads:
            layer_density[L] += 1.0 / n_heads

        task_info[task] = {
            "heads": heads,
            "layers_used": set(layers_used),
            "norm_positions": norm_positions,
            "layer_density": layer_density,
            "n_heads": len(heads),
            "mean_layer": float(np.mean([L for L, H in heads])),
            "layer_span": max(layers_used) - min(layers_used) if len(layers_used) > 1 else 0,
        }

    task_list = sorted(task_info.keys())
    if len(task_list) < 2:
        log("  Need >= 2 tasks with circuit heads")
        return results

    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            ta, tb = task_list[i], task_list[j]
            info_a, info_b = task_info[ta], task_info[tb]

            # 1. Head overlap (Jaccard of exact head positions)
            head_jaccard = (len(info_a["heads"] & info_b["heads"])
                           / max(len(info_a["heads"] | info_b["heads"]), 1))

            # 2. Layer overlap (Jaccard of layers used)
            layer_jaccard = (len(info_a["layers_used"] & info_b["layers_used"])
                            / max(len(info_a["layers_used"] | info_b["layers_used"]), 1))

            # 3. Density correlation (cosine similarity of layer density vectors)
            da = info_a["layer_density"]
            db = info_b["layer_density"]
            dot = float(np.dot(da, db))
            norm_a = float(np.linalg.norm(da))
            norm_b = float(np.linalg.norm(db))
            density_cosine = dot / max(norm_a * norm_b, 1e-8)

            # 4. Size similarity (ratio of smaller to larger circuit)
            size_ratio = min(info_a["n_heads"], info_b["n_heads"]) / max(info_a["n_heads"], info_b["n_heads"], 1)

            # Composite structural overlap score
            structural_overlap = (head_jaccard + layer_jaccard + density_cosine + size_ratio) / 4.0
            passed = structural_overlap > OVERLAP_THRESHOLD

            log(f"  {ta} x {tb}: overlap={structural_overlap:.3f} "
                f"(head_J={head_jaccard:.2f}, layer_J={layer_jaccard:.2f}, "
                f"density_cos={density_cosine:.2f}, size_r={size_ratio:.2f}) "
                f"[{'PASS' if passed else 'FAIL'}]")

            results.append(EvalResult(
                metric_id="GN4.convergent_evolution",
                value=structural_overlap,
                n_samples=info_a["n_heads"] + info_b["n_heads"],
                instrument_info=INSTRUMENT_INFO,
                metadata={
                    "task": f"{ta}_x_{tb}",
                    "task_a": ta,
                    "task_b": tb,
                    "n_heads_a": info_a["n_heads"],
                    "n_heads_b": info_b["n_heads"],
                    "head_jaccard": float(head_jaccard),
                    "layer_jaccard": float(layer_jaccard),
                    "density_cosine": float(density_cosine),
                    "size_ratio": float(size_ratio),
                    "structural_overlap": float(structural_overlap),
                    "shared_heads": sorted(info_a["heads"] & info_b["heads"]),
                    "mean_layer_a": info_a["mean_layer"],
                    "mean_layer_b": info_b["mean_layer"],
                    "passed": passed,
                    "threshold": OVERLAP_THRESHOLD,
                },
            ))

    return results


def main():
    parser = parse_common_args("GN4: Convergent Evolution")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GN4: CONVERGENT EVOLUTION")
    log("=" * 60)

    out = args.out or "GN4_convergent_evolution.json"
    jsonl_out = out.replace(".json", ".jsonl")

    results = run_convergent_evolution(model, tasks, args.n_prompts)
    for r in results:
        save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} task-pairs evaluated.")


if __name__ == "__main__":
    main()
