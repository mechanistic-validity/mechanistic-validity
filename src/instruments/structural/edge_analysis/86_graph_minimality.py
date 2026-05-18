"""Graph Minimality (Graph Structure G5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G05 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G5 Graph Minimality
Establishes:    Whether the circuit's edge set is minimal (no redundant edges)
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Combine edge necessity with direction-aware testing: for each edge,
ablating it must cause >5% drop in logit diff AND the drop must be in
the task-relevant direction (decrease, not increase).

Pass condition: minimality ratio >= 0.8 (at least 80% of edges are necessary).

Usage:
    uv run python 86_graph_minimality.py --tasks ioi --n-prompts 40
    uv run python 86_graph_minimality.py --tasks ioi sva --device cpu
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
    calibrate_mean_z,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def test_edge_minimality(model, prompts, correct_ids, incorrect_ids,
                         edges, mean_z) -> dict:
    """Test each edge for necessity and directional relevance.

    Returns dict mapping edge -> {drop_frac, directional, necessary}.
    """
    results = {}

    for (up_l, up_h, down_l, down_h) in sorted(edges):
        clean_lds = []
        ablated_lds = []

        W_O = model.W_O[up_l]

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_logits, clean_cache = model.run_with_cache(tokens)
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

            # Mean-ablate upstream head's contribution at downstream layer
            up_z = clean_cache[f"blocks.{up_l}.attn.hook_z"][0, -1, up_h, :]
            up_mean = mean_z[up_l, up_h].to(up_z.device)
            diff = (up_z - up_mean) @ W_O[up_h]

            def ablate_hook(activation, hook, _diff=diff):
                activation[0, -1, :] -= _diff
                return activation

            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(f"blocks.{down_l}.hook_resid_pre", ablate_hook)])
            ablated_ld = logit_diff_from_logits(
                ablated_logits, correct_ids[i], incorrect_ids[i])

            clean_lds.append(clean_ld)
            ablated_lds.append(ablated_ld)

        mean_clean = np.mean(clean_lds)
        mean_ablated = np.mean(ablated_lds)

        if abs(mean_clean) < 1e-8:
            drop_frac = 0.0
        else:
            drop_frac = (mean_clean - mean_ablated) / abs(mean_clean)

        # Magnitude test: >5% drop
        magnitude_necessary = bool(drop_frac > 0.05)
        # Direction test: ablation should reduce logit diff (drop_frac > 0),
        # meaning the edge contributes in the task-relevant direction
        directional = bool(drop_frac > 0.0)
        # Both must hold
        necessary = magnitude_necessary and directional

        edge = (up_l, up_h, down_l, down_h)
        results[edge] = {
            "drop_frac": float(drop_frac),
            "magnitude_necessary": magnitude_necessary,
            "directional": directional,
            "necessary": necessary,
        }

    return results


@torch.no_grad()
def run_graph_minimality(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit, all_heads, all_edges = get_circuit_info(task)
        if circuit is None or not all_edges:
            log(f"  {task}: no circuit edges, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(all_edges)} edges, {len(prompts)} prompts")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        edge_results = test_edge_minimality(
            model, prompts, correct_ids, incorrect_ids, all_edges, mean_z)

        for edge, data in sorted(edge_results.items()):
            name = f"L{edge[0]}H{edge[1]}->L{edge[2]}H{edge[3]}"
            status = "NECESSARY" if data["necessary"] else (
                "wrong-dir" if not data["directional"] else "too-small")
            log(f"    {name}: drop={data['drop_frac']:.4f}  [{status}]")

        n_necessary = sum(1 for d in edge_results.values() if d["necessary"])
        n_total = len(edge_results)
        minimality_ratio = float(n_necessary / max(n_total, 1))
        n_wrong_dir = sum(1 for d in edge_results.values() if not d["directional"])

        passed = bool(minimality_ratio >= 0.8)

        log(f"    minimality={n_necessary}/{n_total} ({minimality_ratio:.0%})  "
            f"wrong_dir={n_wrong_dir}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G5.graph_minimality",
            value=minimality_ratio,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_edges": n_total,
                "n_necessary": n_necessary,
                "n_wrong_direction": n_wrong_dir,
                "minimality_ratio": minimality_ratio,
                "per_edge": {
                    f"L{e[0]}H{e[1]}->L{e[2]}H{e[3]}": d
                    for e, d in edge_results.items()
                },
                "passed": passed,
                "threshold_minimality": 0.8,
                "threshold_drop": 0.05,
            },
        ))

    return results


def main():
    parser = parse_common_args("G5: Graph Minimality")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G5: GRAPH MINIMALITY")
    log("=" * 60)

    out = args.out or "86_graph_minimality.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_graph_minimality(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: minimality={r.value:.2f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
