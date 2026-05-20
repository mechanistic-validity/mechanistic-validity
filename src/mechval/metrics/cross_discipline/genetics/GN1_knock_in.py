"""Knock-In Cross-Task Weight Similarity
Tests whether circuit heads performing analogous roles across different
tasks share similar weight structure, analogous to gene knock-in
experiments that test functional conservation.

Pass: mean_cross_task_similarity > 0.3
Ref: Capecchi 1989, Science 244:1288-1292

Usage:
    uv run python GN1_knock_in.py --tasks ioi sva --n-prompts 40
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
    name="Knock-In",
    paper_ref="Capecchi 1989, Science 244:1288-1292",
    paper_cite="Capecchi 1989, Altering the Genome by Homologous Recombination",
    description="Tests cross-task weight similarity of circuit heads (cosine similarity of flattened W_Q/W_K/W_V/W_O)",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

SIMILARITY_THRESHOLD = 0.3


@torch.no_grad()
def run_knock_in(model, tasks: list[str],
                 n_prompts: int = 40) -> list[EvalResult]:
    """Compare weight similarity of circuit heads across task pairs."""
    results = []

    # Collect circuit heads per task
    task_heads: dict[str, set[tuple[int, int]]] = {}
    for task in tasks:
        heads = get_circuit_heads(task)
        if heads:
            task_heads[task] = heads

    if len(task_heads) < 2:
        log("  Need >= 2 tasks with circuit heads for cross-task comparison")
        return results

    # Extract flattened weight vectors per head
    def get_head_weights(L: int, H: int) -> torch.Tensor:
        """Concatenate W_Q, W_K, W_V, W_O for a single head into a flat vector."""
        wq = model.W_Q[L, H].flatten()
        wk = model.W_K[L, H].flatten()
        wv = model.W_V[L, H].flatten()
        wo = model.W_O[L, H].flatten()
        return torch.cat([wq, wk, wv, wo]).float()

    task_list = sorted(task_heads.keys())

    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            task_a, task_b = task_list[i], task_list[j]
            heads_a = sorted(task_heads[task_a])
            heads_b = sorted(task_heads[task_b])

            # Find heads that appear in both circuits (same layer, head index)
            shared = sorted(set(heads_a) & set(heads_b))

            # Also compute similarity for all pairs of heads
            pair_sims = []
            pair_details = []

            if shared:
                # Direct comparison of shared heads
                for (L, H) in shared:
                    wa = get_head_weights(L, H)
                    wb = get_head_weights(L, H)
                    sim = torch.nn.functional.cosine_similarity(
                        wa.unsqueeze(0), wb.unsqueeze(0)).item()
                    pair_sims.append(sim)
                    pair_details.append({
                        "head": f"L{L}H{H}",
                        "type": "shared",
                        "cosine_sim": float(sim),
                    })

            # Cross-pair: each head in A vs each head in B
            cross_sims = []
            for (la, ha) in heads_a:
                wa = get_head_weights(la, ha)
                for (lb, hb) in heads_b:
                    wb = get_head_weights(lb, hb)
                    sim = torch.nn.functional.cosine_similarity(
                        wa.unsqueeze(0), wb.unsqueeze(0)).item()
                    cross_sims.append(sim)
                    pair_details.append({
                        "head_a": f"L{la}H{ha}",
                        "head_b": f"L{lb}H{hb}",
                        "type": "cross",
                        "cosine_sim": float(sim),
                    })

            all_sims = pair_sims + cross_sims if pair_sims else cross_sims
            if not all_sims:
                continue

            mean_sim = float(np.mean(all_sims))
            passed = mean_sim > SIMILARITY_THRESHOLD

            log(f"  {task_a} x {task_b}: mean_sim={mean_sim:.3f} "
                f"({len(shared)} shared, {len(cross_sims)} cross-pairs) "
                f"[{'PASS' if passed else 'FAIL'}]")

            results.append(EvalResult(
                metric_id="GN1.knock_in",
                value=mean_sim,
                n_samples=len(all_sims),
                instrument_info=INSTRUMENT_INFO,
                metadata={
                    "task": f"{task_a}_x_{task_b}",
                    "task_a": task_a,
                    "task_b": task_b,
                    "n_heads_a": len(heads_a),
                    "n_heads_b": len(heads_b),
                    "n_shared": len(shared),
                    "mean_similarity": mean_sim,
                    "max_similarity": float(np.max(all_sims)),
                    "min_similarity": float(np.min(all_sims)),
                    "passed": passed,
                    "threshold": SIMILARITY_THRESHOLD,
                    "pair_details": sorted(pair_details,
                                           key=lambda d: d["cosine_sim"],
                                           reverse=True)[:10],
                },
            ))

    return results


def main():
    parser = parse_common_args("GN1: Knock-In")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GN1: KNOCK-IN")
    log("=" * 60)

    out = args.out or "GN1_knock_in.json"
    jsonl_out = out.replace(".json", ".jsonl")

    results = run_knock_in(model, tasks, args.n_prompts)
    for r in results:
        save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} task-pairs evaluated.")


if __name__ == "__main__":
    main()
