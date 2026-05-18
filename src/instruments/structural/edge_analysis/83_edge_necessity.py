"""Edge Necessity (Graph Structure G2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     G02 — Graph Structure
Categories:     structural
Validity layer: Internal
Criteria:       G2 Edge Necessity
Establishes:    Whether specific edges (not just nodes) are necessary for circuit function
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each edge in the circuit, ablate just that edge (mean-ablate the
upstream head's contribution at the downstream position) and measure
the drop in logit diff.

An edge is "necessary" if ablating it causes >5% drop in logit diff
relative to the full model.

Pass condition: >=50% of edges are individually necessary.

Usage:
    uv run python 83_edge_necessity.py --tasks ioi --n-prompts 40
    uv run python 83_edge_necessity.py --tasks ioi sva --device cpu
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
def ablate_single_edge(model, tokens, correct_id, incorrect_id,
                       up_layer, up_head, down_layer, mean_z):
    """Ablate one edge by replacing upstream head's output with its mean at the downstream layer.

    Returns (clean_ld, ablated_ld).
    """
    W_O = model.W_O[up_layer]

    # Get clean logit diff
    clean_logits, clean_cache = model.run_with_cache(tokens)
    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)

    # Compute the contribution to remove: head output projected through W_O
    up_z_clean = clean_cache[f"blocks.{up_layer}.attn.hook_z"][0, -1, up_head, :]
    up_mean = mean_z[up_layer, up_head].to(up_z_clean.device)
    diff = (up_z_clean - up_mean) @ W_O[up_head]

    def ablate_hook(activation, hook, _diff=diff):
        activation[0, -1, :] -= _diff
        return activation

    ablated_logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{down_layer}.hook_resid_pre", ablate_hook)])
    ablated_ld = logit_diff_from_logits(ablated_logits, correct_id, incorrect_id)

    return clean_ld, ablated_ld


@torch.no_grad()
def run_edge_necessity(model, tasks: list[str],
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

        edge_necessity = {}
        for (up_l, up_h, down_l, down_h) in sorted(all_edges):
            clean_lds = []
            ablated_lds = []

            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                tokens = model.to_tokens(p.text)
                c_ld, a_ld = ablate_single_edge(
                    model, tokens, correct_ids[i], incorrect_ids[i],
                    up_l, up_h, down_l, mean_z)
                clean_lds.append(c_ld)
                ablated_lds.append(a_ld)

            mean_clean = np.mean(clean_lds)
            mean_ablated = np.mean(ablated_lds)
            if abs(mean_clean) < 1e-8:
                drop_frac = 0.0
            else:
                drop_frac = (mean_clean - mean_ablated) / abs(mean_clean)

            necessary = float(drop_frac) > 0.05
            edge = (up_l, up_h, down_l, down_h)
            edge_necessity[edge] = {"drop_frac": float(drop_frac), "necessary": bool(necessary)}

            name = f"L{up_l}H{up_h}->L{down_l}H{down_h}"
            status = "NECESSARY" if necessary else "redundant"
            log(f"    {name}: drop={drop_frac:.4f}  [{status}]")

        n_necessary = sum(1 for d in edge_necessity.values() if d["necessary"])
        frac_necessary = float(n_necessary / max(len(edge_necessity), 1))
        passed = bool(frac_necessary >= 0.5)

        log(f"    necessary={n_necessary}/{len(edge_necessity)} ({frac_necessary:.0%})  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="G2.edge_necessity",
            value=frac_necessary,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_edges": len(all_edges),
                "n_necessary": n_necessary,
                "frac_necessary": frac_necessary,
                "per_edge": {
                    f"L{e[0]}H{e[1]}->L{e[2]}H{e[3]}": d
                    for e, d in edge_necessity.items()
                },
                "passed": passed,
                "threshold_frac": 0.5,
                "threshold_drop": 0.05,
            },
        ))

    return results


def main():
    parser = parse_common_args("G2: Edge Necessity")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("G2: EDGE NECESSITY")
    log("=" * 60)

    out = args.out or "83_edge_necessity.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_edge_necessity(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: frac_necessary={r.value:.2f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
