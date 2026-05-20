"""Ecological Succession in Circuits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EC3 — Ecological Succession
Categories:     cross_discipline, ecology
Evidence family: cross_discipline

Tests how a circuit reorganizes after head ablation -- do surviving
heads compensate by changing their activations? High succession score
indicates active functional reorganization.

Pass condition: succession_score > 0.3

Usage:
    uv run python EC3_succession.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Ecological Succession",
    paper_ref="Clements 1916; Connell & Slatyer 1977",
    paper_cite="Clements 1916, Plant Succession, Carnegie Institution; "
               "Connell & Slatyer 1977, Mechanisms of Succession, American Naturalist 111:1119-1144",
    description="Tests whether surviving circuit heads compensate after single-head ablation",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

SUCCESSION_THRESHOLD = 0.3


@torch.no_grad()
def run_succession(model, tasks: list[str],
                   n_prompts: int = 40) -> list[EvalResult]:
    """Measure activation reorganization in surviving heads after ablation."""
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if len(circuit_heads) < 2:
            log(f"  {task}: need >= 2 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        n_valid = len(prompts)
        head_list = sorted(circuit_heads)
        circuit_layers = sorted({L for L, H in head_list})

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Collect baseline activations
        baseline_acts = np.zeros((len(head_list), n_valid))

        for p_idx in range(n_valid):
            tokens = model.to_tokens(prompts[p_idx].text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "hook_z" in n)
            for h_idx, (L, H) in enumerate(head_list):
                z = cache[f"blocks.{L}.attn.hook_z"]
                baseline_acts[h_idx, p_idx] = z[0, -1, H].norm().item()

        # For each head, ablate it and measure succession in survivors
        per_head_succession = []
        head_details = []

        for abl_idx, (abl_L, abl_H) in enumerate(head_list):
            survivor_changes = []
            ablated_baseline = baseline_acts[abl_idx]

            for p_idx in range(n_valid):
                tokens = model.to_tokens(prompts[p_idx].text)
                captured = {}

                def ablation_hook(z, hook, _H=abl_H):
                    z[0, :, _H, :] = 0.0
                    return z

                def make_capture_hook(layer):
                    def capture_hook(z, hook):
                        captured[layer] = z.detach().clone()
                        return z
                    return capture_hook

                fwd_hooks = [(f"blocks.{abl_L}.attn.hook_z", ablation_hook)]
                for cl in circuit_layers:
                    fwd_hooks.append((f"blocks.{cl}.attn.hook_z", make_capture_hook(cl)))

                model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                abl_base = ablated_baseline[p_idx]
                for h_idx, (L, H) in enumerate(head_list):
                    if h_idx == abl_idx:
                        continue
                    z = captured[L]
                    new_act = z[0, -1, H].norm().item()
                    base = baseline_acts[h_idx, p_idx]
                    change = abs(new_act - base)
                    # Normalize by ablated head's baseline activation
                    if abl_base > 1e-8:
                        survivor_changes.append(change / abl_base)

            succession = float(np.mean(survivor_changes)) if survivor_changes else 0.0
            per_head_succession.append(succession)
            head_details.append({
                "ablated_head": f"L{abl_L}H{abl_H}",
                "succession_score": succession,
                "mean_ablated_activation": float(np.mean(ablated_baseline)),
            })

        succession_score = float(np.mean(per_head_succession))
        max_succession = float(np.max(per_head_succession))
        passed = succession_score > SUCCESSION_THRESHOLD

        log(f"    succession_score={succession_score:.3f}  "
            f"max={max_succession:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EC3.succession",
            value=succession_score,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "succession_score": succession_score,
                "max_succession": max_succession,
                "min_succession": float(np.min(per_head_succession)),
                "passed": passed,
                "threshold": SUCCESSION_THRESHOLD,
                "head_details": sorted(head_details, key=lambda d: -d["succession_score"]),
            },
        ))

    return results


def main():
    parser = parse_common_args("EC3: Ecological Succession")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EC3: ECOLOGICAL SUCCESSION")
    log("=" * 60)

    out = args.out or "EC3_succession.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_succession(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
