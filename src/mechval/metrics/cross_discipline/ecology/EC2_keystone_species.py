"""Keystone Species Detection in Circuits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EC2 — Keystone Species
Categories:     cross_discipline, ecology
Evidence family: cross_discipline

Tests whether removing a single circuit head causes disproportionate
cascade effects on other heads' activations, analogous to keystone
species whose removal reshapes an ecosystem.

Pass condition: keystone_ratio > 2.0

Usage:
    uv run python EC2_keystone_species.py --tasks ioi sva --n-prompts 40
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
    name="Keystone Species",
    paper_ref="Paine 1969, American Naturalist 103:91-93",
    paper_cite="Paine 1969, A Note on Trophic Complexity and Community Stability",
    description="Tests whether single-head ablation causes disproportionate cascade effects on other heads",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

KEYSTONE_RATIO_THRESHOLD = 2.0


@torch.no_grad()
def run_keystone_species(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    """Measure cascade effects of single-head ablation on other circuit heads."""
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

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Collect baseline activations for all circuit heads
        # baseline_acts[h_idx, prompt_idx] = activation norm
        baseline_acts = np.zeros((len(head_list), n_valid))

        for p_idx in range(n_valid):
            tokens = model.to_tokens(prompts[p_idx].text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "hook_z" in n)
            for h_idx, (L, H) in enumerate(head_list):
                z = cache[f"blocks.{L}.attn.hook_z"]
                baseline_acts[h_idx, p_idx] = z[0, -1, H].norm().item()

        # For each head, ablate it and measure cascade on all others
        # cascade_effects[ablated_idx] = mean activation change across other heads
        cascade_effects = np.zeros(len(head_list))
        head_details = []

        # Layers that contain circuit heads (need collection hooks)
        circuit_layers = sorted({L for L, H in head_list})

        for abl_idx, (abl_L, abl_H) in enumerate(head_list):
            other_changes = []

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

                for h_idx, (L, H) in enumerate(head_list):
                    if h_idx == abl_idx:
                        continue
                    z = captured[L]
                    new_act = z[0, -1, H].norm().item()
                    base = baseline_acts[h_idx, p_idx]
                    if base > 1e-8:
                        other_changes.append(abs(new_act - base) / base)
                    else:
                        other_changes.append(abs(new_act - base))

            cascade_effects[abl_idx] = float(np.mean(other_changes)) if other_changes else 0.0
            head_details.append({
                "head": f"L{abl_L}H{abl_H}",
                "cascade_effect": cascade_effects[abl_idx],
            })

        max_cascade = float(np.max(cascade_effects))
        mean_cascade = float(np.mean(cascade_effects))
        keystone_ratio = max_cascade / mean_cascade if mean_cascade > 1e-8 else 0.0
        keystone_idx = int(np.argmax(cascade_effects))

        passed = keystone_ratio > KEYSTONE_RATIO_THRESHOLD

        log(f"    keystone_head=L{head_list[keystone_idx][0]}H{head_list[keystone_idx][1]}  "
            f"ratio={keystone_ratio:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EC2.keystone_species",
            value=keystone_ratio,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "keystone_ratio": keystone_ratio,
                "max_cascade": max_cascade,
                "mean_cascade": mean_cascade,
                "keystone_head": f"L{head_list[keystone_idx][0]}H{head_list[keystone_idx][1]}",
                "passed": passed,
                "threshold": KEYSTONE_RATIO_THRESHOLD,
                "head_details": sorted(head_details, key=lambda d: -d["cascade_effect"]),
            },
        ))

    return results


def main():
    parser = parse_common_args("EC2: Keystone Species")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EC2: KEYSTONE SPECIES")
    log("=" * 60)

    out = args.out or "EC2_keystone_species.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_keystone_species(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
