"""Composition Test (Algorithmic Sufficiency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A05 — MDC/Glennan Mechanisms
Categories:     causal
Validity layer: Internal
Criteria:       A2 Composition Test (proposed)
Establishes:    Whether executing the claimed procedure reproduces behavior
Requires:       GPU or CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complement ablation at the PATHWAY level:

1. Ablate all non-circuit heads -> measure logit diff recovery
   (basic faithfulness via compute_faithfulness).
2. For each pathway in isolation (e.g., only detector->integrator->executor):
   a. Keep only the heads in that pathway, ablate everything else.
   b. Measure logit diff recovery.
3. Report: full_circuit_faithfulness, per_pathway_faithfulness,
   max_single_pathway_faithfulness.
4. Pass: full_circuit > 0.30 OR max_single_pathway > 0.20.

Usage:
    uv run python 78_composition_test.py --tasks ioi sva
    uv run python 78_composition_test.py --device cpu --n-prompts 60
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    compute_faithfulness,
    generate_prompts,
    get_circuit_info,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def compute_pathway_faithfulness(model, prompts, correct_ids, incorrect_ids,
                                 pathway_heads: set[tuple[int, int]],
                                 mean_z: torch.Tensor) -> float:
    """Faithfulness when keeping ONLY pathway_heads, ablating everything else."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_pathway = {(L, H) for L in range(n_layers) for H in range(n_heads)} - pathway_heads
    non_pathway_by_layer = heads_to_layer_dict(non_pathway)
    hooks = make_ablation_hook(non_pathway_by_layer, mean_z, "mean")

    faith_num, faith_den = 0.0, 0.0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])
        faith_num += ablated_ld
        faith_den += clean_ld

    if abs(faith_den) < 1e-8:
        return 0.0
    return faith_num / faith_den


def extract_pathway_head_sets(circuit: dict) -> dict[str, set[tuple[int, int]]]:
    """Extract per-pathway head sets from circuit definition.

    Each pathway is a (sender_role, receiver_role) edge. We also build
    maximal chains and return their head unions.

    Returns dict mapping pathway_name -> set of (layer, head).
    """
    roles = circuit["roles"]
    pathways = circuit["pathways"]

    # Individual edges as pathways
    result = {}
    for sender_role, receiver_role in pathways:
        name = f"{sender_role}->{receiver_role}"
        heads = set(roles.get(sender_role, []))
        heads.update(roles.get(receiver_role, []))
        result[name] = heads

    # Build maximal chains through adjacency
    adj = {}
    in_degree = {}
    for sender, receiver in pathways:
        adj.setdefault(sender, set()).add(receiver)
        in_degree.setdefault(receiver, 0)
        in_degree[receiver] = in_degree.get(receiver, 0) + 1
        if sender not in in_degree:
            in_degree[sender] = 0

    sources = [r for r, deg in in_degree.items() if deg == 0]

    chains = []

    def dfs(role, current_roles):
        current_roles.append(role)
        next_roles = adj.get(role, set())
        if not next_roles:
            chains.append(list(current_roles))
        else:
            for nr in next_roles:
                dfs(nr, current_roles)
        current_roles.pop()

    for src in sources:
        dfs(src, [])

    for chain_roles in chains:
        if len(chain_roles) > 2:
            name = "->".join(chain_roles)
            heads = set()
            for r in chain_roles:
                heads.update(roles.get(r, []))
            result[name] = heads

    return result


def run_composition_test(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit, all_heads, _ = get_circuit_info(task)
        if circuit is None or not all_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(all_heads)} heads, {len(prompts)} prompts)...")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        # Full circuit faithfulness
        full_faith = compute_faithfulness(
            model, prompts, correct_ids, incorrect_ids, all_heads, mean_z,
        )
        log(f"    full_circuit_faithfulness={full_faith:.3f}")

        # Per-pathway faithfulness
        pathway_head_sets = extract_pathway_head_sets(circuit)
        pathway_scores = {}
        for name, heads in pathway_head_sets.items():
            score = compute_pathway_faithfulness(
                model, prompts, correct_ids, incorrect_ids, heads, mean_z,
            )
            pathway_scores[name] = score
            log(f"    {name}: {score:.3f} ({len(heads)} heads)")

        max_pathway = max(pathway_scores.values()) if pathway_scores else 0.0

        passed = full_faith > 0.30 or max_pathway > 0.20

        results.append(EvalResult(
            metric_id="A2.composition_test",
            value=full_faith,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "full_circuit_faithfulness": full_faith,
                "per_pathway_faithfulness": pathway_scores,
                "max_single_pathway_faithfulness": max_pathway,
                "n_circuit_heads": len(all_heads),
                "n_pathways": len(pathway_head_sets),
                "passed": passed,
                "threshold_full": 0.30,
                "threshold_pathway": 0.20,
            },
        ))

    return results


def main():
    parser = parse_common_args("A2: Composition Test")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("A2: COMPOSITION TEST (Algorithmic Sufficiency)")
    log("=" * 60)

    results = run_composition_test(model, tasks, args.n_prompts)

    out = args.out or "78_composition_test.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: full={r.value:.3f}  "
            f"max_pathway={r.metadata['max_single_pathway_faithfulness']:.3f}  [{p}]")


if __name__ == "__main__":
    main()
