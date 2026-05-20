"""Cross-Layer Coalition Tracking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         GT7 — Coalition Tracking (Persist / Split / Merge / Vanish)
Categories:     causal, game_theory
Evidence family: causal
Description mode: implementational-functional

Tracks how coalitions of heads evolve across layers — do early-layer
coalitions persist, split, merge, or vanish as information flows deeper?

Background:
    Coalition dynamics in repeated/sequential games track how groups
    form and dissolve over time (Hart & Kurz 1983, "Endogenous
    Formation of Coalitions", Econometrica 51:1047-1064). In multi-layer
    networks, "time" maps to depth: layer 0 → layer 1 → ... → layer L.

    Chowdhury et al. (2025, "Hedonic Neurons", arXiv:2509.23684)
    introduced cross-layer coalition tracking for neural networks,
    classifying transitions between adjacent layers as:
    - PERSIST: coalition exists in both layers (Jaccard > 0.7)
    - SPLIT: one coalition in layer L becomes two+ in layer L+1
    - MERGE: two+ coalitions in layer L become one in layer L+1
    - VANISH: coalition in layer L has no successor in layer L+1
    - EMERGE: coalition in layer L+1 has no predecessor in layer L

    Applied to circuits: this reveals the circuit's "organizational
    dynamics" — whether functional groups are stable across depth or
    constantly reorganize.

    Connections:
    - Hart & Kurz (1983) — endogenous coalition formation
    - Chowdhury et al. (2025) — cross-layer tracking in neural nets
    - Ray & Vohra (2015) "Coalition Formation", Handbook of Game Theory
      with Economic Applications 4:239-326

Method:
    1. At each layer, identify which circuit heads are present
    2. Compute pairwise synergy between heads within each layer
    3. Cluster heads at each layer into coalitions (by synergy sign)
    4. Track coalitions across layers using Jaccard similarity:
       - J(C_l, C_{l+1}) = |C_l ∩ C_{l+1}| / |C_l ∪ C_{l+1}|
       - where membership is compared by functional role, not position
    5. Classify each transition: persist, split, merge, vanish, emerge
    6. Report transition frequencies and stability score

    Since heads at different layers can't be directly compared by
    position, we compare by their OUTPUT similarity: two heads are
    "the same" across layers if their logit contributions (via W_O @ W_U)
    are highly correlated.

Pass condition: persist_ratio > 0.5 (most coalitions are stable).

Usage:
    mechval.run("coalition_tracking", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Cross-Layer Coalition Tracking",
    paper_ref="Hart & Kurz 1983; Chowdhury et al. 2025; Ray & Vohra 2015",
    paper_cite="Chowdhury et al. 2025 (Hedonic Neurons); Hart & Kurz 1983",
    description="Tracks how head coalitions evolve across layers (persist/split/merge/vanish/emerge)",
    category="causal",
    tier="cross_discipline",
    origin="established",
)

PERSIST_THRESHOLD = 0.5
JACCARD_MATCH = 0.5


@torch.no_grad()
def _head_logit_contribution(model, prompts, correct_ids, incorrect_ids,
                             layer: int, head: int) -> np.ndarray:
    """Get mean logit contribution vector for a head across prompts."""
    W_O = model.W_O[layer, head].detach()
    W_U = model.W_U.detach()
    projection = (W_O @ W_U).cpu().numpy()

    contributions = []
    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: f"blocks.{layer}.attn.hook_z" in n)
        z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head].cpu().numpy()
        contrib = z @ projection
        contributions.append(contrib)

    return np.mean(contributions, axis=0) if contributions else projection[0] * 0


@torch.no_grad()
def _coalition_value(model, prompts, correct_ids, incorrect_ids,
                     active: set[tuple[int, int]],
                     all_heads: set[tuple[int, int]],
                     mean_z: torch.Tensor) -> float:
    ablated = all_heads - active
    if not ablated:
        hooks = []
    else:
        hooks = make_ablation_hook(heads_to_layer_dict(ablated), mean_z, "mean")

    lds = []
    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
        lds.append(ld)
    return float(np.mean(lds)) if lds else 0.0


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@torch.no_grad()
def run_coalition_tracking(model, tasks: list[str],
                           n_prompts: int = 20) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads or len(circuit_heads) < 3:
            log(f"  {task}: <3 circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        heads = sorted(circuit_heads)
        n_heads = len(heads)

        heads_by_layer: dict[int, list[tuple[int, int]]] = {}
        for l, h in heads:
            heads_by_layer.setdefault(l, []).append((l, h))

        layers_with_heads = sorted(heads_by_layer.keys())
        if len(layers_with_heads) < 2:
            log(f"  {task}: heads in <2 layers, skipping coalition tracking")
            continue

        log(f"  {task}: {n_heads} heads across {len(layers_with_heads)} layers")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        per_layer_coalitions: dict[int, list[set[tuple[int, int]]]] = {}

        for layer_idx in layers_with_heads:
            layer_heads = heads_by_layer[layer_idx]

            if len(layer_heads) == 1:
                per_layer_coalitions[layer_idx] = [{layer_heads[0]}]
                continue

            n_layer = len(layer_heads)
            synergy = np.zeros((n_layer, n_layer))

            for i in range(n_layer):
                for j in range(i + 1, n_layer):
                    v_i = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        {layer_heads[i]}, circuit_heads, mean_z)
                    v_j = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        {layer_heads[j]}, circuit_heads, mean_z)
                    v_ij = _coalition_value(
                        model, prompts, correct_ids, incorrect_ids,
                        {layer_heads[i], layer_heads[j]}, circuit_heads, mean_z)
                    s = v_ij - v_i - v_j
                    synergy[i, j] = s
                    synergy[j, i] = s

            assigned = [False] * n_layer
            coalitions = []

            for i in range(n_layer):
                if assigned[i]:
                    continue
                coalition = {layer_heads[i]}
                assigned[i] = True
                for j in range(i + 1, n_layer):
                    if assigned[j]:
                        continue
                    if synergy[i, j] > 0:
                        coalition.add(layer_heads[j])
                        assigned[j] = True
                coalitions.append(coalition)

            per_layer_coalitions[layer_idx] = coalitions

        transitions = []
        n_persist = 0
        n_split = 0
        n_merge = 0
        n_vanish = 0
        n_emerge = 0
        total_transitions = 0

        for idx in range(len(layers_with_heads) - 1):
            l_curr = layers_with_heads[idx]
            l_next = layers_with_heads[idx + 1]
            curr_coalitions = per_layer_coalitions[l_curr]
            next_coalitions = per_layer_coalitions[l_next]

            curr_heads_flat = set()
            for c in curr_coalitions:
                curr_heads_flat.update(c)
            next_heads_flat = set()
            for c in next_coalitions:
                next_heads_flat.update(c)

            curr_head_indices = {h[1] for h in curr_heads_flat}
            next_head_indices = {h[1] for h in next_heads_flat}

            for c_curr in curr_coalitions:
                curr_indices = {h[1] for h in c_curr}
                best_jaccard = 0.0
                matches = []

                for c_next in next_coalitions:
                    next_indices = {h[1] for h in c_next}
                    j = _jaccard(curr_indices, next_indices)
                    if j >= JACCARD_MATCH:
                        matches.append(c_next)
                    best_jaccard = max(best_jaccard, j)

                total_transitions += 1
                if len(matches) == 0:
                    n_vanish += 1
                    transitions.append({
                        "from_layer": l_curr, "to_layer": l_next,
                        "type": "vanish",
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(c_curr)],
                    })
                elif len(matches) == 1:
                    n_persist += 1
                    transitions.append({
                        "from_layer": l_curr, "to_layer": l_next,
                        "type": "persist",
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(c_curr)],
                        "successor": [f"L{h[0]}H{h[1]}" for h in sorted(matches[0])],
                    })
                else:
                    n_split += 1
                    transitions.append({
                        "from_layer": l_curr, "to_layer": l_next,
                        "type": "split",
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(c_curr)],
                        "successors": [[f"L{h[0]}H{h[1]}" for h in sorted(m)] for m in matches],
                    })

            for c_next in next_coalitions:
                next_indices = {h[1] for h in c_next}
                has_predecessor = False
                for c_curr in curr_coalitions:
                    curr_indices = {h[1] for h in c_curr}
                    if _jaccard(curr_indices, next_indices) >= JACCARD_MATCH:
                        has_predecessor = True
                        break
                if not has_predecessor:
                    n_emerge += 1
                    total_transitions += 1
                    transitions.append({
                        "from_layer": l_curr, "to_layer": l_next,
                        "type": "emerge",
                        "coalition": [f"L{h[0]}H{h[1]}" for h in sorted(c_next)],
                    })

        persist_ratio = n_persist / total_transitions if total_transitions > 0 else 0.0
        passed = persist_ratio >= PERSIST_THRESHOLD

        layer_summary = {}
        for layer_idx in layers_with_heads:
            coalitions = per_layer_coalitions[layer_idx]
            layer_summary[f"layer_{layer_idx}"] = {
                "n_coalitions": len(coalitions),
                "coalitions": [
                    [f"L{h[0]}H{h[1]}" for h in sorted(c)]
                    for c in coalitions
                ],
            }

        log(f"    Layer coalitions:")
        for l in layers_with_heads:
            cs = per_layer_coalitions[l]
            desc = ", ".join(
                "{" + ",".join(f"L{h[0]}H{h[1]}" for h in sorted(c)) + "}"
                for c in cs
            )
            log(f"      L{l}: {desc}")

        log(f"    Transitions: persist={n_persist} split={n_split} "
            f"merge={n_merge} vanish={n_vanish} emerge={n_emerge}")
        log(f"    persist_ratio={persist_ratio:.3f}  "
            f"[{'PASS (stable)' if passed else 'FAIL (unstable)'}]")

        results.append(EvalResult(
            metric_id="GT7.coalition_tracking",
            value=persist_ratio,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": n_heads,
                "n_layers_with_heads": len(layers_with_heads),
                "layer_summary": layer_summary,
                "transitions": transitions[:20],
                "n_persist": n_persist,
                "n_split": n_split,
                "n_merge": n_merge,
                "n_vanish": n_vanish,
                "n_emerge": n_emerge,
                "total_transitions": total_transitions,
                "persist_ratio": persist_ratio,
                "passed": passed,
                "threshold": PERSIST_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("GT7: Coalition Tracking")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GT7: CROSS-LAYER COALITION TRACKING")
    log("=" * 60)

    out = args.out or "GT7_coalition_tracking.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_coalition_tracking(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
