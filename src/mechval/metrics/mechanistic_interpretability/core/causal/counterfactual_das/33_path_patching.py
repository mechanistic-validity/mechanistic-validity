"""Path Patching (Metric #22)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A02 — Counterfactual DAS/IIA
Categories:     causal, representational
Validity layer: Internal + Representational
Criteria:       I2 Sufficiency
Establishes:    Individual circuit edges carry causal information between head pairs
Requires:       GPU, model
Doc:            /instruments_v2/causal/a02-counterfactual-das
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of patching a head's entire output, patch only the path from
head A to head B. For each pair of circuit heads (A, B) where A is in
an earlier layer: corrupt head A's output, but only measure the effect
on head B's input (not the full model output). This isolates the A->B
edge contribution.

Reports per-edge path-patching effects and identifies the strongest edges.

Usage:
    uv run python 33_path_patching.py --tasks ioi sva --n-prompts 40
    uv run python 33_path_patching.py --device cuda
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def path_patch_edge(model, clean_tokens, corrupt_tokens,
                    correct_id: int, incorrect_id: int,
                    src_layer: int, src_head: int,
                    dst_layer: int, dst_head: int,
                    mean_z: torch.Tensor) -> float:
    """Measure the path-specific effect from src head to dst head.

    Procedure:
    1. Run clean forward, cache all hook_z values.
    2. Run a modified forward where:
       - src head's output is replaced with corrupt activation
       - BUT we only measure the effect at dst head (by freezing all
         other paths using clean cache).
    3. Return normalized effect on logit diff.
    """
    n_layers = model.cfg.n_layers
    clean_logits = model(clean_tokens)
    clean_ld = logit_diff_from_logits(clean_logits, correct_id, incorrect_id)

    _, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=lambda n: "hook_z" in n)
    _, corrupt_cache = model.run_with_cache(
        corrupt_tokens, names_filter=lambda n: "hook_z" in n)

    # Path patching: corrupt src head, freeze everything else to clean,
    # measure effect at output through dst head only.
    # Step 1: Get clean residual stream at dst_layer input but with
    #         src head corrupted.
    src_hook = f"blocks.{src_layer}.attn.hook_z"
    dst_hook = f"blocks.{dst_layer}.attn.hook_z"

    def corrupt_src_hook(z, hook, _src_head=src_head):
        corrupt_z = corrupt_cache[hook.name]
        seq_len = min(z.shape[1], corrupt_z.shape[1])
        z[0, :seq_len, _src_head, :] = corrupt_z[0, :seq_len, _src_head, :]
        return z

    def freeze_non_dst_hook(z, hook, _dst_head=dst_head, _layer=dst_layer):
        # Freeze all heads in this layer to clean values, except dst_head
        clean_z = clean_cache[hook.name]
        seq_len = min(z.shape[1], clean_z.shape[1])
        for h in range(z.shape[2]):
            if h != _dst_head:
                z[0, :seq_len, h, :] = clean_z[0, :seq_len, h, :]
        return z

    def freeze_other_layers_hook(z, hook):
        # For layers between src and dst (exclusive) and after dst,
        # freeze to clean values
        clean_z = clean_cache[hook.name]
        seq_len = min(z.shape[1], clean_z.shape[1])
        z[0, :seq_len, :, :] = clean_z[0, :seq_len, :, :]
        return z

    hooks = [(src_hook, corrupt_src_hook)]

    # Freeze all intermediate layers (between src and dst) to clean
    for layer in range(src_layer + 1, n_layers):
        if layer == dst_layer:
            hooks.append((dst_hook, freeze_non_dst_hook))
        else:
            hook_name = f"blocks.{layer}.attn.hook_z"
            hooks.append((hook_name, freeze_other_layers_hook))

    patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    patched_ld = logit_diff_from_logits(patched_logits, correct_id, incorrect_id)

    # Normalized effect: how much of the clean LD is lost through this path
    if abs(clean_ld) < 1e-8:
        return 0.0
    return (clean_ld - patched_ld) / clean_ld


@torch.no_grad()
def run_path_patching(model, tasks: list[str], n_prompts: int = 40,
                      n_random_baselines: int = 50) -> list[EvalResult]:
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []
    rng = np.random.RandomState(42)

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

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        # Identify all directed edges A->B where A.layer < B.layer
        sorted_heads = sorted(circuit_heads)
        edges = []
        for i, (la, ha) in enumerate(sorted_heads):
            for lb, hb in sorted_heads[i + 1:]:
                if la < lb:
                    edges.append((la, ha, lb, hb))

        if not edges:
            log(f"    no multi-layer edges, skipping")
            continue

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(50, len(prompts)))

        # Compute path patching effect for each edge
        edge_effects = {}
        for src_l, src_h, dst_l, dst_h in edges:
            effects = []
            for i, p in enumerate(prompts):
                if i >= len(correct_ids):
                    break
                clean_tokens = model.to_tokens(p.text)
                other_idx = rng.choice([j for j in range(len(prompts)) if j != i])
                corrupt_tokens = model.to_tokens(prompts[other_idx].text)

                eff = path_patch_edge(model, clean_tokens, corrupt_tokens,
                                      correct_ids[i], incorrect_ids[i],
                                      src_l, src_h, dst_l, dst_h, mean_z)
                effects.append(eff)

            mean_effect = float(np.mean(effects)) if effects else 0.0
            edge_key = f"L{src_l}H{src_h}->L{dst_l}H{dst_h}"
            edge_effects[edge_key] = mean_effect

        # Sort by absolute effect
        sorted_edges = sorted(edge_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        top_edges = sorted_edges[:10]

        total_effect = sum(abs(v) for v in edge_effects.values())
        mean_abs_effect = total_effect / max(len(edge_effects), 1)

        # Random baseline: random pairs of heads across layers
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
        random_effects = []
        for _ in range(min(n_random_baselines, 100)):
            rand_pair = rng.choice(len(all_heads), size=2, replace=False)
            ra_l, ra_h = all_heads[rand_pair[0]]
            rb_l, rb_h = all_heads[rand_pair[1]]
            if ra_l >= rb_l:
                ra_l, ra_h, rb_l, rb_h = rb_l, rb_h, ra_l, ra_h
            if ra_l >= rb_l:
                continue

            rand_effs = []
            for i, p in enumerate(prompts[:5]):
                if i >= len(correct_ids):
                    break
                clean_tokens = model.to_tokens(p.text)
                other_idx = rng.choice([j for j in range(len(prompts)) if j != i])
                corrupt_tokens = model.to_tokens(prompts[other_idx].text)
                eff = path_patch_edge(model, clean_tokens, corrupt_tokens,
                                      correct_ids[i], incorrect_ids[i],
                                      ra_l, ra_h, rb_l, rb_h, mean_z)
                rand_effs.append(eff)
            if rand_effs:
                random_effects.append(float(np.mean(rand_effs)))

        baseline_random = float(np.mean([abs(e) for e in random_effects])) if random_effects else 0.0

        log(f"    {len(edges)} edges, mean|effect|={mean_abs_effect:.4f}, "
            f"top edge={top_edges[0][0]}={top_edges[0][1]:.4f}" if top_edges else
            f"    {len(edges)} edges")

        results.append(EvalResult(
            metric_id="C33.path_patching",
            value=mean_abs_effect,
            baseline_random=baseline_random,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_edges": len(edges),
                "edge_effects": edge_effects,
                "top_edges": [{"edge": e, "effect": v} for e, v in top_edges],
                "total_abs_effect": total_effect,
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C33: Path Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C33: PATH PATCHING (Metric #22)")
    log("=" * 60)

    results = run_path_patching(model, tasks, args.n_prompts, args.n_random_baselines)

    out = args.out or "33_path_patching.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: mean|path_effect|={r.value:.4f}  vs random={r.baseline_random:.4f}")


if __name__ == "__main__":
    main()
