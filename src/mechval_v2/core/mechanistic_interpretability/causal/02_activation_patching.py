"""Activation Patching (Path Patching)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A01 — SCM / Pearl Causal Hierarchy
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity, I2 Sufficiency
Establishes:    Circuit heads are necessary for task performance under do-calculus interventions
Requires:       GPU, model
Doc:            /instruments_v2/causal/a01-scm-pearl
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each component, measures the fraction of the clean->corrupted logit-diff
gap restored by patching that component's activation. Standard faithfulness
numerator.

Uses TransformerLens hooks only — no new dependencies.

Usage:
    uv run python 02_activation_patching.py --tasks ioi sva --n-prompts 40
    uv run python 02_activation_patching.py --device cuda
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def patch_head_effect(model, clean_tokens, corrupt_tokens, correct_id, incorrect_id,
                      layer: int, head: int) -> float:
    """Patch a single head's z from clean into corrupt run. Return normalized effect."""
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: "hook_z" in n)
    corrupt_logits = model(corrupt_tokens)

    clean_ld = logit_diff_from_logits(model(clean_tokens), correct_id, incorrect_id)
    corrupt_ld = logit_diff_from_logits(corrupt_logits, correct_id, incorrect_id)
    gap = clean_ld - corrupt_ld
    if abs(gap) < 1e-8:
        log(f"    [patch_head L{layer}H{head}] zero gap: clean_ld={clean_ld:.4f} corrupt_ld={corrupt_ld:.4f} correct={correct_id} incorrect={incorrect_id}")
        return 0.0

    def patch_hook(z, hook, _L=layer, _H=head):
        clean_z = clean_cache[hook.name]
        seq_len = min(z.shape[1], clean_z.shape[1])
        z[0, :seq_len, _H, :] = clean_z[0, :seq_len, _H, :]
        return z

    hook_name = f"blocks.{layer}.attn.hook_z"
    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, patch_hook)])
    patched_ld = logit_diff_from_logits(patched_logits, correct_id, incorrect_id)

    return (patched_ld - corrupt_ld) / gap


@torch.no_grad()
def generate_corrupt_prompt(prompt, task: str, all_prompts: list, idx: int, rng) -> str:
    """Generate a corrupted version by swapping with another prompt."""
    other_idx = rng.choice([j for j in range(len(all_prompts)) if j != idx])
    return all_prompts[other_idx].text


@torch.no_grad()
def run_activation_patching(model, tasks: list[str], n_prompts: int = 40,
                            n_random_baselines: int = 100) -> list[EvalResult]:
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

        if prompts:
            p0 = prompts[0]
            t0 = model.to_tokens(p0.text)
            ld0 = logit_diff_from_logits(model(t0), correct_ids[0], incorrect_ids[0])
            log(f"    baseline logit_diff={ld0:.4f} correct={correct_ids[0]} incorrect={incorrect_ids[0]} text={p0.text[:80]!r}")

        head_effects = np.zeros((n_layers, n_heads))
        n_valid = 0

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            clean_tokens = model.to_tokens(p.text)
            other_idx = rng.choice([j for j in range(len(prompts)) if j != i])
            corrupt_tokens = model.to_tokens(prompts[other_idx].text)

            for L, H in circuit_heads:
                eff = patch_head_effect(model, clean_tokens, corrupt_tokens,
                                        correct_ids[i], incorrect_ids[i], L, H)
                head_effects[L, H] += eff
            n_valid += 1

        if n_valid > 0:
            head_effects /= n_valid

        circuit_score = sum(head_effects[L, H] for L, H in circuit_heads)

        random_scores = []
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
        k = len(circuit_heads)
        for _ in range(min(n_random_baselines, 1000)):
            rand_heads = [all_heads[j] for j in rng.choice(len(all_heads), size=k, replace=False)]
            rand_score = sum(head_effects[L, H] for L, H in rand_heads)
            random_scores.append(rand_score)

        baseline_random = float(np.mean(random_scores))
        baseline_std = float(np.std(random_scores))

        per_head = {f"L{L}H{H}": float(head_effects[L, H]) for L, H in circuit_heads}
        top5_non_circuit = []
        flat = [(float(head_effects[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)
                if (L, H) not in circuit_heads]
        flat.sort(reverse=True)
        for val, L, H in flat[:5]:
            top5_non_circuit.append({"head": f"L{L}H{H}", "effect": val})

        log(f"    circuit_sum={circuit_score:.3f}  random_mean={baseline_random:.3f}+/-{baseline_std:.3f}")

        results.append(EvalResult(
            metric_id="C2.activation_patching",
            value=circuit_score,
            baseline_random=baseline_random,
            n_samples=n_valid,
            metadata={
                "task": task,
                "per_head_effects": per_head,
                "top5_non_circuit": top5_non_circuit,
                "random_std": baseline_std,
                "n_circuit_heads": k,
                "circuit_heads": sorted(circuit_heads),
            },
        ))

    return results


def main():
    parser = parse_common_args("C2: Activation Patching")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C2: ACTIVATION PATCHING")
    log("=" * 60)

    results = run_activation_patching(model, tasks, args.n_prompts, args.n_random_baselines)

    out = args.out or "02_activation_patching.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: sum={r.value:.3f}  vs random={r.baseline_random:.3f}")


if __name__ == "__main__":
    main()
