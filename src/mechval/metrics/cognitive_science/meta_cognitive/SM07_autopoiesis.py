"""Autopoiesis Proxy: Context Self-Maintenance Under Perturbation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-07 — Autopoiesis Proxy
Categories:     wildcard, self_model
Evidence family: causal
Description mode: implementational-functional

Tests whether the circuit exhibits "self-repair" — the ability of
later circuit heads to compensate when earlier representations are
perturbed by noise injection.

Background:
    Maturana & Varela (1980, "Autopoiesis and Cognition: The
    Realization of the Living", Boston Studies in Philosophy of
    Science 42) define an autopoietic system as one that maintains
    its own organization through internal processes. The key property
    is self-maintenance: when perturbed, the system actively restores
    its functional output rather than passively degrading.

    In transformer circuits, self-repair has been empirically observed
    (McGrath et al. 2023, "The Hydra Effect: Emergent Self-Repair in
    Language Model Computations"): ablating one attention head often
    causes other heads to compensate, partially restoring the output.
    This is structurally analogous to autopoietic self-maintenance.

    This metric injects calibrated Gaussian noise into the residual
    stream at an early layer and measures how much the circuit's
    downstream heads compensate, preserving the logit difference
    despite the perturbation.

    Connections:
    - Maturana & Varela (1980) — autopoiesis foundations
    - McGann & De Jaegher (2009) "Self-Other Contingencies", Phenomenology
      and the Cognitive Sciences 8:417-437
    - McGrath et al. (2023) — self-repair in language models

Method:
    1. Run model on clean prompts, record circuit head activations
       at the last token position, and compute clean logit diffs
    2. Inject Gaussian noise into the residual stream at layer 1
       (hook_resid_post), scaled to 10%, 20%, and 50% of the
       residual stream norm at that position
    3. Re-run with noise, compute noisy logit diffs
    4. Self-repair score = 1 - |ld_noisy - ld_clean| / |ld_clean|
       - Score near 1.0: circuit self-repairs (output preserved)
       - Score near 0.0: circuit breaks (output destroyed)
       Clamp to [0, 1].
    5. Per-head repair contribution: for each circuit head, measure
       activation change between clean and noisy runs. Heads whose
       activations shift to compensate are "repair heads".
    6. Aggregate self-repair score at each noise level, averaged
       across prompts.

Pass condition: self_repair_score > 0.5 at 20% noise level.

Usage:
    mechval.run("autopoiesis", tasks=["ioi"], device="cpu")

References:
    Maturana & Varela 1980 "Autopoiesis and Cognition";
    McGann & De Jaegher 2009;
    McGrath et al. 2023 "The Hydra Effect"
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
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Autopoiesis Proxy",
    paper_ref="Maturana & Varela 1980; McGann & De Jaegher 2009",
    paper_cite="Maturana & Varela 1980 (autopoiesis); McGrath et al. 2023 (self-repair)",
    description="Measures circuit self-maintenance under residual stream noise injection",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

NOISE_LEVELS = [0.1, 0.2, 0.5]
PASS_NOISE_LEVEL = 0.2
PASS_THRESHOLD = 0.5
INJECTION_LAYER = 1


@torch.no_grad()
def run_autopoiesis(model, tasks: list[str],
                    n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    injection_layer = min(INJECTION_LAYER, n_layers - 1)

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        heads = sorted(circuit_heads)
        log(f"  {task}: {len(heads)} circuit heads, {len(prompts)} prompts, "
            f"injection at layer {injection_layer}")

        # Collect circuit head activation hook names
        head_hook_names = [f"blocks.{l}.attn.hook_z" for l, _ in heads]
        unique_hook_names = sorted(set(head_hook_names))

        repair_scores_by_level = {level: [] for level in NOISE_LEVELS}
        head_activation_deltas = {f"L{l}H{h}": {level: [] for level in NOISE_LEVELS}
                                  for l, h in heads}

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)

            # Clean run
            clean_logits, clean_cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n in unique_hook_names,
            )
            clean_ld = logit_diff_from_logits(clean_logits, correct_ids[idx], incorrect_ids[idx])

            if abs(clean_ld) < 1e-8:
                continue

            # Store clean head activations at last token
            clean_head_acts = {}
            for l, h in heads:
                key = f"L{l}H{h}"
                act = clean_cache[f"blocks.{l}.attn.hook_z"][0, -1, h]
                clean_head_acts[key] = act.clone()

            # Get residual stream norm at injection layer for calibration
            _, resid_cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n == f"blocks.{injection_layer}.hook_resid_post",
            )
            resid = resid_cache[f"blocks.{injection_layer}.hook_resid_post"]
            resid_norm = resid[0, -1].norm().item()

            # Noisy runs at each noise level
            for level in NOISE_LEVELS:
                noise_std = level * resid_norm

                noise_vector = torch.randn_like(resid[0, -1]) * noise_std

                def noise_hook(value, hook, _noise=noise_vector):
                    value[0, -1] = value[0, -1] + _noise
                    return value

                hook_name = f"blocks.{injection_layer}.hook_resid_post"
                noisy_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, noise_hook)],
                )
                noisy_ld = logit_diff_from_logits(noisy_logits, correct_ids[idx], incorrect_ids[idx])

                raw_repair = 1.0 - abs(noisy_ld - clean_ld) / abs(clean_ld)
                repair_score = max(0.0, min(1.0, raw_repair))
                repair_scores_by_level[level].append(repair_score)

                # Capture noisy head activations via hook-based caching
                noisy_head_acts_capture: dict[str, torch.Tensor] = {}

                def _capture_hook(name_key):
                    def hook_fn(value, hook):
                        noisy_head_acts_capture[name_key] = value.detach()
                        return value
                    return hook_fn

                capture_hooks = [(hook_name, noise_hook)]
                for l, h in heads:
                    hk = f"blocks.{l}.attn.hook_z"
                    if hk not in [ch[0] for ch in capture_hooks]:
                        capture_hooks.append((hk, _capture_hook(hk)))

                model.run_with_hooks(tokens, fwd_hooks=capture_hooks)

                for l, h in heads:
                    key = f"L{l}H{h}"
                    hk = f"blocks.{l}.attn.hook_z"
                    if hk in noisy_head_acts_capture:
                        noisy_act = noisy_head_acts_capture[hk][0, -1, h]
                        delta = (noisy_act - clean_head_acts[key]).norm().item()
                        clean_norm = clean_head_acts[key].norm().item()
                        relative_delta = delta / clean_norm if clean_norm > 1e-8 else 0.0
                        head_activation_deltas[key][level].append(relative_delta)

        # Aggregate results
        mean_repair = {}
        for level in NOISE_LEVELS:
            scores = repair_scores_by_level[level]
            mean_repair[level] = float(np.mean(scores)) if scores else 0.0

        pass_score = mean_repair.get(PASS_NOISE_LEVEL, 0.0)
        passed = pass_score > PASS_THRESHOLD

        # Per-head summary
        head_repair_summary = {}
        for key in head_activation_deltas:
            head_repair_summary[key] = {}
            for level in NOISE_LEVELS:
                deltas = head_activation_deltas[key][level]
                head_repair_summary[key][f"noise_{level}"] = {
                    "mean_relative_delta": float(np.mean(deltas)) if deltas else 0.0,
                    "std_relative_delta": float(np.std(deltas)) if deltas else 0.0,
                }

        # Identify repair heads: those whose activations change most at 20% noise
        # (they are compensating for the perturbation)
        repair_candidates = []
        for key in head_repair_summary:
            delta_20 = head_repair_summary[key].get(f"noise_{PASS_NOISE_LEVEL}", {})
            mean_delta = delta_20.get("mean_relative_delta", 0.0)
            repair_candidates.append((key, mean_delta))
        repair_candidates.sort(key=lambda x: x[1], reverse=True)

        log(f"    self-repair scores: " + ", ".join(
            f"{level*100:.0f}%={mean_repair[level]:.3f}" for level in NOISE_LEVELS))
        log(f"    top repair heads (by activation shift at {PASS_NOISE_LEVEL*100:.0f}% noise):")
        for key, delta in repair_candidates[:5]:
            log(f"      {key}: relative_delta={delta:.4f}")
        log(f"    [{'PASS (self-repairing)' if passed else 'FAIL (fragile)'}]")

        results.append(EvalResult(
            metric_id="SM07.autopoiesis",
            value=pass_score,
            n_samples=len(repair_scores_by_level[PASS_NOISE_LEVEL]),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": len(heads),
                "injection_layer": injection_layer,
                "noise_levels": NOISE_LEVELS,
                "mean_repair_by_level": {str(k): v for k, v in mean_repair.items()},
                "head_repair_summary": head_repair_summary,
                "top_repair_heads": [
                    {"head": k, "mean_relative_delta": float(d)}
                    for k, d in repair_candidates[:10]
                ],
                "passed": passed,
                "threshold": PASS_THRESHOLD,
                "pass_noise_level": PASS_NOISE_LEVEL,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-07: Autopoiesis Proxy")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-07: AUTOPOIESIS PROXY")
    log("=" * 60)

    out = args.out or "SM07_autopoiesis.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_autopoiesis(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
