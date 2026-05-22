"""Weber-Fechner / JND — Just-Noticeable Difference and Weber's Law
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX11 — Weber-Fechner / JND
Categories:     behavioral, psychophysics
Evidence family: behavioral
Validity layer: Construct

Tests whether circuit heads follow Weber's law: the just-noticeable
difference (JND) in output is proportional to the stimulus intensity,
yielding a constant Weber fraction.

Background:
    Weber (1834) observed that the smallest detectable change in a
    stimulus is a constant fraction of the stimulus magnitude.
    Fechner (1860) formalized this as a logarithmic relationship
    between physical intensity and perceived magnitude. Gescheider
    (1997) provides the modern psychophysics treatment.

    Applied to circuits: we progressively scale each circuit head's
    output and find the JND — the smallest scaling change that
    produces a detectable output change (>5% drop in logit-diff).
    Weber's law predicts that JND/baseline should be constant across
    different baseline levels.

Method:
    1. For each circuit head, scale its output by factors
       [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0].
    2. Find the JND: smallest scale change from 1.0 that produces a
       detectable output change (logit-diff drops by > 5% of baseline).
    3. Weber's law predicts JND/stimulus is constant (Weber fraction).
    4. Test at different baseline levels: scale all circuit heads by
       0.5, then find per-head JND from that reduced baseline.
    5. Weber fraction = JND / baseline_scale.
    6. Weber consistency = 1 - std(weber_fractions) / mean(weber_fractions).
    7. Pass: JND exists for all heads (all heads contribute detectably).

Refs: Weber 1834; Fechner 1860; Gescheider 1997

Usage:
    uv run python EX11_weber_fechner.py --tasks ioi --n-prompts 40
    uv run python EX11_weber_fechner.py --device cpu
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
    name="Weber-Fechner / JND (Just-Noticeable Difference)",
    paper_ref="Weber 1834; Fechner 1860; Gescheider 1997",
    paper_cite="Weber 1834, De pulsu resorptione auditu et tactu; Fechner 1860, Elemente der Psychophysik; Gescheider 1997, Psychophysics: The Fundamentals (3rd ed.)",
    description="Tests whether circuit heads follow Weber's law by measuring just-noticeable differences at different stimulus intensities",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

SCALE_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
JND_DETECTION_THRESHOLD = 0.05  # 5% drop from baseline logit-diff
BASELINE_SCALES = [1.0, 0.5]  # Test Weber fraction at two baselines


def make_scale_hook(layer: int, head: int,
                    scale: float) -> tuple[str, callable]:
    """Create a hook that scales a single head's output by `scale`."""
    def _hook(z, hook, _H=head, _s=scale):
        z[0, :, _H, :] *= _s
        return z
    return (f"blocks.{layer}.attn.hook_z", _hook)


def make_global_scale_hooks(heads: set[tuple[int, int]],
                            scale: float) -> list[tuple[str, callable]]:
    """Create hooks that scale all specified heads by `scale`."""
    by_layer = heads_to_layer_dict(heads)
    hooks = []
    for layer, head_list in by_layer.items():
        def _hook(z, hook, _heads=head_list, _s=scale):
            for H in _heads:
                z[0, :, H, :] *= _s
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", _hook))
    return hooks


@torch.no_grad()
def find_head_jnd(
    model, prompts, correct_ids, incorrect_ids,
    layer: int, head: int,
    baseline_ld: float,
    global_hooks: list[tuple[str, callable]] | None = None,
) -> tuple[float | None, list[float]]:
    """Find the JND for a single head by sweeping scale factors.

    Returns (jnd_scale_delta, per_scale_lds) where jnd_scale_delta is the
    smallest change from 1.0 that causes > JND_DETECTION_THRESHOLD drop,
    or None if no detectable change.
    """
    n_valid = min(len(prompts), len(correct_ids))
    per_scale_lds = []

    for scale in SCALE_FACTORS:
        scale_hooks = [make_scale_hook(layer, head, scale)]
        if global_hooks:
            all_hooks = global_hooks + scale_hooks
        else:
            all_hooks = scale_hooks

        lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            lds.append(ld)

        mean_ld = float(np.mean(lds))
        per_scale_lds.append(mean_ld)

    # Find JND: iterate from scale=1.0 downward (reversed SCALE_FACTORS)
    # SCALE_FACTORS is ascending, so index -1 is 1.0 (baseline)
    ref_ld = per_scale_lds[-1]  # logit-diff at scale=1.0
    if abs(baseline_ld) < 1e-8:
        return None, per_scale_lds

    jnd = None
    # Walk from scale closest to 1.0 toward lower scales
    for idx in range(len(SCALE_FACTORS) - 2, -1, -1):
        scale = SCALE_FACTORS[idx]
        drop_frac = (ref_ld - per_scale_lds[idx]) / abs(baseline_ld)
        if drop_frac > JND_DETECTION_THRESHOLD:
            jnd = 1.0 - scale  # How far from 1.0 we had to go
            break

    return jnd, per_scale_lds


@torch.no_grad()
def run_weber_fechner(
    model, tasks: list[str], n_prompts: int = 40,
) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no valid token pairs, skipping")
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        n_valid = min(len(prompts), len(correct_ids))

        # Compute baseline logit-diff (no ablation)
        baseline_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            baseline_lds.append(ld)
        mean_baseline_ld = float(np.mean(baseline_lds))

        if abs(mean_baseline_ld) < 1e-8:
            log(f"  {task}: near-zero baseline logit-diff, skipping")
            continue

        # Phase 1: Per-head JND at full baseline (scale=1.0 for other heads)
        per_head_jnd_full: dict[tuple[int, int], float | None] = {}
        per_head_curves: dict[tuple[int, int], list[float]] = {}

        for L, H in sorted(circuit_heads):
            jnd, curve = find_head_jnd(
                model, prompts, correct_ids, incorrect_ids,
                L, H, mean_baseline_ld, global_hooks=None,
            )
            per_head_jnd_full[(L, H)] = jnd
            per_head_curves[(L, H)] = curve

        heads_with_jnd = sum(1 for v in per_head_jnd_full.values() if v is not None)
        all_detectable = heads_with_jnd == len(circuit_heads)

        log(f"    full baseline: {heads_with_jnd}/{len(circuit_heads)} heads have detectable JND")

        # Phase 2: Per-head JND at reduced baseline (scale all circuit heads by 0.5)
        reduced_baseline_scale = 0.5
        global_hooks_reduced = make_global_scale_hooks(circuit_heads, reduced_baseline_scale)

        # Compute reduced-baseline logit-diff
        reduced_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model.run_with_hooks(tokens, fwd_hooks=global_hooks_reduced)
            ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
            reduced_lds.append(ld)
        mean_reduced_ld = float(np.mean(reduced_lds))

        per_head_jnd_reduced: dict[tuple[int, int], float | None] = {}

        for L, H in sorted(circuit_heads):
            jnd, _ = find_head_jnd(
                model, prompts, correct_ids, incorrect_ids,
                L, H, mean_reduced_ld, global_hooks=global_hooks_reduced,
            )
            per_head_jnd_reduced[(L, H)] = jnd

        # Phase 3: Compute Weber fractions
        weber_fractions_full = []
        weber_fractions_reduced = []
        per_head_weber = {}

        for h in sorted(circuit_heads):
            jnd_full = per_head_jnd_full[h]
            jnd_reduced = per_head_jnd_reduced[h]
            wf_full = jnd_full / 1.0 if jnd_full is not None else None
            wf_reduced = (jnd_reduced / reduced_baseline_scale
                          if jnd_reduced is not None else None)

            per_head_weber[h] = {
                "jnd_full": jnd_full,
                "jnd_reduced": jnd_reduced,
                "weber_fraction_full": wf_full,
                "weber_fraction_reduced": wf_reduced,
            }

            if wf_full is not None:
                weber_fractions_full.append(wf_full)
            if wf_reduced is not None:
                weber_fractions_reduced.append(wf_reduced)

        # Weber consistency across all measurements
        all_weber = weber_fractions_full + weber_fractions_reduced
        if len(all_weber) >= 2 and np.mean(all_weber) > 1e-8:
            weber_consistency = 1.0 - float(np.std(all_weber) / np.mean(all_weber))
        elif len(all_weber) >= 1:
            weber_consistency = 1.0  # No variance if only one measurement
        else:
            weber_consistency = 0.0

        passed = all_detectable

        log(f"    weber_fractions_full: {[f'{v:.3f}' for v in weber_fractions_full[:5]]}")
        log(f"    weber_fractions_reduced: {[f'{v:.3f}' for v in weber_fractions_reduced[:5]]}")
        log(f"    weber_consistency={weber_consistency:.4f}  "
            f"all_detectable={all_detectable}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX11.weber_fechner",
            value=weber_consistency,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "weber_consistency": weber_consistency,
                "all_heads_detectable": all_detectable,
                "n_detectable_heads": heads_with_jnd,
                "n_circuit_heads": len(circuit_heads),
                "mean_baseline_ld": mean_baseline_ld,
                "mean_reduced_ld": mean_reduced_ld,
                "reduced_baseline_scale": reduced_baseline_scale,
                "scale_factors": SCALE_FACTORS,
                "jnd_detection_threshold": JND_DETECTION_THRESHOLD,
                "per_head_weber": {
                    f"L{L}H{H}": info for (L, H), info in per_head_weber.items()
                },
                "per_head_curves_full": {
                    f"L{L}H{H}": curve
                    for (L, H), curve in per_head_curves.items()
                },
                "mean_weber_fraction_full": (
                    float(np.mean(weber_fractions_full))
                    if weber_fractions_full else None
                ),
                "mean_weber_fraction_reduced": (
                    float(np.mean(weber_fractions_reduced))
                    if weber_fractions_reduced else None
                ),
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX11: Weber-Fechner / JND")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX11: WEBER-FECHNER / JND (JUST-NOTICEABLE DIFFERENCE)")
    log("=" * 60)

    out = args.out or "EX11_weber_fechner.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_weber_fechner(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
