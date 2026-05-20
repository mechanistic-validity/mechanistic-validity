"""Error Propagation: Noise Amplification Across Circuit Layers
Injects calibrated Gaussian noise at layer 0 residual stream and
measures signal-to-noise ratio at each subsequent layer's circuit head
activations. Tests whether noise is amplified or dampened.

Pass: snr_final / snr_initial > 0.5 (noise doesn't catastrophically amplify)
Ref: Haykin 1999, Neural Networks: A Comprehensive Foundation, Prentice Hall

Usage:
    uv run python EN4_error_propagation.py --tasks ioi --n-prompts 40
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
    name="Error Propagation (Noise Amplification)",
    paper_ref="Haykin 1999, Prentice Hall",
    paper_cite="Haykin 1999, Neural Networks: A Comprehensive Foundation",
    description="Measures whether noise injected at early layers is amplified or dampened across circuit heads",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

SNR_RATIO_THRESHOLD = 0.5
NOISE_SCALE = 0.1  # Fraction of residual stream std


@torch.no_grad()
def run_error_propagation(model, tasks: list[str],
                          n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        # Determine which layers have circuit heads
        circuit_layers = sorted({L for L, H in circuit_heads})
        heads_by_layer = {}
        for L, H in circuit_heads:
            heads_by_layer.setdefault(L, []).append(H)

        # For each prompt: run clean, run with noise at layer 0, compare activations
        per_layer_snrs_all = {L: [] for L in circuit_layers}

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)

            # Clean run: cache circuit head activations
            hook_names = [f"blocks.{L}.attn.hook_z" for L in circuit_layers]
            hook_names.append("blocks.0.hook_resid_pre")
            _, clean_cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names)

            # Calibrate noise to residual stream scale
            resid_pre = clean_cache["blocks.0.hook_resid_pre"]
            noise_std = float(resid_pre.std()) * NOISE_SCALE
            noise = torch.randn_like(resid_pre) * noise_std

            # Noisy run: inject noise at layer 0 residual stream
            def _noise_hook(resid, hook, _noise=noise):
                return resid + _noise

            noisy_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("blocks.0.hook_resid_pre", _noise_hook)],
                return_type="logits",
            )

            # We need the noisy activations too — run again with cache
            # (run_with_hooks doesn't return cache, so use a capture hook)
            noisy_acts = {}

            def _make_capture_hook(layer):
                def _capture(z, hook):
                    noisy_acts[layer] = z.clone()
                    return z
                return _capture

            fwd_hooks = [("blocks.0.hook_resid_pre", _noise_hook)]
            for L in circuit_layers:
                fwd_hooks.append((f"blocks.{L}.attn.hook_z", _make_capture_hook(L)))

            model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            # Compute SNR at each circuit layer
            for L in circuit_layers:
                clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]
                noisy_z = noisy_acts.get(L)
                if noisy_z is None:
                    continue

                # Extract circuit heads only
                for H in heads_by_layer[L]:
                    signal = clean_z[0, -1, H, :]
                    noise_component = noisy_z[0, -1, H, :] - signal
                    signal_power = float((signal ** 2).mean())
                    noise_power = float((noise_component ** 2).mean())

                    if noise_power > 1e-12:
                        snr = signal_power / noise_power
                    else:
                        snr = float("inf")
                    per_layer_snrs_all[L].append(snr)

        # Aggregate SNR per layer
        per_layer_mean_snr = {}
        for L in circuit_layers:
            if per_layer_snrs_all[L]:
                # Use median to be robust to outliers
                per_layer_mean_snr[L] = float(np.median(per_layer_snrs_all[L]))

        if len(per_layer_mean_snr) < 2:
            log(f"    fewer than 2 circuit layers, computing ratio from available")

        snr_values = [per_layer_mean_snr[L] for L in sorted(per_layer_mean_snr)]
        if not snr_values:
            log(f"    no SNR data, skipping")
            continue

        snr_initial = snr_values[0]
        snr_final = snr_values[-1]

        if snr_initial > 1e-12:
            snr_ratio = snr_final / snr_initial
        else:
            snr_ratio = 0.0

        passed = snr_ratio > SNR_RATIO_THRESHOLD

        log(f"    SNR initial (L{sorted(per_layer_mean_snr)[0]})={snr_initial:.2f}  "
            f"final (L{sorted(per_layer_mean_snr)[-1]})={snr_final:.2f}  "
            f"ratio={snr_ratio:.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EN4.error_propagation",
            value=snr_ratio,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "snr_ratio": snr_ratio,
                "snr_initial": snr_initial,
                "snr_final": snr_final,
                "per_layer_snr": {f"L{L}": v for L, v in per_layer_mean_snr.items()},
                "circuit_layers": circuit_layers,
                "noise_scale": NOISE_SCALE,
                "passed": passed,
                "snr_ratio_threshold": SNR_RATIO_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EN4: Error Propagation")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EN4: ERROR PROPAGATION (NOISE AMPLIFICATION)")
    log("=" * 60)

    out = args.out or "EN4_error_propagation.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_error_propagation(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
