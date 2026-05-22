"""Rate-Distortion Analysis of Circuit Activations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         IT2 — Rate-Distortion
Categories:     extended, information_theory
Tier:           extended
Origin:         established

Tests the minimum bits needed to describe circuit activations while
preserving output quality. Progressively quantizes circuit head
activations (32 -> 8, 4, 2, 1 bit) and measures logit diff retention.

Pass: maintains >80% LD at 4-bit quantization
Ref: Shannon 1959 "Coding Theorems for a Discrete Source with a
     Fidelity Criterion", IRE Nat. Conv. Record 7:142-163

Usage:
    uv run python IT2_rate_distortion.py --tasks ioi sva --n-prompts 40
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
    name="Rate-Distortion",
    paper_ref="Shannon 1959",
    paper_cite="Shannon 1959, Coding Theorems for a Discrete Source with a Fidelity Criterion, IRE Nat. Conv. Record 7:142-163",
    description="Tests circuit compression limit via progressive quantization of head activations",
    category="extended",
    tier="extended",
    origin="established",
)

BIT_LEVELS = [8, 4, 2, 1]
PASS_BITS = 4
RETENTION_THRESHOLD = 0.8


@torch.no_grad()
def run_rate_distortion(model, tasks: list[str],
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

        n_valid = min(len(prompts), len(correct_ids))
        head_list = sorted(circuit_heads)

        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        heads_by_layer: dict[int, list[int]] = {}
        for L, H in head_list:
            heads_by_layer.setdefault(L, []).append(H)

        # Compute baseline (clean) logit diff
        baseline_lds = []
        for i in range(n_valid):
            tokens = model.to_tokens(prompts[i].text)
            logits = model(tokens)
            baseline_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))
        baseline_ld = float(np.mean(baseline_lds))

        if abs(baseline_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        # For each bit level, quantize circuit head activations and measure LD
        rd_curve = {}

        for bits in BIT_LEVELS:
            n_levels = 2 ** bits

            def _make_quantize_hook(layer, head_list_for_layer, _n_levels=n_levels):
                def _hook(z, hook):
                    for H in head_list_for_layer:
                        vals = z[0, :, H, :]
                        v_min = vals.min()
                        v_max = vals.max()
                        if v_max - v_min < 1e-10:
                            continue
                        # Uniform quantization
                        normalized = (vals - v_min) / (v_max - v_min)
                        quantized = torch.round(normalized * (_n_levels - 1)) / (_n_levels - 1)
                        z[0, :, H, :] = quantized * (v_max - v_min) + v_min
                    return z
                return _hook

            hooks = []
            for layer, hlist in heads_by_layer.items():
                hooks.append((f"blocks.{layer}.attn.hook_z",
                              _make_quantize_hook(layer, hlist)))

            quant_lds = []
            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                quant_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

            mean_quant_ld = float(np.mean(quant_lds))
            retention = mean_quant_ld / baseline_ld
            rd_curve[bits] = {
                "bits": bits,
                "mean_ld": mean_quant_ld,
                "retention": float(retention),
            }
            log(f"    {bits}-bit: LD={mean_quant_ld:.4f}, retention={retention:.3f}")

        retention_at_pass = rd_curve[PASS_BITS]["retention"]
        passed = retention_at_pass > RETENTION_THRESHOLD

        log(f"    retention@{PASS_BITS}bit={retention_at_pass:.3f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="IT2.rate_distortion",
            value=retention_at_pass,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "baseline_ld": baseline_ld,
                "rd_curve": rd_curve,
                "retention_at_pass_bits": retention_at_pass,
                "pass_bits": PASS_BITS,
                "threshold": RETENTION_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("IT2: Rate-Distortion")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("IT2: RATE-DISTORTION")
    log("=" * 60)

    out = args.out or "IT2_rate_distortion.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_rate_distortion(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
