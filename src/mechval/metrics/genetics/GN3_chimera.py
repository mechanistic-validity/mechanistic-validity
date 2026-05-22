"""Chimera: QK/OV Cross-Layer Transplant
Tests whether mixing the QK circuit (attention pattern) from one
circuit head with the OV circuit (value processing) from another
preserves task performance. Measures functional modularity of the
QK and OV sub-circuits.

Pass: chimera_preservation > 0.3
Ref: McLaren & Wilmut 2003, Reproduction 126:831-838

Usage:
    uv run python GN3_chimera.py --tasks ioi --n-prompts 40
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
    name="Chimera",
    paper_ref="McLaren & Wilmut 2003, Reproduction 126:831-838",
    paper_cite="McLaren & Wilmut 2003, Cloning by Nuclear Transplantation",
    description="Tests whether mixing QK from one head with OV from another preserves function",
    category="extended",
    tier="extended",
    origin="established",
)

PRESERVATION_THRESHOLD = 0.3
MAX_CHIMERAS = 15


@torch.no_grad()
def run_chimera(model, tasks: list[str],
                n_prompts: int = 40) -> list[EvalResult]:
    """Measure function preservation under QK/OV cross-head transplant."""
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

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        n_valid = min(len(prompts), len(correct_ids))
        head_list = sorted(circuit_heads)
        log(f"  {task}: {len(head_list)} heads, {n_valid} prompts")

        # Cache clean activations for all circuit heads
        # We patch Q,K from one head's clean run and V,O from another
        clean_cache_per_prompt = []
        clean_lds = []
        for idx in range(n_valid):
            tokens = model.to_tokens(prompts[idx].text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: "hook_z" in n or "hook_q" in n or "hook_k" in n or "hook_v" in n)
            clean_cache_per_prompt.append(cache)
            logits = model(tokens)
            clean_lds.append(logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx]))

        mean_clean_ld = float(np.mean(clean_lds))
        if abs(mean_clean_ld) < 1e-8:
            log(f"    baseline LD ~ 0, skipping")
            continue

        # Generate chimera pairs: QK from head_qk, OV from head_ov
        # Only pair heads from different layers
        chimera_pairs = []
        for head_qk in head_list:
            for head_ov in head_list:
                if head_qk != head_ov and head_qk[0] != head_ov[0]:
                    chimera_pairs.append((head_qk, head_ov))

        # If no cross-layer pairs, allow same-layer different-head pairs
        if not chimera_pairs:
            for head_qk in head_list:
                for head_ov in head_list:
                    if head_qk != head_ov:
                        chimera_pairs.append((head_qk, head_ov))

        if not chimera_pairs:
            continue

        # Limit pairs
        if len(chimera_pairs) > MAX_CHIMERAS:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(chimera_pairs), MAX_CHIMERAS, replace=False)
            chimera_pairs = [chimera_pairs[k] for k in sorted(indices)]

        preservation_scores = []
        pair_details = []

        for (lq, hq), (lo, ho) in chimera_pairs:
            # For each prompt, run with hooks that:
            # - Patch Q,K at head_qk's position from head_ov's clean Q,K values
            # - Patch V at head_ov's position from head_qk's clean V values
            # This creates a chimera: attention pattern from one head,
            # value processing from another.
            chimera_lds = []

            for idx in range(n_valid):
                tokens = model.to_tokens(prompts[idx].text)
                cache = clean_cache_per_prompt[idx]

                # Extract clean activations
                clean_z_qk = cache[f"blocks.{lq}.attn.hook_z"][0, :, hq, :].clone()
                clean_z_ov = cache[f"blocks.{lo}.attn.hook_z"][0, :, ho, :].clone()

                # Chimera hook: at the QK head's layer, replace its output
                # with the OV head's output (transplant OV contribution)
                def _hook_qk(z, hook, _hq=hq, _clean_ov=clean_z_ov):
                    z[0, :, _hq, :] = _clean_ov.to(z.device)
                    return z

                def _hook_ov(z, hook, _ho=ho, _clean_qk=clean_z_qk):
                    z[0, :, _ho, :] = _clean_qk.to(z.device)
                    return z

                hooks = [
                    (f"blocks.{lq}.attn.hook_z", _hook_qk),
                    (f"blocks.{lo}.attn.hook_z", _hook_ov),
                ]

                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                ld = logit_diff_from_logits(logits, correct_ids[idx], incorrect_ids[idx])
                chimera_lds.append(ld)

            mean_chimera_ld = float(np.mean(chimera_lds))
            preservation = mean_chimera_ld / mean_clean_ld if abs(mean_clean_ld) > 1e-8 else 0.0
            preservation_scores.append(preservation)

            pair_details.append({
                "head_qk": f"L{lq}H{hq}",
                "head_ov": f"L{lo}H{ho}",
                "chimera_ld": mean_chimera_ld,
                "preservation": float(preservation),
            })

        mean_preservation = float(np.mean(preservation_scores))
        passed = mean_preservation > PRESERVATION_THRESHOLD

        log(f"    mean_preservation={mean_preservation:.3f}  "
            f"({len(chimera_pairs)} chimeras) [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="GN3.chimera",
            value=mean_preservation,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "n_chimeras": len(chimera_pairs),
                "mean_preservation": mean_preservation,
                "max_preservation": float(np.max(preservation_scores)),
                "min_preservation": float(np.min(preservation_scores)),
                "baseline_ld": mean_clean_ld,
                "passed": passed,
                "threshold": PRESERVATION_THRESHOLD,
                "pair_details": sorted(pair_details,
                                       key=lambda d: d["preservation"],
                                       reverse=True)[:10],
            },
        ))

    return results


def main():
    parser = parse_common_args("GN3: Chimera")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("GN3: CHIMERA")
    log("=" * 60)

    out = args.out or "GN3_chimera.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_chimera(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
