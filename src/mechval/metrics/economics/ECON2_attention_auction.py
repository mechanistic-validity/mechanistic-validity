"""Attention Auction: Allocative Efficiency of Circuit Heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         ECON2 — Attention Auction
Categories:     extended, economics
Tier:           extended
Origin:         established

Tests whether attention allocation follows auction-theoretic
efficiency. Treats attention weights as bids and measures whether
heads attend most to positions whose information is most valuable
(measured by logit diff contribution from each position).

Pass: mean efficiency > 0.3
Ref: Vickrey 1961 "Counterspeculation, Auctions, and Competitive
     Sealed Tenders", Journal of Finance 16:8-37

Usage:
    uv run python ECON2_attention_auction.py --tasks ioi sva --n-prompts 40
"""

import numpy as np
import torch
from scipy import stats

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
    name="Attention Auction",
    paper_ref="Vickrey 1961",
    paper_cite="Vickrey 1961, Counterspeculation Auctions and Competitive Sealed Tenders, Journal of Finance 16:8-37",
    description="Tests whether attention allocation is efficient by correlating attention weights with position value",
    category="mi",
    tier="core",
    origin="established",
)

EFFICIENCY_THRESHOLD = 0.3


@torch.no_grad()
def run_attention_auction(model, tasks: list[str],
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

        head_efficiencies = {}

        for L, H in head_list:
            per_prompt_corrs = []

            for i in range(n_valid):
                tokens = model.to_tokens(prompts[i].text)
                seq_len = tokens.shape[1]

                if seq_len < 3:
                    continue

                # Get attention pattern and baseline logit diff
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda n: "hook_z" in n or "attn.hook_pattern" in n)

                pattern = cache[f"blocks.{L}.attn.hook_pattern"]
                # Attention weights from last position to all positions
                attn_weights = pattern[0, H, -1, :seq_len].cpu().numpy()

                # Compute position value: for each source position, zero its
                # contribution through this head's residual stream write and
                # measure LD change
                logits_clean = model(tokens)
                clean_ld = logit_diff_from_logits(logits_clean, correct_ids[i], incorrect_ids[i])

                position_values = np.zeros(seq_len)
                for pos in range(seq_len):
                    def _zero_pos(z, hook, _pos=pos, _H=H):
                        # Zero contribution from this source position
                        # by zeroing the attention-weighted value at that position
                        z[0, _pos, _H, :] = 0.0
                        return z

                    logits_abl = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(f"blocks.{L}.attn.hook_z", _zero_pos)])
                    abl_ld = logit_diff_from_logits(logits_abl, correct_ids[i], incorrect_ids[i])
                    position_values[pos] = clean_ld - abl_ld

                # Correlation between attention weight and position value
                if np.std(attn_weights) < 1e-10 or np.std(position_values) < 1e-10:
                    continue

                corr, _ = stats.spearmanr(attn_weights, position_values)
                if not np.isnan(corr):
                    per_prompt_corrs.append(float(corr))

            if per_prompt_corrs:
                mean_eff = float(np.mean(per_prompt_corrs))
            else:
                mean_eff = 0.0

            head_efficiencies[f"L{L}H{H}"] = {
                "mean_efficiency": mean_eff,
                "n_prompts_used": len(per_prompt_corrs),
            }

        if not head_efficiencies:
            continue

        mean_efficiency = float(np.mean([v["mean_efficiency"] for v in head_efficiencies.values()]))
        passed = mean_efficiency > EFFICIENCY_THRESHOLD

        log(f"    mean_efficiency={mean_efficiency:.4f}  [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="ECON2.attention_auction",
            value=mean_efficiency,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(head_list),
                "mean_efficiency": mean_efficiency,
                "per_head": head_efficiencies,
                "threshold": EFFICIENCY_THRESHOLD,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("ECON2: Attention Auction")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("ECON2: ATTENTION AUCTION")
    log("=" * 60)

    out = args.out or "ECON2_attention_auction.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_attention_auction(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
