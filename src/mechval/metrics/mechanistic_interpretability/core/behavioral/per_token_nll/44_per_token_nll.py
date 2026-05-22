"""Per-Token Negative Log-Likelihood
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D04b — Per-Token NLL
Categories:     behavioral
Validity layer: Internal
Criteria:       I1 Necessity
Establishes:    Identifies which token positions depend most on the circuit
Requires:       GPU, model
Doc:            /instruments_v2/behavioral/d04b-per-token-nll
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of measuring only at the final prediction position, this script
measures NLL at EVERY token position. Compares full model NLL per position
vs circuit-ablated NLL per position to identify which positions show the
biggest NLL increase (where the circuit is most critical).

Reports the mean NLL increase, max positional increase, and correlation
between position-level NLL increase and task-relevant positions.

Framework reference: Behavioral Pillar D05 -- positional faithfulness
analysis revealing where in the sequence the circuit contributes most.

Usage:
    uv run python 44_per_token_nll.py --tasks ioi sva
    uv run python 44_per_token_nll.py --device cuda --n-prompts 40
"""

import numpy as np
import torch
import torch.nn.functional as F

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    make_ablation_hook,
    parse_common_args,
    save_results,
)


@torch.no_grad()
def per_position_nll(logits, tokens):
    """Compute NLL at each position (predicting the next token).

    Returns array of length (seq_len - 1).
    """
    # logits: (1, seq_len, vocab_size), tokens: (1, seq_len)
    # Predict token[t+1] from logits[t]
    shift_logits = logits[0, :-1, :]
    shift_targets = tokens[0, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs[torch.arange(shift_targets.shape[0]), shift_targets]
    return nll.cpu().numpy()


@torch.no_grad()
def run_per_token_nll(model, tasks, n_prompts):
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(L, H) for L in range(n_layers) for H in range(n_heads)}
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")
        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        circuit_by_layer = heads_to_layer_dict(circuit_heads)
        hooks = make_ablation_hook(circuit_by_layer, mean_z, "mean")

        all_nll_increases = []
        last_pos_increases = []

        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            seq_len = tokens.shape[1]
            if seq_len < 2:
                continue

            full_logits = model(tokens)
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            nll_full = per_position_nll(full_logits, tokens)
            nll_ablated = per_position_nll(ablated_logits, tokens)
            nll_increase = nll_ablated - nll_full

            all_nll_increases.append(nll_increase)
            last_pos_increases.append(nll_increase[-1])

        if not all_nll_increases:
            continue

        # Compute statistics across prompts
        mean_increase_per_pos = np.mean(
            [inc for inc in all_nll_increases], axis=0
        ) if all(len(inc) == len(all_nll_increases[0]) for inc in all_nll_increases) else None

        mean_nll_increase = float(np.mean([inc.mean() for inc in all_nll_increases]))
        max_nll_increase = float(np.max([inc.max() for inc in all_nll_increases]))
        mean_last_pos_increase = float(np.mean(last_pos_increases))

        # Positional concentration: what fraction of total increase is at last position
        total_per_prompt = [inc.sum() for inc in all_nll_increases]
        last_fraction = float(np.mean(
            [last_pos_increases[j] / total_per_prompt[j]
             if abs(total_per_prompt[j]) > 1e-8 else 0.0
             for j in range(len(last_pos_increases))]
        ))

        log(f"    mean_increase={mean_nll_increase:.4f}, max={max_nll_increase:.4f}, "
            f"last_pos={mean_last_pos_increase:.4f}, last_fraction={last_fraction:.3f}")

        metadata = {
            "task": task,
            "mean_nll_increase_all_pos": mean_nll_increase,
            "max_nll_increase": max_nll_increase,
            "mean_last_pos_increase": mean_last_pos_increase,
            "last_pos_fraction": last_fraction,
            "n_circuit_heads": len(circuit_heads),
            "n_prompts_used": len(all_nll_increases),
        }
        if mean_increase_per_pos is not None:
            # Store top-5 positions with highest mean increase
            top_positions = np.argsort(mean_increase_per_pos)[-5:][::-1]
            metadata["top_positions"] = top_positions.tolist()
            metadata["top_position_increases"] = mean_increase_per_pos[top_positions].tolist()

        results.append(EvalResult(
            metric_id="D05.per_token_nll",
            value=mean_last_pos_increase,
            n_samples=len(all_nll_increases),
            metadata=metadata,
        ))

    return results


def main():
    parser = parse_common_args("D05: Per-Token Negative Log-Likelihood")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("D05: PER-TOKEN NEGATIVE LOG-LIKELIHOOD")
    log("=" * 60)

    results = run_per_token_nll(model, tasks, args.n_prompts)

    out = args.out or "44_per_token_nll.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        log(f"  {t}: last_pos_NLL_increase={r.value:.4f}, "
            f"all_pos_mean={r.metadata['mean_nll_increase_all_pos']:.4f}")


if __name__ == "__main__":
    main()
