"""Causal Scrubbing (Redwood Research)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     A01 — SCM / Pearl Causal Hierarchy
Categories:     causal
Validity layer: Internal
Criteria:       I1 Necessity, I2 Sufficiency
Establishes:    Circuit hypothesis explains model behavior under strict scrubbing criterion
Requires:       GPU, model
Doc:            /instruments_v2/causal/a01-scm-pearl
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether a circuit hypothesis explains model behavior under the strict
scrubbing criterion: resample activations from compatible inputs (same
predicate) and measure KL divergence.

Requires: pip install git+https://github.com/pranavgade20/causal-verifier.git
Marked as slow=True — skip in fast-eval mode.

Usage:
    uv run python 04_causal_scrubbing.py --tasks ioi --n-prompts 20
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    parse_common_args,
    save_results,
)

SLOW = True

try:
    from causal_verifier import CausalVerifier
    HAS_VERIFIER = True
except ImportError:
    HAS_VERIFIER = False


@torch.no_grad()
def scrub_circuit(model, prompts, correct_ids, incorrect_ids,
                  circuit_heads, rng) -> tuple[float, float]:
    """Lightweight causal scrubbing: resample non-circuit heads from
    random compatible prompts and measure KL divergence.

    Returns (kl_div, logit_diff_recovery).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)

    all_caches = []
    all_clean_logprobs = []
    all_clean_logits_last = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
        all_caches.append({L: cache[f"blocks.{L}.attn.hook_z"][0, -1].cpu()
                           for L in range(n_layers)})
        all_clean_logprobs.append(torch.nn.functional.log_softmax(logits[0, -1], dim=-1).cpu())
        all_clean_logits_last.append(logits[0, -1].cpu())

    total_kl = 0.0
    clean_ld_sum = 0.0
    scrubbed_ld_sum = 0.0
    n_valid = 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids) or i >= len(all_caches):
            break
        tokens = model.to_tokens(p.text)

        donor_idx = rng.choice([j for j in range(len(all_caches)) if j != i])

        hooks = []
        for layer, head_list in non_circuit_by_layer.items():
            def _hook(z, hook, _layer=layer, _heads=head_list, _donor=donor_idx):
                for H in _heads:
                    z[0, -1, H, :] = all_caches[_donor][_layer][H].to(z.device)
                return z
            hooks.append((f"blocks.{layer}.attn.hook_z", _hook))

        scrubbed_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        scrubbed_logprobs = torch.nn.functional.log_softmax(scrubbed_logits[0, -1], dim=-1).cpu()

        clean_probs = all_clean_logprobs[i].exp()
        kl = (clean_probs * (all_clean_logprobs[i] - scrubbed_logprobs)).sum().item()
        total_kl += max(kl, 0.0)

        clean_ld = (all_clean_logits_last[i][correct_ids[i]] -
                    all_clean_logits_last[i][incorrect_ids[i]]).item()
        scrubbed_ld = (scrubbed_logits[0, -1, correct_ids[i]] -
                       scrubbed_logits[0, -1, incorrect_ids[i]]).cpu().item()
        clean_ld_sum += clean_ld
        scrubbed_ld_sum += scrubbed_ld
        n_valid += 1

    mean_kl = total_kl / max(n_valid, 1)
    mean_recovery = scrubbed_ld_sum / clean_ld_sum if abs(clean_ld_sum) > 1e-8 else 0.0
    return mean_kl, mean_recovery


def run_causal_scrubbing(model, tasks: list[str], n_prompts: int = 20) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []
    rng = np.random.RandomState(42)

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

        kl_div, recovery = scrub_circuit(model, prompts, correct_ids, incorrect_ids,
                                         circuit_heads, rng)

        log(f"    KL={kl_div:.4f}  recovery={recovery:.3f}")

        results.append(EvalResult(
            metric_id="C4.causal_scrubbing",
            value=kl_div,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "kl_divergence": kl_div,
                "logit_diff_recovery": recovery,
                "n_circuit_heads": len(circuit_heads),
                "slow": True,
            },
        ))

    return results


def main():
    parser = parse_common_args("C4: Causal Scrubbing")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("C4: CAUSAL SCRUBBING (slow)")
    log("=" * 60)

    if not HAS_VERIFIER:
        log("NOTE: causal-verifier not installed. Using lightweight implementation.")

    results = run_causal_scrubbing(model, tasks, args.n_prompts)

    out = args.out or "04_causal_scrubbing.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
