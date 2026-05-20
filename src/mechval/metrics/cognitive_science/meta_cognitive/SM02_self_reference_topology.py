"""Self-Reference Circuit Topology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-02 — Self-Reference Circuit Topology
Categories:     wildcard, self_model
Evidence family: structural
Description mode: implementational-functional

Maps whether circuit heads participate in self-referential processing
by measuring their weight-space contribution toward self-referential
tokens relative to non-circuit heads.

Background:
    Metzinger (2003, "Being No One", MIT Press) argues that self-models
    are constructed by neural systems that represent their own internal
    states. Berg et al. (2025) extend this to artificial systems,
    asking whether transformer circuits contain components that
    preferentially process self-referential content.

    In a transformer, self-referential processing can be probed via
    weight-space analysis: each attention head's output projection
    (W_O) combined with the unembedding matrix (W_U) determines how
    much that head's output pushes toward specific tokens. Heads that
    disproportionately push toward self-referential tokens (pronouns
    like "I", "my", "me", "itself") relative to the full vocabulary
    may constitute a "self-reference circuit."

    This is a purely weight-space analysis requiring no forward pass.
    It examines whether the circuit identified for a task is enriched
    for self-referential token production compared to random heads.

    Connections:
    - Metzinger (2003) — self-model theory
    - Berg et al. (2025) — self-reference in artificial systems
    - Logit lens / direct logit attribution — W_O @ W_U analysis

Method:
    1. Identify self-referential tokens: pronouns and reflexives
       ("I", "my", "me", "myself", "itself", "we", "our", "self")
    2. For each circuit head, compute its logit contribution toward
       self-referential tokens:
       W_logit_self[h] = W_O[h] @ W_U[:, self_token_ids]
       This measures how much each head's output pushes toward
       self-referential tokens
    3. Compute self-reference score per head:
       score = ||W_logit_self[h]||_F / ||W_O[h] @ W_U||_F
    4. Compare circuit heads vs random heads: is the circuit enriched
       for self-reference?
    5. Enrichment = mean_circuit_score / mean_random_score
    6. Pass: enrichment > 1.0 (circuit heads are more self-referential
       than average)

Pass condition: enrichment > 1.0.

Usage:
    mechval.run("self_reference_topology", tasks=["ioi"], device="cpu")

References:
    - Metzinger 2003, "Being No One", MIT Press
    - Berg et al. 2025
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Self-Reference Circuit Topology",
    paper_ref="Metzinger 2003, Being No One, MIT Press; Berg et al. 2025",
    paper_cite="Metzinger 2003 (self-model theory); Berg et al. 2025",
    description="Tests whether circuit heads are enriched for self-referential token production (weight-space)",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

SELF_REF_TOKENS = [
    "I", " I", "my", " my", "me", " me", "myself", " myself",
    "itself", " itself", "we", " we", "our", " our", "self", " self",
    "My", " My", "Me", " Me", "We", " We", "Our", " Our",
]


@torch.no_grad()
def run_self_reference_topology(model, tasks: list[str],
                                n_prompts: int = 40) -> list[EvalResult]:
    results = []
    tokenizer = model.tokenizer

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U.detach().float()  # (d_model, d_vocab)

    # Find self-referential token IDs
    self_ref_ids = set()
    for token_str in SELF_REF_TOKENS:
        encoded = tokenizer.encode(token_str)
        for tid in encoded:
            self_ref_ids.add(tid)

    self_ref_ids = sorted(self_ref_ids)
    log(f"  Found {len(self_ref_ids)} self-referential token IDs")

    if not self_ref_ids:
        log("  No self-referential tokens found in vocabulary, skipping")
        return results

    # Precompute self-reference score for every head in the model
    all_head_scores = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            W_O_h = model.W_O[layer, head].detach().float()  # (d_head, d_model)

            # Full logit contribution: W_O[h] @ W_U -> (d_head, d_vocab)
            full_logit = W_O_h @ W_U
            full_norm = torch.norm(full_logit, p="fro").item()

            # Self-referential logit contribution
            W_U_self = W_U[:, self_ref_ids]  # (d_model, n_self_tokens)
            self_logit = W_O_h @ W_U_self  # (d_head, n_self_tokens)
            self_norm = torch.norm(self_logit, p="fro").item()

            score = self_norm / full_norm if full_norm > 0 else 0.0
            all_head_scores[(layer, head)] = score

    all_scores_array = np.array(list(all_head_scores.values()))
    global_mean = float(all_scores_array.mean())
    global_std = float(all_scores_array.std())

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        n_circuit = len(circuit_heads)
        all_model_heads = {(l, h) for l in range(n_layers) for h in range(n_heads)}
        non_circuit_heads = all_model_heads - circuit_heads

        # Circuit head scores
        circuit_scores = [all_head_scores[(l, h)] for l, h in sorted(circuit_heads)]
        mean_circuit = float(np.mean(circuit_scores))

        # Non-circuit head scores (the baseline)
        non_circuit_scores = [all_head_scores[(l, h)] for l, h in sorted(non_circuit_heads)]
        mean_non_circuit = float(np.mean(non_circuit_scores))

        # Enrichment ratio
        enrichment = mean_circuit / mean_non_circuit if mean_non_circuit > 0 else 0.0

        passed = enrichment > 1.0

        # Per-head detail
        head_stats = {}
        for l, h in sorted(circuit_heads):
            score = all_head_scores[(l, h)]
            z_score = (score - global_mean) / global_std if global_std > 0 else 0.0
            head_stats[f"L{l}H{h}"] = {
                "self_ref_score": float(score),
                "z_score": float(z_score),
            }

        ranking = sorted(head_stats.items(),
                         key=lambda kv: kv[1]["self_ref_score"], reverse=True)

        log(f"  {task}: {n_circuit} circuit heads")
        log(f"    mean_circuit={mean_circuit:.5f}  mean_non_circuit={mean_non_circuit:.5f}")
        log(f"    enrichment={enrichment:.3f}")
        for k, v in ranking[:5]:
            log(f"      {k}: score={v['self_ref_score']:.5f}  z={v['z_score']:.2f}")
        log(f"    [{'PASS (enriched)' if passed else 'FAIL (not enriched)'}]")

        results.append(EvalResult(
            metric_id="SM02.self_reference_topology",
            value=enrichment,
            n_samples=0,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": n_circuit,
                "n_self_ref_tokens": len(self_ref_ids),
                "mean_circuit_score": mean_circuit,
                "mean_non_circuit_score": mean_non_circuit,
                "enrichment": enrichment,
                "global_mean_score": global_mean,
                "global_std_score": global_std,
                "head_stats": head_stats,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-02: Self-Reference Circuit Topology")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-02: SELF-REFERENCE CIRCUIT TOPOLOGY")
    log("=" * 60)

    out = args.out or "SM02_self_reference_topology.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_self_reference_topology(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
