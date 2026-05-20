"""Priming -- Cross-Prompt Circuit State Persistence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX3 -- Priming
Categories:     behavioral, linguistics
Evidence family: behavioral
Description mode: implementational-functional

Tests whether circuit head activations on prompt B are influenced by
prior context from an unrelated prompt A -- i.e., whether circuit
state persists across prompt boundaries.

Background:
    Priming (Meyer & Schvaneveldt 1971, "Facilitation in Recognizing
    Pairs of Words") demonstrates that processing one stimulus affects
    subsequent processing. In human cognition, this is expected for
    semantically related items but should not occur for unrelated ones.

    Applied to circuits: a well-specified circuit should produce
    consistent activations for a given prompt regardless of what
    preceded it (when the prompts are unrelated). If circuit head
    activations for prompt B differ substantially when preceded by
    prompt A versus presented alone, the circuit measurement is
    context-contaminated -- it captures cross-prompt state leakage
    rather than task-specific computation.

    This matters because mechanistic interpretability claims typically
    assume circuit behavior is prompt-local. If priming effects are
    large, ablation and activation patching results may depend on
    prompt ordering in ways that are not accounted for.

    Connections:
    - Meyer & Schvaneveldt (1971) "Facilitation in Recognizing Pairs
      of Words", Journal of Experimental Psychology 90
    - Neely (1977) "Semantic Priming and Retrieval from Lexical
      Memory", Journal of Experimental Psychology: General 106

Method:
    1. Generate N prompts for the task.
    2. For each pair of prompts (A, B) where A != B:
       a. Run B alone, cache circuit head activations at the last
          token position (standalone baseline).
       b. Concatenate A + separator + B, run the concatenated input,
          cache circuit head activations at the position corresponding
          to B's last token (primed condition).
    3. Priming effect per pair = cosine distance between standalone
       and primed activation vectors (concatenated across all circuit
       heads).
    4. Mean priming effect across all pairs is the metric value.
    5. Pass: mean priming effect < 0.1 (circuit is context-independent
       for unrelated prompts).

Usage:
    mechval.run("priming", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Priming (Cross-Prompt State Persistence)",
    paper_ref="Meyer & Schvaneveldt 1971; Neely 1977",
    paper_cite="Meyer & Schvaneveldt 1971, Facilitation in Recognizing Pairs of Words",
    description="Tests whether circuit head activations on a prompt change based on unrelated preceding context",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

PRIMING_THRESHOLD = 0.1
SEPARATOR = "\n\n"


def _extract_circuit_activations(
    cache: dict,
    circuit_heads: set[tuple[int, int]],
    seq_pos: int,
) -> torch.Tensor:
    """Extract and concatenate circuit head activations at a given sequence position.

    Returns a 1-D vector of all circuit head outputs concatenated.
    """
    vecs = []
    for layer, head in sorted(circuit_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"
        # shape: (batch, seq, n_heads, d_head)
        act = cache[hook_name][0, seq_pos, head, :]
        vecs.append(act.cpu().float())
    return torch.cat(vecs)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance = 1 - cosine_similarity. Returns 0 for zero vectors."""
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    cos_sim = torch.dot(a, b) / (norm_a * norm_b)
    return float(1.0 - cos_sim.clamp(-1.0, 1.0))


@torch.no_grad()
def run_priming(
    model,
    tasks: list[str],
    n_prompts: int = 40,
    max_pairs: int = 100,
) -> list[EvalResult]:
    results = []
    for task in tasks:
        r = _run_priming_single(model, task, n_prompts, max_pairs)
        if r is not None:
            results.append(r)
    return results


@torch.no_grad()
def _run_priming_single(
    model,
    task: str,
    n_prompts: int = 40,
    max_pairs: int = 100,
) -> EvalResult | None:
    tokenizer = model.tokenizer

    circuit_heads = get_circuit_heads(task)
    if not circuit_heads:
        log(f"  {task}: no circuit heads, skipping")
        return None

    prompts = generate_prompts(task, tokenizer, n_prompts)
    if len(prompts) < 2:
        log(f"  {task}: need at least 2 prompts, skipping")
        return None

    log(f"  {task}: {len(circuit_heads)} circuit heads, {len(prompts)} prompts")

    hook_filter = lambda n: "attn.hook_z" in n

    # Cache standalone activations for each prompt
    standalone_acts: list[torch.Tensor] = []
    for p in prompts:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(tokens, names_filter=hook_filter)
        last_pos = tokens.shape[1] - 1
        act = _extract_circuit_activations(cache, circuit_heads, last_pos)
        standalone_acts.append(act)

    # Compute priming effects for prompt pairs
    priming_effects: list[float] = []
    pair_count = 0

    for i in range(len(prompts)):
        for j in range(len(prompts)):
            if i == j:
                continue
            if pair_count >= max_pairs:
                break

            # Concatenate prompt A + separator + prompt B
            concat_text = prompts[i].text + SEPARATOR + prompts[j].text
            concat_tokens = model.to_tokens(concat_text)

            # Find position corresponding to B's last token
            b_tokens = model.to_tokens(prompts[j].text)
            primed_last_pos = concat_tokens.shape[1] - 1

            # Skip if concatenation is too long (would OOM or truncate)
            if concat_tokens.shape[1] > 1024:
                continue

            _, concat_cache = model.run_with_cache(
                concat_tokens, names_filter=hook_filter
            )
            primed_act = _extract_circuit_activations(
                concat_cache, circuit_heads, primed_last_pos
            )

            # Compare primed vs standalone
            dist = _cosine_distance(primed_act, standalone_acts[j])
            priming_effects.append(dist)
            pair_count += 1

        if pair_count >= max_pairs:
            break

    if not priming_effects:
        log(f"  {task}: no valid pairs, skipping")
        return None

    effects_arr = np.array(priming_effects)
    mean_effect = float(effects_arr.mean())
    std_effect = float(effects_arr.std()) if len(effects_arr) > 1 else 0.0
    passed = mean_effect < PRIMING_THRESHOLD

    log(f"    mean_priming={mean_effect:.4f}  std={std_effect:.4f}  "
        f"n_pairs={len(priming_effects)}  "
        f"[{'PASS (context-independent)' if passed else 'FAIL (context-leaking)'}]")

    return EvalResult(
        metric_id="EX3.priming",
        value=1.0 - min(mean_effect, 1.0),
        n_samples=len(priming_effects),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "task": task,
            "n_heads": len(circuit_heads),
            "n_pairs": len(priming_effects),
            "mean_priming_effect": mean_effect,
            "std_priming_effect": std_effect,
            "median_priming_effect": float(np.median(effects_arr)),
            "max_priming_effect": float(effects_arr.max()),
            "min_priming_effect": float(effects_arr.min()),
            "passed": passed,
            "threshold": PRIMING_THRESHOLD,
        },
    )


def main():
    parser = parse_common_args("EX3: Priming (Cross-Prompt State Persistence)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX3: PRIMING (CROSS-PROMPT STATE PERSISTENCE)")
    log("=" * 60)

    out = args.out or "EX3_priming.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        result = _run_priming_single(model, task, args.n_prompts)
        if result is None:
            continue
        results.append(result)
        save_incremental(result, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
