"""Object Permanence — Representation Persistence Across Context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX7 — Object Permanence
Categories:     behavioral, developmental
Evidence family: behavioral
Validity layer: Construct

Tests whether circuit heads maintain stable representations of key
entities (e.g., names in IOI) across sequential positions, rather than
immediately losing track once the entity token has passed.

Background:
    In developmental psychology, object permanence (Piaget 1954) is the
    understanding that objects continue to exist when not directly
    perceived. Baillargeon et al. (1985) showed infants can represent
    hidden objects far earlier than Piaget believed, suggesting that
    representation persistence is a fundamental cognitive capacity.

    Applied to circuits: when a name appears at position p, a well-
    functioning circuit head should maintain a representation of that
    name at later positions — not just at position p itself. If the
    head's activation pattern for a name vanishes immediately after the
    name token, the head does not truly "represent" the entity; it only
    reacts to it locally.

Method:
    1. For each prompt, identify the position of key tokens (the first
       name in the prompt, used as the entity to track).
    2. Run model, cache circuit head activations (hook_result) at each
       position.
    3. Measure how much the name's activation signature persists across
       later positions:
       - At the name's position: capture the activation vector.
       - At each subsequent position: compute correlation with the
         name-position activation.
    4. Persistence score = mean correlation across later positions.
    5. Per-head persistence: which heads maintain representations
       longest?
    6. Pass: mean persistence > 0.3 (representations don't immediately
       decay).

Refs: Piaget 1954; Baillargeon et al. 1985

Usage:
    uv run python EX7_object_permanence.py --tasks ioi --n-prompts 40
    uv run python EX7_object_permanence.py --device cpu
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_circuit_heads,
    heads_to_layer_dict,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Object Permanence (Representation Persistence)",
    paper_ref="Piaget 1954; Baillargeon et al. 1985",
    paper_cite="Piaget 1954, The Construction of Reality in the Child; Baillargeon et al. 1985, Object permanence in five-month-old infants, Cognition 20(3)",
    description="Tests whether circuit heads maintain stable representations of entities across sequential positions, analogous to object permanence in developmental cognition",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

PERSISTENCE_THRESHOLD = 0.3


def find_name_position(tokens_list: list[int], tokenizer) -> int | None:
    """Find the position of the first capitalized name token in the sequence.

    Scans decoded tokens for the first that starts with a capital letter
    and is alphabetic (heuristic for names).
    """
    for pos in range(len(tokens_list)):
        decoded = tokenizer.decode([tokens_list[pos]]).strip()
        if len(decoded) >= 2 and decoded[0].isupper() and decoded.isalpha():
            return pos
    return None


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    a = a.float()
    b = b.float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = (a_centered * b_centered).sum()
    denom = a_centered.norm() * b_centered.norm()
    if denom < 1e-10:
        return 0.0
    return (num / denom).item()


@torch.no_grad()
def run_object_permanence(
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

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        # Collect per-head persistence scores across all prompts
        head_persistences: dict[tuple[int, int], list[float]] = {
            h: [] for h in circuit_heads
        }
        n_valid = 0

        hook_names = [
            f"blocks.{L}.attn.hook_z"
            for L in {l for l, _ in circuit_heads}
        ]

        for p in prompts:
            tokens = model.to_tokens(p.text)
            token_ids = tokens[0].tolist()
            name_pos = find_name_position(token_ids, tokenizer)
            if name_pos is None or name_pos >= len(token_ids) - 2:
                continue

            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names,
            )

            seq_len = tokens.shape[1]
            n_later = seq_len - name_pos - 1
            if n_later < 1:
                continue

            n_valid += 1

            for L, H in circuit_heads:
                hook_key = f"blocks.{L}.attn.hook_z"
                # (batch, seq, n_heads, d_head)
                acts = cache[hook_key][0, :, H, :]  # (seq, d_head)

                name_act = acts[name_pos]  # (d_head,)

                # Correlation with each later position
                corrs = []
                for pos in range(name_pos + 1, seq_len):
                    c = correlation(name_act, acts[pos])
                    corrs.append(c)

                mean_persistence = float(np.mean(corrs)) if corrs else 0.0
                head_persistences[(L, H)].append(mean_persistence)

        if n_valid == 0:
            log(f"  {task}: no valid prompts with identifiable names, skipping")
            continue

        # Aggregate per-head
        per_head_mean = {}
        for h, vals in head_persistences.items():
            per_head_mean[h] = float(np.mean(vals)) if vals else 0.0

        overall_persistence = float(np.mean(list(per_head_mean.values())))

        # Find heads with highest and lowest persistence
        sorted_heads = sorted(per_head_mean.items(), key=lambda x: x[1], reverse=True)
        most_persistent = sorted_heads[:3] if len(sorted_heads) >= 3 else sorted_heads
        least_persistent = sorted_heads[-3:] if len(sorted_heads) >= 3 else sorted_heads

        passed = overall_persistence > PERSISTENCE_THRESHOLD

        log(f"    mean_persistence={overall_persistence:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")
        log(f"    most persistent: {[(f'L{h[0]}H{h[1]}', f'{v:.3f}') for h, v in most_persistent]}")

        results.append(EvalResult(
            metric_id="EX7.object_permanence",
            value=overall_persistence,
            n_samples=n_valid,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "mean_persistence": overall_persistence,
                "n_valid_prompts": n_valid,
                "n_circuit_heads": len(circuit_heads),
                "per_head_persistence": {
                    f"L{L}H{H}": v for (L, H), v in per_head_mean.items()
                },
                "most_persistent_heads": [
                    {"head": f"L{h[0]}H{h[1]}", "persistence": v}
                    for h, v in most_persistent
                ],
                "least_persistent_heads": [
                    {"head": f"L{h[0]}H{h[1]}", "persistence": v}
                    for h, v in least_persistent
                ],
                "passed": passed,
                "threshold": PERSISTENCE_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX7: Object Permanence (Representation Persistence)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX7: OBJECT PERMANENCE (REPRESENTATION PERSISTENCE)")
    log("=" * 60)

    out = args.out or "EX7_object_permanence.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_object_permanence(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
