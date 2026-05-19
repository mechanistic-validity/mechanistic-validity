"""Certified Stability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     M3b — Certified Stability
Categories:     measurement
Validity layer: Measurement
Criteria:       Head-level stability under prompt subsampling
Establishes:    Whether individual circuit heads are robust contributors
                across random subsamples of the evaluation set
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on Anani et al. 2025 ("Certified Circuits", arXiv 2602.22968).

For N random subsamples (each 80% of prompts):
  - Measure each circuit head's individual contribution (ablation effect)
  - A head "passes" a subsample if its individual contribution > 0

Classification:
  - Certified stable: passes in >= 95% of subsamples
  - Contingent: passes in 50-95% of subsamples
  - Unstable: passes in < 50% of subsamples

Pass condition: >= 50% of circuit heads are certified stable.

Usage:
    uv run python M3b_certified_stable.py --tasks ioi --n-prompts 40
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Certified Stability",
    paper_ref="https://arxiv.org/abs/2602.22968",
    paper_cite="Anani et al. 2025",
    description="Head-level stability certification under prompt subsampling",
    category="measurement",
)

CERTIFIED_THRESHOLD = 0.95
CONTINGENT_THRESHOLD = 0.50
SUBSAMPLE_FRACTION = 0.80


@torch.no_grad()
def compute_head_contribution(model, prompts, correct_ids, incorrect_ids,
                              head: tuple[int, int],
                              mean_z: torch.Tensor) -> float:
    """Compute a single head's contribution as mean logit-diff drop when ablated."""
    hooks = make_ablation_hook(heads_to_layer_dict({head}), mean_z, "mean")
    total_drop = 0.0
    count = 0

    for idx, p in enumerate(prompts):
        if idx >= len(correct_ids):
            break

        tokens = model.to_tokens(p.text)
        cid = correct_ids[idx]
        iid = incorrect_ids[idx]

        clean_logits = model(tokens)
        ld_clean = logit_diff_from_logits(clean_logits, cid, iid)

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ld_ablated = logit_diff_from_logits(ablated_logits, cid, iid)

        total_drop += (ld_clean - ld_ablated)
        count += 1

    if count == 0:
        return 0.0
    return total_drop / count


def classify_head(pass_rate: float) -> str:
    if pass_rate >= CERTIFIED_THRESHOLD:
        return "certified"
    if pass_rate >= CONTINGENT_THRESHOLD:
        return "contingent"
    return "unstable"


@torch.no_grad()
def run_certified_stability(model, tasks: list[str],
                            n_prompts: int = 10,
                            n_subsamples: int = 20) -> list[EvalResult]:
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
            continue

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts, "
            f"{n_subsamples} subsamples")

        mean_z = calibrate_mean_z(model, prompts, n_calibration=min(100, len(prompts)))

        n_total = len(prompts)
        subsample_size = max(1, int(SUBSAMPLE_FRACTION * n_total))

        heads = sorted(circuit_heads)
        head_pass_counts: dict[tuple[int, int], int] = {h: 0 for h in heads}

        rng = np.random.RandomState()
        for s in range(n_subsamples):
            indices = rng.choice(n_total, size=subsample_size, replace=False)
            sub_prompts = [prompts[i] for i in indices]
            sub_correct = [correct_ids[i] for i in indices]
            sub_incorrect = [incorrect_ids[i] for i in indices]

            for head in heads:
                contribution = compute_head_contribution(
                    model, sub_prompts, sub_correct, sub_incorrect, head, mean_z)
                if contribution > 0:
                    head_pass_counts[head] += 1

        stability_scores = {}
        certified_heads = []
        contingent_heads = []
        unstable_heads = []

        for head in heads:
            pass_rate = head_pass_counts[head] / n_subsamples
            label = classify_head(pass_rate)
            key = f"L{head[0]}H{head[1]}"
            stability_scores[key] = pass_rate

            if label == "certified":
                certified_heads.append(key)
            elif label == "contingent":
                contingent_heads.append(key)
            else:
                unstable_heads.append(key)

            log(f"    {key}: pass_rate={pass_rate:.2f}  [{label}]")

        frac_certified = len(certified_heads) / len(heads) if heads else 0.0
        passed = frac_certified >= 0.50

        log(f"    certified={len(certified_heads)} contingent={len(contingent_heads)} "
            f"unstable={len(unstable_heads)}  frac_certified={frac_certified:.2f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="M3b.certified_stability",
            value=frac_certified,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(heads),
                "n_subsamples": n_subsamples,
                "subsample_fraction": SUBSAMPLE_FRACTION,
                "stability_scores": stability_scores,
                "certified_heads": certified_heads,
                "contingent_heads": contingent_heads,
                "unstable_heads": unstable_heads,
                "frac_certified": frac_certified,
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("M3b: Certified Stability")
    parser.add_argument("--n-subsamples", type=int, default=20)
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("M3b: CERTIFIED STABILITY")
    log("=" * 60)

    out = args.out or "M3b_certified_stable.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_certified_stability(
            model, [task], args.n_prompts, args.n_subsamples)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: frac_certified={r.value:.2f}  [{p}]")

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
