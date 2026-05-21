"""Differential Item Functioning (DIF) for Circuit Measurement Bias
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         EX2 — DIF (Differential Item Functioning)
Categories:     behavioral, linguistics
Evidence family: behavioral
Description mode: implementational-functional

Tests whether the circuit performs equivalently across different name
types (gendered, ungendered, rare, common, different-origin names),
controlling for overall circuit ability.

Background:
    Differential Item Functioning (DIF; Holland & Wainer 1993,
    "Differential Item Functioning", Lawrence Erlbaum) is a
    psychometric technique that asks: does a test question function
    differently for two groups of equal ability? If test-takers matched
    on total score have different probabilities of answering a specific
    item correctly based on group membership, that item shows DIF —
    it's measuring something confounded with group membership.

    Applied to circuits: the "test" is the circuit's task performance.
    The "groups" are prompts with different name types. If the IOI
    circuit performs differently on prompts with common English names
    vs rare names vs gendered names, controlling for overall model
    ability, the circuit's measurement is confounded with name
    familiarity or cultural associations.

    This is directly relevant to fairness/bias analysis — a circuit
    that only works well on common English names isn't measuring pure
    syntactic role tracking; it's partially measuring token frequency.

    Connections:
    - Holland & Wainer (1993) — DIF handbook
    - Zumbo (1999) "A Handbook on the Theory and Methods of DIF"
    - Angoff (1993) "Perspectives on DIF Methodology", in Holland &
      Wainer
    - Swaminathan & Rogers (1990) "Detecting DIF Using Logistic
      Regression Procedures", Journal of Educational Measurement 27

Method:
    1. Generate prompts with different name categories:
       a. Common English names (John, Mary, etc.)
       b. Less common names
       c. Names from different linguistic origins
    2. Run the circuit on each category
    3. Compute logit-diff for each prompt
    4. For each category pair: is the mean logit-diff significantly
       different, controlling for model confidence?
    5. DIF magnitude = max |mean_ld_group_A - mean_ld_group_B| / pooled_std
    6. Report per-group accuracy, effect sizes, and overall DIF score

Pass condition: DIF effect size (Cohen's d) < 0.5 across all group pairs.

Usage:
    mechval.run("dif", tasks=["ioi"], device="cpu")
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
    name="Differential Item Functioning (DIF)",
    paper_ref="Holland & Wainer 1993; Zumbo 1999",
    paper_cite="Holland & Wainer 1993, Differential Item Functioning",
    description="Tests measurement equivalence across name types — detects confounds with token frequency or cultural associations",
    category="behavioral",
    tier="cogsci",
    origin="established",
)

DIF_THRESHOLD = 0.5

COMMON_NAMES = ["John", "Mary", "James", "Sarah", "David", "Emily",
                "Michael", "Jessica", "Robert", "Jennifer"]
LESS_COMMON_NAMES = ["Nigel", "Mabel", "Rupert", "Bertha", "Horace",
                     "Millicent", "Percival", "Gertrude", "Archibald", "Prudence"]
DIVERSE_NAMES = ["Hiroshi", "Priya", "Oluwaseun", "Xiao", "Dmitri",
                 "Fatima", "Kwame", "Aisha", "Ravi", "Yuki"]


@torch.no_grad()
def run_dif(model, tasks: list[str],
            n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        if task != "ioi":
            log(f"  {task}: DIF currently only supports IOI (name-based task), skipping")
            continue

        log(f"  {task}: running DIF across 3 name groups")

        name_groups = {
            "common": COMMON_NAMES,
            "uncommon": LESS_COMMON_NAMES,
            "diverse_origin": DIVERSE_NAMES,
        }

        group_lds: dict[str, list[float]] = {}

        for group_name, names in name_groups.items():
            lds = []
            n_valid = 0

            for i in range(0, len(names) - 1, 2):
                if n_valid >= n_prompts // 3:
                    break
                name_a = names[i]
                name_b = names[i + 1] if i + 1 < len(names) else names[0]

                templates = [
                    f"When {name_a} and {name_b} went to the store, {name_b} gave a drink to",
                    f"Then, {name_a} and {name_b} had a meeting. {name_b} said hello to",
                    f"{name_a} and {name_b} were in the room. {name_b} handed the book to",
                ]

                for template in templates:
                    if n_valid >= n_prompts // 3:
                        break
                    tokens = model.to_tokens(template)
                    logits = model(tokens)

                    correct_tok = tokenizer.encode(f" {name_a}")
                    incorrect_tok = tokenizer.encode(f" {name_b}")

                    if not correct_tok or not incorrect_tok:
                        continue

                    ld = logit_diff_from_logits(logits, correct_tok[0], incorrect_tok[0])
                    lds.append(ld)
                    n_valid += 1

            group_lds[group_name] = lds
            log(f"    {group_name}: {len(lds)} prompts, "
                f"mean_ld={np.mean(lds):.4f}, std={np.std(lds):.4f}")

        group_names = list(group_lds.keys())
        max_dif = 0.0
        pairwise_dif = {}

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                lds1 = np.array(group_lds[g1])
                lds2 = np.array(group_lds[g2])

                if len(lds1) < 2 or len(lds2) < 2:
                    continue

                mean_diff = np.mean(lds1) - np.mean(lds2)
                pooled_std = np.sqrt(
                    ((len(lds1) - 1) * np.var(lds1) + (len(lds2) - 1) * np.var(lds2))
                    / (len(lds1) + len(lds2) - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                pair_key = f"{g1}_vs_{g2}"
                pairwise_dif[pair_key] = {
                    "mean_diff": float(mean_diff),
                    "cohens_d": float(cohens_d),
                    "mean_g1": float(np.mean(lds1)),
                    "mean_g2": float(np.mean(lds2)),
                    "n_g1": len(lds1),
                    "n_g2": len(lds2),
                }
                max_dif = max(max_dif, abs(cohens_d))
                log(f"    {pair_key}: Cohen's d={cohens_d:.3f}  "
                    f"mean_diff={mean_diff:.4f}")

        passed = max_dif < DIF_THRESHOLD

        log(f"    max_dif={max_dif:.3f}  "
            f"[{'PASS (equitable)' if passed else 'FAIL (biased)'}]")

        per_group_stats = {}
        for g, lds in group_lds.items():
            arr = np.array(lds)
            per_group_stats[g] = {
                "n": len(lds),
                "mean_ld": float(arr.mean()) if len(arr) > 0 else 0.0,
                "std_ld": float(arr.std()) if len(arr) > 0 else 0.0,
                "accuracy": float(np.mean(arr > 0)) if len(arr) > 0 else 0.0,
            }

        results.append(EvalResult(
            metric_id="EX2.dif",
            value=1.0 - min(max_dif, 1.0),
            n_samples=sum(len(v) for v in group_lds.values()),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "name_groups": list(name_groups.keys()),
                "per_group_stats": per_group_stats,
                "pairwise_dif": pairwise_dif,
                "max_dif_cohens_d": max_dif,
                "passed": passed,
                "threshold": DIF_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX2: Differential Item Functioning (DIF)")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX2: DIFFERENTIAL ITEM FUNCTIONING (DIF)")
    log("=" * 60)

    out = args.out or "EX2_dif.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_dif(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
