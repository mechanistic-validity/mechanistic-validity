"""Error Boundary Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D09 — Generalization Gap
Categories:     behavioral
Validity layer: External
Criteria:       CM2 Error Analysis at Boundaries (proposed)
Establishes:    Whether circuit failures correspond to genuine problem boundaries
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether the circuit fails gracefully at problem boundaries:

1. Generate prompts at different difficulty levels:
   - Easy: clear task instances
   - Medium: longer/more complex instances
   - Hard: edge cases, ambiguous instances, garden-path constructions
2. For each difficulty level:
   a. Measure faithfulness of the circuit.
   b. Measure model accuracy (does the full model also struggle?).
3. Compute alignment: does circuit faithfulness track model accuracy?
   - model wrong AND circuit unfaithful -> good alignment
   - model right AND circuit faithful -> good alignment
   - model right AND circuit unfaithful -> bad (circuit misses clear case)
4. Report: boundary_alignment (fraction of aligned cases).
5. Pass: boundary_alignment > 0.60.

Usage:
    uv run python 81_error_boundary_analysis.py --tasks ioi sva
    uv run python 81_error_boundary_analysis.py --device cpu --n-prompts 40
"""

import numpy as np
import torch

from mechanistic_validity.instruments.common import (
    CIRCUIT_TASKS,
    EvalResult,
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
    save_results,
)


# Difficulty-tiered prompt templates per task
# Each returns (text, correct, incorrect) tuples
DIFFICULTY_TEMPLATES = {
    "ioi": {
        "easy": [
            ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
            ("When Alice and Bob went to the park, Bob handed a toy to", " Alice", " Bob"),
            ("When Sarah and Tom went to the office, Tom passed a note to", " Sarah", " Tom"),
            ("When Emily and David went to the library, David lent a book to", " Emily", " David"),
            ("When Lisa and Mike went to the cafe, Mike offered a coffee to", " Lisa", " Mike"),
            ("When Kate and James went to school, James showed a picture to", " Kate", " James"),
            ("When Anna and Chris went to the beach, Chris threw a ball to", " Anna", " Chris"),
            ("When Lily and Mark went to the mall, Mark bought a gift for", " Lily", " Mark"),
        ],
        "medium": [
            ("When Mary and John went to the store and looked around for a while, John eventually gave a small drink to", " Mary", " John"),
            ("After Alice and Bob arrived at the park and sat down on the bench, Bob carefully handed the old toy to", " Alice", " Bob"),
            ("When Sarah, who was tired, and Tom went to the large office building downtown, Tom quickly passed a short note to", " Sarah", " Tom"),
            ("While Emily and David were browsing the shelves of the old library together, David decided to lend his favorite book to", " Emily", " David"),
            ("After Lisa and Mike had been waiting at the crowded cafe for twenty minutes, Mike finally offered a hot coffee to", " Lisa", " Mike"),
            ("When Kate and James arrived at the new school early in the morning, James excitedly showed a beautiful picture to", " Kate", " James"),
        ],
        "hard": [
            ("When John and Mary went to the store, Mary saw that John had forgotten his wallet, so John asked Mary to pay, but then John gave the change to", " Mary", " John"),
            ("The friend of Alice told Bob that Alice and Bob should meet at the park, where Bob would give the present to", " Alice", " Bob"),
            ("When Tom met Sarah and Sarah's sister at the restaurant, Tom gave the menu first to", " Sarah", " Tom"),
            ("Although David initially planned to give the book to Emily's friend, David ended up giving it to", " Emily", " David"),
            ("After the mix-up where Mike thought Lisa was someone else, Mike apologized and gave the correct order to", " Lisa", " Mike"),
            ("John told Mary that when they went to the store together later, John would give the special item to", " Mary", " John"),
        ],
    },
    "sva": {
        "easy": [
            ("The cat on the mat", " is", " are"),
            ("The dogs in the park", " are", " is"),
            ("The boy near the trees", " runs", " run"),
            ("The girls by the lake", " swim", " swims"),
            ("The bird on the branch", " sings", " sing"),
            ("The cars on the road", " move", " moves"),
            ("The man with the hat", " walks", " walk"),
            ("The women at the table", " talk", " talks"),
        ],
        "medium": [
            ("The cat that chased the dogs on the mat", " is", " are"),
            ("The dogs that the boy in the garden saw", " are", " is"),
            ("The teacher of the students at the school", " gives", " give"),
            ("The books on the shelf near the window", " contain", " contains"),
            ("The child with the red and blue balloons", " laughs", " laugh"),
            ("The musicians from the orchestra downtown", " perform", " performs"),
        ],
        "hard": [
            ("The cat that the dogs that the boy owned chased", " is", " are"),
            ("The key to the cabinets that were on the shelves", " is", " are"),
            ("The army of soldiers with the general's medals", " marches", " march"),
            ("The difficulty of the problems that the students solved", " was", " were"),
            ("The report on the studies of the effects of the drug", " shows", " show"),
            ("The cat that the dogs near the large old oak tree chased quickly", " was", " were"),
        ],
    },
}

# For tasks without specific difficulty templates, generate perturbations
PERTURBATION_PREFIXES = {
    "medium": [
        "After thinking about it, ",
        "In the context of the discussion, ",
        "According to the latest reports, ",
    ],
    "hard": [
        "Despite the confusing circumstances and the ambiguity of the situation, ",
        "Notwithstanding the various interpretations one might consider, ",
        "In light of the somewhat contradictory evidence presented earlier, ",
    ],
}


class DifficultyPrompt:
    def __init__(self, text: str, correct: str, incorrect: str, difficulty: str):
        self.text = text
        self.target_correct = correct
        self.target_incorrect = incorrect
        self.difficulty = difficulty
        self.metadata = {"difficulty": difficulty}


def generate_difficulty_prompts(task: str, tokenizer, n_per_level: int = 10):
    """Generate prompts at easy/medium/hard difficulty levels."""
    if task in DIFFICULTY_TEMPLATES:
        templates = DIFFICULTY_TEMPLATES[task]
        all_prompts = []
        for difficulty in ["easy", "medium", "hard"]:
            for text, correct, incorrect in templates.get(difficulty, [])[:n_per_level]:
                all_prompts.append(DifficultyPrompt(text, correct, incorrect, difficulty))
        return all_prompts

    # Fallback: use standard prompts with perturbation prefixes
    standard_prompts = generate_prompts(task, tokenizer, n_per_level)
    if not standard_prompts:
        return []

    all_prompts = []
    for p in standard_prompts:
        # Easy: original prompt
        all_prompts.append(DifficultyPrompt(
            p.text, p.target_correct, p.target_incorrect, "easy",
        ))
        # Medium: add prefix
        for prefix in PERTURBATION_PREFIXES["medium"][:1]:
            all_prompts.append(DifficultyPrompt(
                prefix + p.text, p.target_correct, p.target_incorrect, "medium",
            ))
        # Hard: add longer prefix
        for prefix in PERTURBATION_PREFIXES["hard"][:1]:
            all_prompts.append(DifficultyPrompt(
                prefix + p.text, p.target_correct, p.target_incorrect, "hard",
            ))

    return all_prompts


@torch.no_grad()
def measure_per_prompt(model, prompts, correct_ids, incorrect_ids,
                       circuit_heads: set[tuple[int, int]],
                       mean_z: torch.Tensor):
    """For each prompt, measure model accuracy and circuit faithfulness.

    Returns list of dicts with keys:
        model_correct (bool), clean_ld (float), circuit_ld (float),
        faithfulness (float), difficulty (str)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    non_circuit = {(L, H) for L in range(n_layers) for H in range(n_heads)} - circuit_heads
    non_circuit_by_layer = heads_to_layer_dict(non_circuit)
    hooks = make_ablation_hook(non_circuit_by_layer, mean_z, "mean")

    per_prompt = []
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)

        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        circuit_ld = logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i])

        model_correct = clean_ld > 0
        faithfulness = circuit_ld / clean_ld if abs(clean_ld) > 1e-8 else 0.0

        per_prompt.append({
            "model_correct": model_correct,
            "clean_ld": clean_ld,
            "circuit_ld": circuit_ld,
            "faithfulness": faithfulness,
            "difficulty": getattr(p, "difficulty", "unknown"),
        })

    return per_prompt


def compute_boundary_alignment(per_prompt_results: list[dict],
                               faithfulness_threshold: float = 0.3) -> float:
    """Compute alignment between model accuracy and circuit faithfulness.

    Aligned cases:
    - model correct AND circuit faithful (faithfulness > threshold)
    - model wrong AND circuit unfaithful (faithfulness <= threshold)

    Misaligned cases:
    - model correct AND circuit unfaithful (circuit misses clear case)
    - model wrong AND circuit faithful (circuit succeeds where model fails)
    """
    if not per_prompt_results:
        return 0.0

    aligned = 0
    for r in per_prompt_results:
        circuit_faithful = r["faithfulness"] > faithfulness_threshold
        if r["model_correct"] == circuit_faithful:
            aligned += 1

    return aligned / len(per_prompt_results)


def run_error_boundary_analysis(model, tasks: list[str],
                                n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_difficulty_prompts(task, tokenizer, n_per_level=n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(prompts)} prompts)...")

        # Calibrate mean_z using easy prompts
        easy_prompts = [p for p in prompts if p.difficulty == "easy"]
        if not easy_prompts:
            easy_prompts = prompts
        mean_z = calibrate_mean_z(model, easy_prompts,
                                  n_calibration=min(50, len(easy_prompts)))

        # Measure per-prompt
        per_prompt = measure_per_prompt(
            model, prompts, correct_ids, incorrect_ids, circuit_heads, mean_z,
        )

        # Per-difficulty stats
        difficulty_stats = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in per_prompt if r["difficulty"] == diff]
            if not diff_results:
                continue
            accuracy = float(np.mean([r["model_correct"] for r in diff_results]))
            mean_faith = float(np.mean([r["faithfulness"] for r in diff_results]))
            difficulty_stats[diff] = {
                "n_prompts": len(diff_results),
                "model_accuracy": accuracy,
                "mean_faithfulness": mean_faith,
            }
            log(f"    {diff}: n={len(diff_results)}  "
                f"accuracy={accuracy:.3f}  faith={mean_faith:.3f}")

        # Overall boundary alignment
        boundary_alignment = compute_boundary_alignment(per_prompt)

        # Also compute alignment per difficulty
        per_diff_alignment = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in per_prompt if r["difficulty"] == diff]
            if diff_results:
                per_diff_alignment[diff] = compute_boundary_alignment(diff_results)

        passed = boundary_alignment > 0.60

        log(f"    boundary_alignment={boundary_alignment:.3f}")

        results.append(EvalResult(
            metric_id="CM2.error_boundary_analysis",
            value=boundary_alignment,
            n_samples=len(per_prompt),
            metadata={
                "task": task,
                "boundary_alignment": boundary_alignment,
                "per_difficulty_alignment": per_diff_alignment,
                "difficulty_stats": difficulty_stats,
                "n_circuit_heads": len(circuit_heads),
                "faithfulness_threshold": 0.3,
                "passed": passed,
                "threshold": 0.60,
            },
        ))

    return results


def main():
    parser = parse_common_args("CM2: Error Boundary Analysis")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("CM2: ERROR BOUNDARY ANALYSIS")
    log("=" * 60)

    results = run_error_boundary_analysis(model, tasks, args.n_prompts)

    out = args.out or "81_error_boundary_analysis.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: alignment={r.value:.3f}  [{p}]")
        for diff, stats in r.metadata["difficulty_stats"].items():
            log(f"    {diff}: acc={stats['model_accuracy']:.3f}  "
                f"faith={stats['mean_faithfulness']:.3f}")


if __name__ == "__main__":
    main()
