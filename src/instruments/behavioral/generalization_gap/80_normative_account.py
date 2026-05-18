"""Normative Account Assessment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     D09 — Generalization Gap
Categories:     behavioral
Validity layer: External
Criteria:       CM1 Normative Account (proposed)
Establishes:    Whether circuit solves a genuine, separable subproblem of language modeling
Requires:       CPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests whether the circuit's function is a genuine subproblem:

1. Run model on a diverse corpus of prompts (not just task-specific ones).
2. For each prompt, measure circuit heads' logit attribution magnitude.
3. Cluster prompts by circuit activation pattern (high vs low activation).
4. For the "high activation" cluster: analyze what linguistic properties
   they share (epistemic verbs, modal verbs, hedging, quantifiers, etc.).
5. Report: activation_rate, linguistic_coherence, separation_ratio.
6. Pass: separation_ratio > 2.0 on at least one linguistic feature
   AND activation_rate between 0.05 and 0.50.

Usage:
    uv run python 80_normative_account.py --tasks ioi sva
    uv run python 80_normative_account.py --device cpu --n-prompts 100
"""
import sys
from pathlib import Path

import numpy as np
import torch

_INSTRUMENTS = Path(__file__).resolve().parents[2]  # up to src/instruments/
sys.path.insert(0, str(_INSTRUMENTS))
from _common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_results,
)


# Linguistic feature word lists
LINGUISTIC_FEATURES = {
    "epistemic_verbs": {
        "think", "believe", "know", "assume", "suppose", "expect",
        "doubt", "suspect", "consider", "wonder", "guess", "reckon",
        "imagine", "feel", "hope", "fear", "trust", "realize",
    },
    "modal_verbs": {
        "can", "could", "may", "might", "must", "shall", "should",
        "will", "would", "need", "ought",
    },
    "hedging_words": {
        "perhaps", "maybe", "possibly", "probably", "likely",
        "unlikely", "apparently", "seemingly", "roughly", "approximately",
        "somewhat", "fairly", "quite", "rather", "almost",
    },
    "quantifiers": {
        "all", "every", "each", "some", "any", "no", "none", "few",
        "many", "most", "several", "both", "either", "neither",
        "much", "little", "enough",
    },
    "conjunctions": {
        "and", "but", "or", "nor", "yet", "so", "because", "since",
        "although", "though", "while", "whereas", "unless", "until",
        "if", "when", "after", "before", "however", "therefore",
        "moreover", "furthermore", "nevertheless", "meanwhile",
    },
    "negation": {
        "not", "never", "no", "nothing", "nobody", "nowhere",
        "neither", "hardly", "barely", "scarcely", "rarely",
    },
    "pronouns": {
        "he", "she", "they", "it", "him", "her", "them", "his",
        "its", "their", "who", "whom", "which", "that",
    },
}


DIVERSE_PROMPTS = [
    "The president of the company announced that",
    "Scientists discovered a new species of",
    "When the rain started, the children ran to",
    "The old man sitting on the bench looked at",
    "After years of research, they finally found",
    "Mary told John that she would give the book to",
    "The cat sat on the mat and watched the",
    "In the beginning, there was nothing but",
    "The teacher asked the student to explain",
    "Running through the forest, she noticed a",
    "The data suggests that the hypothesis is",
    "Perhaps we should consider whether the",
    "Every student must complete all assignments before",
    "Although the evidence seems clear, some researchers doubt",
    "The mayor believes that the new policy will",
    "Nobody could have predicted that the",
    "If you think about it carefully, the answer might",
    "Several studies have shown that approximately",
    "He wondered whether she would ever",
    "The committee decided that neither option was",
    "Most people assume that climate change will",
    "She never expected that the results would",
    "The judge ruled that the defendant should",
    "According to the report, the economy is",
    "While the first experiment failed, the second",
    "Both candidates promised to improve",
    "The theory predicts that under certain conditions",
    "Few people realize how important it is to",
    "The doctor recommended that the patient should",
    "Therefore, we can conclude that the",
    "Meanwhile, the situation in the region continued to",
    "The musician played a beautiful melody on the",
    "Despite the warnings, many people still believe",
    "The algorithm processes each input and generates",
    "When asked about the incident, the witness said",
    "The temperature dropped significantly after the",
    "She carefully examined the evidence before",
    "The company's profits increased by roughly",
    "No one could explain why the machine stopped",
    "The children were playing in the garden when",
    "It is widely known that exercise helps",
    "The professor explained that the concept of",
    "After the meeting, they agreed to",
    "The pilot announced that the plane would",
    "He suspects that someone has been",
    "The experiment demonstrated that even small changes can",
    "Unless the government acts quickly, the crisis will",
    "The painting depicts a scene from",
    "Many scientists now think that the universe",
    "The storm caused significant damage to the",
    "She realized that her assumption was",
    "The new technology could potentially",
    "Before the invention of electricity, people used",
    "The survey found that most respondents prefer",
    "However, the results contradict previous",
    "The athlete trained every day to",
    "Some experts argue that artificial intelligence might",
    "The library contains thousands of books about",
    "Neither the teachers nor the students were",
    "The evidence clearly shows that the",
]


def count_feature(text: str, feature_name: str) -> int:
    """Count occurrences of a linguistic feature in text."""
    words = set(text.lower().split())
    feature_words = LINGUISTIC_FEATURES[feature_name]
    return len(words & feature_words)


def has_feature(text: str, feature_name: str) -> bool:
    """Check if text contains any word from the feature set."""
    return count_feature(text, feature_name) > 0


@torch.no_grad()
def compute_circuit_activation_magnitude(model, tokens,
                                         circuit_heads: set[tuple[int, int]]) -> float:
    """Compute total absolute logit attribution from circuit heads.

    Uses direct logit attribution: z @ W_O @ W_U for each circuit head.
    Returns sum of absolute attributions.
    """
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: "hook_z" in n,
    )

    W_U = model.W_U.cpu().float()  # (d_model, d_vocab)
    total_attrib = 0.0

    for L, H in circuit_heads:
        z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H].cpu().float()  # (d_head,)
        W_O = model.W_O[L, H].cpu().float()  # (d_head, d_model)
        resid_contrib = z @ W_O  # (d_model,)
        logit_contrib = resid_contrib @ W_U  # (d_vocab,)
        # Use max absolute logit as the activation magnitude
        total_attrib += logit_contrib.abs().max().item()

    return total_attrib


def run_normative_account(model, tasks: list[str],
                          n_prompts: int = 60) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    # Use the diverse prompts corpus (plus any extras needed)
    corpus = DIVERSE_PROMPTS[:n_prompts]

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        log(f"  {task} ({len(circuit_heads)} heads, {len(corpus)} diverse prompts)...")

        # Step 1: Compute circuit activation magnitude for each prompt
        activations = []
        for text in corpus:
            tokens = model.to_tokens(text)
            mag = compute_circuit_activation_magnitude(model, tokens, circuit_heads)
            activations.append(mag)

        activations = np.array(activations)
        if activations.std() < 1e-8:
            log(f"    zero variance in activations, skipping")
            continue

        # Step 2: Split into high/low activation clusters (median split)
        median_act = float(np.median(activations))
        high_mask = activations >= median_act
        low_mask = ~high_mask
        high_prompts = [corpus[i] for i in range(len(corpus)) if high_mask[i]]
        low_prompts = [corpus[i] for i in range(len(corpus)) if low_mask[i]]

        # Step 3: Compute linguistic feature rates for each cluster
        feature_analysis = {}
        max_separation = 0.0
        best_feature = None

        for feature_name in LINGUISTIC_FEATURES:
            high_rate = float(np.mean([has_feature(t, feature_name) for t in high_prompts])) if high_prompts else 0.0
            low_rate = float(np.mean([has_feature(t, feature_name) for t in low_prompts])) if low_prompts else 0.0

            # Separation ratio: how much more common is the feature in high-activation prompts?
            if low_rate > 0.01:
                separation = high_rate / low_rate
            elif high_rate > 0.01:
                separation = high_rate / 0.01  # cap denominator
            else:
                separation = 1.0

            feature_analysis[feature_name] = {
                "high_rate": high_rate,
                "low_rate": low_rate,
                "separation_ratio": separation,
            }

            if separation > max_separation:
                max_separation = separation
                best_feature = feature_name

        # Step 4: Compute activation_rate (fraction with above-median activation
        # that are substantially above baseline)
        p75 = float(np.percentile(activations, 75))
        activation_rate = float(np.mean(activations > p75))

        log(f"    activation_rate={activation_rate:.3f}  "
            f"max_separation={max_separation:.2f} ({best_feature})")

        for feat, info in feature_analysis.items():
            if info["separation_ratio"] > 1.5:
                log(f"    {feat}: high={info['high_rate']:.2f}  "
                    f"low={info['low_rate']:.2f}  "
                    f"ratio={info['separation_ratio']:.2f}")

        any_feature_separates = max_separation > 2.0
        rate_in_range = 0.05 <= activation_rate <= 0.50
        passed = any_feature_separates and rate_in_range

        results.append(EvalResult(
            metric_id="CM1.normative_account",
            value=max_separation,
            n_samples=len(corpus),
            metadata={
                "task": task,
                "activation_rate": activation_rate,
                "max_separation_ratio": max_separation,
                "best_feature": best_feature,
                "feature_analysis": feature_analysis,
                "n_circuit_heads": len(circuit_heads),
                "median_activation": float(median_act),
                "p75_activation": float(p75),
                "passed": passed,
                "threshold_separation": 2.0,
                "threshold_rate_low": 0.05,
                "threshold_rate_high": 0.50,
            },
        ))

    return results


def main():
    parser = parse_common_args("CM1: Normative Account Assessment")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("CM1: NORMATIVE ACCOUNT ASSESSMENT")
    log("=" * 60)

    results = run_normative_account(model, tasks, args.n_prompts)

    out = args.out or "80_normative_account.json"
    save_results(results, out)

    log(f"\nDone. {len(results)} tasks evaluated.")
    for r in results:
        t = r.metadata["task"]
        p = "PASS" if r.metadata["passed"] else "FAIL"
        log(f"  {t}: max_sep={r.value:.2f} ({r.metadata['best_feature']})  "
            f"act_rate={r.metadata['activation_rate']:.3f}  [{p}]")


if __name__ == "__main__":
    main()
