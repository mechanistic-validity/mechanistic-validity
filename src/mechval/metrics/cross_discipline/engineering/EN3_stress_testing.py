"""Stress Testing: Circuit Robustness Under Extreme Conditions
Tests circuit robustness with stressed prompt variants: padded to 3x
length with random tokens, repeated subject names, and inserted
distractor names. Measures logit diff retention on each variant.

Pass: mean stress_retention > 0.5
Ref: Hamlet 1994, Random Testing, Encyclopedia of Software Engineering

Usage:
    uv run python EN3_stress_testing.py --tasks ioi --n-prompts 40
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
    name="Stress Testing",
    paper_ref="Hamlet 1994, Encyclopedia of Software Engineering",
    paper_cite="Hamlet 1994, Random Testing, Encyclopedia of Software Engineering",
    description="Tests circuit robustness under extreme conditions: long prompts, repeated tokens, distractors",
    category="cross_discipline",
    tier="cross_discipline",
    origin="established",
)

RETENTION_THRESHOLD = 0.5


def _make_padded_prompt(text: str, tokenizer, pad_factor: int = 3) -> str:
    """Prepend random tokens to make the prompt ~pad_factor times longer."""
    tokens = tokenizer.encode(text)
    n_pad = len(tokens) * (pad_factor - 1)
    vocab_size = tokenizer.vocab_size
    pad_ids = torch.randint(100, vocab_size - 100, (n_pad,)).tolist()
    pad_text = tokenizer.decode(pad_ids)
    return pad_text + " " + text


def _make_repeated_subject_prompt(text: str, n_repeats: int = 5) -> str:
    """Repeat the first capitalized word (likely subject) multiple times."""
    words = text.split()
    subject = None
    for w in words:
        if w[0].isupper() and w.isalpha():
            subject = w
            break
    if subject is None:
        return text
    repeated = (subject + " ") * n_repeats
    return repeated + text


def _make_distractor_prompt(text: str) -> str:
    """Insert distractor names before the actual prompt."""
    distractors = ["Alice", "Bob", "Charlie", "Diana", "Eve",
                   "Frank", "Grace", "Henry"]
    distractor_text = ", ".join(distractors) + ". "
    return distractor_text + text


@torch.no_grad()
def run_stress_testing(model, tasks: list[str],
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

        log(f"  {task}: {len(circuit_heads)} heads, {len(prompts)} prompts")

        # Measure clean logit diffs
        clean_lds = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            logits = model(tokens)
            clean_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

        # Stress variant 1: padded with random tokens
        padded_lds = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            stressed_text = _make_padded_prompt(p.text, tokenizer)
            tokens = model.to_tokens(stressed_text)
            # Truncate if too long for model context
            max_len = model.cfg.n_ctx
            if tokens.shape[1] > max_len:
                tokens = tokens[:, -max_len:]
            logits = model(tokens)
            padded_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

        # Stress variant 2: repeated subject
        repeated_lds = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            stressed_text = _make_repeated_subject_prompt(p.text)
            tokens = model.to_tokens(stressed_text)
            max_len = model.cfg.n_ctx
            if tokens.shape[1] > max_len:
                tokens = tokens[:, -max_len:]
            logits = model(tokens)
            repeated_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

        # Stress variant 3: distractor names
        distractor_lds = []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            stressed_text = _make_distractor_prompt(p.text)
            tokens = model.to_tokens(stressed_text)
            max_len = model.cfg.n_ctx
            if tokens.shape[1] > max_len:
                tokens = tokens[:, -max_len:]
            logits = model(tokens)
            distractor_lds.append(logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i]))

        clean_arr = np.array(clean_lds)
        padded_arr = np.array(padded_lds)
        repeated_arr = np.array(repeated_lds)
        distractor_arr = np.array(distractor_lds)

        # Compute per-variant retention (ratio of stressed LD to clean LD)
        # Use sign-aware retention: if clean is positive, stressed should be too
        eps = 1e-8
        mean_clean = float(np.mean(clean_arr))
        if abs(mean_clean) < eps:
            log(f"    clean LD ~ 0, skipping")
            continue

        padded_retention = float(np.mean(padded_arr)) / mean_clean
        repeated_retention = float(np.mean(repeated_arr)) / mean_clean
        distractor_retention = float(np.mean(distractor_arr)) / mean_clean

        mean_retention = float(np.mean([padded_retention, repeated_retention, distractor_retention]))
        passed = mean_retention > RETENTION_THRESHOLD

        log(f"    padded={padded_retention:.3f}  repeated={repeated_retention:.3f}  "
            f"distractor={distractor_retention:.3f}  mean={mean_retention:.3f}")
        log(f"    [{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EN3.stress_testing",
            value=mean_retention,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_heads": len(circuit_heads),
                "mean_retention": mean_retention,
                "padded_retention": padded_retention,
                "repeated_retention": repeated_retention,
                "distractor_retention": distractor_retention,
                "mean_clean_ld": mean_clean,
                "mean_padded_ld": float(np.mean(padded_arr)),
                "mean_repeated_ld": float(np.mean(repeated_arr)),
                "mean_distractor_ld": float(np.mean(distractor_arr)),
                "passed": passed,
                "retention_threshold": RETENTION_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EN3: Stress Testing")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EN3: STRESS TESTING")
    log("=" * 60)

    out = args.out or "EN3_stress_testing.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_stress_testing(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
