"""Metric: Actionability Score --- concreteness and validation of circuit insights

Paper: Orgad, Barez et al. (2026). "Interpretability Can Be Actionable."
ICML 2026. arXiv:2605.11161

Measures two dimensions of actionability:
1. Concreteness --- can the circuit insight enable a specific
   intervention?  Computed as the norm ratio of a steering vector
   derived from circuit heads vs one derived from the full model.
2. Validation --- does the intervention produce the expected
   behavioral change?  Computed as the fraction of prompts where
   applying the circuit-derived steering vector shifts the output
   toward the correct answer.

The product concreteness * validation gives the overall actionability
score.

Actionability Score (Evaluation EX33)
=============================================
Instrument:     EX33 --- Actionability Score
Categories:     evaluation
Validity layer: External
Criteria:       E1 Downstream Utility
Establishes:    Whether circuit-level insights translate into
                actionable interventions (steering)
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Extract mean activation difference at circuit heads between
   correct and incorrect prompts as a steering vector.
2. Extract similar steering vector from the full model's residual
   stream for comparison.
3. Concreteness = norm(circuit_steering) / norm(full_steering).
4. Apply circuit-derived steering to held-out prompts, measure
   fraction that flip toward the correct answer.
5. actionability = concreteness * validation.

Pass condition: actionability > 0.1

Usage:
    uv run python 128_actionability.py --model gpt2 --device cpu
    uv run python 128_actionability.py --tasks ioi --n-prompts 40
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
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Actionability Score",
    paper_ref="Orgad, Barez et al. ICML 2026, arXiv:2605.11161",
    paper_cite=(
        "Orgad, Barez et al. 2026, "
        "Interpretability Can Be Actionable "
        "(ICML 2026, arXiv:2605.11161)"
    ),
    description=(
        "Measures whether circuit insights are actionable via two "
        "dimensions: concreteness (can the circuit produce a steering "
        "vector?) and validation (does steering change behavior as "
        "predicted?). The product gives the actionability score."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

ACTIONABILITY_THRESHOLD = 0.1


@torch.no_grad()
def _extract_circuit_steering_vector(
    model,
    circuit_heads: set[tuple[int, int]],
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
) -> torch.Tensor:
    """Extract mean activation difference at circuit heads between correct/incorrect.

    Collects hook_z activations at the last token position for circuit
    heads, computes the mean across prompts where the model predicts
    correctly vs incorrectly, and returns the difference as a
    residual-stream-shaped steering vector.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    correct_accum = torch.zeros(d_model)
    incorrect_accum = torch.zeros(d_model)
    n_correct, n_incorrect = 0, 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_z" in n
        )

        ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])

        # Sum circuit head contributions through W_O to get residual contribution
        contribution = torch.zeros(d_model)
        for (L, H) in circuit_heads:
            z = cache[f"blocks.{L}.attn.hook_z"][0, -1, H]  # (d_head,)
            W_O = model.W_O[L, H]  # (d_head, d_model)
            contribution += (z @ W_O).cpu()

        if ld > 0:
            correct_accum += contribution
            n_correct += 1
        else:
            incorrect_accum += contribution
            n_incorrect += 1

    if n_correct > 0:
        correct_accum /= n_correct
    if n_incorrect > 0:
        incorrect_accum /= n_incorrect

    return correct_accum - incorrect_accum


@torch.no_grad()
def _extract_full_steering_vector(
    model,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    hook_layer: int,
) -> torch.Tensor:
    """Extract full-model steering vector from residual stream differences."""
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    d_model = model.cfg.d_model

    correct_accum = torch.zeros(d_model)
    incorrect_accum = torch.zeros(d_model)
    n_correct, n_incorrect = 0, 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        ld = logit_diff_from_logits(logits, correct_ids[i], incorrect_ids[i])
        resid = cache[hook_name][0, -1].cpu()  # (d_model,)

        if ld > 0:
            correct_accum += resid
            n_correct += 1
        else:
            incorrect_accum += resid
            n_incorrect += 1

    if n_correct > 0:
        correct_accum /= n_correct
    if n_incorrect > 0:
        incorrect_accum /= n_incorrect

    return correct_accum - incorrect_accum


@torch.no_grad()
def _validate_steering(
    model,
    steering_vec: torch.Tensor,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    hook_layer: int,
    scale: float = 3.0,
) -> float:
    """Fraction of prompts where steering flips output toward correct answer."""
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    direction = steering_vec.to(next(model.parameters()).device)

    n_flipped = 0
    n_tested = 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)

        clean_logits = model(tokens)
        clean_ld = logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i])

        def steer_hook(value, hook):
            value[0, -1] += direction * scale
            return value

        steered_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, steer_hook)]
        )
        steered_ld = logit_diff_from_logits(steered_logits, correct_ids[i], incorrect_ids[i])

        # Count as flip if steering moved logit-diff in the positive direction
        if steered_ld > clean_ld:
            n_flipped += 1
        n_tested += 1

    return n_flipped / max(n_tested, 1)


def run_actionability_score(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    steering_scale: float = 3.0,
) -> list[EvalResult]:
    """Measure actionability of circuit insights.

    For each task, extracts circuit-derived and full-model steering
    vectors, computes concreteness (norm ratio) and validation (flip
    rate), and reports actionability = concreteness * validation.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: all circuit tasks).
        n_prompts: prompts per task.
        steering_scale: magnitude multiplier for steering vector.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    results = []
    all_actionability = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            continue

        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            continue

        # Split prompts: first half for extraction, second half for validation
        mid = len(prompts) // 2
        extract_prompts = prompts[:mid]
        validate_prompts = prompts[mid:]
        extract_correct = correct_ids[:mid]
        extract_incorrect = incorrect_ids[:mid]
        validate_correct = correct_ids[mid:]
        validate_incorrect = incorrect_ids[mid:]

        # Determine hook layer (max layer of circuit heads)
        max_circuit_layer = max(L for L, _ in circuit_heads)
        hook_layer = min(max_circuit_layer + 1, model.cfg.n_layers - 1)

        # Extract steering vectors
        circuit_sv = _extract_circuit_steering_vector(
            model, circuit_heads, extract_prompts,
            extract_correct, extract_incorrect,
        )
        full_sv = _extract_full_steering_vector(
            model, extract_prompts,
            extract_correct, extract_incorrect, hook_layer,
        )

        # Concreteness: norm ratio
        circuit_norm = circuit_sv.norm().item()
        full_norm = full_sv.norm().item()
        concreteness = circuit_norm / max(full_norm, 1e-8)
        concreteness = min(concreteness, 1.0)  # cap at 1

        # Validation: flip rate on held-out prompts
        validation = _validate_steering(
            model, circuit_sv, validate_prompts,
            validate_correct, validate_incorrect,
            hook_layer, scale=steering_scale,
        )

        actionability = concreteness * validation
        all_actionability.append(actionability)
        passed = actionability > ACTIONABILITY_THRESHOLD

        log(f"  {task}: concreteness={concreteness:.4f} "
            f"validation={validation:.4f} actionability={actionability:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX33.actionability_score",
            value=actionability,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "concreteness": concreteness,
                "validation": validation,
                "actionability": actionability,
                "circuit_sv_norm": circuit_norm,
                "full_sv_norm": full_norm,
                "hook_layer": hook_layer,
                "steering_scale": steering_scale,
                "n_circuit_heads": len(circuit_heads),
                "passed": passed,
                "threshold": ACTIONABILITY_THRESHOLD,
            },
        ))

    # Aggregate
    if all_actionability:
        agg = float(np.mean(all_actionability))
        agg_passed = agg > ACTIONABILITY_THRESHOLD
        log(f"  Aggregate: actionability={agg:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")
        results.append(EvalResult(
            metric_id="EX33.actionability_score",
            value=agg,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_actionability": agg,
                "n_tasks": len(all_actionability),
                "passed": agg_passed,
                "threshold": ACTIONABILITY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX33: Actionability Score")
    parser.add_argument("--steering-scale", type=float, default=3.0,
                        help="Steering vector magnitude multiplier")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX33: ACTIONABILITY SCORE")
    log("=" * 60)

    results = run_actionability_score(
        model,
        tasks=args.tasks or CIRCUIT_TASKS,
        n_prompts=args.n_prompts,
        steering_scale=args.steering_scale,
    )

    out = args.out or "128_actionability.json"
    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
