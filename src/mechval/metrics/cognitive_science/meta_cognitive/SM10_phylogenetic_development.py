"""Phylogenetic Self-Model Development — Layer-Wise Developmental Trajectory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-10 — Phylogenetic Development
Categories:     wildcard, self_model
Evidence family: representational
Description mode: implementational-functional

Treats layers as "developmental stages" and measures how the model's
self-model quality (logit-lens accuracy and calibration) improves
across depth, analogous to phylogenetic development of self-awareness.

Background:
    Cleeremans (2011, "The Radical Plasticity Thesis: How the Brain
    Learns to be Conscious", Frontiers in Psychology 2:86) proposes
    that self-awareness develops through a system learning to
    re-represent its own internal states with increasing fidelity.
    Early representations are coarse; later ones are refined and
    calibrated.

    In a transformer, each layer refines the residual stream. Using
    the logit lens (nostalgebraist 2020), we can measure the
    "self-model quality" at each layer: how well does the model's
    intermediate representation predict its own final output?

    Three aspects of self-model quality are tracked:
    - Confidence: max softmax probability of the logit-lens prediction
    - Accuracy: fraction of prompts where logit-lens argmax matches
      the model's final answer
    - Calibration: |mean_confidence - accuracy| (well-calibrated
      systems have this near zero)

    The developmental trajectory reveals:
    - Gradual vs sudden onset of self-model (phase transitions)
    - Whether confidence precedes accuracy (overconfident early layers)
    - The maturation index: where in the network the phase transition
      occurs (early = rapid development, late = gradual)

    Connections:
    - Cleeremans (2011) — radical plasticity thesis
    - nostalgebraist (2020) — logit lens
    - Karmiloff-Smith (1992) "Beyond Modularity" — representational
      redescription across developmental stages

Method:
    1. For each prompt, run the model and record the final predicted
       token (argmax of final logits)
    2. At each layer l, apply the logit lens:
       - Compute predicted_probs = softmax(ln_final(resid_l) @ W_U)
       - Confidence_l = max(predicted_probs) for this prompt
       - Correct_l = 1 if argmax(predicted_probs) == final_answer
    3. Across prompts, compute per-layer:
       - mean_confidence, accuracy (fraction correct), calibration error
    4. Developmental score = final_accuracy - initial_accuracy
       (how much the self-model improves across depth)
    5. Phase transition: layer with the largest single-step accuracy
       improvement
    6. Maturation index = phase_transition_layer / n_layers
       - Low = rapid early development
       - High = gradual late development

Pass condition: developmental_score > 0.3 (model improves
substantially across layers).

Usage:
    mechval.run("phylogenetic_development", tasks=["ioi"], device="cpu")

References:
    Cleeremans 2011 "Radical Plasticity Thesis" Frontiers in Psychology 2:86;
    nostalgebraist 2020 (logit lens)
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
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Phylogenetic Self-Model Development",
    paper_ref="Cleeremans 2011, Frontiers in Psychology 2:86; nostalgebraist 2020 (logit lens)",
    paper_cite="Cleeremans 2011 (radical plasticity thesis); nostalgebraist 2020 (logit lens)",
    description="Measures layer-wise developmental trajectory of self-model quality using logit-lens accuracy and calibration",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

PASS_THRESHOLD = 0.3


@torch.no_grad()
def run_phylogenetic_development(model, tasks: list[str],
                                 n_prompts: int = 40) -> list[EvalResult]:
    tokenizer = model.tokenizer
    results = []

    n_layers = model.cfg.n_layers
    W_U = model.W_U.detach()
    b_U = model.b_U.detach() if hasattr(model, 'b_U') and model.b_U is not None else None

    for task in tasks:
        circuit_heads = get_circuit_heads(task)

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)

        log(f"  {task}: {n_layers} layers, {len(prompts)} prompts")

        # Per-layer, per-prompt tracking
        per_layer_confidence = np.zeros((len(prompts), n_layers))
        per_layer_correct = np.zeros((len(prompts), n_layers))
        final_tokens = []
        valid_count = 0

        for idx, p in enumerate(prompts):
            if idx >= len(correct_ids):
                break

            tokens = model.to_tokens(p.text)

            # Full forward pass
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_resid_post" in n,
            )

            # Final answer: argmax of final logits
            final_token = logits[0, -1].argmax().item()
            final_tokens.append(final_token)

            # Per-layer logit lens
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

                if hasattr(model, 'ln_final'):
                    resid_normed = model.ln_final(resid.unsqueeze(0)).squeeze(0)
                else:
                    resid_normed = resid

                layer_logits = resid_normed @ W_U
                if b_U is not None:
                    layer_logits = layer_logits + b_U

                probs = torch.softmax(layer_logits, dim=-1)
                max_prob = probs.max().item()
                pred_token = probs.argmax().item()

                per_layer_confidence[idx, layer] = max_prob
                per_layer_correct[idx, layer] = 1.0 if pred_token == final_token else 0.0

            valid_count += 1

        if valid_count < 5:
            log(f"    too few valid prompts ({valid_count}), skipping")
            continue

        per_layer_confidence = per_layer_confidence[:valid_count]
        per_layer_correct = per_layer_correct[:valid_count]

        # Compute per-layer aggregates
        layer_accuracy = per_layer_correct.mean(axis=0)
        layer_mean_confidence = per_layer_confidence.mean(axis=0)
        layer_calibration_error = np.abs(layer_mean_confidence - layer_accuracy)

        # Developmental score
        initial_accuracy = float(layer_accuracy[0])
        final_accuracy = float(layer_accuracy[-1])
        developmental_score = final_accuracy - initial_accuracy

        # Phase transition: largest single-step accuracy improvement
        accuracy_deltas = np.diff(layer_accuracy)
        phase_transition_layer = int(np.argmax(accuracy_deltas)) if len(accuracy_deltas) > 0 else 0
        max_accuracy_jump = float(accuracy_deltas[phase_transition_layer]) if len(accuracy_deltas) > 0 else 0.0
        maturation_index = phase_transition_layer / n_layers if n_layers > 0 else 0.0

        passed = developmental_score > PASS_THRESHOLD

        # Build per-layer profile
        layer_profile = {}
        for layer in range(n_layers):
            layer_profile[f"layer_{layer}"] = {
                "accuracy": float(layer_accuracy[layer]),
                "mean_confidence": float(layer_mean_confidence[layer]),
                "calibration_error": float(layer_calibration_error[layer]),
                "std_confidence": float(per_layer_confidence[:, layer].std()),
            }

        # Check if confidence precedes accuracy (overconfident early layers)
        # Find first layer where confidence > 0.5 vs first layer where accuracy > 0.5
        first_confident = n_layers
        first_accurate = n_layers
        for layer in range(n_layers):
            if layer_mean_confidence[layer] > 0.5 and first_confident == n_layers:
                first_confident = layer
            if layer_accuracy[layer] > 0.5 and first_accurate == n_layers:
                first_accurate = layer

        confidence_precedes = first_confident < first_accurate

        log(f"    accuracy: L0={initial_accuracy:.3f} -> L{n_layers-1}={final_accuracy:.3f}")
        log(f"    developmental_score={developmental_score:.3f}")
        log(f"    phase transition at layer {phase_transition_layer} "
            f"(jump={max_accuracy_jump:.3f})")
        log(f"    maturation_index={maturation_index:.3f}")
        log(f"    confidence precedes accuracy: {confidence_precedes} "
            f"(confident@L{first_confident}, accurate@L{first_accurate})")
        log(f"    calibration error: L0={layer_calibration_error[0]:.3f} -> "
            f"L{n_layers-1}={layer_calibration_error[-1]:.3f}")
        log(f"    [{'PASS (develops)' if passed else 'FAIL (no development)'}]")

        results.append(EvalResult(
            metric_id="SM10.phylogenetic_development",
            value=developmental_score,
            n_samples=valid_count,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_layers": n_layers,
                "developmental_score": developmental_score,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "phase_transition_layer": phase_transition_layer,
                "max_accuracy_jump": max_accuracy_jump,
                "maturation_index": maturation_index,
                "confidence_precedes_accuracy": confidence_precedes,
                "first_confident_layer": first_confident,
                "first_accurate_layer": first_accurate,
                "layer_profile": layer_profile,
                "accuracy_trajectory": [float(a) for a in layer_accuracy],
                "confidence_trajectory": [float(c) for c in layer_mean_confidence],
                "calibration_trajectory": [float(e) for e in layer_calibration_error],
                "passed": passed,
                "threshold": PASS_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-10: Phylogenetic Self-Model Development")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-10: PHYLOGENETIC SELF-MODEL DEVELOPMENT")
    log("=" * 60)

    out = args.out or "SM10_phylogenetic_development.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_phylogenetic_development(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
