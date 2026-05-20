"""Free Energy Proxy: Entropy Minimization Cascade
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-06 — Entropy Minimization Cascade
Categories:     wildcard, self_model
Evidence family: representational
Description mode: implementational-functional

Tests whether the transformer implements something like free energy
minimization by measuring layer-wise entropy reduction in the
predictive distribution.

Background:
    Friston's Free Energy Principle (2010, "The Free-Energy Principle:
    A Unified Brain Theory?", Nature Reviews Neuroscience 11:127-138)
    predicts that neural systems systematically reduce uncertainty
    (free energy) through processing. In a transformer, this would
    manifest as monotone entropy reduction across layers: each layer
    further reduces uncertainty about the output.

    The logit lens (nostalgebraist 2020) unembeds residual stream
    activations at intermediate layers to produce token probability
    distributions. If the model implements something like predictive
    coding, entropy should decrease monotonically from early to late
    layers.

    Deviations from monotone decrease reveal:
    - Entropy INCREASE at a layer: the layer adds uncertainty,
      possibly integrating new information or computing intermediate
      representations that temporarily increase entropy before
      resolution
    - Plateau: the layer doesn't contribute to the prediction
    - Non-monotone bump: a "reasoning step" where the model revises
      its prediction (analogous to garden-path reanalysis)

    Connections:
    - Friston (2010) — free energy principle
    - Rao & Ballard (1999) "Predictive Coding in the Visual Cortex",
      Nature Neuroscience 2:79-87
    - nostalgebraist (2020) — logit lens
    - Clark (2013) "Whatever Next? Predictive Brains, Situated Agents,
      and the Future of Cognitive Science", Behavioral and Brain
      Sciences 36:181-204

Method:
    1. For each prompt, compute the residual stream at each layer
    2. Apply the logit lens: unembed each layer's residual stream
       using the model's unembedding matrix (W_U)
    3. Convert to probability distribution via softmax
    4. Compute entropy: H = -sum(p * log(p))
    5. Plot entropy vs layer
    6. Compute monotonicity score: fraction of layer transitions
       where entropy decreases
    7. Identify "entropy elbows" — layers with the largest single-step
       entropy reduction

Pass condition: monotonicity > 0.7 (entropy mostly decreases across
layers).

Usage:
    mechval.run("entropy_cascade", tasks=["ioi"], device="cpu")
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
    name="Entropy Minimization Cascade",
    paper_ref="Friston 2010, Nature Reviews Neuroscience 11:127-138; nostalgebraist 2020 (logit lens)",
    paper_cite="Friston 2010 (FEP); nostalgebraist 2020 (logit lens)",
    description="Tests whether entropy decreases monotonically across layers (free energy minimization proxy)",
    category="cogsci_meta",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

MONOTONICITY_THRESHOLD = 0.7


@torch.no_grad()
def run_entropy_cascade(model, tasks: list[str],
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

        all_entropies = np.zeros((len(prompts), n_layers + 1))

        for idx, p in enumerate(prompts):
            if idx >= len(prompts):
                break
            tokens = model.to_tokens(p.text)

            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_resid_post" in n or "hook_resid_pre" in n,
            )

            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

                if hasattr(model, 'ln_final'):
                    ln = model.ln_final
                    resid_normed = ln(resid.unsqueeze(0)).squeeze(0)
                else:
                    resid_normed = resid

                logits = resid_normed @ W_U
                if b_U is not None:
                    logits = logits + b_U

                probs = torch.softmax(logits, dim=-1)
                probs_clamped = torch.clamp(probs, min=1e-10)
                entropy = -torch.sum(probs_clamped * torch.log(probs_clamped)).item()
                all_entropies[idx, layer] = entropy

            final_logits = model(tokens)
            final_probs = torch.softmax(final_logits[0, -1], dim=-1)
            final_probs_clamped = torch.clamp(final_probs, min=1e-10)
            final_entropy = -torch.sum(final_probs_clamped * torch.log(final_probs_clamped)).item()
            all_entropies[idx, n_layers] = final_entropy

        mean_entropies = all_entropies.mean(axis=0)
        std_entropies = all_entropies.std(axis=0)

        n_decreasing = 0
        n_transitions = 0
        layer_deltas = []
        for l in range(len(mean_entropies) - 1):
            delta = mean_entropies[l + 1] - mean_entropies[l]
            layer_deltas.append(float(delta))
            n_transitions += 1
            if delta < 0:
                n_decreasing += 1

        monotonicity = n_decreasing / n_transitions if n_transitions > 0 else 0.0

        elbow_layer = int(np.argmin(layer_deltas)) if layer_deltas else 0
        max_reduction = -min(layer_deltas) if layer_deltas else 0.0

        total_reduction = mean_entropies[0] - mean_entropies[-1]
        relative_reduction = total_reduction / mean_entropies[0] if mean_entropies[0] > 0 else 0.0

        passed = monotonicity >= MONOTONICITY_THRESHOLD

        log(f"    monotonicity={monotonicity:.3f}  "
            f"total_reduction={total_reduction:.2f} ({relative_reduction:.1%})")
        log(f"    elbow_layer={elbow_layer} (reduction={max_reduction:.2f})")
        log(f"    entropy: L0={mean_entropies[0]:.2f} → "
            f"L{n_layers}={mean_entropies[-1]:.2f}")

        bumps = [l for l, d in enumerate(layer_deltas) if d > 0.1]
        if bumps:
            log(f"    entropy bumps at layers: {bumps}")

        log(f"    [{'PASS (FEP-consistent)' if passed else 'FAIL (non-monotone)'}]")

        entropy_profile = {
            f"layer_{l}": {
                "mean_entropy": float(mean_entropies[l]),
                "std_entropy": float(std_entropies[l]),
            }
            for l in range(len(mean_entropies))
        }

        results.append(EvalResult(
            metric_id="SM06.entropy_cascade",
            value=monotonicity,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_layers": n_layers,
                "monotonicity": monotonicity,
                "total_entropy_reduction": float(total_reduction),
                "relative_reduction": float(relative_reduction),
                "elbow_layer": elbow_layer,
                "max_single_step_reduction": float(max_reduction),
                "entropy_profile": entropy_profile,
                "layer_deltas": layer_deltas,
                "entropy_bumps": bumps if bumps else [],
                "initial_entropy": float(mean_entropies[0]),
                "final_entropy": float(mean_entropies[-1]),
                "passed": passed,
                "threshold": MONOTONICITY_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-06: Entropy Minimization Cascade")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-06: ENTROPY MINIMIZATION CASCADE")
    log("=" * 60)

    out = args.out or "SM06_entropy_cascade.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_entropy_cascade(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
