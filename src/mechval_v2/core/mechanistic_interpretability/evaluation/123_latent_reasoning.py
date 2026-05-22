"""Metric: Latent Reasoning Validity --- steering sensitivity and shortcut detection

Paper: Zhang, Tang, Ju, Duan, Liu (2025). "Do Latent Tokens Think?
A Causal and Adversarial Analysis of Chain-of-Continuous-Thought."
arXiv:2512.21711. Evaluating: Hao et al. (2025). "Training Large
Language Models to Reason in a Continuous Latent Space." COLM 2025,
arXiv:2412.06769.

Tests whether intermediate representations claimed to encode reasoning
states actually do so, by measuring (1) steering sensitivity: whether
perturbing specific dimensions predictably changes output in semantically
consistent ways, and (2) shortcut exploitation: whether the representation's
predictive success depends on dataset artifacts rather than genuine
reasoning. Generalizes the "Do Latent Tokens Think?" methodology to any
representation claim (SAE features, circuit activations, hidden states).

Latent Reasoning Validity (Evaluation EX21)
=============================================
Instrument:     EX21 --- Latent Reasoning Validity
Categories:     evaluation
Validity layer: Internal
Criteria:       E1 Content Validity, I2 Compositional Sufficiency,
                C5 Convergent Validity
Establishes:    Whether representations claimed to encode reasoning
                actually contain steerable, non-shortcut information
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Run model on task prompts, capturing hidden states at specified layer.
2. Steering test: perturb top-k principal components of the hidden state
   and measure output change. High sensitivity = representation encodes
   task-relevant information. Low sensitivity = representation is noise
   or redundant.
3. Shortcut test: compare model accuracy on original vs. shuffled/ablated
   prompts. If accuracy persists on corrupted prompts, the model exploits
   statistical shortcuts rather than genuine reasoning.
4. Compute steering_sensitivity and shortcut_exploitation_rate.

Pass condition: steering_sensitivity > 0.3; shortcut_exploitation_rate < 0.2

Usage:
    uv run python 123_latent_reasoning.py --model gpt2 --device cpu
    uv run python 123_latent_reasoning.py --n-prompts 50 --perturbation-scale 2.0
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Latent Reasoning Validity",
    paper_ref="Zhang et al. 2025",
    paper_cite=(
        "Zhang, Tang, Ju, Duan, Liu 2025, "
        "Do Latent Tokens Think? A Causal and Adversarial Analysis of "
        "Chain-of-Continuous-Thought (arXiv:2512.21711)"
    ),
    description=(
        "Tests whether intermediate representations encode steerable, "
        "non-shortcut reasoning information. Measures steering sensitivity "
        "(output change under targeted perturbation) and shortcut "
        "exploitation (accuracy on corrupted prompts)."
    ),
    category="evaluation",
    tier="emerging",
    origin="external",
)

STEERING_THRESHOLD = 0.3
SHORTCUT_THRESHOLD = 0.2


@torch.no_grad()
def _compute_steering_sensitivity(
    model,
    prompts,
    correct_ids: list[int],
    hook_name: str,
    perturbation_scale: float = 2.0,
    n_components: int = 5,
) -> tuple[float, list[dict]]:
    """Measure how much targeted perturbation of hidden states changes output.

    For each prompt:
    1. Run clean forward, record logit diff.
    2. Capture hidden state, compute top PCA directions.
    3. Perturb along each direction, measure logit change.
    4. Sensitivity = mean |perturbed_logit - clean_logit| / |clean_logit|.

    Returns:
        mean_sensitivity: average across prompts
        per_prompt: list of per-prompt detail dicts
    """
    # First pass: collect hidden states for PCA
    all_hidden = []
    for p in prompts[:min(len(prompts), len(correct_ids))]:
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        h = cache[hook_name][0, -1].cpu()  # (d_model,)
        all_hidden.append(h)

    if len(all_hidden) < 2:
        return 0.0, []

    hidden_mat = torch.stack(all_hidden)  # (n, d_model)
    hidden_centered = hidden_mat - hidden_mat.mean(dim=0, keepdim=True)

    # SVD for top components
    U, S, Vt = torch.linalg.svd(hidden_centered, full_matrices=False)
    n_comp = min(n_components, Vt.shape[0])
    directions = Vt[:n_comp]  # (n_comp, d_model)

    per_prompt = []
    sensitivities = []

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)

        # Clean forward
        clean_logits = model(tokens)
        clean_target = clean_logits[0, -1, correct_ids[i]].item()

        # Perturbed forwards
        prompt_sensitivity = []
        for d_idx in range(n_comp):
            direction = directions[d_idx].to(model.cfg.device)

            def perturb_hook(value, hook, _dir=direction):
                value[:, -1, :] += perturbation_scale * _dir
                return value

            perturbed_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, perturb_hook)]
            )
            perturbed_target = perturbed_logits[0, -1, correct_ids[i]].item()

            if abs(clean_target) > 1e-8:
                rel_change = abs(perturbed_target - clean_target) / abs(clean_target)
            else:
                rel_change = abs(perturbed_target - clean_target)
            prompt_sensitivity.append(rel_change)

        mean_sens = float(np.mean(prompt_sensitivity))
        sensitivities.append(mean_sens)
        per_prompt.append({
            "prompt_index": i,
            "sensitivity": mean_sens,
            "per_direction": prompt_sensitivity,
        })

    return float(np.mean(sensitivities)) if sensitivities else 0.0, per_prompt


@torch.no_grad()
def _compute_shortcut_exploitation(
    model,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
) -> tuple[float, dict]:
    """Measure whether model exploits shortcuts by testing on corrupted prompts.

    Corrupts prompts by reversing token order (destroying syntax while
    preserving token distribution). If model still gets the right answer
    on corrupted prompts, it is using statistical shortcuts.

    Returns:
        shortcut_rate: fraction of prompts where model is correct on
                       corrupted input (high = shortcut exploitation)
        details: diagnostic info
    """
    clean_correct = 0
    corrupt_correct = 0
    n_eval = 0

    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break

        tokens = model.to_tokens(p.text)

        # Clean evaluation
        clean_logits = model(tokens)
        clean_ld = (clean_logits[0, -1, correct_ids[i]] -
                    clean_logits[0, -1, incorrect_ids[i]]).item()
        if clean_ld > 0:
            clean_correct += 1

        # Corrupted evaluation: reverse internal tokens (keep BOS)
        corrupted = tokens.clone()
        if tokens.shape[1] > 2:
            corrupted[0, 1:] = tokens[0, 1:].flip(0)
        corrupt_logits = model(corrupted)
        corrupt_ld = (corrupt_logits[0, -1, correct_ids[i]] -
                      corrupt_logits[0, -1, incorrect_ids[i]]).item()
        if corrupt_ld > 0:
            corrupt_correct += 1

        n_eval += 1

    if n_eval == 0:
        return 0.0, {}

    clean_acc = clean_correct / n_eval
    corrupt_acc = corrupt_correct / n_eval

    # Shortcut rate: fraction correct on corrupted that were also correct clean
    if clean_correct > 0:
        shortcut_rate = corrupt_acc / clean_acc
    else:
        shortcut_rate = 0.0

    return shortcut_rate, {
        "clean_accuracy": clean_acc,
        "corrupt_accuracy": corrupt_acc,
        "n_evaluated": n_eval,
    }


def run_latent_reasoning_validity(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    perturbation_scale: float = 2.0,
    hook_layer: int | None = None,
) -> list[EvalResult]:
    """Run latent reasoning validity diagnostic.

    Tests whether intermediate representations encode steerable,
    non-shortcut reasoning information.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts per task.
        perturbation_scale: magnitude of steering perturbation.
        hook_layer: layer for representation capture (default: middle).

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_resid_mid"

    log(f"  Latent reasoning validity at hook: {hook_name}")
    log(f"  perturbation_scale={perturbation_scale}, n_prompts={n_prompts}")

    results = []
    all_sensitivities = []
    all_shortcut_rates = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        # Steering sensitivity
        sensitivity, per_prompt = _compute_steering_sensitivity(
            model, prompts, correct_ids, hook_name,
            perturbation_scale=perturbation_scale,
        )

        # Shortcut exploitation
        shortcut_rate, shortcut_details = _compute_shortcut_exploitation(
            model, prompts, correct_ids, incorrect_ids,
        )

        all_sensitivities.append(sensitivity)
        all_shortcut_rates.append(shortcut_rate)

        passed_steering = sensitivity > STEERING_THRESHOLD
        passed_shortcut = shortcut_rate < SHORTCUT_THRESHOLD
        passed = passed_steering and passed_shortcut

        log(f"    {task}: steering={sensitivity:.4f}, "
            f"shortcut={shortcut_rate:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX21.latent_reasoning_validity",
            value=sensitivity,
            n_samples=len(correct_ids),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "steering_sensitivity": sensitivity,
                "shortcut_exploitation_rate": shortcut_rate,
                "perturbation_scale": perturbation_scale,
                "hook_name": hook_name,
                "shortcut_details": shortcut_details,
                "passed_steering": passed_steering,
                "passed_shortcut": passed_shortcut,
                "passed": passed,
                "threshold_steering": STEERING_THRESHOLD,
                "threshold_shortcut": SHORTCUT_THRESHOLD,
            },
        ))

    # Aggregate
    if all_sensitivities:
        agg_sens = float(np.mean(all_sensitivities))
        agg_shortcut = float(np.mean(all_shortcut_rates))
        agg_passed = agg_sens > STEERING_THRESHOLD and agg_shortcut < SHORTCUT_THRESHOLD

        log(f"  Aggregate: steering={agg_sens:.4f}, "
            f"shortcut={agg_shortcut:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX21.latent_reasoning_validity",
            value=agg_sens,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "mean_steering_sensitivity": agg_sens,
                "mean_shortcut_rate": agg_shortcut,
                "n_tasks": len(all_sensitivities),
                "per_task": {
                    r.metadata["task"]: {
                        "steering": r.metadata["steering_sensitivity"],
                        "shortcut": r.metadata["shortcut_exploitation_rate"],
                    }
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX21: Latent Reasoning Validity")
    parser.add_argument("--perturbation-scale", type=float, default=2.0,
                        help="Magnitude of steering perturbation")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for representation capture (default: middle)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX21: LATENT REASONING VALIDITY")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_latent_reasoning_validity(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        perturbation_scale=args.perturbation_scale,
        hook_layer=args.hook_layer,
    )

    out = args.out or "123_latent_reasoning.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
