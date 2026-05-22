"""Metric: Functional Localizer --- domain-selective causal unit identification

Paper: AlKhamissi, Tuckute, Tang, Binhuraib, Bosselut, Schrimpf (2025).
"The LLM Language Network: A Neuroscientific Approach for Identifying
Causally Task-Relevant Units." EMNLP 2025 / NeurIPS 2025 Workshop.
https://llm-language-network.epfl.ch

Applies the neuroscience functional localizer paradigm to LLMs. Runs
standardized contrast conditions (linguistic vs. non-linguistic stimuli)
to identify units that respond selectively to a target domain, then
ablates those units to confirm causal necessity. The selectivity index
(d-prime between domain and control activations) and the causal deficit
(performance drop from selective vs. random ablation) are the primary
diagnostics. Validated across 18 LLMs by the EPFL+MIT team.

Functional Localizer (Evaluation EX19)
=============================================
Instrument:     EX19 --- Functional Localizer
Categories:     evaluation
Validity layer: Internal
Criteria:       I1 Component Necessity, I3 Specificity, C5 Convergent
Establishes:    Whether model units are domain-selective and causally
                necessary for domain-specific task performance
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Run model on domain stimuli and control stimuli, capturing per-unit
   activations at each layer.
2. For each unit, compute selectivity d-prime = (mean_domain - mean_control)
   / pooled_std.
3. Select top-k units by selectivity as the "domain-selective network".
4. Ablate domain-selective units and measure task performance deficit.
5. Ablate an equal number of random units and measure deficit.
6. Causal deficit = selective_deficit - random_deficit.

Pass condition: causal_deficit > 0.2; selectivity_dprime > 1.0

Usage:
    uv run python 121_functional_localizer.py --model gpt2 --device cpu
    uv run python 121_functional_localizer.py --n-prompts 100 --top-k-frac 0.05
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
    name="Functional Localizer",
    paper_ref="AlKhamissi et al. EMNLP 2025",
    paper_cite=(
        "AlKhamissi, Tuckute, Tang, Binhuraib, Bosselut, Schrimpf 2025, "
        "The LLM Language Network: A Neuroscientific Approach for "
        "Identifying Causally Task-Relevant Units "
        "(EMNLP 2025, llm-language-network.epfl.ch)"
    ),
    description=(
        "Applies the neuroscience functional localizer paradigm to LLMs. "
        "Identifies domain-selective units via contrast conditions, then "
        "confirms causal necessity via selective ablation. The causal "
        "deficit (selective vs. random ablation) is the primary diagnostic."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

CAUSAL_DEFICIT_THRESHOLD = 0.2
SELECTIVITY_DPRIME_THRESHOLD = 1.0


def _generate_control_stimuli(prompts, tokenizer) -> list[str]:
    """Generate non-linguistic control stimuli by shuffling tokens.

    For each prompt, randomly permute the token sequence to destroy
    linguistic structure while preserving token distribution.
    """
    controls = []
    for p in prompts:
        tokens = tokenizer.encode(p.text, add_special_tokens=False)
        if len(tokens) < 2:
            controls.append(p.text)
            continue
        perm = np.random.permutation(len(tokens)).tolist()
        shuffled = [tokens[i] for i in perm]
        controls.append(tokenizer.decode(shuffled))
    return controls


@torch.no_grad()
def _collect_activations(model, texts: list[str], hook_name: str) -> torch.Tensor:
    """Collect per-unit activations at the given hook point.

    Returns:
        activations: (n_texts, d_hook) mean activation per text at last position.
    """
    all_acts = []
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        act = cache[hook_name][0, -1]  # (d_hook,)
        all_acts.append(act.cpu())
    return torch.stack(all_acts)  # (n_texts, d_hook)


def _compute_selectivity(
    domain_acts: torch.Tensor, control_acts: torch.Tensor
) -> torch.Tensor:
    """Compute d-prime selectivity for each unit.

    d_prime = (mean_domain - mean_control) / pooled_std

    Args:
        domain_acts: (n_domain, d_hook)
        control_acts: (n_control, d_hook)

    Returns:
        d_prime: (d_hook,)
    """
    mean_d = domain_acts.mean(dim=0)
    mean_c = control_acts.mean(dim=0)
    var_d = domain_acts.var(dim=0, unbiased=True)
    var_c = control_acts.var(dim=0, unbiased=True)
    n_d = domain_acts.shape[0]
    n_c = control_acts.shape[0]
    pooled_var = ((n_d - 1) * var_d + (n_c - 1) * var_c) / (n_d + n_c - 2)
    pooled_std = (pooled_var + 1e-8).sqrt()
    return (mean_d - mean_c) / pooled_std


@torch.no_grad()
def _evaluate_with_ablation(
    model,
    prompts,
    correct_ids: list[int],
    incorrect_ids: list[int],
    hook_name: str,
    ablation_indices: list[int],
) -> float:
    """Evaluate logit diff accuracy with specified units zeroed out.

    Returns mean logit diff across prompts.
    """
    idx_set = set(ablation_indices)

    def ablation_hook(value, hook):
        for i in idx_set:
            value[:, :, i] = 0.0
        return value

    total_ld = 0.0
    count = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, ablation_hook)]
        )
        last = logits[0, -1]
        ld = (last[correct_ids[i]] - last[incorrect_ids[i]]).item()
        total_ld += ld
        count += 1
    return total_ld / max(count, 1)


@torch.no_grad()
def _evaluate_clean(
    model, prompts, correct_ids: list[int], incorrect_ids: list[int]
) -> float:
    """Evaluate logit diff without any ablation."""
    total_ld = 0.0
    count = 0
    for i, p in enumerate(prompts):
        if i >= len(correct_ids):
            break
        tokens = model.to_tokens(p.text)
        logits = model(tokens)
        last = logits[0, -1]
        ld = (last[correct_ids[i]] - last[incorrect_ids[i]]).item()
        total_ld += ld
        count += 1
    return total_ld / max(count, 1)


def run_functional_localizer(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    top_k_frac: float = 0.05,
    hook_layer: int | None = None,
    n_random_baselines: int = 5,
) -> list[EvalResult]:
    """Run the functional localizer diagnostic.

    For each task, identifies domain-selective units and tests causal
    necessity via selective vs. random ablation.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts for localization and evaluation.
        top_k_frac: fraction of units to select as domain-selective.
        hook_layer: layer for hook point (default: middle layer).
        n_random_baselines: number of random ablation baselines to average.

    Returns:
        List of EvalResult, one per task plus aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_mlp_out"

    log(f"  Functional localizer at hook: {hook_name}")
    log(f"  top_k_frac={top_k_frac}, n_prompts={n_prompts}")

    results = []
    all_causal_deficits = []

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, model.tokenizer)
        if not correct_ids:
            log(f"    {task}: no valid token ids, skipping")
            continue

        # Split prompts: first half for localization, second half for eval
        n_loc = max(1, len(prompts) // 2)
        loc_prompts = prompts[:n_loc]
        eval_prompts = prompts[n_loc:]
        eval_correct = correct_ids[n_loc:]
        eval_incorrect = incorrect_ids[n_loc:]

        if not eval_correct:
            log(f"    {task}: not enough prompts for eval split, skipping")
            continue

        # Collect domain activations
        domain_texts = [p.text for p in loc_prompts]
        domain_acts = _collect_activations(model, domain_texts, hook_name)

        # Generate and collect control activations
        control_texts = _generate_control_stimuli(loc_prompts, model.tokenizer)
        control_acts = _collect_activations(model, control_texts, hook_name)

        # Compute selectivity
        d_prime = _compute_selectivity(domain_acts, control_acts)
        d_hook = d_prime.shape[0]
        top_k = max(1, int(d_hook * top_k_frac))

        # Select domain-selective units
        selective_indices = torch.topk(d_prime, top_k).indices.tolist()
        mean_selectivity = d_prime[selective_indices].mean().item()

        # Clean baseline
        clean_ld = _evaluate_clean(model, eval_prompts, eval_correct, eval_incorrect)

        # Selective ablation
        selective_ld = _evaluate_with_ablation(
            model, eval_prompts, eval_correct, eval_incorrect,
            hook_name, selective_indices,
        )
        selective_deficit = 1.0 - (selective_ld / clean_ld) if abs(clean_ld) > 1e-8 else 0.0

        # Random ablation baselines
        random_deficits = []
        for _ in range(n_random_baselines):
            random_indices = np.random.choice(d_hook, top_k, replace=False).tolist()
            random_ld = _evaluate_with_ablation(
                model, eval_prompts, eval_correct, eval_incorrect,
                hook_name, random_indices,
            )
            rd = 1.0 - (random_ld / clean_ld) if abs(clean_ld) > 1e-8 else 0.0
            random_deficits.append(rd)

        mean_random_deficit = float(np.mean(random_deficits))
        causal_deficit = selective_deficit - mean_random_deficit

        passed_deficit = causal_deficit > CAUSAL_DEFICIT_THRESHOLD
        passed_selectivity = mean_selectivity > SELECTIVITY_DPRIME_THRESHOLD
        passed = passed_deficit and passed_selectivity

        all_causal_deficits.append(causal_deficit)

        log(f"    {task}: causal_deficit={causal_deficit:.4f}, "
            f"selectivity_dprime={mean_selectivity:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX19.functional_localizer",
            value=causal_deficit,
            n_samples=len(eval_correct),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "causal_deficit": causal_deficit,
                "selective_deficit": selective_deficit,
                "random_deficit_mean": mean_random_deficit,
                "selectivity_dprime_mean": mean_selectivity,
                "n_selective_units": top_k,
                "d_hook": d_hook,
                "top_k_frac": top_k_frac,
                "hook_name": hook_name,
                "clean_logit_diff": clean_ld,
                "passed": passed,
                "threshold_deficit": CAUSAL_DEFICIT_THRESHOLD,
                "threshold_selectivity": SELECTIVITY_DPRIME_THRESHOLD,
            },
        ))

    # Aggregate
    if all_causal_deficits:
        agg_mean = float(np.mean(all_causal_deficits))
        agg_std = float(np.std(all_causal_deficits))
        agg_passed = agg_mean > CAUSAL_DEFICIT_THRESHOLD
        log(f"  Aggregate: causal_deficit={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX19.functional_localizer",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "causal_deficit_mean": agg_mean,
                "causal_deficit_std": agg_std,
                "n_tasks": len(all_causal_deficits),
                "per_task_deficits": {
                    r.metadata["task"]: r.metadata["causal_deficit"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "passed": agg_passed,
                "threshold": CAUSAL_DEFICIT_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX19: Functional Localizer")
    parser.add_argument("--top-k-frac", type=float, default=0.05,
                        help="Fraction of units to select as domain-selective")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for hook point (default: middle)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX19: FUNCTIONAL LOCALIZER")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_functional_localizer(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        top_k_frac=args.top_k_frac,
        hook_layer=args.hook_layer,
        n_random_baselines=args.n_random_baselines,
    )

    out = args.out or "121_functional_localizer.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
