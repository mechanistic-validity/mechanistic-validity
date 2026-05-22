"""Input-Dependent vs Input-Invariant Decomposition (Evaluation EX34)
Paper: Dunefsky, Chanin, Neel Nanda (2024). "Transcoders Find Interpretable
LLM Feature Circuits." arXiv:2406.11944
=============================================
Instrument:     EX34 --- Transcoder Decomposition Fraction
Categories:     evaluation
Validity layer: Construct / Measurement
Criteria:       M6 Construct Coverage, C2 Structural Plausibility
Establishes:    Whether the input-dependent sparse features of a
                transcoder-style decomposition capture the majority of
                MLP output variance, or whether the input-invariant
                (bias/mean) term dominates
Requires:       CPU or GPU, model
=============================================

Transcoders uniquely decompose MLP computation into:
  f(x) = sum_i a_i(x) * d_i   +   b
          ^^^^^^^^^^^^^^^^^^^       ^
          input-dependent           input-invariant
          (sparse features)         (bias/mean term)

A good transcoder should have most MLP output variance explained by
the input-dependent term (sparse features doing real computational
work), not the input-invariant bias. This metric measures the fraction
of MLP output variance captured by each term.

Core logic:
1. Run prompts through the model, capturing MLP input and output
   activations at a target layer.
2. Compute the MLP mean output (input-invariant proxy = mean activation).
3. For each position, decompose MLP output into:
   - Input-invariant: the mean MLP output.
   - Input-dependent: the residual (actual output - mean output).
4. Compute variance of each term across all positions.
5. input_dependent_fraction = Var(input-dependent) / Var(total output).

High fraction (>0.6) means sparse features capture real computation.
Low fraction means the MLP's behavior is dominated by a constant
bias term, making sparse feature analysis less informative.

Pass condition: input_dependent_fraction > 0.6

Usage:
    uv run python 141_transcoder_decomposition.py --model gpt2 --device cpu
    uv run python 141_transcoder_decomposition.py --n-prompts 50
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    load_model,
    log,
    parse_common_args,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Transcoder Decomposition Fraction",
    paper_ref="Dunefsky et al. 2024",
    paper_cite=(
        "Dunefsky, Chanin, Nanda 2024, "
        "Transcoders Find Interpretable LLM Feature Circuits "
        "(arXiv:2406.11944)"
    ),
    description=(
        "Measures what fraction of MLP output variance is captured by "
        "input-dependent sparse features vs the input-invariant bias "
        "term. High input-dependent fraction means sparse features "
        "capture real computational work; low fraction means the MLP "
        "is dominated by a constant bias."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

INPUT_DEPENDENT_THRESHOLD = 0.6


@torch.no_grad()
def _collect_mlp_outputs(
    model,
    layer: int,
    prompts: list,
) -> torch.Tensor:
    """Collect MLP output activations across all positions and prompts.

    Returns: (total_tokens, d_model) tensor of MLP outputs.
    """
    hook_name = f"blocks.{layer}.hook_mlp_out"
    all_outputs = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        mlp_out = cache[hook_name][0]  # (seq, d_model)
        all_outputs.append(mlp_out)

    if not all_outputs:
        return torch.zeros(0, model.cfg.d_model)

    return torch.cat(all_outputs, dim=0)  # (total_tokens, d_model)


@torch.no_grad()
def _compute_decomposition_fraction(
    mlp_outputs: torch.Tensor,
) -> dict:
    """Compute the input-dependent fraction of MLP output variance.

    Decomposes MLP output into:
      - Input-invariant: mean output across all positions.
      - Input-dependent: residual (output - mean).

    Returns dict with variance fractions and diagnostics.
    """
    if mlp_outputs.shape[0] < 2:
        return {
            "input_dependent_fraction": 0.0,
            "total_variance": 0.0,
            "input_dependent_variance": 0.0,
            "input_invariant_norm": 0.0,
            "n_positions": mlp_outputs.shape[0],
        }

    # Input-invariant term: mean MLP output
    mean_output = mlp_outputs.mean(dim=0, keepdim=True)  # (1, d_model)

    # Input-dependent term: residual
    residual = mlp_outputs - mean_output  # (total_tokens, d_model)

    # Variance of total output (across positions, summed over dimensions)
    total_var = mlp_outputs.var(dim=0).sum().item()

    # Variance of input-dependent term
    dep_var = residual.var(dim=0).sum().item()

    # Input-dependent fraction
    if total_var < 1e-10:
        fraction = 0.0
    else:
        fraction = dep_var / total_var

    return {
        "input_dependent_fraction": fraction,
        "total_variance": total_var,
        "input_dependent_variance": dep_var,
        "input_invariant_norm": mean_output.norm().item(),
        "n_positions": mlp_outputs.shape[0],
    }


@torch.no_grad()
def _compute_per_neuron_decomposition(
    model,
    layer: int,
    prompts: list,
    top_k: int = 20,
) -> dict:
    """Compute decomposition at the individual neuron (proxy feature) level.

    For the top-k most variable MLP neurons, measures what fraction
    of each neuron's output contribution is input-dependent.

    Returns dict with per-neuron fractions and aggregate.
    """
    hook_post = f"blocks.{layer}.mlp.hook_post"
    all_post = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_post])
        post = cache[hook_post][0]  # (seq, d_mlp)
        all_post.append(post)

    if not all_post:
        return {"per_neuron_mean_fraction": 0.0, "n_neurons": 0}

    all_post = torch.cat(all_post, dim=0)  # (total_tokens, d_mlp)

    # Find top-k most variable neurons
    neuron_vars = all_post.var(dim=0)  # (d_mlp,)
    k = min(top_k, neuron_vars.shape[0])
    top_indices = torch.topk(neuron_vars, k).indices

    fractions = []
    for idx in top_indices:
        neuron_acts = all_post[:, idx]  # (total_tokens,)
        mean_act = neuron_acts.mean()
        residual = neuron_acts - mean_act
        total_var = neuron_acts.var().item()
        dep_var = residual.var().item()
        if total_var > 1e-10:
            fractions.append(dep_var / total_var)

    mean_fraction = float(np.mean(fractions)) if fractions else 0.0
    return {
        "per_neuron_mean_fraction": mean_fraction,
        "n_neurons": len(fractions),
        "top_neuron_indices": top_indices.tolist(),
    }


def run_transcoder_decomposition(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 50,
    top_k_neurons: int = 20,
) -> list[EvalResult]:
    """Compute input-dependent decomposition fraction across tasks.

    For each task, measures what fraction of MLP output variance is
    captured by input-dependent features vs the input-invariant bias.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names (default: CIRCUIT_TASKS).
        n_prompts: number of prompts per task.
        top_k_neurons: number of top neurons for per-neuron analysis.

    Returns:
        List of EvalResult, one per task plus an aggregate.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    n_layers = model.cfg.n_layers
    results = []
    all_fractions = []

    # Evaluate across multiple layers
    layers_to_check = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    layers_to_check = [l for l in layers_to_check if 0 <= l < n_layers]
    if not layers_to_check:
        layers_to_check = [0]

    log(f"  Transcoder decomposition at layers {layers_to_check}")
    log(f"  n_prompts={n_prompts}")

    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        if not prompts:
            log(f"    {task}: no prompts, skipping")
            continue

        task_fractions = []
        layer_details = {}

        for layer in layers_to_check:
            try:
                mlp_outputs = _collect_mlp_outputs(model, layer, prompts)
                decomp = _compute_decomposition_fraction(mlp_outputs)
                neuron_decomp = _compute_per_neuron_decomposition(
                    model, layer, prompts, top_k_neurons
                )

                task_fractions.append(decomp["input_dependent_fraction"])
                layer_details[f"layer_{layer}"] = {
                    **decomp,
                    **neuron_decomp,
                }
            except Exception as e:
                log(f"    {task} layer {layer}: error {e}")
                continue

        if not task_fractions:
            log(f"    {task}: no valid results")
            continue

        mean_fraction = float(np.mean(task_fractions))
        passed = mean_fraction > INPUT_DEPENDENT_THRESHOLD
        all_fractions.append(mean_fraction)

        log(f"    {task}: input_dep_frac={mean_fraction:.4f} "
            f"({'PASS' if passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.transcoder_decomposition",
            value=mean_fraction,
            n_samples=len(prompts),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "input_dependent_fraction": mean_fraction,
                "layers_evaluated": layers_to_check,
                "n_prompts": len(prompts),
                "passed": passed,
                "threshold": INPUT_DEPENDENT_THRESHOLD,
                "per_layer": layer_details,
            },
        ))

    # Aggregate result across all tasks
    if all_fractions:
        agg_mean = float(np.mean(all_fractions))
        agg_std = float(np.std(all_fractions))
        agg_passed = agg_mean > INPUT_DEPENDENT_THRESHOLD
        log(f"  Aggregate: input_dep_frac={agg_mean:.4f} +/- {agg_std:.4f} "
            f"({'PASS' if agg_passed else 'FAIL'})")

        results.append(EvalResult(
            metric_id="EX34.transcoder_decomposition",
            value=agg_mean,
            n_samples=sum(r.n_samples for r in results),
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": "aggregate",
                "input_dependent_fraction": agg_mean,
                "fraction_std": agg_std,
                "n_tasks_evaluated": len(all_fractions),
                "per_task_fractions": {
                    r.metadata["task"]: r.metadata["input_dependent_fraction"]
                    for r in results if r.metadata.get("task") != "aggregate"
                },
                "layers_evaluated": layers_to_check,
                "passed": agg_passed,
                "threshold": INPUT_DEPENDENT_THRESHOLD,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX34: Transcoder Decomposition Fraction")
    parser.add_argument("--top-k-neurons", type=int, default=20,
                        help="Top-k neurons for per-neuron analysis (default: 20)")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX34: TRANSCODER DECOMPOSITION FRACTION")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_transcoder_decomposition(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        top_k_neurons=args.top_k_neurons,
    )

    out = args.out or "141_transcoder_decomposition.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
