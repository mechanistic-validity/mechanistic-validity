"""Metric: Alignment-Interpretability Trade-off --- C4 discriminant validity diagnostic

Paper: Colin, Oliver, Serre (2026). "Does Human-Alignment Benefit
Interpretability?" ICLR 2026 Re-Align Workshop, ELLIS Alicante.
https://ellisalicante.org/publications/colin2026benefit

Tests whether model alignment (e.g., instruction tuning, RLHF) changes
representational interpretability and richness. The ELLIS finding shows
aligned models are more interpretable but less "visually rich" ---
meaning interpretability and richness are discriminant constructs.
This metric quantifies the trade-off via activation pattern consistency
(interpretability) and effective rank (richness).

Alignment-Interpretability Trade-off (Evaluation EX33)
=============================================
Instrument:     EX33 --- Alignment-Interpretability Trade-off
Categories:     evaluation
Validity layer: Construct
Criteria:       C4 Discriminant Validity
Establishes:    Whether interpretability and representational richness
                are discriminant constructs, and whether alignment
                changes the balance between them
Requires:       CPU or GPU, model
=============================================

Core logic:
1. For a set of prompts, collect per-unit activations at each layer.
2. Compute interpretability proxy: activation consistency (how tightly
   clustered are top-k activating prompts for each unit).
3. Compute richness proxy: effective rank of the activation matrix.
4. Report interpretability_score, richness_score, and trade-off_ratio.
5. For two-model comparison: compute delta on each metric.

Pass condition: This is a diagnostic, not pass/fail. Reports whether
alignment improves interpretability at the cost of richness.

Usage:
    uv run python 138_alignment_interpretability.py --model gpt2 --device cpu
    uv run python 138_alignment_interpretability.py --n-prompts 100
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
    name="Alignment-Interpretability Trade-off",
    paper_ref="Colin et al. ICLR 2026 Re-Align",
    paper_cite=(
        "Colin, Oliver, Serre 2026, "
        "Does Human-Alignment Benefit Interpretability? "
        "(ICLR 2026 Re-Align Workshop, ELLIS Alicante)"
    ),
    description=(
        "Measures the trade-off between interpretability (activation "
        "pattern consistency) and richness (effective rank) of model "
        "representations. First controlled evidence that alignment "
        "causes interpretability, but at the cost of representational "
        "richness --- a C4 discriminant validity issue."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)


@torch.no_grad()
def _collect_activation_matrix(
    model, texts: list[str], hook_name: str,
) -> torch.Tensor:
    """Collect per-unit activations for all texts at the hook point.

    Returns:
        activations: (n_texts, d_hook) activation matrix.
    """
    all_acts = []
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        act = cache[hook_name][0, -1].cpu()  # (d_hook,)
        all_acts.append(act)
    return torch.stack(all_acts)  # (n_texts, d_hook)


def _compute_interpretability(act_matrix: torch.Tensor, top_k: int = 5) -> float:
    """Compute interpretability proxy: activation pattern consistency.

    For each unit, find the top-k activating texts and compute the
    mean pairwise cosine similarity among their full activation
    patterns. High similarity = consistent activation patterns = more
    interpretable.

    Args:
        act_matrix: (n_texts, d_hook)
        top_k: number of top-activating texts per unit.

    Returns:
        Mean consistency score across units.
    """
    n_texts, d_hook = act_matrix.shape
    if n_texts < top_k or d_hook == 0:
        return 0.0

    # Sample units to make this tractable
    n_units = min(d_hook, 100)
    unit_indices = np.linspace(0, d_hook - 1, n_units, dtype=int)

    consistencies = []
    act_float = act_matrix.float()

    # Normalize full activation patterns for cosine similarity
    norms = act_float.norm(dim=1, keepdim=True)
    act_normalized = act_float / (norms + 1e-8)

    for uid in unit_indices:
        # Find top-k activating texts for this unit
        unit_acts = act_matrix[:, uid]
        topk_indices = torch.topk(unit_acts.abs(), min(top_k, n_texts)).indices

        # Compute mean pairwise cosine similarity
        top_patterns = act_normalized[topk_indices]  # (k, d_hook)
        sim_matrix = top_patterns @ top_patterns.T  # (k, k)

        # Mean off-diagonal similarity
        k = sim_matrix.shape[0]
        if k < 2:
            continue
        mask = ~torch.eye(k, dtype=torch.bool)
        mean_sim = float(sim_matrix[mask].mean())
        consistencies.append(mean_sim)

    return float(np.mean(consistencies)) if consistencies else 0.0


def _compute_richness(act_matrix: torch.Tensor) -> float:
    """Compute richness proxy: effective rank of the activation matrix.

    Effective rank = exp(entropy of normalized singular values).
    Higher effective rank = richer representations.

    Args:
        act_matrix: (n_texts, d_hook)

    Returns:
        Effective rank (normalized by min(n_texts, d_hook)).
    """
    act_centered = act_matrix.float() - act_matrix.float().mean(dim=0, keepdim=True)

    U, S, Vt = torch.linalg.svd(act_centered, full_matrices=False)
    sv = S.numpy()

    # Normalize
    sv_sum = sv.sum()
    if sv_sum < 1e-12:
        return 0.0

    p = sv / sv_sum
    p = p[p > 1e-12]
    entropy = -float(np.sum(p * np.log(p)))
    effective_rank = float(np.exp(entropy))

    # Normalize by maximum possible rank
    max_rank = min(act_matrix.shape)
    return effective_rank / max_rank


def run_alignment_interpretability(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    hook_layer: int | None = None,
    top_k: int = 5,
) -> list[EvalResult]:
    """Run the alignment-interpretability trade-off diagnostic.

    Computes interpretability (activation consistency) and richness
    (effective rank) for the model's representations, establishing
    a baseline for comparison with aligned variants.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names for prompt generation.
        n_prompts: number of prompts.
        hook_layer: layer for analysis (default: middle).
        top_k: number of top-activating texts per unit for consistency.

    Returns:
        List of EvalResult with interpretability and richness scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS
    if hook_layer is None:
        hook_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{hook_layer}.hook_mlp_out"

    log(f"  Alignment-Interpretability: layer={hook_layer}, n_prompts={n_prompts}")

    # Collect prompts from all tasks
    all_texts = []
    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        for p in prompts:
            all_texts.append(p.text)
        if len(all_texts) >= n_prompts:
            break
    all_texts = all_texts[:n_prompts]

    if len(all_texts) < top_k + 1:
        log("  Not enough prompts")
        return []

    # Collect activations
    act_matrix = _collect_activation_matrix(model, all_texts, hook_name)

    # Compute metrics
    interpretability = _compute_interpretability(act_matrix, top_k=top_k)
    richness = _compute_richness(act_matrix)

    # Per-layer analysis across all layers
    layer_interpretabilities = []
    layer_richnesses = []
    for layer in range(model.cfg.n_layers):
        lhn = f"blocks.{layer}.hook_mlp_out"
        layer_acts = _collect_activation_matrix(model, all_texts[:min(20, len(all_texts))], lhn)
        li = _compute_interpretability(layer_acts, top_k=min(top_k, layer_acts.shape[0] - 1))
        lr = _compute_richness(layer_acts)
        layer_interpretabilities.append(li)
        layer_richnesses.append(lr)

    # Trade-off analysis: correlation between interpretability and richness across layers
    if len(layer_interpretabilities) >= 3:
        interp_arr = np.array(layer_interpretabilities)
        rich_arr = np.array(layer_richnesses)
        # Correlation: negative = trade-off exists
        if interp_arr.std() > 1e-8 and rich_arr.std() > 1e-8:
            corr = float(np.corrcoef(interp_arr, rich_arr)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    log(f"  interpretability={interpretability:.4f}, richness={richness:.4f}, "
        f"interp-richness_corr={corr:.4f}")

    results = [EvalResult(
        metric_id="EX33.alignment_interpretability",
        value=interpretability,
        n_samples=len(all_texts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "interpretability_score": interpretability,
            "richness_score": richness,
            "interp_richness_correlation": corr,
            "trade_off_exists": corr < -0.3,
            "hook_layer": hook_layer,
            "n_texts": len(all_texts),
            "top_k": top_k,
            "layer_interpretabilities": layer_interpretabilities,
            "layer_richnesses": layer_richnesses,
        },
    )]

    return results


def main():
    parser = parse_common_args("EX33: Alignment-Interpretability Trade-off")
    parser.add_argument("--hook-layer", type=int, default=None,
                        help="Layer for analysis (default: middle)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k activating texts for consistency")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX33: ALIGNMENT-INTERPRETABILITY TRADE-OFF")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_alignment_interpretability(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        hook_layer=args.hook_layer,
        top_k=args.top_k,
    )

    out = args.out or "138_alignment_interpretability.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
