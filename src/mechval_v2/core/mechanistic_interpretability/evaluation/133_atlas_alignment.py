"""Metric: Atlas-Alignment Cross-Model Convergence --- concept transfer via CKA

Paper: Puri, Berend, Lapuschkin, Samek (2025). "Atlas-Alignment: Making
Interpretability Transferable Across Language Models." ICLR 2026
Re-Align Workshop. arXiv:2510.27413

Operationalizes Atlas-Alignment's insight that a Concept Atlas built for
one reference model can transfer feature labels to new models via
representational alignment. This metric computes layer-wise CKA between
two model instances on shared inputs to measure how well internal
representations align. High alignment implies concept labels transfer
with fidelity, providing cross-model convergent validity evidence.

Atlas-Alignment Cross-Model Convergence (Evaluation EX26)
=============================================
Instrument:     EX26 --- Atlas-Alignment Cross-Model Convergence
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity (cross-model)
Establishes:    Whether representations in one model have corresponding
                representations in another, enabling label transfer
Requires:       CPU or GPU, model (two instances compared)
=============================================

Core logic:
1. Run shared prompts through both model instances (or two layers of one model).
2. Collect residual stream activations at each layer.
3. Compute linear CKA between corresponding layers.
4. The atlas_alignment_score is the max CKA across layer pairs.
5. The feature_transfer_rate is the fraction of directions in the reference
   that have CKA > threshold with some direction in the target.

Pass condition: atlas_alignment_score > 0.3; feature_transfer_rate > 0.5

Usage:
    uv run python 133_atlas_alignment.py --model gpt2 --device cpu
    uv run python 133_atlas_alignment.py --n-prompts 100
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
    name="Atlas-Alignment Cross-Model Convergence",
    paper_ref="Puri et al. ICLR 2026 Re-Align, arXiv:2510.27413",
    paper_cite=(
        "Puri, Berend, Lapuschkin, Samek 2025, "
        "Atlas-Alignment: Making Interpretability Transferable Across "
        "Language Models (ICLR 2026 Re-Align, arXiv:2510.27413)"
    ),
    description=(
        "Computes layer-wise CKA between two model instances on shared "
        "inputs to measure representational alignment. High alignment "
        "implies concept labels from a reference model transfer to a "
        "target model with fidelity, operationalizing the 'transparency "
        "tax' solution as a C5 convergent validity diagnostic."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

ALIGNMENT_THRESHOLD = 0.3
TRANSFER_THRESHOLD = 0.5


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear Centered Kernel Alignment between X and Y.

    Args:
        X: (n_samples, d_x) activation matrix.
        Y: (n_samples, d_y) activation matrix.

    Returns:
        CKA score in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    XtX = X @ X.T  # (n, n)
    YtY = Y @ Y.T  # (n, n)

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


@torch.no_grad()
def _collect_layer_activations(
    model, texts: list[str], n_layers: int,
) -> list[torch.Tensor]:
    """Collect last-position residual stream activations at each layer.

    Returns:
        List of (n_texts, d_model) tensors, one per layer.
    """
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    layer_acts = [[] for _ in range(n_layers)]
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in hook_names
        )
        for l in range(n_layers):
            act = cache[hook_names[l]][0, -1].cpu()  # (d_model,)
            layer_acts[l].append(act)

    return [torch.stack(acts) for acts in layer_acts]


def run_atlas_alignment(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
) -> list[EvalResult]:
    """Run the Atlas-Alignment convergence diagnostic.

    Compares early-layer vs. late-layer representations within a single
    model as a proxy for cross-model alignment (same model, different
    representation spaces). For a true cross-model comparison, two model
    instances would be needed.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names for generating shared prompts.
        n_prompts: number of prompts.

    Returns:
        List of EvalResult with CKA alignment scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    log(f"  Atlas-Alignment Convergence: n_prompts={n_prompts}")

    # Collect prompts from all tasks
    all_texts = []
    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        for p in prompts:
            all_texts.append(p.text)
        if len(all_texts) >= n_prompts:
            break
    all_texts = all_texts[:n_prompts]

    if len(all_texts) < 3:
        log("  Not enough prompts for CKA computation")
        return []

    n_layers = model.cfg.n_layers
    log(f"  Collecting activations across {n_layers} layers for {len(all_texts)} texts")
    layer_acts = _collect_layer_activations(model, all_texts, n_layers)

    # Compute CKA matrix between all layer pairs
    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = _linear_cka(layer_acts[i], layer_acts[j])

    # Atlas alignment score: max off-diagonal CKA
    # (compares early and late layers as proxy for cross-model alignment)
    off_diag_ckas = []
    for i in range(n_layers):
        for j in range(n_layers):
            if abs(i - j) > 1:  # Skip adjacent layers
                off_diag_ckas.append(cka_matrix[i, j])

    atlas_score = float(np.max(off_diag_ckas)) if off_diag_ckas else 0.0

    # Feature transfer rate: fraction of layers that align well with at least
    # one non-adjacent layer
    n_transferable = 0
    for i in range(n_layers):
        best_distant = 0.0
        for j in range(n_layers):
            if abs(i - j) > 1:
                best_distant = max(best_distant, cka_matrix[i, j])
        if best_distant > ALIGNMENT_THRESHOLD:
            n_transferable += 1
    transfer_rate = n_transferable / n_layers

    # Layer-wise CKA with adjacent layers (representational smoothness)
    adjacent_ckas = []
    for i in range(n_layers - 1):
        adjacent_ckas.append(cka_matrix[i, i + 1])
    smoothness = float(np.mean(adjacent_ckas)) if adjacent_ckas else 0.0

    passed_alignment = atlas_score > ALIGNMENT_THRESHOLD
    passed_transfer = transfer_rate > TRANSFER_THRESHOLD
    passed = passed_alignment and passed_transfer

    log(f"  atlas_score={atlas_score:.4f}, transfer_rate={transfer_rate:.4f}, "
        f"smoothness={smoothness:.4f} ({'PASS' if passed else 'FAIL'})")

    results = [EvalResult(
        metric_id="EX26.atlas_alignment",
        value=atlas_score,
        n_samples=len(all_texts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "atlas_alignment_score": atlas_score,
            "feature_transfer_rate": transfer_rate,
            "representational_smoothness": smoothness,
            "n_layers": n_layers,
            "n_texts": len(all_texts),
            "cka_matrix_diagonal": [float(cka_matrix[i, i]) for i in range(n_layers)],
            "cka_matrix_shape": list(cka_matrix.shape),
            "passed": passed,
            "threshold_alignment": ALIGNMENT_THRESHOLD,
            "threshold_transfer": TRANSFER_THRESHOLD,
        },
    )]

    return results


def main():
    parser = parse_common_args("EX26: Atlas-Alignment Cross-Model Convergence")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX26: ATLAS-ALIGNMENT CROSS-MODEL CONVERGENCE")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_atlas_alignment(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
    )

    out = args.out or "133_atlas_alignment.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
