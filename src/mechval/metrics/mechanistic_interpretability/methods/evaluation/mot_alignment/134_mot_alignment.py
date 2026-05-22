"""Metric: MOT Global Alignment Score --- optimal transport over representations

Paper: Shah, Khosla (2025). "Representational Alignment Across Model
Layers and Brain Regions with Multi-Level Optimal Transport." ICLR 2026
Poster. arXiv:2510.01706

Computes a global alignment score between two sets of representations
using optimal transport over CKA similarity. Unlike greedy layer-wise
CKA, MOT distributes representation mass across multiple target layers,
yielding a single globally consistent score that handles depth mismatches.
The soft layer-to-layer couplings reveal hierarchical correspondences
between networks.

MOT Global Alignment Score (Evaluation EX27)
=============================================
Instrument:     EX27 --- MOT Global Alignment Score
Categories:     evaluation
Validity layer: Construct
Criteria:       C5 Convergent Validity (cross-architecture)
Establishes:    Whether two networks have globally consistent
                representational alignment, accounting for depth mismatch
Requires:       CPU or GPU, model
=============================================

Core logic:
1. Collect layer-wise activations from two sets of layers (or two models).
2. Compute pairwise CKA similarity between all layer pairs -> cost matrix.
3. Solve optimal transport (Sinkhorn) to find soft coupling that minimizes
   total transport cost.
4. The MOT global score = total transported similarity.
5. Report coupling matrix and hierarchical correspondence.

Pass condition: mot_global_score > 0.3

Usage:
    uv run python 134_mot_alignment.py --model gpt2 --device cpu
    uv run python 134_mot_alignment.py --n-prompts 50 --sinkhorn-reg 0.1
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
    name="MOT Global Alignment Score",
    paper_ref="Shah, Khosla, ICLR 2026, arXiv:2510.01706",
    paper_cite=(
        "Shah, Khosla 2025, "
        "Representational Alignment Across Model Layers and Brain "
        "Regions with Multi-Level Optimal Transport "
        "(ICLR 2026, arXiv:2510.01706)"
    ),
    description=(
        "Computes a global alignment score between two sets of "
        "representations using optimal transport over CKA similarity. "
        "Handles depth mismatches natively via mass distribution and "
        "produces a single globally consistent comparison."
    ),
    category="evaluation",
    tier="established",
    origin="external",
)

MOT_THRESHOLD = 0.3


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtX = X @ X.T
    YtY = Y @ Y.T
    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def _sinkhorn(
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    n_iter: int = 100,
) -> np.ndarray:
    """Solve entropy-regularized optimal transport via Sinkhorn iterations.

    Args:
        cost_matrix: (n_source, n_target) cost matrix (lower = better).
        reg: regularization strength (higher = smoother coupling).
        n_iter: number of Sinkhorn iterations.

    Returns:
        coupling: (n_source, n_target) transport plan.
    """
    n_s, n_t = cost_matrix.shape
    # Uniform marginals
    mu = np.ones(n_s) / n_s
    nu = np.ones(n_t) / n_t

    # Gibbs kernel
    K = np.exp(-cost_matrix / max(reg, 1e-8))
    K = np.clip(K, 1e-30, None)

    u = np.ones(n_s)
    v = np.ones(n_t)

    for _ in range(n_iter):
        u = mu / (K @ v + 1e-30)
        v = nu / (K.T @ u + 1e-30)

    coupling = np.diag(u) @ K @ np.diag(v)
    return coupling


@torch.no_grad()
def _collect_layer_activations(
    model, texts: list[str], n_layers: int,
) -> list[torch.Tensor]:
    """Collect last-position residual stream activations at each layer."""
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    layer_acts = [[] for _ in range(n_layers)]
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in hook_names
        )
        for l in range(n_layers):
            act = cache[hook_names[l]][0, -1].cpu()
            layer_acts[l].append(act)
    return [torch.stack(acts) for acts in layer_acts]


def run_mot_alignment(
    model,
    tasks: list[str] | None = None,
    n_prompts: int = 40,
    sinkhorn_reg: float = 0.1,
    sinkhorn_iter: int = 100,
) -> list[EvalResult]:
    """Run the MOT global alignment diagnostic.

    Compares first-half and second-half of layers within a single model
    as a proxy for cross-architecture alignment. For a true cross-model
    comparison, two model instances would be provided.

    Args:
        model: HookedTransformer instance.
        tasks: list of task names.
        n_prompts: number of prompts.
        sinkhorn_reg: Sinkhorn regularization parameter.
        sinkhorn_iter: number of Sinkhorn iterations.

    Returns:
        List of EvalResult with MOT alignment scores.
    """
    if tasks is None:
        tasks = CIRCUIT_TASKS

    log(f"  MOT Alignment: n_prompts={n_prompts}, reg={sinkhorn_reg}")

    # Collect prompts
    all_texts = []
    for task in tasks:
        prompts = generate_prompts(task, model.tokenizer, n_prompts=n_prompts)
        for p in prompts:
            all_texts.append(p.text)
        if len(all_texts) >= n_prompts:
            break
    all_texts = all_texts[:n_prompts]

    if len(all_texts) < 3:
        log("  Not enough prompts")
        return []

    n_layers = model.cfg.n_layers
    log(f"  Collecting activations across {n_layers} layers")
    layer_acts = _collect_layer_activations(model, all_texts, n_layers)

    # Split into source (first half) and target (second half) layers
    mid = n_layers // 2
    source_layers = list(range(mid))
    target_layers = list(range(mid, n_layers))

    n_s = len(source_layers)
    n_t = len(target_layers)

    # Compute CKA similarity matrix
    similarity = np.zeros((n_s, n_t))
    for i, si in enumerate(source_layers):
        for j, tj in enumerate(target_layers):
            similarity[i, j] = _linear_cka(layer_acts[si], layer_acts[tj])

    # Convert similarity to cost (1 - similarity)
    cost = 1.0 - similarity

    # Solve optimal transport
    coupling = _sinkhorn(cost, reg=sinkhorn_reg, n_iter=sinkhorn_iter)

    # MOT global score = total transported similarity
    mot_score = float(np.sum(coupling * similarity))

    # Normalized by number of source layers for interpretability
    mot_normalized = mot_score * n_s

    # Coupling sparsity: how many target layers each source layer couples to
    coupling_entropy = 0.0
    for i in range(n_s):
        row = coupling[i]
        row_sum = row.sum()
        if row_sum > 1e-12:
            p = row / row_sum
            p = p[p > 1e-12]
            coupling_entropy -= float(np.sum(p * np.log(p)))
    coupling_entropy /= max(n_s, 1)

    # Hierarchical ordering: check if coupling preserves relative position
    ordering_score = 0.0
    n_ordered = 0
    for i in range(n_s):
        peak_j = int(np.argmax(coupling[i]))
        for i2 in range(i + 1, n_s):
            peak_j2 = int(np.argmax(coupling[i2]))
            n_ordered += 1
            if peak_j2 >= peak_j:
                ordering_score += 1.0
    ordering_score = ordering_score / max(n_ordered, 1)

    passed = mot_normalized > MOT_THRESHOLD

    log(f"  mot_score={mot_normalized:.4f}, coupling_entropy={coupling_entropy:.4f}, "
        f"ordering={ordering_score:.4f} ({'PASS' if passed else 'FAIL'})")

    results = [EvalResult(
        metric_id="EX27.mot_alignment",
        value=mot_normalized,
        n_samples=len(all_texts),
        instrument_info=INSTRUMENT_INFO,
        metadata={
            "mot_global_score": mot_normalized,
            "mot_raw_score": mot_score,
            "coupling_entropy": coupling_entropy,
            "hierarchical_ordering": ordering_score,
            "n_source_layers": n_s,
            "n_target_layers": n_t,
            "n_texts": len(all_texts),
            "sinkhorn_reg": sinkhorn_reg,
            "similarity_matrix_mean": float(np.mean(similarity)),
            "similarity_matrix_max": float(np.max(similarity)),
            "passed": passed,
            "threshold": MOT_THRESHOLD,
        },
    )]

    return results


def main():
    parser = parse_common_args("EX27: MOT Global Alignment Score")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1,
                        help="Sinkhorn regularization parameter")
    parser.add_argument("--sinkhorn-iter", type=int, default=100,
                        help="Number of Sinkhorn iterations")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX27: MOT GLOBAL ALIGNMENT SCORE")
    log("=" * 60)

    tasks = args.tasks or CIRCUIT_TASKS
    results = run_mot_alignment(
        model,
        tasks=tasks,
        n_prompts=args.n_prompts,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
    )

    out = args.out or "134_mot_alignment.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
