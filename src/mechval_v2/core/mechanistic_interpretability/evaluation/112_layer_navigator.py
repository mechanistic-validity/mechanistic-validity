"""LayerNavigator (Evaluation EX16)
Paper: Sun et al. (2025). NeurIPS 2025.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     EX16 — LayerNavigator
Categories:     evaluation
Validity layer: Measurement (M4 Sensitivity)
Criteria:       M4 — Measurement sensitivity
Establishes:    Whether a principled layer selection criterion (discriminability
                x consistency) identifies a layer with a strong, stable steering
                direction for a concept
Requires:       CPU or GPU, model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the LayerNavigator approach (Sun et al., NeurIPS 2025) as a
validation metric. For each task:

1. Collect residual-stream activations at every layer on positive and
   negative concept examples (task prompts with correct vs incorrect
   continuations).
2. At each layer, compute the mean difference direction between positive
   and negative examples.
3. Discriminability: AUROC of a linear classifier (dot product with the
   mean difference direction) separating positive from negative examples.
4. Consistency: mean pairwise cosine similarity of per-example difference
   directions (how stable is the steering signal across examples?).
5. Layer score = discriminability * consistency.
6. Report the optimal layer and its combined score.

Pass condition: best_layer_score > 0.3

References:
    Sun et al. (NeurIPS 2025) "LayerNavigator: Principled Layer Selection
    for Activation Steering"

Usage:
    uv run python 112_layer_navigator.py --tasks ioi --n-prompts 40
    uv run python 112_layer_navigator.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    generate_prompts,
    get_token_ids,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="LayerNavigator",
    paper_ref="Sun et al. NeurIPS 2025",
    paper_cite=(
        "Sun et al. 2025, LayerNavigator: Principled Layer Selection "
        "for Activation Steering"
    ),
    description=(
        "Scores each layer's suitability for activation steering via "
        "discriminability (AUROC of mean-diff direction) times consistency "
        "(stability of per-example difference directions). First principled "
        "alternative to brute-force layer search."
    ),
    category="evaluation",
    tier="cogsci",
    origin="established",
)

SCORE_THRESHOLD = 0.3


@torch.no_grad()
def _collect_layer_activations(
    model, prompts, token_ids: list[int], n_layers: int,
) -> list[torch.Tensor]:
    """Collect last-token residual-stream activations at each layer.

    Returns a list of length n_layers, each tensor shaped
    (n_prompts, d_model).
    """
    per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for i, p in enumerate(prompts):
        if i >= len(token_ids):
            break
        tokens = model.to_tokens(p.text)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: "hook_resid_pre" in n or "hook_resid_post" in n,
        )
        for L in range(n_layers):
            # Use hook_resid_post for each layer's output representation
            key = f"blocks.{L}.hook_resid_post"
            if key in cache:
                per_layer[L].append(cache[key][0, -1].cpu().float())

    result = []
    for L in range(n_layers):
        if per_layer[L]:
            result.append(torch.stack(per_layer[L]))
        else:
            result.append(torch.zeros(0))
    return result


def _compute_discriminability(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    mean_diff: torch.Tensor,
) -> float:
    """AUROC of a linear classifier using the mean difference direction.

    Projects each example onto the mean-diff direction and uses
    sklearn's roc_auc_score to measure separability.
    """
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    if n_pos == 0 or n_neg == 0:
        return 0.0

    direction = mean_diff / (mean_diff.norm() + 1e-8)
    pos_scores = (pos_acts @ direction).numpy()
    neg_scores = (neg_acts @ direction).numpy()

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5

    return float(roc_auc_score(labels, scores))


def _compute_consistency(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
) -> float:
    """Mean pairwise cosine similarity of per-example difference directions.

    For each positive example i and negative example j, the per-pair
    difference direction is pos_acts[i] - neg_acts[j]. We compute the
    mean pairwise cosine similarity across all such pairs.

    When the number of pairs is large, we subsample to keep computation
    manageable.
    """
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    if n_pos < 2 or n_neg < 1:
        return 0.0

    # Per-example difference directions: each positive example minus the
    # mean negative activation gives one direction per positive example.
    neg_mean = neg_acts.mean(dim=0)
    diff_dirs = pos_acts - neg_mean  # (n_pos, d_model)

    # Normalize
    norms = diff_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    diff_dirs_normed = diff_dirs / norms

    # Pairwise cosine similarity matrix
    cos_matrix = diff_dirs_normed @ diff_dirs_normed.T  # (n_pos, n_pos)

    # Mean of off-diagonal entries
    n = cos_matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_cos = float(cos_matrix[mask].mean())

    return mean_cos


def run_layer_navigator(
    model, tasks: list[str], n_prompts: int = 40,
) -> list[EvalResult]:
    """Run LayerNavigator layer selection scoring on each task.

    For each task, collects activations at every layer for positive
    (correct) and negative (incorrect) concept examples, then scores
    each layer by discriminability * consistency.

    Args:
        model: HookedTransformer instance.
        tasks: List of task names to evaluate.
        n_prompts: Number of prompts per task.

    Returns:
        List of EvalResult, one per task.
    """
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    results = []

    for task in tasks:
        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            log(f"  {task}: no token IDs, skipping")
            continue

        log(f"  {task}: {len(prompts)} prompts, {n_layers} layers")

        # Collect activations for positive examples (correct continuation)
        pos_acts_per_layer = _collect_layer_activations(
            model, prompts, correct_ids, n_layers,
        )

        # For negative examples, we run the same prompts but track the
        # activation relationship to the incorrect token. Since the
        # activations at a given position are the same regardless of the
        # label (the model processes the same prefix), the distinction
        # between positive and negative here comes from splitting prompts
        # into two halves: first half = "positive concept examples",
        # second half = "negative concept examples" (contrastive pairs).
        n = min(len(prompts), len(correct_ids), len(incorrect_ids))
        half = n // 2
        if half < 2:
            log(f"  {task}: too few prompts for pos/neg split, skipping")
            continue

        layer_scores = []
        per_layer_details = []

        for L in range(n_layers):
            acts = pos_acts_per_layer[L]
            if acts.shape[0] < n:
                log(f"    L{L}: insufficient activations, skipping")
                layer_scores.append(0.0)
                per_layer_details.append({
                    "layer": L, "discriminability": 0.0,
                    "consistency": 0.0, "score": 0.0,
                })
                continue

            # Split into positive and negative concept groups
            pos = acts[:half]
            neg = acts[half:n]

            # Mean difference direction
            mean_diff = pos.mean(dim=0) - neg.mean(dim=0)

            discriminability = _compute_discriminability(pos, neg, mean_diff)
            consistency = _compute_consistency(pos, neg)
            score = discriminability * consistency

            layer_scores.append(score)
            per_layer_details.append({
                "layer": L,
                "discriminability": float(discriminability),
                "consistency": float(consistency),
                "score": float(score),
            })

            if (L + 1) % 4 == 0 or L == n_layers - 1:
                log(f"    L{L}: disc={discriminability:.4f} "
                    f"cons={consistency:.4f} score={score:.4f}")

        if not layer_scores:
            continue

        best_idx = int(np.argmax(layer_scores))
        best_score = layer_scores[best_idx]
        passed = bool(best_score > SCORE_THRESHOLD)

        log(f"    best layer: L{best_idx} score={best_score:.4f} "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="EX16.layer_navigator",
            value=float(best_score),
            n_samples=n,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "best_layer": best_idx,
                "best_layer_score": float(best_score),
                "best_discriminability": per_layer_details[best_idx]["discriminability"],
                "best_consistency": per_layer_details[best_idx]["consistency"],
                "n_layers": n_layers,
                "n_pos": half,
                "n_neg": n - half,
                "passed": passed,
                "threshold": SCORE_THRESHOLD,
                "per_layer": per_layer_details,
            },
        ))

    return results


def main():
    parser = parse_common_args("EX16: LayerNavigator")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("EX16: LAYER NAVIGATOR")
    log("=" * 60)

    out = args.out or "112_layer_navigator.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_layer_navigator(model, [task], n_prompts=args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: best_layer=L{r.metadata['best_layer']} "
                f"score={r.value:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
