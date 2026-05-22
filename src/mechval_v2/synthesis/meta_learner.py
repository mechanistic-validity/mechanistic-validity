"""Meta-Learner Circuit Predictor (S07)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta-protocol:  Consumes results from other protocols
Categories:     meta, PARCEL Layer 3
Validity layer: Internal + External
Establishes:    Learned fusion weights + cross-task generalization
Requires:       CPU, protocol results + labeled circuits as input
Source:         PARCEL meta-learning architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trains a logistic regression on known labeled circuits (IOI, GT, SVA,
gendered pronoun) using all method scores as features. Learns per-method
importance weights. Evaluates via leave-one-task-out cross-validation.

Reports: per-method importance, AUROC, predicted circuits for unlabeled
tasks, and method redundancy analysis.

Usage:
    uv run python meta_learner.py --results-json modal_sweep_results.json
"""
import json
import time

import numpy as np
from scipy.special import expit

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    get_circuit_heads,
    log,
    parse_common_args,
    save_results,
)

PROTOCOL_ID = "S07"
PROTOCOL_NAME = "Meta-Learner Circuit Predictor (PARCEL L3)"

GPT2_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_HEADS = len(GPT2_HEADS)


def _parse_head_key(key: str) -> tuple[int, int] | None:
    try:
        if key.startswith("L") and "H" in key:
            parts = key[1:].split("H")
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _extract_all_scores(protocol_results: list[dict],
                        task: str) -> dict[str, np.ndarray]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    vectors = {}

    for result in protocol_results:
        if result.get("status") != "success":
            continue
        proto_id = result.get("protocol_id", "unknown")
        scores = np.zeros(N_HEADS)
        found = False
        for mname, evals in result.get("metrics", {}).items():
            for ev in evals:
                meta = ev if isinstance(ev, dict) else {}
                ev_task = meta.get("metadata", {}).get("task", meta.get("task", ""))
                if ev_task != task:
                    continue
                head_scores = meta.get("metadata", {}).get("head_scores", {})
                for hkey, score in head_scores.items():
                    parsed = _parse_head_key(hkey)
                    if parsed and parsed in head_to_idx:
                        val = abs(score) if isinstance(score, (int, float)) else 0.0
                        scores[head_to_idx[parsed]] = max(scores[head_to_idx[parsed]], val)
                        found = True
        if found:
            ranks = np.zeros(N_HEADS)
            order = np.argsort(-scores)
            for rank, idx in enumerate(order):
                ranks[idx] = 1.0 - (rank + 1) / N_HEADS
            vectors[proto_id] = ranks

    return vectors


def _build_dataset(protocol_results: list[dict],
                   tasks: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    head_to_idx = {h: i for i, h in enumerate(GPT2_HEADS)}
    all_methods = set()
    per_task_scores = {}

    for task in tasks:
        scores = _extract_all_scores(protocol_results, task)
        if scores:
            per_task_scores[task] = scores
            all_methods.update(scores.keys())

    method_names = sorted(all_methods)
    if not method_names:
        return np.array([]), np.array([]), [], []

    X_rows = []
    y_rows = []
    task_labels = []

    for task in tasks:
        if task not in per_task_scores:
            continue
        gt_heads = get_circuit_heads(task)
        if not gt_heads:
            continue

        gt_set = set(gt_heads)
        scores = per_task_scores[task]

        for i, head in enumerate(GPT2_HEADS):
            features = [scores.get(m, np.zeros(N_HEADS))[i] for m in method_names]
            X_rows.append(features)
            y_rows.append(1.0 if head in gt_set else 0.0)
            task_labels.append(task)

    return np.array(X_rows), np.array(y_rows), method_names, task_labels


def _logistic_regression_fit(X: np.ndarray, y: np.ndarray,
                             class_weight_balanced: bool = True,
                             lr: float = 0.01, n_iter: int = 500) -> np.ndarray:
    n_features = X.shape[1]
    w = np.zeros(n_features + 1)

    if class_weight_balanced:
        pos_weight = (1 - y.mean()) / (y.mean() + 1e-10)
        sample_weights = np.where(y == 1, pos_weight, 1.0)
    else:
        sample_weights = np.ones(len(y))

    X_aug = np.column_stack([X, np.ones(len(X))])

    for _ in range(n_iter):
        logits = X_aug @ w
        preds = expit(logits)
        grad = X_aug.T @ (sample_weights * (preds - y)) / len(y)
        w -= lr * grad

    return w


def _auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_scores = y_scores[pos]
    neg_scores = y_scores[neg]

    count = 0
    for ps in pos_scores:
        count += (neg_scores < ps).sum()
        count += 0.5 * (neg_scores == ps).sum()

    return float(count / (n_pos * n_neg))


def run_meta_learner(model=None, tasks: list[str] | None = None,
                     device: str = "cpu",
                     protocol_results: list[dict] | None = None) -> list[EvalResult]:
    if tasks is None:
        tasks = list(CIRCUIT_TASKS)
    if protocol_results is None:
        return []

    X, y, method_names, task_labels = _build_dataset(protocol_results, tasks)
    if len(X) == 0 or len(method_names) == 0:
        log("  No labeled data available")
        return []

    log(f"  Dataset: {len(X)} samples, {len(method_names)} methods, "
        f"{int(y.sum())} positive, {int((1-y).sum())} negative")
    log(f"  Methods: {method_names}")

    results = []

    w = _logistic_regression_fit(X, y)
    method_weights = {m: float(w[i]) for i, m in enumerate(method_names)}
    bias = float(w[-1])

    sorted_weights = sorted(method_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    log(f"\n  Method importance (absolute weight):")
    for m, weight in sorted_weights[:10]:
        log(f"    {m:12s}: {weight:+.4f}")

    X_aug = np.column_stack([X, np.ones(len(X))])
    all_preds = expit(X_aug @ w)
    overall_auroc = _auroc(y, all_preds)
    log(f"\n  Overall AUROC: {overall_auroc:.4f}")

    unique_tasks = sorted(set(task_labels))
    loo_aucs = {}
    for held_out in unique_tasks:
        train_mask = np.array([t != held_out for t in task_labels])
        test_mask = ~train_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if y[test_mask].sum() == 0 or (1 - y[test_mask]).sum() == 0:
            continue

        w_loo = _logistic_regression_fit(X[train_mask], y[train_mask])
        X_test_aug = np.column_stack([X[test_mask], np.ones(test_mask.sum())])
        preds_loo = expit(X_test_aug @ w_loo)
        auc = _auroc(y[test_mask], preds_loo)
        loo_aucs[held_out] = auc
        log(f"    LOO {held_out}: AUROC={auc:.4f}")

    mean_loo = np.mean(list(loo_aucs.values())) if loo_aucs else 0.0
    log(f"  Mean LOO AUROC: {mean_loo:.4f}")

    results.append(EvalResult(
        metric_id="S07.overall_auroc",
        value=overall_auroc,
        n_samples=len(X),
        metadata={
            "n_methods": len(method_names),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "method_weights": method_weights,
            "bias": bias,
            "loo_aurocs": loo_aucs,
            "mean_loo_auroc": mean_loo,
        },
    ))

    results.append(EvalResult(
        metric_id="S07.mean_loo_auroc",
        value=mean_loo,
        n_samples=len(loo_aucs),
        metadata={"loo_aurocs": loo_aucs, "tasks": unique_tasks},
    ))

    corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]])
    redundant_pairs = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            if abs(corr[i, j]) > 0.8:
                redundant_pairs.append((method_names[i], method_names[j], float(corr[i, j])))

    if redundant_pairs:
        log(f"\n  Redundant method pairs (|r| > 0.8):")
        for a, b, r in redundant_pairs:
            log(f"    {a} <-> {b}: r={r:.3f}")

    results.append(EvalResult(
        metric_id="S07.n_redundant_pairs",
        value=float(len(redundant_pairs)),
        n_samples=len(method_names) * (len(method_names) - 1) // 2,
        metadata={
            "redundant_pairs": [
                {"a": a, "b": b, "correlation": r}
                for a, b, r in redundant_pairs
            ],
            "method_names": method_names,
        },
    ))

    return results


def run_protocol(model, tasks, n_prompts=40, device="cpu", run_cals=False,
                 protocol_results=None):
    from protocols import ProtocolResult
    t0 = time.time()
    evals = run_meta_learner(model, tasks, device=device,
                             protocol_results=protocol_results)
    result = ProtocolResult(
        protocol_id=PROTOCOL_ID,
        protocol_name=PROTOCOL_NAME,
        tasks=tasks,
        elapsed_seconds=time.time() - t0,
    )
    result.metrics["meta_learner"] = evals
    return result


def summarize(result) -> str:
    lines = [f"=== {PROTOCOL_NAME} ==="]
    for ev in result.metrics.get("meta_learner", []):
        lines.append(f"  {ev.metric_id}: {ev.value:.4f}")
    return "\n".join(lines)


def main():
    parser = parse_common_args("S07: Meta-Learner Circuit Predictor")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args()
    tasks = args.tasks or CIRCUIT_TASKS
    with open(args.results_json) as f:
        protocol_results = json.load(f)

    log("=" * 60)
    log("S07: META-LEARNER CIRCUIT PREDICTOR (PARCEL L3)")
    log("=" * 60)

    results = run_meta_learner(tasks=tasks, protocol_results=protocol_results)
    out = args.out or "meta_p7_meta_learner.json"
    save_results(results, out)
    log(f"\nDone. {len(results)} results.")


if __name__ == "__main__":
    main()
