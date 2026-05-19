"""Attention Pattern Clustering (Structural S96)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrument:     S96 — Attention Pattern Clustering
Categories:     structural
Validity layer: Internal
Criteria:       Attention-space separability of circuit heads
Establishes:    Whether circuit heads form distinct clusters in attention-pattern
                space compared to non-circuit heads
Requires:       CPU or GPU, model, sklearn
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each task:
1. Run the model on prompts, collect attention patterns from all heads
   at the last token position.
2. Flatten each head's attention pattern into a vector.
3. Run k-means clustering (k = number of circuit heads) on ALL heads'
   attention patterns.
4. Compute cluster purity: what fraction of circuit heads land in the
   same cluster(s)?
5. Compute silhouette score for circuit vs non-circuit grouping.
6. Pass condition: silhouette score > 0.1 (circuit heads are more similar
   to each other than to non-circuit heads in attention space).

Usage:
    uv run python 96_attention_clustering.py --tasks ioi --n-prompts 40
    uv run python 96_attention_clustering.py --tasks ioi sva --device cpu
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mechanistic_validity.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    generate_prompts,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)


@torch.no_grad()
def collect_attention_patterns(model, prompts, n_prompts: int) -> np.ndarray:
    """Collect mean attention patterns at the last token for every head.

    Returns array of shape (n_layers * n_heads, seq_len) averaged over prompts.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads

    # Accumulate attention patterns across prompts
    sum_patterns = None
    count = 0

    for p in prompts[:n_prompts]:
        tokens = model.to_tokens(p.text)
        seq_len = tokens.shape[1]
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: "hook_pattern" in n)

        # Gather attention pattern at last token for every head
        prompt_patterns = []
        for layer in range(n_layers):
            # hook_pattern shape: (batch, n_heads, seq_q, seq_k)
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            # Take last query position -> (n_heads, seq_k)
            last_tok_pattern = pattern[0, :, -1, :].cpu().numpy()
            prompt_patterns.append(last_tok_pattern)

        # Stack to (total_heads, seq_len)
        prompt_all = np.concatenate(prompt_patterns, axis=0)

        if sum_patterns is None:
            sum_patterns = np.zeros((total_heads, seq_len), dtype=np.float64)

        # Pad or truncate to handle varying sequence lengths
        actual_seq = prompt_all.shape[1]
        common = min(actual_seq, sum_patterns.shape[1])
        if actual_seq > sum_patterns.shape[1]:
            new_sum = np.zeros((total_heads, actual_seq), dtype=np.float64)
            new_sum[:, :sum_patterns.shape[1]] = sum_patterns
            sum_patterns = new_sum
        sum_patterns[:, :common] += prompt_all[:, :common]
        count += 1

    if count == 0 or sum_patterns is None:
        return np.zeros((total_heads, 1))

    return sum_patterns / count


def compute_cluster_purity(labels: np.ndarray, circuit_mask: np.ndarray) -> float:
    """Fraction of circuit heads in the most-populated cluster(s).

    For each cluster that contains at least one circuit head, count how many
    circuit heads are in that cluster. Purity = (max such count) / n_circuit.
    """
    n_circuit = int(circuit_mask.sum())
    if n_circuit == 0:
        return 0.0

    circuit_labels = labels[circuit_mask]
    unique_labels, counts = np.unique(circuit_labels, return_counts=True)
    return float(counts.max()) / n_circuit


def run_attention_clustering(model, tasks: list[str],
                             n_prompts: int = 40) -> list[EvalResult]:
    results = []
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        prompts = generate_prompts(task, model.tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        log(f"  {task}: {len(circuit_heads)} circuit heads, "
            f"{total_heads} total heads, {len(prompts)} prompts")

        # Step 1-2: collect and flatten attention patterns
        patterns = collect_attention_patterns(model, prompts, n_prompts)

        # Build circuit mask: True for circuit heads
        circuit_mask = np.zeros(total_heads, dtype=bool)
        for (layer, head) in circuit_heads:
            idx = layer * n_heads + head
            if idx < total_heads:
                circuit_mask[idx] = True

        n_circuit = int(circuit_mask.sum())
        if n_circuit < 2:
            log(f"  {task}: fewer than 2 circuit heads in model, skipping")
            continue

        n_non_circuit = total_heads - n_circuit
        if n_non_circuit < 1:
            log(f"  {task}: all heads are circuit heads, skipping")
            continue

        # Step 3: k-means with k = number of circuit heads
        k = n_circuit
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(patterns)

        # Step 4: cluster purity
        purity = compute_cluster_purity(cluster_labels, circuit_mask)

        # Step 5: silhouette score for circuit vs non-circuit grouping
        binary_labels = circuit_mask.astype(int)
        sil_score = float(silhouette_score(patterns, binary_labels))

        # Step 6: pass condition
        passed = sil_score > 0.1

        log(f"    silhouette={sil_score:.4f}  purity={purity:.4f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

        results.append(EvalResult(
            metric_id="S96.attention_clustering",
            value=sil_score,
            n_samples=len(prompts),
            metadata={
                "task": task,
                "n_circuit_heads": n_circuit,
                "n_total_heads": total_heads,
                "k_clusters": k,
                "cluster_purity": purity,
                "silhouette_score": sil_score,
                "passed": passed,
                "threshold": 0.1,
                "circuit_heads": sorted([list(h) for h in circuit_heads]),
            },
        ))

    return results


def main():
    parser = parse_common_args("S96: Attention Pattern Clustering")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("S96: ATTENTION PATTERN CLUSTERING")
    log("=" * 60)

    out = args.out or "96_attention_clustering.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_attention_clustering(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)
            p = "PASS" if r.metadata["passed"] else "FAIL"
            log(f"  {task}: silhouette={r.value:.4f}  "
                f"purity={r.metadata['cluster_purity']:.4f}  [{p}]")

    save_results(results, out)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
