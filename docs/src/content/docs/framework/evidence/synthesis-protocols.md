---
title: "Synthesis Protocols"
description: "Higher-order analyses that aggregate across multiple protocol outputs."
---

# Synthesis Protocols

A synthesis protocol consumes the outputs of multiple protocols and produces higher-order structure — consensus estimates, parcellations, stability metrics, or learned classifiers. Where a protocol bundles metrics around one validity question, a synthesis protocol bundles protocol *results* to extract patterns that no single protocol can see.

The outputs are scored measurements that feed back into the standard criteria-scoring pipeline. A synthesis protocol does not bypass criteria — it generates richer evidence for them.

## The nine synthesis protocols

### S01 -- Functional Parcellation

Adapts [Glasser et al. (2016)](https://doi.org/10.1038/nature18933)'s multimodal brain parcellation to circuits. Takes ranked component lists from 4+ protocols across different evidence families, computes representational similarity (RSA) between the rankings, and clusters components that are consistently co-ranked. The output is a set of functional groups — components that multiple independent methods agree belong together.

**Strengthens:** C5 Convergent validity, I3 Specificity (by identifying functional subgroups within a circuit).

### S02 -- Dawid-Skene Consensus

Treats each protocol as a noisy annotator and jointly estimates (via EM) both the true circuit membership and each protocol's reliability. The key insight from [Dawid & Skene (1979)](https://doi.org/10.2307/2346806): some annotators are systematically wrong, and the consensus should weight reliable annotators more. Applied to protocols, this means a protocol that consistently disagrees with the majority gets downweighted — unless it is the only one that is right, in which case the consensus shifts.

**Strengthens:** C5 Convergent validity, M1 Reliability (provides per-protocol reliability estimates).

### S03 -- Robust Rank Aggregation

Computes [Kolde et al. (2012)](https://doi.org/10.1093/bioinformatics/btr709) RRA p-values and Borda counts across protocol rankings. Identifies components that rank consistently high across protocols (robust members) and components that rank high in one protocol but low in others (method-dependent members). The latter are candidates for convergent validity failure.

**Strengthens:** C5, I5 Confound control (method-dependent components may reflect method artifacts).

### S04 -- Parallel Ensemble

Implements three fusion rules — equal weighting, protocol-reliability weighting, and minimum-across-protocols — on rank-normalized scores. Produces a single composite ranking with uncertainty bounds. This is the simplest aggregation and serves as the baseline for more sophisticated methods.

**Strengthens:** C5, M3 Baseline separation (composite ranking has tighter confidence intervals than individual protocols).

### S05 -- Sequential Ensemble

A two-stage pipeline: cheap protocols (weight-space, information-theoretic) run first and filter to the top 20% of components, then expensive protocols (causal, behavioral) run only on the filtered set. Reduces computational cost by 5–10x while preserving the ranking of the top components. The filtering threshold is a parameter; the default (top 20%) is calibrated to retain all components that any single expensive protocol would rank in its top set.

**Strengthens:** Practical efficiency. Does not introduce new evidence types, but makes it feasible to run expensive protocols on large circuits.

### S06 -- Wasserstein Stability

Computes the Wasserstein-1 distance between component score distributions across runs (different random seeds, prompt samples, or tasks). A circuit with low W₁ across prompt samples but high W₁ across tasks is robust to measurement noise but sensitive to task context — an informative distinction that individual protocol results cannot make.

**Strengthens:** M1 Reliability (cross-run stability), E5 Robustness (cross-task stability).

### S07 -- Meta-Learner

Trains a logistic regression on known-labeled circuits (components with established ground truth from published analyses) using protocol features as predictors. Evaluated by leave-one-out cross-validation. The trained model predicts circuit membership for novel components, and the learned coefficients reveal which protocol features are most predictive of true membership.

**Strengthens:** M4 Sensitivity (predictive accuracy on held-out circuits), C5 (which protocol features are redundant vs. complementary).

### S08 -- Granger Causality Graph

Constructs a directed graph where edges represent Granger-causal relationships between protocol scores — does component A's score in protocol X predict component B's score in protocol Y, conditional on B's own score in X? The graph reveals information flow between evidence families: structural evidence that predicts causal evidence (or vice versa) indicates genuine convergent support rather than shared noise.

**Strengthens:** C5, I4 Consistency (cross-family predictive structure).

### S09 -- ModCirc Vocabulary

Adapts He et al. (ICML 2025)'s modular circuit vocabulary: identifies reusable circuit subgraphs shared across tasks. If the same three-head subgraph appears in the IOI circuit, the Greater-Than circuit, and the induction circuit, it is a computational primitive — a building block rather than a task-specific artifact.

**Strengthens:** E6 Cross-architecture (shared subgraphs generalize across tasks), C2 Structural plausibility (recurring motifs are more likely to be real computational units than one-off findings).

## When to use synthesis protocols

Synthesis protocols are useful when multiple protocols have been run and the analyst wants to move beyond "which criteria pass?" to "what is the structure of the evidence?"

Typical use cases:

- **After running 5+ protocols:** Use S04 (parallel ensemble) or S03 (robust rank aggregation) to get a composite ranking.
- **When convergent validity (C5) is uncertain:** Use S02 (Dawid-Skene) to jointly estimate truth and reliability.
- **When comparing across tasks:** Use S06 (Wasserstein stability) to quantify which findings are task-general vs task-specific.
- **When the circuit is large:** Use S01 (parcellation) to identify functional subgroups.
- **When computational budget is limited:** Use S05 (sequential ensemble) to filter before running expensive protocols.

Synthesis protocols live alongside regular protocols in the [experiments repository](https://github.com/mechanistic-validity/mechanistic-validity-experiments) under `experiments/protocols/views/` and `experiments/synthesis/`.

---

## Detailed protocol specifications

The sections below expand each synthesis protocol with algorithmic details, input requirements, output metrics, and usage examples. For the summary descriptions above, each protocol's "Strengthens" line names the criteria it provides evidence for.

### S01 -- Functional Parcellation

**What it aggregates.** Independent clustering signals from weight features, causal importance scores, circuit membership across tasks, and graph topology. Each signal produces an independent hierarchical clustering (using Ward or average linkage, depending on the distance metric). Final parcels require agreement across a configurable threshold of signals (default: 3 out of 4+).

**Input signals.**

1. **Weight features**: per-head W_Q, W_K, W_V, W_O norms, QK spectral concentration, OV spectral concentration. Z-scored, clustered with Euclidean distance.
2. **Causal importance**: per-head scores from activation patching, EAP, DAS-IIA, sigma ablation. Z-scored, clustered with cosine distance.
3. **Circuit membership**: binary membership vectors across all tasks. Clustered with Jaccard distance.
4. **Graph topology**: per-head in-degree, out-degree, and total degree from circuit edge lists. Z-scored, clustered with Euclidean distance.

**Algorithm.** For each signal, hierarchical clustering produces k clusters (default k=14). An RSA (representational similarity analysis) matrix measures second-order Spearman correlation between all pairs of signal clusterings. Then a convergent parcellation step counts, for each pair of heads, how many signals place them in the same cluster. Pairs that co-cluster in at least `min_agreement` signals (default: 3) are grouped into the same parcel.

**Output metrics.**

- `S01.n_parcels`: number of convergent parcels discovered.
- `S01.silhouette`: silhouette score of the parcellation over concatenated signal features.
- `S01.mean_rsa`: mean pairwise RSA between signal clusterings (higher = signals agree more).

**What it establishes.** Functional subgroups within the model -- components that multiple independent methods agree belong together. Strengthens C5 (convergent validity) and I3 (specificity) by revealing whether the circuit contains internally coherent functional clusters.

**Usage example.**

```bash
uv run python functional_parcellation.py \
  --results-json modal_sweep_results.json \
  --n-clusters 14 --min-agreement 3
```

---

### S02 -- Dawid-Skene Consensus

**What it aggregates.** Binary circuit-membership labels from multiple protocols. Each protocol is treated as a noisy annotator that labels each head as in-circuit or not (thresholded at 0.5 by default). The EM algorithm jointly estimates true labels and per-protocol confusion matrices.

**Input protocols.** Any protocol that produces per-head scores. Requires at least 2 protocols with per-head annotations for the target task.

**Algorithm.** Dawid-Skene EM (1979):

1. Initialize: assume 70% prevalence of non-circuit, uniform confusion matrices.
2. E-step: compute posterior probability of true label for each head, given all protocol annotations and current confusion matrices.
3. M-step: update confusion matrices using the posteriors as soft labels. Update prevalence.
4. Repeat until convergence (max delta < 1e-4) or 50 iterations.

The confusion matrix for each protocol yields sensitivity (true positive rate) and specificity (true negative rate), providing a reliability estimate per protocol.

**Output metrics.**

- `S02.consensus_jaccard`: Jaccard similarity between the consensus circuit and the ground-truth circuit.
- Per-protocol reliability scores (sensitivity, specificity, accuracy) in metadata.

**What it establishes.** A consensus circuit that accounts for differential protocol reliability. Protocols that systematically disagree with the majority are downweighted. Strengthens C5 (convergent validity) and M1 (reliability) by providing per-protocol reliability estimates.

**Usage example.**

```bash
uv run python dawid_skene.py \
  --results-json modal_sweep_results.json \
  --task ioi --threshold 0.5
```

---

### S03 -- Robust Rank Aggregation

**What it aggregates.** Ranked lists of heads from multiple protocols. Each protocol ranks all 144 GPT-2 heads by importance. RRA tests whether a head appears consistently near the top of multiple ranked lists, more than expected under a uniform null.

**Input protocols.** Any protocol that produces per-head scores (converted to rankings). Requires at least 2 protocols.

**Algorithm.** Kolde et al. (2012) Robust Rank Aggregation:

1. For each head, collect its normalized rank (position / total) from each protocol.
2. Sort the normalized ranks. For each order statistic, compute the probability under the Beta(k, n-k) null.
3. The RRA score is min(p-value * n_protocols) -- the minimum corrected p-value across order statistics.
4. Apply Bonferroni correction across all heads.

Also computes Borda count (mean normalized rank) and Dowdall (harmonic mean rank) as simpler baselines.

**Output metrics.**

- `S03.rra_jaccard`: Jaccard similarity between Bonferroni-significant heads and the ground-truth circuit.
- `S03.n_significant`: number of heads with corrected p < 0.05.
- Top-20 RRA p-values and Borda scores in metadata.

**What it establishes.** Which components are robustly important across methods (robust members) and which are method-dependent (appear in one list but not others). Method-dependent components are candidates for convergent validity failure. Strengthens C5 and I5 (confound control).

**Usage example.**

```bash
uv run python rank_aggregation.py \
  --results-json modal_sweep_results.json \
  --task ioi
```

---

### S04 -- Parallel Ensemble

**What it aggregates.** Rank-normalized component scores from all available protocols, fused using three rules:

1. **Equal-weight average**: simple mean across protocols.
2. **Method-type-weighted average**: causal protocols weighted 2x, spectral 1.5x, statistical and structural 1x.
3. **Minimum (conservative)**: takes the minimum score across protocols -- only keeps components that all methods agree on.

**Input protocols.** Any protocol that produces per-head scores. Requires at least 2 protocols.

**Algorithm.** Scores from each protocol are rank-normalized to [0, 1] (higher = more important). The three fusion rules produce three composite rankings. For each, the top-k heads (default k=20) are compared to ground truth via precision, recall, F1, and Jaccard. A cross-method correlation matrix is also computed.

**Output metrics.**

- `S04.equal_avg_f1`, `S04.weighted_avg_f1`, `S04.minimum_f1`: F1 score for each fusion rule.
- `S04.method_correlation`: mean pairwise correlation between protocol score vectors.
- Top-10 heads per fusion rule in metadata.

**What it establishes.** A composite ranking with tighter confidence intervals than individual protocols. The simplest aggregation baseline. Strengthens C5 (convergent validity) and M3 (baseline separation).

**Usage example.**

```bash
uv run python parallel_ensemble.py \
  --results-json modal_sweep_results.json \
  --task ioi --top-k 20
```

---

### S05 -- Sequential Ensemble

**What it aggregates.** A two-phase pipeline that separates cheap and expensive protocols:

- **Phase 1 (filter):** Run cheap protocols (structural, spectral, statistical -- no interventions) on all 144 heads. Rank by mean score. Keep top P% (default: 20%, i.e., 29 heads).
- **Phase 2 (refine):** Run expensive protocols (causal interventions, full patching) only on the filtered candidate set.
- **Phase 3 (re-rank):** Re-rank candidates using expensive protocol scores.

**Input protocols.** Cheap protocols: B01--B04 (structural), WC_M3/M5/M6/M8/M9/M10/M11/M13 (statistical/spectral). Expensive protocols: A01--A06 (causal), MB_KH/RE/TE (biology-causal).

**Output metrics.**

- `S05.sequential_f1`: F1 of the final refined circuit vs ground truth.
- `S05.filter_recall`: fraction of ground-truth heads retained after Phase 1 filtering.
- Compression ratio (144 / n_candidates) in metadata.

**What it establishes.** Practical computational efficiency. Reduces forward passes by 5--10x while preserving ranking quality for top components. Does not introduce new evidence types but makes it feasible to run expensive protocols on large circuits. The filter_recall metric validates that cheap methods do not discard important heads.

**Usage example.**

```bash
uv run python sequential_ensemble.py \
  --results-json modal_sweep_results.json \
  --task ioi --filter-pct 0.2
```

---

### S06 -- Wasserstein Stability

**What it aggregates.** Score distributions across tasks, runs, or models. Uses the Wasserstein-1 (Earth Mover's) distance to quantify how different two circuit score distributions are, optionally using weight-space cosine similarity as the ground metric.

**Input protocols.** Any protocol results that produce per-head scores. Requires at least 2 tasks for cross-task comparison.

**Algorithm.** Three applications:

1. **Cross-task distance**: W_1 between score distributions for different tasks (e.g., IOI vs SVA). Low distance = tasks use similar circuits.
2. **Predicted vs ground truth**: W_1 between the protocol-predicted importance distribution and the binary ground-truth distribution.
3. **Cross-run stability** (when multiple runs available): W_1 between score distributions from different random seeds.

For the 1D case, W_1 is computed as the mean absolute difference between sorted distributions.

**Output metrics.**

- `S06.cross_task_w1`: Wasserstein-1 distance between each pair of tasks.
- `S06.pred_gt_w1`: distance between predicted and ground-truth distributions.

**What it establishes.** A principled measure of circuit similarity that accounts for the relative positions of components in score space. A circuit with low W_1 across prompt samples but high W_1 across tasks is robust to measurement noise but sensitive to task context -- an informative distinction. Strengthens M1 (reliability, cross-run stability) and E5 (robustness, cross-task stability).

**Usage example.**

```bash
uv run python wasserstein_stability.py \
  --results-json modal_sweep_results.json
```

---

### S07 -- Meta-Learner

**What it aggregates.** Protocol features from all available methods, used as predictors in a logistic regression trained on known-labeled circuits. Learns which protocol features are most predictive of true circuit membership.

**Input protocols.** Any protocol results with per-head scores, plus ground-truth circuit labels for at least 2 tasks.

**Algorithm.**

1. For each task with ground-truth labels, build a feature matrix: each row is a head, columns are rank-normalized scores from each protocol.
2. Fit logistic regression with class-weight balancing (positive class upweighted by prevalence ratio) for 500 iterations.
3. Evaluate via leave-one-task-out cross-validation: train on all tasks except the held-out, predict on the held-out, compute AUROC.
4. Extract learned method weights as importance indicators.
5. Compute pairwise feature correlations to identify redundant methods (|r| > 0.8).

**Output metrics.**

- `S07.overall_auroc`: AUROC of the full model on all labeled data.
- `S07.mean_loo_auroc`: mean AUROC across leave-one-task-out folds (the key generalization metric).
- `S07.n_redundant_pairs`: number of method pairs with |correlation| > 0.8.
- Per-method importance weights in metadata.

**What it establishes.** Which protocol features are predictive of true circuit membership and which are redundant. High LOO-AUROC indicates that circuit membership generalizes across tasks. Strengthens M4 (sensitivity, predictive accuracy) and C5 (which features are complementary vs redundant).

**Usage example.**

```bash
uv run python meta_learner.py \
  --results-json modal_sweep_results.json
```

---

### S08 -- Granger Causality Graph

**What it aggregates.** Pairwise conditional independence tests between head activations, treating layer depth as time. Constructs a directed graph where edges represent Granger-causal relationships between head activation time series.

**Input protocols.** Requires the model directly (GPU forward passes). Protocol results are optional.

**Algorithm.**

1. For each task, collect head activations across all prompts: z[:, :, head, :].mean(d_head) gives a scalar per head per position per prompt, flattened into a time series.
2. For each directed pair (A, B) where layer_A < layer_B, run a Granger F-test: does A's activation improve prediction of B's activation beyond B's own autoregressive history? Uses max_lag=2.
3. Retain edges with p < 0.05, then apply Bonferroni correction.
4. Compute hub statistics (in-degree, out-degree) and compare edge set to ground-truth circuit edges via Jaccard similarity.

**Output metrics.**

- `S08.n_granger_edges`: number of significant directed edges after Bonferroni correction.
- Edge Jaccard with ground-truth edges, hub heads ranked by out-degree, and top-20 edges by F-statistic in metadata.

**What it establishes.** A directed circuit graph from purely observational data (no interventions). Reveals information flow between components: structural evidence that predicts downstream activations. Strengthens C5 (convergent validity, cross-family predictive structure) and I4 (consistency).

**Usage example.**

```bash
uv run python granger_circuit.py \
  --tasks ioi --n-prompts 40 --device cuda
```

---

### S09 -- ModCirc Vocabulary

**What it aggregates.** Circuit definitions across multiple tasks, identifying reusable subgraphs shared across tasks. Adapts He et al. (ICML 2025)'s modular circuit vocabulary.

**Input protocols.** Ground-truth circuit head lists and edge lists for at least 2 tasks. Protocol results are optional (used for consistency scoring).

**Algorithm.**

1. For each head, count how many task circuits include it.
2. Identify "shared heads" that appear in at least `min_tasks` (default: 2) circuits.
3. Cluster shared heads by spatial proximity (layer distance <= 2) and task overlap: heads that are close in layer space and share task participation are grouped into a module.
4. Each module must have at least 2 heads shared by at least `min_tasks` tasks.
5. Compute per-task coverage (fraction of circuit heads that belong to any vocabulary module).
6. If protocol results are available, compute consistency: for each module, measure CV of protocol scores across the tasks that share it (low CV = consistent function).
7. Compute locality: 1 - mean(layer_span / 12) across modules (higher = more spatially compact).

**Output metrics.**

- `S09.vocab_size`: number of discovered vocabulary modules.
- `S09.mean_coverage`: mean fraction of each task's circuit explained by vocabulary modules.
- `S09.consistency`: mean functional consistency across modules (1 - CV of cross-task scores).
- `S09.locality`: spatial compactness of modules (1 = single layer, 0 = all layers).

**What it establishes.** Computational primitives -- building blocks that appear in multiple task circuits. If the same three-head subgraph appears in the IOI circuit, the Greater-Than circuit, and the induction circuit, it is a reusable computational unit rather than a task-specific artifact. Strengthens E6 (cross-architecture, shared subgraphs generalize) and C2 (structural plausibility, recurring motifs are more likely real).

**Usage example.**

```bash
uv run python modcirc_vocabulary.py \
  --results-json modal_sweep_results.json \
  --min-tasks 2
```
