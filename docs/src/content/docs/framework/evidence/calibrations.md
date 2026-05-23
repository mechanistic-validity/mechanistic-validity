---
title: "Calibrations"
description: "Quality gates for metric outputs: bootstrap stability, seed variance, convergent validity, and 13 other calibration checks."
---

# Calibrations

A calibration is a quality gate on a metric's output. It answers: *can we trust this measurement?* A metric result without calibration is a measurement without error bars.

Calibrations are themselves metrics -- they return scored measurements and are registered in the same metric registry. The distinction is functional: calibrations evaluate the trustworthiness of other measurements rather than measuring the model directly.

The framework defines 16 calibrations. Each checks a specific aspect of measurement quality -- resampling stability, cross-method agreement, template sensitivity, or statistical rigor. When a calibration fails, it does not mean the underlying metric is wrong; it means the measurement is not reliable enough to draw conclusions from.

## Reference table

| ID | Name | Criterion | Requires |
|---|---|---|---|
| C11 | Bootstrap Stability | M1 Reliability | GPU, model |
| C12 | Convergent Validity | C5 Convergent validity | CPU, data-only |
| C13 | Measurement Invariance | M2 Invariance | GPU, model |
| C14 | Derived Metrics (Sensitivity) | M4 Sensitivity | CPU, data-only |
| C16 | Reliability Suite | M1 Reliability | GPU, model |
| C17 | Discriminant Validity | M3 Baseline separation | GPU, model |
| C23 | Nomological Validity | C5 Convergent validity | CPU, data-only |
| C30 | Seed Variance | M1 Reliability | GPU, model |
| C36 | Incremental Validity | C5 Convergent validity | GPU, model |
| F59 | Inter-Rater Reliability | M1 Reliability | GPU, model |
| S1 | Distributional Characterization | S1 Distributional Characterization | GPU, model |
| S3 | Distributional Stability | S3 Distributional Stability | GPU, model |
| M93 | Multiple Comparisons Correction | M5 Statistical Rigor | CPU, data-only |
| M98 | Ablation Method Invariance | Method Robustness | GPU, model |
| E1b | Method Invariance | E1b Method Invariance | GPU, model |
| M3b | Certified Stability | Head-level stability | GPU, model |

---

## C11 -- Bootstrap Stability

**What it checks.** Whether a metric produces consistent results when the prompt set is resampled with replacement. Wraps any inner metric (faithfulness or completeness) and computes bootstrap confidence intervals across 1000 resamples.

**Applies to.** Faithfulness and completeness metrics, per task.

**Pass threshold.** The 95% confidence interval (2.5th--97.5th percentile of bootstrap distribution) should be narrow relative to the point estimate. A sigma greater than 0.1 on a metric in the [0, 1] range indicates unstable measurement.

**Passing means.** The metric gives consistent results across different prompt samples -- the point estimate is not an artifact of the specific prompts chosen.

**Failing means.** The metric is sensitive to which prompts are included. The point estimate is unreliable; a different sample could yield a substantially different result. Increase sample size or investigate prompt heterogeneity.

**Metric ID:** `C11.bootstrap_{inner}` (where inner is `faithfulness` or `completeness`).

---

## C12 -- Convergent Validity

**What it checks.** Whether different metric families agree on which heads are circuit components. Computes Spearman rank correlation between per-head scores from activation patching (C2), oCSE (C7), LLC (C10), and mediation (C5). Optionally computes ICC(2) via pingouin.

**Applies to.** Any task with per-head scores from at least two metric families.

**Pass threshold.** Mean pairwise Spearman rho > 0.5 indicates moderate convergent validity. ICC > 0.7 indicates good agreement.

**Passing means.** Independent measurement approaches agree on the relative importance of circuit components. The construct "circuit membership" is robust across operationalizations.

**Failing means.** Different methods rank heads differently. The circuit definition may be method-dependent rather than reflecting a genuine structural property. Investigate which methods disagree and why.

**Metric ID:** `C12.convergent_validity`.

---

## C13 -- Measurement Invariance

**What it checks.** Whether circuit faithfulness scores are stable across prompt templates. Splits prompts into groups (by metadata template or by length: short/medium/long), computes faithfulness on each group, and tests whether the groups differ using Welch ANOVA with partial eta-squared.

**Applies to.** Faithfulness scores, per task.

**Pass threshold.** Eta-squared < 0.01 = invariant. Eta-squared 0.01--0.06 = moderate. Eta-squared > 0.06 = template-sensitive.

**Passing means.** The circuit's faithfulness score measures a stable property that does not depend on prompt surface form. Analogous to measurement invariance in confirmatory factor analysis.

**Failing means.** The circuit behaves differently on different kinds of prompts. A faithfulness score of 0.87 on one template type and 0.31 on another means the aggregate score is misleading. Investigate which templates break the circuit.

**Metric ID:** `C13.measurement_invariance`.

---

## C14 -- Derived Metrics (Sensitivity)

**What it checks.** Whether circuit metrics produce signal-detection statistics that discriminate circuit from non-circuit heads. Computes approximately 20 derived quantities from previously computed results -- no forward passes required. Key sub-metrics include:

- **Sparsity** (`C14.sparsity`): fraction of all model heads in the circuit (n_circuit / 144 for GPT-2).
- **Node overlap** (`C14.node_overlap_jaccard`): size ratio between discovered and published circuit as a Jaccard proxy.
- **Spectral norm ratio** (`C14.spectral_norm_ratio`): circuit-head mean spectral norm divided by non-circuit mean.
- **d-prime** (`C14.d_prime`): signal-detection discriminability between circuit and random head patching effects.
- **LLC d-prime** (`C14.llc_d_prime`): discriminability from local learning coefficient values.
- **Hit rate / False alarm rate** (`C14.hit_rate`, `C14.false_alarm_rate`): from weight-classifier recall and precision.
- **Partial eta-squared** (`C14.partial_eta_sq`): forwarded from C13 measurement invariance.
- **Attribution AUROC** (`C14.attribution_auroc`): area under ROC for attribution patching predictions.
- **Faithfulness / Completeness** (`C14.faithfulness`, `C14.completeness`): from pillar data.
- **Minimality** (`C14.minimality_mean_importance`): mean individual head importance.
- **Logit-diff recovery** (`C14.logit_diff_recovery`): from causal scrubbing.

**Applies to.** All tasks with existing per-head metric outputs.

**Pass threshold.** d-prime > 1.0 indicates clear discriminability. Hit rate > 0.9 with false alarm rate < 0.2 indicates a useful classifier.

**Passing means.** The metric ensemble produces enough signal to reliably distinguish circuit from non-circuit components.

**Failing means.** Circuit and non-circuit heads are not separable by the available metrics. Either the circuit definition is wrong, or the metrics are not measuring the right thing.

**Metric IDs:** `C14.*` (approximately 20 sub-metrics).

---

## C16 -- Reliability Suite

**What it checks.** Three psychometric reliability measures:

1. **Test-retest** (`C16.test_retest`): faithfulness measured with three different prompt seeds (42, 123, 456). Reports 1 - CV as a reliability index.
2. **Split-half** (`C16.split_half`): Spearman-Brown corrected correlation between odd- and even-indexed prompt faithfulness values.
3. **Cronbach's alpha** (`C16.cronbach_alpha`): internal consistency of per-head activation patching effects across prompts. Each circuit head is an "item"; alpha measures whether heads contribute consistently.

**Applies to.** Faithfulness and per-head patching effects, per task.

**Pass threshold.** Test-retest reliability > 0.8. Split-half corrected r > 0.7. Cronbach's alpha > 0.7 (conventional psychometric threshold).

**Passing means.** The circuit measurement is reliable in the psychometric sense: repeatable across prompt samples (test-retest), internally consistent across prompt halves (split-half), and coherent across circuit components (alpha).

**Failing means.** Low test-retest: different prompt samples give different scores. Low split-half: even and odd prompts measure different things. Low alpha: circuit heads do not form a coherent "instrument" -- some heads contribute erratically.

**Metric IDs:** `C16.test_retest`, `C16.split_half`, `C16.cronbach_alpha`.

---

## C17 -- Discriminant Validity

**What it checks.** Whether a circuit identified for task A shows high causal importance on unrelated task B. It should not -- a good circuit is task-specific. For each ordered pair (A, B), computes the mean activation patching effect of task-A's circuit heads on task-B's prompts. Also computes random-head baselines.

**Applies to.** All pairs of tasks with defined circuits (requires at least 2 tasks).

**Pass threshold.** Discriminant ratio (diagonal / mean off-diagonal) > 3.0 indicates good task specificity. Off-diagonal effects should be comparable to random baselines.

**Passing means.** The circuit is specific to its target task. Task A's heads are important for task A but not for task B. The circuit captures task-relevant computation, not general model activity.

**Failing means.** The circuit's heads are important for multiple tasks -- either the circuit is capturing shared computation (which may be correct for overlapping tasks) or the circuit definition is too broad.

**Metric IDs:** `C17.discriminant_validity`, `C17.discriminant_matrix`.

---

## C23 -- Nomological Validity

**What it checks.** Whether circuit structure correlates with expected layer-depth patterns. Computes two Spearman correlations:

1. **Layer density** (`C23.layer_density_correlation`): correlation between layer index (0--11) and count of circuit heads at that layer.
2. **Role-depth** (`C23.role_depth_correlation`): correlation between each head's layer and its functional role's ordinal position (early=1, mid=2, late=4).

**Applies to.** All tasks with defined circuit head roles and band assignments.

**Pass threshold.** Significant positive correlation (p < 0.05) for role-depth indicates that early-layer roles are assigned to early layers and late roles to late layers, consistent with the hierarchical processing hypothesis.

**Passing means.** The circuit's structure is consistent with known computational principles -- early layers handle low-level features, late layers handle task-specific computation. The circuit obeys the "nomological network" of transformer computation.

**Failing means.** Circuit roles do not align with layer depth. This is not necessarily wrong (some computations may genuinely skip layers), but it warrants investigation.

**Metric IDs:** `C23.layer_density_correlation`, `C23.role_depth_correlation`.

---

## C30 -- Seed Variance

**What it checks.** Whether faithfulness scores are stable across different random seeds for prompt subsampling. Generates a large prompt pool (3x the evaluation size), subsamples with 5 different seeds (42, 123, 456, 789, 1337), computes faithfulness on each subsample, and reports the coefficient of variation (CV = std / |mean|) across seeds.

**Applies to.** Faithfulness scores, per task.

**Pass threshold.** CV < 0.05 indicates excellent stability. CV < 0.10 is acceptable. CV > 0.20 indicates problematic variance.

**Passing means.** The faithfulness measurement is reproducible -- different random prompt samples produce similar results. The evaluation pipeline is not seed-sensitive.

**Failing means.** Results change substantially depending on which prompts happen to be sampled. Increase sample size or investigate heterogeneity in the prompt distribution.

**Metric ID:** `C30.seed_variance`.

---

## C36 -- Incremental Validity

**What it checks.** Whether the weight-classifier circuit outperforms the simplest possible baseline: top-k heads by mean activation magnitude (where k = circuit size). Computes faithfulness for both the circuit and the top-k baseline, and reports the delta.

**Applies to.** Any task with a defined circuit and a measurable faithfulness score.

**Pass threshold.** Positive delta indicates the circuit method adds value beyond the naive baseline. Delta > 0.1 indicates substantial improvement.

**Passing means.** The circuit discovery method finds better circuits than simply selecting the most active heads. The method captures structural information beyond raw activation magnitude.

**Failing means.** Top-k by activation magnitude performs comparably to or better than the circuit. The circuit method is not adding value, or the circuit is effectively selecting high-activity heads.

**Metric ID:** `C36.incremental_validity`.

---

## F59 -- Inter-Rater Reliability

**What it checks.** Whether independent circuit discovery methods agree on which heads are circuit members. Treats three ranking methods as "raters":

1. **Activation patching**: logit-diff restoration per head.
2. **Direct Logit Attribution (DLA)**: projection of head output onto the answer direction.
3. **Weight-space OV norm**: Frobenius norm of W_OV projected onto the answer direction.

Reports three agreement statistics:
- **Kendall's W** (`F59.kendalls_w`): concordance across all three ranked lists.
- **Mean Spearman rho** (`F59.mean_spearman`): pairwise rank correlation.
- **Mean Cohen's kappa** (`F59.mean_kappa`): agreement on binary top-k classification.

**Applies to.** All tasks with defined circuits.

**Pass threshold.** Kendall's W > 0.5 indicates moderate agreement. Mean kappa > 0.4 indicates fair-to-moderate agreement on binary classification.

**Passing means.** Multiple independent methods converge on similar head rankings. Circuit membership is not an artifact of one particular method.

**Failing means.** Methods disagree on which heads matter. The circuit may be method-dependent. Investigate which method pair disagrees most.

**Metric IDs:** `F59.kendalls_w`, `F59.mean_spearman`, `F59.mean_kappa`.

---

## S1 -- Distributional Characterization

**What it checks.** Full distributional profile of circuit component activations: mean (with bootstrap 95% CI), variance, skewness, kurtosis, sparsity, and effective rank. Compares circuit heads vs non-circuit heads. The primary output is the attribution ratio: mean |logit attribution| of circuit heads divided by mean |logit attribution| of non-circuit heads.

**Applies to.** All tasks with defined circuits.

**Pass threshold.** Attribution ratio > 2.0 indicates that circuit heads contribute substantially more to the output than non-circuit heads.

**Passing means.** Circuit heads are distributional outliers -- they have larger, more variable, and more structured logit attributions than non-circuit heads. The circuit captures genuinely important components.

**Failing means.** Circuit heads are indistinguishable from non-circuit heads in their activation statistics. The circuit may be capturing noise.

**Metric ID:** `S1.distributional_characterization`.

---

## S3 -- Distributional Stability

**What it checks.** Whether per-head logit attribution statistics (mean, variance, skewness) are stable across random data subsets. Splits prompts into 5 random subsets, computes statistics on each, and reports the coefficient of variation (CV) of each statistic across subsets.

**Applies to.** Per-head logit attributions for circuit heads.

**Pass threshold.** CV < 0.20 for mean and variance across subsets. The primary value is the fraction of circuit heads that pass both thresholds.

**Passing means.** The activation statistics are stable properties of the heads, not artifacts of the specific data sample. A head that is important on one subset is also important on another.

**Failing means.** Attribution statistics fluctuate across subsets. The head's behavior is data-dependent. Increase sample size or investigate whether specific prompts drive the instability.

**Metric ID:** `S3.distributional_stability`.

---

## M93 -- Multiple Comparisons Correction

**What it checks.** Whether significant results survive family-wise error correction when all testable claims from all calibrations are pooled. Scans all numbered result JSON files, extracts value-vs-baseline pairs and pass/fail verdicts, derives approximate p-values, then applies both Bonferroni and Benjamini-Hochberg corrections.

**Applies to.** All results across all calibrations and metrics, per task and globally.

**Pass threshold.** BH survival rate > 50% -- more than half of nominally significant results remain significant after correction.

**Passing means.** The finding pattern is robust to the multiple-testing problem. The significant results are not false positives from running many tests.

**Failing means.** Too many nominally significant results fail correction. The apparent significance is inflated by the number of comparisons. Reduce the number of tests or increase sample sizes.

**Metric IDs:** `M93.multiple_comparisons`, `M93.multiple_comparisons_global`.

---

## M98 -- Ablation Method Invariance

**What it checks.** Whether circuit faithfulness scores are consistent across three ablation methods: zero ablation (replace with zeros), mean ablation (replace with mean activation), and resample ablation (replace with activations from a different prompt). Inspired by Miller et al. (2024), "Transformer Circuit Faithfulness Metrics are not Robust."

**Applies to.** Faithfulness scores, per task.

**Pass threshold.** Maximum divergence between any pair of ablation methods < 0.20 (scores agree within 20 percentage points).

**Passing means.** The circuit's faithfulness is a genuine property of the circuit, not an artifact of the ablation method. The result would hold regardless of how non-circuit components are handled.

**Failing means.** Faithfulness depends on how non-circuit heads are ablated. A circuit that scores 0.85 under zero ablation but 0.45 under resample ablation does not have a meaningful faithfulness score. Investigate which ablation method produces the divergence and why.

**Metric ID:** `M98.ablation_invariance`.

---

## E1b -- Method Invariance

**What it checks.** Similar to M98 but uses zero, mean, and noise ablation methods (rather than resample). Reports 1 - max_divergence as the value, so higher is better. Based on "Towards Best Practices of Activation Patching" (ICLR 2024) and Miller et al. (2024).

**Applies to.** Faithfulness scores, per task.

**Pass threshold.** Value > 0.8 (equivalently, max divergence < 0.20).

**Passing means.** The circuit's faithfulness measurement is robust to the choice of ablation method. The value reported is interpretable without caveats about methodology.

**Failing means.** The ablation method matters more than the circuit. Report results with the caveat that they are method-dependent, or investigate which ablation assumptions are violated.

**Metric ID:** `E1b.method_invariance`.

---

## M3b -- Certified Stability

**What it checks.** Whether individual circuit heads are robust contributors across random subsamples of the evaluation set. For N subsamples (each 80% of prompts), measures each head's individual contribution (mean-ablation effect). A head "passes" a subsample if its contribution > 0. Based on Anani et al. (2025), "Certified Circuits."

**Applies to.** All circuit heads, per task.

**Pass threshold.** Classification per head: certified stable (passes in 95%+ subsamples), contingent (50--95%), unstable (<50%). Overall pass: 50%+ of circuit heads are certified stable.

**Passing means.** The circuit's heads are individually robust -- each head contributes positively in nearly all data subsamples. The circuit is not held up by a few heads that only sometimes contribute.

**Failing means.** Too many heads are contingent or unstable. They contribute positively in some subsamples but not others. The circuit may include heads that are not reliably important. Consider pruning unstable heads from the circuit definition.

**Metric ID:** `M3b.certified_stability`.
