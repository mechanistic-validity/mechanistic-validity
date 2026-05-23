---
title: "MI Benchmarks Metrics"
description: "Benchmark metrics from the mechanistic interpretability lens: AxBench, SAEBench, CE-Bench, and MIB evaluations."
---

# MI Benchmarks Metrics

This page documents the six benchmark metrics implemented in `mechval_v2.core.mechanistic_interpretability.benchmarks`. Each benchmark evaluates interpretability artifacts (SAEs, features, circuits) against standardized external evaluation protocols from the literature. These benchmarks contextualize mechanistic claims within community baselines.

---

## B20 -- AxBench Concept Detection and Steering

**ID.** `B20.axbench` | **File.** `101_axbench.py`

**What it evaluates.** Whether artifact features detect concepts better than simple baselines (DiffMean, random directions) and whether feature directions steer model behavior.

**Evidence family.** External (Behavioral)

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| Detection AUROC | Best-feature AUROC at separating positive vs. negative concept examples | `> 0.6` |
| DiffMean baseline AUROC | AUROC of the mean-difference direction (supervised baseline) | comparison |
| Steering effect | Log-probability change on concept-related continuations when adding/subtracting the feature direction | `> 0.0` |

**What it establishes.** Whether an artifact's features provide a practical advantage over the simplest possible baselines for concept detection and behavioral steering. AxBench (Huang et al., ICML 2025 Spotlight) showed that SAEs are not competitive with prompting, finetuning, or DiffMean on these tasks. A passing score means the artifact exceeds the baseline threshold, not that it outperforms all alternatives.

**What it does not establish.** That the features capture genuine computational structure. High detection AUROC is achievable by any direction correlated with the concept -- it does not require the direction to be mechanistically meaningful or causally relevant. Steering effectiveness confirms behavioral impact but not that the feature is used by the model's native computation.

**Reference.** Huang et al. (2025), "AxBench: Steering and Concept Detection Benchmarks for Language Models", ICML 2025 Spotlight.

---

## B22 -- Failure Prediction

**ID.** `B22.failure_prediction` | **File.** `103_failure_prediction.py`

**What it evaluates.** Whether artifact features can predict model failures (incorrect outputs), not just successes, using per-feature AUROC at separating success vs. failure prompts.

**Evidence family.** External (Behavioral)

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| Best feature AUROC | AUROC of the single best-separating feature at distinguishing success from failure | `> 0.6` |
| Best neuron AUROC | Same metric on raw activation dimensions (baseline) | comparison |
| Combined top-k AUROC | AUROC using mean of top-k most discriminative features | `> 0.6` |

**What it establishes.** Whether individual features carry information about when the model will fail, not just what it knows. Mathew et al. (AAAI 2026) found that linear probes on learned features achieve near-chance performance at predicting failures, suggesting features encode capability but not failure modes.

**What it does not establish.** That failure-predictive features are mechanistically informative. A feature that fires on "hard" inputs may predict failure by detecting input difficulty rather than identifying the mechanism of failure.

**Reference.** Mathew et al. (2026), "Why It Failed: Probing Failure Modes of Learned Representations", AAAI 2026.

---

## C17 -- MIB Causal Variable Localization

**ID.** `C17.mib_causal_variable` | **File.** `103_mib_causal_variable.py`

**What it evaluates.** Whether artifact features localize causal variables better than raw neurons, with DAS (Distributed Alignment Search) providing a supervised upper bound.

**Evidence family.** Internal (Causal)

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| Feature AUROC | Best per-feature AUROC at separating positive vs. negative prompts (by logit-diff sign) | comparison |
| Neuron AUROC | Best per-neuron AUROC (baseline) | comparison |
| DAS AUROC | AUROC of the supervised mean-difference direction (ceiling) | comparison |
| Normalized score | `(feature - neuron) / (DAS - neuron)` -- fraction of the neuron-to-DAS gap closed by features | `> 0.1` |

**What it establishes.** Whether the artifact's decomposition provides a measurable advantage over raw neurons for localizing causal variables. Mueller et al. (ICML 2025) found that SAE features are NOT systematically better than raw neurons, making this a calibration benchmark rather than a validation test. A normalized score near 0 means features add no value over neurons; a score near 1 means features match the supervised DAS ceiling.

**What it does not establish.** That features are causally meaningful even if they beat neurons. The AUROC metric measures linear separability of a binary classification, which does not require the feature to be a genuine causal variable -- it only requires correlation with the task outcome.

**Reference.** Mueller et al. (2025), "Missed Insights, Burned Budgets: Evaluating Sparse Autoencoders for Causal Variable Localization", ICML 2025.

---

## B23 -- Evaluation Awareness Format Control

**ID.** `B23.eval_awareness` | **File.** `104_eval_awareness.py`

**What it evaluates.** Whether artifact features track evaluation FORMAT (benchmark-canonical MCQ structure) rather than evaluation CONTENT (knowledge-testing questions), using a 2x2 format-by-content factorial design.

**Evidence family.** External (Behavioral)

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| Format effect | Mean activation difference between eval-format and free-format conditions | diagnostic |
| Content effect | Mean activation difference between eval-content and non-eval-content conditions | diagnostic |
| Format ratio | `|format| / (|format| + |content| + eps)` -- fraction of feature activation driven by format | `< 0.7` per feature |
| Format confound fraction | Fraction of features with `format_ratio > 0.7` | `< 0.5` |

**What it establishes.** Whether claimed "evaluation awareness" features actually track MCQ formatting artifacts (lettered options, benchmark-canonical structure) rather than genuine awareness that the model is being evaluated. Devbunova (ICLR 2026 Workshop) showed that probe-based evidence for evaluation awareness largely collapses under format control.

**What it does not establish.** That features passing the format control are genuinely tracking evaluation awareness. A feature could respond to content features (e.g., question marks, academic vocabulary) that correlate with evaluation content without representing awareness per se.

**Reference.** Devbunova (2026), "Evaluation Awareness in LLMs: Probes Track Format, Not Awareness", ICLR 2026 Workshop.

---

## EX10 -- CE-Bench Contrastive Evaluation

**ID.** `EX10.ce_bench` | **File.** `EX10_ce_bench.py`

**What it evaluates.** Whether artifact features respond selectively to specific semantic dimensions, measured via contrastive minimal pairs that differ in exactly one dimension (gender, sentiment, tense, location, quantity, formality, agency, certainty).

**Evidence family.** Measurement

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| Contrastive score | Cohen's-d-like effect size between feature activations on paired stories | `> 0.3` |
| Independence score | `1 - |corr|` with all other contrastive dimensions (a feature selective for gender should not also separate sentiment) | high is better |
| Mean contrastive score | Average across features | `> 0.3` |

**What it establishes.** Whether features detect specific semantic dimensions rather than responding to broad distributional properties. CE-Bench (Gulko et al., BlackboxNLP 2025) is fully deterministic and requires no LLM judge, making it a reliable complement to autointerp-based evaluations.

**What it does not establish.** That semantically selective features are causally relevant. A feature that responds selectively to gender in its activations may not causally influence the model's gender-related outputs. Selectivity is a necessary but not sufficient condition for causal relevance.

**Reference.** Gulko et al. (2025), "CE-Bench: A Contrastive Evaluation Benchmark for Feature Interpretability", BlackboxNLP 2025.

---

## EX9 -- SAEBench Multi-Metric Evaluation

**ID.** `EX9.saebench` | **File.** `EX9_saebench.py`

**What it evaluates.** Overall SAE quality across five sub-metrics spanning proxy, interpretability, and disentanglement dimensions.

**Evidence family.** Measurement / External

**Key metrics.**

| Sub-metric | What it measures | Pass threshold |
|---|---|---|
| EX9a Reconstruction loss | MSE between original activations and SAE reconstruction | `< 0.1` |
| EX9b L0 sparsity | Mean number of active features per token | `< 50` |
| EX9c Explained variance | `1 - var(residual) / var(original)` | `> 0.85` |
| EX9d Feature detection | Per-feature AUROC on built-in concept pairs (sentiment, formal/informal, scientific/everyday) | `> 0.6` |
| EX9e Feature disentanglement | Mean pairwise |cosine similarity| of sampled decoder direction pairs | `< 0.1` |

**What it establishes.** Whether an SAE meets basic quality thresholds across multiple evaluation axes. Karvonen et al. (ICML 2025) showed that proxy metric gains (better reconstruction, lower L0) do NOT reliably translate to practical performance improvements on downstream tasks. SAEBench calibrates expectations: passing all sub-metrics is necessary but not sufficient for useful features.

**What it does not establish.** That the SAE captures genuine computational structure. An SAE can achieve low reconstruction loss, high sparsity, and reasonable detection AUROCs while learning an arbitrary basis that does not correspond to the model's internal computation. The benchmarks test output quality, not mechanistic correspondence.

**Reference.** Karvonen et al. (2025), "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Models", ICML 2025.

---

## Summary Table

| ID | Name | File | Evidence Family | Primary Metric | Threshold |
|---|---|---|---|---|---|
| B20 | AxBench | `101_axbench.py` | External | Detection AUROC | `> 0.6` |
| B22 | Failure Prediction | `103_failure_prediction.py` | External | Best feature AUROC | `> 0.6` |
| C17 | MIB Causal Variable | `103_mib_causal_variable.py` | Internal | Normalized score | `> 0.1` |
| B23 | Eval Awareness | `104_eval_awareness.py` | External | Format confound fraction | `< 0.5` |
| EX9 | SAEBench | `EX9_saebench.py` | Measurement | Multi-metric composite | varies |
| EX10 | CE-Bench | `EX10_ce_bench.py` | Measurement | Mean contrastive score | `> 0.3` |

---

## Reading the Benchmarks Together

### Proxy vs. practical performance

The central lesson from SAEBench and AxBench is that proxy metrics (reconstruction, L0, explained variance) do not predict practical utility (concept detection, steering, failure prediction). When evaluating an artifact:

- **Start with SAEBench** (EX9) to confirm basic quality thresholds are met.
- **Then run AxBench** (B20) to test whether features beat simple baselines on practical tasks.
- **Check CE-Bench** (EX10) for semantic selectivity without LLM-judge confounds.
- **Use MIB** (C17) to calibrate against the neuron baseline and the DAS ceiling.

### Failure modes to watch for

| Pattern | What it means |
|---|---|
| SAEBench passes, AxBench fails | Proxy quality is high but features are not practically useful -- the SAE may have learned an arbitrary basis |
| AxBench passes, MIB fails | Features detect concepts but do not localize causal variables -- correlation without causation |
| All pass, Eval Awareness fails | Features track formatting artifacts, not genuine evaluation content |
| MIB normalized score near 0 | Features add no value over raw neurons for causal localization |
| Failure Prediction near chance | Features encode capability but not failure modes |

## Relationship to Other Pages

For the **evaluation** metrics (circuit faithfulness, safety, transcoders, crosscoders, CLT), see [MI Evaluation Metrics](/framework/lenses/core/mi-evaluation-metrics). For the genetics lens benchmarks and protocols, see [Genetics Metrics](/framework/lenses/supporting/genetics-metrics). For the overall MI lens framework, see [Mechanistic Interpretability](/framework/lenses/core/mechanistic-interpretability).
