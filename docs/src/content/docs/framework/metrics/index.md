---
title: "Metrics"
description: "Concrete, runnable tests that produce evidence about mechanistic claims — 59 in the paper, 64 on this site, organized into six families."
---

# Metrics

A metric is a concrete, runnable test that produces a measurement about a neural network. Metrics are Layer 3 of the pipeline — where empirical contact with the model actually happens. Everything above (evidence families, criteria, validity types, verdicts) depends on what metrics measure and how well they measure it.

The paper defines 59 metrics; this site lists 5 additional ones, for 64 total. They are organized into six families, each producing a distinct kind of signal. Metrics marked *(site-only)* appear on this site but are not in the current paper; all others are in both.

## A. Causal (12 metrics)

Metrics that intervene on the model's computation — [ablation](/mechanistic-validity/glossary/#ablation), [patching](/mechanistic-validity/glossary/#activation-patching), [causal scrubbing](/mechanistic-validity/glossary/#causal-scrubbing) — and measure the downstream effect. Causal metrics answer: **does this component matter for this behavior?**

| Metric | Description |
|---|---|
| <span id="a01"></span>A01 SCM | Structural causal model fit between circuit components and task variables *(site-only)* |
| <span id="a02"></span>A02 Counterfactual DAS | Distributed Alignment Search and IIA for testing causal abstraction hypotheses |
| <span id="a03"></span>A03 Rubin CATE | Conditional average treatment effects across input subpopulations |
| <span id="a06"></span>A06 Mediation | Natural direct and indirect effects decomposition for circuit paths |
| <span id="a07"></span>A07 Granger / TE | Granger causality and transfer entropy for directed information flow *(site-only)* |
| <span id="a08"></span>A08 PID | Partial information decomposition of causal contributions *(site-only)* |
| <span id="a09"></span>A09 MDL / SLT | Minimum description length and singular learning theory for circuit complexity |
| <span id="a10"></span>A10 Regularity / INUS | INUS condition analysis for insufficient but necessary parts of circuits *(site-only)* |
| <span id="a11"></span>A11 Actual Cause | Halpern-Pearl actual causation on specific inputs, not just potential causes |
| <span id="a12"></span>A12 Transportability | Formal conditions for circuit generalization across models and distributions |
| <span id="a13"></span>A13 Causal Discovery | NOTEARS / PC algorithms for learning DAGs over circuit components |
| <span id="a14"></span>A14 Onset–Offset Coupling | Whether a mechanism's measured strength tracks the capability in both directions: rising as the capability is acquired during training, falling as it is removed by unlearning or fine-tuning, and returning if the capability returns |

## B. Structural (9 metrics)

Metrics that analyze the model's weight matrices directly — without running any input through the model. Structural metrics answer: **what does the architecture encode before any data flows?**

| Metric | Description |
|---|---|
| <span id="b01"></span>B01 SVD / Spectral | Singular value decomposition to identify dominant computational directions |
| <span id="b02"></span>B02 Effective Rank | Entropy-based dimensionality as a scalar summary of spectral concentration |
| <span id="b03"></span>B03 OV / QK Decomp. | Decomposing attention heads into OV (what to write) and QK (where to attend) |
| <span id="b04"></span>B04 Weight Alignment | Cosine similarity between principal weight directions across heads |
| <span id="b05"></span>B05 Norm Trajectory | Spectral norm ratios tracking signal amplification through components |
| <span id="b06"></span>B06 Template Distance | Graph-edit and metric distances between circuits discovered for different tasks |
| <span id="b07"></span>B07 Polysemanticity | Measuring whether components encode multiple unrelated features in superposition |
| <span id="b08"></span>B08 ICA / NMF | Independent component analysis for decomposing weights into interpretable parts |
| <span id="b09"></span>B09 Weight Classifier | Training classifiers on weight matrices to predict circuit membership |

## C. Information-theoretic (9 metrics)

Metrics that quantify information flow through the network using entropy, mutual information, and related quantities. Information metrics answer: **how much does this component know about the task variable, and where did that knowledge come from?**

| Metric | Description |
|---|---|
| <span id="c01"></span>C01 Mutual Info. | Total shared information between circuit components and task performance |
| <span id="c02"></span>C02 Conditional MI | Information shared after conditioning on other parts of the circuit |
| <span id="c03"></span>C03 Transfer Entropy | Directed information flow between components across layers |
| <span id="c04"></span>C04 PID | Decomposing shared information into unique, redundant, and synergistic atoms |
| <span id="c05"></span>C05 Info. Bottleneck | How efficiently circuits compress input while preserving task-relevant signal |
| <span id="c06"></span>C06 O-Information | Whether a group of components interacts redundantly or synergistically |
| <span id="c07"></span>C07 Granger Causality | Whether one component's past activations improve prediction of another's future |
| <span id="c08"></span>C08 OCSE | Estimating causal influence between components using only observational data |
| <span id="c09"></span>C09 NOTEARS | Continuous optimization for learning DAGs over circuit components *(site-only)* |

## D. Behavioral (9 metrics)

Metrics that measure the model's input-output behavior under controlled conditions — ablation recovery, distribution matching, and generalization testing. Behavioral metrics answer: **does the proposed circuit actually produce the behavior it is supposed to explain?**

| Metric | Description |
|---|---|
| <span id="d01"></span>D01 Faithfulness | Whether an identified circuit faithfully reproduces the full model's behavior |
| <span id="d02"></span>D02 Logit Diff | How much of the model's logit difference the circuit recovers |
| <span id="d03"></span>D03 KL Divergence | Information-theoretic distance between circuit and full model distributions |
| <span id="d04"></span>D04 CE Delta | Change in cross-entropy loss when the circuit is ablated |
| <span id="d05"></span>D05 Top-K Accuracy | Whether the circuit preserves the model's top-K predicted tokens |
| <span id="d06"></span>D06 Cross-Task | Whether a circuit discovered on one task transfers to a different task |
| <span id="d07"></span>D07 Cross-Scale | Whether circuit structure replicates in larger or smaller models |
| <span id="d08"></span>D08 Prompt Paraphrase | Circuit consistency across semantically equivalent prompt templates |
| <span id="d09"></span>D09 Generalization Gap | Sensitivity of circuit discovery to hyperparameters and methodological choices |

## E. Representational (10 metrics)

Metrics that characterize what information is encoded in the model's internal representations and how it is organized geometrically. Representational metrics answer: **what does this component represent, and how is that representation structured?**

| Metric | Description |
|---|---|
| <span id="e01"></span>E01 DAS-IIA | Whether a learned linear subspace causally encodes a target variable |
| <span id="e02"></span>E02 Linear Probe | Whether a target variable is linearly decodable from intermediate representations |
| <span id="e03"></span>E03 RSA | Comparing representation geometry by correlating pairwise distance matrices |
| <span id="e04"></span>E04 CKA | Kernel alignment for cross-layer and cross-model representation comparison |
| <span id="e05"></span>E05 Subspace Align. | Cosine alignment between SVD-derived principal directions of weight matrices |
| <span id="e06"></span>E06 PCA Dim. | Effective dimensionality of circuit subspaces via activation covariance spectrum |
| <span id="e07"></span>E07 Intrinsic Dim. | True manifold dimensionality, connecting to geometric complexity |
| <span id="e08"></span>E08 Participation Ratio | How many dimensions are effectively active in a representation |
| <span id="e09"></span>E09 Persistent Homology | Topological data analysis detecting loops and voids in activation manifolds |
| <span id="e10"></span>E10 Cross-Task Overlap | Representational structure shared between tasks via IIA transfer |

## F. Measurement (15 metrics)

Metrics that evaluate the measurement properties of other metrics — reliability, validity, and invariance. Measurement metrics answer: **can we trust the measurements that the other metrics produce?**

| Metric | Description |
|---|---|
| <span id="f01"></span>F01 Test–Retest | Whether rerunning the same metric on the same model, task, and intervention gives the same answer |
| <span id="f02"></span>F02 Bootstrap Stability | Whether the metric remains stable under bootstrap resampling of prompts, examples, or activation samples |
| <span id="f03"></span>F03 Seed Variance | How much the metric changes across random seeds for probes, SAEs, optimization, or sampling-based estimators |
| <span id="f04"></span>F04 Checkpoint Variance | Whether the metric is stable across nearby checkpoints or depends on one training snapshot |
| <span id="f05"></span>F05 Prompt Variance | How much the metric changes across prompt templates, paraphrases, lexical choices, or dataset slices |
| <span id="f06"></span>F06 Baseline Separation | Whether the score is distinguishable from random, untrained, shuffled-label, or permuted-circuit baselines |
| <span id="f07"></span>F07 Negative Controls | Whether the metric stays low where the claimed effect should be absent |
| <span id="f08"></span>F08 Positive Controls | Whether the metric detects known-true or synthetic effects of the relevant size |
| <span id="f09"></span>F09 Intervention Robustness | Whether the conclusion survives reasonable intervention variants such as mean ablation, zero ablation, resample ablation, patching, or causal scrubbing |
| <span id="f10"></span>F10 Hyperparam. Sensitivity | Whether the metric changes under reasonable choices of sparsity, probe regularization, thresholds, localization cutoffs, or discovery hyperparameters |
| <span id="f11"></span>F11 Estimator Uncertainty | Confidence intervals, standard errors, permutation tests, or posterior intervals for the reported metric value |
| <span id="f12"></span>F12 Calibration Curve | Whether larger reported scores correspond to larger empirical effects or higher probability of success |
| <span id="f13"></span>F13 Cross-Metric Convergence | Whether independent metrics intended to test the same criterion agree despite different failure modes |
| <span id="f14"></span>F14 Measurement Invariance | Whether the metric behaves consistently across model sizes, tasks, prompt distributions, and intervention contexts |
| <span id="f15"></span>F15 Multiple Comparisons | Whether scores are corrected for the number of components tested, methods tried, or thresholds swept |

## How metrics connect to the rest of the framework

Each metric produces **evidence** (Layer 2) that is evaluated against **criteria** (Layer 4). The criteria are grouped by **validity type** (Layer 5), and the aggregate assessment across validity types produces a **verdict** (Layer 6) tagged with a **description mode**.

A metric alone cannot establish a claim. A claim requires evidence from multiple metrics, evaluated against the criteria appropriate to the validity type being asserted. The dependency order is strict: no skipping from Layer 3 to Layer 6.
