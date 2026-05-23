---
title: "Glossary"
description: "Every term used in the framework, defined in one line."
---

# Glossary

Terms are grouped by domain. Each entry gives a concise definition and, where relevant, a pointer to the page where the term is used in depth.

---

## Framework Structure

**Analytical lens.** A scientific tradition (neuroscience, pharmacology, measurement theory, etc.) whose concepts and standards are adapted to evaluate mechanistic claims. Each lens contributes metrics and interpretation heuristics; no single lens is authoritative. See [Analytical Lenses](/framework/lenses/).

**Calibration.** A quality gate applied to a metric before its output is trusted -- bootstrap stability, convergent validity, measurement invariance, and similar checks. Calibrations do not produce evidence directly; they certify that evidence-producing metrics are functioning correctly. See [Calibrations](/framework/evidence/calibrations).

**Criterion.** One of 27 specific questions a mechanistic claim must answer, organized under five validity types. Each criterion has a short ID (e.g., C4 Minimality, I1 Necessity) and a pass condition defined in terms of metric outputs. See [Validity Types](/framework/validity-types/).

**Description mode.** A level of abstraction at which a mechanistic claim is stated, adapted from Marr's levels. The framework distinguishes computational, algorithmic, representational, and four implementational sub-modes (topographic, connectomic, activation-statistical, functional). See [Description Modes](/framework/description-modes/).

**Evidence family.** One of six categories of empirical evidence: causal (A), structural (B), information-theoretic (C), behavioral (D), representational (E), and measurement-theoretic (F). Each family targets different validity criteria and uses different experimental techniques. See [Evidence Families](/framework/evidence-families/).

**Metric.** A quantitative measurement applied to a model and task, producing a scalar or structured output that bears on one or more criteria. Metrics are the atomic units of evidence. See [Core Metrics](/framework/metrics/) and [MI Extended Metrics](/framework/evidence/mi-metrics-overview).

**Protocol.** A curated bundle of metrics and calibrations organized around a specific validity question, interpreted through a theoretical framework. Running a protocol is optional -- metrics alone are sufficient -- but protocols provide structured depth. See [Protocols](/framework/evidence/protocols).

**Synthesis protocol.** A higher-order analysis that consumes outputs from multiple protocols and extracts patterns no single protocol can see -- consensus estimates, functional parcellations, stability metrics. See [Synthesis Protocols](/framework/evidence/synthesis-protocols).

**Verdict.** The framework's overall assessment of a mechanistic claim, assigned after evaluating criteria across all five validity types. Verdicts range from Proposed (tier 1) through Validated (tier 5), with Underdetermined and Disconfirmed as non-tier outcomes. See [Verdicts](/framework/verdicts/).

---

## Validity Types

**Construct validity.** Whether the claim defines a coherent, falsifiable entity with clear boundaries. Criteria: C1 Falsifiability, C2 Structural plausibility, C3 Task specificity, C4 Minimality, C5 Convergent validity. See [Construct](/framework/validity-types/construct).

**Measurement validity.** Whether the metrics used to evaluate the claim are reliable, calibrated, and measure what they purport to measure. Criteria: M1 Reliability, M2 Invariance, M3 Baseline separation, M4 Sensitivity, M5 Calibration, M6 Construct coverage. See [Measurement](/framework/validity-types/measurement).

**Internal validity.** Whether the claimed causal mechanism is genuine -- necessary, sufficient, specific, consistent, and not confounded. Criteria: I1 Necessity, I2 Sufficiency, I3 Specificity, I4 Consistency, I5 Confound control. See [Internal](/framework/validity-types/internal).

**External validity.** Whether the claim generalizes beyond the specific conditions under which it was established. Criteria: E1 Intervention reach, E2 Graded response, E3 Selectivity, E4 Effect magnitude, E5 Robustness, E6 Cross-architecture. See [External](/framework/validity-types/external).

**Interpretive validity.** Whether the narrative accompanying the claim is appropriate to the evidence -- not overclaiming, not underclaiming, and honest about scope. Criteria: V1 Level declaration, V2 Level-evidence match, V3 Narrative coherence, V4 Alternative exclusion, V5 Scope honesty. See [Interpretive](/framework/validity-types/interpretive).

---

## Construct Validity Criteria

**C1 Falsifiability.** The claim specifies conditions under which it would be considered false. A circuit claim that cannot fail any test is unfalsifiable and therefore unscientific.

**C2 Structural plausibility.** The claimed entity has a weight-space signature consistent with the proposed computation -- e.g., a "copying head" has high OV copy scores.

**C3 Task specificity.** The claim identifies a bounded set of inputs on which the mechanism operates, rather than claiming universal relevance.

**C4 Minimality.** The circuit contains no components whose removal leaves performance intact. Every component must be load-bearing.

**C5 Convergent validity.** Independent methods (causal, structural, representational) agree on circuit membership. Agreement across methods is stronger evidence than agreement within a single method.

---

## Measurement Validity Criteria

**M1 Reliability.** The metric produces consistent results across repeated measurements -- different random seeds, prompt samples, and runs.

**M2 Invariance.** The metric's ranking of components is stable across prompt templates, not just across samples from the same template.

**M3 Baseline separation.** The metric's scores for circuit components are distinguishable from scores for random or non-circuit components.

**M4 Sensitivity.** The metric detects known differences -- it assigns different scores to components with known different roles.

**M5 Calibration.** Reported confidence intervals have correct coverage; a 95% CI contains the true value 95% of the time.

**M6 Construct coverage.** The set of metrics used collectively addresses the claim's full scope, not just one aspect.

---

## Internal Validity Criteria

**I1 Necessity.** Ablating the circuit degrades task performance. The circuit is required for the behavior.

**I2 Sufficiency.** The circuit alone (with non-circuit components ablated or replaced) reproduces the behavior.

**I3 Specificity.** The circuit's effect is concentrated on the claimed task and does not equally affect unrelated tasks.

**I4 Consistency.** The circuit produces the same effect across different valid measurements of the same construct.

**I5 Confound control.** The observed effect is not attributable to methodological artifacts -- off-manifold activations, backup circuits, or shared variance with non-circuit components.

---

## External Validity Criteria

**E1 Intervention reach.** The fraction of the model's attention heads (or analogous components) whose ablation produces a measurable effect on the task.

**E2 Graded response.** Ablating progressively more of the circuit produces monotonically increasing degradation -- a dose-response relationship.

**E3 Selectivity.** The circuit's ablation effect on the target task is larger than its effect on control tasks, expressed as a ratio.

**E4 Effect magnitude.** The standardized size of the circuit's contribution, measured in units like Cohen's d or logit-diff recovery fraction.

**E5 Robustness.** The claim holds across prompt paraphrases, held-out examples, and distribution shifts.

**E6 Cross-architecture.** The claim transfers across model scales or architectures, or the boundaries of non-transfer are characterized.

---

## Interpretive Validity Criteria

**V1 Level declaration.** The claim explicitly states which description mode it operates at (computational, algorithmic, representational, or implementational).

**V2 Level-evidence match.** The evidence type matches the declared level -- implementational claims require causal evidence, not just representational.

**V3 Narrative coherence.** The mechanistic story is internally consistent and does not invoke contradictory sub-mechanisms.

**V4 Alternative exclusion.** Competing explanations (simpler circuits, confounded signals, backup mechanisms) have been tested and ruled out.

**V5 Scope honesty.** The claim does not generalize beyond the conditions under which it was tested.

---

## Causal Methods and Metrics

**Ablation.** Replacing a component's output with a reference value (zero, mean, or random) and measuring the effect on task performance. Mean ablation uses the dataset mean; zero ablation uses the zero vector.

**Activation patching.** Replacing a component's activation in a clean run with its activation from a corrupted run (or vice versa), isolating the component's causal contribution to the output difference.

**ACDC (Automated Circuit DisCovery).** An algorithm that iteratively prunes edges from the computational graph by testing whether each edge's removal changes model output beyond a threshold. Conmy et al. (2023).

**Causal scrubbing.** A protocol that tests whether a proposed computational graph is consistent with the model's behavior by resampling activations according to the graph's structure. Chan et al. (2022).

**DAS/IIA (Distributed Alignment Search / Interchange Intervention Accuracy).** Learns a linear subspace such that swapping activations within that subspace between two inputs swaps the model's behavior correspondingly. The accuracy of this swap is the IIA score. Geiger et al. (2024).

**EAP (Edge Attribution Patching).** Estimates each edge's contribution to the output by computing the gradient of the output with respect to the edge's activation, scaled by the activation difference between clean and corrupted runs. Syed et al. (2023).

**Mediation analysis.** Decomposes the total causal effect of a variable into direct and indirect (mediated) paths, quantifying what fraction of the effect flows through a specific intermediate component.

**Path patching.** A variant of activation patching that intervenes on a specific computational path (e.g., head 9.1's output as received by head 10.0) rather than on a component's total output.

**Sigma ablation.** Ablation calibrated in units of the activation's standard deviation rather than absolute magnitude, controlling for the natural scale of each component's activations.

---

## Structural Methods and Metrics

**Effective rank.** The number of significant singular values in a weight matrix, measuring the dimensionality of the subspace the matrix actually uses. Computed as the exponential of the entropy of the normalized singular value distribution.

**OV circuit / QK circuit.** The output-value and query-key compositional circuits within an attention head. The OV circuit ($W_V W_O$) determines what information is written to the residual stream; the QK circuit ($W_Q^T W_K$) determines which positions attend to which.

**SVD (Singular Value Decomposition).** Factorization of a weight matrix into orthogonal components ranked by magnitude, revealing the principal directions of computation.

**Weight alignment.** The cosine similarity between weight matrices of different heads or layers, measuring whether they operate in similar subspaces.

---

## Information-Theoretic Methods

**Granger causality.** A statistical test of whether past values of variable A improve prediction of variable B beyond B's own history. Applied to neural networks by treating layer depth as time.

**Mutual information.** The reduction in uncertainty about one variable given knowledge of another, quantifying the statistical dependence between two representations.

**O-information.** A multivariate measure distinguishing redundancy-dominated from synergy-dominated interactions among three or more variables.

**PID (Partial Information Decomposition).** Decomposes the mutual information between source variables and a target into unique, redundant, and synergistic components, revealing how information is distributed across circuit elements.

**Transfer entropy.** A directed, non-parametric measure of information flow from one time series to another, generalizing Granger causality to nonlinear relationships.

---

## Representational Methods

**CKA (Centered Kernel Alignment).** A similarity measure between two sets of representations that is invariant to orthogonal transformations and isotropic scaling, used to compare representations across layers, models, or tasks.

**Linear probe.** A linear classifier trained on a model's internal activations to test whether a specific feature (e.g., grammatical number, entity identity) is linearly decodable from those activations.

**Probing.** Training a simple classifier on internal representations to test what information is present. The probe's accuracy measures information availability, not necessarily information use.

**RSA (Representational Similarity Analysis).** Compares two representation spaces by computing the correlation between their pairwise distance matrices, testing whether two representations encode the same relational structure.

---

## Decomposition and Feature Methods

**CLT (Circuit-Level Tracing).** A method that attributes model outputs to specific features in a sparse autoencoder by tracing contributions through the computational graph. Produces an attribution graph with error nodes quantifying unexplained variance.

**Crosscoder.** A sparse autoencoder variant trained jointly on activations from multiple models, learning shared features that transfer across architectures.

**SAE (Sparse Autoencoder).** A neural network trained to reconstruct model activations through a sparse bottleneck, decomposing activations into interpretable features. Quality is measured by reconstruction fidelity, sparsity, and feature interpretability.

**Transcoder.** A sparse autoencoder variant that maps from an MLP layer's input to its output (rather than reconstructing activations), learning interpretable approximations of the MLP's computation.

---

## Steering and Editing Methods

**CAA (Contrastive Activation Addition).** Computes the mean activation difference between contrastive prompt pairs (e.g., positive vs. negative sentiment) and adds this steering vector to the residual stream at inference, shifting model behavior along the contrast direction. Panickssery et al. (2024).

**LEACE (Least-squares Concept Erasure).** Projects out a linear subspace from model activations to remove information about a target concept, testing whether the concept is necessary for a downstream behavior. Belrose et al. (2023).

**RepE (Representation Engineering).** Identifies linear directions in activation space corresponding to high-level concepts and uses them to read or control model behavior. Zou et al. (2023).

---

## Pharmacology Concepts

**Dose-response curve.** The relationship between the fraction of a circuit ablated and the resulting performance degradation. A genuine causal mechanism should produce monotonic, graded degradation.

**Effect size.** The magnitude of an intervention's effect in standardized units (Cohen's d, Glass's delta, Hedges' g), making results comparable across tasks and models.

**Recovery score.** The fraction of the clean model's performance retained after an intervention -- logit-diff recovery, cross-entropy recovery, or top-k accuracy recovery.

**Selectivity ratio.** The ratio of a circuit's effect on the target task to its effect on a control task. High selectivity means the circuit is functionally specific rather than globally important.

---

## Neuroscience Concepts

**Constitutive relevance.** A component is constitutively relevant to a mechanism if it is both a part of the mechanism and makes a difference to the mechanism's behavior -- Craver's mutual manipulability criterion adapted to neural networks.

**Double dissociation.** Evidence that two components are functionally independent: ablating A impairs task X but not Y, while ablating B impairs Y but not X. Stronger than single dissociation.

**Functional parcellation.** Dividing a system into non-overlapping functional groups based on convergent evidence from multiple independent methods.

**Lesion study.** The neuroscience analogue of ablation -- removing or inactivating a brain region and observing the behavioral deficit. Circuit ablation studies are computational lesion studies.

**Single dissociation.** Ablating component A impairs task X. Does not establish that A is specific to X -- A might be necessary for all tasks.

---

## Causal Inference Concepts

**Actual causation.** Halpern and Pearl's formal definition: C is an actual cause of E if there exists a contingency (setting of other variables) under which changing C changes E. Distinguishes actual causes from merely background conditions.

**Counterfactual.** A hypothetical scenario in which one variable is set to a different value while all else is held fixed. Causal claims are tested by comparing factual and counterfactual outcomes.

**INUS condition.** An Insufficient but Necessary part of an Unnecessary but Sufficient condition -- a component that is individually insufficient but contributes essentially to one sufficient set of conditions for the outcome. Mackie (1965).

**SCM (Structural Causal Model).** A directed acyclic graph plus functional equations specifying how each variable is determined by its parents, encoding the causal structure of a system. Pearl (2009).

**Transportability.** Whether a causal relationship established in one setting (model, task, distribution) can be validly transferred to another setting, and what assumptions are required for the transfer.

---

## Measurement Theory Concepts

**Baseline separation.** The degree to which a metric's scores for circuit components differ from scores for non-circuit or random components -- the signal-to-noise ratio of the measurement.

**Bootstrap stability.** The consistency of a metric's output across bootstrap resamples of the input data, quantifying sensitivity to sampling noise.

**Convergent validity.** The degree to which independent measurement methods agree on the same construct -- metrics from different evidence families should rank components similarly if they are measuring the same thing.

**Discriminant validity.** The degree to which a metric differentiates between constructs that should be different -- a circuit metric should not give the same scores to circuits for different tasks.

**ICC (Intraclass Correlation Coefficient).** A reliability measure for continuous ratings, quantifying the fraction of total variance attributable to true differences between items rather than measurement error.

**Measurement invariance.** A metric's ranking of components is stable across measurement conditions -- different prompt templates, model checkpoints, or evaluation datasets.

---

## Philosophy of Science Concepts

**Falsifiability.** A claim is scientific only if it specifies conditions under which it would be considered false. Circuit claims must state what pattern of evidence would disconfirm them.

**Marr's levels.** David Marr's three levels of analysis for information-processing systems: computational (what problem is solved), algorithmic (what representations and procedures are used), and implementational (how the algorithm is physically realized). The framework extends this with sub-modes.

**Mechanistic explanation.** An explanation that describes how a system's components and their interactions produce the phenomenon, rather than merely predicting or correlating with it.

**Model-to-mechanism mapping (3M).** The principle that a model explains a phenomenon only if there is a mapping from the model's variables to the mechanism's components that preserves causal and organizational relationships. Kaplan and Craver (2011).

---

## Neural Network Architecture

**Attention head.** A component within a transformer layer that computes a weighted combination of value vectors, where weights are determined by query-key similarity. The fundamental unit of circuit analysis in attention-based models.

**Attention pattern.** The matrix of attention weights produced by a head for a given input, indicating which positions attend to which other positions.

**Circuit.** A subgraph of the model's computational graph -- a subset of heads, MLP neurons, and edges -- hypothesized to implement a specific computation.

**Embedding / Unembedding.** The embedding matrix maps tokens to vectors in the residual stream; the unembedding matrix maps residual stream vectors back to logits over the vocabulary. Together they define the model's input-output interface.

**Logit.** The unnormalized score assigned to each vocabulary token before the softmax; the model's raw prediction. Logit difference (logit-diff) between the correct and incorrect token is a standard measure of circuit effect.

**Residual stream.** The vector that accumulates contributions from all attention heads and MLP layers across the model's depth. Each component reads from and writes to this shared communication channel.

---

## Tasks and Datasets

**IOI (Indirect Object Identification).** A benchmark task where the model must predict which name is the indirect object in sentences like "When Mary and John went to the store, John gave a drink to" (answer: Mary). The most extensively studied circuit in mechanistic interpretability.

**Induction.** A task requiring the model to complete repeated patterns -- given "AB...AB", predict "B" after the second "A". Induction heads are a well-characterized two-head circuit (previous-token head + induction head).

**Greater-Than.** A task where the model must predict that a year is greater than a preceding year in temporal expressions. Tests ordinal reasoning circuits.

---

## Synthesis and Aggregation

**Borda count.** A rank aggregation method that assigns each item a score equal to the number of items ranked below it, then sums across rankers. Used in synthesis protocol S03.

**Dawid-Skene.** An EM algorithm that treats each protocol as a noisy annotator and jointly estimates true labels and per-annotator reliability. Used in synthesis protocol S02.

**RRA (Robust Rank Aggregation).** A statistical method that tests whether an item appears near the top of multiple ranked lists more often than expected under a uniform null. Kolde et al. (2012). Used in synthesis protocol S03.

**Wasserstein distance.** A metric between probability distributions that accounts for the geometry of the underlying space, measuring the minimum "work" required to transform one distribution into another. Used in synthesis protocol S06 to quantify circuit stability.

---

## Naming Convention

**CRIT.** Typed prefix for criteria (e.g., CRIT-C4 for Minimality). 27 total.

**MET.** Typed prefix for metrics (e.g., MET-activation-patching). Full kebab-case slugs, never abbreviated.

**CAL.** Typed prefix for calibrations (e.g., CAL-01 for Bootstrap Stability). 16 total.

**PROT.** Typed prefix for protocols (e.g., PROT-A01 for Pearl SCM). Letter indicates evidence family.

**SYN.** Typed prefix for synthesis protocols (e.g., SYN-02 for Dawid-Skene). 9 total.

See [Naming Convention](/framework/evidence/naming-convention) for the full disambiguation system.

---

## Acronyms

| Acronym | Expansion |
|---|---|
| ACDC | Automated Circuit DisCovery |
| AUROC | Area Under the Receiver Operating Characteristic curve |
| CAA | Contrastive Activation Addition |
| CKA | Centered Kernel Alignment |
| CLT | Circuit-Level Tracing |
| DAS | Distributed Alignment Search |
| EAP | Edge Attribution Patching |
| ICC | Intraclass Correlation Coefficient |
| IIA | Interchange Intervention Accuracy |
| INUS | Insufficient but Necessary part of an Unnecessary but Sufficient condition |
| IOI | Indirect Object Identification |
| LEACE | Least-squares Concept Erasure |
| MDL | Minimum Description Length |
| MI | Mechanistic Interpretability |
| NMF | Non-negative Matrix Factorization |
| OOD | Out-of-Distribution |
| OV | Output-Value (circuit) |
| PID | Partial Information Decomposition |
| QK | Query-Key (circuit) |
| RepE | Representation Engineering |
| RRA | Robust Rank Aggregation |
| RSA | Representational Similarity Analysis |
| SAE | Sparse Autoencoder |
| SCM | Structural Causal Model |
| SLT | Statistical Learning Theory |
| SVD | Singular Value Decomposition |
| TE | Transfer Entropy |
