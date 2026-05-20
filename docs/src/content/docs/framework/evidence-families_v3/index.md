---
title: "Evidence Families"
description: "Six kinds of signal that metrics can produce — classified by the nature of the evidence, not the tool that generates it."
---

# Evidence Families

An evidence family classifies a metric's output by the **kind of signal** it produces, independent of the specific tool used to generate it. Evidence families are Layer B of the taxonomy, sitting between Metrics (Layer A) and Criteria (Layer C). Metrics produce raw measurements; evidence families describe what *type* of epistemic content those measurements carry; criteria then evaluate whether that content meets the bar for a validity claim.

The classification matters because **convergent validity** — two independent lines of evidence agreeing — is the strongest form of empirical support for a mechanistic claim. But not all agreement is equally informative. Two causal metrics agreeing (e.g., activation patching and causal scrubbing both implicate the same heads) shares failure modes: both are sensitive to the same confounds (backup circuits, distributed computation). Two metrics from *different* families agreeing (e.g., a causal ablation and a weight-space structural analysis both implicate the same component) is substantially stronger, because the two families have structurally different failure modes. A confound that fools a causal metric (backup recovery) cannot fool a structural metric (which never runs a forward pass), and vice versa.

The six evidence families map one-to-one with the six metric families, but the distinction is between the *tool* (metric) and the *signal type* (evidence family). A single metric can sometimes contribute evidence to multiple families — for example, DAS/IIA produces both representational evidence (a direction encodes a variable) and causal evidence (intervening on that direction changes behavior). The family labels attach to the *evidence produced*, not to the metric itself.

The families are not ranked. They are complementary perspectives on the same underlying system, each with characteristic strengths and blind spots. No single family is sufficient; no single family is dispensable.

## 1. Causal

**Signal type:** Changes in model behavior caused by interventions on its computation.

Causal evidence answers the question: *does this component matter for this behavior?* The signal is a measured difference between the model's output under normal conditions and its output when a specific component is ablated, patched, or otherwise intervened upon.

**Characteristic strengths:** Establishes necessity and sufficiency directly. Can distinguish load-bearing components from correlated bystanders. Supports counterfactual reasoning about mechanism.

**Characteristic blind spots:** Vulnerable to backup circuits (components that compensate after ablation), intervention artifacts (the intervention itself introducing out-of-distribution states), and granularity limits (ablation at the wrong level obscures fine-grained structure).

**Metrics that produce this evidence:** Activation patching, path patching, causal scrubbing, counterfactual interventions (DAS/IIA when used causally), ablation studies, causal discovery algorithms.

## 2. Structural

**Signal type:** Properties of the model's weight matrices, analyzed without running any input through the network.

Structural evidence answers the question: *what does the architecture encode before any data flows?* The signal comes from direct mathematical analysis of learned parameters — their spectra, rank, alignment, composition, or factored structure.

**Characteristic strengths:** Cannot be confounded by runtime compensation or distribution effects. Reveals the model's *capacity* (what it could compute) independently of any particular input. Enables analysis at arbitrary scale without requiring a dataset.

**Characteristic blind spots:** Cannot distinguish used capacity from unused capacity. A weight-space structure that exists may never be activated on natural data. Cannot establish behavioral relevance without supplementary evidence from another family.

**Metrics that produce this evidence:** SVD/spectral analysis, effective rank, OV/QK circuit composition scores, weight alignment measures, norm trajectories, template matching, blind source separation (ICA/NMF).

## 3. Representational

**Signal type:** Information encoded in internal activations and its geometric organization.

Representational evidence answers the question: *what does this component represent, and how is that representation structured?* The signal is a demonstrated correspondence between internal activation patterns and external variables, together with a characterization of the geometric form of that correspondence (linear direction, subspace, nonlinear manifold).

**Characteristic strengths:** Identifies the *content* carried by intermediate computations. Can reveal hidden structure invisible to behavioral tests. Connects internal states to interpretable variables.

**Characteristic blind spots:** Decodability does not imply use — a representation may be present but ignored by downstream computation. Probe flexibility confounds (a powerful enough decoder can "find" any variable). Cannot establish causal role without interventional evidence.

**Metrics that produce this evidence:** Linear probing, DAS/IIA (representational component), RSA, CKA, subspace alignment, PCA/dimensionality analysis, intrinsic dimension estimation, persistent homology.

## 4. Behavioral

**Signal type:** The model's input-output behavior under controlled conditions.

Behavioral evidence answers the question: *does the proposed circuit actually produce the behavior it is supposed to explain?* The signal is a measured correspondence between a circuit's outputs (when run in isolation or under ablation) and the full model's outputs on held-out inputs that test specific aspects of the claimed functionality.

**Characteristic strengths:** Directly tests the claim's real-world consequence. Reveals generalization failures and overfitting to discovery prompts. Provides the ground truth that all other evidence types ultimately serve.

**Characteristic blind spots:** Behavioral equivalence does not imply mechanistic equivalence — different circuits can produce the same behavior for different reasons. Sensitive to prompt selection and distribution shift. A circuit that reproduces behavior perfectly on the test set may do so via a different mechanism than the one claimed.

**Metrics that produce this evidence:** Faithfulness scores, logit diff recovery, KL divergence (circuit vs. full model), cross-task transfer, cross-scale transfer, prompt paraphrase robustness, generalization gap analysis, distribution-shift benchmarks.

## 5. Information-Theoretic

**Signal type:** Quantified information flow through the network, measured in bits or related units.

Information-theoretic evidence answers the question: *how much does this component know about the task variable, and where did that knowledge come from?* The signal is a quantity — mutual information, transfer entropy, or a decomposition thereof — that measures the statistical dependence between internal states and external variables, or between internal states at different points in the computation.

**Characteristic strengths:** Provides a principled, unit-bearing measure of information content that is (in principle) independent of the decoder used to access it. Decomposition methods (PID) can separate redundant, unique, and synergistic contributions — distinctions invisible to other families.

**Characteristic blind spots:** Estimation is notoriously difficult in high dimensions; practical estimators introduce bias. Cannot distinguish information that is *used* from information that is merely *present*. High mutual information between a component and a variable does not establish that the component *processes* that variable — it may simply copy it passively.

**Metrics that produce this evidence:** Mutual information estimation, conditional MI, transfer entropy, partial information decomposition (PID), O-information, information bottleneck methods, Granger causality (information-theoretic formulation), structure-learning algorithms (NOTEARS).

## 6. Measurement-Theoretic

**Signal type:** Meta-evidence about the reliability, calibration, and validity of other metrics' measurements.

Measurement-theoretic evidence answers the question: *can we trust the measurements that the other metrics produce?* The signal is a property of a metric or measurement procedure — its stability under repetition, its invariance across conditions, its agreement with other metrics measuring the same construct, or its discrimination between constructs that should differ.

**Characteristic strengths:** Establishes whether findings from other families are trustworthy before interpreting them. Detects spurious results caused by metric instability, overfitting, or threshold sensitivity. Enables principled comparison across studies and methods.

**Characteristic blind spots:** Cannot generate first-order evidence about the model — it evaluates evidence, not the model itself. A perfectly reliable metric can reliably measure the wrong thing. Measurement-theoretic evidence is necessary but never sufficient for a mechanistic claim.

**Metrics that produce this evidence:** Bootstrap stability analysis, seed variance, convergent validity testing (MTMM), discriminant validity testing, internal consistency (split-half), inter-rater agreement, measurement invariance testing, incremental validity analysis.

## How evidence families enable convergent validity

The primary function of the family classification is to structure convergent validity assessments. When evaluating a mechanistic claim, the analyst asks:

1. **How many evidence families support the claim?** A claim supported by three families is stronger than one supported by three metrics from the same family.
2. **Which families are represented?** The combination of causal + structural evidence is particularly powerful because their failure modes are complementary (runtime compensation cannot confound weight-space analysis; weight-space structure may be unused, but causal testing verifies actual use).
3. **Are there families that *should* support the claim but do not?** Absence of expected evidence from a family is informative — it may indicate a gap in the claim or a limitation of the proposed mechanism.

The convergent validity principle: **agreement across families with independent failure modes provides evidence that the finding reflects the target system rather than a shared artifact of the measurement approach.**

## Relationship to the rest of the framework

Evidence families sit at Layer B in the hierarchy:

- **Layer A (Metrics):** The concrete tools that produce measurements.
- **Layer B (Evidence Families):** The classification of those measurements by signal type. *You are here.*
- **Layer C (Criteria):** The standards that evidence must meet to support a validity claim.
- **Layer D (Validity Types):** The five independent dimensions along which a claim can succeed or fail.
- **Layer E (Verdicts):** The aggregate assessment, tagged with a description mode.

Each criterion (Layer C) specifies which evidence families are relevant to it. For example, the *Necessity* criterion under Internal Validity primarily draws on causal evidence, but structural evidence (showing a component's weight-space contribution is non-redundant) provides supporting convergent evidence. The criteria pages specify these mappings explicitly.
