---
title: "MI Behavioral Metrics"
description: "Behavioral metrics from the mechanistic interpretability lens: faithfulness, generalization, and functional characterization of circuit behavior."
---

# MI Behavioral Metrics

These 15 protocols implement the behavioral criteria defined in the [mechanistic interpretability lens overview](/framework/lenses/core/mechanistic-interpretability). Each protocol tests a different aspect of circuit behavior -- whether the circuit faithfully reproduces the model's output, whether that faithfulness generalizes across conditions, and whether the circuit's functional role is well-characterized. The behavioral evidence family (D) complements causal and structural evidence by asking what the circuit *does*, not merely what it is connected to.

The protocols below are organized into four groups: **faithfulness and sufficiency** (D01, C20--C22), **loss-based and distributional fidelity** (D04--D06), **generalization and compression** (D07--D09), and **functional characterization** (A5, CM1, CM2, E6b, RI3b).

---

## Group 1: Faithfulness and Sufficiency

### D01 -- Derived Metrics (`14_derived_metrics.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I2 Sufficiency

#### Question

What summary statistics can be computed from previously collected evaluation data without additional forward passes? This protocol reads results from earlier evaluations (activation patching, causal scrubbing, measurement invariance, etc.) and computes approximately 20 derived quantities: sparsity ratios, node overlap with published circuits, spectral norm ratios, $d'$ (signal detection discriminability), hit/false-alarm rates from weight classifiers, composition $p$-values, attribution AUROC, faithfulness, completeness, and minimality statistics.

The discriminability index $d'$ is computed as:

$$
d' = \frac{\mu_{\text{signal}} - \mu_{\text{noise}}}{\sigma_{\text{pooled}}}
\quad\text{where}\quad
\sigma_{\text{pooled}} = \sqrt{(\sigma_{\text{signal}}^2 + \sigma_{\text{noise}}^2) / 2}
$$

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `C14.sparsity` | $n_{\text{circuit}} / n_{\text{total}}$ (fraction of heads in circuit) | Informational |
| `C14.d_prime` | Discriminability between circuit and random patching effects | $\geq 1.0$ |
| `C14.node_overlap_jaccard` | Size-ratio proxy for overlap with published circuits | $\geq 0.5$ |
| `C14.faithfulness` | Circuit-only logit diff / full-model logit diff | $\geq 0.8$ |
| `C14.completeness` | Ablating circuit destroys performance | $\geq 0.8$ |

#### Interpretation

- **What it establishes:** A unified summary view that identifies inconsistencies across evaluation methods. High $d'$ confirms that circuit vs. non-circuit is a meaningful distinction by signal-detection criteria. High faithfulness and completeness together demonstrate both sufficiency and necessity.
- **What it does not establish:** Any new causal claim -- all values are derived from earlier experiments. The protocol cannot detect errors in the upstream data it aggregates.

---

### C20 -- Corrupt-Restore Patching (`20_corrupt_restore.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I2 Sufficiency

#### Question

Can the circuit alone restore model performance from a fully corrupted baseline? Standard activation patching tests necessity (ablate circuit, measure degradation). This protocol tests the converse: start with *all* heads mean-ablated, then restore only the circuit heads from the clean cache, and measure how much of the clean-to-corrupted logit-difference gap is recovered.

$$
\text{restoration\_rate} = \frac{LD_{\text{restored}} - LD_{\text{corrupted}}}{LD_{\text{clean}} - LD_{\text{corrupted}}}
$$

Per-head restoration contributions are also computed by restoring each circuit head individually.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `C20.corrupt_restore` | Fraction of logit-diff gap recovered by restoring circuit heads | $\geq 0.5$ |
| (baseline) | Same metric for random head sets of equal size | Comparison |

#### Interpretation

- **What it establishes:** Sufficiency of the circuit -- the circuit heads alone can reconstruct the model's task behavior from a destroyed baseline. A restoration rate substantially above the random baseline confirms that the circuit is not merely a size-matched arbitrary subset.
- **What it does not establish:** Necessity (a different subset might also suffice) or that the circuit's internal mechanism matches the model's. Restoration rate depends on the choice of ablation method (mean ablation).

---

### C21 -- Output Metric Variants (`21_output_variants.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I2 Sufficiency

#### Question

Is the circuit's faithfulness robust to the choice of output metric? The standard metric is logit difference, but this is one of many possible summaries of the model's output distribution. This protocol computes faithfulness under five alternative metrics -- logit difference, log probability, raw probability, top-1 accuracy, and KL divergence -- and reports their coefficient of variation:

$$
\sigma_{\text{output}} = \frac{\text{std}(\{f_1, \ldots, f_5\})}{\text{mean}(\{f_1, \ldots, f_5\})}
$$

where $f_k$ is the faithfulness score under metric $k$.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `C21.output_variants` | $\sigma_{\text{output}}$ -- coefficient of variation across 5 output metrics | $\leq 0.3$ |

#### Interpretation

- **What it establishes:** Measurement robustness -- a low $\sigma_{\text{output}}$ confirms that the circuit claim does not depend on an idiosyncratic property of logit difference. The circuit faithfully reproduces the full model's output distribution, not just one scalar summary of it.
- **What it does not establish:** That any individual faithfulness score is high. A circuit can have low $\sigma_{\text{output}}$ while being uniformly unfaithful across all metrics.

---

### C22 -- Mean-Centered Logit Diff (`22_mean_centered_logit.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I2 Sufficiency

#### Question

Is the logit-difference metric invariant to mean-centering the logit vector? This is a sanity check: because mean-centering subtracts a constant from both the correct and incorrect logits, the difference should be unchanged. The protocol computes both the standard logit diff $LD = \ell_{\text{correct}} - \ell_{\text{incorrect}}$ and the centered variant $LD_c = (\ell_{\text{correct}} - \bar{\ell}) - (\ell_{\text{incorrect}} - \bar{\ell})$, then reports their ratio.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `C22.mean_centered_logit` | Ratio of centered to standard logit diff (should be $\approx 1.0$) | $\in [0.99, 1.01]$ |

#### Interpretation

- **What it establishes:** That the logit-diff metric is algebraically well-behaved -- centering cancels in the difference. A ratio deviating from 1.0 would indicate a numerical issue in the evaluation pipeline.
- **What it does not establish:** Any property of the circuit itself. This is a measurement-infrastructure validation, not a circuit evaluation.

---

## Group 2: Loss-Based and Distributional Fidelity

### D04 -- Cross-Entropy Delta (`43_ce_delta.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I1 Necessity

#### Question

How much of the model's loss sensitivity is attributable to the circuit? Logit difference captures the margin between two tokens; cross-entropy loss captures the full output distribution. This protocol computes the CE increase when ablating the circuit ($\Delta_{\text{circuit}}$) vs. ablating the complement ($\Delta_{\text{complement}}$):

$$
\text{CE\_ratio} = \frac{\Delta_{\text{circuit}}}{\Delta_{\text{complement}}}
= \frac{\text{CE}(\text{circuit ablated}) - \text{CE}(\text{full})}{\text{CE}(\text{complement ablated}) - \text{CE}(\text{full})}
$$

A ratio $\gg 1$ means ablating the circuit hurts more than ablating everything else -- the circuit is the dominant contributor to model performance on this task.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D04.ce_delta` | Ratio of CE increase from circuit ablation to complement ablation | $\geq 1.0$ |

#### Interpretation

- **What it establishes:** That the circuit accounts for more of the model's loss sensitivity than the complement, as measured by full-distribution cross-entropy rather than a two-token summary. This is a stronger necessity claim than logit-diff-based ablation.
- **What it does not establish:** Sufficiency. The complement's CE increase may be nonzero, indicating that non-circuit components also contribute.

---

### D05 -- Per-Token NLL (`44_per_token_nll.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I1 Necessity

#### Question

At which token positions does the circuit contribute most to the model's predictions? Rather than measuring only at the final prediction position, this protocol computes the NLL increase at every position when the circuit is ablated. It reports the mean NLL increase, the maximum positional increase, and the fraction of total NLL increase concentrated at the last position.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D05.per_token_nll` | Mean NLL increase at the last (prediction) position | $\geq 0.5$ nats |
| `last_pos_fraction` | Fraction of total NLL increase at the prediction position | Informational |

#### Interpretation

- **What it establishes:** Positional specificity -- whether the circuit's contribution is concentrated at the task-relevant position or spread across the entire sequence. A high `last_pos_fraction` confirms that the circuit is specialized for the prediction position, not a general-purpose component.
- **What it does not establish:** Whether the circuit's contribution at non-prediction positions is beneficial or harmful. Distributed NLL increases may indicate that the circuit performs useful intermediate computations at earlier positions.

---

### D06 -- Calibration (`45_calibration.py`)

**Lens:** MI | **Validity type:** Measurement | **Criterion:** M5 Calibration

#### Question

Does the circuit preserve the model's probability calibration? A circuit can achieve high accuracy while distorting the probability distribution -- predicting the right answer with the wrong confidence. This protocol bins the model's predicted probabilities into deciles, computes the actual accuracy in each bin, and reports the Expected Calibration Error:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|
$$

where $\text{acc}(b)$ and $\text{conf}(b)$ are the accuracy and mean confidence in bin $b$.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D06.calibration_ece` | ECE of circuit-only model | $\leq 0.15$ |
| `ece_ratio` | $\text{ECE}_{\text{circuit}} / \text{ECE}_{\text{full}}$ | $\leq 2.0$ |

#### Interpretation

- **What it establishes:** Distributional faithfulness -- the circuit not only gets the right answer but assigns well-calibrated probabilities. A circuit with low ECE ratio preserves the model's uncertainty structure, not just its point predictions.
- **What it does not establish:** That the circuit's calibration matches the model's across the full vocabulary. ECE is computed only for the correct-vs-incorrect token pair, not the complete softmax distribution.

---

## Group 3: Generalization and Compression

### D07 -- Generalization Gap (`46_generalization_gap.py`)

**Lens:** MI | **Validity type:** External | **Criterion:** E1 Replication

#### Question

Does the circuit's faithfulness generalize beyond the standard prompt templates? This protocol compares faithfulness on in-distribution prompts (standard templates) against out-of-distribution variants created by prepending or appending padding text ("Well, you know, ...", "That was what happened."). The gap between in-distribution and OOD faithfulness reveals template dependence:

$$
\text{gap} = f_{\text{ID}} - f_{\text{OOD}}
$$

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D07.generalization_gap` | Absolute faithfulness drop from ID to OOD prompts | $\leq 0.15$ |

#### Interpretation

- **What it establishes:** That the circuit captures genuine computational structure rather than template-specific correlations. A small gap means the circuit's behavior is robust to superficial changes in prompt format.
- **What it does not establish:** Robustness to semantic distribution shifts. The OOD perturbations are syntactic (padding text), not semantic (different task instances or domains).

---

### D08 -- MDL Compression (`47_mdl_compression.py`)

**Lens:** MI | **Validity type:** Construct | **Criterion:** C4 Minimality

#### Question

How efficiently does the circuit compress the model's task behavior? The Minimum Description Length framework evaluates a circuit by the ratio of its faithfulness to its size. This protocol computes:

- **Compression ratio:** $n_{\text{circuit}} / n_{\text{total}}$
- **Efficiency:** faithfulness / compression ratio
- **KL coding cost:** $D_{\text{KL}}(P_{\text{full}} \| P_{\text{circuit}})$ at the last position

A good circuit achieves high faithfulness with few components (high efficiency) and low KL divergence from the full model.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D08.mdl_compression` | Efficiency = faithfulness / compression ratio | $\geq 3.0$ |
| `kl_coding_cost` | $D_{\text{KL}}(\text{full} \| \text{circuit})$ in nats | $\leq 1.0$ |

#### Interpretation

- **What it establishes:** That the circuit is a parsimonious explanation -- it achieves high faithfulness relative to its size. High efficiency means the circuit is not merely "large enough to contain the computation" but is specifically organized around it.
- **What it does not establish:** That the circuit is *minimal*. A smaller subset might achieve comparable efficiency. MDL compression is a quality measure, not a minimality proof.

---

### D09 -- Subnetwork Probe (`48_subnetwork_probe.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** I2 Sufficiency

#### Question

Does the circuit concentrate linearly decodable task information relative to random subnetworks? This protocol trains a logistic regression probe on the circuit's residual-stream output (top-100 logit magnitudes at the last position) to predict whether the model answers correctly. The same probe is trained on random subnetworks of equal size, and the circuit's probe accuracy is compared to the random baseline.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `D09.subnetwork_probe` | Probe accuracy on circuit residual stream | $\geq 0.7$ |
| `advantage` | Circuit accuracy minus random baseline accuracy | $\geq 0.1$ |
| `z_score` | Advantage / std of random baseline accuracies | $\geq 2.0$ |

#### Interpretation

- **What it establishes:** That the circuit's output representation is more informative about task correctness than a size-matched random subset. A high $z$-score confirms that the information concentration is statistically significant, not a consequence of subnetwork size alone.
- **What it does not establish:** That the circuit's representation is *causally* relevant. Linear decodability is an associative (Rung 1) measure -- it identifies correlation between circuit output and task correctness, not a causal mechanism.

---

## Group 4: Functional Characterization

### CM1 -- Normative Account (`80_normative_account.py`)

**Lens:** MI | **Validity type:** External | **Criterion:** CM1 Normative Account (proposed)

#### Question

Does the circuit solve a genuine, separable subproblem of language modeling? This protocol runs the model on a diverse corpus (not task-specific prompts), measures circuit-head logit attribution magnitude via direct logit attribution ($z \cdot W_O \cdot W_U$), splits prompts into high- and low-activation clusters (median split), and tests whether the high-activation prompts share specific linguistic properties (epistemic verbs, modal verbs, hedging, quantifiers, negation, pronouns, conjunctions).

The key quantity is the **separation ratio**: the prevalence of a linguistic feature in high-activation prompts divided by its prevalence in low-activation prompts.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `CM1.normative_account` | Maximum separation ratio across linguistic features | $\geq 2.0$ |
| `activation_rate` | Fraction of prompts with above-P75 circuit activation | $\in [0.05, 0.50]$ |

Pass requires both: separation ratio $> 2.0$ on at least one feature AND activation rate between 0.05 and 0.50.

#### Interpretation

- **What it establishes:** That the circuit activates selectively on prompts sharing a coherent linguistic property, suggesting it implements a genuine subcomputation rather than an arbitrary collection of components.
- **What it does not establish:** That the identified linguistic feature is the *correct* normative account of the circuit's function. The feature analysis uses a fixed word-list heuristic, which may miss the true functional characterization or produce spurious matches.

---

### CM2 -- Error Boundary Analysis (`81_error_boundary_analysis.py`)

**Lens:** MI | **Validity type:** External | **Criterion:** CM2 Error Analysis at Boundaries (proposed)

#### Question

Do the circuit's failures align with the model's failures? This protocol generates prompts at three difficulty levels (easy, medium, hard -- using progressively more complex syntax, longer sentences, and garden-path constructions) and measures both model accuracy and circuit faithfulness at each level. The **boundary alignment** score is the fraction of prompts where the circuit's faithfulness status (faithful/unfaithful at threshold 0.3) agrees with the model's accuracy status (correct/incorrect):

$$
\text{alignment} = \frac{|\{i : \text{correct}_i = \text{faithful}_i\}|}{N}
$$

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `CM2.error_boundary_analysis` | Boundary alignment (fraction of aligned cases) | $\geq 0.60$ |

#### Interpretation

- **What it establishes:** That the circuit fails where the model fails and succeeds where the model succeeds. High alignment means the circuit's failure modes correspond to genuine problem boundaries (ambiguity, complexity) rather than arbitrary gaps in the circuit description.
- **What it does not establish:** Why the circuit fails on hard cases. The protocol detects alignment between circuit and model failure patterns but does not explain the mechanism of failure.

---

### A5 -- Epistemic Gradient (`A5_epistemic_gradient.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** Minimal-pair gradient test

#### Question

Does the circuit respond monotonically to graded manipulation of a target construct? This protocol creates minimal pairs at four epistemic-framing levels -- neutral ("The door is open"), weak ("Maybe the door is open"), medium ("I think the door is open"), strong ("I firmly believe the door is open") -- and measures mean absolute `hook_z` activation at circuit heads on the last token. The key metric is **monotonicity**: the fraction of adjacent level pairs where activation increases with framing strength.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `A5.epistemic_gradient` | Monotonicity (fraction of ordered adjacent pairs) | $\geq 0.75$ |
| `gradient_slope` | Linear regression slope of activation across levels | Informational |

#### Interpretation

- **What it establishes:** That the circuit's internal activations track the strength of a linguistic construct in a graded, monotonic fashion. This is evidence that the circuit encodes a continuous quantity, not merely a binary feature presence/absence.
- **What it does not establish:** That the circuit *uses* the epistemic framing for its computation. The activation gradient may be a consequence of input length or word frequency differences between levels rather than genuine sensitivity to epistemic strength.

---

### E6b -- Cross-Task Generalization (`E6b_cross_task_generalization.py`)

**Lens:** MI | **Validity type:** External | **Criterion:** Task-transfer test

#### Question

Does a circuit identified on one task also activate on related tasks? For a source task's circuit heads, this protocol computes mean absolute logit-diff attribution (the effect of ablating the circuit) across multiple tasks. The **transfer score** is the ratio of mean held-out-task attribution to source-task attribution. **Selectivity** is the inverse: source attribution divided by mean across all tasks.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `E6b.cross_task_generalization` | Transfer score (held-out attribution / source attribution) | Informational |
| `selectivity` | Source attribution / mean all-task attribution | Informational |

Pass condition: circuit attribution exceeds $2\times$ baseline on at least 2 held-out tasks.

#### Interpretation

- **What it establishes:** Whether the circuit is task-specific (high selectivity, low transfer) or shared infrastructure (low selectivity, high transfer). Both outcomes are informative -- a task-specific circuit confirms that the model has dedicated machinery; a shared circuit reveals common computational substrate across tasks.
- **What it does not establish:** The mechanism of transfer. High cross-task attribution could reflect genuine shared computation or a confound (e.g., the circuit heads are in early layers that process all inputs similarly).

---

### RI3b -- Boundary Sweep (`RI3b_boundary_sweep.py`)

**Lens:** MI | **Validity type:** Internal | **Criterion:** Construct boundary identification

#### Question

Where does the circuit's activation boundary lie across different prompt categories? This protocol defines six prompt categories (epistemic, evidential, temporal, spatial, causal, neutral), runs each through the model, and measures mean absolute attention score at circuit heads on the last token. The **activation ratio** is the maximum category activation divided by the minimum.

#### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `RI3b.boundary_sweep` | Activation ratio (max category / min category) | $\geq 2.0$ |
| `category_ranking` | Ordering of categories by circuit activation strength | Informational |

#### Interpretation

- **What it establishes:** That the circuit responds differentially to semantically distinct prompt categories. A high activation ratio identifies which categories most strongly engage the circuit, providing a behavioral fingerprint of its functional role.
- **What it does not establish:** That the category distinction is causal. The categories differ in multiple features (word choice, sentence structure, semantic content), so the activation differences may be driven by confounds rather than the intended category variable.

---

## Cross-protocol evidence patterns

| Pattern | Protocols | What it establishes |
|---|---|---|
| High restoration + low $\sigma_{\text{output}}$ + low generalization gap | C20 + C21 + D07 | Robust sufficiency: the circuit faithfully reproduces the model across metrics and distributions |
| High CE ratio + high last-position NLL fraction | D04 + D05 | Concentrated necessity: the circuit dominates loss at the task-critical position |
| High efficiency + high boundary alignment | D08 + CM2 | Parsimonious and well-bounded: the circuit is small, faithful, and fails where the model fails |
| High $d'$ + high subnetwork probe advantage | D01 + D09 | Clear separability: circuit and non-circuit components are statistically distinguishable by both causal effect and linear decodability |
| High transfer + high monotonicity | E6b + A5 | Shared graded computation: the circuit is engaged across tasks and responds continuously to construct strength |
| Low restoration + high CE ratio | C20 + D04 | Necessary but insufficient: the circuit is critical (ablation hurts) but cannot alone reconstruct the computation |

## Relationship to other lenses

The behavioral metrics documented here operationalize criteria from the [mechanistic interpretability lens](/framework/lenses/core/mechanistic-interpretability). They provide evidence at Pearl's Rung 1 (association, via probing and feature analysis) and Rung 2 (intervention, via ablation and patching), complementing the causal metrics (A01 activation patching, causal scrubbing) which operate at Rungs 2--3.

- **Causal metrics** (A01) establish that specific components are causally load-bearing. Behavioral metrics establish that the circuit's *output* matches the model's -- a distinct claim. A circuit can be causally necessary without being behaviorally faithful (if the remaining components compensate), and behaviorally faithful without being causally isolated (if redundant pathways exist).
- **Structural metrics** characterize weight-space properties (spectral structure, composition scores). Behavioral metrics test the *functional* consequences of those structural properties. High spectral norm ratio (structural) predicts high CE ratio (behavioral), but the correspondence must be verified empirically.
- **Supporting lenses** (dynamical systems, control theory, information theory) provide orthogonal evidence about the circuit's internal dynamics. Behavioral metrics test the circuit's input-output function; supporting lenses test its internal organization. Together they establish that the circuit not only produces the right output but does so via an identifiable, structured computation.
