---
title: "Neuroscience -- Metrics & Protocols"
description: "Reference for neuroscience-lens metrics and protocols: mediation analysis, cross-task overlap, attribution patching, SCM protocols, counterfactual DAS, Rubin CATE, Woodward interventionism, MDC/Glennan mechanisms, INUS conditions, actual causation, and linguistic probes."
---

# Neuroscience -- Metrics & Protocols

This page documents the metrics and protocols under the [Neuroscience lens](/framework/lenses/core/neuroscience). These metrics adapt causal inference methods from neuroscience -- mediation analysis, counterfactual interventions, path-specific effects, cross-task representation overlap, and attribution patching -- to evaluate whether proposed circuits are genuine causal mechanisms rather than mere correlates.

All metrics in this page follow one principle: **does the proposed circuit pass the same evidential standards that neuroscience uses to establish that a brain region is causally involved in a cognitive function?** Some operate at the single-head level (mediation, PSE), some at the circuit level (AtP*, cross-task overlap), and some orchestrate multiple metrics into protocols that map onto specific causal frameworks (Pearl's SCM, Rubin's potential outcomes, Woodward's interventionism).

---

## Causal Intervention Metrics

These metrics directly test whether circuit components are causally involved in the task via intervention experiments.

### C5v2 -- Mediation Analysis (v2)

**Source:** Pearl (2001), "Direct and Indirect Effects"; Vig et al. (2020).

**Criteria:** Causal / I1 Necessity

**What it establishes:** Whether each circuit head mediates the model's task performance. Decomposes the total effect (TE) into natural indirect effect (NIE, the effect flowing through the head) and natural direct effect (NDE, the effect bypassing the head). A head with high NIE relative to TE is a genuine mediator -- the model's output depends on information flowing through it.

**What it does not establish:** Whether the head is sufficient for the task. High mediation means the head is necessary (information passes through it), but other heads may also be necessary. It also does not reveal what computation the head performs.

**Method:**

1. Run the model on clean prompts, record the logit-diff (total effect baseline).
2. For each circuit head, apply counterfactual mediation:
   - **NIE**: intervene on the head's input (set to corrupted/mean), let the effect propagate naturally. NIE = clean_ld - intervened_ld.
   - **NDE**: freeze the head's output to its clean value, corrupt everything else. NDE = TE - NIE.
3. Compute `prop_mediated` = NIE / TE for each head.
4. Report `mean_prop_mediated` across all circuit heads.

**Key quantities:**

- `mean_prop_mediated` -- mean proportion of total effect mediated through circuit heads
- `nie_per_head` -- natural indirect effect per head
- `nde_per_head` -- natural direct effect per head

**Pass condition:** No explicit threshold (report-only). Higher `mean_prop_mediated` indicates stronger mediation.

**Usage:**

```bash
uv run python 05_mediation_v2.py --model gpt2 --device cpu
uv run python 05_mediation_v2.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| mean_prop_mediated > 0.7 | Strong mediation -- most of the task effect flows through circuit heads |
| mean_prop_mediated 0.3--0.7 | Moderate mediation -- circuit heads carry some but not all information |
| mean_prop_mediated < 0.3 | Weak mediation -- the circuit may be misspecified or the effect flows through other paths |
| One head dominates NIE | A single bottleneck head -- circuit is effectively a serial pipeline |

---

### C5 -- Mediation Analysis (Original)

**Source:** Pearl (2001), "Direct and Indirect Effects"; Baron & Kenny (1986).

**Criteria:** Causal / I1 Necessity

**What it establishes:** Same as C5v2 but uses the original NDE/NIE decomposition via Pearl's formula. Reports `nie_fraction` as the primary metric.

**What it does not establish:** Same limitations as C5v2. The original version is retained for backward compatibility; C5v2 is the recommended implementation.

**Key quantities:**

- `nie_fraction` -- fraction of total effect attributable to indirect (mediated) path

**Pass condition:** Report-only.

---

### C24 -- Path-Specific Effect (PSE)

**Source:** Pearl (2001); Vig et al. (2020).

**Criteria:** Causal / I1 Necessity

**What it establishes:** The causal contribution of each individual head in isolation. PSE patches ONE head at a time from mean to clean and measures the change in logit-diff. Unlike mediation analysis (which decomposes TE into NIE/NDE), PSE directly quantifies how much each head contributes to restoring the correct output.

**What it does not establish:** Whether heads interact -- PSE is a marginal (one-at-a-time) measure that misses synergistic or redundant interactions between heads. If two heads are redundant, PSE will overcount their combined contribution.

**Method:**

1. Start from a corrupted baseline (all circuit heads mean-ablated).
2. For each head, patch it from mean back to clean activation.
3. PSE = clean_ld - patched_ld for each head.
4. `pse_ratio` = sum of all PSEs / total effect.

**Key quantities:**

- `pse_ratio` -- ratio of summed path-specific effects to total effect
- `pse_per_head` -- individual head PSE values

**Pass condition:** Report-only. `pse_ratio` near 1.0 indicates the circuit accounts for the full effect additively; values > 1.0 suggest redundancy.

**Usage:**

```bash
uv run python 24_pse.py --model gpt2 --device cpu
uv run python 24_pse.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| pse_ratio near 1.0 | Circuit heads additively account for the total effect |
| pse_ratio > 1.0 | Redundancy -- some heads contribute overlapping information |
| pse_ratio < 1.0 | Synergy or missing components -- heads interact non-additively or circuit is incomplete |
| One head has PSE >> others | Single dominant head -- the "circuit" may be mostly one component |

---

### G2b -- AtP* Attribution Patching

**Source:** Kramar et al. (2024). arXiv:2403.00745.

**Criteria:** Causal

**What it establishes:** Which edges in the computational graph carry task-relevant information, with correction for cancellation artifacts. Standard attribution patching (AtP) can miss important edges when positive and negative contributions cancel. AtP* decomposes attributions to detect and correct for this cancellation, producing more reliable edge importance scores.

**What it does not establish:** Whether the identified edges form a sufficient circuit. AtP* ranks edges by importance but does not test whether the selected subset reproduces the model's behavior.

**Method:**

1. Run the model on clean and corrupted prompt pairs.
2. Compute standard AtP scores: gradient of output metric with respect to each edge activation, evaluated at the corrupted baseline.
3. Compute AtP* scores: decompose AtP into positive and negative contributions, flag edges where cancellation occurs (large positive and negative terms that nearly cancel).
4. For flagged edges, use the uncancelled (absolute) attribution.
5. Report fraction of circuit edges above the attribution threshold.

**Key quantities:**

- `frac_above_threshold` -- fraction of circuit edges with AtP* score above threshold
- `mean_atp_star` -- mean AtP* score across circuit edges
- `cancellation_rate` -- fraction of edges where AtP and AtP* disagree substantially

**Pass condition:** >= 80% of circuit edges above threshold.

**Usage:**

```bash
uv run python G2b_atp_star.py --model gpt2 --device cpu
uv run python G2b_atp_star.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| frac_above_threshold > 0.8 | Circuit edges are reliably important -- AtP* confirms the circuit definition |
| High cancellation_rate | Standard AtP would miss important edges -- AtP* correction is valuable |
| Low frac_above_threshold | Many circuit edges have weak attribution -- circuit may include non-essential components |

---

## Cross-Task Metrics

### E63 -- Cross-Task Overlap

**Source:** Kornblith et al. (2019), CKA; Raghu et al. (2021).

**Criteria:** Representational / E5 Cross-task

**What it establishes:** Whether the same representations are used across different tasks. Measures subspace overlap (via principal angles between task activation matrices) and CKA (Centered Kernel Alignment) at each layer. If two tasks share high CKA in early layers but diverge later, the model reuses low-level features but develops task-specific representations at higher layers.

**What it does not establish:** Whether shared representations serve the same function in both tasks. High overlap means similar activation patterns, not necessarily the same computation.

**Method:**

1. Run the model on prompts from two different tasks, collecting activations at each layer.
2. Compute CKA between the two task activation matrices at each layer.
3. Compute principal angles between the top-$k$ principal subspaces of each task's activations.
4. Report `cka_decline` (whether CKA decreases across layers) and `subspace_overlap` (mean cosine of principal angles).

**Key quantities:**

- `cka_decline` -- whether CKA decreases from early to late layers (indicates increasing task specialization)
- `subspace_overlap` -- mean subspace overlap across layers

**Pass condition:** Report-only. Both quantities are diagnostic.

**Usage:**

```bash
uv run python 63_cross_task_overlap.py --model gpt2 --device cpu
uv run python 63_cross_task_overlap.py --tasks ioi,greater_than --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| High CKA early, low CKA late | Model reuses low-level representations, develops task-specific features in later layers |
| High CKA throughout | Tasks use similar representations at all levels -- possibly a shared mechanism |
| Low CKA throughout | Tasks use distinct representations from the start -- independent circuits |
| High subspace_overlap | Activation subspaces align closely -- same directions encode both tasks |

---

## Protocols

Protocols orchestrate multiple metrics into a coherent evaluation that maps onto a specific causal framework. Each protocol defines required metrics, pass thresholds, and the theoretical tradition that justifies the evaluation structure.

### Protocol A01 -- SCM (Pearl)

**Source:** Pearl (2000), "Causality"; Wang et al. (2022); Chan et al. (2022); Conmy et al. (2023).

**Framework:** Structural Causal Models. Evaluates circuits through Pearl's causal hierarchy: Association (observational correlation), Intervention (do-calculus), and Counterfactual (what-if reasoning). Higher rungs provide stronger evidence.

**Metrics and thresholds:**

| Metric | Threshold | Rung |
|---|---|---|
| `logit_diff` | > 0.0 | Association |
| `activation_patching` | > 0.7 | Intervention |
| `causal_scrubbing` | > 0.5 | Counterfactual |
| `role_ablation` | Report-only | Intervention |

**What it establishes:** Whether the circuit satisfies the evidential requirements of Pearl's causal hierarchy. Passing at all three rungs means the circuit is not merely correlated with the task (Association), actually does something when intervened on (Intervention), and behaves correctly under counterfactual manipulations (Counterfactual).

**What it does not establish:** That the SCM is the unique correct causal model. Multiple SCMs may be compatible with the same intervention data.

---

### Protocol A02 -- Counterfactual DAS

**Source:** Geiger et al. (2021, 2023); Wu et al. (2023).

**Framework:** Distributed Alignment Search. Tests whether model representations can be aligned with causal variables in an interpretive theory via interchange interventions.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `das_iia` | > 0.8 |
| `iia_variants` | > 0.5 |
| `corrupt_restore` | > 0.7 |
| `multi_axis_iia` | > 0.5 |
| `counterfactual_consistency` | > 0.7 |
| `path_patching` | > 0.5 |
| `intermediate_state_prediction` | > 0.5 |

**What it establishes:** Whether there exist linear subspaces in the model's activations that correspond to interpretive variables, such that swapping those subspace components between inputs produces the counterfactually correct output. This is the strongest form of causal alignment -- it tests not just "is this head important" but "does this subspace encode this specific variable."

**What it does not establish:** Uniqueness of the alignment. Multiple subspaces may pass DAS, and the method assumes linearity of the representation.

---

### Protocol A03 -- Rubin CATE

**Source:** Rubin (1974); Holland (1986); Imbens & Rubin (2015).

**Framework:** Rubin Causal Model / Potential Outcomes. Evaluates circuits using the framework of randomized experiments and average treatment effects.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `cate` | > 0.5 |
| `intervention_specificity` | > 0.7 |

**What it establishes:** Whether the circuit has a measurable, specific causal effect. CATE (Conditional Average Treatment Effect) measures the average effect of circuit ablation, while intervention specificity ensures the effect is targeted (ablating the circuit affects the target task more than non-target tasks).

**What it does not establish:** The mechanism by which the circuit produces its effect. The Rubin framework is agnostic about mechanisms -- it quantifies effects without requiring a structural model.

---

### Protocol A04 -- Woodward Interventionism

**Source:** Woodward (2003), "Making Things Happen"; Craver & Bechtel (2007).

**Framework:** Woodward's interventionist account of causation. Tests three properties: stability (the causal relationship holds under different ablation methods), proportionality (the intervention is at the right level of abstraction), and invariance (the relationship holds across contexts).

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `sigma_ablation` | > 0.5 |
| `resample_complement` | > 0.7 |
| `misalignment` | < 0.3 |

**What it establishes:** Whether the circuit is a stable, proportional, invariant cause of the task behavior. Stability (low CV across ablation methods) means the finding is not an artifact of a particular ablation technique. Proportionality (low misalignment between noising and denoising) means the circuit is at the right level of granularity. Invariance (resample complement) means non-circuit components are genuinely uninvolved.

**What it does not establish:** Whether the circuit is the unique mechanism. Woodward's framework allows for multiple causes at different levels of description.

---

### Protocol A05 -- MDC/Glennan Mechanisms

**Source:** Machamer, Darden & Craver (2000); Glennan (2017); Bechtel & Abrahamsen (2005).

**Framework:** New Mechanistic Philosophy. Tests whether the circuit qualifies as a mechanism in the MDC sense: organized entities and activities producing the phenomenon.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `operation_specification` | > 0.5 |
| `held_out_prediction` | > 0.5 |
| `replacement_test` | > 0.7 |
| `procedure_specification` | > 0.5 |
| `composition_test` | > 0.5 |
| `logic_gates` | > 0.5 |

**What it establishes:** Whether the circuit's components have specifiable operations (each head does a characterizable thing), the operations predict behavior on held-out data, components are not trivially replaceable, and the components compose into a coherent procedure.

**What it does not establish:** That we have correctly identified what each component does -- only that the structural properties of a mechanism are present.

---

### Protocol A06 -- Mediation Analysis

**Source:** Pearl (2001); Baron & Kenny (1986); Vig et al. (2020).

**Framework:** Causal mediation. Orchestrates the mediation metrics (C5, C5v2, C24) into a unified evaluation of whether the circuit mediates the model's task performance.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `mediation` (C5) | > 0.5 |
| `mediation_v2` (C5v2) | > 0.5 |
| `pse` (C24) | > 0.7 |

**What it establishes:** Convergent evidence from three mediation measures that the circuit is on the causal pathway between input and output.

**What it does not establish:** Whether the circuit is the only causal pathway. Mediation analysis identifies mediators but cannot rule out unmeasured alternative pathways.

---

### Protocol A10 -- Regularity/INUS

**Source:** Mackie (1965), "Causes and Conditions."

**Framework:** INUS conditions (Insufficient but Necessary part of an Unnecessary but Sufficient condition). Tests whether circuit heads satisfy INUS conditions for the task behavior.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `minimality_class` | > 0.5 |

**What it establishes:** Whether each head is an INUS condition for the task -- necessary within some sufficient subset, even if alternative sufficient subsets exist. This is weaker than strict necessity but stronger than mere correlation.

**What it does not establish:** That the circuit is the only sufficient set. By definition, INUS allows for multiple sufficient sets; the metric classifies heads within the identified circuit.

---

### Protocol A11 -- Actual Causation

**Source:** Halpern & Pearl (2005); Shapley (1953).

**Framework:** Halpern-Pearl actual causation (AC1--AC3 conditions) plus Shapley value interaction analysis.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `eap` | > 0.5 |
| `atp_star` | > 0.5 |
| `pairwise_synergy` | > 0.0 |
| `shapley_interactions` | > 0.0 |

**What it establishes:** Whether circuit heads are actual causes (not just counterfactual dependencies) of the task behavior. AC conditions handle preemption and overdetermination -- cases where standard counterfactual tests give misleading results. Shapley interactions reveal whether heads work synergistically or redundantly.

**What it does not establish:** A complete causal model. Actual causation identifies specific causes in specific contexts; it does not provide a general law-like relationship.

---

### Protocol EX03 -- Linguistic Probes

**Source:** Linzen et al. (2016); Marvin & Linzen (2018); Futrell et al. (2019).

**Framework:** Psycholinguistic evaluation. Tests whether the circuit handles specific linguistic phenomena that the original task requires.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `priming` | > 0.0 |
| `garden_path` | > 0.0 |
| `binding_theory` | > 0.5 |
| `animacy` | > 0.5 |

**What it establishes:** Whether the circuit's behavior is consistent with known psycholinguistic phenomena. If the IOI circuit correctly handles binding theory constraints and animacy distinctions, this provides content validity evidence -- the circuit engages with the linguistic structure the task requires.

**What it does not establish:** That the circuit implements these linguistic computations in the way humans do. Behavioral consistency with psycholinguistic patterns does not imply mechanistic similarity.

---

## Summary Table

| Metric ID | Name | Criteria | Evidence Family | Pass Condition |
|---|---|---|---|---|
| C5 | Mediation Analysis | I1 Necessity | Causal | Report-only |
| C5v2 | Mediation Analysis v2 | I1 Necessity | Causal | Report-only |
| C24 | Path-Specific Effect | I1 Necessity | Causal | Report-only |
| E63.cka | Cross-Task CKA Decline | E5 Cross-task | Representational | Report-only |
| E63.sub | Cross-Task Subspace Overlap | E5 Cross-task | Representational | Report-only |
| G2b | AtP* Attribution Patching | Causal | Causal | >= 80% edges above threshold |
| p_a01 | SCM Pearl | Pearl hierarchy | Protocol | See metric thresholds |
| p_a02 | Counterfactual DAS | DAS alignment | Protocol | See metric thresholds |
| p_a03 | Rubin CATE | Potential outcomes | Protocol | cate > 0.5, specificity > 0.7 |
| p_a04 | Woodward Interventionism | Stability/proportionality | Protocol | See metric thresholds |
| p_a05 | MDC/Glennan Mechanisms | Mechanistic explanation | Protocol | See metric thresholds |
| p_a06 | Mediation Analysis | Mediation | Protocol | See metric thresholds |
| p_a10 | Regularity/INUS | INUS conditions | Protocol | minimality_class > 0.5 |
| p_a11 | Actual Causation | HP actual causation | Protocol | See metric thresholds |
| p_ex03 | Linguistic Probes | Psycholinguistic | Protocol | See metric thresholds |

---

## Connection to Neuroscience Lens

The neuroscience lens is documented at the [Neuroscience lens page](/framework/lenses/core/neuroscience). The core insight is that mechanistic interpretability faces the same evidential challenges as cognitive neuroscience: establishing that a component is causally involved in a function requires multiple converging lines of evidence, not a single ablation experiment.

The metrics on this page operationalize that insight:

- **Mediation metrics** (C5, C5v2, C24) decompose total effects into direct and indirect paths, following Pearl's mediation framework -- the same approach used in neuroimaging to establish that a brain region mediates between stimulus and response.
- **Attribution patching** (G2b) adapts gradient-based attribution with cancellation correction, analogous to effective connectivity analysis in neuroscience.
- **Cross-task overlap** (E63) tests whether circuits share representations across tasks, paralleling cross-decoding analyses in cognitive neuroscience.
- **Protocols** (A01--A11, EX03) organize these individual tests into coherent evaluation strategies drawn from specific philosophical traditions of causal inference.
