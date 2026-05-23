---
title: "Pharmacology -- Metrics & Protocols"
description: "Reference for pharmacology-lens metrics and protocols: effect size reporting, dose-response curves, faithfulness and recovery protocols, generalization and transfer, and probing and representation analysis."
---

# Pharmacology -- Metrics & Protocols

This page documents the metrics and protocols under the [Pharmacology lens](/framework/lenses/core/pharmacology). These metrics adapt the pharmacological toolkit -- dose-response analysis, effect size quantification, generalization testing, and probing -- to evaluate whether circuit interventions produce reliable, well-characterized effects.

The pharmacology lens asks: **does the intervention behave like a specific drug acting on a known target?** A good circuit ablation should produce a dose-dependent effect (more ablation = more degradation), the effect should be large and reproducible (strong effect size), the circuit should generalize beyond the training distribution, and probing should confirm that the circuit encodes the representations its mechanism requires.

---

## Effect and Dose Metrics

These metrics quantify the magnitude and dose-dependence of circuit interventions.

### M90 -- Effect Size Reporting

**Source:** Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences."

**Criteria:** Measurement

**What it establishes:** The magnitude of the difference between circuit and non-circuit head recovery scores, expressed in standardized units. Raw logit-diff numbers are hard to interpret without context; effect sizes (Cohen's d, Glass's delta, Hedges' g) normalize the difference by variability, making results comparable across tasks and models.

**What it does not establish:** Whether the effect is causally due to the circuit. A large effect size means the circuit and non-circuit populations differ substantially, but this is a descriptive statistic, not a causal test.

**Method:**

1. Compute recovery scores (logit-diff after ablation / clean logit-diff) for circuit heads and non-circuit heads separately.
2. Compute three effect size measures:
   - **Cohen's d**: $(M_{\text{circuit}} - M_{\text{non-circuit}}) / s_{\text{pooled}}$
   - **Glass's delta**: $(M_{\text{circuit}} - M_{\text{non-circuit}}) / s_{\text{non-circuit}}$ (uses only non-circuit SD as reference)
   - **Hedges' g**: bias-corrected Cohen's d for small samples

**Key quantities:**

- `cohens_d` -- standardized mean difference (pooled SD)
- `glass_delta` -- standardized mean difference (control SD)
- `hedges_g` -- bias-corrected effect size

**Pass condition:** Cohen's d > 0.8 (large effect by conventional standards).

**Usage:**

```bash
uv run python 90_effect_size.py --model gpt2 --device cpu
uv run python 90_effect_size.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| d > 1.2 | Very large effect -- circuit heads are clearly distinct from non-circuit heads |
| d 0.8--1.2 | Large effect -- conventional threshold for a meaningful difference |
| d 0.5--0.8 | Medium effect -- circuit heads differ moderately from non-circuit |
| d < 0.5 | Small effect -- circuit/non-circuit distinction is weak |
| Glass's delta >> Cohen's d | Non-circuit SD is small (homogeneous baseline) -- effect is clear relative to baseline variability |

---

### B95 -- Ablation Dose-Response

**Source:** Inspired by pharmacological dose-response curve methodology.

**Criteria:** Behavioral

**What it establishes:** Whether circuit ablation produces a monotonic, dose-dependent effect. A genuine causal mechanism should show graded degradation as more of it is removed -- ablating 25% of the circuit should produce less degradation than ablating 75%. Non-monotonic responses suggest the circuit definition includes compensatory or redundant components.

**What it does not establish:** The mechanism of the dose-response relationship. Monotonicity is necessary but not sufficient for a clean causal mechanism -- the circuit could be monotonic simply because it contains a single dominant head.

**Method:**

1. Sweep ablation fractions: [0.0, 0.25, 0.5, 0.75, 1.0] of circuit heads (ordered by individual effect size).
2. At each fraction, mean-ablate the selected heads and measure faithfulness (logit-diff recovery).
3. Compute monotonicity: fraction of adjacent (fraction, faithfulness) pairs where faithfulness decreases as ablation increases.
4. Compute selectivity: slope of circuit dose-response / slope of random-head dose-response. High selectivity means ablating circuit heads is more damaging than ablating random heads at the same dose.

**Key quantities:**

- `monotonicity` -- fraction of dose steps showing monotonic decrease (1.0 = perfectly monotonic)
- `selectivity` -- ratio of circuit slope to random-head slope
- `faithfulness_at_fraction` -- faithfulness values at each ablation dose

**Pass condition:** monotonicity >= 0.8 AND selectivity > 2.0.

**Usage:**

```bash
uv run python 95_dose_response.py --model gpt2 --device cpu
uv run python 95_dose_response.py --tasks ioi --n-prompts 50
```

**Reading the scores:**

| Pattern | What it means |
|---|---|
| Monotonicity = 1.0, selectivity > 3.0 | Textbook dose-response -- circuit ablation cleanly degrades performance |
| Monotonicity >= 0.8, selectivity > 2.0 | Good dose-response -- passes threshold |
| Low monotonicity | Non-monotonic response -- suggests compensatory mechanisms or redundancy within the circuit |
| Low selectivity | Circuit heads are not more important than random heads -- circuit definition may be incorrect |
| Flat response until high dose | Redundancy -- many heads must be removed before the effect appears |

---

## Protocols

### Protocol D01 -- Faithfulness and Recovery

**Source:** Wang et al. (2022); Conmy et al. (2023); Chan et al. (2022); Cohen & Saphra (2024).

**Framework:** Does the circuit faithfully reproduce the model's behavior? Evaluates both the magnitude of circuit effects (via effect size) and the quality of circuit reconstruction (via multiple output comparison methods).

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `effect_size` | > 0.8 |
| `dose_response` | monotonicity >= 0.8, selectivity > 2.0 |
| `corrupt_restore_behavioral` | > 0.5 |
| `output_variants` | > 0.7 |
| `output_variants_kl` | < 1.0 |
| `output_variants_topk` | > 0.7 |
| `mean_centered_logit` | Report-only |

**What it establishes:** Whether the circuit produces the same outputs as the full model when isolated, and whether removing the circuit disrupts model behavior. This is the most basic requirement for a valid circuit: it should be functionally equivalent to the model on the task.

**What it does not establish:** Why the circuit works. Faithfulness is a necessary but not sufficient condition -- a circuit that passes all faithfulness tests could still be a "shortcut" that correlates with the correct answer without implementing the intended algorithm.

---

### Protocol D02 -- Generalization and Transfer

**Source:** Geiger et al. (2021); Olsson et al. (2022); Rissanen (1978); Nanda et al. (2023).

**Framework:** Does the circuit generalize beyond the prompts used to define it? Overfitting to the definition distribution is a fundamental threat to circuit validity.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `ce_delta` | > 0.5 |
| `per_token_nll` | Report-only |
| `calibration` | < 0.1 |
| `generalization_gap` | < 0.2 |
| `mdl_compression` | > 0.5 |
| `normative_account` | Report-only |
| `error_boundary` | Report-only |

**What it establishes:** Whether the circuit's behavior holds on held-out data, different prompt templates, and different difficulty levels. A small generalization gap (low difference between train and test faithfulness) means the circuit captures a genuine regularity rather than memorizing specific inputs. MDL compression tests whether the circuit is a parsimonious description of the model's behavior.

**What it does not establish:** That the circuit generalizes to all possible inputs. Generalization is always relative to a test distribution; novel distributions may break the circuit.

---

### Protocol D03 -- Probing and Representation

**Source:** Belinkov (2022); Hewitt & Liang (2019); Geiger et al. (2021); Nanda et al. (2023).

**Framework:** Do the circuit's internal representations encode the information that its proposed mechanism requires? If a circuit is hypothesized to perform indirect object identification, its internal activations should encode the identity of the indirect object.

**Metrics and thresholds:**

| Metric | Threshold |
|---|---|
| `subnetwork_probe` | > 0.7 |
| `boundary_sweep` | Report-only |
| `epistemic_gradient` | Report-only |
| `cross_task_generalization` | > 0.5 |

**What it establishes:** Whether the circuit's activations contain the information needed for the proposed computation. Subnetwork probing trains a linear classifier on circuit activations and tests whether task-relevant features are linearly decodable. Boundary sweep tests how performance degrades as the probe is restricted to smaller subsets.

**What it does not establish:** Whether the circuit actually uses the probed information. A representation can be present without being functionally relevant (the "encoding vs. use" distinction from the probing literature).

---

## Summary Table

| Metric ID | Name | Criteria | Evidence Family | Pass Condition |
|---|---|---|---|---|
| M90 | Effect Size Reporting | Measurement | Behavioral | Cohen's d > 0.8 |
| B95 | Ablation Dose-Response | Behavioral | Behavioral | monotonicity >= 0.8, selectivity > 2.0 |
| p_d01 | Faithfulness & Recovery | Faithfulness | Protocol | See metric thresholds |
| p_d02 | Generalization & Transfer | Generalization | Protocol | generalization_gap < 0.2 |
| p_d03 | Probing & Representation | Representational | Protocol | subnetwork_probe > 0.7 |

---

## Connection to Pharmacology Lens

The pharmacology lens is documented at the [Pharmacology lens page](/framework/lenses/core/pharmacology). The core analogy is that circuit ablation is an intervention analogous to pharmacological manipulation: just as a drug's effect on a biological target is characterized by dose-response curves, effect sizes, generalization across patient populations, and mechanistic probes, circuit interventions should be characterized by the same rigorous quantitative standards.

The metrics on this page operationalize that analogy:

- **Effect size** (M90) quantifies intervention magnitude in standardized units, following Cohen's conventions -- the same standards used in clinical trial reporting.
- **Dose-response** (B95) tests whether the intervention produces graded, monotonic effects, paralleling pharmacological dose-response curve analysis.
- **Faithfulness** (D01) tests whether the circuit reproduces the model's behavior, analogous to confirming that a drug target is functionally relevant.
- **Generalization** (D02) tests whether findings transfer beyond the training distribution, paralleling external validity in clinical research.
- **Probing** (D03) confirms that the circuit encodes the necessary representations, analogous to confirming receptor binding in pharmacology.
