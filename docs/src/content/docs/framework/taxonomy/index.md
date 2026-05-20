---
title: "Framework Overview"
description: "The full Mechanistic Validity taxonomy: six layers, five validity types, seven description modes, verdict tiers, and design principles."
---

# Framework Overview

Every circuit claim in mechanistic interpretability is a chain from a concrete measurement to a conclusion. This framework names every link in that chain. Reading bottom-up is how you *build* a claim. Reading top-down is how you *evaluate* one.

## Six-layer hierarchy

<img src="/mechanistic-validity/figures/framework/pipeline-vert-example.png" alt="Mechanistic Validity Pipeline with worked example" width="100%"/>

The framework can be read in three directions:

- **Bottom-up (1 → 6):** Building a new claim — what does my metric establish?
- **Top-down (6 → 1):** Auditing an existing claim — what evidence would this verdict require?
- **Sideways (across layer 2):** Checking convergent validity — do independent evidence families agree?

### 1. Description Mode

The level of description at which the claim operates. This is set first because it determines what counts as relevant evidence. A claim about *where* an effect occurs (implementational) requires different metrics than a claim about *what* computation is performed (algorithmic) or *how* information is encoded (representational).

There are three top-level modes and four implementational sub-types (7 total):

| If the evidence establishes... | Licensed tag |
|---|---|
| What function the system computes and why | `[computational]` |
| What procedure the system executes | `[algorithmic]` |
| What information is encoded and in what geometry | `[representational]` |
| Which components are involved | `[implementational–topographic]` |
| How components are wired to each other | `[implementational–connectomic]` |
| Distributional properties of activations | `[implementational–statistical]` |
| What input-output transformation a component performs | `[implementational–functional]` |

### 2. Evidence Family

Classifies a metric's output by the *kind of signal* it produces. Two metrics from different families that agree constitute stronger evidence than two from the same family, because they have structurally different failure modes.

| Family | Signal type | Metrics |
|---|---|---|
| Causal | Interventions and counterfactuals | 13 |
| Structural | Weight-space analysis (no forward pass) | 12 |
| Representational | Latent geometry and decodability | 11 |
| Behavioral | Held-out task and generalization tests | 10 |
| Information-theoretic | Coding properties and information flow | 9 |

### 3. Metrics & Calibrations

**Metrics** are the concrete runnable tests: the things you actually execute on the model. Metrics include activation patching, resample ablation, DAS-IIA, K-composition, copying score, and spectral SVD of weight matrices. A metric produces a number or a set of numbers. By itself, a number from this layer is not a finding — it is data that must be interpreted upward through the hierarchy.

**Calibrations** gate metric outputs before they can serve as evidence. They determine whether a metric's numbers are trustworthy — stable across seeds, separable from random baselines, and replicable. Without calibration, apparent findings may be noise artifacts. Key calibrations include bootstrap stability (F01), random baseline separation (F09), and untrained model baseline (F10).

### 4. Criteria

27 falsifiable, operationally defined conditions across five validity types. Each criterion has a clear pass/fail threshold — no subjective judgment calls.

| Validity type | # | Examples |
|---|---|---|
| Construct | 5 | Falsifiability, structural plausibility, task specificity, minimality, convergent validity |
| Internal | 5 | Necessity, sufficiency, specificity, consistency, confound control |
| External | 6 | Intervention reach, graded response, selectivity, effect magnitude, robustness, cross-architecture generalization |
| Measurement | 6 | Reliability, invariance, baseline separation, sensitivity, calibration, construct coverage |
| Interpretive | 5 | Level declaration, level–evidence match, narrative coherence, alternative exclusion, scope honesty |

### 5. Validity Type

The five abstract dimensions a circuit claim must address. A validity type is satisfied when all its criteria are met; partial satisfaction is reported as partial validity. A claim can be internally valid without being externally valid, and can pass all four traditional types while failing interpretive validity.

### 6. Verdict

The claim itself, stated with explicit scope. A verdict names: (1) the component or set of components, (2) the computation or behavior attributed to them, (3) the model and task, (4) the verdict-strength tier, and (5) the description-mode tag.

## Verdict tiers

| Tier | What it means |
|---|---|
| **Proposed** | No intervention evidence — structural or representational only |
| **Causally suggestive** | Necessity shown (ablation degrades behavior), sufficiency not yet established |
| **Mechanistically supported** | Necessity + sufficiency (ablation + patching, at least 2 ablation variants) |
| **Triangulated** | All internal criteria met, plus at least 1 external and 1 construct criterion |
| **Validated** | All five validity types addressed with explicit baselines and convergent evidence |
| **Underdetermined** | Evidence present but consistent with multiple mechanisms |
| **Disconfirmed** | Fails decisively on a key criterion |

**Example:** L8.MLP causally mediates subject-verb number agreement in GPT-2 Small on the Linzen SVA dataset — *Mechanistically supported* `[implementational–functional]`; external validity and interpretive validity unaddressed.

The "unaddressed" clause is not optional — it prevents silent upgrading by selective citation.

## Dependency order

The validity types gate each other. Skipping a step is the most common structural problem in MI papers.

1. **Construct** — define the construct before calibrating metrics
2. **Measurement** — calibrate metrics before making causal claims
3. **Internal** — establish causal evidence with trustworthy measurements
4. **External** — generalize only after establishing a local result
5. **Interpretive** — audit the assembled verdict

<details class="worked-example">
<summary>Common dependency violations</summary>

| Pattern | What it looks like | Violation |
|---|---|---|
| Circular construct | Circuit named for what metrics found | Construct: no independent definition |
| Uncalibrated IIA | IIA 0.48 without random-vector baseline | Measurement gates internal |
| Single-seed generalization | Reported as robust from one seed | Measurement (reliability) gates internal |
| Cross-arch before local | Cross-model result before within-model necessity | Internal gates external |
| Algorithmic tag from ablation | "implements SVA" from ablation alone | Interpretive: tag not licensed |
</details>

## Design principles

The framework produces a structured verdict, not a scalar score. Two circuits with identical faithfulness numbers can have radically different validity profiles — you cannot upgrade a claim to *Validated* by piling up more evidence of one type while the others remain unaddressed. Most circuit-discovery work should land at *Causally suggestive* or *Mechanistically supported*, and that is a real result, not a failure.

Description level is a property of the claim, not the metric. Activation patching is not inherently an algorithmic-level test; DAS-IIA is not inherently representational. The description tag is determined by what the researcher concludes, not by which tool produced the number.

When two metrics disagree about which components constitute the circuit, the disagreement is itself a primary finding — it points to a methodological gap, sensitivity to different properties, or an underspecified construct. Multi-metric studies should report overlap (e.g., Jaccard similarity) as a first-class result rather than picking a winner.

An IIA score of 0.48 at L8.MLP is a measurement. Whether it constitutes *evidence* that L8.MLP implements a mechanism is a validity question — one that requires knowing the random-vector baseline, the untrained-model baseline, the cross-task control, and the replication rate across seeds. Evidence tells us what we measured; validity tells us what we can conclude.
