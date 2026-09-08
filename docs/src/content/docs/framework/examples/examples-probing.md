---
title: "Case Study: Probing Classifiers"
description: "Linear probing as a methodology for representational claims, evaluated through all five validity lenses."
---

# Case Study: Probing Classifiers

Linear probing (Alain & Bengio 2017, Belinkov 2022) trains a linear classifier on model activations to test whether a concept (part-of-speech, syntax tree depth, sentiment, factual knowledge) is **linearly decodable** from the representation. The claim is representational: if a probe succeeds, the model "encodes" or "represents" the probed concept.

This case study evaluates probing *as a methodology* rather than a specific circuit. The question is not "does probe X work?" but "what does probe success actually establish?"

## Composite Verdict

> **Verdict (framework paper, Table 6):** Proposed. **Capped by:** I1 (necessity).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability (confirmed) | C4 Discriminant validity (disconfirmed) | Weak |
| Internal | I5 Rival mechanism exclusion, I7 Confound control (both partial) | I4 Specificity (disconfirmed), I1 Necessity (inconclusive, the capping criterion) | Very weak |
| External | E4 Cross-model generalization (confirmed) | E1 Intervention reach (inconclusive), E5 Graded response (N/A) | Partial |
| Measurement | M2 Baseline separation (confirmed) | M4 Calibration (disconfirmed), M3 Stability (inconclusive) | Weak |
| Interpretive | V3 Alternative level, V5 Scope declaration (both confirmed) | V2 Level-evidence match (disconfirmed) | Partial |

**Overall verdict: Proposed (without causal follow-up).** Standard linear probing, without causal intervention or control tasks, does not advance a claim beyond *Proposed*. The evidence establishes decodability but not encoding, representation, or use.

Probing *with* causal follow-up (DAS, causal abstraction, intervention along probe direction) can advance to *Causally suggestive*. Probing *with* control tasks + causal intervention + cross-method convergence can reach *Mechanistically supported*.

This case study illustrates a fundamental principle of the framework: **a measurement without an intervention is a measurement without internal validity.** Probing measures a property of the activation space. Whether that property is causally relevant to the model's computation requires a different kind of evidence entirely.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Linear probing | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| Control tasks (Hewitt & Liang 2019) | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| DAS / causal intervention along probe direction (Geiger et al.) | [E01 DAS-IIA](/mechanistic-validity/framework/metrics/#e01) | Representational |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "the model represents X" a coherent construct when supported only by probe success?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Partial.** A probe failure (low accuracy) would disconfirm the representation claim. But probe *success* is ambiguous — it could reflect genuine encoding or could reflect that the concept is linearly separable in any high-dimensional space, even without genuine representation. The falsifiability is asymmetric: failure is informative, success is not clearly so.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Weak.** Probes operate on activations, not weights. They do not identify *which* parameters encode the concept or *how* the encoding is implemented in the model's architecture. A probe success is consistent with intentional encoding, accidental encoding, and encoding-as-artifact.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Variable.** Some probes test specific concepts (syntax tree depth). Others test broad concepts (sentiment). The discriminant question — does the probe *fail* on closely related non-matches? — is often not tested. A "syntax depth" probe might succeed because the activation space also linearly encodes sentence length, which correlates with depth.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Not applicable.** Probes identify a direction/subspace, not a circuit. The analog of minimality (is this the minimal subspace encoding the concept?) is rarely tested — probes operate at the layer level and do not investigate whether a lower-dimensional subspace would suffice.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Weak.** Probing is typically the *only* method used. Convergent validity would require confirming the representation claim via an independent method — causal intervention along the probe direction, weight-space analysis, or cross-method agreement. When probing alone is the evidence, convergent validity is absent by definition.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Partial | Failure informative; success ambiguous |
| C2 Structural plausibility | Weak | Activation-level, not weight-level |
| C4 Discriminant validity | Variable | Confounds often untested |
| I3 Minimality | N/A | Not a circuit-level claim |
| C3 Convergent validity | Weak | Usually single-method |

### Key Distinctions

- **Confirmation vs corroboration:** Probe success is confirmation (the probe finds what you looked for) without corroboration (no independent method verifies the same representation claim). A concept that is decodable by a probe AND causally manipulable by intervention along the probe direction would constitute genuine corroboration.
- **Underdetermination:** High probe accuracy is consistent with multiple explanations — genuine encoding, incidental linear separability, confound encoding (correlated feature rather than target feature). The probe accuracy underdetermines the representational claim.
- **Operationalism vs realism:** "The model represents syntax depth" is a realist claim. "A linear classifier achieves 85% accuracy on syntax depth labels from layer 6 activations" is an operationalist statement. Probing provides the operationalist evidence; the realist interpretation requires additional justification.

### Nomological Network

The probing-based representation claim connects to:
- **Linear decodability** — a probe can extract the concept from activations (observational, confirmed by definition)
- **Causal use** — the model reads from this direction during inference (causal, almost always untested)
- **Weight-space grounding** — specific parameters implement the encoding (structural, untested)
- **Cross-layer consistency** — the representation persists or transforms predictably across layers (partial, sometimes tested)
- **Control task separation** — accuracy exceeds random-label baseline (confound control, often untested)
- **Cross-distribution transfer** — probe generalizes to new text domains (robustness, sometimes tested)
- **Intervention effect** — patching along probe direction changes behavior (causal, tested only in DAS-style follow-ups)

One node confirmed by construction (decodability), one or two partially tested in good papers, four typically untested. The nomological network for standard probing is extremely thin — most of the predictive power of the "model represents X" claim is never tested.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that the probed representation is causally used?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Not tested (typically).** Standard probing does not ablate the identified direction and measure behavioral change. Without this, we do not know if the probed representation is causally used by the model. It could be a byproduct that the model never reads from.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Not tested (typically).** Patching along the probe direction to shift behavior is the sufficiency test. DAS (Geiger et al. 2023) does this — it is a probe + causal intervention combined. Standard probing without intervention tests neither necessity nor sufficiency.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Disconfirmed.** A control dataset holds the property non-discriminative for the original task and the probe recovers it anyway, so decodability does not localize to the task.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Weak — the critical gap.** The fundamental confound: in high-dimensional activation spaces, *many* concepts are linearly decodable even from random representations ([Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368), "control tasks"). A probe may succeed not because the model encodes the concept but because the concept is linearly separable in any space with sufficient dimensionality. Without a control task (probing on random labels or on a related-but-different concept), probe accuracy is uninterpretable.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Not tested | No ablation of probed direction |
| I2 Sufficiency | Not tested | No intervention along probe |
| I4 Specificity | Not tested | No off-target measurement |
| M1 Reliability | Partial | Single distribution typical |
| I7 Confound control | Weak | High-dimensional separability confound |

### Key Distinctions

- **Single vs double dissociation:** Standard probing provides no dissociation evidence at all — it is purely observational. DAS-style extensions provide single dissociation (intervening on the direction changes the target behavior). Double dissociation (intervening on direction A does NOT change behavior B, and intervening on direction B does NOT change behavior A) is almost never tested in the probing literature.
- **Lesion vs stimulation:** Probing uses neither. It is a passive measurement — the neural analog of recording without intervening. The transition from "recordable" to "causally relevant" requires an intervention that probing does not provide.

### Dissociation Matrix

|  | Probed concept (behavior) | Related concept (behavior) | Unrelated concept (behavior) |
|---|---|---|---|
| Ablate probed direction | ? | ? | ? |
| Ablate related direction | ? | ? | ? |
| Patch along probed direction | ? (DAS only) | ? | ? |

Entirely empty for standard probing. Even DAS-style extensions fill at most one cell (patch along probed direction → probed concept changes). The matrix makes visible the complete absence of causal evidence in standard probing — every cell is a "?" that would need to be filled to establish that the probed representation is causally real.

---

## Pharmacology Lens — External Validity

*Does intervening on the probed direction produce predictable downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Not tested.** Standard probes do not intervene. DAS-style extensions do, and they sometimes find that intervention along the probe direction does *not* produce the expected behavioral change — the representation is present but not causally used.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) N/A.** No intervention = no dose-response.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Variable.** Some probes generalize across text types. Many do not — performance degrades on out-of-distribution text.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Confirmed.** Portability is established without qualification, from static word embeddings through recurrent and recursive networks to transformers, and outward to speech recognition and computer vision.

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Not tested | Probes don't intervene |
| I4 Specificity | N/A | — |
| E5 Graded response | N/A | — |
| E2 Prompt generalization | Variable | Domain-dependent |
| E4 Cross-model generalization | Partial | Applied broadly, rarely compared |

### Key Distinctions

- **Affinity vs efficacy:** Probing measures affinity only (the concept is present/decodable in the activation space). Efficacy (whether the model uses this information to drive behavior) is entirely untested. A representation with high affinity and zero efficacy is a byproduct, not a functional encoding.
- **The fundamental pharmacological gap:** A drug that binds to a receptor (affinity) but produces no physiological effect (no efficacy) is not a therapeutic agent. Similarly, a probe that detects a concept (affinity) but provides no evidence of causal use (no efficacy) does not establish functional representation. Probing is a binding assay, not a clinical trial.

### Dose-Response Curve

Standard probing produces no dose-response data — there is no intervention to dose. The closest analog:
- **Probe accuracy as a function of layer** — sometimes shows a curve (accuracy increases, peaks, then decreases across layers). This is informative about where information is available but says nothing about causal use.
- **DAS intervention strength** — when DAS-style follow-ups are performed, patching at varying strengths can produce a dose-response. But this is no longer standard probing; it is a different methodology.

For standard probing, the dose-response section is N/A — no intervention means no curve to characterize.

---

## Measurement Theory Lens — Measurement Validity

*Is the probe a reliable and well-calibrated metric?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Partial.** Probe accuracy varies with: probe architecture, training hyperparameters, layer choice, and dataset. The same concept probed with different settings gives different accuracy numbers. Reliability is conditional on methodological choices.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Weak.** A probe trained on one dataset may not transfer to another. The measurement is distribution-specific rather than model-intrinsic.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Confirmed.** This is the criterion the literature solved: three control families are formalized, including control tasks [Hewitt & Liang (2019)](https://arxiv.org/abs/1909.03368), random-feature baselines and selectivity. What caps the claim is necessity (I1), not the baseline.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Variable.** Linear probes may miss nonlinear representations. This is a known limitation — choosing a linear probe is both a strength (constraining the hypothesis) and a weakness (potentially missing genuine encodings that are nonlinear).

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Poorly understood.** What does 75% probe accuracy mean? Without calibration against models of known representational capacity, the number is hard to interpret. Is 75% "the model represents this concept weakly" or "the probe is underpowered"?

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Partial | Sensitive to methodological choices |
| M6 Invariance | Weak | Distribution-specific |
| M2 Baseline separation | Critical gap | Without control tasks, uninterpretable |
| M5 Sensitivity | Variable | Linear constraint: feature and limitation |
| M4 Calibration | Poorly understood | Accuracy numbers hard to interpret |
| C3 Convergent validity | Partial | Single-layer decodability only |

### Key Distinctions

- **Reliability vs validity:** Probe results are unreliable (vary with hyperparameters, architecture, dataset) and of uncertain validity (decodability does not equal representation). The combination is particularly concerning — we cannot even confirm that the metric produces stable measurements, let alone that those measurements reflect something real about the model.
- **Convergent vs discriminant validity:** Standard probing provides neither. Convergent: does a different method (causal intervention, weight analysis) confirm the same representation? Discriminant: does the probe for concept A give *low* scores when applied to concept B? Both are almost always absent.
- **The metric may create the signal:** A sufficiently powerful probe can decode many concepts from any high-dimensional space. The "measurement" may be a property of the probe (its capacity to find linear separability) rather than a property of the model (its representational content). This is the Hewitt & Liang insight formalized as a measurement-theoretic concern.

### MTMM Matrix

| | Linear probe (concept A) | Linear probe (concept B) | DAS (concept A) | Weight analysis (concept A) |
|---|---|---|---|---|
| **Linear probe (concept A)** | — | ? (discriminant) | ? (convergent) | ? (convergent) |
| **Linear probe (concept B)** | ? | — | ? | ? |
| **DAS (concept A)** | ? | ? | — | ? |
| **Weight analysis (concept A)** | ? | ? | ? | — |

Entirely empty. Standard probing does not produce MTMM data because it uses one method on one concept at a time. The absence is not incidental — it reflects the fundamental single-method nature of probing. Filling even one convergent cell (does DAS confirm what the probe found?) would substantially strengthen the representational claim; filling a discriminant cell (does the probe for A score low on B?) would address confound concerns. Neither is standard practice.

---

## MI Lens — Interpretive Validity

*Is "the model represents X" warranted by probe success?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** Probing makes a [representational](/mechanistic-validity/framework/modes/representational) claim — the model encodes a concept.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Disconfirmed.** The evidence supports extractability; the claim made is representation. Probe success is reported as "the model represents X" when what holds is "X is decodable from layer L by a classifier of the stated capacity" — and the model may not *use* the information at all.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Confirmed.** Every rival reading of a good probe score is given a measure of its own — control tasks, selectivity, random-feature baselines and description length each target a different alternative account.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Confirmed.** The anchor declares scope end to end, conceding limits before any content.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Pass | Representational |
| V2 Level-evidence match | Partial | Decodable does not equal encoded |
| V3 Alternative level | Weak | Geometric artifact not excluded |
| V5 Scope declaration | Often violated | "Represents" exceeds "decodable" |

### Key Distinctions

- **Description vs explanation:** Probing is purely descriptive — it identifies that a concept is decodable but does not explain the mechanism that produces the encoding or the computation that uses it. The gap from description to explanation requires causal evidence that probing does not provide.
- **Component identity vs component role:** Probing identifies a direction (component identity) but not its role in the model's computation. The probe direction may be a real computational axis or a geometric artifact — probing alone cannot distinguish these.
- **Faithfulness vs understanding:** Probe accuracy measures faithfulness of the external classifier, not understanding of the model. High probe accuracy means the external classifier faithfully decodes the concept — it does not mean we understand how or why the model represents it.

### Evidence Convergence Map

- **Implementational → Interpretation:** Absent. Probing does not identify implementing parameters or structural mechanisms. The direction found by the probe is not linked to specific weights.
- **Algorithmic → Interpretation:** Absent. Probing does not specify the algorithm that produces or reads the representation. The computational steps are entirely uncharacterized.
- **Computational → Interpretation:** Weak. Probing establishes that the concept is available in the representation (a computational-level observation). But availability does not imply use — the model may have access to information it never reads.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Standard probing | — | — | observational only | — | — |
| Probe + control task | — | — | partial (excludes artifact) | — | — |
| DAS (probe + intervention) | partial | partial | ✓ | — | partial |
| Probe + weight analysis | — | — | convergent | — | — |

Standard probing fills zero interventional cells. It provides observational representational evidence only. Each methodological extension (control tasks, DAS, weight-space convergence) fills additional cells, progressively strengthening the claim. The matrix makes visible that standard probing is a starting point, not an endpoint — the interpretive claim requires evidence that probing alone cannot provide.

### Causal Sufficiency Graph

- Input properties → activation pattern: **solid** (by construction — the model processes the input)
- Activation pattern → linear decodability: **solid** (the probe demonstrates this)
- Linear decodability → model representation: **dashed** (the inferential leap — decodable does not imply represented)
- Model representation → downstream computation: **absent** (probing provides no evidence about use)
- Probe direction → model's actual encoding direction: **dashed** (the probe finds *a* direction; whether it is *the* direction the model uses is unverified)

Two solid edges (trivial: inputs produce activations, probes can decode them) and three dashed-or-absent edges (substantive: does decodability imply representation? does the model use this direction?). The causal sufficiency graph makes visible that the interesting claims — those that go beyond "a classifier works" — have no solid causal support from probing alone.

---
