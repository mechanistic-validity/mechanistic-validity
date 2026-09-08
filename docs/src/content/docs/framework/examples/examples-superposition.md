---
title: "Case Study: Superposition"
description: "The superposition hypothesis (Elhage et al. 2022) evaluated through the five core lenses."
---

# Case Study: Superposition

[Elhage et al. (2022)](https://arxiv.org/abs/2209.10652) propose the **superposition hypothesis**: neural networks represent more features than they have dimensions by encoding features as nearly-orthogonal directions in activation space. When features are sparse (rarely co-active), the model can "pack" many features into a lower-dimensional space with minimal interference, because the near-orthogonal directions rarely collide.

This is a *theoretical claim about the representational strategy* of neural networks, demonstrated primarily in toy models. It is not a circuit claim — it is a claim about how representations are organized.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I6 (double dissociation).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability, C2 Structural plausibility, C3 Convergent validity, C5 Nomological validity (all confirmed) | C6 Complementation validity (untested) | Strong |
| Internal | I1/I2 (toy) | I6 (double dissociation) | Mechanistically Supported |
| External | E5 Graded response, E6 Novel prediction (both confirmed) | E1, E2, E3, E4 (all partial) | Partial |
| Measurement | M2/M3 (toy) | M6 (real) | Partial |
| Interpretive | V1, V2, V4, V5 (all confirmed) | V3 Alternative level (partial) | Strong |

**Overall verdict: Mechanistically Supported.** The superposition hypothesis has strong structural and causal evidence in toy models — the theory is mathematically precise and empirically confirmed in that setting. The capping criterion is I6 (double dissociation): no study has performed a crossed design testing a second representational property that superposition spares while a matched control feature-packing scheme impairs.

The gap to real models remains the central open question. SAE success is consistent with superposition but does not confirm it — SAEs could work for other reasons, and the geometric structure of superposition in real models has not been measured with the precision achieved in toy models.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Weight geometry analysis (near-orthogonal feature directions) | [B01 SVD/Spectral](/mechanistic-validity/framework/metrics/#b01) | Structural |
| Feature ablation (projection out of feature direction) | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Feature stimulation (activation addition) | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| Phase diagram analysis (feature/dimension ratio) | [B01 SVD/Spectral](/mechanistic-validity/framework/metrics/#b01) | Structural |
| Interference / crosstalk measurement | [C01 Mutual Information](/mechanistic-validity/framework/metrics/#c01) | Information |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "superposition" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The hypothesis predicts: (1) in toy models with more features than dimensions, the model should learn nearly-orthogonal feature directions, (2) interference (crosstalk) should be proportional to feature co-occurrence frequency, (3) features should transition from dedicated neurons (no superposition) to superposed representations as the feature-to-dimension ratio increases. All three predictions are testable in toy settings.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Pass (in toy models).** The toy model weights are verified to encode features as nearly-orthogonal directions. The Johnson-Lindenstrauss lemma guarantees that this is geometrically possible. The structural evidence directly confirms the theoretical prediction — in the toy setting.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) **Partial.** The measures are built to separate outcomes the planted data distinguishes in advance: a feature is unlearned, superposed, or given a dedicated dimension, and the norm of the embedding vector together with the projection of other features onto it decides which. The correlated-feature experiments add a second separation, between superposition and PCA, by driving the model from one to the other as sparsity falls. Discriminant validity for the wider claim rests on polysemanticity implying capacity pressure, which is not itself tested.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** The load-bearing parts are named and the smallest case is solved in closed form. The data assumptions are untested.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Confirmed.** Theory and experiment agree, and three outside groups replicated the result before publication. The toy-to-real gap is scored under E2 and E4, not here.

### Key Distinctions

- **Observable vs theoretical:** Superposition is a theoretical construct in real models (inferred from necessity of packing) but directly observable in toy models (weight geometry is fully inspectable). The gap between these is the central open question.
- **Confirmation vs corroboration:** Toy model results confirm the theory (strong prediction + observation). Real-model evidence merely corroborates it — SAE success is consistent with superposition but does not uniquely entail it.
- **Underdetermination:** In real models, the observation "features outnumber neurons" is underdetermined — it could arise from superposition, from redundant/distributed encoding, or from dictionary overcomplexity in the measurement tool.

### Nomological Network

The superposition hypothesis connects to:
- **Weight geometry** — toy model weights encode features as nearly-orthogonal directions (structural, confirmed)
- **Interference patterns** — crosstalk proportional to feature co-occurrence (predicted, confirmed in toy)
- **Phase transitions** — sharp transition from dedicated to superposed regime as feature/dimension ratio increases (predicted, confirmed in toy)
- **SAE feature count** — SAE dictionaries outnumber neurons in real models (observational, consistent but not uniquely entailed)
- **Polysemanticity** — individual neurons respond to multiple unrelated features (observational, consistent)
- **Geometric verification in real models** — directly measuring near-orthogonal feature directions in transformer activations (untested at toy-model precision)
- **Training dynamics** — does superposition emerge gradually or as a phase transition during training? (partially explored in toy models, untested in real)

Five nodes confirmed or consistent (in toy models), two unconnected at real-model scale. A thick network within its scope, with the toy-to-real gap as the primary missing bridge.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that the model implements superposition?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Pass (toy).** In toy models, ablating a feature direction (projecting it out) removes the model's ability to detect that feature. The effect is specific and proportional.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Pass (toy).** Stimulating along a feature direction (adding activation in that direction) produces the expected output — the model acts as if the feature is present. In toy models, this is a clean sufficiency result.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Not applicable.** There is one task and one loss, and the paper states that cross-superposition loss comparison fails, so there is no matched control task to be more specific than.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) **Partial.** The generator is under complete control, so sparsity, importance, correlation and dimension are set independently rather than adjusted for after the fact, and the phase-change experiment is designed specifically to stop many features from moving at once. The confound that remains is optimization: the authors report that the m = 2 case is much harder for gradient descent than three dimensions and that the reported solutions are selected by loss, so which geometry appears depends on the optimizer.

**For real models:** All criteria drop to "inferred but not demonstrated." Superposition is the *explanation* for why SAE features outnumber neurons, but direct confirmation (measuring the geometry of superposed features in GPT-2, verifying that interference matches predictions) is limited.

### Key Distinctions

- **Single vs double dissociation:** No double dissociation is run, in toy models or real ones. Importance and sparsity are crossed, but on a single outcome, and no converse arm shows some property intact while superposition is removed. This is the criterion that caps the claim.
- **Lesion vs stimulation:** Both lesion (projection out of feature direction) and stimulation (activation addition along feature direction) produce predicted effects in toy models — a strong converging pair. Real models have stimulation evidence (SAE steering) but weaker lesion evidence.
- **Localization vs distributed:** The theory explicitly predicts distributed representation (features spread across dimensions via superposition). This is a case where "not localized" is the correct mechanistic claim, not a failure of localization.

### Dissociation Matrix

|  | Feature A detection | Feature B detection | Feature C detection |
|---|---|---|---|
| Ablate direction A | **↓↓ (confirmed, toy)** | Minimal (predicted crosstalk) | Minimal (predicted crosstalk) |
| Ablate direction B | Minimal (predicted crosstalk) | **↓↓ (confirmed, toy)** | Minimal (predicted crosstalk) |
| Ablate direction C | Minimal (predicted crosstalk) | Minimal (predicted crosstalk) | **↓↓ (confirmed, toy)** |

The diagonal is filled by construction — ablating a feature direction removes that feature — and the off-diagonal cells are predicted crosstalk rather than measured converse arms. No experiment shows a property intact while superposition is removed, which is why I6 caps the claim.

---

## Pharmacology Lens — External Validity

*Does intervening on superposed features produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) **Partial.** Interventions reach four different parts of the setup: the architecture, the data, L1 regularization on the hidden activations, and adversarial training. The last two are aimed at removing superposition and both do, with the cost reported rather than buried — including the admission that attacks had to reach 80% of the input L2 norm to eliminate it. Every intervention is on a toy model, which bounds the reach.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Pass (toy).** Stimulation magnitude produces graded effects in toy models.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) **Partial.** The input distribution is varied along each axis the theory identifies, and the sweeps are wide. What does not vary is the family: synthetic vectors that are zero with a set probability and uniform otherwise, with the authors calling the uniform choice arbitrary. No natural-data distribution appears anywhere, and Open Questions asks whether the importance and sparsity curves of real models can be estimated at all.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) The critical gap.** The theory is demonstrated in toy ReLU networks. Whether it applies to transformers, to models with attention, to models trained on natural language — this is the entire question. The theory *predicts* it should, but direct confirmation is limited to indirect evidence (SAE success implies superposition exists).

### Key Distinctions

- **Affinity vs efficacy:** In toy models, feature directions demonstrate both affinity (they correlate with the feature) and efficacy (intervening on them causally changes behavior). In real models, SAE feature directions show affinity but efficacy is inconsistent — many features steer weakly or incoherently.
- **The system compensates:** The theory predicts that crosstalk from interference should be small when features are sparse. In real models, compensatory mechanisms (attention, MLPs) may actively correct interference, meaning the "raw" superposition geometry may not reflect the effective computation.

### Dose-Response Curve

In toy models, the dose-response relationship is fully characterized:
- **Graded stimulation**: adding activation along a feature direction at varying magnitudes produces proportional output changes
- **Predictable interference**: crosstalk magnitude scales with the dot product between feature directions, quantitatively matching geometric predictions
- **Sharp phase transitions**: as the feature/dimension ratio increases, the model transitions from dedicated (no interference) to superposed (graded interference) — a well-characterized dose-response at the architectural level

What's missing for real models:
- **No parametric sweep of SAE feature steering magnitude** with systematic measurement of both on-target and off-target effects
- **No interference measurement** — when steering one SAE feature, how much do correlated features activate as a function of steering strength?
- **No therapeutic window estimate** — at what steering magnitude do off-target effects begin to dominate?

The toy-model curve is complete and quantitatively predicted by theory. The real-model curve is scattered data points (SAE steering sometimes works) without systematic dose-response characterization.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics measuring superposition reliably?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) **Partial.** Repetition happens and the rule is stated each time, which separates this from an unreported single fit. The rule is selection in two of three cases: the phase diagram averages ten models per point after discarding the worst, the m = 2 geometries are the lowest-loss of several fits, and the neuron-level figures are the best of a thousand. No standard deviation, interval or spread across the repeated fits accompanies any of them.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) **Partial.** The phenomenon survives changes of scale and data structure: results are qualitatively similar as features and hidden dimensions grow, and the number of input features stops mattering once it exceeds the hidden dimension. Held fixed are the activation function, the importance-weighted squared-error loss and the optimizer. The one report of varying the activation function comes from an external replicator and finds the phase diagrams differ.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Pass.** Superposition is clearly distinguishable from non-superposition (dedicated-neuron regime). The phase transition between regimes is sharp.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Pass (toy).** The geometric measurements (cosine similarity between feature directions) precisely quantify the degree of superposition.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) **Partial.** Both measures come with an interpretation attached rather than a threshold chosen after the fact: the embedding norm says whether a feature is represented, and the summed projection of other features onto its direction has a stated meaning at 1. Feature dimensionality is anchored by summing to the number of hidden dimensions, which makes it a capacity accounting. Nothing is tied to a quantity outside the toy model.

### Key Distinctions

- **Reliability vs validity:** In toy models, the measurement is both reliable (deterministic, replicable) and valid (measures what it claims — the geometry of superposition). In real models, indirect measurements (SAE feature count) are reliable but their validity as measures of superposition specifically is uncertain.
- **True score vs observed score:** In toy models, the true score (actual geometric structure) is directly accessible. In real models, observed scores (SAE reconstruction loss, feature count) are noisy proxies for the true underlying superposition geometry.

### MTMM Matrix

| | Weight geometry (toy) | Interference measurement (toy) | SAE feature count (real) | Polysemanticity score (real) |
|---|---|---|---|---|
| **Weight geometry (toy)** | — | High (converge on same features) | N/A (different domain) | N/A |
| **Interference measurement (toy)** | High | — | N/A | N/A |
| **SAE feature count (real)** | N/A | N/A | — | Moderate (correlated) |
| **Polysemanticity score (real)** | N/A | N/A | Moderate | — |

Within the toy domain: strong convergent validity — multiple methods agree on which features are superposed and to what degree. Within the real-model domain: moderate convergent validity — SAE feature count and polysemanticity co-occur but neither directly measures superposition geometry. Cross-domain (toy-to-real): no direct validity comparison possible because the measurement metrics differ fundamentally. This gap in the MTMM is the measurement-theoretic expression of the toy-to-real transfer problem.

---

## MI Lens — Interpretive Validity

*Is "superposition" the right interpretation?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** The claim is [representational](/mechanistic-validity/framework/modes/representational) — about how features are geometrically organized in activation space.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Pass.** Representational evidence (geometric analysis) supports a representational claim. Match is direct.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** PCA is examined and closed as a limiting case. The optimization alternative is raised and left open: the $m = 2$ case is reported as hard for gradient descent, its solutions are selected by loss, and an invited comment in the same article shows the global minimum can have a smaller basin of attraction than nearby local minima.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Mostly honest.** The paper is clear that toy models are the primary evidence. The jump to real models is framed as a hypothesis, not a conclusion. Subsequent work (SAE papers) sometimes treats superposition as established in real models, which overstates the evidence.

### Key Distinctions

- **Description vs explanation:** The superposition hypothesis moves beyond description (features are polysemantic) to explanation (because features outnumber dimensions and sparsity permits packing). This explanatory depth is a key strength.
- **Faithfulness vs understanding:** The toy model analysis is both faithful (correctly describes what happens) and provides understanding (explains why). In real models, the theory provides understanding (a plausible why) but faithfulness to the actual mechanism is unconfirmed.
- **Activation evidence vs weight evidence:** The primary evidence is weight-based (geometric structure of learned directions). This is appropriate because the claim is about representational geometry, not about activation patterns on specific inputs.

### Evidence Convergence Map

- **Implementational → Interpretation:** Strong (toy). Weight geometry directly shows feature directions packed into the available dimensions. In real models, indirect — SAE dictionaries suggest overcomplete representations but do not directly display the geometric packing.
- **Representational → Interpretation:** Strong (toy). The claim IS representational and the evidence IS representational. Direct match. In real models, representational evidence (SAE features) is consistent but underdetermining.
- **Computational → Interpretation:** Moderate. The theory explains WHY the model adopts superposition (it needs more features than dimensions), connecting representational structure to computational demands.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Feature ablation (toy) | ✓ | — | ✓ | ∅ | ∅ |
| Feature stimulation (toy) | — | ✓ | ✓ | ∅ | ∅ |
| Phase diagram (toy) | — | — | ✓ | — | ✓ |
| SAE steering (real) | — | Partial | Partial | ∅ | ∅ |
| SAE reconstruction (real) | — | — | Partial | — | — |

Strong coverage in the representational column for toy models. Real-model evidence clusters at "Partial" — consistent with but not uniquely establishing the interpretation.

### Causal Sufficiency Graph

- Feature direction → model output: **solid (toy)** — stimulating or ablating feature directions directly and predictably changes output
- Feature sparsity → superposition regime: **solid (toy)** — the phase diagram causally links sparsity to the degree of superposition
- SAE feature direction → model output: **dashed (real)** — SAE steering sometimes works, causal link is inconsistent
- Superposition (theory) → polysemanticity (observation): **dashed** — the theory predicts polysemanticity, but polysemanticity has other possible causes

The toy-model causal graph is fully connected with solid edges. The real-model causal graph has only dashed edges — consistent correlational evidence without confirmed causal pathways at the same precision.

---
