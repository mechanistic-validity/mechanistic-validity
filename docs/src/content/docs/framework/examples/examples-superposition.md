---
title: "Case Study: Superposition"
description: "The superposition hypothesis (Elhage et al. 2022) evaluated through all five validity lenses."
---

# Case Study: Superposition

[Elhage et al. (2022)](https://arxiv.org/abs/2209.10652) propose the **superposition hypothesis**: neural networks represent more features than they have dimensions by encoding features as nearly-orthogonal directions in activation space. When features are sparse (rarely co-active), the model can "pack" many features into a lower-dimensional space with minimal interference, because the near-orthogonal directions rarely collide.

This is a *theoretical claim about the representational strategy* of neural networks, demonstrated primarily in toy models. It is not a circuit claim — it is a claim about how representations are organized.

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1/C2 | C3 (real models) | Strong (toy) |
| Internal | I1/I2 (toy) | I6 (double dissociation) | Mechanistically Supported |
| External | E1–E2, E4–E5 (toy) | E4 Cross-model recurrence | Partial |
| Measurement | M2/M3 (toy) | M6 (real) | Partial |
| Interpretive | V2 Level-evidence match | V5 (community overstatement) | Partial |

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

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) N/A.** Superposition is a representational strategy, not a task-specific mechanism. It should appear whenever features outnumber dimensions and sparsity allows packing. This is a general theory, not a task-specific circuit.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) N/A.** Not a circuit claim — minimality does not directly apply.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** In toy models, superposition is confirmed through multiple analyses (weight geometry, interference patterns, phase diagrams). In real models, superposition is *inferred* (SAE features outnumber neurons, polysemanticity exists) but not directly *confirmed* with the same precision as in toy models.

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

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Pass (toy).** In the toy setting, ablating one feature direction selectively impairs that feature without catastrophically affecting others (because directions are nearly orthogonal). Some interference exists (crosstalk), but it is proportional to the geometric dot product — exactly as predicted.

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Pass.** Superposition emerges consistently across toy model training runs. The phenomenon is robust to random seed, model size (within the overcomplete regime), and training details.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Pass (toy).** In toy models, the mechanistic account is complete enough to rule out confounds — the geometry fully explains the interference patterns.

**For real models:** All criteria drop to "inferred but not demonstrated." Superposition is the *explanation* for why SAE features outnumber neurons, but direct confirmation (measuring the geometry of superposed features in GPT-2, verifying that interference matches predictions) is limited.

### Key Distinctions

- **Single vs double dissociation:** Toy models achieve double dissociation — ablating feature A impairs task A but not task B, and vice versa. In real models, only single dissociations (SAE feature ablation degrades related inputs) have been shown.
- **Lesion vs stimulation:** Both lesion (projection out of feature direction) and stimulation (activation addition along feature direction) produce predicted effects in toy models — a strong converging pair. Real models have stimulation evidence (SAE steering) but weaker lesion evidence.
- **Localization vs distributed:** The theory explicitly predicts distributed representation (features spread across dimensions via superposition). This is a case where "not localized" is the correct mechanistic claim, not a failure of localization.

### Dissociation Matrix

|  | Feature A detection | Feature B detection | Feature C detection |
|---|---|---|---|
| Ablate direction A | **↓↓ (confirmed, toy)** | Minimal (predicted crosstalk) | Minimal (predicted crosstalk) |
| Ablate direction B | Minimal (predicted crosstalk) | **↓↓ (confirmed, toy)** | Minimal (predicted crosstalk) |
| Ablate direction C | Minimal (predicted crosstalk) | Minimal (predicted crosstalk) | **↓↓ (confirmed, toy)** |

In toy models: full matrix filled with clean double dissociation plus predicted small off-diagonal crosstalk. In real models: no equivalent matrix constructed — only single dissociations via SAE feature ablation have been demonstrated.

---

## Pharmacology Lens — External Validity

*Does intervening on superposed features produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Pass (toy).** Feature directions can be stimulated and the effect propagates to outputs. In real models, SAE feature steering is the analog — and it works sometimes.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Pass (toy).** Stimulation magnitude produces graded effects in toy models.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Pass (toy) with predicted crosstalk.** Intervention along one direction slightly activates correlated features — exactly as the interference theory predicts.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Strong (toy).** Features account for all model behavior in toy settings.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Pass (toy).** Consistent across the input distribution.

**[E4 — Cross-model recurrence:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) The critical gap.** The theory is demonstrated in toy ReLU networks. Whether it applies to transformers, to models with attention, to models trained on natural language — this is the entire question. The theory *predicts* it should, but direct confirmation is limited to indirect evidence (SAE success implies superposition exists).

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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Pass (toy).** Measurements are deterministic and replicable.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Pass (toy).** The phenomenon is invariant across model sizes (in the overcomplete regime).

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Pass.** Superposition is clearly distinguishable from non-superposition (dedicated-neuron regime). The phase transition between regimes is sharp.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Pass (toy).** The geometric measurements (cosine similarity between feature directions) precisely quantify the degree of superposition.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Pass (toy).** Predicted interference matches measured interference quantitatively.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** In toy models, coverage is complete. In real models, we measure indirect indicators (SAE feature count > neuron count) rather than directly measuring the geometric structure of superposition.

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

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Strong.** The story (features > dimensions → exploit sparsity → pack into nearly-orthogonal directions → accept crosstalk proportional to co-occurrence) is mathematically precise, mechanistically clear, and empirically confirmed in toy settings.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Pass (toy).** In toy models, the geometric account is the only explanation for the observed weight structure. In real models, alternatives exist (polysemanticity could arise from other causes, SAE dictionaries could be overcomplete artifacts).

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
