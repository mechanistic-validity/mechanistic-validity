---
title: "Case Study: Successor Heads"
description: "The general-purpose successor mechanism (Hanna et al. 2023, Gould et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Successor Heads

Building on the Greater-Than analysis, subsequent work ([Gould et al. 2023](https://arxiv.org/abs/2312.09230)) identifies **successor heads** as a general-purpose mechanism — attention heads whose $W_{OV}$ matrices encode *ordinal succession* across multiple domains: days of the week (Monday → Tuesday), months (January → February), numbers (1 → 2), and alphabetical sequences (A → B). The claim is that these heads do not merely implement year comparison but encode a general ordinal-successor function that the model reuses across sequence types.

This extends the Greater-Than claim from a task-specific circuit to a *general computational primitive* — a reusable building block. The construct validity question shifts accordingly: is "successor head" a natural kind (a real computational unit) or a family resemblance (a label applied to heads that happen to do ordinal things)?

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C2 Structural plausibility | C5 Convergent | Strong |
| Internal | I1 Necessity | I5 Confound control | Causally suggestive |
| External | E5 Robustness | E1/E6 | Partial |
| Measurement | M2/M3 Invariance + Separation | M1 Reliability | Strong |
| Interpretive | V2/V3 Match + Coherence | V4 Alternative exclusion | Strong |

**Overall verdict: Causally suggestive, approaching Mechanistically supported.** Successor heads benefit from the same structural clarity as the Greater-Than circuit, with the additional strength of cross-domain generalization. The multi-domain pattern makes the "general computational primitive" claim more convincing than a single-task circuit claim. The case for successor heads as a natural kind is stronger than for most circuits because the same structural signature appears across unrelated domains — this is convergent evidence from the phenomenon itself, even without formal C5 convergent validity from multiple discovery methods.

---

## Philosophy of Science Lens — Construct Validity

*Is "successor head" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim predicts that successor heads should encode ordinal structure across *multiple* domains in their $W_{OV}$ matrices. A head that encodes year ordering but not month ordering is a year-specific head, not a general successor head. This is a discriminating prediction.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Pass.** $W_{OV}$ matrices are inspected and shown to encode ordinal structure across domains. The same heads that boost "32 → 33" also boost "Monday → Tuesday" and "B → C." The structural evidence spans domains.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) N/A (honest scope — general purpose).** Successor heads are claimed to be general-purpose, not task-specific. The evidence confirms this — they fire across domains. This is the same honest-scope pattern as induction heads: a general mechanism, honestly described as general.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Pass.** A small number of heads show the multi-domain successor pattern. Not every head in the model does this — the set is selective.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** Evidence from structural analysis ($W_{OV}$) and behavioral analysis (ablation effects on successor tasks) converges. A third method (e.g., probing for ordinal features, EAP discovery) would strengthen convergence.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Pass | Cross-domain structural predictions |
| C2 Structural plausibility | Pass | Multi-domain $W_{OV}$ ordering |
| C3 Task specificity | N/A (general) | Multi-domain by design |
| C4 Minimality | Pass | Small selective set |
| C5 Convergent validity | Partial | Structural + behavioral |

### Key Distinctions

- **Confirmation vs corroboration:** The multi-domain pattern provides genuine corroboration — the same structural signature independently discovered in years, months, days, and letters constitutes multiple independent tests of the "general successor" hypothesis, not a single confirmation replayed across domains.
- **Natural kind vs family resemblance:** The structural signature ($W_{OV}$ encoding ordinal succession) is consistent across domains, suggesting "successor head" is a natural kind rather than a loose family resemblance label. The same mechanism, not just the same behavior.
- **Operationalism vs realism:** "Successor head" is defined by both observable behavior (cross-domain ordinal effects) and structural properties ($W_{OV}$ geometry). This dual grounding makes the label more realist than purely operationalist — the mechanism exists in the weights, not just in the measurements.

### Nomological Network

The successor head construct connects to:
- **Weight structure** — $W_{OV}$ encodes ordinal ordering across multiple domains (structural, confirmed)
- **Cross-domain behavior** — same heads produce successor effects for years, months, days, letters (behavioral, confirmed)
- **Ablation effects** — removing successor heads degrades ordinal predictions (causal, confirmed)
- **Non-ordinal control** — successor heads do not drive non-ordinal tasks (specificity, partially confirmed)
- **Training dynamics** — when does the successor structure emerge during training? (untested)
- **Cross-model prediction** — do other architectures develop the same mechanism? (untested)
- **Probing convergence** — do probes for ordinal features align with successor head directions? (untested)

Four nodes confirmed, three unconnected. A moderately thick network — the cross-domain confirmation is particularly strong as it represents multiple independent tests of the same structural prediction.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that successor heads implement ordinal computation?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Ablating successor heads degrades performance on ordinal/successor tasks across domains. The effect is measurable and domain-general (not just years).

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Partial.** The $W_{OV}$ structure implies the heads *can* compute succession. But full isolation (can these heads alone drive successor behavior with everything else ablated?) is not reported.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Partial.** The cross-domain pattern provides implicit specificity: successor heads are specific to *ordinal* tasks. They should not fire on tasks without ordinal structure (sentiment, syntax). This is partially verified — ablation on non-ordinal tasks shows smaller effects.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** Cross-domain consistency is strong (the same heads work across years, months, letters). Cross-model consistency is limited — are the same heads successors in GPT-2 Medium? In Pythia?

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not tested.** Single ablation method.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Pass | Cross-domain ablation effects |
| I2 Sufficiency | Partial | Structural implication, not isolation |
| I3 Specificity | Partial | Ordinal vs. non-ordinal contrast |
| I4 Consistency | Partial | Cross-domain strong; cross-model limited |
| I5 Confound control | Not tested | Single method |

### Key Distinctions

- **Single vs double dissociation:** Cross-domain ablation provides partial double dissociation — successor heads impair ordinal tasks but show smaller effects on non-ordinal tasks. This is stronger than pure single dissociation but not a formal double-dissociation design with a matched control circuit.
- **Lesion vs stimulation:** Only lesion-style evidence (ablation) is reported. Stimulation (amplifying successor head signals to force ordinal predictions) would test whether the mechanism is steerable, not just necessary.

### Dissociation Matrix

|  | Year succession | Month succession | Letter succession | Non-ordinal control |
|---|---|---|---|---|
| Ablate successor heads | **↓↓** | **↓↓** | **↓↓** | ↓ (small) |
| Ablate non-successor heads | ? | ? | ? | ? |

The top row is well-filled across domains — a strength. The contrast between ordinal (large effect) and non-ordinal (small effect) provides implicit specificity. However, the converse (ablating non-successor heads and measuring ordinal task impact) is not tested, leaving the formal double-dissociation incomplete.

---

## Pharmacology Lens — External Validity

*Does intervening on successor heads produce predictable downstream effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Not tested.** Can you steer the model toward successor behavior (make it always predict the next item in any sequence) by stimulating successor heads? Untested.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Implicit.** The $W_{OV}$ structure implies graded effects — items further from the reference should receive proportionally stronger boosts. Not directly measured as a dose-response.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Partial.** The cross-domain generality is both a strength and a limitation: the mechanism is selective for ordinal tasks but not selective for any *particular* ordinal domain.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Moderate.** Successor heads contribute meaningfully to ordinal predictions but are not the sole mechanism.

**[E5 — Robustness:](/framework/criteria/external/robustness) Strong (within scope).** Works across years, months, days, letters, numbers. The robustness across domains is the primary evidence.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** Is the successor mechanism a universal attention-head computation or specific to GPT-2's architecture?

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Not tested | — |
| E2 Graded response | Implicit | Structural prediction |
| E3 Selectivity | Partial | Selective for ordinal class |
| E4 Effect magnitude | Moderate | Contributing, not sole mechanism |
| E5 Robustness | Strong | Multi-domain generalization |
| E6 Cross-architecture | Not tested | — |

### Key Distinctions

- **Affinity vs efficacy:** Successor heads show both affinity (they activate on ordinal sequences) and efficacy (ablation degrades ordinal predictions). The cross-domain evidence means this affinity-efficacy pairing is confirmed across multiple independent test cases.
- **Therapeutic window:** Since successor heads are claimed as general-purpose primitives (not task-specific), the concept of a "therapeutic window" shifts — any intervention affects all ordinal tasks simultaneously. There is no selective dosing for one domain.
- **Receptor reserve:** Whether backup mechanisms compensate when successor heads are ablated is not characterized. The partial (non-total) effect of ablation suggests some redundancy exists.

### Dose-Response Curve

The successor heads' dose-response is largely uncharacterized. We have:
- **Complete ablation**: measurable degradation across ordinal domains
- **Cross-domain confirmation**: the same intervention degrades multiple domains (consistent direction)
- **Non-ordinal control**: smaller effect on non-ordinal tasks (selectivity boundary exists)

What's missing:
- **No parametric sweep** — no graded ablation between 0% and 100%
- **No stimulation experiment** — amplifying successor signals to test whether predictions shift toward "next item"
- **No EC₅₀ characterization** — at what intervention strength does the ordinal effect become detectable?

The structural prediction (items further from reference get proportionally larger $W_{OV}$ boosts) implies a dose-response exists in the weights, but this has not been measured as a behavioral curve.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Not reported.** No confidence intervals on ordinal structure measurements.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Pass (within model).** The same heads show successor structure across domains — strong within-model invariance.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** Non-successor heads do not show multi-domain ordinal structure. The measurement cleanly separates.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Good.** The multi-domain criterion is more sensitive than a single-domain criterion — it distinguishes general successor heads from domain-specific ordinal heads.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.**

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Good.** Both structural ($W_{OV}$) and behavioral (multi-domain ablation) evidence. Good coverage.

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Not reported | — |
| M2 Invariance | Pass | Cross-domain consistency |
| M3 Baseline separation | Pass | Clear successor/non-successor distinction |
| M4 Sensitivity | Good | Multi-domain criterion is discriminating |
| M5 Calibration | Not reported | — |
| M6 Construct coverage | Good | Structural + behavioral |

### Key Distinctions

- **Reliability vs validity:** The multi-domain criterion provides strong face validity (the measurement captures something real about ordinal computation). But without confidence intervals or test-retest measurements, reliability is assumed rather than demonstrated.
- **Convergent vs discriminant validity:** The same measurement (ordinal structure in $W_{OV}$) converges across domains — this is implicit convergent validity from the phenomenon. Discriminant validity (do these heads score *low* on non-ordinal structure metrics?) is partially demonstrated through the non-ordinal control.

### MTMM Matrix

| | $W_{OV}$ analysis (years) | $W_{OV}$ analysis (months) | Ablation (years) | Ablation (months) |
|---|---|---|---|---|
| **$W_{OV}$ analysis (years)** | — | high (same heads) | moderate | ? |
| **$W_{OV}$ analysis (months)** | high | — | ? | moderate |
| **Ablation (years)** | moderate | ? | — | high (same heads) |
| **Ablation (months)** | ? | moderate | high | — |

Cross-domain convergence (off-diagonal same-method cells) is high — the same heads identified structurally in one domain appear in another. Cross-method convergence (structural vs. ablation for the same domain) is moderate — structural identification and causal effects point to overlapping head sets. This is an unusually well-filled MTMM for an MI result, though formal correlation values are not reported.

---

## MI Lens — Interpretive Validity

*Is the "general successor primitive" interpretation warranted?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** Structural + algorithmic — names what the heads compute (ordinal succession) and how ($W_{OV}$ encodes ordering).

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Strong.** Structural evidence ($W_{OV}$ analysis) directly supports a structural/algorithmic claim.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** "Some heads are reusable successor-computing primitives" is a clean, falsifiable story that explains cross-domain generalization.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Partial.** Could these heads be doing something more general (attention to "related items") that happens to include succession? The structural evidence constrains this — $W_{OV}$ specifically encodes *ordering*, not general similarity. But whether "successor" is exactly right versus "ordinal proximity" is debatable.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Good.** "General-purpose ordinal mechanism" matches the evidence scope.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Pass | Structural + algorithmic |
| V2 Level-evidence match | Strong | Direct structural support |
| V3 Narrative coherence | Strong | Cross-domain generalization explained |
| V4 Alternative exclusion | Partial | "Successor" vs. "ordinal proximity" |
| V5 Scope honesty | Good | Matches evidence |

### Key Distinctions

- **Description vs explanation:** The "general successor primitive" account is genuinely explanatory — it explains *why* the same heads appear across ordinal domains (shared mechanism) and predicts where they should appear (any ordinal task). This goes beyond mere description of which heads are active.
- **Component identity vs component role:** The role label "successor head" is well-grounded: the structural signature ($W_{OV}$ ordering) independently confirms the functional label (ordinal succession behavior). The label is not just based on behavioral observation but has architectural backing.
- **Faithfulness vs understanding:** The evidence supports both — the identified heads are causally important (faithfulness via ablation) AND the mechanism is understood (ordinal structure in $W_{OV}$). This combination is rare in MI.

### Evidence Convergence Map

- **Implementational → Interpretation:** Strong. $W_{OV}$ weight analysis directly shows ordinal encoding; ablation confirms causal role. Multiple implementational sub-modes converge.
- **Algorithmic → Interpretation:** Strong. "Compute the next item in an ordinal sequence" is a specified algorithm that the structural evidence directly supports. The cross-domain pattern confirms it is a general algorithm, not a task-specific shortcut.
- **Computational → Interpretation:** Moderate. "Ordinal succession" is well-defined as a computational goal. Whether it is exactly "successor" (next item) or "ordinal proximity" (nearby items) is the remaining ambiguity.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ (cross-domain) | — | ∅ | ∅ | ∅ |
| $W_{OV}$ analysis | — | — | ✓ | ✓ (partial) | — |
| Steering | — | — | — | — | — |
| Cross-domain generalization | — | implicit | — | ✓ | ✓ |

The filled cells span both rows and columns more broadly than most MI results. Ablation provides necessity; weight analysis provides representational and partial algorithmic evidence; cross-domain generalization provides algorithmic and computational support. The main gap is sufficiency (no isolation experiment) and steering (no stimulation test).

### Causal Sufficiency Graph

- Ordinal input → successor head attention: **solid** (heads attend to ordinal tokens, confirmed across domains)
- Successor head $W_{OV}$ → boosted next-item logit: **solid** (structural analysis confirms the weight pathway)
- Successor heads → output prediction: **solid** (ablation confirms causal contribution)
- Input encoding → successor head selection: **dashed** (how the model identifies that a token is part of an ordinal sequence is not characterized)

The output pathway (successor head → prediction) is solid and multi-domain confirmed. The input pathway (how tokens are identified as ordinal) is the main uncharacterized link — the heads clearly compute succession, but the upstream mechanism that routes ordinal inputs to them is not described.

---
