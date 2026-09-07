---
title: "Criteria"
description: "All 36 criteria, grouped by validity type, with pass conditions."
---

# Criteria

Each criterion is a specific, falsifiable condition that must be met for a validity type to be satisfied. The 36 criteria are grouped into five validity types following a dependency chain: Construct → Measurement → Internal → External → Interpretive. A failure early in the chain limits what later evidence can establish.

Each criterion receives one of six statuses: **Confirmed**, **Partially confirmed**, **Inconclusive**, **Disconfirmed**, **Untested**, or **Not applicable**.

## Construct validity (C1–C6) — Is the target concept well-defined?

| # | Criterion | One-line description | Page |
|---|---|---|---|
| C1 | Falsifiability | Can the claim be refuted? | [falsifiability](construct/falsifiability/) |
| C2 | Structural plausibility | Is the mechanism physically possible in the architecture? | [structural-plausibility](construct/structural-plausibility/) |
| C3 | Convergent validity | Do multiple independent methods agree? | [convergent-validity](construct/convergent-validity/) |
| C4 | Discriminant validity | Does the measure distinguish this from neighboring constructs? | [discriminant-validity](construct/discriminant-validity/) |
| C5 | Nomological validity | Does the claim fit into a broader theory? | [nomological-validity](construct/nomological-validity/) |
| C6 | Complementation validity | Are the construct's labeled subdivisions functionally distinct? | [complementation-validity](construct/complementation-validity/) |

## Measurement validity (M1–M7) — Are the instruments trustworthy?

| # | Criterion | One-line description | Page |
|---|---|---|---|
| M1 | Reliability | Do repeated measurements give the same answer? | [reliability](measurement/reliability/) |
| M2 | Baseline separation | Is the score distinguishable from random/untrained baselines? | [baseline-separation](measurement/baseline-separation/) |
| M3 | Stability | Is the classification robust to perturbation? | [stability](measurement/stability/) |
| M4 | Calibration | Are the numbers meaningful? | [calibration](measurement/calibration/) |
| M5 | Sensitivity | Can the instrument detect known-true effects? | [sensitivity](measurement/sensitivity/) |
| M6 | Invariance | Does the metric behave consistently across conditions? | [invariance](measurement/invariance/) |
| M7 | Selection correction | When k findings are selected from N candidates, is N reported and multiplicity controlled? | [selection-correction](measurement/selection-correction/) |

## Internal validity (I1–I12) — Does the evidence support the causal claim?

The twelve internal criteria fall into four blocks:

- **I1–I3**: Properties of the set as a whole (necessity, sufficiency, minimality)
- **I4–I6**: Discrimination across tasks, rival circuits, and both (specificity, rival exclusion, [double dissociation](/mechanistic-validity/glossary/#double-dissociation))
- **I7–I8**: Measured and unmeasured confounders
- **I9–I10**: Internal structure probes (epistatic interaction, rescue reversibility)
- **I11–I12**: Developmental coupling (onset, offset)

| # | Criterion | One-line description | Page |
|---|---|---|---|
| I1 | Necessity | Is the circuit required for the behavior? | [necessity](internal/necessity/) |
| I2 | Sufficiency | Is the circuit enough to produce the behavior? | [sufficiency](internal/sufficiency/) |
| I3 | Minimality | Does every component earn its place? | [minimality](internal/minimality/) |
| I4 | Specificity | Does intervening on the circuit affect this task more than matched control tasks? | [specificity](internal/specificity/) |
| I5 | Rival mechanism exclusion | Is this *the* mechanism, or *a* mechanism? | [rival-mechanism-exclusion](internal/rival-mechanism-exclusion/) |
| I6 | Double dissociation | Do two interventions cross, each breaking what the other spares? | [double-dissociation](internal/double-dissociation/) |
| I7 | Confound control | Are alternative explanations ruled out? | [confound-control](internal/confound-control/) |
| I8 | Confounding sensitivity | How strong must an unmeasured confounder be to explain the result? | [confounding-sensitivity](internal/confounding-sensitivity/) |
| I9 | Epistatic interaction | Do circuit components interact non-additively, and does the direction of interaction distinguish shared pathways from mutual compensation? | [epistatic-interaction](internal/epistatic-interaction/) |
| I10 | Rescue reversibility | Does restoring a corrupted component recover behavior? | [rescue-reversibility](internal/rescue-reversibility/) |
| I11 | Onset coupling | Does the mechanism appear when the capability appears? | [onset-coupling](internal/onset-coupling/) |
| I12 | Offset coupling | Does the mechanism go when the capability is removed? | [offset-coupling](internal/offset-coupling/) |

I6 (double dissociation) caps every claim that reaches Mechanistically Supported in the sixteen audited case studies. I8 (confounding sensitivity) is untested in all sixteen. No claim reaches Validated.

## External validity (E1–E6) — Does the mechanism generalize?

| # | Criterion | One-line description | Page |
|---|---|---|---|
| E1 | Intervention reach | Has the result been reproduced under at least two intervention families, and do they agree? | [intervention-reach](external/intervention-reach/) |
| E2 | Prompt generalization | Does it work on diverse prompts? | [prompt-generalization](external/prompt-generalization/) |
| E3 | Cross-task generalization | Does the mechanism transfer to related tasks? | [cross-task-generalization](external/cross-task-generalization/) |
| E4 | Cross-model generalization | Does the mechanism appear in other models? | [cross-model-recurrence](external/cross-model-recurrence/) |
| E5 | Graded response | Does partial [ablation](/mechanistic-validity/glossary/#ablation) produce partial effects? | [graded-response](external/graded-response/) |
| E6 | Novel prediction | Does the mechanism predict new, untested behaviors? | [novel-prediction](external/novel-prediction/) |

## Interpretive validity (V1–V5) — Is the interpretation correct?

| # | Criterion | One-line description | Page |
|---|---|---|---|
| V1 | Level declaration | At what description mode is the claim made? | [level-declaration](interpretive/level-declaration/) |
| V2 | Level-evidence match | Does the evidence support claims at that level? | [level-evidence-match](interpretive/level-evidence-match/) |
| V3 | Alternative level | Could the evidence be explained at a different level? | [alternative-level](interpretive/alternative-level/) |
| V4 | Unlicensed labeling | Does a name import a property that was not measured? | [unlicensed-labeling](interpretive/unlicensed-labeling/) |
| V5 | Scope declaration | What does the claim explicitly not cover? | [scope-declaration](interpretive/scope-declaration/) |

V3, V4, and V5 have no counterpart in the validity frameworks surveyed from other fields. They address failure modes specific to mechanistic interpretability: claiming an algorithm when only an implementation was shown (V3), calling a representation a "world model" when only a state summary was demonstrated (V4), and silently generalizing beyond the tested system (V5).
