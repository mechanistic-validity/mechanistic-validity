---
title: "Verdicts"
description: "A five-tier evidential grading system for mechanistic claims — what each tier requires and how claims move between them."
---

# Verdicts

A verdict is a **composite assessment of evidential status** — it answers the question "how well-established is this mechanistic claim, given all available evidence across all five validity types?" The verdict is not a quality judgment on the paper. It is a characterization of where the claim stands on the path from initial proposal to full validation, with specific gaps named.

Each tier subsumes all requirements of lower tiers.

## Why tiers, not scores

A continuous score (e.g., "this claim has validity 0.73") implies a precision the evidence does not support and obscures qualitative transitions. The difference between a claim with one causal experiment and a claim with five convergent lines of evidence is not well-captured by assigning them 0.4 and 0.8 on a scale — the second has crossed a qualitative threshold (convergence) that changes what the claim means.

The tier system makes these thresholds explicit. Each tier has a *minimum evidence package* — a set of criteria that must be satisfied (not merely partially addressed) for the claim to occupy that tier. Movement between tiers is governed by specific upgrade conditions that name what additional evidence is required.

## Intellectual origins

| Source | Year | Contribution |
|---|---|---|
| [Hill](https://doi.org/10.1177/003591576505800503) | 1965 | Nine criteria for causal inference in epidemiology (strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy). Hill did not intend these as a checklist — he intended them as considerations that, in aggregate, make a causal claim more or less credible. The tier system operationalizes this aggregate judgment. |
| [Grading of Recommendations Assessment (GRADE)](https://doi.org/10.1136/bmj.328.7454.1490) | 2004 | Evidence grading in clinical medicine: High / Moderate / Low / Very Low. Each grade has specific upgrade and downgrade conditions (inconsistency, indirectness, imprecision, publication bias). The framework demonstrated that tiered grading with named transitions is more actionable than continuous scores. |
| [Shadish, Cook & Campbell](https://psycnet.apa.org/record/2001-18082-000) | 2002 | Threats to validity as *reasons for doubt* — each threat, if present and unaddressed, downgrades the evidential status of the claim. The tier system formalizes which threats block which transitions. |
| [Lakatos, *The Methodology of Scientific Research Programmes*](https://doi.org/10.1017/CBO9780511621123) | 1978 | Progressive vs. degenerative research programmes — a programme is progressive if it generates novel predictions that are subsequently confirmed, degenerative if it only accommodates known facts. The Proposed → Triangulated progression tracks this: a claim that only explains the data it was discovered from is at a lower tier than one that generates and confirms novel predictions. |

## The five tiers

### Tier 1: Proposed

**Meaning:** Structural or representational evidence only.

**Minimum evidence requirements:**
- Construct defined under a declared [description mode](/mechanistic-validity/framework/description-modes/) (C1–C2)
- At least one admissible measurement conducted

**Characteristic occupants:** Probing classifiers (capped by I1); sparse autoencoder features (capped by M2).

---

### Tier 2: Causally Suggestive

**Meaning:** Necessity shown, sufficiency not established.

**Minimum evidence requirements (in addition to Tier 1):**
- Necessity via causal intervention ([I1](/mechanistic-validity/framework/criteria/internal/necessity/) confirmed)
- At least one measurement passes baseline separation ([M2](/mechanistic-validity/framework/criteria/measurement/baseline-separation/))

**Characteristic occupants:** Docstring circuit (capped by I4); gender bias circuits (capped by E1, I4); IOI circuit (capped by E1, I4); Othello world model (capped by E1).

---

### Tier 3: Mechanistically Supported

**Meaning:** Necessity + sufficiency with consistent methods.

**Minimum evidence requirements (in addition to Tier 2):**
- Sufficiency established ([I2](/mechanistic-validity/framework/criteria/internal/sufficiency/))
- Intervention reach across ≥2 ablation methods ([E1](/mechanistic-validity/framework/criteria/external/intervention-reach/))
- Specificity test conducted ([I4](/mechanistic-validity/framework/criteria/internal/specificity/) at least partially confirmed)

**Characteristic occupants:** Copy suppression (capped by I6); greater-than circuit (capped by I5, I6); modular addition (capped by I6); refusal direction (capped by I6); successor heads (capped by I6); superposition (capped by I6); global workspace (capped by I6).

Double dissociation (I6) caps every one of the seven claims that reach this tier.

---

### Tier 4: Triangulated

**Meaning:** Multiple converging lines of independent evidence.

**Minimum evidence requirements (in addition to Tier 3):**
- Convergent evidence from ≥3 evidence families ([C3](/mechanistic-validity/framework/criteria/construct/convergent-validity/)), where families count as independent when the failure of one's core assumption would not automatically invalidate the other
- Discriminant validity ([C4](/mechanistic-validity/framework/criteria/construct/discriminant-validity/))
- Rival mechanism exclusion ([I5](/mechanistic-validity/framework/criteria/internal/rival-mechanism-exclusion/))
- Confound control ([I7](/mechanistic-validity/framework/criteria/internal/confound-control/))
- Cross-distribution replication — [E4](/mechanistic-validity/framework/criteria/external/cross-model-recurrence/) where the claim asserts reach beyond the systems tested, [E2](/mechanistic-validity/framework/criteria/external/prompt-generalization/) where it does not
- Double dissociation attempted ([I6](/mechanistic-validity/framework/criteria/internal/double-dissociation/))

**Characteristic occupants:** Induction heads, token copying (capped by C6, I3, I10, I12).

---

### Tier 5: Validated (within scope)

**Meaning:** All five validity types addressed.

**Minimum evidence requirements (in addition to Tier 4):**
- Measurement calibration ([M1](/mechanistic-validity/framework/criteria/measurement/reliability/)–[M6](/mechanistic-validity/framework/criteria/measurement/invariance/)) explicitly addressed
- Interpretive validity ([V1](/mechanistic-validity/framework/criteria/interpretive/level-declaration/)–[V5](/mechanistic-validity/framework/criteria/interpretive/scope-declaration/)) audited
- Nomological and complementation validity ([C5](/mechanistic-validity/framework/criteria/construct/nomological-validity/)–[C6](/mechanistic-validity/framework/criteria/construct/complementation-validity/))
- Minimality, rescue reversibility and onset–offset coupling (I3, I10–I12)
- External validity across models, prompts, tasks, dose and novel prediction ([E2](/mechanistic-validity/framework/criteria/external/prompt-generalization/)–[E6](/mechanistic-validity/framework/criteria/external/novel-prediction/))

**Characteristic occupants:** No claim in the sixteen audited reaches Validated.

---

## When criteria cannot be tested

Some criteria cannot be tested for some claims, and the tier system treats that as information. Specificity (I4) asks whether a mechanism does this task and not everything, which a model trained on a single task cannot answer: there is no second task to fail on. Onset–offset coupling (I11) asks whether a mechanism appears when the capability appears, which no claim about a released model without training checkpoints can answer. In each case the claim caps at the tier below, and the cap is a statement about the setting. A result on a single-task toy model licenses less than one on a model doing many things, and a field that cannot observe a system's history establishes less than one that can.

## Diagnostic labels

Three labels sit outside the tier progression. They replace the tier rather than occupying a position within it — a claim carrying one of them does not also hold a position in the hierarchy.

### Underdetermined

**What it means:** The evidence is consistent with multiple mechanisms. The claim cannot resolve between rival specifications.

**When to assign:** When two or more mechanistic accounts have comparable evidential support and no available experiment distinguishes them. The informative response is to name the competing accounts and identify what experiment would distinguish them.

### Insufficient

**What it means:** The construct is not defined well enough, or no admissible measurement exists, to score the claim. Unlike Proposed (which has a defined construct and at least one measurement), Insufficient means the claim cannot enter the evaluation pipeline in its current form. Following the US Preventive Services Task Force's *I statement*, Insufficient marks a claim whose construct is not defined well enough to score rather than one that scores badly.

### Disconfirmed

**What it means:** The claim fails decisively on a key criterion. A negative result on a required criterion.

**Types of disconfirmation:**
- **Prediction failure**: The mechanism predicts behavior $X$ and the model produces $\neg X$ in the relevant conditions.
- **Artifact demonstration**: The claimed mechanism is shown to be an artifact of the measurement procedure.

**Disconfirmation is not failure.** A disconfirmed claim is informative — it narrows the space of possible mechanisms. A field that never disconfirms is not doing science. The lateral position of Disconfirmed (rather than placing it below Proposed) reflects this: disconfirmation is a *different kind of conclusion*, not a worse one.

**Characteristic occupants:** Induction heads, general in-context learning (fails I1); knowledge neurons (fails I4).

## Verdicts on sixteen claims

| Verdict | Claim (criteria that cap it) |
|---|---|
| **Triangulated** | Induction heads, token copying (C6, I3, I10, I12) |
| **Mechanistically Supported** | Copy suppression (I6); greater-than circuit (I5, I6); modular addition (I6); refusal direction (I6); successor heads (I6); superposition (I6); global workspace (I6) |
| **Causally Suggestive** | Docstring circuit (I4); gender bias circuits (E1, I4); IOI circuit (E1, I4); Othello world model (E1) |
| **Proposed** | Probing classifiers (I1); sparse autoencoder features (M2) |
| **Disconfirmed** | Induction heads, general in-context learning (I1); knowledge neurons (I4) |

Two patterns run across the set rather than within any claim. Double dissociation caps every one of the seven claims that reach Mechanistically Supported, so the tier above is held by a single criterion the field rarely attempts. And confounding sensitivity is Untested in all sixteen: no claim in the set bounds how strong an unmeasured confounder would have to be to explain its result.

## Upgrade mechanics

### Upgrade conditions are conjunctive

To move from Tier $n$ to Tier $n+1$, *all* conditions for Tier $n+1$ must be satisfied simultaneously. Partial satisfaction of Tier $n+1$ conditions does not produce a position "between" tiers — the claim remains at Tier $n$ with the partial evidence noted.

### Downgrades are possible

New evidence can move a claim to a lower tier. Miller et al. (2024) effectively downgraded the IOI circuit's internal validity by demonstrating method-conditional results — what was presented as a stable finding is actually conditional on the ablation choice. Downgrades are not punitive — they are the system updating on new evidence.

### The weakest-link principle

The tier is determined by the *weakest gating criterion*, not by an average over criteria or over validity types. The gates are conjunctive: every criterion a tier names must be met, so one unmet gate holds the claim at the tier below regardless of how strong the rest is. A claim with excellent internal, external and interpretive evidence but no baseline separation (M2) is bounded at Proposed, because M2 gates the first upgrade. This is what prevents impressive evidence in one dimension from masking a fundamental problem in another.
