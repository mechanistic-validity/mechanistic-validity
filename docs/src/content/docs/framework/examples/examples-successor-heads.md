---
title: "Case Study: Successor Heads"
description: "The general-purpose successor mechanism (Hanna et al. 2023, Gould et al. 2023) evaluated through the five core lenses."
---

# Case Study: Successor Heads

Building on the Greater-Than analysis, subsequent work ([Gould et al. 2023](https://arxiv.org/abs/2312.09230)) identifies **successor heads** as a general-purpose mechanism — attention heads whose $W_{OV}$ matrices encode *ordinal succession* across multiple domains: days of the week (Monday → Tuesday), months (January → February), numbers (1 → 2), and alphabetical sequences (A → B). The claim is that these heads do not merely implement year comparison but encode a general ordinal-successor function that the model reuses across sequence types.

This extends the Greater-Than claim from a task-specific circuit to a *general computational primitive* — a reusable building block. The construct validity question shifts accordingly: is "successor head" a natural kind (a real computational unit) or a family resemblance (a label applied to heads that happen to do ordinal things)?

## Composite Verdict

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I6 (double dissociation).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C2 Structural plausibility | C3 Convergent | Strong |
| Internal | I1 Necessity | I7 Confound control | Causally suggestive |
| External | E1 Intervention reach, E3 Cross-task generalization, E6 Novel prediction (all confirmed) | E2 Prompt generalization, E4 Cross-model generalization, E5 Graded response (all partial) | Strong |
| Measurement | M6/M2 Invariance + Separation | M1 Reliability | Strong |
| Interpretive | V2 Level-evidence match | V3 Alternative level | Strong |

**Overall verdict: Mechanistically Supported.** Successor heads benefit from the same structural clarity as the Greater-Than circuit, with the additional strength of cross-domain generalization. The capping criterion is I6 (double dissociation): no crossed design has been attempted. The multi-domain pattern makes the "general computational primitive" claim more convincing than a single-task circuit claim. The case for successor heads as a natural kind is stronger than for most circuits because the same structural signature appears across unrelated domains. Convergent validity (C3) is confirmed independently: a linear probe at 0.708 cosine and single-neuron ablation recover the mod-10 features, sharing nothing with the sparse autoencoder.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| $W_{OV}$ decomposition (multi-domain ordinal structure) | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |
| Ablation | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Cross-task generalization (years, months, days, letters) | [D06 Cross-Task Transfer](/mechanistic-validity/framework/metrics/#d06) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "successor head" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The claim predicts that successor heads should encode ordinal structure across *multiple* domains in their $W_{OV}$ matrices. A head that encodes year ordering but not month ordering is a year-specific head, not a general successor head. This is a discriminating prediction.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Pass.** $W_{OV}$ matrices are inspected and shown to encode ordinal structure across domains. The same heads that boost "32 → 33" also boost "Monday → Tuesday" and "B → C." The structural evidence spans domains.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** Separation from the null case is shown twice and quantified: one model has no head above the cutoff, and in the case-study model the top head's score is eight times the runner-up. Separation from a neighboring construct is never attempted — the successor score is not computed alongside an induction, copying or duplicate-token score on the same heads, so nothing rules out a head scoring highly on both.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** The mechanism is minimal by construction — a product of four weight matrices — and the one droppable component contributes under a percent.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Confirmed.** Three instruments recover the mod-10 features independently: sparse autoencoders, a linear probe at 0.708 cosine, and single-neuron ablation, sharing nothing with each other.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Pass | Cross-domain structural predictions |
| C2 Structural plausibility | Pass | Multi-domain $W_{OV}$ ordering |
| C4 Discriminant validity | N/A (general) | Multi-domain by design |
| I3 Minimality | Pass | Small selective set |
| C3 Convergent validity | Partial | Structural + behavioral |

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

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Partial.** The authors name the gap themselves and then close part of it. On 64 numbered-listing prompts collected from real text, the successor head is the winning head every time, which is a strong result about ranking. It is not a necessity measurement: winning means the head whose ablation most reduces the correct-token logit, so the experiment never reports how far performance falls when the head is removed, and 64 out of 64 could hold with a small absolute effect.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Confirmed.** Sufficiency is demonstrated at the weight level, which is stronger than the activation-level version the criterion usually gets. The effective OV circuit alone — four matrices, no attention pattern, no downstream computation, no prompt — ranks the successor above every competitor in its task class for more than half the dataset, and for the case-study head across most individual tasks.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Partial, with an adverse result.** Off-target function is measured across five ablation regimes and found present. On natural text, acronym and greater-than behavior take 23.8% and 18.9% of the head's winning cases, and the authors reframe the head as interpretably polysemantic. The stronger reading — that the head is specific to succession — is disconfirmed.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Two confounds are controlled by construction. Tokenization is handled by carrying spaced and capitalized variants of every token through the whole pipeline and by taking the MLP0 output rather than the raw embedding as the object of study. Leakage is handled by holding out whole tasks for the probe and reserving Roman numerals, which are absent from the succession dataset. Frequency, position and prompt length are untouched.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Pass | Cross-domain ablation effects |
| I2 Sufficiency | Partial | Structural implication, not isolation |
| I4 Specificity | Partial | Ordinal vs. non-ordinal contrast |
| M1 Reliability | Partial | Cross-domain strong; cross-model limited |
| I7 Confound control | Not tested | Single method |

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

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Confirmed.** Five intervention forms spanning removal and addition are run, and they agree.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Implicit.** The $W_{OV}$ structure implies graded effects — items further from the reference should receive proportionally stronger boosts. Not directly measured as a dose-response.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Partial.** Two distributions of very different kinds are used and the harder one is genuinely hard. The succession dataset is a constructed token list, which makes the weight test clean and says nothing about text. The natural-language evaluation samples 128 long contexts at random from the model's own training distribution and scores every prefix. The supporting experiments are much narrower: two hand-written templates for the Tuned Lens control, 64 hand-collected prompts for the numbered-listing study.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Partial.** Behavior recurs across GPT-2, Pythia and Llama-2 from 31M to 12B parameters. The mechanism-level account is single-model: one case study on Pythia-1.4B L12H0 plus two appendix replications.

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Not tested | — |
| I4 Specificity | Partial | Selective for ordinal class |
| E5 Graded response | Moderate | Contributing, not sole mechanism |
| E2 Prompt generalization | Strong | Multi-domain generalization |
| E4 Cross-model generalization | Not tested | — |

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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Partial.** Modes are taken over more than a hundred autoencoders, but the central successor score carries no interval.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Pass (within model).** The same heads show successor structure across domains — strong within-model invariance.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** Two negative baselines, one of them sharp. The Tuned Lens control cuts the head out of the path and asks what the embedding plus MLP0 can do alone; the answer is under one percent, and what comes back instead are bigram continuations, which localizes the work to the head rather than to the representation it reads. A chance level is stated, but for the decoding experiments rather than for the successor score itself, whose null depends on the size of each task class.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Untested.** Every control run is a known-negative; no planted circuit of known extent is recovered, so the instrument's detection of a known-true effect is never demonstrated.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The primary measure is calibrated by construction: the successor score is a proportion of a stated token set with a stated threshold, and ΔL is a loss change on real text. The importance measure carried into the natural-data study is not — a winning case records only which head's ablation hurts most, so a head can win every prompt while accounting for an arbitrarily small share of performance. This is why the 64-out-of-64 result establishes rank and not magnitude.

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Not reported | — |
| M6 Invariance | Pass | Cross-domain consistency |
| M2 Baseline separation | Pass | Clear successor/non-successor distinction |
| M5 Sensitivity | Good | Multi-domain criterion is discriminating |
| M4 Calibration | Not reported | — |
| C3 Convergent validity | Good | Structural + behavioral |

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** Structural + algorithmic — names what the heads compute (ordinal succession) and how ($W_{OV}$ encodes ordering).

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** Three claims sit at three levels and the evidence matches unevenly. The mechanism claim is about weights and is tested on weights, the cleanest match in the paper. The feature-level claim is supported by three methods and is stated by the authors to be incomplete, since the features fail to reproduce a property the weights have. The universality claim is loosest: made about mechanism and supported mostly by behavior.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** Could these heads be doing something more general (attention to "related items") that happens to include succession? The structural evidence constrains this — $W_{OV}$ specifically encodes *ordering*, not general similarity. But whether "successor" is exactly right versus "ordinal proximity" is debatable.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Partial.** Three limits are declared in the main text, each next to the result it bounds: the feature account does not reproduce the greater-than bias, the features do not steer two of the eight tasks, and no general necessity demonstration was run. What is not declared is the distance between the abstract's claim about architectures and sizes and the single model the mechanism was worked out in.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Pass | Structural + algorithmic |
| V2 Level-evidence match | Strong | Direct structural support |
| V3 Alternative level | Partial | "Successor" vs. "ordinal proximity" |
| V5 Scope declaration | Good | Matches evidence |

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
