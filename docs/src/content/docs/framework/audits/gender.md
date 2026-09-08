---
title: "Gender Bias Circuits"
description: "Criterion audit of the gender bias circuits claim, as submitted."
---

# Gender Bias Circuits

**Source.** *Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias* ([Vig et al., 2020](https://arxiv.org/abs/2004.12265)).

**Description.** Causal mediation analysis treats a gender edit to the input as a treatment and each neuron, layer, head or attention weight as a mediator, splitting the effect on a two-pronoun probability ratio into a natural direct and a natural indirect effect. Three properties of the effect distribution are reported — sparsity, with ten of 144 heads reproducing the effect of intervening on all of them; synergy among neurons; and decomposition into direct and indirect parts — against a randomly initialized GPT2-small carried through as a matched negative control. Evaluated on: five GPT-2 sizes on 17 templates over 169 professions plus WinoBias Type 1 and Winogender, with five further model families added in §5.5. Description mode: implementational-topographic.

## Readings of the claim

**Gender bias circuits.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** Ten of 144 attention heads carry the indirect effect of a gender edit on the pronoun probability ratio, and neurons combine synergistically to produce it | Causally Suggestive | Specificity, which has no second outcome to register on (I4); necessity, which the design's substitution estimands cannot express (I1); correction for a greedy maximum over 144 candidates (M7); and dispersion on the head-level and neuron-level effects that carry the sparsity claim (M1) |
| These components mediate gender bias rather than gender processing | Underdetermined | A localized gender-competence mechanism to compute an overlap against; none is measured, and the definitionally gendered professions that would carry the test are excluded from the total-effect calculation (C4, V3, I5) |
| Training is what installs the sparse mediation structure | Underdetermined | A trajectory. The coupling is a single endpoint contrast against one untrained model whose total effects are small rather than absent — 0.07 on WinoBias against GPT2-small's 0.25 (I12) — with no intermediate checkpoint anywhere (I11) |
| The effect grows with model size and the pattern holds across model families | Disconfirmed | Nothing — tested and failed: the neuron-level pattern does not transfer to the masked language models and the authors state they have no theory for the difference (E4, M6), and the size trend holds on the Winograd sets but not on Professions, where model size is not well correlated with total effect (E4) |

## Verdict

**Gender bias circuits.** Verdict: **Causally Suggestive**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked. **E1** intervention reach — One instrument throughout, so every agreement is the method with itself; **I4** specificity — A two-element outcome set leaves off-target damage nowhere to register |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Mechanistically Supported. **C4** discriminant validity — No second construct is ever localized, so nothing can separate from it; **I6** double dissociation — One behavior by construction, so the second dissociation cannot be run |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Mechanistically Supported. **E3** cross-task generalization — One two-candidate setup throughout; generality claimed, not shown; **I10** rescue reversibility — Counterfactual activations exist for every example; no recovery is read; **I11** onset coupling — Two states, before training and after; no sequence to locate an onset in |

## 36-criterion audit

**Gender bias circuits.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | The answer could have been 100 heads and was 10; one negative case fires |
| C2 | Structural plausibility | PC | Mediators taken from the architecture; no weight-level mechanism anywhere |
| C3 | Convergent validity | C | Two mediator families, three datasets, four scales – all one instrument |
| C4 | Discriminant validity | U | No second construct is ever localized, so nothing can separate from it |
| C5 | Nomological validity | C | Pearl's estimands under his definitions, with conditions stated (Pearl, 2001) |
| C6 | Complementation validity | N/A | Mediators carry an index and an estimand, not a labeled role |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Dispersion reported where it is incidental and absent where it is load-bearing |
| M2 | Baseline separation | C | An untrained model matched on everything but training, run at both levels |
| M3 | Stability | PC | Selection rule and outcome scale both varied; the target word is not |
| M4 | Calibration | PC | A fixed point at 1 and a sign convention; the effect scale has no referent |
| M5 | Sensitivity | PC | Effects track published occupational statistics, but at the wrong level |
| M6 | Invariance | C | Six axes tested, and where the invariance stops is reported too |
| M7 | Selection correction | U | Pool, rule and shortcut all disclosed; nothing corrected for multiplicity |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | N/A | Neither estimand can be put in the form necessity requires |
| I2 | Sufficiency | PC | Ten heads of 144 reproduce the all-mediator effect; no capability measured |
| I3 | Minimality | PC | Greedy admission by marginal gain; no ablation of the selected ten |
| I4 | Specificity | U | A two-element outcome set leaves off-target damage nowhere to register |
| I5 | Rival mechanism exclusion | PC | The untrained control excludes architecture; the stimulus rival stands |
| I6 | Double dissociation | U | One behavior by construction, so the second dissociation cannot be run |
| I7 | Confound control | PC | Definitional gender dropped by crowdsourced rating; frequency untouched |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | C | Concurrent against summed intervention, run at both levels, and it splits |
| I10 | Rescue reversibility | U | Counterfactual activations exist for every example; no recovery is read |
| I11 | Onset coupling | U | Two states, before training and after; no sequence to locate an onset in |
| I12 | Offset coupling | PC | Behavior and structure both absent untrained; nothing in between |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | U | One instrument throughout, so every agreement is the method with itself |
| E2 | Prompt generalization | C | Three prompt sets built differently, with the differences reported |
| E3 | Cross-task generalization | U | One two-candidate setup throughout; generality claimed, not shown |
| E4 | Cross-model recurrence | PC | Transfers within the autoregressive class and fails outside it |
| E5 | Graded response | PC | Dose is the number of mediators, not the strength of the intervention |
| E6 | Novel prediction | PC | One prediction confirmed outside the model, one sharper one falsified |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | Structural-behavioral declared twice, and no claim sits outside it |
| V2 | Level-evidence match | C | Every headline claim is a property of the measured effect distribution |
| V3 | Alternative level | PC | The architecture rival is defeated; the decisive one is never raised |
| V4 | Unlicensed labeling | C | Components carry no names at all, only an index and an estimand |
| V5 | Scope declaration | C | Declared where each restriction binds, not collected at the end |
| **Total:** 11 Confirmed, 14 Partially confirmed, 9 Untested, 2 Not applicable |   |   |   |
| **Verdict: Causally Suggestive** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on.

| Criterion | In tension | What would settle it |
|---|---|---|
| I6 | Untested or Not applicable | Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Gender Bias Circuits — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-gender-bias/).
