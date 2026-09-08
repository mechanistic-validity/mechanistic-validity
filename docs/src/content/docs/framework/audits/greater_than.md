---
title: "Greater-Than Circuit"
description: "Criterion audit of the greater-than circuit claim, as submitted."
---

# Greater-Than Circuit

**Source.** *How Does GPT-2 Compute Greater-Than?: Interpreting Mathematical Abilities in a Pre-Trained Language Model* ([Hanna et al., 2023](https://arxiv.org/abs/2305.00586)).

**Description.** Iterative path patching over GPT-2 small recovers a circuit for prompts that supply a two-digit start year and ask for an end year: attention heads carry the start year to the final position, and MLPs 8–11 raise the probability of every year above it. Feeding the circuit real input while the rest of the model receives 01-input recovers 72.7% of an 81.7% probability-difference baseline, measured over 10,000 examples spanning 120 nouns and 768 years, and the inverse assignment drives the same quantity to -36.6%. The circuit also fires on prompts requiring less-than, where the model answers greater, which the authors report as a property of the mechanism rather than a defect in the account. Description mode: implementational, with the algorithmic question in the title left open.

## Readings of the claim

**Greater-than circuit.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A subgraph of attention heads and MLPs 8–11 performs greater-than on year-span prompts in GPT-2 small | Mechanistically Supported | A second instrument for circuit discovery (C3), a dissociation from the components serving the tasks the circuit fails (I6), and a second model (E4) |
| The circuit computes the comparison rather than retrieving memorized year associations | Underdetermined | The paper's own structured-number-representation account fails two causal tests and memorization is never tested (I5); no experiment separates a lookup table from a computation (V3) |
| The circuit implements greater-than as such, firing where the relation applies and not elsewhere | Disconfirmed | Nothing — tested and failed: two prompts requiring less-than recruit the same circuit and the model answers greater, and a non-monotone sequence case does the same (I4, E3) |
| The same circuit is the mechanism for greater-than beyond GPT-2 small | Disconfirmed | Nothing — tested and failed: the disconfirming evidence is post-origin and in three other model families (E4), and the origin itself runs GPT-2 small only and names the gap |

## Verdict

**Greater-than circuit.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **E4** cross-model recurrence — One model at origin; run later, and the circuit differs per model ([Xu, 2026](https://arxiv.org/abs/2606.05378)); **I5** rival mechanism exclusion — Two rivals tested and both left live, the lookup table included; **I6** double dissociation — The makings of one half; the converse is never run |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **C6** complementation validity — Heads and MLPs divided by role, with no joint ablation to test it; **I10** rescue reversibility — Direction fixed by path patching; the restore is never framed; **M1** reliability — Dispersion reported twice, both pre-circuit; no interval on any result; **M3** stability — Membership read off heatmaps; no threshold stated, none swept; **M5** sensitivity — Two experiments, both on the real model; no planted mechanism |

## 36-criterion audit

**Greater-than circuit.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Probability difference moves +81.7% to -36.6%, past indifference |
| C2 | Structural plausibility | PC | Flow located at three granularities; the derivation fails two causal tests |
| C3 | Convergent validity | PC | Logit lens and path patching agree; single-method where they do not overlap |
| C4 | Discriminant validity | PC | Three tasks recruit different components; every contrast is one kind |
| C5 | Nomological validity | PC | Placed between memorization and generalization, not derived from either |
| C6 | Complementation validity | U | Heads and MLPs divided by role, with no joint ablation to test it |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | U | Dispersion reported twice, both pre-circuit; no interval on any result |
| M2 | Baseline separation | PC | A size-matched comparison run and reported only in prose |
| M3 | Stability | U | Membership read off heatmaps; no threshold stated, none swept |
| M4 | Calibration | PC | Outcome metrics calibrated; the faithfulness quantity is not |
| M5 | Sensitivity | U | Two experiments, both on the real model; no planted mechanism |
| M6 | Invariance | PC | Per-condition variation visible; one tokenizer, one model, one frame |
| M7 | Selection correction | U | Every candidate screened and disclosed; no multiplicity control |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | C | A sign reversal rather than degradation: +81.7% to -36.6% |
| I2 | Sufficiency | C | Complement recovers 72.7% under conditions chosen to favor failure |
| I3 | Minimality | PC | Most of MLP 10 ablates freely; one result runs the other way |
| I4 | Specificity | PC | Fires on less-than prompts its label does not cover |
| I5 | Rival mechanism exclusion | I | Two rivals tested and both left live, the lookup table included |
| I6 | Double dissociation | U | The makings of one half; the converse is never run |
| I7 | Confound control | PC | Tokenization confound designed out; frequency handled by restriction |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | PC | Non-additivity is the neuron-level finding; no interaction statistic |
| I10 | Rescue reversibility | U | Direction fixed by path patching; the restore is never framed |
| I11 | Onset coupling | N/A | No published checkpoints, so onset cannot be timed at all |
| I12 | Offset coupling | N/A | One checkpoint observed once; no interval across which to lapse |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Three intervention families, but one counterfactual behind them all |
| E2 | Prompt generalization | PC | Broad inside one frame, and the frame is held fixed by design |
| E3 | Cross-task generalization | PC | Transfers across the year frame at 67.8–98.8% and fails outside it |
| E4 | Cross-model recurrence | D | One model at origin; run later, and the circuit differs per model ([Xu, 2026](https://arxiv.org/abs/2606.05378)) |
| E5 | Graded response | PC | Circuit extent is graded; intervention strength has no scale to sweep |
| E6 | Novel prediction | PC | Prediction confirmed at 98.8% and 88.9%; one corollary falsified |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | The declaration is implementational; the title asks an algorithmic question |
| V2 | Level-evidence match | PC | Matched on flow and topography, not on the claim the title advertises |
| V3 | Alternative level | PC | The lookup-table reading is entertained and bounded, never excluded |
| V4 | Unlicensed labeling | C | Every component label redescribes a measurement made first |
| V5 | Scope declaration | C | Scope declared before the evidence and again after it |
| **Total:** 5 Confirmed, 19 Partially confirmed, 1 Inconclusive, 8 Untested, 1 Disconfirmed, 2 Not applicable |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on.

| Criterion | In tension | What would settle it |
|---|---|---|
| E4 | Disconfirmed or Untested | Whether the later cross-model test was run and failed, or reports a different circuit |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Greater-Than Circuit — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-greater-than/).
