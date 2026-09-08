---
title: "Knowledge Neurons"
description: "Criterion audit of the knowledge neurons claim, as submitted."
---

# Knowledge Neurons

**Source.** *Knowledge Neurons in Pretrained Transformers* ([Dai et al., 2022](https://arxiv.org/abs/2104.08696)).

**Description.** Integrated gradients over the intermediate activations of a feed-forward layer attribute a relational fact to about four neurons, which the paper reads as the value slots of the key–value memory that layer implements. Zeroing those activations lowers the correct-answer probability by 29.03% and doubling them raises it by 31.17%, where a count-matched attribution baseline moves the same quantity by $-1.47%$ and $-1.27%$; rewriting the value slots installs a substituted entity as the top prediction 34.4% of the time against the baseline's 0.0%. Evaluated on: BERT-base-cased at one released checkpoint, on 253,448 ParaRel cloze prompts covering 27,738 facts across 34 relations, with no training run anywhere in the paper. Description mode: implementational-functional.

## Readings of the claim

**Knowledge neurons.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** These roughly four feed-forward neurons store the relational fact | Disconfirmed | Nothing — tested and failed: all three of the paper's own summaries report a correlation between activation and expression while the title and abstract claim storage (V2), and the same editing machinery moves non-factual linguistic patterns, so the construct never separates from its neighbor (C4) |
| Manipulating these neurons changes how strongly the model expresses the fact | Causally Suggestive | Specificity, tested and inconclusive (I4); sufficiency reaches a 34.4% edit success rate on a fact the rest of the model still expresses (I2) |
| Editing these neurons edits that fact and leaves unrelated knowledge alone | Disconfirmed | Nothing — tested and failed: Table 6 gives an inter-relation perplexity rise of 7.2 for the identified neurons against 4.3 for random ones, and §5.1 reads the same table as little negative influence on other knowledge (I4) |
| The account holds beyond BERT-base-cased | Insufficient | One model, and the generalization is asserted with no experiment behind it (E4, V5); what has since transferred is the attribution procedure rather than the storage reading |

## Verdict

**Knowledge neurons.** Verdict: **Disconfirmed**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Blocked. **I4** specificity — Holds at identification and erasure, fails on other knowledge: results point both ways |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked at Mechanistically Supported. **C4** discriminant validity — Competitor named and filtered out; the same machinery moves it ([Niu et al., 2024](https://arxiv.org/abs/2405.02421)); **I5** rival mechanism exclusion — The one alternative ruled out is about method; the FFN framing forecloses the rest; **I6** double dissociation — One mechanism localized and one behavior measured, so neither arm exists |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Mechanistically Supported. **C6** complementation validity — Relation sets separate at identification; no joint suppression is run; **E3** cross-task generalization — Single-word cloze throughout, named first among the authors' limitations; **I3** minimality — Set size is an input, held near four neurons before any effect is measured; **I10** rescue reversibility — The damage is analytic and invertible, and the restore is never run; **M1** reliability — Every quantity a mean reported once, with no interval or repeated run; **M3** stability — Three free settings stated and none of them perturbed; **M5** sensitivity — One known-negative control and no case where the answer is known in advance |

## 36-criterion audit

**Knowledge neurons.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Signed predictions run both ways on 34 relations, against a matched control |
| C2 | Structural plausibility | C | Eq. (3) beside Eq. (2): the same key-value operation under another nonlinearity |
| C3 | Convergent validity | PC | One attributor against one contrast, agreeing on one coarse property |
| C4 | Discriminant validity | D | Competitor named and filtered out; the same machinery moves it ([Niu et al., 2024](https://arxiv.org/abs/2405.02421)) |
| C5 | Nomological validity | PC | Two theories say where to look; neither is tested as a network |
| C6 | Complementation validity | U | Relation sets separate at identification; no joint suppression is run |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | U | Every quantity a mean reported once, with no interval or repeated run |
| M2 | Baseline separation | C | Activation magnitude through the identical pipeline, succeeding 0.0% of the time |
| M3 | Stability | U | Three free settings stated and none of them perturbed |
| M4 | Calibration | PC | Outcomes calibrated; the selection quantity is driven to a target range |
| M5 | Sensitivity | U | One known-negative control and no case where the answer is known in advance |
| M6 | Invariance | PC | Per-relation spread across 34 relations; the generalization claim has no experiment |
| M7 | Selection correction | U | Selection at four places, disclosed at all four and corrected at none |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | PC | 29.03% against the control's 1.47%, consistent across relations but a decrement |
| I2 | Sufficiency | PC | Doubling gives +31.17% and rewriting succeeds 34.4%; neither reaches sufficiency |
| I3 | Minimality | U | Set size is an input, held near four neurons before any effect is measured |
| I4 | Specificity | I | Holds at identification and erasure, fails on other knowledge: results point both ways |
| I5 | Rival mechanism exclusion | U | The one alternative ruled out is about method; the FFN framing forecloses the rest |
| I6 | Double dissociation | U | One mechanism localized and one behavior measured, so neither arm exists |
| I7 | Confound control | PC | Wording controlled by requiring recurrence across nine templates; frequency is not |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder |
| I9 | Epistatic interaction | U | Every manipulation hits the whole set at once, so no contribution separates |
| I10 | Rescue reversibility | U | The damage is analytic and invertible, and the restore is never run |
| I11 | Onset coupling | N/A | One released checkpoint, so there is no sequence to locate an onset in |
| I12 | Offset coupling | N/A | One checkpoint supplies no interval over which either could go |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Three forms agreeing in direction, all downstream of one attributor |
| E2 | Prompt generalization | C | Survival across roughly nine templates is built into the identification |
| E3 | Cross-task generalization | U | Single-word cloze throughout, named first among the authors' limitations |
| E4 | Cross-model recurrence | PC | One model at origin; the attribution method has since transferred to three domains |
| E5 | Graded response | PC | Sign reverses between zero and double; nothing reported between them |
| E6 | Novel prediction | C | Unseen text fires the neurons where it expresses the fact and not otherwise |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | PC | The unit is declared exactly and the level is not; storage and expression run together |
| V2 | Level-evidence match | D | The title claims storage; the measurement is correlation with expression |
| V3 | Alternative level | I | The method alternative is rejected; the storage-versus-expression reading is untouched |
| V4 | Unlicensed labeling | PC | The name arrives in the sentence that introduces the method, before the measurement |
| V5 | Scope declaration | PC | Four limits declared; the one asserted past is generalization beyond BERT |
| **Total:** 5 Confirmed, 13 Partially confirmed, 2 Inconclusive, 12 Untested, 2 Disconfirmed, 2 Not applicable |   |   |   |
| **Verdict: Disconfirmed** |   |   |   |

## Sensitivity

Criterion judgments this audit record leaves contested: each carries an argument on both sides, and the status shipped is the one the record settled on.

| Criterion | In tension | What would settle it |
|---|---|---|
| I6 | Untested or Not applicable | Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Knowledge Neurons — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-knowledge-neurons/).
