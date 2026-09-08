---
title: "Case Study: Greater-Than Circuit"
description: "The Greater-Than circuit (Hanna et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Greater-Than Circuit

[Hanna et al. (2023)](https://arxiv.org/abs/2305.00586) identify a subgraph of GPT-2 Small that performs the **Greater-Than task** — given "The war lasted from the year 1732 to the year 17\_\_", the model must predict a two-digit suffix greater than 32. The circuit is a set of attention heads that carry the start year YY to the final position, plus MLPs 8–11, which raise the probability of every year above YY. It is found by iterative path patching against a corruption dataset in which every start year is replaced by 01.

The claim is implementational: it locates where the computation happens and what each component contributes to the output distribution. What the comparison operation itself *is* remains open, and the authors say so — "we cannot provide a conclusive answer" (§4).

## Composite Verdict

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I5 (rival mechanism exclusion), I6 (double dissociation).

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C1 Falsifiability (confirmed) | C3/C4/C5 (all partial) | Strong |
| Internal (Neuroscience) | I1/I2 Necessity + Sufficiency (both confirmed) | I5 Rival mechanism exclusion (inconclusive), I6 Double dissociation (untested) | Mechanistically supported |
| External (Pharmacology) | E6 Novel prediction | E4 Cross-model generalization (disconfirmed) | Partial |
| Measurement (Meas. Theory) | M4/M6 Calibration + Invariance (partial) | M1/M3/M5/M7 (all untested) | Partial |
| Interpretive (MI) | V4/V5 Unlicensed labeling + Scope (both confirmed) | V1/V2 Level declaration + match (partial) | Strong |

**Overall verdict: Mechanistically Supported.** Necessity and sufficiency are both confirmed, and by the strong forms of each: routing the counterfactual through the circuit takes probability difference from +81.7% to −36.6%, a sign reversal rather than a degradation, and the complement test recovers 72.7% under conditions the authors chose to favor failure. The claim is capped by rival mechanism exclusion (I5) — memorization is never tested, and the paper's own structured-number-representation account fails two causal tests — and by double dissociation (I6), where the second mechanism is located observationally and never intervened on.

Three further readings of the same evidence do not reach that tier. Whether the circuit *computes* the comparison rather than retrieving memorized year associations is **Underdetermined**. Whether it implements greater-than *as such* is **Disconfirmed**: two prompts requiring less-than recruit the same circuit and the model answers greater. Whether the same circuit is the mechanism beyond GPT-2 Small is **Disconfirmed** on post-origin evidence across three other model families.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Iterative path patching at four granularities | [D01 Faithfulness](/mechanistic-validity/framework/metrics/#d01) | Behavioral |
| Probability difference over year tokens | [D02 Logit-Diff Recovery](/mechanistic-validity/framework/metrics/#d02) | Behavioral |
| Logit lens on component outputs | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| LEACE erasure of linearly-extractable YY | [A06 Mediation](/mechanistic-validity/framework/metrics/#a06) | Causal |
| Cross-task transfer (§5) | [D06 Cross-Task](/mechanistic-validity/framework/metrics/#d06) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "the Greater-Than circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Confirmed.** The claim is stated so that it could have come out flat and did not. Probability difference carries a sign and a two-sided range, the prediction attached to each component is directional and named before the test, and the necessity experiment moves the quantity from +81.7% to −36.6% — past the indifference point, not merely toward it. A second metric is designed against a specific way of passing for the wrong reason: cutoff sharpness "checks that the model depends on YY, and does not produce constant (but valid) output, e.g. by always outputting $p(99) = 1$" (§2).

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** Information flow is established at three granularities: heads attend to the YY position and put YY at the top of the whole vocabulary in unembedding space; MLPs 9 and 10 show the upper-triangular logit-lens signature of greater-than; and successively larger neuron subsets of MLP 10 sharpen into the same pattern. The comparison itself is left open, and the one structural hypothesis about how it might work — number representations with linear structure — is put to two causal tests and survives neither. Mechanism located and not derived.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** One cross-validation at origin is genuine and run for a stated reason: the logit lens ignores the final LayerNorm, so Appendix E recomputes every logit-lens result as a direct effect measured by path patching, and the two agree up to scale. Circuit discovery has no such second instrument — every component enters through iterative path patching against one corruption dataset.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** Separation from a neighboring construct is tested and found: three tasks a greater-than mechanism should cover recruit a different set of components entirely. The contrasts are all of one kind — every comparison task changes the operation, the frame, or both, so nothing isolates the frame while holding the operation fixed.

**[C5 — Nomological validity:](/mechanistic-validity/framework/criteria/construct/nomological-validity) Partial.** The claim is placed between memorization and generalization rather than derived from either. It connects to the model's year-token geometry and to the task's output space, but not to a wider account that would predict where else the mechanism should appear.

### Key Distinctions

- **Confirmation vs corroboration:** §5's cross-task experiment is corroboration rather than confirmation — the transfer results were not what the circuit was selected on, and the mixed outcome (67.8% to 98.8% on related tasks, different components entirely on three others) is reported at full width.
- **Operationalism vs realism:** The construct name is a mathematical relation defined by the task, not a faculty attributed to the model. Every component label redescribes a measurement made first.
- **Observable vs theoretical:** The flow evidence sits close to the observations; the comparison operation, which is the theoretical part, is the part left open.

### Nomological Network

The Greater-Than circuit connects to:
- **Behavioral prediction** — routing the counterfactual through the circuit reverses the sign of probability difference (causal, confirmed)
- **Sufficiency** — the circuit alone recovers 72.7% while every path outside it carries 01-input (causal, confirmed)
- **Representational structure** — MLPs 9 and 10 show an upper-triangular logit-lens signature (structural, confirmed)
- **Novel prediction** — prompts supplying YY in a different frame recruit the same components, recovering 98.8% and 88.9% (confirmed)
- **Cross-task boundary** — transfer holds inside the year frame and fails outside it (scope, confirmed and bounded)
- **Cross-model prediction** — tested post-origin; the circuit differs per model across three families (disconfirmed)
- **Training dynamics** — no published checkpoints, so onset cannot be timed at all (structurally unavailable)

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Confirmed.** The necessity test is the strong form. Routing the 01-dataset through the circuit while the rest of the model receives real input drives probability difference to −36.6% against a +81.7% baseline, so the model stops preferring valid years and starts preferring invalid ones. Degradation alone would be consistent with the circuit carrying part of a distributed signal; a sign reversal says the circuit's output determines which side of YY the mass falls on.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Confirmed.** The complement test is run under conditions that favor failure and passes. Every path outside the circuit carries 01-input, which the authors state should push the model to poor performance, and the circuit alone recovers 72.7% of the 81.7% baseline. The anti-degeneracy metric moves the same way — cutoff sharpness rises to 8%, so the recovered behavior still depends on YY rather than collapsing onto a constant prediction. Appendix B repeats the test for the full input-to-logits circuit and reaches 71.5%.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** Minimality is asserted in the definition of a circuit and probed by decomposition rather than by pruning. Neuron-level patching shows most of MLP 10 can be ablated with near-zero effect, and each MLP's direct and indirect contributions are separated. But no component of the published circuit is ever removed on its own and the remainder rescored, so no member has been shown to earn its place. The one accretion result runs the other way: adding MLP 7 and two heads lifts a related task from 67.8% to 90.3%.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Partial, with an adverse result.** Specificity is tested in the direction that can embarrass the claim, and it fails there. Two prompts requiring less-than recruit the same circuit and produce greater-than output, and a non-monotone random-sequence case does the same. The authors report this as a property of the mechanism rather than smoothing it away. The converse test — ablate the circuit and measure damage to capabilities that should be untouched — is missing.

**[I5 — Rival mechanism exclusion:](/mechanistic-validity/framework/criteria/internal/rival-mechanism-exclusion) Inconclusive — a capping criterion.** Two rival accounts are on the table and neither is disposed of. Structured number representations are tested causally twice and fail both times: ablating the PCA dimensions changes little, and LEACE erasure of linearly-extractable YY costs 17 points against a baseline of 81.7%. A failed test of a rival leaves the rival unsupported, not the main account supported. Memorization is not tested at all, and the authors state their circuit could be a lookup table.

**[I6 — Double dissociation:](/mechanistic-validity/framework/criteria/internal/double-dissociation) Untested — a capping criterion.** A double dissociation needs two mechanisms and two behaviors, with each intervention moving one and sparing the other. Ablating the circuit destroys year-span prediction, and a different set of components is identified for the tasks the circuit does not serve — but that second set is located observationally and never intervened on. Nothing shows it intact under circuit ablation, and nothing removes it while leaving greater-than behavior.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** One confound is controlled to the point of shaping the dataset. Byte-pair encoding makes frequent years single tokens, which would let the model score well for a reason unrelated to comparison, so single-token years are excluded and each century's extremes are dropped. Lexical content is randomized over 120 nouns. The confound the authors themselves identify is left in: nouns carry duration priors that shape which years get probability.

**[I9 — Epistatic interaction:](/mechanistic-validity/framework/criteria/internal/epistatic-interaction) Partial.** Non-additivity is the neuron-level finding rather than a side observation, and the argument for it is structural: a neuron's activation scales its logit distribution without reshaping it, so the greater-than pattern appears only in sums. What is missing is any test of interaction as such — no pair of components is ablated together and compared against the sum of its singles.

### Key Distinctions

- **Single vs double dissociation:** The paper has the makings of one half and never runs the second. This is the field's most common gap, and it is what caps this claim alongside I5.
- **Localization vs distributed:** The sign reversal establishes that the circuit's output determines the direction of the effect, which is a stronger localization result than a magnitude drop would be.
- **Lesion vs stimulation:** Both directions of patching appear — the circuit receives corrupted input in one experiment and clean input in the other — but there is no amplification experiment.

### Dissociation Matrix

|  | Year-span prediction | Tasks recruiting other components | Less-than prompts | General LM |
|---|---|---|---|---|
| Ablate greater-than circuit | **↓↓ (sign reversal, +81.7% → −36.6%)** | ? | ? | ? |
| Ablate the other component set | ? | ? | ? | ? |

One cell filled. The second component set is identified but never ablated, so the converse arm of the crossed design does not exist. The less-than column is informative in the wrong direction: the circuit fires there and the model answers greater.

---

## Pharmacology Lens — External Validity

*Does intervening on the circuit produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** Three intervention families appear and they are not variants of each other: path patching at four granularities, PCA-dimension ablation, and LEACE erasure. The families disagree usefully — patching supports the circuit while both erasure methods decline to support the representational story behind it. What none of them varies is the counterfactual: one corruption dataset sits behind every patching result.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Partial.** Coverage is broad inside one frame and the frame is held fixed by design. The origin runs 10,000 examples over 120 nouns and 768 years, then three variant prompts that change the surface form while keeping the two-digit year answer space, recovering 98.8% and 88.9% on the first two. Nothing tests a prompt whose answer space differs.

**[E3 — Cross-task generalization:](/mechanistic-validity/framework/criteria/external/cross-task-generalization) Partial.** §5 is this criterion's experiment and it is reported at full width. Three related tasks recruit similar circuits with recoveries from 67.8% to 98.8%, three tasks that should have used the circuit recruit different components entirely, and two tasks requiring less-than fire the circuit anyway. Transfer holds for tasks sharing the year frame and fails outside it.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Disconfirmed.** GPT-2 Small at origin, with the gap named. Run later across three other model families, the circuit differs per model. This is tested and failed, not untested.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** One axis is graded and it is the wrong one for this criterion. Circuit extent varies smoothly — the logit lens of MLP 10 is shown for the top 3, 10, 100 and 200 neurons, with the greater-than pattern sharpening as the set grows. Intervention strength never varies: patching replaces a node's input with 01-input outright, with no interpolation and no scaling parameter to sweep.

**[E6 — Novel prediction:](/mechanistic-validity/framework/criteria/external/novel-prediction) Partial.** The mechanism yields a prediction before the test that could have failed: if MLPs 9–11 perform the comparison on a YY value delivered by attention heads, any prompt supplying YY in a different frame should recruit the same components — and the first two variants recover 98.8% and 88.9%. A second, implicit prediction fails informatively: a circuit tracking the greater-than operation should stay silent when less-than is required, and instead it fires.

### Key Distinctions

- **The counterfactual is part of the finding:** every patching number here is a number about the 01-dataset. Nothing in the paper varies it, so the intervention families agree only where they share it.
- **Disagreement reported rather than averaged:** the erasure results decline to support the representational story while patching supports the circuit, and both are reported. That is what E1 asks for; what is missing is a second counterfactual.
- **A failed prediction reported as a finding:** the less-than result is the strongest evidence on the page about what the circuit is not, and it is in the paper rather than in a rebuttal.

### Dose-Response Curve

- **Graded axis available:** circuit extent — top 3, 10, 100 and 200 neurons of MLP 10, with the greater-than pattern sharpening as the set grows; and circuit membership, where adding MLP 7 and two heads lifts a related task from 67.8% to 90.3%.
- **Graded axis absent:** intervention strength. Patching is all-or-nothing per node, so $\alpha_{\text{thresh}}$, $\alpha_{\text{plat}}$ and the therapeutic window are undefined for this claim.
- **Off-target at dose:** not measured at any extent.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Untested.** Dispersion is reported twice, both times pre-circuit. No interval accompanies any result, including the headline 72.7% and 89.5%.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** A size- and location-matched comparison exists and it is the right comparison to make: circuits of comparable extent are scored, with performance reported to track how much of the original path structure they retain. The comparison is delivered entirely in prose — no count, no figure, no distribution — so 89.5% has no reference class to sit in.

**[M3 — Stability:](/mechanistic-validity/framework/criteria/measurement/stability) Untested.** Circuit membership is read off heatmaps. No threshold is stated and none is swept.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** Both outcome metrics are well calibrated: each is bounded on [−1, 1], each carries a sign, zero means indifference between valid and invalid years, and each has a measured model value over 10,000 examples. The validation quantity inherits none of that — faithfulness as a percentage of the unpatched model is interpretable only against a distribution of comparable circuits the paper does not supply.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Untested.** Two experiments, both on the real model. No mechanism of known extent is planted and recovered, so the instrument's floor is unmeasured.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Partial.** The design makes per-condition variation visible instead of averaging it away: every behavioral result is a heatmap over start year crossed with predicted year on 10,000 examples spanning 120 nouns and 768 years. One non-invariance is documented and costed — attention to the YY position varies non-monotonically with YY, bottoms out at YY=01, and patching queries and keys costs 15% of partial-circuit performance. One tokenizer, one model, one frame throughout.

**[M7 — Selection correction:](/mechanistic-validity/framework/criteria/measurement/selection-correction) Untested.** Every candidate is screened and disclosed, which is more than most papers do, but no multiplicity control is applied to the screen.

### Key Distinctions

- **Sensitivity vs specificity:** the outcome metrics are well-calibrated and the validation metric is not. A percentage-of-unpatched figure looks like a calibrated number and is not one.
- **Disclosure is not correction:** screening every candidate and reporting it addresses transparency, not the family-wise error rate.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Partial.** The declaration is implementational — where the computation happens and what each component contributes. The title asks an algorithmic question, and the paper does not say which of the two the evidence is meant to settle.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** The evidence matches the claim on flow and topography. It does not match the claim the title advertises: no experiment addresses how the comparison is computed, and the authors say as much.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** The competing description is named, entertained seriously, and partly accepted. A lookup table over memorized year associations would produce the same circuit diagram; the authors say so, and then state what would remain true under that reading — that the retrieval is selective and operand-sensitive. No experiment separates the two, so the alternative is bounded rather than excluded.

**[V4 — Unlicensed labeling:](/mechanistic-validity/framework/criteria/interpretive/unlicensed-labeling) Confirmed.** Every component label redescribes a measurement made first. Heads "communicate YY" after the logit lens shows YY as the top-ranked token in their output; MLPs "upweight precisely those years greater than YY" after the upper-triangular pattern is shown; MLP 11 "enforces a maximum duration" after the 50-year window is visible in its logit lens. No component is given an intention.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Confirmed.** Scope is declared before the evidence and again after it, including the single-model restriction and the frame the prompt variants hold fixed.

### Key Distinctions

- **Description vs explanation:** the paper describes the flow completely and declines to explain the operation. Declining is the right move given the evidence, and it is why V1 and V2 sit at partial rather than lower.
- **Component identity vs component role:** every role name here is licensed by a measurement, which is why V4 is confirmed where most audited claims are not.
- **Faithfulness vs understanding:** faithfulness is established and understanding is not. The two axes come apart cleanly in this case.

### Evidence Convergence Map

- **Implementational → Interpretation:** strong. Path patching at four granularities plus the logit lens locate the flow.
- **Algorithmic → Interpretation:** absent. The comparison operation is stated as unresolved, and the one structural hypothesis about it fails two causal tests.
- **Computational → Interpretation:** partial. §5 bounds the task family the account covers and reports where it does not.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Path patching | ✓ | ✓ | — | ∅ | ∅ |
| Logit lens | — | — | ✓ | — | — |
| PCA-dimension ablation | — | — | ✗ (negative) | — | — |
| LEACE erasure | — | — | ✗ (negative) | — | — |
| Cross-task testing (§5) | — | — | — | — | ✓ (bounded) |

Both erasure rows return negative results about the representation, which is why the algorithmic column stays empty. The two causal columns are filled by one method against one counterfactual.

### Causal Sufficiency Graph

- Input (year tokens) → attention heads carrying YY: **solid** (logit lens puts YY at the top of the vocabulary in the heads' output)
- MLPs 8–11 → output logit modulation: **solid** (upper-triangular logit-lens signature, sharpening with neuron subset size)
- Full path (input → circuit → greater-than output): **solid** (sign reversal under corruption, 72.7% recovery under complement ablation)
- YY value → comparison operation: **unknown** (the authors state no conclusive answer; the structured-representation account fails two causal tests)
- Circuit → behavior on inputs the label does not cover: **adverse** (less-than prompts recruit the circuit and the model answers greater)

Three solid edges, one unknown, one adverse. The unknown edge is the comparison itself, which is what the title asks about.
