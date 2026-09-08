---
title: "Case Study: Knowledge Neurons"
description: "Factual knowledge localization in feed-forward neurons (Dai et al. 2022) evaluated through the five core lenses."
---

# Case Study: Knowledge Neurons

[Dai et al. (2022)](https://arxiv.org/abs/2104.08696) attribute a relational fact to roughly four feed-forward neurons in BERT-base-cased, using integrated gradients over the intermediate activations of an FFN layer read as the value slots of a key–value memory. The evaluation runs over 253,448 ParaRel cloze prompts covering 27,738 facts and 34 relations.

Zeroing those activations lowers the correct-answer probability by 29.03% and doubling them raises it by 31.17%, against a count-matched activation-magnitude baseline that moves the same quantity by −1.47% and −1.27%. Rewriting the value slots installs a substituted entity as the top prediction 34.4% of the time, against the baseline's 0.0%.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Disconfirmed. **Capped by:** I4 (specificity).

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C1 Falsifiability, C2 Structural plausibility (both confirmed) | C4 Discriminant validity (disconfirmed) | Weak |
| Internal (Neuroscience) | I1/I2 Necessity + Sufficiency (both partial) | I4 Specificity (inconclusive, the capping criterion) | Weak |
| External (Pharmacology) | E2 Prompt generalization, E6 Novel prediction (both confirmed) | E3 Cross-task generalization (untested) | Weak |
| Measurement (Meas. Theory) | M2 Baseline separation (confirmed) | M1/M3/M5/M7 (all untested) | Weak |
| Interpretive (MI) | V1 Level declaration (partial) | V2 Level-evidence match (disconfirmed) | Weak |

**Overall verdict: Disconfirmed.** The primary reading — that these roughly four neurons *store* the relational fact — fails on two counts that were tested rather than skipped. All three of the paper's own summaries report a *correlation* between activation and expression while the title and abstract claim storage (V2). And the same editing machinery moves non-factual linguistic patterns [Niu et al. 2024], so the construct never separates from its neighbor (C4).

Weaker readings of the same evidence survive. That manipulating these neurons changes how strongly the model expresses the fact is **Causally Suggestive**. That editing them edits *that* fact and leaves unrelated knowledge alone is **Disconfirmed**: Table 6 of the origin gives an inter-relation perplexity rise of 7.2 for the identified neurons against 4.3 for random ones, and §5.1 reads the same table as showing little negative influence on other knowledge. That the account holds beyond BERT-base-cased is **Insufficient** — one model, and the generalization is asserted with no experiment behind it.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Integrated gradients over FFN activations | [A06 Mediation](/mechanistic-validity/framework/metrics/#a06) | Causal |
| Suppression and amplification of activations | [D02 Logit-Diff Recovery](/mechanistic-validity/framework/metrics/#d02) | Behavioral |
| Value-slot rewriting | [D01 Faithfulness](/mechanistic-validity/framework/metrics/#d01) | Behavioral |
| Activation-magnitude control attributor | [F06 Baseline Separation](/mechanistic-validity/framework/metrics/#f06) | Measurement |
| Cross-relation perplexity | [D03 KL Divergence](/mechanistic-validity/framework/metrics/#d03) | Behavioral |

---

## Philosophy of Science Lens — Construct Validity

*Is "knowledge neuron" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Confirmed.** The hypothesis is a signed prediction on a measured quantity, run in both directions on 34 relations. Zeroing should lower the correct-answer probability, doubling should raise it, and a count-matched control attributor should do neither. The control could have reproduced either effect and reproduced neither: 1.47% against 29.03% under suppression, and a 1.27% *decrease* against a 31.17% increase under amplification, where the control moves in the wrong direction entirely.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Confirmed.** The account names a concrete component and shows the algebra that makes it a memory. Equation (3) is put beside Equation (2) and the two are read as the same query–key–value operation under a different nonlinearity, licensing the first FFN layer as keys and the second as values. The intervention then lands on the object the account says holds the memory: editing modifies the value slots indexed by the attributed neuron.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** One attributor against one contrast, agreeing on one coarse property. Nothing independent of integrated gradients selects the same neurons.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Disconfirmed.** The neighboring construct is named in the source: §3.3 says the coarse set may hold neurons that express syntactic or lexical information. The paper removes them by a sharing rule resting on two stated hypotheses — that true positives recur across paraphrases and false positives do not — and then never measures whether the retained neurons carry syntactic or lexical signal. [Niu et al. (2024)](https://arxiv.org/abs/2405.02421) show the same editing machinery moves non-factual linguistic patterns, so the competitor was filtered out by assumption rather than separated by measurement.

**[C5 — Nomological validity:](/mechanistic-validity/framework/criteria/construct/nomological-validity) Partial.** Two theories say where to look — the key–value memory reading of FFN layers, and the localization premise behind model editing. Neither is tested as a network.

### Key Distinctions

- **Operationalism vs realism:** "Knowledge neuron" names the output of a procedure, and the paper's own summaries describe what the procedure finds as a correlate of expression. The construct is operational; the title's claim is realist.
- **Confirmation vs corroboration:** the web-crawled-text prediction (E6) is genuine corroboration — the account made it before the test and it could have failed.
- **Underdetermination:** storage and expression predict the same activation correlation, and no experiment here separates them.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Partial.** Zeroing the identified activations produces a deficit the matched control does not reproduce — 29.03% against 1.47% — consistent across all 34 relations rather than driven by a few. It is a decrement, not a collapse: the reported quantity is an average change ratio, so a 29.03% drop leaves most of the correct-answer probability intact and the fact still expressed after the neurons are switched off.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** Two results push toward sufficiency and neither reaches it. Doubling the activations raises the correct probability by 31.17%, and rewriting the value slots makes the substituted entity the top prediction 34.4% of the time against 0.0% for random neurons. Both are changes to a fact the model already expresses through the rest of its machinery, and the authors' own remedy for the 34.4% is to include more neurons.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Untested.** Set size is an input, not a result: thresholds are tuned to hold the count near four neurons before any effect is measured.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Inconclusive — the capping criterion.** The origin tests specificity three times and gets two answers. It holds at identification: fact pairs from different relations share 0.09 neurons against the control's 1.92. It holds at erasure: zeroing a relation's twenty most frequent neurons raises that relation's perplexity by 36.7% to 141.2% while raising other relations' by 1.1% to 10.1%. It reverses at update: rewriting the value slots raises inter-relation perplexity by 7.2 against 4.3 for random neurons, so the located units damage other knowledge *more* than random ones do.

**[I5 — Rival mechanism exclusion:](/mechanistic-validity/framework/criteria/internal/rival-mechanism-exclusion) Untested.** One alternative is ruled out and it is an alternative about method — the activation baseline stands for the possibility that the units are merely input-sensitive. A rival *mechanism* would be a different account of what these units do in the forward pass: routing a fact computed elsewhere, or amplifying a signal already present. The paper's framing forecloses the question by assuming the FFN module is where facts live.

**[I6 — Double dissociation:](/mechanistic-validity/framework/criteria/internal/double-dissociation) Untested.** One mechanism localized and one behavior measured, so neither arm of the crossed design exists.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Wording is controlled by requiring recurrence across roughly nine templates. Fact frequency in the training corpus is not.

**[I9 — Epistatic interaction:](/mechanistic-validity/framework/criteria/internal/epistatic-interaction) Untested.** Every manipulation hits the whole set at once, so no individual contribution separates.

**[I10 — Rescue reversibility:](/mechanistic-validity/framework/criteria/internal/rescue-reversibility) Untested.** The damage is analytic and invertible — the activations are set to zero and could be set back — and the restore is never run.

### Key Distinctions

- **Lesion vs stimulation:** both directions are present, which is unusual, and the sign reversal between them rules out a generic damage effect.
- **Localization vs distributed:** a 29.03% decrement on an average change ratio is consistent with a distributed representation in which these neurons carry one share.
- **Single vs double dissociation:** neither arm exists. The specificity result that does exist points in two directions depending on which intervention produced it.

### Dissociation Matrix

|  | Target relation | Other relations | Syntactic/lexical patterns | General LM |
|---|---|---|---|---|
| Zero identified neurons | **↓ 29.03% (control 1.47%)** | ↑ perplexity 1.1–10.1% | ? | ? |
| Rewrite value slots | top-1 substitution 34.4% | **↑ perplexity 7.2 (random 4.3)** | moves them [Niu et al. 2024] | ? |

The erasure row and the update row disagree about specificity, which is why I4 is Inconclusive rather than confirmed or failed. The third column is the neighboring construct C4 never separates from.

---

## Pharmacology Lens — External Validity

*Does intervening on the neurons produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** Three intervention forms appear and they agree in direction: zeroing an activation, doubling it, and rewriting the value slot the activation weights. Two act on activations and the third on the weights those activations multiply, so all three enter the forward pass at the same point, and all three are applied to a set chosen by a single attributor.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Confirmed.** Survival across roughly nine paraphrase templates is built into the identification procedure, so every retained neuron is one that recurs across restatements of the same fact.

**[E3 — Cross-task generalization:](/mechanistic-validity/framework/criteria/external/cross-task-generalization) Untested.** Single-word cloze throughout, named first among the authors' own limitations.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Partial.** The origin runs on one model, asserts that the method generalizes, and supplies no experiment; its conclusion lists the multilingual case as future work. What has since transferred is the attribution procedure, across three domains, rather than the storage reading placed on its output.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** The intervention takes two values, zero and double, and the response reverses sign between them — from −29.03% to +31.17% — which rules out a generic damage effect. Nothing between or beyond those points is reported: no interpolation, no multiplier sweep, no curve of effect against strength. The authors call the manipulation a proof of concept and defer precise control.

**[E6 — Novel prediction:](/mechanistic-validity/framework/criteria/external/novel-prediction) Confirmed.** The prediction is derived before the test and could have failed: if these neurons carry a relational fact, text the model never saw during identification should activate them when it expresses that fact and leave them alone when it merely mentions the entities. Both halves hold on web-crawled text — 0.485 for relation-bearing prompts against 0.019 for head-only prompts — and Table 3 gives the sharper form, where the least-activating prompts contain both entities and still fail to fire the neurons.

### Key Distinctions

- **Affinity vs efficacy:** the E6 result is affinity evidence of an unusually clean kind. Efficacy at the level the title claims — installing a fact — reaches 34.4%.
- **The metric is part of the finding:** the same Table 6 supports "little negative influence on other knowledge" in the paper's reading and an adverse specificity result in ours, because a raw perplexity rise and a rise relative to random controls are different quantities.
- **Naming requires criteria:** the label arrives in the sentence that introduces the method, before any measurement.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Untested.** Every quantity is a mean reported once, with no interval and no repeated run.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Confirmed.** One negative control runs through the whole paper and is matched on the quantity that would otherwise explain any gap. Raw activation magnitude is put through the identical refining pipeline with thresholds tuned to the same [2, 5] target, yielding 3.96 neurons against 4.13 — the same size. It then separates at every stage measured: suppression, amplification, prompt discrimination, and edit success, where the control succeeds 0.0% of the time.

**[M3 — Stability:](/mechanistic-validity/framework/criteria/measurement/stability) Untested.** Three free settings are stated and none of them is perturbed.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The outcome quantities are calibrated and the selection quantity is not. A correct-probability change ratio carries a sign, a zero meaning no effect, and a control value measured on the same relations. The attribution score is thresholded at 0.2 of its own per-prompt maximum, which rescales it per prompt and leaves its absolute value uninterpretable, and the surviving neuron count is then set by driving a selection quantity into a target range.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Untested.** One known-negative control and no case where the answer is known in advance.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Partial.** Per-relation spread is reported across all 34 relations. The generalization claim beyond BERT has no experiment.

**[M7 — Selection correction:](/mechanistic-validity/framework/criteria/measurement/selection-correction) Untested.** Selection happens at four places, is disclosed at all four, and is corrected at none.

### Key Distinctions

- **Reliability vs validity:** M2 is the strongest cell on this page and it does not rescue the claim. A control that separates cleanly establishes that *something* was found, not that what was found is storage.
- **True score vs observed score:** the attribution score is normalized per prompt, so its scale carries no information across prompts.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Partial.** The unit is declared exactly — a named neuron in a named FFN layer — and the level is not. Storage and expression run together throughout.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Disconfirmed.** The claim and the evidence sit at different levels, and the paper says both in its own summary sentences. The title and abstract claim storage, a claim about where a fact is held in the weights. The abstract's statement of the result and the contribution list both report a positive correlation between activation and the expression of the fact, a claim about a signal covarying with an output. The strongest level-matched evidence is the value-slot surgery, and it installs the intended fact 34.4% of the time.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Inconclusive.** The method alternative is rejected. The storage-versus-expression reading — the one that decides what the title means — is untouched.

**[V4 — Unlicensed labeling:](/mechanistic-validity/framework/criteria/interpretive/unlicensed-labeling) Partial.** The name arrives before the measurement. §1 introduces the method and the label in one sentence: the neurons the method identifies "are named knowledge neurons." The term states what the procedure is assumed to find rather than abbreviating a result.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Partial.** Four limits are declared. The one asserted past is generalization beyond BERT.

### Key Distinctions

- **Description vs explanation:** the description is precise at the level of the unit and imprecise at the level of the claim. That combination is what V2 is built to catch.
- **Component identity vs component role:** identity is established by an attributor with a matched control. The role — storage — is asserted in the name and measured as correlation.
- **Faithfulness vs understanding:** the manipulations are real and the reading placed on them is the disconfirmed part.

### Evidence Convergence Map

- **Implementational → Interpretation:** strong for identity. The control separates at every stage.
- **Representational → Interpretation:** absent. Nothing measures what the retained neurons encode, which is why C4 fails.
- **Computational → Interpretation:** partial and adverse. The account predicts that editing these neurons edits this fact and spares others; the measurement says they damage other relations more than random neurons do.
