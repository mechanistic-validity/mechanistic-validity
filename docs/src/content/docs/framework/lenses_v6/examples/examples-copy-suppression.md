---
title: "Case Study: Copy Suppression"
description: "The copy suppression mechanism (McDougall et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Copy Suppression

[McDougall et al. (2023)](https://arxiv.org/abs/2310.04625) identify **copy suppression heads** in GPT-2 Small — attention heads that detect when the model is about to copy a token and actively suppress that copying. The mechanism functions as an anti-induction circuit: where induction heads promote copying repeated tokens, copy suppression heads inhibit it, preventing the model from naively repeating tokens that appear in context but are not the correct next prediction.

This is unusual because it is defined by what it *prevents* rather than what it produces — a negative-effect mechanism.

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C2 Structural plausibility | C5 Convergent | Partial–Strong |
| Internal | I3 Specificity | I5 Confound control | Causally suggestive |
| External | E3 Selectivity | E6 Cross-architecture | Partial |
| Measurement | M3 Baseline separation | M1 Reliability | Partial |
| Interpretive | V3 Narrative coherence | V4 Alternative exclusion | Strong |

**Overall verdict: Causally suggestive, approaching Mechanistically supported.** Copy suppression is notable for its unusually clean specificity result (I3) — ablation produces a specific error type rather than general degradation. This is rare in MI and provides stronger evidence than typical necessity results. The mechanism is a good example of how negative-effect components (inhibitory mechanisms) can be as well-characterized as positive-effect ones.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Ablation (mean ablation) | [A01 Pearl SCM](/framework/metrics/causal/a01-scm-pearl) | Causal |
| Direct logit attribution (DLA) | [D02 Logit-Diff Recovery](/framework/metrics/behavioral/d02-logit-diff-recovery) | Behavioral |
| $W_{OV}$ decomposition (anti-copying structure) | [B03 OV/QK Decomposition](/framework/metrics/structural/b03-ov-qk-decomposition) | Structural |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "copy suppression" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim predicts: (1) these heads should produce negative direct logit attribution on tokens that are about to be copied, (2) ablating them should *increase* the probability of incorrect token repetition. Both are testable and concrete.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Pass.** The $W_{OV}$ matrices of copy suppression heads show anti-copying structure — they project negatively onto the tokens they attend to, effectively subtracting those tokens from the output logits. This is the structural mirror of the positive copying structure in name-mover/induction heads.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Partial.** Copy suppression is not task-specific — it operates across any context where token repetition is likely but incorrect. This is an honest scope claim (like induction heads), but the boundary of when suppression activates versus when copying is appropriate is not precisely characterized.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Pass.** A small number of heads are identified. Each contributes independently measurable negative DLA.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** Identified through DLA (behavioral) and confirmed through $W_{OV}$ analysis (structural). Two method families, partially independent.

### Key Distinctions

- **Confirmation vs corroboration:** The $W_{OV}$ anti-copying structure provides partial corroboration of the behavioral DLA finding — the two methods have different assumptions (one is static weight analysis, the other is dynamic attribution). However, both ultimately rely on the same linear decomposition of logit contributions, so the assumption independence is partial rather than complete.
- **Operationalism vs realism:** "Copy suppression" is operationally grounded (negative DLA on tokens about to be copied), descriptive of function rather than asserting deeper theory, and directly falsifiable. The naming discipline is exemplary — a head so named that does not suppress copying would be misclassified.
- **Underdetermination:** Could "copy suppression" be a side effect of a more general computation? The structural evidence ($W_{OV}$ anti-copying pattern) constrains alternatives, but does not fully exclude the possibility that these heads implement a broader "novelty detection" or "prediction correction" function that incidentally manifests as copy suppression.

### Nomological Network

The copy suppression construct connects to:
- **Weight structure** — $W_{OV}$ matrices project negatively onto attended tokens (structural, confirmed)
- **Behavioral prediction** — ablation increases incorrect token repetition (causal, confirmed)
- **DLA signature** — negative direct logit attribution on copy-tempting tokens (behavioral, confirmed)
- **Interaction with induction heads** — functionally opposes the copying circuit (theoretical, partially confirmed through complementary effects)
- **Activation boundary** — when does suppression activate vs. permit appropriate copying? (untested)
- **Cross-model prediction** — do other architectures have analogous anti-copying mechanisms? (untested)
- **Training dynamics** — does copy suppression emerge after induction heads? (untested)

Four nodes confirmed, three unconnected. The confirmed nodes establish a coherent negative-effect mechanism, but the boundaries of its activation and its developmental relationship to copying mechanisms remain unexplored.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Ablating copy suppression heads causes the model to *over-copy* — token repetition probability increases on prompts where copying would be incorrect. This is a specific and interpretable necessity result.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Partial.** The mechanism is demonstrated through its effect (suppressing logits), but a full isolation test (can these heads alone prevent copying when the rest of the model promotes it?) is not reported.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Pass.** The effect is specific to suppression of incorrect copying. Ablating these heads does not generally degrade model performance — it specifically increases token repetition errors. This is an unusually clean specificity result because the effect direction is distinctive (increase in a specific error type, not general degradation).

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** Demonstrated across varied prompts where copying is inappropriate. Cross-model consistency not reported.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not tested.** Single ablation method.

### Key Distinctions

- **Single vs double dissociation:** Copy suppression comes close to double dissociation: ablating these heads increases copying errors specifically (not general degradation), and general-purpose heads do not produce this specific error pattern when ablated. The error-type specificity approximates the discriminative power of a formal double dissociation without requiring a second circuit as control.
- **Lesion vs stimulation:** Only the lesion direction is tested (ablation increases copying). The stimulation direction — amplifying copy suppression heads to test whether the model becomes overly reluctant to repeat tokens — would provide complementary evidence but is not reported. This leaves the sufficiency side incomplete.
- **Localization vs distributed:** Copy suppression is relatively localized (a small number of identifiable heads), but it operates in a distributed context — it must interact with induction heads and name-mover heads to detect when copying is inappropriate. The mechanism is localized; the computation it participates in is distributed.

### Dissociation Matrix

|  | Over-copying errors | General LM quality | Non-repetition tasks |
|---|---|---|---|
| Ablate copy suppression heads | **↑↑ (specific increase)** | Minimal change | Minimal change |
| Ablate induction heads | ↓ (reduced copying overall) | Some degradation | ? |
| Ablate random heads | ? | General degradation | General degradation |

The distinctive pattern: ablating copy suppression heads produces a *specific error type increase* (over-copying) without general degradation. This is stronger than typical single dissociation because the effect direction (increase in a specific error) is diagnostically distinctive — it would not be produced by removing a general-purpose component. The complementary row (induction head ablation reduces copying) provides implicit double-dissociation structure.

---

## Pharmacology Lens — External Validity

*Does intervening on the mechanism produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Partial.** The mechanism operates wherever the model encounters copy-tempting contexts. Its reach is defined by the breadth of such contexts in natural text.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Not tested.** Does stronger copy signal produce stronger suppression? A parametric relationship is not measured.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Pass.** The intervention (ablation) selectively produces over-copying without general performance degradation. This is clean selectivity.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Moderate.** The effect is measurable and specific, but copy suppression is one of many mechanisms contributing to output quality. It is not the dominant mechanism for any single task.

**[E5 — Robustness:](/framework/criteria/external/robustness) Partial.** Works across varied copy-tempting prompts. Not tested on edge cases (when should the model copy vs. suppress?).

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** GPT-2 Small only.

### Key Distinctions

- **Affinity vs efficacy:** The anti-copying $W_{OV}$ structure demonstrates affinity (the mechanism is structurally suited to suppress copying), while the ablation-induced over-copying demonstrates efficacy (it actually performs this function in vivo). Both sides are established, though graded efficacy — whether stronger copy signals produce proportionally stronger suppression — is not measured.
- **The metric is part of the finding:** The choice to measure over-copying rate (a specific error type) rather than general perplexity is what makes the specificity result so clean. A less targeted metric would have shown generic degradation and obscured the mechanistic insight. The metric design is integral to the finding's strength.
- **Naming requires criteria:** "Copy suppression" is an exemplary mechanistic name: it is operationally grounded (negative DLA on tokens about to be copied), descriptive of function rather than asserting deeper theory, and directly falsifiable (a head so named that does not suppress copying would be misclassified). The naming discipline here could serve as a model for the field.

### Dose-Response Curve

The copy suppression mechanism's dose-response is largely unmapped:
- **α = 0** (no intervention): normal model behavior (appropriate mix of copying and suppression)
- **α = 1** (complete ablation): over-copying on copy-tempting prompts, minimal general degradation
- **Missing:** No intermediate ablation strengths tested
- **Missing:** No measurement of whether stronger copy signals in the input produce proportionally stronger suppression activation
- **Missing:** No off-target curve (at what intervention strength do non-copying behaviors begin to degrade?)

The key insight from a pharmacological perspective: the selectivity of the full-ablation effect (over-copying without general degradation) implies a wide therapeutic window — the mechanism can be fully removed without collateral damage. But without intermediate points, we cannot characterize the curve's shape (linear? threshold? sigmoidal?).

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Not reported.** No confidence intervals.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Partial.** Works across prompt types. Layer/position invariance not tested.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** Copy suppression heads show clearly negative DLA on copy-tempting tokens, while other heads do not. Clean separation.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Good.** The negative-DLA criterion cleanly identifies suppression heads.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.**

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Good.** Both behavioral (ablation effect) and structural ($W_{OV}$) measurements used.

### Key Distinctions

- **Sensitivity vs specificity:** The negative-DLA criterion has excellent specificity — it cleanly separates copy suppression heads from all others without false positives. Sensitivity is also good: the metric reliably identifies the relevant heads across varied copy-tempting contexts. This measurement quality is a notable strength of the study.
- **Convergent vs discriminant validity:** Two methods (DLA and $W_{OV}$ structure) converge on the same heads — partial convergent validity. Discriminant validity is implicit: heads with positive DLA (name-movers, induction heads) are clearly distinguished from heads with negative DLA. But a formal discriminant test across multiple constructs is not reported.

### MTMM Matrix

| | Negative DLA (copy supp.) | $W_{OV}$ anti-copy (copy supp.) | Negative DLA (other heads) | $W_{OV}$ anti-copy (other heads) |
|---|---|---|---|---|
| **Negative DLA (copy supp.)** | — | High (convergent) | Low (discriminant) | Low (discriminant) |
| **$W_{OV}$ anti-copy (copy supp.)** | High | — | Low (discriminant) | Low (discriminant) |
| **Negative DLA (other heads)** | Low | Low | — | ? |
| **$W_{OV}$ anti-copy (other heads)** | Low | Low | ? | — |

The convergent diagonal is strong: heads identified by negative DLA are the same ones with anti-copying $W_{OV}$ structure. The discriminant cells show clean separation — other heads lack both signatures. The pattern is well-behaved, though the number of methods (two) is minimal for a full MTMM analysis.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** Algorithmic — names what the heads do (suppress copying) and how ($W_{OV}$ anti-copying).

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Pass.** Structural + behavioral evidence matches algorithmic claim.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** The story is mechanistically precise: model is tempted to copy → copy suppression heads detect this → they subtract the copy signal from logits. The negative-effect framing is clean and testable.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Partial.** Could these heads be doing something else that incidentally suppresses copying? The structural evidence (anti-copying $W_{OV}$) constrains alternatives, but the possibility that "suppression" is a side effect of a more general computation is not fully excluded.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Good.** Presented as general-purpose anti-copying, which matches the evidence scope.

### Key Distinctions

- **Description vs explanation:** The copy suppression narrative is explanatory — it specifies not just that these heads are active, but *what they do* (subtract copy signal from logits) and *why this matters* (prevents incorrect repetition). The negative-effect framing adds explanatory power by connecting the mechanism to a functional role in the broader system.
- **Component identity vs component role:** The role label "copy suppression" is well-supported by the anti-copying $W_{OV}$ structure — the structural evidence independently confirms the behavioral label. However, whether "copy suppression" fully characterizes these heads' function or is one aspect of a broader role remains open.
- **Faithfulness vs understanding:** The mechanism is faithful (ablation confirms importance) and understood (the structural basis is characterized). The understanding is incomplete only in the sense that the activation boundary (when to suppress vs. permit copying) is not precisely mapped.

### Evidence Convergence Map

- **Implementational → Interpretation:** Moderate-strong. Ablation identifies specific heads; $W_{OV}$ structure confirms anti-copying role. Two implementational lines converge.
- **Algorithmic → Interpretation:** Moderate. "Detect copy temptation → subtract copy signal" is a specified algorithm, but the detection mechanism (how do these heads know when copying is inappropriate?) is not fully characterized.
- **Computational → Interpretation:** Moderate. The computational description (suppress incorrect copying) matches evidence, but the decision boundary between "appropriate copying" and "incorrect repetition" is not mapped.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ | — | ∅ | ∅ | ∅ |
| DLA analysis | ✓ | — | ✓ | — | — |
| Weight analysis | — | — | ✓ | ✓ (partial) | — |
| Stimulation (amplification) | — | — | — | — | — |

The ablation and DLA rows provide necessity and representational evidence. Weight analysis adds algorithmic understanding (the $W_{OV}$ structure specifies *how* suppression occurs). Stimulation is entirely absent — amplifying suppression heads to test whether the model under-copies has not been attempted. The sufficiency column is empty.

### Causal Sufficiency Graph

- Copy-tempting context → copy suppression head activation: **dashed** (the mechanism is active on copy-tempting tokens, but how it *detects* copy temptation — vs. being tonically active — is not fully characterized)
- Copy suppression head activation → negative logit contribution: **solid** (the anti-copying $W_{OV}$ structure directly produces negative DLA on the attended token)
- Negative logit contribution → suppressed copying in output: **solid** (logit subtraction mechanically reduces the probability of the suppressed token)
- Interaction with induction/name-mover heads: **dashed** (the functional opposition is observed but the causal pathway of interaction is not directly patched)

Two solid edges (the output pathway is verified), two dashed edges (the input/detection pathway and the interaction with complementary mechanisms). The mechanism's *effect* is causally verified; its *activation trigger* is characterized observationally but not causally.

---

