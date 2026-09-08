---
title: "Case Study: Copy Suppression"
description: "The copy suppression mechanism (McDougall et al. 2023) evaluated through the five core lenses."
---

# Case Study: Copy Suppression

[McDougall et al. (2023)](https://arxiv.org/abs/2310.04625) identify **copy suppression heads** in GPT-2 Small — attention heads that detect when the model is about to copy a token and actively suppress that copying. The mechanism functions as an anti-induction circuit: where induction heads promote copying repeated tokens, copy suppression heads inhibit it, preventing the model from naively repeating tokens that appear in context but are not the correct next prediction.

This is unusual because it is defined by what it *prevents* rather than what it produces — a negative-effect mechanism.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I6 (double dissociation).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C2 Structural plausibility | C3 Convergent | Partial–Strong |
| Internal | I4 Specificity | I7 Confound control | Causally suggestive |
| External | E2/E3 Prompt + Cross-task generalization | E1 Intervention reach | Partial |
| Measurement | M2 Baseline separation | M1 Reliability | Partial |
| Interpretive | V2 Level-evidence match | V3 Alternative level | Strong |

**Overall verdict: Mechanistically Supported.** Copy suppression is notable for the direction of its specificity result (I4) — ablation produces a specific error type rather than general degradation — though the discrimination is partial, since either half of the instrument alone clears 50% recovered KL for many layer 9–11 heads. The capping criterion is I6 (double dissociation): the materials are present in Table 2 and never assembled into a crossed design.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Ablation (mean ablation) | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Direct logit attribution (DLA) | [D02 Logit-Diff Recovery](/mechanistic-validity/framework/metrics/#d02) | Behavioral |
| $W_{OV}$ decomposition (anti-copying structure) | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "copy suppression" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The claim predicts: (1) these heads should produce negative direct logit attribution on tokens that are about to be copied, (2) ablating them should *increase* the probability of incorrect token repetition. Both are testable and concrete.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Pass.** The $W_{OV}$ matrices of copy suppression heads show anti-copying structure — they project negatively onto the tokens they attend to, effectively subtracting those tokens from the output logits. This is the structural mirror of the positive copying structure in name-mover/induction heads.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** Copy suppression is not task-specific — it operates across any context where token repetition is likely but incorrect. This is an honest scope claim (like induction heads), but the boundary of when suppression activates versus when copying is appropriate is not precisely characterized.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** The object under study is a single attention head, so minimality can only be tested inside the explanation rather than over a component set. Preserving the QK mechanism alone recovers 95.2% and the OV mechanism alone 81.1%, while preserving both and deleting everything else recovers 76.9% — each half is individually mild and their conjunction is what costs, which is evidence that both earn their place. There is no component set left to prune.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Confirmed.** Four instruments bear on the same claim and agree. Weight products give 84.70% and 95.72% with no intervention at all; hand-coding of dataset examples gives 80%; a structured ablation that deletes everything except the two weight mechanisms preserves 76.9% of the head's effect; and out-of-distribution behavior on repeated random tokens ranks the same head first. These differ in kind rather than in granularity — parameters, observation, causation, a different distribution — so their agreement is not arithmetic.

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
- **Cross-model prediction** — anti-copying scores correlate across GPT, Pythia and SoLU (phenomenon, confirmed); the mechanism itself is tested only in GPT-2 Small
- **Training dynamics** — does copy suppression emerge after induction heads? (untested)

Four nodes confirmed, three unconnected. The confirmed nodes establish a coherent negative-effect mechanism, but the boundaries of its activation and its developmental relationship to copying mechanisms remain unexplored.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Partial.** Necessity is established for a path and not for the head. Decomposing the head's effect into direct and indirect routes shows the direct path carries most of the loss change, which localizes the contribution cleanly. In absolute terms nothing depends on the head: its whole direct effect is a thousandth of the model's loss, the sign of its contribution is close to balanced across completions, and the same is true of most heads in its layers.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** The mechanism is demonstrated through its effect (suppressing logits), but a full isolation test (can these heads alone prevent copying when the rest of the model promotes it?) is not reported.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Partial.** Specificity is tested and comes out split, with both halves in the same appendix. CSPA applied to every layer 9–11 head recovers most for L10H7, which is the result the criterion asks for. The components of that instrument do not discriminate: the OV and QK ablations taken separately clear 50% recovered KL for many other heads, so what is specific to L10H7 is the conjunction rather than either mechanism.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Tied embeddings and LayerNorm are handled explicitly. Data confounds are untouched.

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

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** Four intervention forms are run, but all four are activation edits on one head, so they do not constitute independent intervention families.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** There is a genuine dose-response at the query side. Every other intervention in the paper is binary.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Confirmed.** This is the claim's distinguishing feature. The headline number is an average over OpenWebText, GPT-2's own pretraining distribution, rather than over a template set built for the hypothesis, and it is reported per percentile of effect size, with the preserved fraction highest where mean ablation is most destructive. Two further distributions are checked — IOI and repeated sequences of uniformly random tokens — the second sharing no structure with the first.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Partial.** Copy-suppression scores correlate head by head with anti-induction scores across GPT, Pythia and SoLU models, so the phenomenon replicates in three systems. The structured ablation that establishes the mechanism is never run outside GPT-2 Small.

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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Partial.** Spread is reported across data in 100 percentiles, but no interval accompanies any estimate. Cross-model consistency is reported and uneven: GPT-2 Medium recovers two of its three most negative heads, Pythia's copy suppression is weaker, and Stanford GPT-2 Small E's analogue attends to IO and S2 equally.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Partial.** Works across prompt types. Layer/position invariance not tested.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** Two baselines carry the argument and both are the right shape. Mean ablation of the direct effect is the denominator of the effect-explained ratio, so every percentage is already expressed against a null, and the same-matching QK circuit is a baseline chosen because tied embeddings make that rival live. What is missing is a null for 76.9% itself: Appendix J.3 shows the component ablations clear 50% for many heads, so the reference distribution the headline number needs is the one not constructed.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Not tested.** Every control in the paper is a known-negative — the same-matching baseline, the layer 9–11 head sweep, mean ablation — and each establishes what the measurement returns when copy suppression should be absent. None establishes what it returns when a mechanism of known strength is present, because none is planted and recovered. A head with half the mechanism would score some number, and nothing says which.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The validation metric is well chosen and its failure mode is stated by the authors. KL divergence has a meaningful zero, is linear in the log-probabilities the ablation produces, and Appendix J.1 names the case it will miss — a logit change too small to move a small probability, which costs loss and not KL. Calibration of the underlying effect thins out: the head's whole direct contribution is about a thousandth of the model's OpenWebText loss, and the same explanation scores 82% or 45% depending on which loss metric is chosen.

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** Algorithmic — names what the heads do (suppress copying) and how ($W_{OV}$ anti-copying).

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** Evidence and claim are matched on two of three axes. The mechanism claim is carried by weight-level evidence, the right level for a claim about what a matrix does, and the coverage claim by a distribution-level average, the right level for a claim about a training distribution. The unmatched axis is the one the authors name: when and how much copy suppression fires. Both idealized approximations shift real attention substantially, and the query-side direction that matters most is perpendicular to the one the account uses.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** Could these heads be doing something else that incidentally suppresses copying? The structural evidence (anti-copying $W_{OV}$) constrains alternatives, but the possibility that "suppression" is a side effect of a more general computation is not fully excluded.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Good.** Presented as general-purpose anti-copying, which matches the evidence scope.

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

