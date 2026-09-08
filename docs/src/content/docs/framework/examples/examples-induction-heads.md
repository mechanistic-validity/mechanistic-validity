---
title: "Exploratory Lens Analysis: Induction Heads"
description: "The induction head mechanism (Olsson et al. 2022) evaluated through the five core lenses."
prev:
  link: /mechanistic-validity/framework/examples/
  label: Lens Applications
next:
  link: /mechanistic-validity/framework/examples/examples-copy-suppression/
  label: "Exploratory Lens Analysis: Copy Suppression"
---

# Exploratory Lens Analysis: Induction Heads

:::note[Disclaimer]
This page is an exploratory reading, not part of the paper. It applies the framework's five lenses to the claim as an illustration. For the audit as published, see [Case Studies: Induction Heads](/mechanistic-validity/framework/audits/induction/).
:::


[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) describe **induction heads** — a two-component mechanism (previous-token head + induction head) that implements in-context learning by attending to the token following the previous occurrence of the current token, then copying it to the output. This is arguably the strongest mechanistic claim in MI, combining structural clarity with broad replication.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Triangulated. **Capped by:** C6 (complementation validity), I3 (minimality), I10 (rescue reversibility), I12 (offset coupling).


| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C3 Convergent validity | — | Strong |
| Internal (Neuroscience) | I1/I11 Necessity + Onset coupling | I3/I10/I12 Minimality, Rescue, Offset | Triangulated |
| External (Pharmacology) | E4 Cross-model generalization | — | Strong |
| Measurement (Measurement Theory) | M6 Invariance | M4 Calibration | Strong |
| Interpretive (MI) | V3 Alternative level | — | Strong |

**Overall verdict: Triangulated.** Induction heads reach Triangulated on 14 Confirmed and 16 Partially confirmed criteria, with six Untested. Evidence converges across five families — eigenvalues, behavioral evaluators, ablation, scaling and training dynamics — with non-overlapping assumptions. Validated is blocked by four criteria the paper never attempts: complementation validity (C6), because the two named roles are ablated one at a time and never together; minimality (I3); rescue reversibility (I10); and offset coupling (I12).

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Ablation | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Path patching | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| $W_{OV}$ CopyScore / weight analysis | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |
| $W_{QK}$ compositional analysis | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |
| Training dynamics (loss curve phase transition) | [D04 CE Delta](/mechanistic-validity/framework/metrics/#d04) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "induction head" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The claim generates specific predictions: induction heads should attend to the token *after* the previous occurrence, not the occurrence itself. On non-repeated sequences, the attention pattern should be absent or diffuse. These predictions are concrete enough to be testable — if the attention pattern does not match, the claim is disconfirmed.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Pass.** An induction head requires a $W_{OV}$ matrix that copies the attended-to token's identity to the output. It also requires a compositional partner — a previous-token head in an earlier layer whose $W_{QK}$ enables the attending-to-next pattern. Both are verified: the $W_{OV}$ matrices have copying structure, and the compositional partner exists.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** The two behavioral evaluators are built to separate induction from repeated-token memorization, which is a real discriminant test. It is not extended to further neighboring constructs.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Not tested.** The account names two components, but no leave-one-out establishes that each earns its place; the authors themselves name the obstacle, which is that every ablation effect is marginal. This is one of four criteria capping the claim.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Pass.** The mechanism was discovered through behavioral analysis (in-context learning curves) and confirmed through structural analysis ($W_{OV}$ and $W_{QK}$ inspection) — two evidence families with non-overlapping assumptions. It replicates across model families.

### Key Distinctions

- **Confirmation vs corroboration:** Genuine corroboration, not mere confirmation. The mechanism was discovered via behavioral observation (in-context learning curves), then independently confirmed via weight-space analysis, then further corroborated by training-dynamic predictions (phase change). These methods have non-overlapping assumptions — a finding that satisfies all three simultaneously is unlikely to be an artifact of any single method's biases. This is the gold standard for MI corroboration.
- **Operationalism vs realism:** "Induction head" is operationally defined (CopyScore on $W_{OV}$ + characteristic attention pattern), which grounds the construct in measurable criteria. The operational definition is precise enough that independent researchers reliably identify the same heads.
- **Observable vs theoretical:** The gap between observable and theoretical is small here — the claimed mechanism (attend to token after previous occurrence, copy it forward) is directly visible in attention patterns and $W_{OV}$ structure without long inference chains.

### Nomological Network

The induction head construct connects to:

| Prediction the construct makes | How you test it | Confirmed? |
|---|---|---|
| Copies attended token to output | CopyScore on $W_{OV}$ matrix | Yes |
| Attends to token after previous occurrence | Attention pattern on repeated sequences | Yes |
| Composes with a previous-token head | Path patching the compositional circuit | Yes |
| Drives in-context learning on repeated text | Ablation: repeated vs. non-repeated sequences | Yes |
| Emerges as a training phase transition | Loss curve inflection point | Yes |
| Appears across model families | Cross-model comparison (GPT-2, GPT-3, etc.) | Yes |

Six independent predictions, six independent confirmations. A thick nomological network.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation, not just participation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Pass.** Ablating induction heads degrades in-context learning performance. The effect is specific to tasks involving repeated sequences — on non-repeated text, ablation has a smaller effect. This includes an implicit specificity control.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** No experiment reconstructs the copying behavior from induction heads alone. The two nearest things are both weaker than sufficiency: Argument 2 shows that the architectural capacity for induction suffices to produce in-context learning in a one-layer model that previously had none, which is sufficiency of the architecture rather than of the heads, and Scherlis's replication substitutes an idealized induction pattern and recovers most of the head's loss contribution — a comment on the paper rather than an experiment in it.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Partial.** Every head is ablated, so off-target effects are measured throughout, and the effect concentrates on repetition-bearing sequences. The crossed design that would make this a dissociation is a separate criterion (I6) and was not run at origin.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Three exogenous confounds are checked, including scheduled hyperparameters. What is not addressed is a shared latent cause that would produce heads and capability together.

### Key Distinctions

- **Single vs double dissociation:** Partial double dissociation is present. Ablating induction heads impairs in-context copying but not non-repetition tasks (forward direction). The converse — other heads impairing non-repetition tasks without affecting copying — is implicit in the specificity analysis but not formally reported as a double-dissociation design. The evidence is stronger than single dissociation but stops short of a textbook double dissociation.
- **Lesion vs stimulation:** Both are present. Ablation (lesion) shows necessity; the compositional mechanism itself — previous-token head feeding positional information to the induction head — constitutes a demonstrated sufficiency pathway (stimulation-equivalent). Path patching directly demonstrates that activating this specific path is sufficient to restore behavior, which is the functional equivalent of electrical stimulation in neuroscience.
- **Structural vs functional connectivity:** Both demonstrated. Structural connectivity is shown via $W_{QK}$ and $W_{OV}$ weight analysis (the heads are wired to compose). Functional connectivity is shown via path patching (they actually communicate during inference on relevant inputs). The alignment of structural and functional evidence is a key reason this mechanism achieves Triangulated status.

### Dissociation Matrix

|  | In-context copying task | Non-repetition tasks |
|---|---|---|
| Ablate induction heads | **↓↓ (strong)** | No significant effect |
| Ablate non-induction heads | No significant effect on copying | ? (not formally tested) |

The forward dissociation is demonstrated: ablating induction heads selectively impairs in-context copying. The partial converse (induction head ablation does NOT impair non-repetition tasks) is also established. The remaining cell (other heads' effect on non-repetition tasks) is implicit but not formally reported as a double-dissociation design.

---

## Pharmacology Lens — External Validity

*Does intervening on induction heads produce the expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** One intervention primitive at origin: pattern-preserving zero ablation of a head. A second intervention family that agrees with it has not been run.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** Measurement is graded and intervention is not. Both evaluators return continuous scores on stated ranges, footnote 5 says the construct is a continuum rather than a threshold, and the in-context learning score is tracked at every snapshot. The intervention is a zero vector substituted for one head: binary, one head at a time, with no interpolation between ablated and intact.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Strong.** The mechanism operates on arbitrary repeated sequences — not just specific token types, syntactic structures, or prompt templates. This is robustness by scope: the mechanism is defined over a broad input class.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Pass.** Identified in GPT-2, GPT-3-scale models, and other architectures. Structurally analogous heads appear wherever attention + residual stream composition is available.

### Key Distinctions

- **Affinity vs efficacy:** Both clearly demonstrated. Induction heads show high attention scores on repeated sequences (affinity) AND ablating them degrades in-context learning (efficacy). In two-layer models, they account for ~95% of the in-context learning signal, making the efficacy evidence unusually strong. The gap between "active" and "causally important" is essentially closed for this mechanism.
- **The system compensates:** Multiple induction heads exist in larger models, making compensation likely, but it is not systematically tested. The paper documents that larger models have redundant induction heads, implicitly acknowledging receptor reserve, but does not measure how ablation of one head changes the activation of others. This is a minor gap given the otherwise strong evidence.
- **Naming requires criteria:** "Induction head" is well-operationalized with explicit criteria: CopyScore on $W_{OV}$ above threshold, characteristic attention pattern on repeated sequences, and composition with a previous-token head. This makes it one of the few MI constructs where the naming criteria are precise enough that independent researchers reliably identify the same heads — satisfying the pharmacological standard for receptor classification.

### Dose-Response Curve

The induction head mechanism shows a clear dose-response relationship:
- **Dose axis:** Number of repetitions / context length containing repeated patterns
- **Response axis:** Strength of in-context copying (probability assigned to the copied token)
- **Observed relationship:** More repetitions produce stronger copying — monotonic, graded
- **EC₅₀ equivalent:** In two-layer models, even a single repetition produces a strong signal (~95% attribution to induction heads)
- **Ceiling:** The mechanism saturates in small models (it accounts for nearly all ICL) but in larger models, other mechanisms contribute at higher "doses" of contextual information

The key pharmacological insight: this is one of the few MI mechanisms where the dose-response is naturally observable across the input distribution, rather than requiring artificial parametric sweeps.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Partial.** Later work repeats the measurements across seeds in two-layer models. The origin measurements themselves are never repeated, and no interval accompanies any of them.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Strong.** The measurement is invariant across model sizes — the same identification criteria work from small models to GPT-3-scale. This is measurement invariance in the measurement-theoretic sense: the metric generalizes.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** Every head in every ablated model is ablated at every snapshot, so an induction head's effect is read against the empirical distribution of all head effects in the same model — an implicit matched comparison at more than fifty thousand ablations. The one-layer model supplies a clean architectural negative control: no induction heads form and no in-context learning develops. What is missing is an explicit random-head or random-init control, and footnote 17 shows why one is needed.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Partial.** A known-positive exists and is used: the two-layer attention-only induction circuit was derived from weights in the previous paper, so the Appendix can ask whether the activation evaluators recover a mechanism already established independently, and they do. The smeared-key architecture is a designed positive of a different kind. Neither is a planted circuit of known strength, so nothing says how weak a real induction head could be and still be detected.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The behavioral calibration is implicit — heads identified by CopyScore do in fact drive in-context learning when tested causally. Formal calibration curves are not reported.

### Key Distinctions

- **Convergent vs discriminant validity:** Strong convergent validity: behavioral analysis (loss curves), structural analysis ($W_{OV}$ CopyScore), compositional analysis (path patching), and training dynamics (phase transition) all identify the same mechanism. Discriminant validity is also present — CopyScore cleanly separates induction heads from non-induction heads with a bimodal distribution, showing the measurement is specific to the construct.
- **Reliability vs validity:** Both are strong. The CopyScore identification is perfectly reliable (deterministic, reproducible across analyses) and valid (identified heads are causally confirmed via ablation). This closes the gap between measurement quality and construct quality.

### MTMM Matrix

| | CopyScore (induction) | Attention pattern (induction) | CopyScore (non-induction) | Attention pattern (non-induction) |
|---|---|---|---|---|
| **CopyScore (induction)** | — | High (convergent) | Low (discriminant) | Low (discriminant) |
| **Attention pattern (induction)** | High | — | Low (discriminant) | Low (discriminant) |
| **CopyScore (non-induction)** | Low | Low | — | ? |
| **Attention pattern (non-induction)** | Low | Low | ? | — |

The convergent diagonal (two methods, same construct) shows high agreement. The discriminant cells (same method, different construct) show clean separation via the bimodal CopyScore distribution. This is a well-behaved MTMM pattern — convergent correlations exceed discriminant correlations, confirming that the measurement captures the intended construct rather than method variance.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [algorithmic](/mechanistic-validity/framework/modes/algorithmic) level — it names a two-step computation (attend to token after previous occurrence → copy to output) and identifies the components that implement it.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Pass.** The evidence includes behavioral effects (ablation), structural signatures ($W_{OV}$, $W_{QK}$), and compositional analysis (path patching). This multi-modal evidence supports an algorithmic-level claim.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** One alternative at a competing level is addressed — basic copying heads — and it is handled by argument rather than test. The paper's own higher-level redescription, in-context nearest neighbor, is offered as compatible with the copying account rather than as a rival, and two candidate relations between the levels are proposed. Neither is tested and the paper does not claim otherwise.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Pass.** The claim is "general-purpose in-context copying on repeated sequences" — and the evidence covers this full scope. The authors do not overclaim task-specific function.

### Key Distinctions

- **Description vs explanation:** The induction head narrative is a genuine explanation — it specifies a two-step algorithm and explains *why* each component is needed (previous-token head provides positional shift, induction head uses it to attend and copy). This is stronger than merely describing which components are active.
- **Component identity vs component role:** The role labels ("previous-token head," "induction head") are well-supported by structural evidence ($W_{QK}$ for position shift, $W_{OV}$ for copying). This is one case where the role claim has independent structural backing beyond behavioral observation.
- **Faithfulness vs understanding:** Both axes are strong. The mechanism is faithful (ablation confirms it matters) AND understood (the compositional algorithm is specified). The combination is rare in MI.

### Evidence Convergence Map

- **Implementational → Interpretation:** Strong. Ablation identifies specific heads; $W_{OV}$ CopyScore confirms structural roles; path patching isolates the compositional path. Multiple implementational sub-modes converge.
- **Algorithmic → Interpretation:** Strong. "Attend to token after previous occurrence → copy" is a specified two-step process with layer ordering that matches the narrative. The algorithm is simple enough that alternatives are constrained.
- **Computational → Interpretation:** Strong. The computational description (in-context learning on repeated sequences) matches the full scope of evidence. No overclaiming.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ | — | ∅ | ∅ | ∅ |
| Path patching | ✓ | ✓ | — | ✓ | — |
| Weight analysis | — | — | ✓ | — | — |
| Training dynamics | — | — | — | ✓ (emergence) | — |

More cells filled than typical MI studies. The path-patching row provides both necessity and sufficiency plus algorithmic evidence (isolating the compositional path). Weight analysis provides representational confirmation. Training dynamics provide independent algorithmic evidence (the mechanism emerges as a phase transition, confirming it is a coherent unit).

### Causal Sufficiency Graph

- Previous-token head → induction head: **solid** (path patching confirms information flow along this specific compositional path)
- Induction head → output: **solid** (DLA and ablation directly measure the causal contribution to output logits)
- Input → previous-token head: **solid** (the head attends to the previous position by design — its $W_{QK}$ encodes this pattern)
- Full path (input → prev-token → induction → output): **solid** (the complete causal chain is verified end-to-end via path patching)

All edges in the causal graph are solid — no dashed/inferred links. This is the only MI circuit where the full input-to-output causal chain is verified without gaps. The simplicity of the mechanism (two components, one path) makes complete causal verification tractable.
