---
title: "Case Study: Induction Heads"
description: "The induction head mechanism (Olsson et al. 2022) evaluated through all five validity lenses."
---

# Case Study: Induction Heads

[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) describe **induction heads** — a two-component mechanism (previous-token head + induction head) that implements in-context learning by attending to the token following the previous occurrence of the current token, then copying it to the output. This is arguably the strongest mechanistic claim in MI, combining structural clarity with broad replication.

## Composite Verdict

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C5 Convergent validity | — | Strong |
| Internal (Neuroscience) | I4 Consistency | I5 Confound control | Mechanistically supported |
| External (Pharmacology) | E6 Cross-architecture | — | Strong |
| Measurement (Measurement Theory) | M2 Invariance | M5 Calibration | Strong |
| Interpretive (MI) | V4 Alternative exclusion | — | Strong |

**Overall verdict: Triangulated.** Induction heads pass all five criteria for construct validity, four of five for internal validity, all six for external validity, and all five for interpretive validity. Evidence converges across multiple independent lenses with non-overlapping assumptions. This is the strongest mechanistic claim in MI — it reaches Triangulated status primarily because the mechanism is simple, general-purpose, and independently verifiable from multiple angles. The single remaining gap (I5 — systematic multi-method comparison) is a reporting gap, not an evidence gap, since path patching partially addresses confounds.

---

## Philosophy of Science Lens — Construct Validity

*Is "induction head" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim generates specific predictions: induction heads should attend to the token *after* the previous occurrence, not the occurrence itself. On non-repeated sequences, the attention pattern should be absent or diffuse. These predictions are concrete enough to be testable — if the attention pattern does not match, the claim is disconfirmed.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Pass.** An induction head requires a $W_{OV}$ matrix that copies the attended-to token's identity to the output. It also requires a compositional partner — a previous-token head in an earlier layer whose $W_{QK}$ enables the attending-to-next pattern. Both are verified: the $W_{OV}$ matrices have copying structure, and the compositional partner exists.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) N/A (honest scope).** Induction heads fire on any repeated sequence. This is not a specificity failure — it is an honest scope claim. The construct is described as a general-purpose mechanism, not a task-specific circuit.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Pass.** The mechanism requires exactly two components: the previous-token head and the induction head. Removing either breaks the mechanism. No redundancy.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Pass.** The mechanism was discovered through behavioral analysis (in-context learning curves) and confirmed through structural analysis ($W_{OV}$ and $W_{QK}$ inspection) — two evidence families with non-overlapping assumptions. It replicates across model families.

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

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Ablating induction heads degrades in-context learning performance. The effect is specific to tasks involving repeated sequences — on non-repeated text, ablation has a smaller effect. This includes an implicit specificity control.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Pass (path-level).** Path patching confirms that patching only the output of the previous-token head through the induction head is sufficient to restore the behavior. This is sufficiency at the *path* level — stronger than component-level sufficiency because it isolates the compositional mechanism.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Pass (honest scope).** The double dissociation holds: ablating induction heads impairs in-context learning but not tasks that do not involve repetition. The mechanism is specific to repetition-based in-context learning.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Pass (cross-scale).** The mechanism is identified across multiple model sizes (small to GPT-3-scale) and across model families. The training dynamics signature — a phase change in loss curves — replicates across independent training runs. Unusually strong consistency.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not directly tested.** Path patching partially addresses confounds by isolating the compositional path rather than ablating entire components. A systematic multi-method comparison has not been reported.

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

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Implicit pass.** The mechanism operates naturally across all repeated-sequence contexts — it does not need to be artificially activated. Any repeated sequence triggers the mechanism, demonstrating that it has broad reach within its scope.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Pass.** The in-context learning effect strengthens with more repetitions and longer contexts. The mechanism shows the expected dose-response: more signal (more repetitions) produces more effect (stronger copying).

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Pass.** Ablating induction heads selectively impairs in-context copying without catastrophically degrading other model capabilities. The intervention is targeted.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Large.** In two-layer attention-only models, induction heads account for nearly all of the in-context learning signal ($F \approx 0.95$). In larger models, the effect is distributed but induction heads remain major contributors.

**[E5 — Robustness:](/framework/criteria/external/robustness) Strong.** The mechanism operates on arbitrary repeated sequences — not just specific token types, syntactic structures, or prompt templates. This is robustness by scope: the mechanism is defined over a broad input class.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Pass.** Identified in GPT-2, GPT-3-scale models, and other architectures. Structurally analogous heads appear wherever attention + residual stream composition is available.

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

*Are the instruments reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Strong (implicit).** The induction head identification criterion (CopyScore on $W_{OV}$ + attention pattern on repeated sequences) produces consistent results across analyses. Different researchers examining the same model identify the same heads.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Strong.** The measurement is invariant across model sizes — the same identification criteria work from small models to GPT-3-scale. This is measurement invariance in the measurement-theoretic sense: the instrument generalizes.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** Non-induction heads clearly fail the CopyScore criterion. The measurement cleanly separates heads that implement the mechanism from heads that do not.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Good.** CopyScore has a natural threshold that separates induction from non-induction heads. The bimodal distribution (most heads score low, induction heads score high) makes the measurement sensitive without requiring arbitrary threshold choices.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Partial.** The behavioral calibration is implicit — heads identified by CopyScore do in fact drive in-context learning when tested causally. Formal calibration curves are not reported.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Good.** The identification uses multiple signals (CopyScore, attention pattern, training dynamics signature), providing good coverage of the construct from different angles.

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

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [algorithmic](/framework/modes/algorithmic) level — it names a two-step computation (attend to token after previous occurrence → copy to output) and identifies the components that implement it.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Pass.** The evidence includes behavioral effects (ablation), structural signatures ($W_{OV}$, $W_{QK}$), and compositional analysis (path patching). This multi-modal evidence supports an algorithmic-level claim.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** The two-head composition story is mechanistically precise: the previous-token head shifts the key one position back, enabling the induction head to attend to the token *after* the match and copy it forward. The story explains *why* both heads are needed and what each contributes.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Strong.** The mechanism is simple enough (two components, one compositional path) that alternative explanations are constrained. You could dispute whether "induction head" is the right name, but it is hard to dispute the mechanical account of what the heads do.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Pass.** The claim is "general-purpose in-context copying on repeated sequences" — and the evidence covers this full scope. The authors do not overclaim task-specific function.

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

---

