---
title: "Case Study: Greater-Than Circuit"
description: "The Greater-Than circuit (Hanna et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Greater-Than Circuit

[Hanna et al. (2023)](https://arxiv.org/abs/2305.00586) identify attention heads in GPT-2 Small that implement the **Greater-Than task** — given "The war lasted from 1732 to 17," the model must predict a two-digit year suffix greater than 32. The claimed mechanism involves "successor heads" whose $W_{OV}$ matrices encode ordinal relationships between year tokens.

This is a structural-level claim with algorithmic aspects: it names specific weight-space signatures ($W_{OV}$ encodes ordinal structure) and a computation (compare magnitude, suppress smaller years, boost larger ones).

## Composite Verdict

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C2 Structural plausibility | C5 Convergent validity | Strong |
| Internal (Neuroscience) | I1 Necessity | I5 Confound control | Causally suggestive |
| External (Pharmacology) | E4 Effect magnitude | E1/E6 Reach + Cross-arch | Partial |
| Measurement (Measurement Theory) | M3/M4 Separation + Sensitivity | M1/M5 Reliability + Calibration | Partial |
| Interpretive (MI) | V2 Level-evidence match | V4 Alternative exclusion | Strong |

**Overall verdict: Causally suggestive, approaching Mechanistically supported.** The Greater-Than circuit's distinguishing strength is structural plausibility (C2) — the $W_{OV}$ ordering evidence is among the most precise weight-space characterizations published. This makes it a model case for how structural evidence can support an algorithmic claim. Its primary gaps are the same as most circuits: single model, single ablation method, no formal double dissociation. The honest scope investigation (generalization to other ordinal tasks) is a genuine strength that most papers lack.

---

## Philosophy of Science Lens — Construct Validity

*Is "the Greater-Than circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim makes a concrete structural prediction: the $W_{OV}$ matrices of successor heads should encode monotonic ordering over year suffixes. Specifically, for a proposed successor head $h$:

$$\text{effect}(y_1, y_2) = e_{y_2}^\top \, W_U \, W_{OV}^{(h)} \, W_E \, e_{y_1}$$

should be positive when $y_2 > y_1$ and negative when $y_2 < y_1$. A head labeled "successor" whose $W_{OV}$ shows no such ordering would be disconfirmed. This is unusually precise falsifiability — the prediction is quantitative and structural.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Strong pass.** This is the paper's primary contribution. Hanna et al. verify that the $W_{OV}$ matrices of their proposed heads encode a monotonic ordering over two-digit year suffixes. The structural signature directly matches the claimed computational role — this is among the strongest structural plausibility demonstrations in published MI.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Partial.** The circuit is evaluated on the Greater-Than task specifically. Some cross-task evaluation exists — the authors test whether the circuit generalizes to other ordinal comparisons (months, numbers outside the year range). The circuit partially generalizes, suggesting it captures ordinal structure more broadly than just year comparison. This is an honest scope expansion, not a specificity failure.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Pass.** The identified heads are few (a small subset of attention heads) and each contributes measurably. No systematic redundancy is reported.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** The claim is primarily established through weight-space analysis ($W_{OV}$ inspection) and behavioral evidence (ablation). These are somewhat independent methods (one is static structural analysis, the other is dynamic causal intervention). However, a third fully independent method (e.g., probing, or automated circuit discovery via EAP) has not been applied.

### Key Distinctions

- **Confirmation vs corroboration:** The cross-task generalization (months, arbitrary numbers) provides genuine corroboration rather than mere confirmation — the Fourier ordering hypothesis was not designed to predict this generalization, so finding it strengthens the claim beyond the original test set.
- **Operationalism vs realism:** "Successor heads" is a descriptive name grounded in operational criteria (monotonic $W_{OV}$ ordering over sequential tokens). The naming is honest — it describes the observed structural property rather than asserting a broader theoretical role.
- **Observable vs theoretical:** The $W_{OV}$ ordering structure is directly observable in the weights — no long inference chain is required. This makes the construct unusually close to the raw observations, reducing underdetermination.

### Nomological Network

The Greater-Than circuit connects to:
- **Weight structure** — $W_{OV}$ of successor heads shows monotonic year ordering (structural, confirmed)
- **Behavioral prediction** — ablation degrades ordinal comparison accuracy (causal, confirmed)
- **Cross-task generalization** — circuit extends to months and other ordinal comparisons (scope, confirmed)
- **Quantitative prediction** — effect magnitude follows the $W_{OV}$ ordering relationship (structural-behavioral link, confirmed)
- **Cross-model prediction** — do other GPT-2 scales use the same structure? (untested)
- **Training dynamics** — does the ordering structure emerge at a specific training phase? (untested)
- **Naturalistic robustness** — does the circuit work in diverse syntactic contexts? (untested)

Four nodes confirmed, three unconnected. The confirmed nodes are stronger than average (the quantitative structural prediction is unusually precise), but the generalization edges remain open.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Ablating the identified successor heads degrades Greater-Than performance. The effect size is substantial — the model loses its ability to preferentially predict years greater than the reference. Random-component baselines are included.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Partial.** The paper demonstrates that the $W_{OV}$ structure is consistent with the claimed computation, but a full circuit isolation test (ablate everything outside the circuit, measure whether Greater-Than still works) is not the primary methodology. Sufficiency is partially established through the structural argument: if the $W_{OV}$ matrices encode the ordering, and the heads are active, the computation follows. But this is a structural sufficiency argument, not a causal one.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Partial.** The generalization to other ordinal tasks (months, arbitrary numbers) is informative — it suggests the circuit is specific to *ordinal comparison* rather than just year tokens. But a formal double dissociation (ablate Greater-Than circuit → measure IOI; ablate IOI circuit → measure Greater-Than) is not reported.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** The mechanism is demonstrated on GPT-2 Small. Cross-model and cross-seed replication are not reported. The generalization across ordinal tasks (years, months, numbers) provides some within-model consistency.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not tested.** The ablation method is not varied. Multi-method comparison (mean vs. resample ablation) is not reported.

### Key Distinctions

- **Single vs double dissociation:** Only single dissociation is established: ablating successor heads degrades Greater-Than. A double dissociation (ablating other circuits does *not* degrade Greater-Than, and ablating successor heads does *not* degrade other tasks) is partially explored via cross-task generalization but never formally demonstrated.
- **Localization vs distributed:** The circuit is relatively localized (a small set of attention heads), which is a strength for interpretability. However, whether these heads are the complete mechanism or one node in a distributed computation is not fully resolved — the partial sufficiency result leaves this open.
- **Lesion vs stimulation:** Only the lesion direction (ablation) is reported. No stimulation experiment (amplifying successor head activations to steer the model toward predicting larger/smaller years) is demonstrated.

### Dissociation Matrix

|  | Greater-Than task | IOI task | SVA task | General LM |
|---|---|---|---|---|
| Ablate successor heads | **↓↓ (strong)** | ? | ? | ? |
| Ablate IOI circuit | ? | ? | ? | ? |
| Ablate SVA circuit | ? | ? | ? | ? |

One cell filled. The on-diagonal entry is strong, but without off-diagonal measurements we cannot distinguish "ordinal-comparison-specific mechanism" from "general numerical processing bottleneck." The cross-task generalization to months/numbers suggests the mechanism is ordinal-specific rather than year-specific, but formal dissociation testing is absent.

---

## Pharmacology Lens — External Validity

*Does intervening on the circuit produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Not tested.** Can you steer the model toward predicting larger or smaller years by manipulating the successor heads' activations? This would test whether the circuit is genuinely manipulable beyond ablation.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Implicit.** The $W_{OV}$ structure implies a graded response — years further from the reference should receive stronger suppression/boosting. This is structurally predicted but not directly measured as a parametric dose-response.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Partial.** The cross-task generalization to other ordinal comparisons suggests the intervention would not be purely selective for year comparison. This is an honest scope description rather than a selectivity failure.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Strong on-task.** The circuit accounts for a large portion of the model's ordinal comparison ability on the tested prompts.

**[E5 — Robustness:](/framework/criteria/external/robustness) Partial.** Works across different year ranges and extends to other ordinal tasks. Not tested on naturalistic text (where Greater-Than appears in diverse syntactic contexts).

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** GPT-2 Small only.

### Key Distinctions

- **Affinity vs efficacy:** The $W_{OV}$ ordering demonstrates strong structural affinity (the mechanism has the capacity to perform ordinal comparison), and the ablation results confirm efficacy (it actually does so in practice). Both sides are present, which is unusual.
- **The metric is part of the finding:** The probability-difference metric over year tokens is well-chosen and specific to the claimed computation. Unlike metrics that could be inflated by general model degradation, this one tracks the precise ordinal relationship the circuit is supposed to implement.
- **Naming requires criteria:** "Successor heads" is a descriptive name grounded in operational criteria (monotonic $W_{OV}$ ordering over sequential tokens). The naming is honest — it describes the observed structural property rather than asserting a broader theoretical role.

### Dose-Response Curve

The Greater-Than circuit's dose-response is structurally predicted but empirically incomplete:
- **Structural prediction:** The $W_{OV}$ ordering implies that years further from the reference should receive stronger boosting/suppression — a monotonic dose-response
- **Discrete points:** Ablating individual heads produces graded effects (some heads contribute more than others)
- **Missing:** No parametric sweep of intervention strength (partial ablation at varying magnitudes)
- **Missing:** No off-target measurement at any dose level
- **Missing:** No threshold characterization (at what intervention strength does ordinal comparison begin to degrade?)

The structural evidence *predicts* a smooth dose-response curve, but the empirical evidence consists only of binary on/off ablation points. The prediction is unusually well-grounded (because the $W_{OV}$ structure is precisely characterized), but remains unverified as a parametric relationship.

---

## Measurement Theory Lens — Measurement Validity

*Are the instruments reliable?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Not reported.** No bootstrap confidence intervals on effect sizes or structural measurements.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Partial.** The measurement generalizes across year ranges (not just 1700s). This provides some invariance evidence. But invariance across different prompt formats or naturalistic contexts is not tested.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** The $W_{OV}$ ordering structure is clearly present in successor heads and absent in non-successor heads. The signal-to-noise separation is clean.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Good.** The monotonic ordering metric provides a clear threshold for identifying successor heads — heads either show the pattern or they don't. The measurement is sensitive to the structural signature.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.** No gold-standard comparison.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Good.** The analysis covers both structural ($W_{OV}$ inspection) and behavioral (logit prediction) aspects of the claim, providing good construct coverage from multiple angles.

### Key Distinctions

- **Sensitivity vs specificity:** The $W_{OV}$ monotonicity metric has high sensitivity (it reliably detects successor heads) and good specificity (non-successor heads do not show the pattern). This clean bimodal separation is a measurement strength that many circuit studies lack.
- **Convergent vs discriminant validity:** Convergent validity is partial (weight-space and behavioral methods agree), but discriminant validity is not tested — we do not know if the $W_{OV}$ ordering metric would incorrectly flag heads that are not doing ordinal comparison.

### MTMM Matrix

| | $W_{OV}$ ordering (successor) | Ablation effect (successor) | $W_{OV}$ ordering (non-successor) | Ablation effect (non-successor) |
|---|---|---|---|---|
| **$W_{OV}$ ordering (successor)** | — | High (convergent) | Low (discriminant) | Low (discriminant) |
| **Ablation effect (successor)** | High | — | Low | Low |
| **$W_{OV}$ ordering (non-successor)** | Low | Low | — | ? |
| **Ablation effect (non-successor)** | Low | Low | ? | — |

The convergent cells (two methods agree on the same heads) are strong — heads identified by $W_{OV}$ structure are the same ones whose ablation degrades ordinal comparison. The discriminant cells show clean separation (non-successor heads lack both the structural signature and the ablation effect). However, no cross-task discriminant comparison exists: we do not know whether these methods would produce the same agreement pattern for an unrelated circuit.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [structural](/framework/modes/structural) level with algorithmic aspects — it names specific weight-space signatures and the computation they implement.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Strong.** The evidence is primarily structural ($W_{OV}$ analysis), which directly supports a structural-level claim. The match between evidence type and claim level is unusually tight.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** The story is mechanistically precise: successor heads encode year-token ordering in their $W_{OV}$ matrices; when the model processes "from 17XX to 17," these heads suppress years ≤ XX and boost years > XX. The narrative directly connects structure to behavior.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Partial.** The structural evidence constrains alternatives — it is hard to explain why $W_{OV}$ encodes monotonic year ordering if not for ordinal comparison. But whether this is the *complete* mechanism (vs. one component of a larger distributed computation) is not fully addressed.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Good.** The paper tests scope boundaries (does it generalize to months? to arbitrary numbers?) and reports honestly where the mechanism extends and where it does not.

### Key Distinctions

- **Description vs explanation:** The narrative bridges description and explanation — it not only identifies which heads are involved but specifies the structural basis ($W_{OV}$ ordering) that explains *why* those heads perform ordinal comparison. This is stronger than pure behavioral description.
- **Component identity vs component role:** "Successor head" is well-grounded: the role label (ordinal comparison) is backed by structural evidence (monotonic $W_{OV}$ ordering), not merely behavioral observation. The structural signature independently confirms the claimed role.
- **Faithfulness vs understanding:** Both are present — the circuit is faithful (ablation confirms importance) and understood (the $W_{OV}$ structure specifies the mechanism). The understanding axis is particularly strong because the structural characterization is quantitative.

### Evidence Convergence Map

- **Implementational → Interpretation:** Strong. $W_{OV}$ weight analysis directly reveals the ordinal ordering structure. Ablation confirms these specific heads are causally important.
- **Algorithmic → Interpretation:** Moderate. The "suppress smaller, boost larger" algorithm is consistent with the structural evidence, but whether this is the complete algorithm or one step in a multi-stage process is not fully resolved.
- **Computational → Interpretation:** Moderate. The computational description (ordinal comparison) matches evidence, but scope boundaries (where does ordinal comparison end and general numerical processing begin?) are partially characterized.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ | — | ∅ | ∅ | ∅ |
| Weight analysis | — | — | ✓ (strong) | ✓ (partial) | — |
| Cross-task testing | — | — | — | — | ✓ (scope) |
| Steering | — | — | — | — | — |

The weight-analysis row is the distinguishing feature: it provides both representational evidence (the ordering exists in weights) and partial algorithmic evidence (the structure specifies the computation). The ablation row provides necessity. Cross-task testing addresses computational scope. Steering is entirely absent.

### Causal Sufficiency Graph

- Input (year tokens) → successor head attention: **solid** (the heads attend to year tokens in the prompt)
- Successor head $W_{OV}$ → output logit modulation: **solid** (the ordering structure directly modulates which year tokens are boosted/suppressed in output logits)
- Full path (input → successor heads → ordinal output): **dashed** (the end-to-end causal path is established via ablation, but whether the circuit is the *complete* mechanism or one contributor to a distributed computation is not fully resolved)
- Interaction with other components: **unknown** (how successor heads interact with other attention heads or MLP layers is not characterized)

Two solid edges (individual links verified), one dashed edge (end-to-end completeness uncertain), one unknown interaction. The structural evidence makes each individual link strong, but the circuit's completeness as a standalone mechanism is the remaining question.

---

