---
title: "Case Study: IOI Circuit"
description: "The indirect object identification circuit (Wang et al. 2022) evaluated through all five validity lenses."
---

# Case Study: IOI Circuit

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) identify 26 attention heads in GPT-2 Small that form the **indirect object identification circuit** — a mechanism that detects duplicated names, suppresses them, and copies the remaining name to the output. This is the most thoroughly analyzed circuit in mechanistic interpretability.

Below, we evaluate this claim through each of the five validity lenses, applying the full criteria set.

## Composite Verdict

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C2 Structural plausibility | C3 Task specificity | Partial |
| Internal (Neuroscience) | I1/I2 Necessity + Sufficiency | I3/I5 Specificity + Confound | Causally suggestive |
| External (Pharmacology) | E4 Effect magnitude | E1/E6 Reach + Cross-arch | Weak |
| Measurement (Measurement Theory) | M3 Baseline separation | M1/M5 Reliability + Calibration | Partial |
| Interpretive (MI) | V3 Narrative coherence | V4 Alternative exclusion | Strong |

**Overall verdict: Causally suggestive, approaching Mechanistically supported.** The IOI circuit has strong necessity and sufficiency evidence (I1, I2), strong narrative coherence (V3), and confirmed structural plausibility (C2). It stalls short of Mechanistically supported because specificity (I3/C3), confound control (I5), measurement reliability (M1), and alternative exclusion (V4) are all untested or weak. This is the most thoroughly analyzed circuit in MI — the remaining gaps reflect the difficulty of the bar, not deficiencies of the paper.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Activation patching | [A02 Counterfactual DAS](/framework/metrics/causal/a02-counterfactual-das) | Causal |
| Path patching | [A02 Counterfactual DAS](/framework/metrics/causal/a02-counterfactual-das) | Causal |
| Mean ablation | [A01 Pearl SCM](/framework/metrics/causal/a01-scm-pearl) | Causal |
| Direct logit attribution (DLA) | [D02 Logit-Diff Recovery](/framework/metrics/behavioral/d02-logit-diff-recovery) | Behavioral |
| $W_{OV}$ / $W_{QK}$ decomposition | [B03 OV/QK Decomposition](/framework/metrics/structural/b03-ov-qk-decomposition) | Structural |
| Logit difference | [D02 Logit-Diff Recovery](/framework/metrics/behavioral/d02-logit-diff-recovery) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "the IOI circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Implicit pass.** The claim generates testable predictions — name-mover heads should have $W_{OV}$ matrices that copy names, S-inhibition heads should attend from the IO position to the S position. These were not pre-registered but are concrete enough that failure would disconfirm the claim. The label "name mover" would be falsified by a $W_{OV}$ that does not preferentially copy name tokens.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Pass.** The $W_{OV}$ matrices of name-mover heads (9.9, 9.6, 10.0) show copying structure — high singular values along name-token directions. S-inhibition heads (7.3, 7.9, 8.6) attend from the final position to the position of the repeated subject. The structural signatures match the claimed roles.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Not tested.** The IOI circuit is not evaluated on related tasks (subject-verb agreement, gendered pronouns, etc.). If the same 26 heads also rank highly for other syntactic tasks, the circuit may be capturing general syntactic processing rather than task-specific IOI computation.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Complicated.** The 26-head circuit includes backup name-mover heads that are individually unnecessary — the primary name movers suffice. The backups activate compensatorily when primaries are ablated, raising the question of whether the circuit is over-inclusive under normal operation. Whether backups are "in the circuit" depends on the definition of minimality.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** The circuit was discovered primarily through activation patching and direct logit attribution — methods that share interventionist assumptions. Weight-space analysis confirms structural plausibility (C2), providing partial convergent evidence from a different method family. However, a systematic Jaccard comparison between activation-based and weight-based circuit definitions has not been reported.

### Key Distinctions

- **Confirmation vs corroboration:** The circuit was found by activation patching and evaluated by activation patching — the same methodological family. This is confirmation rather than genuine corroboration, which would require an independent method (e.g., weight-space identification or training dynamics). Partial weight-space analysis exists but is not used as the primary identification tool.
- **Underdetermination:** Meloux et al. find alternative faithful circuits with comparable faithfulness but different head membership, directly demonstrating that the behavioral data does not uniquely determine which circuit implements IOI. The "detect-inhibit-copy" algorithm may be multiply realizable within the same model.
- **Operationalism vs realism:** "Name-mover" is a theoretical label applied to heads whose observable behavior is high DLA for name tokens. The label implies a richer functional role than the evidence strictly supports — a head that copies names under specific distributional assumptions may not be a general "name mover" in the realist sense.

### Nomological Network

The IOI circuit connects to:
- **Attention pattern** — S-inhibition heads attend to the repeated subject position (observable, confirmed)
- **Weight structure** — $W_{OV}$ of name-mover heads shows copying geometry (structural, confirmed)
- **Behavioral prediction** — ablation degrades IOI logit difference (causal, confirmed)
- **Template generalization** — ABBA/BABA variants activate the same circuit (scope, confirmed)
- **Cross-task prediction** — does the circuit fire on related syntactic tasks? (untested)
- **Training dynamics** — does the circuit emerge at a specific phase? (untested)
- **Cross-model prediction** — do other GPT-2 scales use the same structure? (untested)

Four nodes confirmed, three unconnected. A moderately thick network — strong, but with clear gaps at the generalization edges.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation, not just participation?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass (method-conditional).** The authors ablate each head individually (mean ablation) and measure the change in logit difference. The name-mover heads each produce large effects when ablated — removing head 9.9 alone drops the logit difference by approximately 1.2 points. An equal-size random-component baseline is included. But [Miller et al. (2024)](https://arxiv.org/abs/2407.08734) show that the same circuit's faithfulness varies from 87% under mean ablation to below 50% under other methods. The necessity claim is conditional on the ablation method.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Pass (method-conditional).** Wang et al. test sufficiency by running the model with everything *outside* the 26-head circuit mean-ablated. The circuit alone recovers 87% of the full model's logit difference. This is the strongest form of sufficiency — isolation rather than just restoration. However, this number is also ablation-method-dependent.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Not tested.** The authors do not systematically test whether the IOI circuit degrades unrelated tasks when ablated. A formal double-dissociation test — ablate the IOI circuit and measure SVA, ablate the SVA circuit and measure IOI — has not been reported.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** The circuit generalizes across name substitutions and ABBA/BABA template variants. Cross-seed and cross-checkpoint consistency are not tested. Cross-model consistency (does GPT-2 Medium use the same circuit?) is not evaluated.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not tested.** All analysis uses mean ablation. Resample ablation would control for distributional disruption, but the full circuit analysis was not replicated under alternative methods.

### Key Distinctions

- **Single vs double dissociation:** Only single dissociation is performed — ablating the IOI circuit impairs IOI. The converse (ablating a different circuit and showing IOI remains intact) is never tested. Without double dissociation, the circuit could be a general-purpose module whose removal impairs many tasks.
- **Lesion vs stimulation:** The paper uses only lesion-style evidence (mean ablation). No stimulation experiment (amplifying name-mover signals or steering the circuit to produce a specific name) is reported, leaving open whether the circuit is merely necessary infrastructure or a genuinely steerable mechanism.

### Dissociation Matrix

|  | IOI task | SVA task | Factual recall | Pronoun resolution |
|---|---|---|---|---|
| Ablate IOI circuit | **↓↓ (87%)** | ? | ? | ? |
| Ablate SVA circuit | ? | ? | ? | ? |
| Ablate factual circuit | ? | ? | ? | ? |

One cell filled out of twelve. The diagonal entry is strong ($D_{11}$ = 87% drop), but without off-diagonal measurements we cannot distinguish "IOI-specific mechanism" from "general syntactic bottleneck." The matrix makes visible exactly what's missing: every `?` is an untested double-dissociation leg.

---

## Pharmacology Lens — External Validity

*Does intervening on the circuit produce the expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Not tested.** No activation steering experiments have been reported for the IOI circuit. Can you steer the model toward or away from IOI behavior by injecting signal along the circuit's principal directions? This would test whether the circuit is merely descriptive or genuinely manipulable.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Partial.** Ablating individual heads produces graded effects — removing head 9.9 has a larger effect than removing head 10.7. But a parametric dose-response (ablating at varying strengths, or patching at varying magnitudes) is not systematically reported.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Not tested.** Does intervening on the IOI circuit affect only IOI behavior, or does it produce off-target effects? If steering the name-mover heads also changes factual recall or pronoun resolution, the intervention is not selective — same gap as I3 specificity.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Strong on-task.** The circuit accounts for 87% of the logit difference, which is a large effect. This establishes that the circuit is a major contributor, not a marginal one.

**[E5 — Robustness:](/framework/criteria/external/robustness) Partial.** Name substitutions preserve the effect. But robustness across syntactic variations, sentence lengths, and naturalistic (non-template) prompts is not systematically tested.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** The analysis is restricted to GPT-2 Small. Whether the same circuit structure exists in GPT-2 Medium/Large, Pythia, or other architectures is unknown.

### Key Distinctions

- **Affinity vs efficacy:** The IOI circuit demonstrates both — heads are highly active on IOI prompts (affinity) AND ablation degrades performance (efficacy). The 87% faithfulness figure combines both into a single measurement.
- **The system compensates:** Backup name-movers activate compensatorily when primaries are ablated, directly demonstrating receptor reserve. This is one of the few MI results that explicitly documents compensation, though the compensation ceiling is not fully characterized.
- **The metric is part of the finding:** The circuit was discovered using logit difference under mean ablation and evaluated using the same metric under the same intervention. Miller et al. show faithfulness drops below 50% under alternative methods, confirming that the metric choice inflates apparent circuit quality.

### Dose-Response Curve

The IOI circuit's dose-response curve is mostly unknown. We have:
- **α = 0** (no intervention): full performance
- **α = 1** (complete mean ablation): 87% logit difference drop
- **Individual heads**: removing 9.9 drops ~1.2 points, removing 10.7 drops less — discrete points, not a sweep

What's missing:
- **No parametric sweep** — no intermediate α values between 0 and 1
- **No off-target measurement** — we don't know where collateral damage begins
- **No therapeutic window estimate** — can't compute the gap between threshold and off-target onset

The curve is two endpoints with no interior. We know the maximum effect is large, but we cannot characterize threshold, EC₅₀, or selectivity boundary. The strongest statement possible: "full ablation produces a large effect." The shape of the mechanism remains invisible.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Not reported.** No bootstrap confidence intervals or test-retest measurements are provided for the faithfulness scores. We do not know whether the 87% faithfulness figure has a confidence interval of ±2% or ±15%.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Partial.** The ABBA/BABA template comparison provides some measurement invariance evidence — the circuit identification is stable across template variants. But invariance across prompt distributions (formal text vs. dialogue vs. code) is not tested.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** A random-component baseline is included. The IOI circuit's effect size is clearly separated from the baseline distribution, establishing that the measurement is detecting a real signal above noise.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Not explicitly tested.** Can the measurement distinguish the IOI circuit from a slightly different circuit (e.g., 24 of the 26 heads)? The sensitivity curve — faithfulness as a function of circuit size — is partially implicit in the analysis but not reported as a formal sensitivity assessment.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.** Is 87% faithfulness "good"? Without calibration against a gold standard or against known-correct circuits, the number is hard to interpret in absolute terms.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Partial.** The primary metric is logit difference. This captures the direction of the model's prediction but not its confidence, calibration, or full distributional effects. Accuracy and cross-entropy are occasionally mentioned but not systematically reported as complementary metrics.

### Key Distinctions

- **Reliability vs validity:** The 87% faithfulness is reported without confidence intervals — we don't know if the metric is reliable. A reliable metric pointed at the wrong target produces confident wrong answers, but we can't even confirm reliability here.
- **Convergent vs discriminant validity:** One convergent comparison exists (activation patching vs. weight-space analysis partially agree). Zero discriminant comparisons — we don't know if the methods agree more about IOI than about everything else.

### MTMM Matrix

| | Act. patching (IOI) | Weight analysis (IOI) | Act. patching (GT) | Weight analysis (GT) |
|---|---|---|---|---|
| **Act. patching (IOI)** | — | ~0.61 (partial) | ? | ? |
| **Weight analysis (IOI)** | ~0.61 | — | ? | ? |
| **Act. patching (GT)** | ? | ? | — | ? |
| **Weight analysis (GT)** | ? | ? | ? | — |

One convergent cell partially filled (activation patching vs. weight-space for the same circuit: estimated Jaccard ~0.61). No discriminant cells filled — we don't know if the methods agree *more* about IOI than they agree about everything. Without the discriminant comparison, the convergent evidence could reflect method bias rather than genuine construct convergence.

Reliability: unknown (no confidence intervals reported for the 87% figure). The MTMM cannot be interpreted until reliability establishes a ceiling on correlations.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is clearly at the [algorithmic](/framework/modes/algorithmic) level — it names a multi-step computation (detect duplicates → inhibit → copy) implemented by specific components.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Pass.** The evidence includes both behavioral effects (ablation changes outputs) and structural signatures ($W_{OV}$ analysis), which jointly support an algorithmic-level claim. The evidence is not solely behavioral (which would support only a computational-level claim).

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** The three-stage story (duplicate detection → S-inhibition → name-mover copying) is logically coherent and consistent with the layer ordering of the identified heads. The narrative explains *why* each component is needed, not just that it is needed.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Weak.** [Méloux et al. (2025)](https://arxiv.org/abs/2410.10186) find alternative circuits for IOI with comparable faithfulness but different membership. The IOI circuit is *a* faithful circuit, possibly not *the* unique one. The "detect → inhibit → copy" algorithm might also be implementable by different head subsets, meaning the algorithm is underdetermined by the data.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Partial.** The paper tests on template-generated prompts and generalizes to "the IOI task." Whether the circuit handles naturalistic IOI (where the names are not cleanly separated, or where there are more than two names) is unclear. The scope of the claim slightly exceeds the scope of the evidence.

### Key Distinctions

- **Description vs explanation:** The "detect-inhibit-copy" narrative is an explanation (it specifies a multi-step algorithm), not merely a description. This is a strength — most MI papers only describe which components are active without explaining the algorithm.
- **Component identity vs component role:** The slide from "head 9.9 is in the circuit" to "head 9.9 is a name-mover" is well-supported here — the $W_{OV}$ structure independently confirms the role label. This is one case where the role claim has structural backing.
- **Faithfulness vs understanding:** High faithfulness (87%) AND a coherent mechanistic narrative. The two axes are both strong, which is rare. The weakness is that alternative circuits (Méloux et al.) suggest the understanding may be one of several valid accounts.

### Evidence Convergence Map

- **Implementational → Interpretation:** Strong. Ablation and patching identify specific heads; $W_{OV}$ signatures confirm structural roles. Multiple implementational sub-modes converge.
- **Algorithmic → Interpretation:** Moderate. "Detect-inhibit-copy" is a specified multi-step process with layer ordering that matches the narrative. But whether this is the *unique* algorithm or one of several compatible implementations is unresolved (Méloux et al.).
- **Computational → Interpretation:** Weak. We know the circuit performs IOI on templates. Whether IOI is the right computational description — or a special case of contextual coreference — is untested.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Ablation | ✓ | ✓ | ∅ | ∅ | ∅ |
| Act. patching | ✓ | — | — | ∅ | ∅ |
| IIA/DAS | — | — | — | — | — |
| Steering | — | — | — | — | — |
| Weight analysis | — | — | ✓ (partial) | — | — |

Most cells empty or structurally ∅. The filled cells cluster in the ablation row (necessity + sufficiency) and one weight-analysis cell (representational). The algorithmic and computational columns have no valid evidence — yet the paper makes algorithmic claims ("detect-inhibit-copy"). The gap is visible.

### Causal Sufficiency Graph

- S-inhibition heads → name-mover heads: **solid** (path patching confirms information flow)
- Duplicate token heads → S-inhibition heads: **dashed** (compositional structure inferred from layer order, not directly patched)
- Name-mover heads → output: **solid** (DLA directly measures contribution)
- Primary → backup name-movers: **dashed** (compensatory activation observed, causal pathway not precisely characterized)

The solid-edge subgraph has a gap: the path from input to S-inhibition is dashed (inferred, not causally confirmed). The circuit has a confirmed output stage and a confirmed intermediate link, but the full input-to-output causal chain has one unverified step.

