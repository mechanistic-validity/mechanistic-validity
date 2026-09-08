---
title: "Case Study: IOI Circuit"
description: "The indirect object identification circuit (Wang et al. 2022) evaluated through the five core lenses."
---

# Case Study: IOI Circuit

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) identify 26 attention heads in GPT-2 Small that form the **indirect object identification circuit** — a mechanism that detects duplicated names, suppresses them, and copies the remaining name to the output. This is the most thoroughly analyzed circuit in mechanistic interpretability.

Below, we evaluate this claim through each of the five core lenses, applying the full criteria set.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Causally Suggestive. **Capped by:** E1 (intervention reach), I4 (specificity).


| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C2 Structural plausibility | C4 Discriminant validity | Partial |
| Internal (Neuroscience) | I1/I2 Necessity + Sufficiency | I4/I7 Specificity + Confound | Causally suggestive |
| External (Pharmacology) | E5 Graded response | E1/E4 Reach + Cross-model | Weak |
| Measurement (Measurement Theory) | M2 Baseline separation | M1/M4 Reliability + Calibration | Partial |
| Interpretive (MI) | V2 Level-evidence match | V3 Alternative level | Strong |

**Overall verdict: Causally Suggestive.** The IOI circuit has confirmed falsifiability (C1) and novel prediction (E6), and sufficiency holds on average — mean-ablating everything outside the circuit leaves 87% of the logit difference (I2). Component-level necessity is weaker than the narrative suggests: knocking out all three Name Mover heads costs a 5% drop, because backup heads take over (I1). It stops short of Mechanistically Supported on intervention reach (E1), where methods disagree across three granularities, and specificity (I4), where head overlap and task effect give opposite verdicts.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Activation patching | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| Path patching | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| Mean ablation | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Direct logit attribution (DLA) | [D02 Logit-Diff Recovery](/mechanistic-validity/framework/metrics/#d02) | Behavioral |
| $W_{OV}$ / $W_{QK}$ decomposition | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |
| Logit difference | [D02 Logit-Diff Recovery](/mechanistic-validity/framework/metrics/#d02) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "the IOI circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Confirmed.** Each of the 26 heads carries a role assignment tied to a path-patching prediction the experiment could have contradicted, and the three validation criteria are stated as quantities with the direction of failure fixed before they are computed. The strongest evidence that the criteria are live is that the paper reports failing one of them, and says so in the introduction.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** Weight-level validation exists and is quantitative for the two Name Mover classes: the copy score pushes the residual stream through the head's OV matrix and the unembedding, and it separates from an average head by a wide margin in both directions. For the remaining classes the paper names the parameter-level checks it would need and states it did not run them, and §5 records that one class's attention pattern is not understood at all.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Inconclusive.** The origin never compares the IOI circuit against a circuit for a different task, so nothing in it bears on the question. Post-origin the evidence conflicts: [Merullo et al. (2024)](https://arxiv.org/abs/2410.01434) find 78% head overlap between the IOI and Colored Objects circuits in GPT-2 Medium, while [Hanna et al. (2024)](https://arxiv.org/abs/2403.17806) find cross-task faithfulness near zero for most task pairs in GPT-2 Small. Different models, different quantities, unreconciled.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Complicated.** The 26-head circuit includes backup name-mover heads that are individually unnecessary — the primary name movers suffice. The backups activate compensatorily when primaries are ablated, raising the question of whether the circuit is over-inclusive under normal operation. Whether backups are "in the circuit" depends on the definition of minimality.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** The circuit was discovered primarily through activation patching and direct logit attribution — methods that share interventionist assumptions. Weight-space analysis confirms structural plausibility (C2), providing partial convergent evidence from a different method family. However, a systematic Jaccard comparison between activation-based and weight-based circuit definitions has not been reported.

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
- **Cross-model prediction** — GPT-2 Medium at origin, Pythia post-origin (partially confirmed)

Four nodes confirmed, three unconnected. A moderately thick network — strong, but with clear gaps at the generalization edges.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation, not just participation?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Partial.** Necessity is demonstrated by knockout, and the source reports the demonstration failing at the level of the circuit's headline component class: removing all three Name Mover Heads leaves 95% of the logit difference. The per-node necessity test is existentially quantified — "there exists a subset K" — so each minimality score is conditional on a co-ablated set selected to make that score large, and the paper states the resulting effects are small.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** Sufficiency is measured directly: everything outside the circuit is mean-ablated against the p_ABC reference distribution, and the surviving average logit difference is 87% of the model's. The source reports a single average with no dispersion and no per-example figure, and the paper itself flags that this number does not settle the account.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Inconclusive.** Merullo et al. (2024) measure both head overlap and task effect, and the two give opposite verdicts. Alongside E1, this is the criterion capping the claim.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Sequence length is controlled twice. Name frequency is not, and the analysis is not replicated under an alternative ablation method.

### Key Distinctions

- **Single vs double dissociation:** Only single dissociation is performed — ablating the IOI circuit impairs IOI. The converse (ablating a different circuit and showing IOI remains intact) is never tested. Without double dissociation, the circuit could be a general-purpose module whose removal impairs many tasks.
- **Lesion vs stimulation:** The paper uses only lesion-style evidence (mean ablation). No stimulation experiment (amplifying name-mover signals or steering the circuit to produce a specific name) is reported, leaving open whether the circuit is merely necessary infrastructure or a genuinely steerable mechanism.

### Dissociation Matrix

|  | IOI task | SVA task | Factual recall | Pronoun resolution |
|---|---|---|---|---|
| Ablate IOI circuit | **↓ (5% for Name Mover knockout)** | ? | ? | ? |
| Ablate SVA circuit | ? | ? | ? | ? |
| Ablate factual circuit | ? | ? | ? | ? |

One cell filled out of twelve. The diagonal entry is weaker than the narrative suggests — knocking out all three Name Mover heads costs a 5% drop, because backup heads take over — and without off-diagonal measurements we cannot distinguish "IOI-specific mechanism" from "general syntactic bottleneck." The matrix makes visible exactly what's missing: every `?` is an untested double-dissociation leg.

---

## Pharmacology Lens — External Validity

*Does intervening on the circuit produce the expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Inconclusive.** One ablation value at origin across three granularities, and the methods disagree with one another. This is the criterion that caps the claim: what it needs is adjudication between existing results, not a first experiment.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** Ablating individual heads produces graded effects — removing head 9.9 has a larger effect than removing head 10.7. But a parametric dose-response (ablating at varying strengths, or patching at varying magnitudes) is not systematically reported.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Disconfirmed.** At 106 clean/corrupted prompt pairs, model and circuit diverge. This is tested and failed, not merely unextended.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Partial.** GPT-2 Medium is run at origin, and the circuit has since been reproduced across Pythia models. What has not been shown is that the seven head classes recur with the same roles.

### Key Distinctions

- **Affinity vs efficacy:** The IOI circuit demonstrates both — heads are highly active on IOI prompts (affinity) AND ablation degrades performance (efficacy). The 87% faithfulness figure combines both into a single measurement.
- **The system compensates:** Backup name-movers activate compensatorily when primaries are ablated, directly demonstrating receptor reserve. This is one of the few MI results that explicitly documents compensation, though the compensation ceiling is not fully characterized.
- **The metric is part of the finding:** The circuit was discovered using logit difference under mean ablation and evaluated using the same metric under the same intervention. Miller et al. show faithfulness drops below 50% under alternative methods, confirming that the metric choice inflates apparent circuit quality.

### Dose-Response Curve

The IOI circuit's dose-response curve is mostly unknown. We have:
- **α = 0** (no intervention): full performance
- **α = 1** (mean-ablating everything outside the circuit): 87% of the logit difference is retained; knocking out the Name Movers themselves costs 5%
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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Inconclusive.** Sample sizes are disclosed for every estimate, and dispersion is shown where the quantity is an attention probability. The quantity the argument rests on gets neither: faithfulness is reported once, as 0.46, with no interval, no resampling and no per-example spread, and the same holds for the incompleteness and minimality scores. Dispersion reported for the peripheral measurements and withheld from the central one.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Disconfirmed.** Faithfulness differs across the origin's own ABBA/BABA templates. This is not an untested extension: the measurement has been shown not to hold across the conditions the paper itself uses.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Partial.** Three baselines are present. The copy-score baselines are average-head comparisons and separate cleanly (>95% vs <20%; 98% vs 12%), and the completeness test carries a random-subset baseline over ten uniformly sampled K. The circuit-level baseline is the naïve circuit of §4.3, and it does not separate: the paper's own word is "comparable", and because faithfulness is the gap |F(M) − F(C)|, the naïve circuit's 0.1 is a *smaller* gap than the published circuit's 0.46.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Not explicitly tested.** Can the measurement distinguish the IOI circuit from a slightly different circuit (e.g., 24 of the 26 heads)? The sensitivity curve — faithfulness as a function of circuit size — is partially implicit in the analysis but not reported as a formal sensitivity assessment.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The outcome metric is well calibrated: logit difference carries a sign, a zero point meaning indifference between IO and S, and a measured model value of 3.56 over 100,000 examples. The validation metric inherits the units and none of the calibration — faithfulness is an unsigned distance with no upper bound, no null value and no stated threshold separating faithful from unfaithful, reported as a single average.

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Partial.** The unit of description is declared exactly: a subgraph over model components, with the contrast against feature-level circuits stated rather than left implicit. The algorithm is also stated, as three steps mapped one-to-one onto three head classes. What is never declared is the relation between the two descriptions — whether the three-step algorithm is a claim about the computation the heads perform or a summary of where information travels.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** The evidence matches the claim at one step of three. For S-Inhibition the paper measures the content of what is moved, decomposes it into token and position components, and shows an additive model of the two predicts logit difference to within 7% — algorithm-level evidence for an algorithm-level claim. For duplicate detection and induction the corresponding parameter-level evidence is named and not run, and the paper states it did not investigate how the induction-to-position mechanism works.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Inconclusive.** A strictly simpler description of the same behavior is built at origin. The naïve circuit drops the Backup and Negative Name Movers and half the Duplicate Token and Induction heads, and it matches the published circuit on the criterion the paper leads with. Two of the three completeness samplers separate them; the third separates neither. The source constructs its own simpler rival and does not defeat it on faithfulness. [Méloux et al. (2025)](https://arxiv.org/abs/2410.10186) find further faithful circuits with different membership.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Confirmed.** Scope is declared in five places, and in the two that matter most: the abstract concedes remaining gaps before any result is given, and the introduction states the circuit fails the hardest of the paper's own tests. The exclusions are stated as design decisions rather than discovered afterwards, and §5 frames the whole result as a model organism, which names the generalization limit rather than leaving the reader to infer it.

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

