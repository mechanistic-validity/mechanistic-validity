---
title: "Case Study: Gender Bias Circuits"
description: "Gender bias localization and debiasing (Bolukbasi et al. 2016, Vig et al. 2020, Ravfogel et al. 2020) evaluated through all five validity lenses."
---

# Case Study: Gender Bias Circuits

Multiple papers attempt to locate and remove **gender bias** in language models. [Bolukbasi et al. (2016)](https://arxiv.org/abs/1607.06520) identify a "gender direction" in word embeddings. [Vig et al. (2020)](https://arxiv.org/abs/2010.06032) use causal mediation analysis to identify attention heads that mediate gender bias in GPT-2. Ravfogel et al. (2020) use iterative nullspace projection (INLP) to remove gender information from representations. The shared claim: gender bias is localized in identifiable components (directions, heads, subspaces) and can be surgically removed.

This case study is important because it connects mechanistic claims to real-world consequences — debiasing tools are deployed in practice. The stakes for getting the mechanism wrong are higher than for academic circuit analysis.

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability | C3/C5 Specificity + Convergence | Weak |
| Internal | I1 Necessity (partial) | I3/I4/I5 | Weak |
| External | E1 Intervention reach | E3/E5 Selectivity + Robustness | Weak |
| Measurement | M3 Baseline separation | M1/M2/M6 | Weak |
| Interpretive | V1 Level declaration | V3/V4/V5 | Weak |

**Overall verdict: Proposed to Causally suggestive — with fundamental construct problems.** Gender bias circuits score weakly across all five lenses. The core issue is not lack of evidence but **construct incoherence** (C3): "gender bias" and "gender knowledge" are not separable at the mechanistic level, which means the construct itself may not be well-defined enough to have a circuit. This is not a measurement problem or an evidence problem — it is a *theory* problem.

This case study illustrates the framework's most important function: sometimes the right verdict is not "the evidence is insufficient" but "the construct is not coherent enough to evaluate." When task specificity (C3) fails fundamentally — when the phenomenon cannot be separated from a related phenomenon that uses the same components — the mechanistic claim cannot be established regardless of how much evidence is collected. The framework names this problem rather than hiding it behind aggregate scores.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Gender direction projection (embedding geometry) | [B01 SVD/Spectral](/framework/metrics/structural/b01-svd-spectral) | Structural |
| Causal mediation analysis (Vig et al.) | [A06 Mediation](/framework/metrics/causal/a06-mediation) | Causal |
| Iterative nullspace projection / INLP (Ravfogel et al.) | [E02 Linear Probe](/framework/metrics/representational/e02-linear-probe) | Representational |
| Activation steering along gender direction | [A02 Counterfactual DAS](/framework/metrics/causal/a02-counterfactual-das) | Causal |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "gender bias circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Partial.** Each approach makes testable predictions: removing the gender direction should reduce bias on benchmark tests; ablating mediating heads should reduce gendered predictions. These are testable. But "bias" itself is a contested construct — different benchmarks measure different things, and success on one does not guarantee success on others.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Partial.** A single "gender direction" is structurally plausible in embedding space (it exists and is measurable). Whether bias in a deep transformer is captured by a single direction per layer, rather than being distributed across many parameters, is a much stronger structural assumption. Vig et al.'s identification of mediating attention heads is more structurally detailed but still does not explain *how* the heads encode bias.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Weak — critical gap.** The core problem: "gender information" is not separable from "gender-related knowledge." Removing the model's ability to distinguish gender also removes its ability to correctly resolve gendered pronouns, understand gendered language, or perform tasks that require gender knowledge. The circuit for bias and the circuit for legitimate gender processing may be the same circuit. Task specificity cannot be established because the two "tasks" are not separable.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Unclear.** Is one direction minimal? INLP iteratively finds multiple directions, suggesting the first direction is not sufficient. Is one set of attention heads minimal? Vig et al. identify many heads, not a clean minimal set.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Weak.** Different methods (direction removal, INLP, causal mediation, activation steering) identify different components as "where bias lives." They do not converge on the same locus. This may reflect genuine distribution of bias rather than method disagreement — but without convergence, the construct "gender bias circuit" is method-dependent.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Partial | Benchmark predictions testable; "bias" contested |
| C2 Structural plausibility | Partial | Direction exists; deep localization unclear |
| C3 Task specificity | Weak | Bias and gender knowledge inseparable |
| C4 Minimality | Unclear | Multiple methods find multiple loci |
| C5 Convergent validity | Weak | Methods disagree on localization |

### Key Distinctions

- **Confirmation vs corroboration:** Debiasing interventions are "confirmed" by the same benchmark used to define the bias. A model debased on WinoBias scores better on WinoBias — this is circular confirmation. Genuine corroboration would require showing bias reduction on a held-out benchmark the intervention was not optimized for, which typically fails (Gonen & Goldberg 2019).
- **Natural kind vs family resemblance:** "Gender bias" may not be a natural kind at the mechanistic level — it may be a family resemblance concept grouping disparate phenomena (stereotyped associations, pronoun statistics, name-occupation correlations) that share a surface label but lack a unified mechanism.
- **Underdetermination:** The divergence between methods (direction removal, INLP, causal mediation) finding different "bias loci" directly demonstrates underdetermination — the behavioral data (bias benchmark scores) does not uniquely determine which components implement bias.

### Nomological Network

The "gender bias circuit" construct connects to:
- **Embedding geometry** — a gender direction exists in word/token embedding space (structural, confirmed)
- **Benchmark reduction** — removing the direction reduces scores on tested benchmarks (behavioral, confirmed on trained benchmark)
- **Cross-benchmark transfer** — debiasing transfers to untested benchmarks (robustness, often fails)
- **Knowledge preservation** — debiasing preserves legitimate gender knowledge (specificity, often fails)
- **Cross-method convergence** — different methods find the same bias locus (convergent, fails)
- **Mechanism specification** — how bias is *computed* by the identified components (algorithmic, untested)
- **Training origin** — how bias enters the model during training (developmental, untested)

Two nodes confirmed (direction exists, trained-benchmark scores improve), three nodes that actively *fail* (cross-benchmark, knowledge preservation, cross-method convergence). A network with failing nodes is worse than one with untested nodes — it suggests the construct is incoherent rather than merely underexplored.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that bias is implemented in the identified components?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Partial.** Removing the gender direction / ablating mediating heads reduces bias on tested benchmarks. But the reduction is often incomplete, and bias re-emerges on different benchmarks or in different contexts. Necessity is benchmark-specific.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Not demonstrated.** Can you *induce* gender bias by stimulating the identified components? Activation steering along gender directions does produce gendered outputs — but this is sufficiency for gender *information*, not specifically for *bias*. The distinction matters.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Weak.** Removing the gender direction reduces bias *and* degrades gender-related task performance. The intervention is not specific to bias — it removes gender information broadly. This is the fundamental problem with the approach: bias and knowledge share components.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Weak.** Debiasing interventions that work on one benchmark often fail on others (Gonen & Goldberg 2019). The effect does not replicate robustly across evaluation settings. Different prompt templates, different bias metrics, and different downstream tasks give different results.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Weak.** The primary confound: removing gender information (debiasing) may simply make the model *worse at predicting* in gendered contexts, producing apparent debiasing as a side effect of degradation. Without controlling for overall quality loss, the debiasing effect is confounded.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Partial | Benchmark-specific reduction |
| I2 Sufficiency | Not demonstrated | Gender information does not equal bias specifically |
| I3 Specificity | Weak | Removes knowledge with bias |
| I4 Consistency | Weak | Benchmark-specific; does not generalize |
| I5 Confound control | Weak | Degradation confound |

### Key Distinctions

- **Single vs double dissociation:** Debiasing provides partial single dissociation only (removing components reduces bias on one benchmark). The crucial double dissociation — removing *bias* without removing *gender knowledge* — is precisely what cannot be achieved, because the two are mechanistically entangled.
- **Lesion vs stimulation:** Direction removal is a lesion. Steering along the gender direction is a stimulation. Critically, stimulation produces *gendered* output, not specifically *biased* output — confirming that the direction encodes gender information generally, not bias specifically.

### Dissociation Matrix

|  | Bias benchmark A | Bias benchmark B | Gender knowledge task | General capability |
|---|---|---|---|---|
| Remove gender direction | ↓ (partial) | ? or ↓ (weak) | **↓↓ (degradation)** | ↓ (some) |
| Ablate mediating heads | ↓ (partial) | ? | ? | ? |
| INLP projection | ↓ (partial) | ↓ (partial) | **↓↓** | ↓ |

The critical finding: the "bias benchmark" column and the "gender knowledge" column both show degradation from the same intervention. This is the anti-dissociation — the intervention cannot distinguish bias from knowledge because they share the same mechanistic substrate. The matrix makes visible that surgical bias removal is impossible if the target and the side effect occupy the same components.

---

## Pharmacology Lens — External Validity

*Does intervening on the bias circuit produce selective behavioral change?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Partial.** Debiasing interventions change model outputs. But whether they change the *right* thing (bias without knowledge loss) is disputed.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Sometimes.** Scaling the projection magnitude produces graded effects. But the useful range (enough to reduce bias, not enough to degrade performance) is narrow and context-dependent.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Weak.** Interventions are not selective — they affect both bias and legitimate gender knowledge. This is the pharmacological equivalent of a drug with severe side effects that cannot be separated from the therapeutic effect.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Variable.** Large on the benchmarks used during development. Smaller or absent on held-out benchmarks.

**[E5 — Robustness:](/framework/criteria/external/robustness) Weak.** The most robust finding is that debiasing is brittle — it works on tested settings and fails on untested ones (Gonen & Goldberg 2019).

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Partial.** Bias exists across architectures. Whether the same debiasing technique transfers is model-dependent.

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Partial | Changes outputs; not always correctly |
| E2 Graded response | Sometimes | Narrow useful range |
| E3 Selectivity | Weak | Bias + knowledge inseparable |
| E4 Effect magnitude | Variable | Benchmark-specific |
| E5 Robustness | Weak | Brittle across settings |
| E6 Cross-architecture | Partial | Technique transfer variable |

### Key Distinctions

- **Affinity vs efficacy:** The gender direction has high affinity (it clearly relates to gender-associated tokens) but questionable efficacy for the intended purpose (removing bias selectively). The intervention binds to the right target but produces both therapeutic (bias reduction) and toxic (knowledge loss) effects simultaneously.
- **Therapeutic window:** The therapeutic window for debiasing is extremely narrow or nonexistent — the dose that reduces bias on one benchmark simultaneously degrades gender knowledge. This is the pharmacological signature of a target that cannot be selectively modulated because the "disease" and normal function share the same receptor.
- **Off-target effects as the diagnosis:** The fact that off-target effects (knowledge degradation) are inseparable from on-target effects (bias reduction) is not a failure of methodology — it is the diagnosis. The construct "gender bias circuit" may not exist as a separable entity, and the off-target effects reveal this.

### Dose-Response Curve

For gender direction removal (varying projection strength):
- **0% projection**: full model, bias intact
- **Partial projection**: some bias reduction + some knowledge loss (the two track together)
- **Full projection**: maximum bias reduction on trained benchmark + significant knowledge degradation + bias persistence on untested benchmarks

The critical feature of this dose-response: there is no regime where bias decreases without knowledge also decreasing. The two curves are coupled, not separable. This is pharmacological evidence that the target is not specific — the "bias" pathway and the "knowledge" pathway share the same substrate.

What's missing:
- **No selective dose** — no intervention strength produces bias reduction without knowledge cost
- **No plateau identification** — does bias reduction saturate before knowledge loss becomes critical?
- **No cross-benchmark dose-response** — the curve may look different on each bias measure

---

## Measurement Theory Lens — Measurement Validity

*Are the bias metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Weak.** Different bias benchmarks give different answers. The measurement of "bias" itself is unreliable across metrics.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Weak.** A model that appears debiased on one benchmark appears biased on another. The measurement is not invariant across evaluation conditions.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Partial.** Gender direction projections clearly separate male/female-associated words. But separating "bias" from "knowledge" in the measurement is the unsolved problem.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Unknown.** Can the metric distinguish between "the model is unbiased" and "the model has learned to hide bias from the benchmark"? Gonen & Goldberg's "lipstick on a pig" result suggests the latter is common.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Poorly understood.** What level of bias-benchmark performance constitutes "debiased"? There is no agreed threshold.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Weak.** Each benchmark measures one facet of bias. No single metric captures the full construct. "Bias" is a multi-dimensional construct measured by one-dimensional metrics.

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Weak | Benchmark disagreement |
| M2 Invariance | Weak | Results don't transfer across benchmarks |
| M3 Baseline separation | Partial | Direction separates; bias/knowledge don't |
| M4 Sensitivity | Unknown | Hiding vs. removing |
| M5 Calibration | Poorly understood | No agreed threshold |
| M6 Construct coverage | Weak | Multi-dimensional construct, 1D metrics |

### Key Distinctions

- **Reliability vs validity:** Bias measurements are unreliable (different benchmarks disagree) AND of uncertain validity (they may measure surface patterns rather than genuine bias). When reliability is low, validity cannot be established — you cannot validate a measurement that produces different results each time.
- **Convergent vs discriminant validity:** Multiple bias benchmarks should *converge* (score the same model similarly) — they often do not. Different benchmarks should *discriminate* between bias and non-bias — but they cannot distinguish "debiased" from "degraded." Both convergent and discriminant validity fail for bias measurement.
- **The construct precedes the metric:** Measurement theory assumes a well-defined construct that the metric measures. If the construct itself (separable gender bias) is incoherent, no metric can validly measure it — the problem is pre-measurement.

### MTMM Matrix

| | WinoBias | StereoSet | CrowS-Pairs | Direction projection |
|---|---|---|---|---|
| **WinoBias** | — | low-moderate | low | moderate |
| **StereoSet** | low-moderate | — | low-moderate | ? |
| **CrowS-Pairs** | low | low-moderate | — | ? |
| **Direction projection** | moderate | ? | ? | — |

Cross-benchmark convergence (the off-diagonal cells) is low to moderate — different metrics disagree about how biased a model is. This is a reliability crisis for the construct: if multiple metrics measuring "the same thing" produce different results, either they are measuring different things (the construct is multi-dimensional) or they are all poorly calibrated. For gender bias, both are likely true simultaneously.

---

## MI Lens — Interpretive Validity

*Is "gender bias is localized and removable" warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Partial.** The claims range from representational ("bias lives in a direction") to implementational ("these heads mediate bias") without always distinguishing the levels.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Partial.** Direction-based methods provide representational evidence. Causal mediation provides causal evidence. But the stronger claim ("bias can be surgically removed") requires implementational evidence that is not provided.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Weak.** "Bias lives in one place and can be removed" is narratively simple but inconsistent with the evidence (bias re-emerges, removal causes side effects). The narrative coherence breaks down under scrutiny.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Weak.** The primary alternative: bias is not a localized property but an emergent property of the full model — a consequence of training data distribution reflected throughout all parameters. Under this alternative, surgical removal is fundamentally impossible, and apparent debiasing is actually degradation-masking. This alternative is not excluded.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Often violated.** "We removed gender bias from the model" claims far more than "we reduced scores on bias benchmark X by projecting out direction Y." The scope inflation is particularly concerning given practical deployment.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Partial | Mixed levels |
| V2 Level-evidence match | Partial | Representational evidence for implementational claims |
| V3 Narrative coherence | Weak | Narrative contradicted by replication failures |
| V4 Alternative exclusion | Weak | Distributed bias alternative not excluded |
| V5 Scope honesty | Often violated | "Debiased" exceeds evidence |

### Key Distinctions

- **Description vs explanation:** Debiasing papers describe *where* bias correlates (a direction, certain heads) but do not explain *why* bias and knowledge are entangled or *how* the model computes biased predictions. The description is accurate (the direction exists) but the explanation implied by the intervention (bias is localized and removable) is contradicted by the evidence.
- **Component identity vs component role:** The gender direction is identified (component identity) but its role is ambiguous — is it "the bias direction" or "the gender information direction" or "one of many correlated directions"? The role label "bias" is applied based on desired outcome rather than mechanistic evidence.
- **Faithfulness vs understanding:** Debiasing interventions are "faithful" to their trained benchmark (they reduce the target metric) but do not reflect genuine understanding of how bias is implemented. Benchmark faithfulness without mechanistic understanding produces interventions that are fragile and side-effect-prone.

### Evidence Convergence Map

- **Implementational → Interpretation:** Weak. Causal mediation identifies mediating heads, but multiple methods identify different components. The implementational evidence diverges rather than converges.
- **Algorithmic → Interpretation:** Absent. No paper specifies the algorithm by which the model produces biased outputs through the identified components. The computational steps from "gender direction exists" to "biased prediction emerges" are uncharacterized.
- **Computational → Interpretation:** Moderate. The computational-level claim ("the model produces biased outputs") is clearly supported. But the mechanistic claims (where, how, and whether it can be removed) have much weaker support.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Direction removal | partial | — | ✓ | — | partial |
| Head ablation (Vig) | partial | — | — | — | partial |
| INLP | partial | — | ✓ | — | partial |
| Activation steering | — | partial (gender, not bias) | partial | — | — |

The "partial" entries cluster in necessity and representational columns. Crucially, no row demonstrates bias-*specific* sufficiency — steering produces gender effects, not bias effects specifically. The algorithmic column is entirely empty, reflecting the absence of mechanistic explanation for how bias is computed. The pattern reveals that the evidence supports "gender information exists in identifiable components" much more strongly than "gender *bias* is localized and removable."

### Causal Sufficiency Graph

- Training data statistics → model weights: **solid** (bias enters through training data, well-established)
- Gender direction → gendered predictions: **solid** (projecting out the direction changes gendered outputs)
- Gender direction → bias specifically: **dashed** (the direction encodes gender broadly, not bias specifically)
- Bias removal → sustained debiasing: **broken** (bias re-emerges on untested benchmarks — Gonen & Goldberg)
- Localized components → full bias explanation: **broken** (different methods find different loci; no convergence)

Two solid edges, one dashed, two broken. The broken edges are particularly informative — they represent claims that are not merely untested but actively disconfirmed by evidence (bias re-emergence, method divergence). A causal sufficiency graph with broken edges indicates a theory that is partially falsified, not merely incomplete.

---
