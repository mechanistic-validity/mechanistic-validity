---
title: "Case Study: Gender Bias Circuits"
description: "Causal mediation analysis of gender bias in GPT-2 (Vig et al. 2020) evaluated through the five core lenses."
---

# Case Study: Gender Bias Circuits

[Vig et al. (2020)](https://arxiv.org/abs/2010.06032) use causal mediation analysis to identify the attention heads and neurons that mediate gender bias in GPT-2, decomposing the effect of a gendered intervention into a natural direct effect and a natural indirect effect routed through named components. The claim audited here is theirs: gender bias is sparsely mediated, and ten of 144 heads reproduce the effect of intervening on all of them.

The wider debiasing literature — a "gender direction" in embeddings [Bolukbasi et al. 2016](https://arxiv.org/abs/1607.06520), iterative nullspace projection [Ravfogel et al. 2020] — makes the stronger claim that bias can be surgically removed. That claim is not the one scored below, and where the two diverge the difference is noted.

This case study is important because it connects mechanistic claims to real-world consequences — debiasing tools are deployed in practice. The stakes for getting the mechanism wrong are higher than for academic circuit analysis.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Causally Suggestive. **Capped by:** E1 (intervention reach), I4 (specificity).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1 Falsifiability | C4/C3 Discriminant + Convergence | Weak |
| Internal | I1 Necessity (partial) | I4/M1/I7 | Weak |
| External | E2 Prompt generalization | E1/E3 Intervention reach + Cross-task | Weak |
| Measurement | M2 Baseline separation | M1/M6/C3 | Weak |
| Interpretive | V1 Level declaration | V2/V3/V5 | Weak |

**Overall verdict: Causally Suggestive.** The claim is capped by intervention reach (E1) and specificity (I4). Causal mediation is the single instrument throughout, so every agreement between results is the method with itself, and no second intervention family has been run. Underneath that sits a construct problem: **discriminant validity** (C4) is untested, because "gender bias" and "gender knowledge" are not separated at the mechanistic level, and a construct that cannot be told apart from its neighbor cannot be localized to a circuit.

This case study illustrates the framework's most important function: sometimes the right verdict is not "the evidence is insufficient" but "the construct is not coherent enough to evaluate." When discriminant validity (C4) fails fundamentally — when the phenomenon cannot be separated from a related phenomenon that uses the same components — the mechanistic claim cannot be established regardless of how much evidence is collected. The framework names this problem rather than hiding it behind aggregate scores.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Gender direction projection (embedding geometry) | [B01 SVD/Spectral](/mechanistic-validity/framework/metrics/#b01) | Structural |
| Causal mediation analysis (Vig et al.) | [A06 Mediation](/mechanistic-validity/framework/metrics/#a06) | Causal |
| Iterative nullspace projection / INLP (Ravfogel et al.) | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| Activation steering along gender direction | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "gender bias circuit" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Confirmed.** The sparsity claim is put as a quantity that could have come out any way. The search runs over all 144 heads and reports how many are needed to reach the all-head effect; the answer could have been 100, and it was 10. Two negative cases are named in advance and one of them fires: the untrained model shows neither the effect nor the structure, and the gender-neutral arm — the same instrument on the same templates with a different treatment target — fails to show the concentrated structure, reported as a negative result rather than dropped.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** A single "gender direction" is structurally plausible in embedding space (it exists and is measurable). Whether bias in a deep transformer is captured by a single direction per layer, rather than being distributed across many parameters, is a much stronger structural assumption. Vig et al.'s identification of mediating attention heads is more structurally detailed but still does not explain *how* the heads encode bias.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Not tested — the construct problem.** Nothing separates the bias mechanism from a mechanism for gender competence, because no second construct is ever localized. The outcome is a ratio over two pronouns and nothing else is measured, so there is no quantity on which a competence mechanism could show up. Where the two constructs collide the paper handles it by deletion: definitionally gendered professions are dropped from the total-effect calculation rather than analyzed as the contrasting case.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Unclear.** Is one direction minimal? INLP iteratively finds multiple directions, suggesting the first direction is not sufficient. Is one set of attention heads minimal? Vig et al. identify many heads, not a clean minimal set.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Confirmed.** Convergence at origin is broad and shallow. Two mediator families run on three datasets give the same sparsity picture, and four outcome scales give it again, so the result is robust to the unit of analysis and to the metric. What they share is the instrument: every number is a natural indirect effect produced by the same interchange intervention. The one check of a different kind is the TE ≈ NDE + NIE decomposition, which holds the mediation numbers against a quantity they were not fitted to.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Partial | Benchmark predictions testable; "bias" contested |
| C2 Structural plausibility | Partial | Direction exists; deep localization unclear |
| C4 Discriminant validity | Weak | Bias and gender knowledge inseparable |
| I3 Minimality | Unclear | Multiple methods find multiple loci |
| C3 Convergent validity | Weak | Methods disagree on localization |

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

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) N/A.** The design's estimands are natural direct and indirect effects. Neither can be put in the form necessity requires, so the criterion has nothing to register on.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** Ten of 144 heads reproduce the effect of intervening on all of them, which is sufficiency for the measured effect. No capability is measured, so this is sufficiency for the estimand rather than for a behavior.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Not tested.** Off-target effects have nowhere to register. The primary measure and all three alternates are computed over a two-element outcome set, so an intervention that wrecked the model's syntax or its factual recall would leave every reported number untouched. The paper sees this and says so — it notes the distributions could be extended to the full vocabulary, that doing so would reveal consequences it currently cannot see, and then declines.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Weak.** The primary confound: removing gender information (debiasing) may simply make the model *worse at predicting* in gendered contexts, producing apparent debiasing as a side effect of degradation. Without controlling for overall quality loss, the debiasing effect is confounded.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Partial | Benchmark-specific reduction |
| I2 Sufficiency | Not demonstrated | Gender information does not equal bias specifically |
| I4 Specificity | Weak | Removes knowledge with bias |
| M1 Reliability | Weak | Benchmark-specific; does not generalize |
| I7 Confound control | Weak | Degradation confound |

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

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Not tested.** Causal mediation is the single instrument throughout, so every agreement is the method with itself. No second intervention family has been run. This is one of the two criteria capping the claim.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Sometimes.** Scaling the projection magnitude produces graded effects. But the useful range (enough to reduce bias, not enough to degrade performance) is narrow and context-dependent.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Confirmed.** The claim is tested on three prompt sets that were not built the same way: a seventeen-template generator over 169 professions, and two Winograd-schema coreference sets whose bias lives in which of two entities a pronoun resolves to. The result that carries the paper — sparsity of the indirect effect — holds on all of them. The paper also refuses to pool: it reports that effect magnitudes differ across the sets and names the construction property it attributes the difference to.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Partial.** Bias exists across architectures. Whether the same debiasing technique transfers is model-dependent.

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Partial | Changes outputs; not always correctly |
| I4 Specificity | Weak | Bias + knowledge inseparable |
| E5 Graded response | Variable | Benchmark-specific |
| E2 Prompt generalization | Weak | Brittle across settings |
| E4 Cross-model generalization | Partial | Technique transfer variable |

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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Weak.** Different bias benchmarks give different answers. The measurement of "bias" itself is unreliable across metrics.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Confirmed.** Invariance is tested along more axes than the criterion usually sees — five GPT-2 sizes, three datasets, two mediator families, four metrics, six model families — and the answer differs by axis. Inside the autoregressive family the sparsity and decomposability patterns hold, and the paper says which conclusion that licenses. Outside it the neuron-level pattern does not transfer, and the paper reports the failure in its own words and declines to explain it.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Pass.** A randomly initialized GPT2-small, matched on everything but training, is carried through the identical pipeline at both the head and the neuron level as a negative control.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Unknown.** Can the metric distinguish between "the model is unbiased" and "the model has learned to hide bias from the benchmark"? Gonen & Goldberg's "lipstick on a pig" result suggests the latter is common.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Poorly understood.** What level of bias-benchmark performance constitutes "debiased"? There is no agreed threshold.

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Weak | Benchmark disagreement |
| M6 Invariance | Weak | Results don't transfer across benchmarks |
| M2 Baseline separation | Partial | Direction separates; bias/knowledge don't |
| M5 Sensitivity | Unknown | Hiding vs. removing |
| M4 Calibration | Poorly understood | No agreed threshold |
| C3 Convergent validity | Weak | Multi-dimensional construct, 1D metrics |

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Confirmed.** The level is declared twice and in the places where it binds. The introduction names the analysis as structural-behavioral and says which half each claim comes from — components on the structural side, model outputs on the behavioral side. The methodology fixes the admissible mediators before any measurement and defines the response variable as a function of the model's predictions rather than of its internal state. No claim sits at a level the declaration fails to cover.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Pass.** Every headline claim — sparsity, synergy, decomposition into direct and indirect parts — is a property of the measured effect distribution, at the implementational-topographic level the paper declares.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Weak.** The primary alternative: bias is not a localized property but an emergent property of the full model — a consequence of training data distribution reflected throughout all parameters. Under this alternative, surgical removal is fundamentally impossible, and apparent debiasing is actually degradation-masking. This alternative is not excluded.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Pass.** Each restriction is declared where it binds rather than collected at the end, and components carry no names at all — only an index and an estimand. The scope inflation belongs to the downstream debiasing literature, not to the audited claim.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Partial | Mixed levels |
| V2 Level-evidence match | Partial | Representational evidence for implementational claims |
| V3 Alternative level | Weak | Distributed bias alternative not excluded |
| V5 Scope declaration | Often violated | "Debiased" exceeds evidence |

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
