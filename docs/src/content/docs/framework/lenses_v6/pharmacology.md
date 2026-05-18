---
title: "Pharmacology"
description: "The external validity lens: how much effect, at what strength, on what target, with what margin?"
---

# The Pharmacology Lens

This lens asks one question: **how much effect, at what strength, on what target, and with what margin of generalization?**

When we ablate a circuit component and report a behavioral change, we are making a claim with a structure pharmacology recognized over a century ago: *this intervention, at this site, through this mechanism, produces this effect.* The entire history of drug development is a history of learning what goes wrong when we report that claim incompletely — when we skip the step of confirming the drug reaches its target, when we report one dose instead of a curve, when we measure on-target effects without measuring off-target ones.

[External validity](/framework/validity-types_v4/external) is the pharmacological question: not whether the effect is real (that is [internal validity](/framework/validity-types_v4/internal)), but how much effect, at what strength, on what target, with what margin. A circuit that satisfies every internal-validity criterion at a single intervention strength can still fail here. The effect may not scale. It may disappear in a different model. Its absolute magnitude may be too small to support the computational story.

There is also a disanalogy worth naming. A drug in a living organism faces degradation, metabolism, plasma protein binding, and blood-brain barrier transport. A steering vector or ablation in a language model does not — the intervention reaches its target instantaneously and completely (by construction), so target engagement is trivially satisfied at the level of the ablation. What is not trivially satisfied is *selective* engagement: ablating a head removes everything it does, not just the computation we are interested in. In MI, the target is a specific subspace or computation — not the component as a whole. Ablating an entire head removes everything it does, not just the one computation we care about. Confirming that an intervention engages the specific target subspace (rather than the full component) is the MI analog of confirming target selectivity in pharmacology.

## Key Distinctions

### Affinity is not efficacy

Stephenson (1956) drew a core distinction: a drug can bind to a receptor without producing a functional response. Binding (affinity) is necessary but not sufficient for effect (efficacy). A ligand — any molecule that binds to a receptor — with high affinity and zero efficacy occupies the site, prevents other drugs from acting, but does nothing itself.

In MI: a component can be highly active during a task — large activations, high attribution scores, strong attention to the relevant tokens — without being causally responsible for the behavior. Participation is affinity. Causation is efficacy. Activation patching measures something closer to affinity (is this component engaged?). Ablation measures something closer to efficacy (does removing it change the output?). But even ablation conflates efficacy with the system's compensatory capacity — which is where Black & Leff come in.

### The system compensates

Black & Leff (1983) formalized what pharmacologists already knew: the observed effect of a drug is not just a property of the drug. It is a joint property of the drug's intrinsic activity and the system's capacity to amplify or dampen that activity. A highly efficacious drug in a system with massive receptor reserve will produce a maximal response even at low occupancy. A moderately efficacious drug in a system with no reserve will produce a submaximal response even at full occupancy.

In MI: ablating a genuinely load-bearing head can produce a small behavioral effect if the network has backup mechanisms that compensate. Conversely, ablating a marginally important head can produce a large effect if the network happens to have no redundancy at that point. The observed effect magnitude reflects both the component's contribution and the network's compensatory capacity — and a single ablation cannot separate the two. This is why sufficiency tests (complement ablation, circuit-only runs) are necessary alongside necessity tests: they probe the system's reserve.

### The metric is part of the finding

Kenakin (2004) identified functional selectivity: the same drug, acting on the same receptor, produces different downstream effects depending on which signaling pathway is measured. A ligand can be an agonist on one readout and an antagonist on another. The choice of assay — a standardized test procedure that measures a specific biological response — is not neutral. It determines what you find.

In MI: the same head, under the same ablation, can appear essential or redundant depending on whether you measure logit difference, KL divergence, top-1 accuracy, or cross-entropy. Miller et al. (2024) demonstrated this empirically — the IOI circuit's faithfulness varies dramatically across ablation methods. But it also varies across metrics at a fixed method. A circuit that recovers 87% of logit difference may recover only 60% of full distributional KL. These are not contradictions; they are different functional readouts of the same intervention. The metric must be named as part of the claim, not treated as an interchangeable measurement of "effect."

### Naming requires criteria

Rang (2006), summarizing IUPHAR's receptor classification framework, argued that naming a new drug target requires standardized evidence: a selective ligand that binds it, a functional assay that demonstrates its downstream effect, and ideally genetic validation (knockout or knockin) — though there is no direct analog of genetic validation in MI. The closest equivalent is cross-architecture evidence, which tests whether the mechanism appears in models with different training, initialization, and structure. Without these, a putative target is a hypothesis, not an entity.

In MI: naming a circuit ("the IOI circuit," "induction heads," "a deception feature") is an act of classification. The pharmacological standard asks: what is the selective ligand (the intervention that engages this circuit and not others)? What is the functional assay (the behavioral metric that tracks this circuit's contribution)? What is the cross-architecture evidence (does the mechanism appear in other model families)? A circuit that has been named but lacks a selective intervention, a specific functional readout, and any form of cross-model validation is a hypothesis with a label — not an established entity.

## Analytical Constructs

### The dose-response curve

The signature artifact of pharmacological evaluation is the dose-response curve: a plot of intervention strength (x-axis) against behavioral effect (y-axis). In MI, the x-axis is the interpolation parameter α (from 0 = no intervention to 1 = full ablation), and the y-axis is the behavioral metric (logit difference, accuracy, KL divergence).

The curve reveals structure that no single point can:

- **Threshold** — the intervention strength at which the effect first becomes detectable. A mechanism with a high threshold may be buffered by redundancy.
- **EC₅₀** — the strength producing half-maximal effect. This is the sensitivity of the mechanism to disruption.
- **Plateau** — the maximum effect achievable by intervening on this component alone. A plateau below 100% means other components contribute.
- **Off-target onset** — the strength at which behaviors unrelated to the target task begin degrading. This is the specificity boundary.
- **Therapeutic window** — the gap between threshold (mechanism engages) and off-target onset (unrelated behaviors degrade). A wide window means the intervention is specific; a narrow window means any effective intervention also causes collateral damage.

A circuit with a wide therapeutic window — large gap between threshold and off-target onset — is one where you can modulate the target behavior without breaking other things. A circuit with no therapeutic window (threshold ≈ off-target onset) cannot be cleanly separated from the network's general processing.

To construct the curve: sweep α from 0 to 1 in increments (e.g., 0.05), measuring both the target metric and at least one off-target metric at each point. Plot both. The shape tells you more about the mechanism than any single ablation ever could.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Hill, "The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves"](https://doi.org/10.1113/jphysiol.1910.sp001386) | 1910 | Pharmacology | **Sigmoidal dose-response (Hill equation)** — the standard mathematical form for graded responses, parameterized by $E_{\max}$, $\text{EC}_{50}$, and Hill coefficient $n$ |
| [Clark, "The reaction between acetyl choline and muscle cells"](https://doi.org/10.1113/jphysiol.1926.sp002299) | 1926 | Pharmacology | **Quantitative dose-response** — effects scale monotonically with concentration; threshold and plateau are distinct quantities; a single dose is not a curve |
| [Gaddum, "The quantitative effects of antagonistic drugs"](https://doi.org/10.1113/jphysiol.1937.sp003535) | 1937 | Pharmacology | **Competitive antagonism and selectivity** — selectivity requires measuring on-target and off-target effects at matched doses |
| [Stephenson, "A modification of receptor theory"](https://doi.org/10.1111/j.1476-5381.1956.tb00081.x) | 1956 | Pharmacology | **Efficacy vs affinity** — a ligand can bind its receptor (affinity) without producing a functional effect (efficacy); binding is not activation |
| [Black & Leff, "Operational models of pharmacological agonism"](https://doi.org/10.1098/rspb.1983.0093) | 1983 | Pharmacology | **Operational model of agonism** — separates the intrinsic property of the drug from the system's capacity to amplify or compensate; observed effect reflects both |
| [Kenakin, "Efficacy as a vector"](https://doi.org/10.1016/j.tips.2004.02.012) | 2004 | Pharmacology | **Functional selectivity (biased agonism)** — the same ligand produces different downstream effects depending on which pathway is measured; the readout is part of the result |
| [Rang, "The receptor concept: pharmacology's big idea"](https://doi.org/10.1038/nrd2009) | 2006 | Pharmacology | **Target classification criteria** — naming a new drug target requires standardized evidence (selective ligand, functional assay, genetic validation); naming without criteria is labeling without content |
| [Miller, Chughtai & Saunders, "Transformers are uninterpretable with myopic methods"](https://arxiv.org/abs/2407.08734) | 2024 | Mechanistic Interpretability | **Faithfulness as a joint property** — circuit faithfulness is a function of circuit $\times$ ablation method; absolute magnitude must be reported with the method named |

## Validity type: [External validity](/framework/validity-types_v4/external)

> **The Hill equation:** $E = E_{\max} \cdot C^n / (C^n + \text{EC}_{50}^n)$, where $E$ is the effect, $C$ is the intervention strength, $E_{\max}$ is the maximal response, $\text{EC}_{50}$ is the strength producing half-maximal effect, and $n$ controls the curve's steepness. In MI: $C$ is the ablation fraction, steering multiplier, or patching proportion; $E$ is the behavioral metric.

Ablation in MI is typically binary: zero out a head entirely, or replace it with its mean. But interventions can be graded. Instead of fully zeroing a component, interpolate — replace the activation with $\alpha \cdot \text{ablated} + (1 - \alpha) \cdot \text{original}$ and sweep $\alpha$ from 0 to 1. For steering vectors, vary the multiplier. For IIA patching, interpolate partway toward the counterfactual activation. Each value of $\alpha$ is a point on a dose-response curve.

The shape of this curve contains information that no single point can reveal. An effect that appears only at extreme intervention strengths suggests force-override of the network rather than mechanism-specific modulation. An effect that saturates at low strengths suggests a highly specific mechanism with low redundancy. An effect that scales smoothly between threshold and plateau, with a wide gap before off-target degradation begins, is the signature of a mechanism that is genuinely load-bearing.

A wide therapeutic window — a large gap between threshold and off-target onset — is what separates "this component implements a specific mechanism" from "this component was hit hard enough to break something." Without the curve, that distinction is invisible.

## The criteria

### Intervention reach

Before interpreting what an ablation reveals, we need to confirm that the ablation actually engaged the computation we intended to engage.

When we zero-ablate an attention head, we zero its entire output — every subspace it contributes to, every downstream computation that depends on it. If the head carries three independent computations and we are interested in one, the ablation has engaged all three. The behavioral change we observe is the joint effect of disrupting all of them, not the specific contribution of the target computation.

Mean ablation has its own version of this problem. For components with large mean activations — early-layer heads, high-variance MLP neurons — replacing the activation with its mean leaves substantial residual signal at the target. The ablation is incomplete. The intervention has not fully reached the computation we intend to isolate.

Confirming intervention reach means measuring the activation delta at the target component itself, reported separately from the downstream behavioral outcome. A logit lens projection at the hook point shows whether the component's contribution to the residual stream changed in the predicted direction. An occupancy curve sweeping intervention strength shows how the component's internal state responds to graded modulation.

**What to report.** Activation delta at the target component. Reported separately from the behavioral outcome. If the ablation is known to remove more than the claimed target, this should be stated.

### Graded response

A single ablation strength is a single point on a curve. It is not the curve.

A mechanism-specific intervention should produce a graded behavioral response: negligible at very low strengths, growing monotonically, saturating at high strengths before off-target degradation begins. The threshold, the plateau, and the gap between the plateau and off-target onset are three distinct quantities. A paper that reports only one point cannot characterize any of them.

The dose-response curve to report:

$$E(\alpha) = E_{\max} \cdot \frac{\alpha^n}{\alpha^n + \alpha_{50}^n}$$

where $\alpha$ is the intervention strength (ablation fraction, steering multiplier), $\alpha_{50}$ is the strength producing half-maximal effect, and $n$ controls steepness. We sweep $\alpha$ across at least five and preferably seven values, spanning from below the expected threshold to above the expected plateau. We plot the on-task metric and an off-task collateral-damage metric on shared axes. We report $\alpha_{\text{thresh}}$ (the smallest $\alpha$ with a reliable on-task effect), $\alpha_{\text{plat}}$ (the $\alpha$ at which on-task effect saturates), and the **therapeutic window** $[\alpha_{\text{thresh}}, \alpha_{\text{off}}]$ where $\alpha_{\text{off}}$ is where off-target degradation begins.

A wide therapeutic window is evidence for mechanism specificity. A narrow or absent window — where the on-task threshold is near the off-target onset — is evidence that the intervention is not selective.

**What to report.** At least five intervention strengths. $\alpha_{\text{thresh}}$, $\alpha_{\text{plat}}$, and $\alpha_{\text{off}}$ identified or stated as not yet determined. Both on-task and off-task metrics on the same axes.

<details class="worked-example">
<summary>Worked example: steering with a learned direction</summary>

Consider a direction $w \in \mathbb{R}^{d_{\text{model}}}$ identified via DAS as encoding subject-verb agreement (SVA) information. We add $\alpha \cdot w$ to the residual stream at the MLP hook point for each token in a held-out SVA evaluation set. We evaluate accuracy on the SVA task (on-task) and perplexity on a general-language evaluation corpus (off-task) at $\alpha \in \{0.5, 1, 2, 5, 10, 20\}$.

Suppose we observe:

| $\alpha$ | SVA accuracy (on-task) | Perplexity (off-task) |
|---|---|---|
| 0 | 62.1% | 18.4 |
| 0.5 | 63.0% | 18.5 |
| 1 | 65.4% | 18.7 |
| 2 | 69.2% | 19.1 |
| 5 | 71.8% | 21.3 |
| 10 | 71.9% | 26.7 |
| 20 | 68.1% | 41.2 |

From this: $\alpha_{\text{thresh}} \approx 1$ (the smallest strength with a noticeable on-task effect), $\alpha_{\text{plat}} \approx 5$ (where on-task accuracy saturates), $\alpha_{\text{off}} \approx 5$ (where off-task perplexity begins rising meaningfully). The therapeutic window is narrow — $[1, 5]$. The selectivity index at $\alpha = 2$ is $(69.2 - 62.1) / (19.1 - 18.4) = 7.1/0.7 \approx 10$, which is at the pharmacological selectivity threshold. The direction has a modest, specific effect on SVA, but no headroom above the plateau before off-target degradation.

A paper reporting only the $\alpha = 20$ result would report on a point outside the therapeutic window where the intervention is simultaneously damaging the model generally.
</details>

### Selectivity

Selectivity quantifies whether the intervention has greater effect on the claimed target than on everything else.

The selectivity index:

$$SI = \frac{E_{\text{on-task}}}{E_{\text{off-task}}}$$

measured at the on-task threshold $\alpha_{\text{thresh}}$. In drug development: $SI < 10$ is not selective, $SI > 100$ is acceptable, $SI > 1000$ is excellent. These thresholds are conventions, not laws — we use them as orientation.

The choice of off-task benchmark matters more than the threshold. The IOI circuit tested against arithmetic is a trivially easy bar — a general-purpose bottleneck component would also score high on selectivity against an unrelated task. The informative comparison is a related task: IOI against subject-verb agreement, both of which require tracking syntactic roles across positions. A circuit that is selective against related tasks is more specifically defined than one that is only selective against unrelated ones.

**What to report.** $SI$ at $\alpha_{\text{thresh}}$ with at least one related-task comparison. Random-component selectivity — the same ablation applied to a size-matched random set of components — as a baseline for the $SI$ value.

![Selectivity Index across circuit claims — bar chart with SI = 10 threshold](/figures/selectivity_index_minimal.svg)

### Effect magnitude

A statistically reliable and selective effect can still be too small to support the computational story being told.

The recovery fraction alone is not enough. An 87% recovery fraction means something very different when the full-model logit difference is 3.56 (the IOI baseline under Wang et al.'s setup) than when it is 0.05. Both have the same percentage, but the first is a large absolute effect and the second is noise.

The minimum report for effect magnitude is three quantities:

$$M_{\text{circuit}}, \quad M_{\text{full}}, \quad R = \frac{M_{\text{circuit}}}{M_{\text{full}}}$$

where $M$ is the behavioral metric (logit difference, probability, accuracy). And a fourth: the ablation method, named as part of the claim. [Miller, Chughtai, and Saunders (2024)](https://arxiv.org/abs/2407.08734) demonstrated that the IOI circuit's 87% recovery is under mean ablation with the Wang et al. prompt set; under resample ablation or with different prompt distributions, the number changes substantially.

A finding where $R > 0.8$ and $M_{\text{full}}$ is itself large relative to the random baseline supports a strong computational claim. A finding where $R > 0.8$ but $M_{\text{full}} \approx M_{\text{random}}$ does not — the circuit is recovering a large fraction of a small signal.

**What to report.** $M_{\text{circuit}}$, $M_{\text{full}}$, $R$, and the ablation method. If multiple ablation methods are used, report $R$ for each.

### Robustness

A circuit finding that holds on a single prompt template is a finding about that template.

Robustness asks whether the effect survives prompt paraphrase, alternative template structures, and transfer to other model sizes within the same family. The IOI circuit was originally evaluated on the template "When Mary and John went to the store, John gave a drink to…" Whether its faithfulness holds on paraphrased versions — "Mary and John visited the shop, and the one who handed over the drink was…" — is a robustness question, not an internal validity question. The mechanism may be real and the paraphrase result may differ; both can be true. Reporting both is more informative than reporting only the original.

Cross-checkpoint replication (does the circuit appear at multiple points in training history?) and cross-scale transfer within a model family (do the same heads in GPT-2 Medium carry the same computation?) are also robustness questions.

**What to report.** At least one held-out prompt distribution not used during discovery. Cross-checkpoint or cross-scale results reported if available.

### Cross-architecture generalization

The strongest form of external validity asks whether the mechanism appears in a model with a different tokenizer, training corpus, and depth.

This is kept separate from robustness because the evidence required is qualitatively different: matching circuits across families requires an explicit, falsifiable matching criterion stated before testing. The criterion can be functional (candidate heads show the same selectivity for indirect objects under activation patching), structural (weight-space cosine similarity to the reference circuit heads exceeds a threshold), or both. Without a pre-stated criterion, post-hoc matching can always be made to succeed.

The absence of cross-architecture evidence does not invalidate a finding. It bounds its scope. The recommended language is "in the GPT-2 family, a circuit consistent with IOI" rather than "language models implement IOI through…" unless cross-architecture evidence exists.

**What to report.** The matching criterion, stated before testing. Whether an analogous circuit was found and in which model families.

## Evidence patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Single ablation, no curve | One point, not a mechanism | "At strength $\alpha$, ablation produces $\Delta M$" |
| Curve, no selectivity | Magnitude, not specificity | "Dose-dependent effect; selectivity untested" |
| High $R$, small $M_{\text{full}}$ | Large fraction of a small signal | "Recovery $R = X\%$ of baseline $M_{\text{full}} = Y$" |
| Selective within family | Family-specific generalization | "Generalizes within [family]; cross-architecture untested" |
| Cross-architecture match | Broad generalization | "Mechanism found in [families] under [criterion]" |

## Verdicts

- **Proposed → Causally suggestive:** Requires I1 (necessity) from the neuroscience lens. Pharmacology does not gate the first upgrade.
- **Causally suggestive → Mechanistically supported:** Requires at minimum E4 (effect magnitude) — the absolute effect must be large enough that the computational story is coherent.
- **Mechanistically supported → Triangulated:** Requires E2 (graded response) and E3 (selectivity). Without a dose-response curve, the claimed mechanism's strength cannot be characterized.
- **Triangulated → Validated:** Requires E5 (robustness) and, ideally, E6 (cross-architecture generalization).

## Protocol

For a proposed circuit $C$ and behavior $B$:

1. **Intervention reach.** Measure activation delta at $C$. Report separately from behavioral outcome.
2. **Graded response.** Sweep at least five intervention strengths from below threshold to above plateau. Report $\alpha_{\text{thresh}}$, $\alpha_{\text{plat}}$, $\alpha_{\text{off}}$, and the therapeutic window.
3. **Selectivity.** Compute $SI$ at $\alpha_{\text{thresh}}$ with at least one related-task comparison. Include random-component $SI$.
4. **Effect magnitude.** Report $M_{\text{circuit}}$, $M_{\text{full}}$, and $R$. Name the ablation method.
5. **Robustness.** Held-out prompt evaluation. Cross-checkpoint and cross-scale if available.
6. **Cross-architecture generalization.** State matching criterion before testing. Report findings.

A skipped step must be named in the verdict.

## Case studies

For full worked examples applying all five lenses (including external validity) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) — method-conditional faithfulness; single-dose reporting
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) — cross-architecture generalization demonstrated
- [Greater-Than](/framework/lenses_v6/examples/examples-greater-than) — effect magnitude well-characterized
- [Successor Heads](/framework/lenses_v6/examples/examples-successor-heads) — cross-domain generalization as convergent evidence
- [Copy Suppression](/framework/lenses_v6/examples/examples-copy-suppression) — unusually clean selectivity
- [Grokking](/framework/lenses_v6/examples/examples-grokking) — full dose-response in toy scope
- [Knowledge Neurons](/framework/lenses_v6/examples/examples-knowledge-neurons) — strong intervention, weak selectivity
