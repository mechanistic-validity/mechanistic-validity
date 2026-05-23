---
title: "Neuroscience"
description: "The internal validity lens: does the evidence establish implementation, not just participation?"
---

# The Neuroscience Lens

This lens asks one question: **does the component implement the computation, or just participate in it?**

Activating during a task, or even contributing causally to a task, is not enough. A bottleneck component is causally necessary for every computation that routes through it, but it does not implement any of them in particular. Systems neuroscience developed a checklist — necessity, sufficiency, specificity, consistency, confound control — to draw exactly this distinction. The same checklist applies to circuits in neural networks without modification.

The metrics of MI — ablation, activation patching, path patching, steering — are direct analogs of the neuroscience techniques. What MI sometimes lacks is the interpretive framework that makes the results meaningful. An ablation study that reports only necessity, without specificity controls, establishes less than it appears to.

There is also a disanalogy worth naming. In biological systems, lesion studies observe a system that cannot adapt to the damage — the deficit is permanent. Neural network ablation is instantaneous: the model never gets to recalibrate around the missing component. This makes ablation a weaker test than it appears, because the observed degradation may partly reflect the network's inability to route around the damage, not the component's irreplaceability. The stronger test is Craver's mutual manipulability in full: not just "does ablating the component degrade the behavior?" (bottom-up) but "does training away the behavior change the component?" (top-down). If you fine-tune a model to remove IOI capability, do the name-mover heads lose their copying structure in $W_{OV}$? This top-down direction is rarely tested in MI, but it is what constitutive relevance actually requires.

Neuroscience also offers a prior question that MI tends to skip: how do you define the units in the first place? Before testing whether a brain region implements a function, you need a parcellation — a principled decomposition of cortex into regions. [Glasser et al. (2016)](https://doi.org/10.1038/nature18933) define 180 cortical areas by convergence of architecture, function, connectivity, and topography across 210 subjects. An area boundary is placed where multiple independent signals agree. The analog in MI is circuit discovery: activation patching, EAP, weight-based methods, and probing each propose a set of components. When these methods agree, the circuit boundary is well-defined. When they disagree, the "circuit" may be an artifact of the discovery method — the same problem neuroscience solved with multimodal parcellation. Similarly, [Gallego et al. (2017)](https://doi.org/10.1146/annurev-neuro-092917-025811) argue that the right unit of analysis for motor cortex is the population-level neural mode, not the individual neuron. Most variance in neural activity is captured by a few latent dimensions. The analog in MI is the argument for circuits and features over individual weights or neurons: the computational unit is the pattern, not the parameter.

:::note
For the full metrics and protocols reference, see [Neuroscience -- Metrics & Protocols](/framework/lenses/core/neuroscience-metrics).
:::

## Key Distinctions

### Single vs double dissociation

Shallice (1988) formalized the logic of dissociation in neuropsychology. A single dissociation — lesioning component A impairs function X — establishes that A contributes to X. But it does not establish specificity, because A might contribute to everything. A double dissociation — lesioning A impairs X but not Y, and lesioning B impairs Y but not X — establishes that A and B are functionally distinct and that neither is simply a general-purpose bottleneck.

In MI: most circuit evaluations perform single dissociations. We ablate the IOI circuit and show that IOI performance drops. But we rarely test the converse — does ablating a different circuit (e.g., the Greater-Than circuit) leave IOI performance intact? Without the second leg of the dissociation, we cannot distinguish "this circuit implements IOI" from "this circuit is a general-purpose component that many tasks route through." Double dissociation is the test for specificity (I3) that the field most consistently skips.

### Lesion vs stimulation

Neuroscience distinguishes two intervention types: lesions (destroy tissue, test necessity) and stimulation (inject current, test sufficiency). These give different and complementary information. A region can be necessary without being sufficient (it provides input but doesn't compute the output), and sufficient without being necessary (a backup exists).

In MI: ablation is a lesion study — it tests necessity. Activation addition (steering vectors, patching clean activations into corrupted runs) is a stimulation study — it tests sufficiency. Most MI papers report only ablation. Adding stimulation evidence (can we restore the behavior by injecting signal into the circuit?) closes the gap between "this component matters" and "this component implements the computation."

### Localization vs distributed processing

A century of neuroscience debate about localization (Broca) vs distributed processing (Lashley) resolved into a nuanced answer: some functions are highly localized, others are distributed, and the level of analysis determines which you find. Gallego et al. (2017) showed that population-level neural modes — not individual neurons — are often the right unit of analysis for motor control.

In MI: the equivalent question is whether a computation lives in specific heads (localized) or is distributed across many components. The answer depends on the granularity of analysis. A circuit found by activation patching at the head level may appear localized, while the same computation analyzed at the neuron or subspace level appears distributed. The level of analysis is a methodological choice that shapes the finding, not a neutral window onto ground truth.

### Structural vs functional connectivity

Two brain regions can be structurally connected (axons linking them) without being functionally connected during a specific task (no task-relevant information flows between them). Conversely, regions without direct structural connections can be functionally coupled through intermediaries.

In MI: two attention heads can be compositionally connected (one's output is in the other's input subspace) without actually passing task-relevant information during the computation we care about. Structural connectivity (weight-space composition scores, OV circuit analysis) and functional connectivity (activation correlation during task performance, path patching) can diverge. A circuit claim based only on structural connectivity ("these heads compose") is weaker than one grounded in functional connectivity ("task-relevant information flows along this path during this computation").

## Analytical Constructs

### The dissociation matrix

The signature artifact of neuroscientific evaluation is the dissociation matrix: a k-circuits × k-tasks grid where each cell records the effect of lesioning circuit $i$ on task $j$.

$$D_{ij} = \Delta\text{performance}(\text{task}_j \mid \text{ablate circuit}_i)$$

The ideal pattern for a set of well-defined circuits is diagonal dominance: each circuit's ablation strongly impairs its own task (large $D_{ii}$) and minimally affects other tasks (small $D_{ij}$ for $i \neq j$). This is double dissociation generalized to k circuits simultaneously.

The matrix reveals structure that individual ablation studies cannot:

- **Diagonal entries** ($D_{ii}$) — necessity of each circuit for its claimed task. These are what most papers report.
- **Off-diagonal entries** ($D_{ij}, i \neq j$) — cross-talk. These are what most papers skip. A row with uniformly high values means the "circuit" is actually a general bottleneck, not a task-specific mechanism.
- **Column patterns** — if a task is impaired by ablating many different circuits, the task may recruit distributed processing rather than a localized mechanism.
- **Row patterns** — if a circuit's ablation impairs many tasks, it is a shared resource, not a task-specific module.

To construct the matrix: identify k candidate circuits and k tasks. For each circuit, ablate it and measure all k tasks. Fill the k×k grid. Inspect for diagonal dominance. Report the ratio $D_{ii} / \text{mean}(D_{ij}, j \neq i)$ as a specificity index for each circuit.

A perfect diagonal matrix (each circuit affects only its own task) is the idealized case. Real matrices will have off-diagonal entries — the question is whether the diagonal dominates or whether the pattern is diffuse. The matrix makes this visible at a glance.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Shallice, *From Neuropsychology to Mental Structure*](https://doi.org/10.1017/CBO9780511526817) | 1988 | Neuropsychology | **Double dissociation** — two functions dissociate iff lesioning A impairs X not Y, and lesioning B impairs Y not X |
| [Woodward, *Making Things Happen*](https://global.oup.com/academic/product/making-things-happen-9780195189537) | 2003 | Philosophy of Science | **Invariant difference-making** under intervention — the "holding fixed" clause is where the real work happens |
| [Craver, *Explaining the Brain*](https://global.oup.com/academic/product/explaining-the-brain-9780199568222) | 2007 | Neuroscience / Philosophy | **Constitutive relevance** and mutual manipulability — implementation requires that intervening on the component changes the behavior *and* vice versa |
| [Laird et al., "Large-scale automated synthesis of human functional neuroimaging data"](https://doi.org/10.1162/jocn_a_00077) | 2011 | Neuroimaging | **Functional characterization of data-driven components** — ICA identifies networks bottom-up, behavioral taxonomy characterizes them top-down |
| [Glasser et al., "A multi-modal parcellation of human cerebral cortex"](https://doi.org/10.1038/nature18933) | 2016 | Neuroimaging | **Multimodal parcellation** — brain region boundaries defined by convergence of architecture, function, connectivity, and topography; a classifier trained on multimodal fingerprints identifies regions in new subjects |
| [Gallego et al., "Neural manifolds for the control of movement"](https://doi.org/10.1016/j.neuron.2017.05.025) | 2017 | Neuroscience | **Neural manifolds** — population-level patterns (neural modes), not individual neurons, are the computational units; the right level of analysis determines what you can discover |
| [Miller, Chughtai & Saunders, "Transformers are uninterpretable with myopic methods"](https://arxiv.org/abs/2407.08734) | 2024 | Mechanistic Interpretability | **Faithfulness as joint property** — faithfulness is a function of circuit × ablation method; ablation type is part of the claim |

## Validity type: [Internal validity](/framework/validity-types/internal)

> **Constitutive relevance ([Craver 2007](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001)):** A component is constitutively relevant to a mechanism if and only if intervening on the component changes the behavior, *and* intervening on the behavior changes the component's activity. This is strictly stronger than mere causal relevance.

A power line is causally relevant to a factory (cutting power stops production), but it is not constitutively relevant to any particular manufacturing process. In MI, a layer-0 residual stream position is causally necessary for every computation the model performs — ablating it degrades everything. That does not make it part of the IOI circuit, the Greater-Than circuit, or any task-specific mechanism.

The gap between "causally implicated" and "implements" is the reason this lens has five criteria rather than one.

For formal definitions, quantitative thresholds, and calibration data, see [Internal Validity — Formal Specification](/framework/validity-types/internal).

## Criteria

| Code | Criterion | What it asks | Page |
|---|---|---|---|
| I1 | Necessity | Does removing the component degrade the behavior? | [I1](/framework/criteria/internal/necessity) |
| I2 | Sufficiency | Does restoring or isolating the component reproduce the behavior? | [I2](/framework/criteria/internal/sufficiency) |
| I3 | Specificity | Is the effect selective for the claimed function, not generic disruption? | [I3](/framework/criteria/internal/specificity) |
| I4 | Consistency | Does the effect replicate across prompts, seeds, and checkpoints? | [I4](/framework/criteria/internal/consistency) |
| I5 | Confound control | Is the effect not explained by collateral disruption to non-circuit components? | [I5](/framework/criteria/internal/confound-control) |

Necessity (I1) is the easiest to demonstrate and the easiest to overclaim. Sufficiency (I2) is the strongest and the most underreported. Specificity (I3) is what separates a circuit finding from a bottleneck finding.

### Necessity

Removing the component should degrade the behavior.

Necessity is the easiest criterion to demonstrate in practice: ablate a component, measure the change in output, and report the effect size. It is also the easiest to overclaim. A positive necessity result — "ablating head 9.1 reduces the logit difference on IOI prompts by 40%" — establishes that the component contributes causally to the behavior under the tested conditions. It does not establish that the component is the locus of the computation, that it is unique in its contribution, that it is task-specific, or that the behavior cannot be produced without it.

The distinction between causal contribution and implementation is the entire point of having five criteria rather than one. A residual stream position in layer 0 is causally necessary for every computation the model performs — ablating it degrades everything. That does not make it part of the IOI circuit, the Greater-Than circuit, or any particular task-specific mechanism. Necessity tells us that the component matters; it does not tell us what it does or why it matters.

**What necessity establishes.** The component contributes causally to the behavior under the tested conditions and the tested ablation method.

**What necessity does not establish.** Localization, uniqueness, task-specificity, sufficiency, or that the component is constitutively relevant rather than an upstream prerequisite.

#### Ablation semantics

The semantics of an ablation study depend on the counterfactual distribution — what we replace the component's output with. Different ablation types answer different questions, and conflating them is a common source of invalid inference.

**Zero ablation** replaces the component's output with the zero vector. This answers: "what happens if this component contributes nothing?" The problem is that a zero vector is typically off the manifold of activations the model has learned to process. Downstream components may respond to the out-of-distribution input rather than to the absence of the ablated component's contribution.

**Mean ablation** replaces the component's output with its mean activation across a reference dataset. This answers: "what happens if this component contributes only its average effect, losing all input-dependent information?" Mean ablation stays closer to the training distribution than zero ablation, but it still makes the strong assumption that the mean activation is a neutral baseline.

**Resample ablation** (also called noising or corruption) replaces the component's output with its activation on a different input from the same distribution. This answers: "what happens if this component receives unrelated but in-distribution information?" Resample ablation is the closest analog to the neuroscientific ideal of lesion + stimulation: it removes the component's task-relevant signal while preserving the statistical properties of its output.

The practical consequence is that ablation type must be reported as part of the claim, not relegated to a methods appendix. "Head 9.1 is necessary for IOI (zero ablation)" and "Head 9.1 is necessary for IOI (resample ablation)" are different claims with different evidential force.

**Failure modes.** *Bottleneck confound* — the component is necessary because many computations route through it, not because it implements the target one. *Off-manifold disruption* — zero or mean ablation pushes activations outside the training distribution. *Hydra effect* ([McGrath et al. 2023](https://arxiv.org/abs/2312.09230)) — the mechanism is distributed, and removing one component triggers compensatory activation in another. *Single-metric coupling* — logit difference can degrade while accuracy is unchanged, or vice versa.

**Minimum reporting.** Ablation type and counterfactual distribution stated explicitly. Equal-size random-component baseline included. At least two behavioral metrics.

<details class="worked-example">
<summary>Worked example: IOI ablation under three counterfactual distributions</summary>

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) evaluate the IOI circuit's necessity primarily through mean ablation: each head's output is replaced with its mean over the dataset, and the change in logit difference $\Delta LD = LD_{\text{clean}} - LD_{\text{ablated}}$ is measured. Consider head 9.9 (a name-mover head). Under mean ablation, removing it reduces the logit difference by approximately 1.2 points on the clean distribution — a substantial effect.

Now consider the same ablation under the resample paradigm. We construct a counterfactual distribution by pairing each IOI prompt with a corrupted version where the subject and indirect object names are swapped. Resample ablation replaces head 9.9's output on the clean prompt with its output on the matched corrupted prompt. The effect size changes: the reduction in logit difference is now approximately 0.8 points, and the variance across prompts is tighter.

Under zero ablation, the effect is larger still — approximately 2.1 points — but this number is less interpretable because the zero vector is far from the manifold of activations that downstream heads have learned to process.

The three numbers tell different stories:

- Zero ablation ($\Delta LD \approx 2.1$): this head matters, but some of the effect may be distributional disruption.
- Mean ablation ($\Delta LD \approx 1.2$): removing input-dependent information from this head degrades IOI, controlling for the average contribution.
- Resample ablation ($\Delta LD \approx 0.8$): replacing this head's task-relevant signal with in-distribution but task-irrelevant signal still degrades IOI, the strongest evidence for necessity.

A complete necessity report includes all three (or at minimum, the resample number with the ablation type stated explicitly). Reporting only the zero-ablation number overstates the evidence.
</details>

### Sufficiency

Restoring, isolating, or stimulating the component should reproduce the behavior.

Where necessity asks "does removing this component break the behavior?", sufficiency asks the converse: "does this component, on its own or in isolation, produce the behavior?" Sufficiency is the stronger claim and the harder one to establish honestly.

The standard metric is activation patching in the restoration direction (also called denoising). We run the model on a corrupted input (where the behavior is absent), then patch in the clean activation of the proposed circuit, and measure how much of the behavior is recovered. The faithfulness score $F$ for a circuit $C$ on task $T$ is typically defined as:

$$F(C, T) = \frac{M_{\text{patched}}(C) - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}$$

where $M$ is the behavioral metric. $F = 1$ means the circuit fully restores the behavior; $F = 0$ means patching the circuit has no effect beyond the corrupted baseline.

A related and stronger test is *circuit isolation*: ablate everything *outside* the circuit and measure whether the behavior persists. Circuit isolation removes the possibility that the circuit's sufficiency depends on intact backup mechanisms.

**What sufficiency establishes.** The component contains enough causal state to drive the behavior in the tested context.

**What sufficiency does not establish.** That this is the model's natural route during unperturbed forward passes. That the result generalizes beyond the prompt family tested. That there is no alternative circuit with equal sufficiency.

**Failure modes.** *Backup-dependent sufficiency* — the circuit passes restoration patching only because intact backup mechanisms compensate for any imperfections. *Magnitude artifacts* — steering works only at activation magnitudes that simultaneously degrade unrelated capabilities. *Discovery-prompt overfitting* — the circuit is sufficient on the prompts used to discover it and fails on paraphrases.

**Minimum reporting.** Faithfulness score $F$ on held-out prompts. Isolation test where feasible. If steering is used, a sweep across at least five multiplier levels.

<details class="worked-example">
<summary>Worked example: induction head sufficiency via isolation</summary>

[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) claim that induction heads implement in-context learning of repeated sequences. The sufficiency claim is that induction heads, by themselves, are sufficient to produce the characteristic "copy the token that followed the previous occurrence" behavior.

We can test this with circuit isolation. Define the induction circuit as the set of induction heads plus the previous-token heads in earlier layers that compose with them. Ablate every other attention head in the model by replacing their outputs with their mean activations. Then present the model with a repeated random sequence: `[A, B, C, D, A, B, C, ...]` and measure whether it predicts `D` at the position after the second `C`.

If the isolated induction circuit predicts `D` with high probability (say, $F > 0.8$ compared to the full model), we have strong sufficiency evidence. Olsson et al. find that induction heads in two-layer attention-only models are almost perfectly sufficient ($F \approx 0.95$). In larger models, sufficiency is lower because the behavior is distributed across more components — an honest sufficiency result with the degradation explained by distributional redundancy.
</details>

### Specificity

The component should be selective for the claimed function and not produce the same effect through generic disruption.

Specificity is what separates "this head implements IOI" from "this head is used by many tasks, one of which is IOI." A component can be necessary and sufficient for a behavior while serving a general-purpose role — ablating it degrades everything, and patching it restores everything, but these results reflect the component's centrality in the network, not its task-specific computational role.

The logic of specificity comes from classical neuropsychology's double dissociation ([Shallice 1988](https://doi.org/10.1017/CBO9780511526817)). A single dissociation — lesioning region $A$ impairs function $X$ — is consistent with $A$ being a general-purpose prerequisite for all functions. A double dissociation — lesioning $A$ impairs $X$ but not $Y$, while lesioning $B$ impairs $Y$ but not $X$ — establishes that $A$ and $B$ are specifically associated with $X$ and $Y$. The crossed pattern rules out the possibility that either region is merely a bottleneck.

In MI, the double dissociation translates directly. Let $C_{\text{IOI}}$ be the proposed IOI circuit and $C_{\text{GT}}$ be the proposed Greater-Than circuit. Specificity requires:

1. Ablating $C_{\text{IOI}}$ degrades IOI performance but not Greater-Than performance.
2. Ablating $C_{\text{GT}}$ degrades Greater-Than performance but not IOI performance.

If ablating $C_{\text{IOI}}$ degrades both tasks equally, then $C_{\text{IOI}}$ is not specific to IOI — it is a shared resource.

The most common specificity control in current MI practice is the random-component baseline. For a circuit $C$ of size $|C| = k$, we sample a random set of $k$ heads and measure its faithfulness on the same task. The circuit's specificity is captured by the excess over baseline:

$$\Delta F = F(C, T) - F_{\text{rand}}(k, T)$$

**Failure modes.** *Baseline-free numbers* — reporting faithfulness without a random baseline renders the number uninterpretable. *Generic-capability capture* — induction heads, copy heads, and positional attention heads are necessary for many tasks and easily mistaken for task-specific circuits. *Easy contrasts* — using a control task that is trivially distinguishable from the target inflates apparent specificity.

**Minimum reporting.** Random-component baseline for every reported score. At least one collateral-damage measurement. If a double dissociation is claimed, both ablation directions reported.

### Consistency

The effect should replicate across contexts.

Consistency converts a result from a property of a specific prompt family, random seed, or checkpoint into a property of the model. An inconsistent result is not necessarily false — it is a claim of narrower scope than the original report suggests.

Consistency operates along several axes: cross-prompt replication (new templates and paraphrases), cross-checkpoint (different points in training), cross-seed (independently trained models), and cross-model (different families). Each axis provides different evidential value, and each requires its own matching criterion.

We recommend reporting consistency with bootstrap confidence intervals on the principal behavioral metric. For a circuit $C$ evaluated on $n$ prompts with metric values $m_1, \ldots, m_n$, the bootstrap 95% confidence interval is computed by resampling with replacement $B$ times (typically $B = 10{,}000$) and taking the 2.5th and 97.5th percentiles of the resampled means.

**Failure modes.** *One-distribution science* — the discovery prompts are reused for evaluation. *Single-seed generalization* — a result from one trained model is presented as a property of the architecture. *Loose cross-model matching* — qualitative resemblance without a quantitative criterion for what counts as "the same circuit."

**Minimum reporting.** At least two of: cross-prompt, cross-checkpoint, cross-seed replication. Bootstrap 95% confidence intervals on the principal metric.

### Confound control

The observed effect should not be explained by collateral disruption to non-circuit components.

Ablation is not a scalpel — it is a sledgehammer applied to one location. The damage may propagate. Mean ablation pushes activations toward their dataset mean, which may be off-manifold for specific inputs. The degradation observed could partly reflect distributional disruption rather than mechanism-specific removal. Confound control asks: could the effect be explained by something other than the component's computational role?

The strongest confound control is multi-method comparison: if the same circuit shows necessity under mean ablation, resample ablation, and path patching, the result is unlikely to be an artifact of any single method's distributional assumptions. A weaker but useful control is measuring whether ablation pushes activations off the data manifold — if downstream residual stream norms spike after ablation, distributional disruption is a plausible confound.

**Failure modes.** *Single-method reliance* — all evidence comes from one ablation type, leaving method-specific artifacts uncontrolled. *Cascading disruption* — ablation of one component disrupts inputs to downstream components, and the observed effect is downstream failure rather than mechanism-specific removal. *Activation scale confound* — the ablation changes activation magnitudes without changing directions, and downstream components fail because they are calibrated for a particular scale.

**Minimum reporting.** At least two ablation methods with different distributional assumptions. If only one method is used, a distributional integrity check (residual stream norm before/after ablation).

## What each criterion does and does not establish

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Necessity alone | Causal contribution | "Causally implicated" |
| Sufficiency alone | A capable route exists | "A sufficient route, not shown necessary" |
| Necessity + sufficiency, no specificity | General-purpose mechanism | "Real mechanism, not shown task-specific" |
| Necessity + specificity, no consistency | Prompt-family-specific mechanism | "Task-specific on the tested distribution" |
| All five criteria met | Strong implementation claim | "Implements [function] robustly" |

## Verdicts

Internal validity is the workhorse of the [verdict system](/framework/verdicts/):

- **Proposed → Causally suggestive:** Requires I1 (necessity). A single ablation result with a random-component baseline.
- **Causally suggestive → Mechanistically supported:** Requires I1 + I2 (necessity + sufficiency). The circuit must not only be necessary but capable of driving the behavior.
- **Mechanistically supported → Triangulated:** Requires I1–I5 complete, plus at least one criterion from another validity type.

Most published MI papers reach Causally suggestive. The gap between Causally suggestive and Mechanistically supported — sufficiency — is where most claims stall.

## Protocol

For a proposed circuit $C$ and behavior $B$, the following protocol operationalizes internal validity. Each step specifies both what to do and what the result means. A skipped step must be named in the verdict rather than silently omitted.

1. **Necessity.** Ablate $C$ using resample ablation with a matched counterfactual distribution. Compare the effect against an equal-size random-component baseline. Report at least two behavioral metrics. State the ablation type as part of the claim.

2. **Sufficiency.** Restoration-patch $C$ from corrupted to clean on held-out prompts not used during discovery. Report the faithfulness score $F$. Where feasible, run an isolation test (ablate everything outside $C$) and report the isolated faithfulness.

3. **Specificity.** Provide a random-component baseline for every reported score. Measure collateral damage on at least one unrelated task. If a double dissociation is claimed, report both ablation directions.

4. **Consistency.** Replicate across at least two of: prompt templates, training checkpoints, random seeds. Report bootstrap 95% confidence intervals on the principal metric.

5. **Confound control.** Apply at least two ablation methods with different distributional assumptions. If only one method is feasible, report a distributional integrity check.

Partial evidence is informative; unreported gaps are not.

## Case studies

For full worked examples applying all five lenses (including internal validity) to published claims:

- [IOI Circuit](/framework/examples/examples/examples-ioi) — strong I1/I2, weak I3/I5; method-conditional results
- [Induction Heads](/framework/examples/examples/examples-induction-heads) — reaches Mechanistically supported; path-level sufficiency
- [SAE Features](/framework/examples/examples/examples-sae-features) — necessity/sufficiency sometimes; bulk untested
- [Copy Suppression](/framework/examples/examples/examples-copy-suppression) — unusually clean specificity (I3)
- [Grokking](/framework/examples/examples/examples-grokking) — all five criteria pass (toy model)
- [Knowledge Neurons](/framework/examples/examples/examples-knowledge-neurons) — strong I1/I2, weak I3/I5
- [Probing Classifiers](/framework/examples/examples/examples-probing) — measurement without intervention; I1–I3 all untested
