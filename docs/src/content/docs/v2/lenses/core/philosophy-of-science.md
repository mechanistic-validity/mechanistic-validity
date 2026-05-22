---
title: "Philosophy of Science"
description: "The construct validity lens: is the entity you named a coherent construct?"
---

# The Philosophy of Science Lens

This lens asks one question: **is the entity you named a coherent construct?**

Every mechanistic claim names a theoretical entity — "the IOI circuit," "an induction head," "the copy-suppression mechanism," "a deception feature." The philosophy of science lens asks whether that entity refers to a determinate computational concept or to a post-hoc grouping of whatever the discovery procedure returned.

The other four lenses evaluate evidence *about* a claim. This one evaluates *the claim itself*. A circuit with strong causal evidence and a poorly defined construct is a strong measurement attached to a weak theory.

## Key Distinctions

### Observable vs theoretical

Philosophy of science distinguishes observable terms (directly measurable) from theoretical terms (inferred entities postulated to explain observations). "The attention pattern of head 9.9 on IOI prompts" is observable — we can directly compute and display it. "Name-mover head" is theoretical — it is a label we assign based on interpreting the observable pattern. The gap between observable and theoretical is where construct validity lives.

In MI: the move from "head 9.9 has high direct logit attribution for the IO token" (observable) to "head 9.9 is a name-mover" (theoretical) is an interpretive leap. The observable fact is secure; the theoretical label carries assumptions about mechanism, role, and generality that require independent validation. Hempel (1965) argued that theoretical terms must be connected to observables via explicit correspondence rules. "Name-mover" needs a rule: "a head is a name-mover if and only if [specific measurable conditions]." Without this rule, the label is evocative but empirically empty.

### Confirmation vs corroboration

Popper (1959) distinguished confirmation (accumulating supporting evidence) from corroboration (surviving genuine attempts at falsification). A theory that has been tested only in ways it was designed to pass has been confirmed but not corroborated. Corroboration requires risky predictions — tests the theory could fail.

In MI: a circuit discovered by activation patching and evaluated by activation patching has been confirmed (it passes the test it was built to pass). A circuit discovered by activation patching and evaluated by an independent method (weight-space analysis, causal scrubbing, a behavioral prediction on held-out prompts) has been corroborated — it survived a test it was not optimized for. The distinction matters because confirmation is cheap and corroboration is expensive, and the field's evidence base is dominated by confirmation.

### Operationalism vs realism

Bridgman (1927) proposed that a scientific concept is nothing more than the operations used to measure it — "length" is what a ruler measures. This is operationalism. The alternative (scientific realism) holds that concepts refer to real entities in the world, and measurements are fallible attempts to detect them.

In MI: is "the IOI circuit" whatever activation patching returns (operational definition), or does it refer to a real computational structure in the model that we're trying to discover (realist interpretation)? Under operationalism, different methods finding different circuits is not a contradiction — they define different constructs. Under realism, it means at least one method is wrong. The framework adopts a moderate realism: circuits are real computational structures, but our metrics are fallible, and disagreement between methods is informative rather than definitional.

### Underdetermination

Duhem (1906) and Quine (1951) argued that observations underdetermine theory — any finite set of observations is consistent with multiple incompatible theories. You cannot test a hypothesis in isolation; every test relies on auxiliary assumptions (the metric works, the setup is correct, background conditions are stable). When a test fails, it could be the hypothesis that's wrong, or any of the auxiliaries.

In MI: Méloux et al. (2025) showed that multiple circuits are equally faithful for the same task — the behavioral evidence underdetermines which is "the" circuit. More broadly, when an ablation test fails (low effect), it could mean the component isn't part of the circuit (hypothesis wrong), or that the ablation method is inappropriate (auxiliary wrong), or that the prompt distribution doesn't engage the mechanism (background condition wrong). Quine's holism means no single failed test definitively refutes a circuit claim — but it also means no single passed test definitively confirms one.

## Analytical Constructs

### The nomological network

Cronbach and Meehl (1955) proposed that a construct is valid only insofar as it occupies a determinate position in a *nomological network* — a web of lawful relations connecting the construct to other constructs and to observable measurements. A construct that relates to nothing except the specific observations it was designed to explain is not a construct. It is a re-description of the data.

The construct "induction head" ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) is a good example of what a well-connected nomological network looks like. It connects to a specific attention pattern (attending to the token after the previous occurrence of the current token), to a compositional mechanism (a previous-token head in an earlier layer provides the key), to a behavioral prediction (in-context learning on repeated sequences), and to a training dynamics prediction (a phase change in loss curves). Each of these connections is independently testable. The construct earns its validity by surviving those tests, not by being defined once and accepted.

A "circuit" that connects to nothing outside the specific behavior it was discovered on — where the only evidence for the construct is the same data that motivated it — sits at the degenerate edge of this network. It may still be real, but it has not yet earned the status of a validated theoretical entity.

[Campbell and Fiske (1959)](https://doi.org/10.1037/h0046016) made this more precise with the multitrait-multimethod matrix (MTMM). If we measure $k$ traits (circuits) using $m$ methods (metrics), we can arrange the correlations into a $km \times km$ matrix. Convergent validity shows up as high correlations between different methods measuring the same trait. Discriminant validity shows up as low correlations between the same method measuring different traits. For trait $i$ measured by methods $a$ and $b$:

$$r_{ia, ib} > r_{ia, jb} \quad \text{for } i \neq j$$

Two methods should agree more about the same circuit than about different circuits. When EAP and weight classification agree on the IOI circuit more than EAP-on-IOI agrees with EAP-on-Greater-Than, that is genuine convergent validity. When the cross-method agreement is no higher than the cross-trait agreement, that is a measurement artifact.

[Craver (2007)](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) and Woodward (2003) add the interventionist dimension. Craver's constitutive relevance requires that a component be mutually manipulable with the mechanism it belongs to — intervening on the component changes the behavior, and intervening on the behavior (by changing inputs) changes the component's activity. Woodward's invariant difference-making requires that the causal relationship hold under a range of interventions, not just the one that happened to be tested. Together, these require that circuit claims be grounded in manipulation, not association, and that the manipulations be robust across contexts.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Duhem, *The Aim and Structure of Physical Theory*](https://doi.org/10.1515/9780691233857) | 1906 | Philosophy of Science | **Underdetermination** — any observation is consistent with multiple theories; behavioral evidence alone cannot determine "the" circuit because alternative circuits fit the same data equally well |
| [Bridgman, *The Logic of Modern Physics*](https://archive.org/details/logicofmodernphy0000brid) | 1927 | Philosophy of Science | **Operationalism** — a concept is defined by the operations used to measure it; if "the IOI circuit" is just whatever activation patching returns, it is not a theoretical entity but a measurement artifact |
| [Quine, "Two Dogmas of Empiricism"](https://doi.org/10.2307/2181906) | 1951 | Philosophy of Science | **Holism** — no single observation confirms or disconfirms an isolated hypothesis; a failed ablation test indicts the circuit claim, the ablation method, or the prompt distribution jointly |
| [Cronbach & Meehl, "Construct validity in psychological tests"](https://doi.org/10.1037/h0040957) | 1955 | Measurement Theory | **Nomological network** — a construct is valid only insofar as it occupies a determinate position in a web of lawful relations with other constructs and observables |
| [Popper, *The Logic of Scientific Discovery*](https://doi.org/10.4324/9780203994627) | 1959 | Philosophy of Science | **Falsifiability** — a construct has empirical content only if there exist observations that would disconfirm it; a circuit claim consistent with any experimental outcome is vacuous |
| [Campbell & Fiske, "Convergent and discriminant validation by the multitrait-multimethod matrix"](https://doi.org/10.1037/h0046016) | 1959 | Measurement Theory | **Multitrait-multimethod matrix** — convergence across independent metrics; discriminant validity across distinct traits |
| [Hempel, *Aspects of Scientific Explanation*](https://doi.org/10.1029/EO067i020p00253) | 1965 | Philosophy of Science | **Theoretical terms** — labels like "name-mover" or "induction head" must be connected to observables via explicit correspondence rules; without these, the terms are empirically empty |
| [Lakatos, *Falsification and the Methodology of Scientific Research Programmes*](https://doi.org/10.1017/CBO9780511621123.010) | 1970 | Philosophy of Science | **Progressive vs degenerating programmes** — a construct is progressive if it predicts novel facts beyond the data it was discovered on; a circuit found by patching and only ever evaluated by patching is degenerating |
| [Woodward, *Making Things Happen*](https://global.oup.com/academic/product/making-things-happen-9780195189537) | 2003 | Philosophy of Science | **Invariant difference-making** — causes must hold under a range of interventions, not just the one tested |
| [Craver, *Explaining the Brain*](https://global.oup.com/academic/product/explaining-the-brain-9780199568222) | 2007 | Neuroscience / Philosophy | **Constitutive relevance** — mechanistic explanation requires that components make a difference, not merely be present |
| [Méloux et al., "Not all circuits are the same"](https://arxiv.org/abs/2410.10186) | 2025 | Mechanistic Interpretability | **Construct non-uniqueness** — multiple equally faithful circuits exist for the same task; "the circuit" may not refer to a determinate entity |

## Validity type: [Construct validity](/v2/validity-types/construct)

> **Nomological network ([Cronbach & Meehl 1955](https://doi.org/10.1037/h0040957)):** A construct is valid only insofar as it occupies a determinate position in a web of lawful relations with other constructs and observables. A circuit that relates to no other theoretical construct except the specific behavior it was discovered on is not a construct — it is a re-description of the data.

This lens applies to any mechanistic claim, not just circuits. An SAE feature, an MLP neuron, a residual-stream direction, a learned decomposition — anything given a name and a computational role is a construct that can be evaluated.

The difficulty of the construct validity question depends on the [description mode](/v2/description-modes/). A structural claim ("$W_{OV}$ has rank-1 copying structure") is almost self-verifying — the construct is the measurement. An implementational claim ("these heads are name-movers") requires asking whether "name-mover" is a coherent category. An algorithmic claim ("the circuit implements token copying via OV composition") requires asking whether the named algorithm is the real one or just one of many consistent explanations. The higher the mode, the harder construct validity is to establish.

For formal definitions, quantitative thresholds, and calibration data, see [Construct Validity — Formal Specification](/v2/validity-types/construct).

## Criteria

| Code | Criterion | What it asks | Page |
|---|---|---|---|
| C1 | Falsifiability | Was a disconfirming condition stated before collecting evidence? | [C1](/framework/criteria/construct/falsifiability) |
| C2 | Structural plausibility | Do weight-space signatures match the claimed computational role? | [C2](/framework/criteria/construct/structural-plausibility) |
| C3 | Task specificity | Does the circuit score highly only on its discovery task, not unrelated ones? | [C3](/framework/criteria/construct/task-specificity) |
| C4 | Minimality | Is it the smallest set that satisfies sufficiency, with no redundant members? | [C4](/framework/criteria/construct/minimality) |
| C5 | Convergent validity | Do independent metrics identify the same components? | [C5](/framework/criteria/construct/convergent-validity) |

Falsifiability (C1) is a precondition — without it, a claim cannot advance beyond [Proposed](/v2/verdicts/proposed) regardless of other evidence. Convergent validity (C5) is the most powerful and the most frequently absent.

### Falsifiability

A circuit claim is falsifiable when we can state, before collecting evidence, what observation would disconfirm it. Without a pre-registered disconfirming condition, post-hoc rationalization is indistinguishable from genuine confirmation. Every outcome can be explained away: low faithfulness becomes "the circuit is distributed," low specificity becomes "the circuit is general-purpose," inconsistency becomes "the mechanism is context-dependent."

Falsifiability requires three things stated in advance: the metric, the threshold, and the dataset. "If the circuit fails" is not a falsifiability condition. "If interchange intervention accuracy on a held-out paraphrase set falls below $\text{IIA} < 0.10$, measured with the same DAS procedure used during discovery" is one.

In this framework, falsifiability is a precondition rather than a scored criterion. Its absence does not lower a circuit's score — it blocks scoring entirely. A circuit claim with no pre-registered disconfirming condition cannot advance beyond the "Proposed" verdict, regardless of how compelling the post-hoc evidence appears.

**Failure modes.** The most common is implicit falsifiability — a tacit assumption that any sufficiently low number would count as disconfirmation, without naming what "sufficiently low" means. A subtler failure is metric switching: pre-registering a threshold on logit difference but reporting accuracy when the logit difference threshold is not met.

**Minimum reporting.** The disconfirming condition, stated before evidence collection, in the methods section. If retrospective, this must be disclosed.

### Structural plausibility

The components of a proposed circuit should occupy layers and positions consistent with the claimed computational role, and their weight-space signatures should match.

A name-mover head should appear in late layers and have a $W_{OV}$ matrix that approximates a copying operation for name tokens. An S-inhibition head should attend from the final token position to the position of the repeated subject name. A successor head ([Hanna et al. 2023](https://arxiv.org/abs/2305.00586)) should fire on tokens with ordinal or sequential relationships. When a component's structural signature contradicts its claimed role — a "name mover" whose $W_{OV}$ does not copy names, an "inhibition" head that does not attend to the relevant position — the claim has a gap that behavioral evidence alone cannot close.

This is the construct-validity analog of nomological coherence: a circuit component must cohere with what the transformer architecture can mechanistically do at that position. A head in layer 0 cannot read from a head in layer 5. A head with a full-rank $W_{OV}$ is not performing low-rank copying. These are architectural constraints.

**Metrics.** $W_{OV}$ singular value analysis and copying score, attention pattern visualization at task-relevant token positions, direct logit attribution decomposed by component, cross-model structural comparison for the same claimed role.

**Failure modes.** *Role inflation* — the behavioral output matches the claimed role but the weight-space mechanism does not. *Layer mislocation* — a component at the wrong layer for its claimed role produces the right output through compositional coincidence.

**Minimum reporting.** Weight-space signature measurement for every named component role. Any mismatch between role label and signature should be flagged explicitly.

<details class="worked-example">
<summary>Worked example: structural plausibility of Greater-Than successor heads</summary>

Hanna et al. (2023) identify attention heads in GPT-2 Small that implement the Greater-Than task — given "The war lasted from 1732 to 17," the model must predict a year greater than 32. The claimed mechanism involves heads that have learned ordinal relationships between year tokens.

Structural plausibility here means checking whether the $W_{OV}$ matrices actually encode ordinal structure. For a proposed successor head $h$, we compute:

$$\text{effect}(y_1, y_2) = e_{y_2}^\top \, W_U \, W_{OV}^{(h)} \, W_E \, e_{y_1}$$

where $e_y$ is a one-hot vector for year token $y$, $W_E$ is the embedding matrix, and $W_U$ is the unembedding matrix. Structural plausibility requires that $\text{effect}(y_1, y_2) > 0$ when $y_2 > y_1$ and $\text{effect}(y_1, y_2) < 0$ when $y_2 < y_1$, at least on average across the relevant year-token pairs. Hanna et al. confirm this pattern: the $W_{OV}$ matrices of their proposed heads encode a monotonic ordering over two-digit year suffixes. A head labeled "successor" whose $W_{OV}$ showed no such ordering would fail structural plausibility regardless of its behavioral effect.
</details>

### Task specificity

The proposed circuit should not score highly on unrelated tasks under the same evaluation procedure. This is the construct-validity analog of discriminant validity (Campbell and Fiske 1959): measures of distinct traits should not correlate highly.

If a circuit discovered for IOI also ranks at the top of Greater-Than, subject-verb agreement, and gendered-pronoun resolution under the same faithfulness metric, one of two things is happening. Either the circuit is genuinely general-purpose — in which case it should be reported as such, not as a task-specific mechanism — or the evaluation procedure is picking up on a confound (such as a bottleneck component that all tasks route through).

The bar for informative cross-task evaluation is discriminability between related tasks, not trivially different ones. Testing the IOI circuit against arithmetic is a low bar; testing it against subject-verb agreement is informative, because both require tracking syntactic roles across positions.

We can quantify this with a selectivity ratio. For a circuit $C$ discovered on task $T_{\text{disc}}$ and evaluated on a related task $T_{\text{off}}$:

$$S(C) = \frac{F(C, T_{\text{disc}}) - F(C, T_{\text{off}})}{F(C, T_{\text{disc}})}$$

where $F(C, T)$ is the faithfulness score. $S = 1$ means zero off-task faithfulness; $S = 0$ means equal faithfulness on both; $S < 0$ means the circuit is *more* faithful on the off-task — a red flag.

**Failure modes.** *No cross-task evaluation performed* is the most common. *Trivially distinct off-tasks* inflate apparent specificity. *Bottleneck capture* — the circuit includes a general-purpose component (an early-layer induction head, say) that all tasks depend on, so the circuit "passes" every task because it contains a shared prerequisite.

**Minimum reporting.** At least one evaluation on a related task not used during discovery. Discriminant faithfulness reported alongside discovery-task faithfulness, with the selectivity ratio $S$.

### Minimality

The circuit should be the smallest set of components that satisfies sufficiency. No member should be redundant.

Craver (2007) defines the components of a mechanism as those whose presence makes a *difference*, not those that are merely present during operation. Adding components to a circuit can only increase apparent sufficiency — an over-inclusive circuit is therefore unfalsifiable by any sufficiency test. If we include every head whose removal causes a nonzero decrease in performance, we will include heads that are incidental rather than constitutive, and the resulting "circuit" will describe the model's general-purpose infrastructure rather than the task-specific mechanism.

The practical test is leave-one-out ablation within the proposed circuit. For each component $c_i$ in circuit $C = \{c_1, \ldots, c_n\}$, we measure the faithfulness of $C \setminus \{c_i\}$. A component is individually necessary if removing it causes a meaningful drop; it is individually redundant if the circuit performs equally well without it.

But individual necessity is not the whole story. Two components can be individually necessary yet jointly redundant — each compensates for the other's absence when tested alone, but removing both reveals that only one is needed. [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) found exactly this with the backup name-mover heads in the IOI circuit: each backup head is individually unnecessary (the primary name movers suffice), but the backups activate when the primaries are ablated. Whether the backups are "part of the circuit" depends on whether we define the circuit as the minimal sufficient set under normal operation or under ablation.

**Failure modes.** *Over-inclusion* — every component above a contribution threshold is included rather than the minimal sufficient set. *Redundancy masking* — compensatory mechanisms hide the fact that a smaller circuit would suffice.

**Minimum reporting.** Per-component leave-one-out results. Explicit distinction between individually necessary and jointly necessary components. If backup or compensatory mechanisms are observed, these should be reported as such.

### Convergent validity

Multiple independent metrics should identify the same components.

When two methods that share no major assumptions identify the same circuit, their agreement is informative — it is unlikely to be an artifact of either method's biases. When one method's output is the other's input (using activation patching to discover a circuit and then path patching to "confirm" it), agreement is mechanical. The shared assumption must be named.

We quantify convergent validity using Jaccard similarity at the component level. For circuits $C_A$ and $C_B$ identified by methods $A$ and $B$:

$$J(C_A, C_B) = \frac{|C_A \cap C_B|}{|C_A \cup C_B|}$$

$J = 1.0$ means perfect agreement; $J = 0$ means no overlap. In practice, $J > 0.6$ between genuinely independent methods (weight-space classification and activation-based attribution patching, say) is strong convergent validity. $J < 0.3$ between independent methods is a warning that the circuit is method-dependent.

The independence requirement is strict. Two gradient-based attribution methods share the linearity assumption. Two patching methods share the interventionist assumption. Two methods that both threshold at a percentile share the assumption that the relevant components are in the tail of some distribution. For convergent validity to hold, the methods must differ in their *major* assumptions — the ones that determine which components are selected.

<details class="worked-example">
<summary>Worked example: convergent validity for the IOI circuit</summary>

Wang et al. (2022) identified the IOI circuit primarily through activation patching and direct logit attribution — both grounded in the interventionist framework. Suppose we independently apply a weight-space classifier (which examines $W_{OV}$ and $W_{QK}$ matrices without running any forward passes) and it identifies 20 of the 26 heads, plus 4 heads not in the original circuit.

The two circuits are:

- $C_{\text{AP}}$ = the 26-head activation patching circuit
- $C_{\text{WC}}$ = the 24-head weight classifier circuit

Their overlap is $|C_{\text{AP}} \cap C_{\text{WC}}| = 20$ heads. Their union is $|C_{\text{AP}} \cup C_{\text{WC}}| = 30$ heads. The Jaccard similarity is:

$$J(C_{\text{AP}}, C_{\text{WC}}) = \frac{20}{30} = 0.67$$

This is informative because the methods share almost no assumptions — one intervenes on activations during forward passes, the other examines static weights. The 6 heads found only by activation patching may implement their role through a dynamic mechanism invisible to weight inspection. The 4 heads found only by weight classification may have the structural signature but not the activation profile on the tested prompts. Both discrepancies are scientifically interesting and should be reported rather than resolved by picking one method's output.
</details>

**Failure modes.** *Shared-bias convergence* — two metrics converge because they share an assumption (linearity, gradient-based attribution), not because the claim is true. *Pipeline convergence* — one metric's output feeds the other, making agreement circular. *Threshold-dependent agreement* — two methods agree at one threshold but diverge at another; Jaccard should be reported across a range of thresholds.

**Minimum reporting.** At least two metrics with non-overlapping major assumptions. Agreement reported as Jaccard similarity at the component level, ideally across a threshold sweep.

## Underdetermination

When evidence is consistent with multiple incompatible explanations, the correct verdict is *underdetermined*, not *solved*. Underdetermination is not a failure — it is a state of evidence that construct validity is equipped to name.

[Méloux et al. (ICLR 2025)](https://arxiv.org/abs/2410.10186) show empirically that underdetermination is the norm in circuit discovery. Using multiple discovery algorithms on the same tasks and models, they find that the circuit you get depends on the search heuristic you use. Different algorithms return circuits with substantially different membership — sometimes with Jaccard similarities as low as $J = 0.2$ — yet each circuit individually passes faithfulness and necessity tests. The circuits are not wrong. They are underdetermined.

This is consistent with a basic observation from philosophy of science (Duhem 1906, [Quine 1951](https://doi.org/10.2307/2181906)): any finite body of evidence is consistent with multiple theories. In the circuit setting, a single behavioral output (logit difference on IOI prompts) constrains the circuit only to the set of component subsets that can produce that output. There are, in general, many such subsets — especially when the model contains redundant or partially overlapping mechanisms. Faithfulness tests reduce this set but do not reduce it to one.

Reporting underdetermination explicitly is a stronger finding than suppressing it. "We found circuit $X$; EAP finds a distinct circuit $Y$; both are faithful; they agree on $n$ components ($J = 0.4$)" tells the reader more than "we found circuit $X$ and it is faithful." The former locates the claim in the space of possible circuits and identifies which components are robust to method variation. The latter presents a method-dependent result as a method-independent one.

## Verdicts

Construct validity gates advancement through the [verdict tiers](/v2/verdicts/):

- **Proposed → Causally suggestive:** Requires C1 (falsifiability). Without a pre-registered disconfirming condition, no amount of ablation evidence upgrades the verdict.
- **Mechanistically supported → Triangulated:** Requires at least one construct criterion beyond C1. Typically C2 (structural plausibility) or C5 (convergent validity).
- **Triangulated → Validated:** Requires substantial construct validity coverage — C1 through C5.

A claim can have perfect [internal validity](/v2/validity-types/internal) (all ablations, all patching, full consistency) and still stall at Mechanistically supported because the construct itself is poorly defined.

## Protocol

For a proposed circuit $C$ and behavior $B$, the following protocol operationalizes construct validity. A skipped step must be named in the verdict rather than silently omitted.

1. **Falsifiability.** State the disconfirming condition — metric, threshold, and dataset — before collecting evidence. If the condition is specified retrospectively, disclose this.

2. **Structural plausibility.** For every named component role, confirm that the weight-space signature matches the claimed mechanism. Report the specific measurements ($W_{OV}$ copying score, attention pattern at relevant positions, direct logit attribution sign) and flag any mismatches.

3. **Task specificity.** Evaluate $C$ on at least one related task not used during discovery. Report faithfulness on the off-task alongside the discovery-task faithfulness, and compute the selectivity ratio $S$.

4. **Minimality.** Per-component leave-one-out ablation within the circuit. Report which components are individually necessary versus jointly necessary. If compensatory or backup mechanisms are observed, report them as such.

5. **Convergent validity.** Apply at least two metrics with non-overlapping major assumptions. Report Jaccard similarity at the component level. If the methods disagree, characterize the disagreement.

6. **Underdetermination.** If an alternative circuit $C'$ with comparable faithfulness is known, report it. State the Jaccard overlap and identify the robust core — the components present in all known faithful circuits.

## Case studies

For full worked examples applying all five lenses (including construct validity) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) — the most thoroughly analyzed circuit; strong C2, weak C3/C5
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) — the strongest mechanistic claim; passes C1–C5
- [SAE Features](/framework/lenses_v6/examples/examples-sae-features) — weakest construct validity; thin nomological network
- [Greater-Than](/framework/lenses_v6/examples/examples-greater-than) — best structural plausibility (C2)
- [Grokking](/framework/lenses_v6/examples/examples-grokking) — the ceiling: Validated within toy scope
- [Knowledge Neurons](/framework/lenses_v6/examples/examples-knowledge-neurons) — tool works, but construct may be wrong
- [Gender Bias](/framework/lenses_v6/examples/examples-gender-bias) — construct incoherence (C3 fails fundamentally)
