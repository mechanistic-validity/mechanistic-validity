---
title: "Causal Inference"
description: "The formal language lens: do-calculus as the language of ablation claims, potential outcomes as the framework for heterogeneous effects, and transportability as the condition for cross-model transfer."
---

# The Causal Inference Lens

This lens asks one question: **does the ablation claim have a well-defined causal semantics, and do the formal conditions for that semantics hold?**

Every ablation study in mechanistic interpretability makes a causal claim. Zeroing out a head and observing a behavioral change is an intervention, and the conclusion "this head implements name-moving" is a causal inference from that intervention. The causal inference literature — Pearl's structural causal models, Rubin's potential outcomes, Woodward's interventionism, Spirtes' causal discovery algorithms — provides the formal language for stating these claims precisely, the conditions under which they are valid, and the failure modes when those conditions are violated.

The contribution of this lens is not a new experimental technique. It is the formal language that makes the existing techniques' assumptions visible. An ablation study already performs a do-operation. Stating it as one makes the conditioning set, the exclusion restriction, and the transportability conditions explicit — and makes their violations checkable.

## Key distinctions

### do-calculus as the language of ablation

[Pearl (2009)](https://doi.org/10.1017/CBO9780511803161) formalized the distinction between observation and intervention. Observing that a component is active during a task establishes $P(Y \mid X)$ — the conditional probability of the output given the component's state. Ablating the component and measuring the output establishes $P(Y \mid \text{do}(X := 0))$ — the probability under intervention. These are not the same quantity, and the difference is the confounding bias.

In MI, the distinction separates two classes of evidence:

- **Observational:** probing, attention pattern analysis, correlation between component activations and behavior. These establish association, not causation.
- **Interventional:** [ablation](/mechanistic-validity/glossary/#ablation), [activation patching](/mechanistic-validity/glossary/#activation-patching), [causal scrubbing](/mechanistic-validity/glossary/#causal-scrubbing), [DAS-IIA](/mechanistic-validity/glossary/#das). These establish (conditional) causation, subject to the assumptions of the specific do-operation.

The structural causal model makes the causal structure explicit. For IOI, the claimed mechanism is:

$$S \to H_{\text{SI}} \to H_{\text{NM}} \to Y \quad \text{and} \quad IO \to H_{\text{NM}} \to Y$$

where $S$ = subject name, $IO$ = indirect object, $H_{\text{SI}}$ = S-inhibition heads, $H_{\text{NM}}$ = name-mover heads, and $Y$ = output logit. The necessity claim (I1) is $P(Y \mid \text{do}(H_{\text{NM}} := 0)) \neq P(Y)$. The sufficiency claim (I2) is $P(Y \mid \text{do}(\text{all except circuit} := 0)) \approx P(Y)$. The specificity claim (I4) — currently untested — is $P(Y_{\text{SVA}} \mid \text{do}(H_{\text{NM}} := 0)) \approx P(Y_{\text{SVA}})$ for an unrelated task like subject-verb agreement. The SCM makes the untested link visible: I4 is a conditional independence claim that has never been checked.

### Interventionism and counterfactuals

[Woodward (2003)](https://doi.org/10.1093/0195155270.001.0001) defined interventionist causation: $X$ causes $Y$ if and only if there exists an intervention on $X$ that changes $Y$, holding all other variables fixed. This maps directly to MI's necessity and sufficiency criteria:

- **Necessity (I1):** "Would the behavior persist if the component were absent?" — an interventionist counterfactual.
- **Sufficiency (I2):** "Would the component alone produce the behavior?" — a stronger interventionist claim.

The critical word is "holding all other variables fixed." In MI, ablating a head does not hold other variables fixed — downstream components receive different inputs and may compensate or malfunction. This is the cascading-disruption confound, and it is a formal violation of the interventionist condition. Resample ablation partially addresses it by keeping activations on-manifold; [path patching](/mechanistic-validity/glossary/#path-patching) addresses it more directly by intervening on a specific path rather than a component's total output.

### Potential outcomes and heterogeneous effects

[Rubin (1974)](https://doi.org/10.1037/h0037350) formalized treatment effects through potential outcomes: for each unit (here, each prompt), there is a potential outcome under treatment $Y_i(1)$ (ablated) and under control $Y_i(0)$ (intact). The average treatment effect is $\text{ATE} = \mathbb{E}[Y_i(1) - Y_i(0)]$.

In MI, the "units" are prompts and the "treatment" is ablation. The ATE framework makes a crucial fact visible: **the average ablation effect can be large even if many individual prompts show zero effect, or small even if a few prompts show catastrophic effects.** Reporting a mean logit difference change across 100 prompts hides this heterogeneity. The potential outcomes framework asks: what is the distribution of individual treatment effects, and are there identifiable subpopulations for which the effect differs?

This grounds E5 (graded response): not all prompts may respond to ablation equally. A circuit that is necessary for IOI on "When Mary and John went to the store, John gave a drink to..." may be unnecessary on a paraphrased version. The heterogeneous treatment effect is information about the circuit's scope.

### Transportability

[Pearl & Bareinboim (2011)](https://doi.org/10.1609/aaai.v25i1.8018) formalized the conditions under which causal effects estimated in one population transfer to another. Transportability theory asks: given a causal graph and known differences between populations, can the causal effect in the target population be identified from experiments in the source population?

In MI, "populations" are models. A circuit discovered in GPT-2 Small is a finding about GPT-2 Small. Whether it transfers to GPT-2 Medium, Pythia, or Llama depends on which variables in the causal graph differ across models (architecture, training data, tokenizer, scale). Transportability theory provides the formal conditions: the effect transfers if and only if the differences between models do not d-separate the intervention from the outcome through a path that is not blocked by the circuit. This grounds E4 (cross-model generalization).

### Causal discovery

[Spirtes, Glymour & Scheines (2000)](https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/) developed algorithms for learning causal structure from data — constraint-based (PC, FCI) and score-based (GES) methods that recover the causal graph from observational data under assumptions (faithfulness, causal sufficiency, acyclicity).

In MI, [attribution patching](/mechanistic-validity/glossary/#attribution-patching) (EAP) and ACDC are causal discovery algorithms: they learn which edges in the computational graph carry task-relevant information. The assumptions of causal discovery apply directly — faithfulness (every statistical dependence reflects a causal connection) and causal sufficiency (no unmeasured common causes) — and their violations are the failure modes. ACDC's greedy edge-pruning assumes causal sufficiency; if two heads share an unmeasured common input (e.g., a residual stream direction not in the circuit's scope), the algorithm may attribute one's effect to the other.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Pearl, *Causality: Models, Reasoning, and Inference*](https://doi.org/10.1017/CBO9780511803161) | 2009 | Causal inference | **do-calculus and SCMs** — the formal language for distinguishing observation from intervention; the rules under which interventional distributions can be identified from observational data |
| [Woodward, *Making Things Happen: A Theory of Causal Explanation*](https://doi.org/10.1093/0195155270.001.0001) | 2003 | Philosophy of science | **Interventionist causation** — $X$ causes $Y$ iff there exists an intervention on $X$ that changes $Y$; the framework for necessity and sufficiency claims |
| [Rubin, "Estimating causal effects of treatments in randomized and nonrandomized studies"](https://doi.org/10.1037/h0037350) | 1974 | Statistics | **Potential outcomes** — each unit has outcomes under treatment and control; the average treatment effect is defined as their expected difference; heterogeneity across units is information |
| [Pearl & Bareinboim, "Transportability of causal and statistical relations: a formal approach"](https://doi.org/10.1609/aaai.v25i1.8018) | 2011 | Causal inference | **Transportability** — formal conditions under which causal effects transfer across populations; grounds cross-model generalization claims |
| [Spirtes, Glymour & Scheines, *Causation, Prediction, and Search*](https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/) | 2000 | Causal inference | **Causal discovery algorithms** — learning causal structure from data under faithfulness and sufficiency; the formal framework for circuit-discovery methods |
| [Shadish, Cook & Campbell, *Experimental and Quasi-Experimental Designs for Generalized Causal Inference*](https://psycnet.apa.org/record/2002-17373-000) | 2002 | Methodology | **Validity types taxonomy** — construct, internal, external, and statistical conclusion validity; the framework this paper adapts for mechanistic claims |

## Validity type: [Internal validity](/mechanistic-validity/framework/validity-types/internal/)

> **An ablation is a do-operation.** Every time an MI paper ablates a component and draws a conclusion, it is performing an intervention in the formal sense. Stating the intervention as $P(Y \mid \text{do}(X := x'))$ rather than prose makes the conditioning set visible and the exclusion restriction checkable.

The causal inference lens contributes to the framework through formal grounding rather than additional criteria. It provides the language in which the criteria from other lenses — necessity, sufficiency, specificity, generalization — are stated precisely:

| Concept | Source | Criteria it grounds | MI analog |
|---|---|---|---|
| do-calculus / SCM | Pearl (2009) | I1, I2 | Formal language for ablation claims |
| Interventionism | Woodward (2003) | I1 | "Would behavior persist if component absent?" |
| Transportability | Pearl & Bareinboim (2011) | E4 | Conditions for circuit generalization across models |
| Causal discovery | Spirtes, Glymour & Scheines (2000) | — | Learning circuit structure from data |
| Potential outcomes | Rubin (1974) | E5 | Heterogeneous effects across input subpopulations |

## Evidence patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Single ablation, no SCM stated | Informal necessity claim | "Ablation degrades behavior; causal model not specified" |
| SCM with do-calculus derivation | Formal necessity/sufficiency | "$P(Y \mid \text{do}(X := 0)) \neq P(Y)$; necessity holds under the stated graph" |
| Path-specific effects computed | Mediation identified | "Effect mediated through [path]; direct effect [magnitude]" |
| Heterogeneous effects across prompts | Treatment effect varies by subpopulation | "ATE = [X]; prompt-level effects range from [min] to [max]" |
| Cross-model transfer with transportability check | Formal generalization | "Transportability conditions satisfied for [target model]; effect transfers" |

## Verdicts

The causal inference lens does not gate specific verdict transitions — unlike pharmacology or genetics, it does not add numbered criteria that must be passed. Its contribution is formal: it provides the language in which the criteria from other lenses are stated, and violations of the formal conditions (cascading disruption, hidden confounding, non-transportability) are the failure modes those criteria detect.

- **Proposed → Causally suggestive:** I1 (necessity) is the gate. The causal inference lens formalizes I1 as a do-operation: $P(Y \mid \text{do}(X := 0)) \neq P(Y)$.
- **Causally suggestive → Mechanistically supported:** I2 (sufficiency) is stated as a stronger do-operation. The cascading-disruption confound is a formal violation of the interventionist condition.
- **Mechanistically supported → Triangulated:** Multiple causal identification strategies (ablation + path patching + causal scrubbing) provide identification under different assumptions. Convergence across identification strategies is stronger than convergence across variations of one strategy.
- **Triangulated → Validated:** E4 (cross-model generalization) is formally a transportability claim. The causal inference lens provides the conditions under which it is valid.
