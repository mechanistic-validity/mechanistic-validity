---
title: "Internal Validity"
---

# Internal Validity

Does the evidence support the causal claim? Given a well-defined construct and trustworthy instruments, the evidence must establish that the identified component is causally involved in the behavior — not merely correlated with it. Ablation shows that removing a head degrades performance; patching shows that restoring a head's activation from a clean run recovers performance. These are formally distinct interventions, and the distinction matters: a component can be necessary without being sufficient, sufficient without being specific, and specific without being the only mechanism that fits the data.

## Position in the Dependency Chain

```
Construct → Measurement → Internal → External → Interpretive
                            ▲
                          you are here
```

Internal validity occupies the third position in the chain. It depends on both construct validity and measurement validity. A causal claim about a component that corresponds to no coherent construct (construct failure) cannot be evaluated — there is nothing for the component to be causally involved *in*. A causal claim supported by an unreliable metric (measurement failure) cannot be trusted — Spearman's attenuation formula bounds the correlation between an unreliable measure and any true effect at the geometric mean of their reliabilities, so noise in the instrument caps the strength of the causal inference.

Internal validity is what most MI evidence addresses. Ablation, activation patching, path patching, and causal scrubbing are all internal-validity methods. They ask whether the identified components are causally involved in the behavior within the experimental setup. What they do not ask — and what depends on internal validity being established first — is whether the result generalizes (external validity) or whether the narrative about the mechanism is correct (interpretive validity).

## Theoretical Lineage

The concept of internal validity originates with Campbell and Stanley (1963), who developed a taxonomy of threats to causal inference in experimental design. Their object of analysis is a study; their question is whether the study's design licenses the conclusion drawn. Shadish, Cook, and Campbell (2002) extended this framework with a fuller catalogue of threats and a distinction between statistical conclusion validity and internal validity proper. We adopt their vocabulary throughout.

Pearl (2009) provides the formal language in which ablation claims can be stated. The *do*-calculus distinguishes observational association from interventional effect: $P(Y \mid X)$ is not $P(Y \mid \text{do}(X))$. In MI terms, observing that a head's activation correlates with task performance is not the same as showing that setting that activation to a counterfactual value changes task performance. Ablation and activation patching are formally distinct interventions — ablation sets a value to a default (zero, mean, resample), while patching transplants a value from a different input. Woodward (2003) complements Pearl with an interventionist account of causation that makes the connection between counterfactual dependence and causal claims explicit: a variable $X$ is a cause of $Y$ if and only if there exists an intervention on $X$ that changes $Y$, holding fixed all other variables on paths that do not go through $X$.

Craver (2007) contributes the framework of mutual manipulability from neuroscience. A component is part of a mechanism if intervening on the component changes the system's behavior (bottom-up, corresponding to I1 necessity) *and* intervening on the system's input-output relation changes the component's activity (top-down, corresponding to I12 offset coupling). Shallice (1988) contributes the double dissociation from neuropsychology: two interventions cross, each impairing the function the other spares. This is the strongest form of specificity evidence and maps directly to I6.

Three of Hill's (1965) nine viewpoints for causal inference in epidemiology reappear as criteria: specificity (I4), convergence (addressed at construct level as C3), and graded response (addressed at external level as E5). Mayo (1996, 2018) contributes the error-statistical tradition and the concept of severe testing — a test that a hypothesis was not built to pass. From genetics, we take epistasis for non-additive component interactions (I9), rescue experiments for reversibility (I10), and sensitivity analysis for unmeasured confounders (I8). From medical microbiology, Koch's postulates provide the template for graded causal evidence: a pathogen must be found in all cases of the disease, isolated and grown in pure culture, and reproduce the disease when introduced into a healthy host. The parallel to circuit claims is direct — necessity, isolation sufficiency, and rescue — though the analogy is partial because circuits, unlike pathogens, are not discrete entities with sharp boundaries.

## Criteria

| ID | Name | Question |
|---|---|---|
| I1 | Necessity | Is the circuit required for the behavior? |
| I2 | Sufficiency | Is the circuit enough to produce the behavior? |
| I3 | Minimality | Does every component earn its place? |
| I4 | Specificity | Does intervening on the circuit affect this task more than matched control tasks? |
| I5 | Rival mechanism exclusion | Is this THE mechanism, or A mechanism? |
| I6 | Double dissociation | Do two interventions cross, each breaking what the other spares? |
| I7 | Confound control | Are alternative explanations ruled out? |
| I8 | Confounding sensitivity | How strong must an unmeasured confounder be to explain the result? |
| I9 | Epistatic interaction | Do circuit components interact non-additively? |
| I10 | Rescue reversibility | Does restoring a corrupted component recover behavior? |
| I11 | Onset coupling | Does the mechanism appear when the capability appears? |
| I12 | Offset coupling | Does the mechanism go when the capability is removed? |

The twelve criteria fall into five blocks:

- **I1–I3 (set-level properties):** Necessity, sufficiency, and minimality characterize the circuit as a whole. A circuit that is necessary but not sufficient is incomplete. A circuit that is sufficient but not minimal contains passengers.
- **I4–I6 (discrimination):** Specificity discriminates across tasks, rival mechanism exclusion discriminates across alternative circuits, and double dissociation discriminates across both simultaneously. I6 is the one criterion that requires both arms to be run: a single dissociation, however clean, does not establish it. Across sixteen audited claims, I6 was met once — by Feucht et al. (2025), three years after the origin paper.
- **I7–I8 (confounders):** I7 addresses measured confounds (off-manifold ablation, backup suppression, layer-norm redistribution). I8 asks how strong an unmeasured confounder would need to be to explain the result — the analog of Rosenbaum's sensitivity analysis in observational studies.
- **I9–I10 (internal structure):** Epistatic interaction (I9) asks whether circuit components interact non-additively — whether the joint effect of ablating two components differs from the sum of their individual effects. Rescue reversibility (I10) asks whether restoring a corrupted component recovers the behavior, the circuit analog of a genetic rescue experiment.
- **I11–I12 (developmental coupling):** Onset coupling (I11) asks whether the mechanism appears during training when the capability appears. Offset coupling (I12) asks whether the mechanism disappears when the capability is removed — Craver's top-down leg of mutual manipulability.

## Failure Examples

**Cardiac stents and specificity (I4).** The ORBITA trial randomized patients with stable angina to percutaneous coronary intervention or a sham procedure. The sham group showed zero benefit difference. The stent was necessary for opening the artery (the intervention reached its target) but not specific to the symptom — the symptom improvement was a placebo effect. The MI parallel: a component can be necessary for a behavior (ablating it degrades performance) without being specific to the behavior (ablating it degrades many behaviors equally, because it is a bottleneck).

**Candidate gene psychiatry and confound control (I7).** Border et al. (2019) tested 18 candidate genes for depression in a sample of 620,000 individuals. None was associated with depression more than a randomly selected gene. Two decades of candidate gene studies had reported positive results from samples of hundreds to low thousands, where confounding by population stratification, publication bias, and flexible analysis produced consistent false positives. The MI parallel: a circuit component reported as causally involved in a behavior on a small prompt set, without controlling for the confound that ablating *any* component of similar size degrades performance by a similar amount.

**IOI circuit specificity (I4).** Wang et al. (2022) discovered the IOI circuit on a specific set of prompts. When we examine circuits discovered on different prompt samples for the same task, the Jaccard similarity between sample-specific circuits is 0.126. Transferring one sample's circuit to another sample drives the behavioral metric in the opposite direction. The circuit is specific to the prompt sample, not to the task.

**Circuit non-uniqueness (I5).** Chen et al. (2024) found two circuits for IOI in GPT-2 Small, both achieving 100% faithfulness, sharing 4.1% of edges. The existence of multiple high-faithfulness circuits for the same behavior means that presenting one circuit as *the* mechanism for the behavior fails rival mechanism exclusion. The appropriate claim is "a sufficient mechanism," not "the mechanism."

## Cross-Disciplinary Foundations

Internal validity draws on more source disciplines than any other validity type — four of the eight theoretical foundations contribute directly.

| Discipline | Transfer to internal validity |
|---|---|
| Causal inference | *do*-calculus provides the formal language for ablation and patching claims; distinguishes observational correlation from interventional effect |
| Neuroscience | Lesion and stimulation experiments map to necessity and sufficiency; double dissociation (Shallice 1988) maps to I6; mutual manipulability (Craver 2007) maps to I1 + I12 |
| Genetics | Epistasis for non-additive interactions (I9); rescue experiments for reversibility (I10); sensitivity analysis for unmeasured confounders (I8); complementation test (Benzer 1955) for C6 |
| Medical microbiology | Koch's postulates provide the template for graded causal evidence — necessity, isolation, and rescue — with partial satisfaction as the norm rather than the exception |
