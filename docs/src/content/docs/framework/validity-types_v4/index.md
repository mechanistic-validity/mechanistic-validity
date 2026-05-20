---
title: "The Five Validity Types"
description: "Five independent, named failure modes for circuit claims — from construct coherence to interpretive warrant."
---

# The Five Validity Types

Validity is the organizing framework for what a circuit claim must satisfy to count as evidence rather than coincidence. The four-validity taxonomy originates in experimental design ([Cook & Campbell 1979](https://doi.org/10.2307/2529755); [Shadish, Cook & Campbell 2002](https://psycnet.apa.org/record/2002-17109-000)), where it was developed to evaluate causal claims in social science and biomedical research. The fifth type — interpretive validity — is added to address the gap between validated mechanisms and defensible explanations that the MI-specific literature has surfaced ([Geiger et al. 2024](https://arxiv.org/abs/2301.04709); [Méloux et al. 2026](https://openreview.net/forum?id=vERnMGBqxJ)).

The intellectual lineage of the five-type framework:

| Validity type | Origin | Foundational references |
|---|---|---|
| Construct | Philosophy of science / measurement theory | [Cronbach & Meehl (1955)](https://doi.org/10.1037/h0040957); [Craver (2007)](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) |
| Internal | Experimental methodology / systems neuroscience | [Campbell (1957)](https://doi.org/10.1037/h0040945); [Woodward (2003)](https://doi.org/10.1093/0195155270.001.0001) |
| External | Experimental methodology / pharmacology | [Cook & Campbell (1979)](https://doi.org/10.2307/2529755); [Shadish et al. (2002)](https://psycnet.apa.org/record/2002-17109-000) |
| Measurement | Classical test theory / measurement theory | [Lord & Novick (1968)](https://psycnet.apa.org/record/1969-02031-000); [Campbell & Fiske (1959)](https://doi.org/10.1037/h0046016) |
| Interpretive | Mechanistic interpretability methodology | [Marr (1982)](https://doi.org/10.7551/mitpress/9780262514620.001.0001); [Geiger et al. (2021)](https://arxiv.org/abs/2106.02997) |

## Why five types rather than one global score

A single score summarizing "how good a circuit is" obscures the fact that circuit claims can fail in qualitatively different ways. A circuit can be measured by a reliable metric and still correspond to no coherent computational concept. A circuit can correspond to a coherent concept and rest on purely correlational evidence. A circuit can survive rigorous causal testing at one intervention strength on one prompt distribution and collapse under any other. A circuit can pass all of those tests and still be described at the wrong level of abstraction. These are not points on a continuum; they are independent failures that demand independent remedies.

The five-type taxonomy makes those failures named and reportable. A verdict that satisfies internal validity but not construct validity is honestly described as *causally implicated but theoretically underspecified*, and the remedy is a clearer construct rather than more interventions. A verdict that satisfies construct validity but not external validity is *coherent but local*, and the remedy is replication rather than redefinition. A verdict that passes all four traditional types but fails interpretive validity is *validated but overclaimed*, and the remedy is scoping the narrative to match the evidence level. These distinctions are routinely collapsed in MI write-ups, with the result that claims of different types are presented as if they were equivalent.

## The five types

| Type | Question | Parent discipline | Key criteria |
|---|---|---|
| **Construct** | Is the claimed entity a coherent theoretical concept? | Philosophy of science (Cronbach & Meehl 1955; Craver 2007) | C1 Falsifiability, C2 Structural plausibility, C3 Task specificity, C4 Minimality, C5 Convergent validity |
| **Measurement** | Is the metric that produced the evidence trustworthy? | Measurement theory (Campbell & Fiske 1959; Lord & Novick 1968) | M1 Reliability, M2 Invariance, M3 Baseline separation, M4 Sensitivity, M5 Calibration, M6 Construct coverage |
| **Internal** | Does the evidence establish that the component implements the computation? | Systems neuroscience (Woodward 2003; Craver 2007) | I1 Necessity, I2 Sufficiency, I3 Specificity, I4 Consistency, I5 Confound control |
| **External** | Does the claim generalize beyond the tested conditions? | Pharmacology (Clark 1926; Hill 1910; Gaddum 1937) | E1 Intervention reach, E2 Graded response, E3 Selectivity, E4 Effect magnitude, E5 Robustness, E6 Cross-architecture |
| **Interpretive** | Is the narrative about the mechanism licensed by the evidence? | Mechanistic interpretability methodology (Marr 1982; Geiger et al. 2024) | V1 Level declaration, V2 Level-evidence match, V3 Narrative coherence, V4 Alternative exclusion, V5 Scope honesty |

## How the five types interact

The types are not independent in the sense that they can be evaluated in any order; they have an implicit dependency structure:

1. **Construct validity comes first.** A construct that is not clearly defined cannot be measured, and an ambiguous construct cannot have its causal role meaningfully tested.
2. **Measurement validity gates internal validity.** A metric that is unreliable cannot support a causal inference. A high IIA score computed without a random-vector baseline does not license a representational claim, regardless of how well the internal interventions performed.
3. **Internal validity gates external validity.** A finding that has not been established causally within the discovery conditions cannot be said to generalize. External validity asks about the *reach* of an established result, not the credibility of an unestablished one.
4. **External validity gates upgrade from result to property.** An internally valid claim that does not generalize is a local result rather than a finding. The upgrade from *result on a benchmark* to *property of the model* requires external validity evidence.
5. **Interpretive validity is downstream of all four.** A narrative about a mechanism cannot be evaluated until the mechanism itself has been established as real, generalizable, coherent, and well-measured.

The dependency order does not mean that work on any single type must wait for the previous one to be finished. It means that a verdict at any level should name the types at which evidence is missing, rather than upgrading the verdict on the strength of evidence from a different type.

## How each type connects to its casebook

Each validity type has a dedicated casebook that translates the type's abstract requirements into operational criteria, metrics, and reporting rules.

- *Construct validity* is operationalized by the [Philosophy of Science Casebook](../lenses_v6/philosophy_of_science), which provides falsifiability, structural plausibility, task specificity, minimality, and convergent validity criteria.
- *Measurement validity* is operationalized by the [Measurement Theory Casebook](../lenses_v6/measurement-theory), which provides reliability, invariance, baseline separation, sensitivity, calibration, and construct coverage criteria.
- *Internal validity* is operationalized by the [Neuroscience Casebook](../lenses_v6/neuroscience), which provides necessity, sufficiency, specificity, consistency, and confound-control criteria.
- *External validity* is operationalized by the [Pharmacology Casebook](../lenses_v6/pharmacology), which provides intervention reach, graded response, selectivity, effect magnitude, robustness, and cross-architecture generalization criteria.
- *Interpretive validity* is operationalized by the [Mechanistic Interpretability Casebook](../lenses_v6/mechanistic_interpretability), which provides level declaration, level-evidence match, narrative coherence, alternative exclusion, and scope honesty criteria.

## How validity types connect to verdicts

A circuit claim must eventually address all five validity types. In practice, evidence accumulates incrementally, and the verdict tiers encode which types have been addressed so far:

| Verdict tier | Validity types addressed | What's still open |
|---|---|---|
| Proposed (A) | Construct (partial) | Internal, external, measurement, interpretive |
| Causally suggestive (B) | Construct (partial) + Internal (I1 only) | Sufficiency, specificity, consistency, external, measurement |
| Triangulated (D) | Construct (partial) + Internal (I1–I4) + External (partial) | Full construct, measurement, interpretive |
| Mechanistically supported (E) | Construct + Internal + External + Measurement | Interpretive |
| Validated (F) | All five types | — |

No published MI paper has yet reached the Validated tier under this framework. Most published circuits sit between Causally suggestive and Triangulated.

## Where the literature most often goes wrong

The taxonomy makes recurring error patterns diagnosable:

- **Construct conflation.** A circuit named for a behavior is treated as though the behavior and the circuit were the same concept. "The IOI circuit" conflates the behavior (indirect object identification) with the particular set of components found by a particular method.
- **Causal overreach.** Internal-validity evidence (ablation degrades performance) is reported as establishing external validity ("the model uses this circuit for IOI") without testing generalization.
- **Baseline omission.** Measurement-validity failures (M3) are presented as internal-validity successes. An IIA of 0.48 is impressive until the random-vector baseline turns out to be 0.44 ([Sutter et al. 2025](https://arxiv.org/abs/2412.09659)).
- **Single-prompt generalization.** External-validity claims are made from a single prompt distribution, treating the distribution as the phenomenon rather than a sample from it.
- **Level-evidence mismatch.** Implementational evidence (ablation) is presented as licensing algorithmic-level claims ("this head implements name-moving") without the causal abstraction evidence (IIA) required for the upgrade.

Each error is named by one of the five types, and each has a specific remedy.
