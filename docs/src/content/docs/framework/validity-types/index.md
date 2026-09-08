---
title: "The Five Validity Types"
description: "Five independent, named failure modes for circuit claims — from construct coherence to interpretive warrant."
---

# The Five Validity Types

The framework organizes validity into five types, each asking a distinct question about a mechanistic claim. The vocabulary is Shadish, Cook & Campbell's (2002), developed for evaluating causal claims in social science and biomedical research. The ordering departs from theirs: we place construct validity first, following Messick (1995), who holds that construct validity "undergirds all score-based interpretations" and is "the essence of a unitary validity conception." The fifth type — interpretive validity — is added to address a gap the original taxonomy does not cover: whether the natural-language description of a mechanism is licensed by the evidence for it.

## Intellectual lineage

| Validity type | Origin | Foundational references |
|---|---|---|
| [Construct](/mechanistic-validity/framework/validity-types/construct/) | Philosophy of science, psychometrics | Cronbach & Meehl (1955); Messick (1995); Craver (2007) |
| [Measurement](/mechanistic-validity/framework/validity-types/measurement/) | Classical test theory, pharmacology (assay validation) | Spearman (1904); Borsboom, Mellenbergh & van Heerden (2004) |
| [Internal](/mechanistic-validity/framework/validity-types/internal/) | Experimental methodology, neuroscience, genetics | Campbell & Stanley (1963); Pearl (2009); Woodward (2003); Craver (2007) |
| [External](/mechanistic-validity/framework/validity-types/external/) | Pharmacology, experimental methodology | Rang et al. (2006); Pearl (2011); Steel (2008) |
| [Interpretive](/mechanistic-validity/framework/validity-types/interpretive/) | Philosophy of mind, mechanistic interpretability | Marr (1982); Geiger et al. (2021); Méloux et al. (2025) |

## Why five types rather than one global score

A single score summarizing "how good a circuit is" obscures the fact that circuit claims can fail in qualitatively different ways. A circuit can be measured by a reliable metric and still correspond to no coherent computational concept. A circuit can correspond to a coherent concept and rest on purely correlational evidence. A circuit can survive rigorous causal testing at one intervention strength on one prompt distribution and collapse under any other. A circuit can pass all of those tests and still be described at the wrong level of abstraction.

These are not points on a continuum; they are independent failures that demand independent remedies. The five-type taxonomy makes those failures named and reportable:

- A verdict that satisfies internal validity but not construct validity is *causally implicated but theoretically underspecified*. The remedy is a clearer construct, not more interventions.
- A verdict that satisfies construct validity but not external validity is *coherent but local*. The remedy is replication, not redefinition.
- A verdict that passes all four traditional types but fails interpretive validity is *validated but overclaimed*. The remedy is scoping the narrative to match the evidence level.

## The five types and their criteria

| Type | Question | Criteria |
|---|---|---|
| **[Construct](/mechanistic-validity/framework/validity-types/construct/)** | Is the thing being measured well-defined? | C1 Falsifiability · C2 Structural plausibility · C3 Convergent validity · C4 Discriminant validity · C5 Nomological validity · C6 Complementation validity |
| **[Measurement](/mechanistic-validity/framework/validity-types/measurement/)** | Are the instruments trustworthy? | M1 Reliability · M2 Baseline separation · M3 Stability · M4 Calibration · M5 Sensitivity · M6 Invariance · M7 Selection correction |
| **[Internal](/mechanistic-validity/framework/validity-types/internal/)** | Does the evidence support the causal claim? | I1 Necessity · I2 Sufficiency · I3 Minimality · I4 Specificity · I5 Rival mechanism exclusion · I6 Double dissociation · I7 Confound control · I8 Confounding sensitivity · I9 Epistatic interaction · I10 Rescue reversibility · I11 Onset coupling · I12 Offset coupling |
| **[External](/mechanistic-validity/framework/validity-types/external/)** | Does the mechanism generalize? | E1 Intervention reach · E2 Prompt generalization · E3 Cross-task generalization · E4 Cross-model generalization · E5 Graded response · E6 Novel prediction |
| **[Interpretive](/mechanistic-validity/framework/validity-types/interpretive/)** | Is the interpretation of the mechanism correct? | V1 Level declaration · V2 Level-evidence match · V3 Alternative level · V4 Unlicensed labeling · V5 Scope declaration |

## Dependency chain

The five types form a dependency chain:

> **Construct → Measurement → Internal → External → Interpretive**

The ordering reflects logical precedence: a construct that is not well-defined cannot be reliably measured, an unreliable measurement cannot support a causal claim, a causal claim in one setting cannot be generalized, and an interpretation cannot be evaluated until the underlying evidence is characterized.

The dependency is logical, not temporal. Work on any type need not wait for the previous one to be finished. The chain means that a verdict at any level should name the types at which evidence is missing, rather than upgrading the verdict on the strength of evidence from a different type.

## Cross-disciplinary foundations

Eight disciplines contribute to the framework. Each grounds one or two validity types:

| Discipline | Validity type(s) | Principal transfer |
|---|---|---|
| Philosophy of science | Construct (C) | Falsifiability, severe testing, confirmation/corroboration distinction |
| Psychometrics | Construct (C), Measurement (M) | Convergent/discriminant validity (C3, C4); reliability and baseline separation |
| Causal inference | Internal (I) | do-calculus; ablation vs patching as formally distinct interventions |
| Neuroscience | Internal (I) | Lesion vs stimulation, double dissociation, convergence across modalities |
| Genetics | Internal (I) | Epistasis, rescue experiments, sensitivity analysis |
| Medical microbiology | Internal (I) | Graded presence, partial satisfaction for causal conclusion |
| Pharmacology | External (E) | Dose-response, affinity vs efficacy, system reserve |
| Mechanistic interpretability | Interpretive (V) | Description vs explanation, faithfulness vs understanding |

## Verdict tiers

The framework assigns each audited claim to one of five verdict tiers, or replaces the tier with one of three diagnostic labels. The tiers encode which validity types have been addressed:

| Tier | Meaning | Minimum evidence |
|---|---|---|
| **Proposed** | Structural or representational evidence only | C1–C2 defined; at least one admissible measurement |
| **Causally Suggestive** | Necessity shown, sufficiency not established | I1 confirmed; M2 passes |
| **Mechanistically Supported** | Necessity + sufficiency with consistent methods | I2 established; E1 across ≥2 methods; I4 at least partially confirmed |
| **Triangulated** | Multiple converging lines of independent evidence | C3 convergence; E2/E4 replication; I6 dissociation; I5 rival exclusion; C4 discriminant; I7 confound control |
| **Validated** | Characterized, not merely identified | M1–M6; V1–V5; E2–E6; C5; C6, I3, I10–I12 |
| **Underdetermined** | Multiple mechanisms fit | Cannot resolve between rivals |
| **Insufficient** | Cannot be assessed | Construct not defined enough to score |
| **Disconfirmed** | Fails decisively | Negative result on required criterion |

No published MI paper has reached the Validated tier under this framework. Most published circuits sit between Causally Suggestive and Mechanistically Supported. The single Triangulated verdict — induction heads for token copying — required three years of follow-up work across multiple research groups.

## Common error patterns

The taxonomy makes recurring error patterns diagnosable:

- **Construct conflation.** A circuit named for a behavior is treated as though the behavior and the circuit were the same concept. "The IOI circuit" conflates the behavior (indirect object identification) with the particular set of components found by a particular method.
- **Causal overreach.** Internal-validity evidence (ablation degrades performance) is reported as establishing external validity ("the model uses this circuit for IOI") without testing generalization across prompts, methods, or models.
- **Baseline omission.** Measurement-validity failures (M2) are presented as internal-validity successes. An IIA of 0.48 is uninterpretable until the random-vector baseline turns out to be 0.44.
- **Single-prompt generalization.** External-validity claims (E2) are made from a single prompt distribution, treating the distribution as the phenomenon rather than a sample from it.
- **Level-evidence mismatch.** Implementational evidence (ablation) is presented as licensing algorithmic-level claims ("this head implements name-moving") without the causal abstraction evidence (IIA) required at V2 for that upgrade.

Each error is named by one of the five types, and each has a specific remedy described on the relevant subpage.
