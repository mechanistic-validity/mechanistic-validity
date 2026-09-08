---
title: "Construct Validity"
description: "Is the thing being claimed a coherent theoretical entity? Construct validity asks whether the target concept is well-defined before any measurement is taken."
---

# Construct Validity

Construct validity asks whether the thing being measured is well-defined. Before any measurement is taken, the construct must be specified precisely enough that the claim is falsifiable, structurally plausible, and distinguishable from neighboring constructs. A circuit claim whose target concept is incoherent — where "induction head" could mean token-copying, prefix-matching, or in-context learning depending on which paragraph the reader is in — cannot be rescued by any amount of causal evidence.

## Position in the Dependency Chain

```
Construct → Measurement → Internal → External → Interpretive
    ▲
  you are here
```

Construct validity comes first in the dependency chain. The ordering follows Messick (1995), who argued that construct validity "undergirds all score-based interpretations" and is "the essence of a unitary validity conception." A construct that is not well-defined cannot be reliably measured (Measurement depends on Construct), an unreliable measurement cannot support a causal claim (Internal depends on Measurement), a causal claim in one setting cannot be generalized (External depends on Internal), and an interpretation cannot be evaluated until the underlying evidence is characterized (Interpretive depends on External).

The standard Shadish, Cook & Campbell ordering places construct validity third, paired with external validity. We place it first because a claim whose target concept is incoherent cannot be rescued by any amount of causal evidence, so construct has to come first for the same reason Messick gives.

All four downstream validity types inherit construct validity failures. If the construct is ambiguous, measurement validity evaluates instruments against an ambiguous target. Internal validity establishes causal involvement in an ambiguous behavior. External validity generalizes an ambiguous finding. Interpretive validity narrates an ambiguous mechanism. Fixing any of those cannot fix the ambiguity at the source.

## Theoretical Lineage

Cronbach and Meehl (1955) developed construct validity for measurement instruments in psychology. The object was a test score, and the question was whether the score means what its user takes it to mean. Their framework required that a construct be embedded in a nomological network — a set of lawful relationships linking the construct to other constructs and to observable indicators. A construct that participates in no such network is, in their formulation, scientifically empty. We transfer this requirement directly: a circuit claim must specify what the circuit does, what it does not do, and how those two relate to other known circuits and behaviors.

Messick (1995) unified construct validity with all other forms of validity, arguing that reliability, criterion validity, and content validity are facets of a single construct-validity question rather than independent properties. Under this unification, asking whether a metric is reliable (Measurement) or whether an intervention is specific (Internal) are both, ultimately, construct-validity questions asked at different stages of evidence accumulation. We adopt the unification but preserve the five-type decomposition because the remedies differ: a construct problem is fixed by redefining the target, not by adding baselines or interventions.

The falsifiability requirement (C1) draws on Popper's demarcation criterion and on Mayo's (1996, 2018) error-statistical refinement. Popper required that a scientific claim specify conditions under which it would be refuted. Mayo sharpened this into the notion of severe testing: a claim passes a severe test only when the test had a high probability of detecting the error, if the error were present. A circuit claim that specifies no falsification condition (no metric, no threshold, no held-out dataset) cannot be severely tested.

The operationalism of Bridgman (1927) contributes a further constraint: a concept is defined by the operations used to measure it. In mechanistic interpretability, this means that "induction head" is defined by the behavioral and structural tests that identify it — prefix-matching attention pattern, high copying score, formation during training — not by an intuitive notion of what induction means. The psychometrics tradition contributes convergent and discriminant validity from the multitrait-multimethod matrix (Campbell & Fiske, 1959), which we transfer as C3 and C4: multiple independent methods should agree on the same construct (convergent), and the construct should be distinguishable from neighboring constructs measured by the same method (discriminant).

## Criteria

| ID | Name | Question |
|---|---|---|
| C1 | Falsifiability | Can the claim be refuted by a specified observation? |
| C2 | Structural plausibility | Is the mechanism physically possible in the architecture? |
| C3 | Convergent validity | Do multiple independent methods agree on the same components? |
| C4 | Discriminant validity | Does the measure distinguish this construct from neighboring constructs? |
| C5 | Nomological validity | Does the claim fit into a broader network of established relationships? |
| C6 | Complementation validity | Are the construct's labeled subdivisions functionally distinct? |

C1–C2 define the construct: C1 requires that it be falsifiable, C2 that it be structurally possible given the architecture. C3–C4 position the construct relative to other constructs and other methods, transferring the multitrait-multimethod logic of Campbell and Fiske (1959). C5 embeds the construct in a nomological network in Cronbach and Meehl's sense. C6 asks whether subdivisions within the construct (e.g., "name-mover heads" vs. "backup name-mover heads" within the IOI circuit) are functionally distinct, borrowing from the complementation test in genetics (Benzer, 1955).

## Failure Examples

**Google Flu Trends (C1, M6).** Google Flu Trends correlated search-query volume with CDC influenza surveillance data and was initially accurate. Over time, the system became "part flu detector, part winter detector" — the construct drifted from influenza incidence to seasonal search behavior without any falsification condition that could have detected the shift. The failure is a construct validity failure: the target concept was not specified precisely enough to distinguish flu-driven queries from winter-driven queries, and no threshold was stated in advance that would have flagged the divergence.

**Gender bias circuits (C4).** Work on gender bias circuits in language models presupposes that bias separates from gender competence — that a circuit implementing gendered-pronoun prediction can be isolated from a circuit implementing grammatical agreement. This separation was assumed, not tested. Similarly, a reported "deception feature" in a sparse autoencoder cannot be distinguished from an "uncertainty feature" because the two constructs predict the same activation pattern on the tested inputs. Both failures are discriminant validity failures: the construct was not shown to be distinguishable from a neighboring construct that makes overlapping predictions.

**Attribution method disagreement (C3).** Krishna et al. applied six attribution methods to the same models and found that they disagree substantially on which components are important. The field has no framework for reading this disagreement as information about the instruments rather than noise to be averaged away. The failure is a convergent validity failure: if multiple independent methods do not agree on the same construct, either the methods are measuring different things or the construct is not well-defined enough to produce agreement. Without construct validity as an organizing frame, the disagreement is uninterpretable.

## Cross-Disciplinary Foundations

| Discipline | Transfer to construct validity |
|---|---|
| Philosophy of science | Falsifiability (C1), severe testing, and the confirmation/corroboration distinction — separating a test a claim was built to pass from one it could have failed |
| Psychometrics | Convergent and discriminant validity from the multitrait-multimethod matrix (C3, C4); the nomological network requirement (C5) |

The remaining six disciplines in the framework's theoretical foundations (causal inference, neuroscience, genetics, medical microbiology, pharmacology, mechanistic interpretability) ground other validity types. Construct validity draws primarily from philosophy of science and psychometrics because the question — is the concept well-defined? — is a question those two disciplines have addressed most directly.
