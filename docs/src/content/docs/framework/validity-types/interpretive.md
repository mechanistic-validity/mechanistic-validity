---
title: "Interpretive Validity"
---

# Interpretive Validity

Interpretive validity asks whether the natural-language description attached to a mechanism is justified by the evidence. A head called a "name mover" on the basis of direct logit attribution carries theoretical implications beyond what the evidence strictly supports — the label asserts an algorithmic role, while the evidence establishes only a causal contribution at the implementational level. Interpretive validity is the standard that separates licensed descriptions from overclaimed narratives.

## Position in the Dependency Chain

```
Construct → Measurement → Internal → External → **Interpretive**
```

Interpretive validity occupies the fifth and final position in the dependency chain. An interpretation cannot be evaluated until the underlying evidence is characterized: the construct must be defined (construct validity), the instruments must be trustworthy (measurement validity), the causal claim must be established (internal validity), and the generalization must be tested (external validity). Only then can we ask whether the story told about the mechanism matches the evidence that supports it.

This ordering reflects the fact that interpretive failures are qualitatively different from evidential failures. A mechanism can pass every causal and generalization test and still be described at the wrong level of abstraction, or given a name that imports unmeasured properties. The remedy for an interpretive failure is not more evidence — it is a more careful narrative.

## Theoretical Lineage

Marr (1982) introduced the three-level framework — computational, algorithmic, and implementational — that structures how we describe information-processing systems. A computational-level claim states what problem a system solves; an algorithmic-level claim states how it solves it; an implementational-level claim states which physical components carry the computation. Each level licenses different inferences and requires different evidence. The mechanistic validity framework extends Marr's three levels to seven description modes, but the core insight is Marr's: a description at one level does not automatically license claims at another.

Shagrir (2017) formalized how Marr's computational level delineates the phenomenon a mechanistic explanation targets. The computational level is not merely a high-level summary — it defines what counts as the explanandum. A mismatch between the computational-level description and the phenomenon under investigation renders downstream mechanistic claims vacuous, because the explanation targets the wrong thing. This work grounds V1 (level declaration): a claim that does not state its level cannot be evaluated for whether it targets the right phenomenon.

Messick (1995) reframed validity as the warrant for interpretations of scores, not as a property of the test itself. Under Messick's unified conception, a test score is valid or invalid only relative to a specific interpretation and use. We adopt the same stance for mechanistic claims: a circuit is not "valid" or "invalid" in isolation — the question is whether a specific interpretive claim about the circuit is warranted by the evidence. This framing grounds V4 (unlicensed labeling) and V5 (scope declaration), where the gap between evidence and interpretation is the object of evaluation.

Craver (2006) distinguished how-possibly explanations from how-actually explanations in neuroscience. A how-possibly explanation shows that a mechanism *could* produce the target phenomenon; a how-actually explanation shows that it *does*. Most circuit claims in mechanistic interpretability are how-possibly explanations presented as how-actually explanations — ablation shows that a component is causally involved, but the narrative asserts a specific algorithmic role that was not directly tested. Méloux et al. (2025) made the how-possibly/how-actually gap concrete by proving that on Boolean MLPs small enough to enumerate exhaustively, multiple circuits replicate the same behavior and multiple interpretations fit the same circuit. Alternative exclusion is therefore a required step, not an optional one, and V3 (alternative level) operationalizes it.

## Criteria

| ID | Name | Question |
|---|---|---|
| V1 | Level declaration | At what description mode is the claim made? |
| V2 | Level-evidence match | Does the evidence support claims at that level? |
| V3 | Alternative level | Could the evidence be explained at a different level? |
| V4 | Unlicensed labeling | Does a name import a property that was not measured? |
| V5 | Scope declaration | What does the claim explicitly not cover? |

V1–V3 form a sequence: declare the level, check that the evidence matches the level, then check whether a simpler level would account for the same evidence. V4 targets the labels themselves — names like "world model" or "knowledge neuron" that carry theoretical commitments not present in the supporting evidence. V5 requires that the claim state its own boundaries.

### Novelty

Three of the interpretive criteria — V1, V2, and V3 — have no counterpart in the validity frameworks we survey. Traditional validity theory evaluates whether a score means what its user takes it to mean, but does not ask at which level of description the claim is pitched or whether the evidence licenses a claim at that level rather than a simpler one. These criteria address a failure mode specific to mechanistic interpretability, where the gap between implementational evidence and algorithmic or computational narratives is routinely crossed without additional evidence.

## Failure Examples

**Othello "world model" (V4).** Li et al. trained a probe showing that an Othello-playing network maintains an internal representation isomorphic to the board state. The representation was labeled a "world model." The evidence supports "causally-used state summary" — the network maintains and uses a board-state representation. The label "world model" imports properties that were never measured: generative capacity, counterfactual reasoning, and transfer to novel game states. The gap runs between evidence for one reading and a name that asserts a second. Subsequent work by Neel Nanda and others investigated whether the representation supports counterfactual board states and found mixed evidence, but the original label was adopted widely before these tests were conducted.

**Knowledge neurons (V2).** Dai et al. identified neurons whose activation correlates with the expression of specific factual knowledge and labeled them "knowledge neurons." The evidence is correlational: activating these neurons is associated with correct factual recall. The label "storage" — as in, these neurons *store* knowledge — is a causal and algorithmic claim that requires intervention evidence (does suppressing the neuron remove the knowledge? does activating it insert the knowledge?) and level-appropriate causal abstraction (does the neuron implement a retrieval operation?). Correlation was reported; storage was claimed. The level-evidence mismatch (V2) is between correlational evidence and a causal-algorithmic interpretation.

## Cross-Disciplinary Foundations

| Discipline | Transfer to interpretive validity |
|---|---|
| Mechanistic interpretability | The primary source — V1–V5 address the gap between description and explanation, and between faithfulness (behavioral replication) and understanding (correct narrative about mechanism). |
| Philosophy of science | Falsifiability and severe testing ground V2 (level-evidence match): a claim at a given level must be testable at that level, not merely consistent with evidence from a lower level. The confirmation/corroboration distinction separates a test a claim was built to pass from one it could have failed. |
| Psychometrics | Messick's unified validity conception — validity as warrant for interpretation, not property of the instrument — provides the conceptual frame. Convergent and discriminant validity (C3, C4) feed into V4: a label that cannot be discriminated from a neighboring construct is unlicensed. |
| Neuroscience | Craver's how-possibly vs how-actually distinction maps directly onto the interpretive gap in MI: most circuit claims are how-possibly explanations reported as how-actually. Double dissociation (I6) provides the strongest evidence for distinguishing two interpretations of overlapping circuits. |
