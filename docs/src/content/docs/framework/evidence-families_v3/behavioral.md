---
title: "Behavioral Evidence"
description: "Evidence from input-output analysis testing whether circuits reproduce full model behavior under controlled conditions"
---

# Behavioral Evidence

Behavioral evidence captures whether a proposed circuit actually reproduces the full model's behavior — testing the end-to-end functional claim that the circuit explains the phenomenon.

## What this family measures

Behavioral evidence comes from running a circuit (or ablated/patched model) on controlled inputs and comparing its outputs to the full model. This includes faithfulness metrics (does the circuit's output distribution match the full model's?), logit difference recovery (does the circuit produce the same preference between target tokens?), and transfer tests (does the circuit generalize across tasks or model scales?).

The logic is straightforward: if you claim "this circuit performs indirect object identification," then running the circuit in isolation should produce indirect-object-identification behavior. Behavioral evidence tests this directly by measuring how well the circuit's outputs match the full model's outputs on the relevant task.

This family also includes *negative* behavioral tests: does the circuit fail to produce behavior on tasks it should not explain? A name-mover circuit that also explains arithmetic behavior is either misidentified or captures something more general than claimed. Cross-task transfer and cross-scale transfer tests probe whether the circuit's behavioral profile matches its claimed scope.

## Metrics

- **D01 Faithfulness** — Overall output distribution match between circuit and full model
- **D02 Logit Diff Recovery** — Recovery of the logit difference between target tokens
- **D03 KL Divergence** — KL divergence between circuit and full model output distributions
- **D06 Cross-Task Transfer** — Whether a circuit identified on one task transfers to related tasks
- **D07 Cross-Scale Transfer** — Whether a circuit identified in one model size appears at other scales

## Characteristic strength

Behavioral evidence directly tests the bottom-line claim: does this circuit explain the behavior? All other evidence families provide supporting information, but ultimately a circuit claim is a claim about behavior — "this subset of the model is responsible for this input-output mapping." Behavioral evidence tests this directly.

This family also provides the most intuitive and communicable form of validation. Stakeholders can understand "the circuit recovers 95% of the full model's logit difference on IOI" without needing to understand SVD, mutual information, or counterfactual interventions.

## Characteristic blind spot

A faithful circuit may be faithful for the wrong reasons. If the evaluation distribution is too narrow, a circuit can achieve high faithfulness by overfitting to superficial features of the test set rather than capturing the true computational mechanism. A circuit that achieves 95% logit diff recovery on 100 IOI prompts with specific name frequencies might fail on prompts with different names or structures.

More fundamentally, behavioral equivalence does not establish mechanistic equivalence. Two entirely different computational strategies can produce the same input-output mapping on a finite test set. Behavioral evidence constrains what the circuit *does* but not *how* it does it.

## Criteria served

- **E1 Intervention reach** — Faithfulness metrics directly measure whether the circuit captures enough of the model's computation
- **E2 Graded response** — Partial ablation studies test whether degradation is proportional to circuit damage
- **E5 Robustness** — Cross-distribution behavioral tests reveal whether circuit claims are robust to input variation
- **E6 Cross-architecture** — Transfer tests across model scales probe whether circuits reflect general computational strategies

## Convergent validity role

Behavioral evidence is necessary but not sufficient. High faithfulness establishes that the circuit captures the right behavior, but combining it with causal evidence (which components are necessary) and structural evidence (what mechanisms underlie the behavior) provides a complete picture.

Behavioral + causal is particularly valuable: faithfulness shows the circuit works end-to-end while ablation shows which components are load-bearing within it. Behavioral + behavioral (e.g., faithfulness on two different prompt sets) increases confidence in robustness but does not illuminate mechanism. The strongest claims combine behavioral evidence with at least one mechanistic family (structural or representational) that explains *why* the circuit produces the observed behavior.
