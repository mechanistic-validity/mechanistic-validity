---
title: "Case Studies"
description: "Thirteen published mechanistic claims evaluated through all five validity lenses."
---

# Case Studies

Each case study below takes a published mechanistic claim and evaluates it through all five validity lenses — construct, internal, external, measurement, and interpretive. The goal is not to rank papers but to show what the framework looks like in practice: where evidence is strong, where it is absent, and what the composite verdict means.

The case studies are ordered roughly by overall verdict strength, from the strongest claims to the weakest.

---

## Validated (within scope)

Claims with complete evidence across all lenses — limited only by scope.

| Case Study | Claim | Key insight |
|---|---|---|
| [Grokking / Modular Addition](/framework/lenses_v6/examples/examples-grokking) | Fourier algorithm in toy transformer | The ceiling — what "fully understood" looks like. Every weight matrix explained. |
| [Superposition](/framework/lenses_v6/examples/examples-superposition) | Features packed as near-orthogonal directions | Validated theory awaiting real-model confirmation. Toy → real gap is the open question. |

---

## Triangulated

Evidence converges across multiple independent lenses.

| Case Study | Claim | Key insight |
|---|---|---|
| [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) | Two-head composition for in-context copying | The gold standard in real models. Simple mechanism, broad replication, thick nomological network. |

---

## Causally suggestive

Strong causal evidence with identifiable gaps preventing advancement.

| Case Study | Claim | Key insight |
|---|---|---|
| [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) | 26-head indirect object identification mechanism | Most thoroughly analyzed circuit. Strong I1/I2, but method-conditional and specificity untested. |
| [Greater-Than](/framework/lenses_v6/examples/examples-greater-than) | Successor heads encoding ordinal year comparison | Best structural plausibility in MI. $W_{OV}$ ordering evidence is the model for C2. |
| [Successor Heads](/framework/lenses_v6/examples/examples-successor-heads) | General-purpose ordinal mechanism across domains | Cross-domain generalization as convergent evidence. Stronger "natural kind" case than single-task circuits. |
| [Copy Suppression](/framework/lenses_v6/examples/examples-copy-suppression) | Heads that actively suppress incorrect token copying | Unusually clean specificity — ablation produces a specific error type, not general degradation. |
| [Docstring Circuit](/framework/lenses_v6/examples/examples-docstring) | Variable binding in Python docstrings | Illustrates label risk: "variable binding" vs. simpler "positional copying" not distinguished. |
| [Knowledge Neurons / ROME](/framework/lenses_v6/examples/examples-knowledge-neurons) | Factual knowledge localized in MLP layers | A tool can work for the wrong reasons. Strong intervention, weak mechanistic story. |
| [Othello World Model](/framework/lenses_v6/examples/examples-othello) | Linear board-state representation | Interpretive inflation: "world model" carries implications beyond "linearly decodable." |

---

## Proposed

Claims where evidence has not yet established validity beyond initial identification.

| Case Study | Claim | Key insight |
|---|---|---|
| [SAE Features](/framework/lenses_v6/examples/examples-sae-features) | Dictionary directions as computational units | Thin nomological network. Features may be properties of the dictionary, not the model. |
| [Probing Classifiers](/framework/lenses_v6/examples/examples-probing) | Linear decodability implies representation | Measurement without intervention = no internal validity. Decodable ≠ encoded. |
| [Gender Bias Circuits](/framework/lenses_v6/examples/examples-gender-bias) | Bias localized in removable components | Construct incoherence: bias and knowledge share circuits. The construct itself may not be separable. |

---

## Reading the case studies

Each case study follows the same structure:

1. **Introduction** — what the claim is and why it matters
2. **Five lens evaluations** — each with per-criterion verdicts (Pass / Partial / Not tested / Weak) and a summary table
3. **Composite verdict** — a table showing the strongest and weakest criterion per lens, plus the overall verdict

The per-criterion verdicts use consistent language:
- **Pass** — evidence is present and sufficient
- **Partial** — some evidence exists but with gaps
- **Not tested** — this criterion was not evaluated in the published work
- **Weak** — evidence exists but is inadequate or contradicted
- **N/A** — the criterion does not apply to this type of claim

---

## Patterns across case studies

Several patterns emerge from evaluating these claims side by side:

**The sufficiency gap.** Most circuits demonstrate necessity (I1) but not sufficiency (I2). Only induction heads and grokking demonstrate path-level or full sufficiency.

**Method-conditional results.** IOI's headline numbers (87% faithfulness) are specific to mean ablation. [Miller et al. (2024)](https://arxiv.org/abs/2407.08734) show these drop below 50% under other methods. Ablation type is part of the claim.

**The toy-model ceiling.** Grokking and superposition reach Validated — but only within toy scope. The gap between toy-model proof-of-concept and real-model confirmation is the field's central challenge.

**Interpretive inflation.** "World model," "deception feature," "knowledge neuron" — labels that carry theoretical implications beyond what the evidence supports. The framework systematically identifies where labels exceed evidence (V5 scope honesty).

**Construct incoherence.** Gender bias circuits fail not because evidence is lacking but because the construct itself cannot be separated from legitimate gender processing. Sometimes the right answer is "this question is not well-posed," not "we need more data."
