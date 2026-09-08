---
title: "Case Studies"
description: "Sixteen published mechanistic claims evaluated through all five validity types, with verdicts from Proposed through Triangulated."
---

# Case Studies

Each case study takes a published mechanistic claim and evaluates it through all five validity types — construct, measurement, internal, external, and interpretive. The evaluation uses the framework's [36 criteria](/mechanistic-validity/framework/criteria/) and the [six-category status vocabulary](/mechanistic-validity/framework/criteria/) (Confirmed, Partially confirmed, Inconclusive, Disconfirmed, Untested, Not applicable). The verdict names what the claim has established and what caps it from advancing further.

Sixteen claims from fifteen papers are audited. No claim reaches [Validated](/mechanistic-validity/framework/verdicts/validated/). One reaches [Triangulated](/mechanistic-validity/framework/verdicts/triangulated/). Seven reach [Mechanistically Supported](/mechanistic-validity/framework/verdicts/mechanistically-supported/). Four are [Causally Suggestive](/mechanistic-validity/framework/verdicts/causally-suggestive/). Two are [Proposed](/mechanistic-validity/framework/verdicts/proposed/). Two are [Disconfirmed](/mechanistic-validity/framework/verdicts/disconfirmed/).

---

## Triangulated

Evidence converges across multiple independent lines — no single method's failure collapses the claim.

| Case Study | Claim | Verdict | Capping criteria |
|---|---|---|---|
| [Induction Heads](/mechanistic-validity/framework/audits/induction/) | Two-head composition for in-context token copying | Triangulated | C6 (complementation), I3 (minimality), I10 (rescue), I12 (offset coupling) |

Induction heads are the strongest claim in the corpus. The mechanism is confirmed across models, model sizes, and training checkpoints. Onset coupling (I11) is demonstrated via the phase transition in training loss. What caps it: complementation validity (C6) — the labeled subdivisions of the mechanism (previous-token heads, induction heads) have not been tested for functional distinctness — and three internal criteria that require training-history evidence the original work did not collect.

---

## Mechanistically Supported

Necessity and sufficiency established with consistent methods; specificity at least partially confirmed.

| Case Study | Claim | Verdict | Capping criteria |
|---|---|---|---|
| [Copy Suppression](/mechanistic-validity/framework/audits/copy_suppression/) | Heads that actively suppress incorrect token copying | Mech. Supported | I6 (double dissociation) |
| [Greater-Than](/mechanistic-validity/framework/audits/greater_than/) | Attention heads plus MLPs 8–11 raising probability of every year above the start year | Mech. Supported | I5 (rival exclusion), I6 (double dissociation) |
| [Modular Addition](/mechanistic-validity/framework/audits/grokking/) | Fourier multiplication algorithm in toy transformer | Mech. Supported | I6 (double dissociation) |
| [Refusal Direction](/mechanistic-validity/framework/audits/refusal/) | Single direction mediating safety refusal | Mech. Supported | I6 (double dissociation) |
| [Successor Heads](/mechanistic-validity/framework/audits/successor_heads/) | General-purpose ordinal mechanism across domains | Mech. Supported | I6 (double dissociation) |
| [Superposition](/mechanistic-validity/framework/audits/superposition/) | Features packed as near-orthogonal directions in toy models | Mech. Supported | I6 (double dissociation) |
| [Global Workspace](/mechanistic-validity/framework/audits/workspace/) | Shared representational subspace (J-space) across tasks | Mech. Supported | I6 (double dissociation) |

Double dissociation (I6) caps every claim at this tier. I6 requires a crossed design — two interventions, each breaking what the other spares — rather than an accumulation of evidence. It is scored as met or unmet with no partial credit, because each arm is already scored elsewhere (I1 for the necessity arm, M2 for the baseline arm). Across sixteen audited claims, I6 is met once: by the induction heads claim, via Feucht et al. (2025), three years after the origin paper.

---

## Causally Suggestive

Necessity shown via causal intervention; sufficiency, specificity, or convergence not yet established.

| Case Study | Claim | Verdict | Capping criteria |
|---|---|---|---|
| [Docstring Circuit](/mechanistic-validity/framework/audits/docstring/) | Variable binding in Python docstrings | Causally Suggestive | I4 (specificity) |
| [Gender Bias Circuits](/mechanistic-validity/framework/audits/gender/) | Bias localized in removable components | Causally Suggestive | E1 (intervention reach), I4 (specificity) |
| [IOI Circuit](/mechanistic-validity/framework/audits/ioi/) | 26-head indirect object identification mechanism | Causally Suggestive | E1 (intervention reach), I4 (specificity) |
| [Othello Board State](/mechanistic-validity/framework/audits/othello/) | Nonlinearly decodable board-state representation | Causally Suggestive | E1 (intervention reach) |

The IOI circuit is the most thoroughly analyzed circuit in the literature. Necessity is partial — knocking out all three Name Mover heads costs only 5% of the logit difference, because backup heads take over — and sufficiency is partial at 87% under mean ablation. That headline number is method-conditional: Miller et al. (2024) move the same quantity from below 0% to over 100% across six methodological choices, which is why stability (M3) and invariance (M6) are both Disconfirmed. The ablation method is part of the claim.

---

## Proposed

Structural or representational evidence only — no causal intervention establishes the mechanism.

| Case Study | Claim | Verdict | Capping criteria |
|---|---|---|---|
| [Probing Classifiers](/mechanistic-validity/framework/audits/probing/) | Linear decodability implies representation | Proposed | I1 (necessity) |
| [SAE Features](/mechanistic-validity/framework/audits/sae/) | Sparse autoencoder directions as computational units | Proposed | M2 (baseline separation) |

Probing demonstrates that information is linearly accessible in the representation. It does not demonstrate that the model's own computation accesses it. SAE features face a measurement validity gap: SAEBench showed that some evaluation metrics score higher on random models than trained ones, undermining baseline separation (M2). Moving to Causally Suggestive requires establishing necessity via causal intervention and demonstrating that the measurement distinguishes learned structure from random baselines.

---

## Disconfirmed

A specific prediction of the claimed mechanism was tested and failed.

| Case Study | Claim | Verdict | Capping criteria |
|---|---|---|---|
| [Induction Heads (General ICL)](/mechanistic-validity/framework/audits/induction_broad/) | Induction heads implement general in-context learning | Disconfirmed | I1 (necessity) |
| [Knowledge Neurons](/mechanistic-validity/framework/audits/knowledge_neurons/) | Factual knowledge localized in MLP neurons | Disconfirmed | I4 (specificity) |

The induction heads claim has two readings. Token copying (the narrow claim) reaches Triangulated. General in-context learning (the broad claim) is Disconfirmed: induction heads are necessary for copying-based ICL but not for the broader capability. Knowledge neurons are Disconfirmed on specificity: the editing intervention that updates a fact also raises inter-relation perplexity, indicating that the "knowledge" is not localized in the way the name implies.

---

## Cross-cutting patterns

### I6 caps the field

Double dissociation (I6) is Untested in fourteen of sixteen claims, Inconclusive in one, and caps every claim that reaches Mechanistically Supported. The field rarely attempts the crossed design that I6 requires. The one claim that meets I6 — induction heads, via Feucht et al. (2025) — advances to Triangulated.

### I8 is universally Untested

Confounding sensitivity (I8) asks how strong an unmeasured confounder would have to be to explain the observed result. No claim in the corpus reports this bound. The analogue in observational epidemiology is the E-value; in genetics, it is the sensitivity analysis for unmeasured confounding. The absence is not a verdict-capping failure (I8 is not gating for any tier), but it is a systematic gap.

### The toy-model ceiling

Modular addition and superposition reach Mechanistically Supported, not Validated, despite having complete mechanistic accounts within their toy-model scope. Both are capped by double dissociation (I6): importance and sparsity are crossed, but on one outcome, and no converse arm is run. Above that, complementation validity (C6), rescue reversibility (I10) and offset coupling (I12) remain untested.

### Interpretive inflation

"World model" (Othello), "knowledge neuron" (Dai et al.), "monosemantic feature" (SAEs) — labels that carry theoretical implications beyond what the evidence supports. The framework identifies these via [V4 Unlicensed labeling](/mechanistic-validity/framework/criteria/interpretive/unlicensed-labeling/): a name that imports a property the evidence did not measure.

## Reading the case studies

Each case study follows the same structure:

1. **Introduction** — what the claim is and why it matters
2. **Five validity-type evaluations** — each with per-criterion verdicts using the six-category status vocabulary (Confirmed, Partially confirmed, Inconclusive, Disconfirmed, Untested, Not applicable)
3. **Composite verdict** — a table showing the verdict tier, capping criteria, and primary gap

## Sensitivity

Five of the criterion judgments above are contested within the audit records, and one of them
moves a verdict. See [Sensitivity Analysis](/mechanistic-validity/framework/audits/sensitivity/).

## Lens applications

Alongside each audit there is an exploratory reading of the same claim through the framework's
five lenses, written for this site and not part of the paper. Each audit page links to its own.
