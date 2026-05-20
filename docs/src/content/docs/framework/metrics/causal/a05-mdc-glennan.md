---
title: "A05 — MDC / Glennan New Mechanism"
description: "The new mechanistic philosophy (Machamer-Darden-Craver, Glennan) applied to transformer circuits: components organized to produce phenomena."
---

# A05 — MDC / Glennan New Mechanism

This framework asks: **can the circuit be described as entities and activities organized such that they produce the phenomenon — and can we verify this organization empirically?**

The "new mechanism" philosophy (Machamer, Darden & Craver 2000; Glennan 2002) defines a mechanism as entities and activities organized in such a way that they are responsible for the phenomenon. This goes beyond mere causal sufficiency (A01) to require that the *organization* — the spatial, temporal, and functional arrangement of parts — be explanatorily relevant. In MI terms: it is not enough to show that a set of heads causally mediates IOI behavior; you must show that those heads are organized into a specific information-processing pipeline (name movers receive from S-inhibition heads, which receive from duplicate-token detectors) and that this organization is what produces the behavior.

This framework motivates two distinctive metrics: weight-space circuit verification (checking that the organizational structure exists in the weights independent of any particular input) and logic-gate analysis (verifying that individual components implement specific Boolean or arithmetic operations as the mechanism hypothesis predicts).

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Machamer, Darden & Craver, "Thinking About Mechanisms"](https://doi.org/10.1086/392611) | 2000 | Mechanisms as entities + activities organized to produce phenomena |
| [Glennan, "Rethinking Mechanistic Explanation"](https://doi.org/10.1086/341858) | 2002 | Mechanisms as complex systems with interacting parts |
| [Craver, *Explaining the Brain*](https://global.oup.com/academic/product/explaining-the-brain-9780199568222) | 2007 | Levels of mechanism; constitutive vs. etiological explanation |
| [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | QK/OV circuits as organized entities producing attention phenomena |
| [Olsson et al., "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 2022 | Induction heads as a multi-component mechanism with specific organization |

## Core concept: organization as explanation

The MDC/Glennan framework distinguishes between two kinds of causal claim: (1) "these components are causally relevant to the output" (etiological) and (2) "these components are *organized in this specific way* to produce the output" (constitutive/mechanistic). Standard activation patching (A01) provides type (1). The mechanistic framework demands type (2): you must specify the organizational structure and verify it.

For transformer circuits, organization means: which heads write information that other heads read (via the residual stream), what computation each head performs on its inputs, and how the pipeline's stages depend on each other. Weight-space analysis can verify organization independently of activation measurements — if head A's OV circuit projects into the subspace that head B's QK circuit reads from, the organizational link exists in the weights regardless of whether any particular input activates it.

## Metrics under A05

### C18 — Weight-Extended Circuit (`18_weight_extended.py`)

Verifies circuit organization by checking weight-space connections between components. For each edge in the hypothesized mechanism, measures the alignment between the writing head's output subspace and the reading head's input subspace:

\[
\text{Connection}(A \to B) = \| W_O^A \cdot W_Q^B \|_F / (\|W_O^A\|_F \cdot \|W_Q^B\|_F)
\]

This establishes that the organizational structure hypothesized by the mechanism exists in the weights — the "wiring" is physically present, not just functionally inferred from activations.

**What it establishes:** Structural organization in weight space; that components are "wired" to communicate.

**What it does not establish:** That the wiring is *used* on any particular input (requires activation-level verification from A01/A02).

**Usage:**
```
uv run python 18_weight_extended.py --tasks ioi sva --n-prompts 40
```

### C19 — Logic Gates (`19_logic_gates.py`)

Tests whether individual components implement specific Boolean or arithmetic operations as the mechanism hypothesis predicts. For example, verifying that an "AND-gate" head only fires when both its input conditions are satisfied, or that a "copy" head's OV circuit has high rank-1 structure aligned with the identity.

**What it establishes:** That individual entities in the mechanism perform the *activities* the hypothesis attributes to them.

**What it does not establish:** That these activities are necessary for the overall phenomenon (requires ablation from A01).

**Usage:**
```
uv run python 19_logic_gates.py --tasks ioi --n-prompts 40
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High weight-space connection along hypothesized edges | Organization exists structurally; mechanism is "wired" |
| Logic gates pass for all components | Each entity performs its attributed activity |
| Weight connection exists but logic gate fails | Structure is present but the component's operation is mis-characterized |
| Logic gate passes but weight connection is weak | Component performs the operation but is not strongly connected to the next stage |

