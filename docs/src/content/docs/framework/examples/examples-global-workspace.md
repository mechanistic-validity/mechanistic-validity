---
title: "Exploratory Lens Analysis: Global Workspace"
description: "The global workspace / J-space hypothesis (Gurnee et al. 2026) evaluated through the five core lenses."
prev:
  link: /mechanistic-validity/framework/examples/examples-superposition/
  label: "Exploratory Lens Analysis: Superposition"
next:
  link: /mechanistic-validity/framework/examples/examples-docstring/
  label: "Exploratory Lens Analysis: Docstring Circuit"
---

# Exploratory Lens Analysis: Global Workspace

:::note[Disclaimer]
This page is an exploratory reading, not part of the paper. It applies the framework's five lenses to the claim as an illustration. For the audit as published, see [Case Studies: Global Workspace](/mechanistic-validity/framework/audits/workspace/).
:::


Gurnee et al. (2026) provide evidence for a **global workspace** structure in transformer language models — a low-dimensional subspace of the residual stream (dubbed "J-space"), recovered as the expected Jacobian $J_\ell$ of the final-layer residual with respect to layer $\ell$, proposed as the set of contents the model can verbally report and reason over. The claim is that the residual stream is not used uniformly: a structured, shared subspace mediates inter-component communication, analogous to the global workspace theory in cognitive neuroscience (Baars, 1988; Dehaene et al., 2014).

**Description mode:** `[implementational-functional]`. The claim specifies what the subspace does — carries the information that components read from and write to — without asserting an algorithmic account of the routing procedure itself. See [Description Modes](/mechanistic-validity/framework/description-modes/).

## Verdict: Mechanistically Supported

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I6 (double dissociation).


| Validity type | Status | Key evidence |
|---|---|---|
| Construct | Strong | Well-defined subspace claim; falsifiable predictions about dimensionality and sharing; convergent with independent observations of residual stream structure |
| Measurement | Partial | Subspace identification is reproducible; baseline separation confirmed against logit lens and tuned lens |
| Internal | Strong | Necessity shown via projection ablation; sufficiency demonstrated by information routing through the subspace; specificity across tasks |
| External | Partial | Cross-layer generalization demonstrated; cross-model evidence preliminary |
| Interpretive | Partial | "Global workspace" label imports cognitive science connotations; the evidence supports "shared low-dimensional communication subspace" without requiring the full cognitive theory |

The claim reaches Mechanistically Supported because both necessity and sufficiency are demonstrated with convergent methods, and the subspace structure shows specificity across tasks. It does not reach Triangulated because double dissociation (I6) is inconclusive — the single dissociation is run cleanly and repeatedly, but no converse arm exists showing a control subspace of matched dimension that spares J-space-dependent tasks while impairing a different one. The "global workspace" label carries implications beyond what the evidence establishes, but the underlying structural claim is well-supported.

## Evidence summary

**Structural identification.** The J-space subspace is recovered as the expected Jacobian $J_\ell$ of the final-layer residual with respect to layer $\ell$, derived from the architecture rather than fitted to activations.

**Necessity (I1).** Projecting out the J-space subspace from the residual stream degrades model performance substantially, while projecting out random subspaces of equal dimension produces smaller effects.

**Sufficiency (I2).** Information routed through the J-space subspace is sufficient to support downstream computations. Restricting communication to this subspace preserves model behavior to a greater degree than restricting to random subspaces.

**Baseline separation (M2).** The identified subspace is compared against logit lens and tuned lens, the densest negative controls in the audited set. Confirmed.

## Validity concerns

**Unlicensed labeling (V4).** "Global workspace" in cognitive neuroscience entails conscious access, broadcast, and competition among specialized processors. The evidence in transformers supports a shared communication subspace but does not establish the richer cognitive-science interpretation. "Shared communication subspace" would be a more conservative label.

**Cross-model generalization (E4).** The evidence for J-space structure across different model families is preliminary. Whether the same structure appears in architectures trained on different data or with different objectives remains an open question.

**Confound control (I7).** The subspace could reflect training data statistics rather than a functional architectural property. Distinguishing "the model routes information through this subspace because it is computationally useful" from "this subspace captures the principal variance of the training distribution" requires additional controls.

**Double dissociation (I6).** No crossed design has been published: a demonstration that a second, dimension-matched subspace is necessary and sufficient for a different task while leaving J-space-dependent tasks unaffected (and vice versa) would establish that the two subspaces are functionally distinct communication channels rather than two views of the same variance.
