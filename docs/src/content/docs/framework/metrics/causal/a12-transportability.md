---
title: "A12 — Transportability (Pearl & Bareinboim)"
description: "Transportability theory applied to circuits: formal conditions under which circuit discoveries generalize across models, scales, and distributions."
---

# A12 — Transportability (Pearl & Bareinboim)

This framework asks: **under what formal conditions can a circuit discovered in one model (or scale, or distribution) be validly transferred to another?**

Transportability theory (Pearl & Bareinboim 2011) provides a calculus for determining when causal conclusions derived in one domain (the "source") are valid in a different domain (the "target"). In MI, this addresses the generalization question: if we discover that heads 9.9 and 10.0 form the name-mover circuit in GPT-2 Small, does this conclusion transfer to GPT-2 Medium? To a model trained on different data? To a different task formulation? Transportability theory formalizes exactly which structural assumptions must hold for the transfer to be valid, and identifies when additional target-domain data is needed.

This is not merely an empirical question ("does the same circuit show up?") but a formal one: transportability tells you *which differences between source and target* can be tolerated and which break the transfer. When a circuit fails to transport, the theory identifies the specific causal assumptions that were violated.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Pearl & Bareinboim, "Transportability of Causal and Statistical Relations"](https://doi.org/10.1613/jair.3483) | 2011 | Formal calculus for transporting causal conclusions across domains |
| [Pearl, *Causality*](https://doi.org/10.1017/CBO9780511803161) | 2000/2009 | Structural causal models underlying transportability |
| [Méloux et al., arXiv 2411.10442](https://arxiv.org/abs/2411.10442) | 2025 | Cross-model circuit comparison in language models |
| [Mueller et al., arXiv 2406.14673](https://arxiv.org/abs/2406.14673) | 2024 | MIB benchmark: standardized evaluation enabling cross-model comparison |

## Core concept: selection diagrams and S-nodes

Transportability uses *selection diagrams* — causal graphs augmented with S-nodes that mark variables whose mechanisms differ between source and target domains. A causal conclusion is transportable if the do-calculus can express the target-domain quantity using only source-domain experimental data plus target-domain observational data, accounting for the S-nodes.

In MI terms: if the source model is GPT-2 Small and the target is GPT-2 Medium, the S-nodes mark components whose mechanisms differ (e.g., different head counts, different learned representations). The circuit is transportable if its causal structure does not depend on variables affected by S-nodes. If it does, transportability theory identifies what additional target-domain experiments are needed.

The cross-model invariance test empirically checks transportability by measuring whether the same causal structure (same edges, same IIA scores) holds across models. Formal transportability analysis can then explain *why* certain structures transfer (they don't depend on S-variables) and why others don't.

## Metrics under A12

### C38 — Cross-Model Invariance (`38_cross_model_invariance.py`)

Tests whether a circuit's causal structure is invariant across model variants. Evaluates the same circuit hypothesis (same component roles, same causal edges) in multiple models and measures structural agreement:

\[
\text{Invariance}(M_1, M_2) = 1 - \frac{\text{SHD}(\mathcal{C}_{M_1}, \mathcal{C}_{M_2})}{|\mathcal{C}_{M_1}| + |\mathcal{C}_{M_2}|}
\]

where SHD is structural Hamming distance between the circuits discovered in each model.

**What it establishes:** Whether circuit structure is model-invariant (evidence of transportability) or model-specific.

**What it does not establish:** *Why* the structure transfers or fails to transfer (requires formal transportability analysis with selection diagrams).

**Usage:**
```
uv run python 38_cross_model_invariance.py --tasks ioi sva --n-prompts 40
```

### C41 — Transportability Analysis (`41_transportability.py`)

**Metric status:** This script is a stub awaiting implementation. The directory listing confirms it does not yet exist.

An implementation would require:
1. **Selection diagram construction:** Identify which variables differ between source and target domains (model architecture, training data, scale).
2. **Do-calculus derivation:** Check whether the target-domain causal quantity can be expressed using source-domain experiments + target-domain observations.
3. **Empirical verification:** If transportable, verify the transported conclusion holds empirically. If not, identify the minimal additional target-domain interventions needed.
4. **Output:** Transportability verdict (yes/no/conditional) with the specific S-variables that block or enable transfer.

**Planned usage:**
```
uv run python 41_transportability.py --tasks ioi --source-model gpt2 --target-model gpt2-medium
```

## Reading the scores

| Pattern | What it means |
|---|---|
| High cross-model invariance (> 0.8) | Circuit structure is transportable; generalizes across models |
| Low invariance (< 0.4) | Model-specific circuit; cannot transfer conclusions |
| High invariance on some edges, low on others | Partially transportable; core structure generalizes but periphery is model-specific |
| Transportability fails on scale-dependent S-nodes | Architecture differences (head count, width) break transfer |

## Connection to other frameworks

A12 formalizes the generalization claims tested empirically by A07 (cross-task IIA transfer). Where A07 measures whether causal structure transfers across tasks within a model, A12 asks whether it transfers across models/domains with formal guarantees. A01 (SCM) provides the structural models that transportability operates on. A04 (Woodward) ensures that the interventions used in the source domain satisfy the quality criteria needed for valid transport.
