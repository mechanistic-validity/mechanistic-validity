---
title: "F03 — Convergent Validity"
description: "Tests whether different metrics measuring the same construct produce correlated results."
---

# F03 — Convergent Validity

This framework asks: **Do independent methods that claim to measure the same thing actually agree?**

If weight-based circuit importance and activation-based circuit importance both purport to identify which heads matter for a task, their rankings should correlate. High convergent validity means the construct (circuit importance) is real and measurable — not an artifact of one particular method. Low convergence signals that at least one method is measuring something else entirely.

This is the "same trait, different method" cell of the classic multitrait-multimethod (MTMM) matrix. In circuit discovery, the "trait" is circuit membership and the "methods" are weight analysis, activation patching, edge attribution, and knockout experiments.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Campbell & Fiske, "Convergent and discriminant validation by the MTMM matrix"](https://doi.org/10.1037/h0046016) | 1959 | Defined convergent/discriminant validity framework |
| [Conley et al., "Replication and robustness in developmental research"](https://doi.org/10.1037/dev0000496) | 2018 | Multi-method agreement as evidence for construct validity |
| [Goldstein et al., "Are Neural Network Interpretations Robust?"](https://arxiv.org/abs/2306.10032) | 2023 | Testing agreement between attribution methods |
| [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) | 2022 | IOI circuit validation with multiple methods |

## Core concept

Given two metrics \( A \) and \( B \) that each assign importance scores \( a_i, b_i \) to circuit heads \( i = 1, \ldots, H \), convergent validity is the rank correlation:

\[
r_{\text{conv}} = \text{Spearman}(\mathbf{a}, \mathbf{b})
\]

For binary circuit membership decisions, we use Cohen's kappa:

\[
\kappa = \frac{p_o - p_e}{1 - p_e}
\]

where \( p_o \) is observed agreement and \( p_e \) is chance agreement. Values above 0.6 indicate substantial convergence; above 0.8 indicates near-perfect agreement between methods.

## Metrics under F03

### Convergent Validity (`12_convergent_validity.py`)

Computes pairwise Spearman correlations and Cohen's kappa between all available circuit-identification methods (weight-based ranking, activation patching, edge attribution patching) on the same task. Reports an MTMM-style matrix.

**What it establishes:** That independent methods recover the same circuit structure — the construct is real, not method-specific.
**What it does not establish:** That the shared construct is *faithfulness* specifically — convergence could reflect a shared bias.

**Usage:**
```
uv run python 12_convergent_validity.py --tasks ioi sva --methods weight activation eap
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Spearman > 0.8 | Strong convergence — methods agree on head importance ordering |
| Spearman 0.5–0.8 | Moderate — methods partially agree; investigate divergent heads |
| Spearman < 0.5 | Weak — methods may be measuring different constructs |
| Kappa > 0.6 | Substantial agreement on binary circuit membership |
| Kappa < 0.4 | Poor agreement — circuit boundaries are method-dependent |

