---
title: "Tier 5: Validated (Within Scope)"
description: "The mechanism is fully characterized within a stated scope — all five validity types pass, with quantitative predictions confirmed."
---

# Verdict Tier 5: Validated (Within Scope)

| | |
|---|---|
| Tier | 5 of 5 (progressive) |
| What it means | Complete mechanistic account within stated scope — every component characterized, quantitative predictions confirmed |
| Minimum evidence | All five validity types pass + component-level function ($I_{\text{fun}}$) + novel quantitative prediction confirmed + scope boundary tested + coverage $\kappa > 0.9$ |
| Upgrade | N/A (highest progressive tier) |
| Downgrade to Triangulated | If completeness fails (uncharacterized components discovered) or a quantitative prediction is refuted |

## What this tier establishes

A Validated claim represents what "fully understood" looks like within mechanistic interpretability. The account is *closed*: every component's function is known, the information flow is demonstrated end-to-end, the account generates quantitative predictions that have been tested, and the scope is explicitly bounded.

The scope restriction is not a weakness — it is honesty about what has been established. Validated is not "true of all language models" or even "true of this model on all inputs." It is "true of this model, on this class of inputs, within this explanatory scope." The boundary should be tested: cases just outside the claimed scope should show the mechanism failing or degrading.

Why so few claims reach this tier: Validated requires *completeness*, not just *correctness*. A circuit can be correctly identified (every component it names is causally involved) without being completely characterized (every component's function is known and the information flow is fully traced). For real-model circuits with dozens of components, achieving completeness remains expensive and technically difficult.

## Example verdict statement

> **Verdict:** Validated (within scope) — `[implementational-algorithmic]`
> **Claim:** The one-layer transformer trained on modular addition implements a discrete Fourier transform algorithm for computing $(a + b) \mod p$.
> **Met:** All five validity types. $I_{\text{fun}}$: each neuron's function characterized as a specific Fourier component. Novel prediction: model confidence should be periodic in $(a + b)$ with period $p$ — confirmed. Scope boundary: mechanism fails on multiplication (outside scope).
> **Open:** Generalization to multi-layer or larger models (outside stated scope).
> **Scope:** One-layer transformer, modular addition, $p = 113$, trained to grokking

## Minimum reporting for this tier

- Complete component-level function table (each component's input-output mapping characterized)
- End-to-end information flow diagram with no gaps
- At least one novel quantitative prediction stated before confirmation
- Scope boundary: specific inputs or conditions where the mechanism demonstrably fails
- Coverage metric $\kappa > 0.9$ on a representative distribution within scope
- All five validity types explicitly assessed and passing

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| → (no higher tier) | The claim can expand in *scope* (same mechanism confirmed in larger models, broader tasks) but this expands the scope declaration rather than changing the tier |
| → Triangulated (downgrade) | An uncharacterized component is discovered within the claimed scope. Or a quantitative prediction fails. Or coverage drops below threshold on a sample within the stated distribution. |

## Characteristic occupants

- **Grokking / modular addition** ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)) — a toy transformer where every weight matrix is explained by the Fourier algorithm, quantitative predictions about periodicity are confirmed, and the scope (one-layer model, single arithmetic task) is explicit
- **Superposition in toy models** ([Elhage et al., 2022](https://arxiv.org/abs/2209.10652)) — validated as a mathematical framework within toy models with known feature statistics and controlled geometry

## Why "within scope"

Validated is not "true of all language models" or "true of this model on all inputs." It is "true of this model, on this class of inputs, within this explanatory scope." The scope restriction is epistemic honesty about what has actually been established. Expanding the scope is possible but constitutes a new claim requiring its own validation.

## Key references

- Nanda et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Elhage et al. (2022). *Toy Models of Superposition.* [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Hill, A. B. (1965). *The Environment and Disease: Association or Causation?* [doi:10.1177/003591576505800503](https://doi.org/10.1177/003591576505800503)
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes.* [doi:10.1017/CBO9780511621123](https://doi.org/10.1017/CBO9780511621123)
