---
title: "F06 — Inter-Rater Reliability"
description: "Measures agreement between independent circuit-discovery methods treated as raters."
---

# F06 — Inter-Rater Reliability

This framework asks: **When two independent methods identify circuit boundaries, how often do they agree on which edges belong?**

In measurement theory, inter-rater reliability quantifies whether different raters assign the same scores to the same subjects. In circuit discovery, the "raters" are different algorithms — weight-based identification, EAP (Edge Attribution Patching), activation patching, ACDC — and the "subjects" are model edges. High inter-rater agreement means the circuit boundary is objective, not method-dependent.

This differs from convergent validity (F03) in granularity: F03 correlates continuous importance rankings, while F06 measures agreement on the binary decision "is this edge in the circuit or not?" using set-overlap and chance-corrected agreement coefficients.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Cohen, "A coefficient of agreement for nominal scales"](https://doi.org/10.1177/001316446002000104) | 1960 | Cohen's kappa for two raters |
| [Shrout & Fleiss, "Intraclass correlations: uses in assessing rater reliability"](https://doi.org/10.1037/0033-2909.86.2.420) | 1979 | ICC framework for multiple raters |
| [Jaccard, "The distribution of the flora in the alpine zone"](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x) | 1912 | Jaccard index for set similarity |
| [Conmy et al., "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) | 2023 | ACDC as an independent circuit-discovery rater |
| [Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery"](https://arxiv.org/abs/2310.10348) | 2023 | EAP as alternative rater for circuit edges |

## Core concept

Given two methods that each produce a circuit edge set \( C_A, C_B \subseteq E \), the Jaccard index is:

\[
J(C_A, C_B) = \frac{|C_A \cap C_B|}{|C_A \cup C_B|}
\]

Cohen's kappa for the binary classification (in-circuit vs. not) over all possible edges:

\[
\kappa = \frac{p_o - p_e}{1 - p_e}
\]

where \( p_o = \frac{|C_A \cap C_B| + |\overline{C_A} \cap \overline{C_B}|}{|E|} \) and \( p_e \) accounts for chance agreement given each method's circuit density.

## Instruments under F06

### Edge Jaccard Agreement (`27_edge_jaccard.py`)

Computes the Jaccard index and Cohen's kappa between the weight-circuit edge set and the EAP-derived edge set at matched sparsity levels. Reports agreement at multiple threshold points to show how inter-rater reliability varies with circuit density.

**What it establishes:** That the circuit boundary is reproducible across methods — the identified edges are not artifacts of one algorithm.
**What it does not establish:** That the agreed-upon edges are causally important — agreement could reflect shared biases (e.g., both methods favor high-norm edges).

**Usage:**
```
uv run python 27_edge_jaccard.py --tasks ioi sva --methods weight eap --sparsity 0.1 0.2 0.3
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Kappa > 0.7 | Strong agreement — circuit boundary is method-independent |
| Kappa 0.4–0.7 | Moderate — core edges agree, periphery diverges |
| Kappa < 0.4 | Weak — methods identify substantially different circuits |
| Jaccard decreasing with sparsity | Methods agree on top edges but diverge on marginal ones |

