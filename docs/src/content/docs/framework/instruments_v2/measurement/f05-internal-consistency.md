---
title: "F05 — Internal Consistency"
description: "Assesses whether sub-parts of an evaluation instrument produce coherent scores."
---

# F05 — Internal Consistency

This framework asks: **Do the individual items within our evaluation instrument measure a single coherent construct?**

A circuit faithfulness evaluation typically aggregates scores across multiple prompts, multiple metrics, or multiple ablation targets. Internal consistency asks whether these sub-measurements hang together. If half the prompts suggest the circuit is faithful and the other half suggest it is not, the aggregate score is meaningless — the instrument lacks coherence.

This is the measurement-theoretic analogue of split-half reliability: divide the evaluation items into halves and check whether both halves yield the same conclusion. High internal consistency means the evaluation captures a unitary construct; low consistency means it conflates multiple unrelated signals.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Cronbach, "Coefficient alpha and the internal structure of tests"](https://doi.org/10.1007/BF02310555) | 1951 | Defined Cronbach's alpha as a reliability lower bound |
| [McDonald, "Test Theory: A Unified Treatment"](https://doi.org/10.4324/9781410601087) | 1999 | Omega coefficient as alternative to alpha |
| [Revelle & Zinbarg, "Coefficients alpha, beta, omega, and the glb"](https://doi.org/10.1007/s11336-008-9102-z) | 2009 | Comparison of reliability estimators |
| [Sijtsma, "On the use, misuse, and the very limited usefulness of Cronbach's alpha"](https://doi.org/10.1007/s11336-008-9101-0) | 2009 | Limitations and correct interpretation |

## Core concept

Given \( N \) evaluation items (prompts) each yielding a per-item score \( x_{ij} \) for circuit \( j \), Cronbach's alpha is:

\[
\alpha = \frac{N}{N-1}\left(1 - \frac{\sum_{i=1}^{N} \sigma^2_{x_i}}{\sigma^2_{\text{total}}}\right)
\]

where \( \sigma^2_{x_i} \) is the variance of item \( i \) across circuits and \( \sigma^2_{\text{total}} \) is the variance of the total score. For split-half reliability, we compute faithfulness on two random halves of the prompt set and correlate:

\[
r_{\text{split}} = \text{Pearson}(\theta_{\text{half}_1}, \theta_{\text{half}_2})
\]

applying the Spearman-Brown correction: \( \rho = 2r_{\text{split}} / (1 + r_{\text{split}}) \).

## Instruments under F05

### Split-Half Reliability (`11_bootstrap.py`)

The bootstrap script includes a split-half mode: it partitions the prompt set into two equal halves, computes faithfulness on each, and reports the Spearman-Brown corrected correlation as the reliability coefficient.

**What it establishes:** That the evaluation prompts measure a single coherent construct — faithfulness scores are not driven by a few outlier prompts.
**What it does not establish:** That the construct being measured is the *right* one — only that the instrument is internally coherent.

**Usage:**
```
uv run python 11_bootstrap.py --tasks ioi sva --mode split-half
```

## Reading the scores

| Pattern | What it means |
|---|---|
| Alpha > 0.9 | Excellent internal consistency — items form a tight scale |
| Alpha 0.7–0.9 | Good — acceptable for reporting aggregate scores |
| Alpha 0.5–0.7 | Questionable — some items may not belong to the same construct |
| Alpha < 0.5 | Poor — the aggregate score is not meaningful; inspect item-level results |

