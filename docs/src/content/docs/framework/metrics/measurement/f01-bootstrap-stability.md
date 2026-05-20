---
title: "F01 — Bootstrap Stability"
description: "Measures the stability of faithfulness estimates under resampling of the evaluation corpus."
---

# F01 — Bootstrap Stability

This framework asks: **How much does our faithfulness estimate change if we resample the evaluation prompts?**

A circuit's faithfulness score is computed over a finite set of prompts. If that score swings wildly when we draw a different sample from the same distribution, the measurement is unreliable regardless of its magnitude. Bootstrap stability quantifies this sampling uncertainty by repeatedly resampling the evaluation set and recomputing the metric each time.

High bootstrap stability means the circuit evaluation is precise: the same conclusion would be reached with any comparably-sized sample. Low stability warns that the reported score is an artifact of the particular prompts chosen, not a property of the circuit itself.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Efron, "Bootstrap methods: another look at the jackknife"](https://doi.org/10.1214/aos/1176344552) | 1979 | Introduced the bootstrap for nonparametric confidence intervals |
| [Efron & Tibshirani, "An Introduction to the Bootstrap"](https://doi.org/10.1007/978-1-4899-4541-9) | 1993 | Comprehensive treatment of BCa and percentile intervals |
| [DiCiccio & Efron, "Bootstrap confidence intervals"](https://doi.org/10.1214/ss/1032280214) | 1996 | Bias-corrected accelerated intervals for small samples |
| [Conerly & Mansfield, "Approximate confidence limits for circuit metrics"](https://doi.org/10.1080/00031305.1999.10474445) | 1999 | Application of bootstrap to model evaluation metrics |

## Core concept

Given a set of \( N \) evaluation prompts and a faithfulness metric \( \hat{\theta} \), we draw \( B \) bootstrap samples of size \( N \) with replacement. For each resample \( b \), we compute:

\[
\hat{\theta}^{(b)} = \text{faithfulness}(\text{circuit}, \text{prompts}^{(b)})
\]

The bootstrap standard error is:

\[
\text{SE}_{\text{boot}} = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(\hat{\theta}^{(b)} - \bar{\theta}\right)^2}
\]

A 95% confidence interval is constructed from the 2.5th and 97.5th percentiles of the bootstrap distribution. A stability ratio \( \text{SE}_{\text{boot}} / |\bar{\theta}| \) below 0.05 indicates a highly stable estimate.

## Metrics under F01

### Bootstrap Resampling (`11_bootstrap.py`)

Draws \( B = 1000 \) bootstrap resamples from the evaluation corpus and recomputes the faithfulness metric for the target circuit on each resample. Reports percentile-based confidence intervals and the coefficient of variation.

**What it establishes:** Precision of the faithfulness estimate; whether the reported score is reproducible under resampling.
**What it does not establish:** Whether the metric itself is valid or whether the circuit is "good" — only that the measurement is stable.

**Usage:**
```
uv run python 11_bootstrap.py --tasks ioi sva --n-bootstrap 1000
```

## Reading the scores

| Pattern | What it means |
|---|---|
| SE/mean < 0.03 | Highly stable — estimate is tight |
| SE/mean 0.03–0.10 | Acceptable — report with confidence interval |
| SE/mean 0.10–0.20 | Unstable — increase sample size before drawing conclusions |
| SE/mean > 0.20 | Unreliable — measurement is dominated by sampling noise |

## Connection to other frameworks


F01 pairs naturally with F02 (Seed Variance): where F01 resamples prompts, F02 varies the random seed controlling prompt subset selection. Together they bound the total measurement uncertainty from data sampling.
