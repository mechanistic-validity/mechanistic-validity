---
title: "F02 — Seed Variance"
description: "Quantifies how much circuit evaluation scores vary across different random prompt subsets."
---

# F02 — Seed Variance

This framework asks: **Does the choice of random seed for prompt selection materially change the evaluation outcome?**

Evaluation pipelines typically sample a fixed-size subset of prompts from a larger corpus. The random seed governing this selection introduces a source of variance that is entirely independent of the circuit's quality. Seed variance measures this effect directly by running the same evaluation across multiple seeds and reporting the spread.

If scores are seed-invariant, the evaluation is robust to the particular subset chosen. If they vary substantially, reported differences between circuits may be artifacts of prompt selection rather than genuine performance gaps.

## Theoretical grounding

| Source | Year | Key contribution |
|---|---|---|
| [Efron, "Bootstrap methods: another look at the jackknife"](https://doi.org/10.1214/aos/1176344552) | 1979 | Foundation for resampling-based variance estimation |
| [Dodge et al., "Show Your Work: Improved Reporting of Experimental Results"](https://doi.org/10.18653/v1/2019.emnlp-main.65) | 2019 | Multi-seed reporting norms for NLP |
| [Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks"](https://arxiv.org/abs/2103.03098) | 2021 | Decomposing variance sources in ML evaluation |
| [Sellam et al., "BLEURT: Learning Robust Metrics for Text Generation"](https://doi.org/10.18653/v1/2020.acl-main.704) | 2020 | Variance-aware metric design |

## Core concept

Let \( s_1, s_2, \ldots, s_K \) be \( K \) random seeds, each producing a prompt subset \( \mathcal{P}_{s_k} \) of size \( n \). The seed variance is:

\[
\text{Var}_{\text{seed}} = \frac{1}{K-1} \sum_{k=1}^{K} \left(\theta_{s_k} - \bar{\theta}\right)^2
\]

where \( \theta_{s_k} \) is the faithfulness score on subset \( \mathcal{P}_{s_k} \). We report the coefficient of variation \( \text{CV} = \sqrt{\text{Var}_{\text{seed}}} / |\bar{\theta}| \) as the normalized instability measure.

A paired comparison between two circuits is seed-robust if their difference \( \Delta_{s_k} = \theta^{A}_{s_k} - \theta^{B}_{s_k} \) has consistent sign across all \( K \) seeds.

## Metrics under F02

### Seed Variance Analysis (`30_seed_variance.py`)

Runs the full evaluation pipeline \( K \) times (default \( K = 20 \)) with different random seeds controlling prompt subset selection. Computes per-seed scores, the seed variance, CV, and sign-consistency of pairwise comparisons.

**What it establishes:** Robustness of evaluation scores to the arbitrary choice of prompt subset.
**What it does not establish:** Whether the metric captures the right property — only that it captures *something* consistently.

**Usage:**
```
uv run python 30_seed_variance.py --tasks ioi sva --n-seeds 20
```

## Reading the scores

| Pattern | What it means |
|---|---|
| CV < 0.02 | Negligible seed effect — single-seed results are trustworthy |
| CV 0.02–0.08 | Moderate — report mean ± std across seeds |
| CV > 0.08 | High seed sensitivity — increase subset size or average over seeds |
| Sign-consistency < 100% | Ranking between circuits flips with seed — difference is not meaningful |

