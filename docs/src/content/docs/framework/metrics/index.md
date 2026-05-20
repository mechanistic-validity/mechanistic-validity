---
title: "Metrics"
description: "The 58 concrete, runnable tests that produce evidence about mechanistic claims — organized into six families."
---

# Metrics

A metric is a concrete, runnable test that produces a measurement about a neural network. Metrics are Layer A of the taxonomy — the bottom of the hierarchy, where empirical contact with the model actually happens. Everything above (evidence families, criteria, validity types, verdicts) depends on what metrics measure and how well they measure it.

The 58 metrics are organized into six families, each producing a distinct kind of signal.

## A. Causal (13 metrics)

Metrics that intervene on the model's computation — ablation, patching, causal scrubbing — and measure the downstream effect. Causal metrics answer: **does this component matter for this behavior?**

The family spans the full Pearl hierarchy: observational correlation (A10 Regularity/INUS), interventional effects (A01 SCM, A02 Counterfactual DAS, A03 Rubin CATE), and counterfactual reasoning (A11 Actual Cause). Information-theoretic causal measures (A07 Granger/TE, A08 PID) and structural discovery algorithms (A09 MDL/SLT, A13 Causal Discovery) provide complementary perspectives.

## B. Structural (9 metrics)

Metrics that analyze the model's weight matrices directly — without running any input through the model. Structural metrics answer: **what does the architecture encode before any data flows?**

Spectral decomposition (B01 SVD), rank analysis (B02 Effective Rank), circuit-level decomposition (B03 OV/QK), alignment measures (B04 Weight Alignment), and norm trajectories (B05) characterize the geometry of weight space. Template matching (B06), polysemanticity indices (B07), and blind source separation (B08 ICA/NMF) extract interpretable structure from learned parameters.

## C. Information-theoretic (9 metrics)

Metrics that quantify information flow through the network using entropy, mutual information, and related quantities. Information metrics answer: **how much does this component know about the task variable, and where did that knowledge come from?**

Classical measures (C01 Mutual Information, C02 Conditional MI, C03 Transfer Entropy) are complemented by decomposition frameworks (C04 PID, C06 O-Information) that separate redundant, unique, and synergistic contributions. Bottleneck methods (C05) and causal discovery algorithms (C07 Granger, C09 NOTEARS) connect information flow to causal structure.

## D. Behavioral (9 metrics)

Metrics that measure the model's input-output behavior under controlled conditions — ablation recovery, distribution matching, and generalization testing. Behavioral metrics answer: **does the proposed circuit actually produce the behavior it is supposed to explain?**

Faithfulness (D01), logit diff recovery (D02), and KL divergence (D03) measure how well a circuit reproduces the full model's outputs. Cross-task (D06) and cross-scale (D07) transfer test whether circuits generalize beyond their discovery setting. Prompt paraphrase (D08) and generalization gap (D09) probe robustness to surface variation.

## E. Representational (10 metrics)

Metrics that characterize what information is encoded in the model's internal representations and how it is organized geometrically. Representational metrics answer: **what does this component represent, and how is that representation structured?**

Distributed alignment search (E01 DAS-IIA) and linear probing (E02) test for specific features. Similarity measures (E03 RSA, E04 CKA, E05 Subspace Alignment) compare representational geometry across layers, models, or conditions. Dimensionality measures (E06 PCA, E07 Intrinsic Dimension, E08 Participation Ratio) and topological methods (E09 Persistent Homology) characterize the shape of representation space.

## F. Measurement-Theoretic (8 metrics)

Metrics that evaluate the measurement properties of other metrics — reliability, validity, and invariance. Measurement-theoretic metrics answer: **can we trust the measurements that the other metrics produce?**

Bootstrap stability (F01) and seed variance (F02) test whether results are reproducible. Convergent (F03) and discriminant (F04) validity test whether metrics that should agree do agree, and metrics that should disagree do disagree. Internal consistency (F05), inter-rater agreement (F06), measurement invariance (F07), and incremental validity (F08) round out the measurement-theoretic toolkit.

## How metrics connect to the rest of the framework

Each metric produces **evidence** (Layer B) that is evaluated against **criteria** (Layer C). The criteria are grouped by **validity type** (Layer D), and the aggregate assessment across validity types produces a **verdict** (Layer E) tagged with a **description mode**.

A metric alone cannot establish a claim. A claim requires evidence from multiple metrics, evaluated against the criteria appropriate to the validity type being asserted. The dependency order is strict: no skipping from Layer A to Layer E.
