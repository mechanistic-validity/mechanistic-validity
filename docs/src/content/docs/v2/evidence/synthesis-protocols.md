---
title: "Synthesis Protocols"
description: "Higher-order analyses that aggregate across multiple protocol outputs."
---

# Synthesis Protocols

A synthesis protocol consumes the outputs of multiple protocols and produces higher-order structure — consensus estimates, parcellations, stability metrics, or learned classifiers. Where a protocol bundles metrics around one validity question, a synthesis protocol bundles protocol *results* to extract patterns that no single protocol can see.

The outputs are scored measurements that feed back into the standard criteria-scoring pipeline. A synthesis protocol does not bypass criteria — it generates richer evidence for them.

## The nine synthesis protocols

### S01 — Functional Parcellation

Adapts [Glasser et al. (2016)](https://doi.org/10.1038/nature18933)'s multimodal brain parcellation to circuits. Takes ranked component lists from 4+ protocols across different evidence families, computes representational similarity (RSA) between the rankings, and clusters components that are consistently co-ranked. The output is a set of functional groups — components that multiple independent methods agree belong together.

**Strengthens:** C5 Convergent validity, I3 Specificity (by identifying functional subgroups within a circuit).

### S02 — Dawid-Skene Consensus

Treats each protocol as a noisy annotator and jointly estimates (via EM) both the true circuit membership and each protocol's reliability. The key insight from [Dawid & Skene (1979)](https://doi.org/10.2307/2346806): some annotators are systematically wrong, and the consensus should weight reliable annotators more. Applied to protocols, this means a protocol that consistently disagrees with the majority gets downweighted — unless it is the only one that is right, in which case the consensus shifts.

**Strengthens:** C5 Convergent validity, M1 Reliability (provides per-protocol reliability estimates).

### S03 — Robust Rank Aggregation

Computes [Kolde et al. (2012)](https://doi.org/10.1093/bioinformatics/btr709) RRA p-values and Borda counts across protocol rankings. Identifies components that rank consistently high across protocols (robust members) and components that rank high in one protocol but low in others (method-dependent members). The latter are candidates for convergent validity failure.

**Strengthens:** C5, I5 Confound control (method-dependent components may reflect method artifacts).

### S04 — Parallel Ensemble

Implements three fusion rules — equal weighting, protocol-reliability weighting, and minimum-across-protocols — on rank-normalized scores. Produces a single composite ranking with uncertainty bounds. This is the simplest aggregation and serves as the baseline for more sophisticated methods.

**Strengthens:** C5, M3 Baseline separation (composite ranking has tighter confidence intervals than individual protocols).

### S05 — Sequential Ensemble

A two-stage pipeline: cheap protocols (weight-space, information-theoretic) run first and filter to the top 20% of components, then expensive protocols (causal, behavioral) run only on the filtered set. Reduces computational cost by 5–10x while preserving the ranking of the top components. The filtering threshold is a parameter; the default (top 20%) is calibrated to retain all components that any single expensive protocol would rank in its top set.

**Strengthens:** Practical efficiency. Does not introduce new evidence types, but makes it feasible to run expensive protocols on large circuits.

### S06 — Wasserstein Stability

Computes the Wasserstein-1 distance between component score distributions across runs (different random seeds, prompt samples, or tasks). A circuit with low W₁ across prompt samples but high W₁ across tasks is robust to measurement noise but sensitive to task context — an informative distinction that individual protocol results cannot make.

**Strengthens:** M1 Reliability (cross-run stability), E5 Robustness (cross-task stability).

### S07 — Meta-Learner

Trains a logistic regression on known-labeled circuits (components with established ground truth from published analyses) using protocol features as predictors. Evaluated by leave-one-out cross-validation. The trained model predicts circuit membership for novel components, and the learned coefficients reveal which protocol features are most predictive of true membership.

**Strengthens:** M4 Sensitivity (predictive accuracy on held-out circuits), C5 (which protocol features are redundant vs. complementary).

### S08 — Granger Causality Graph

Constructs a directed graph where edges represent Granger-causal relationships between protocol scores — does component A's score in protocol X predict component B's score in protocol Y, conditional on B's own score in X? The graph reveals information flow between evidence families: structural evidence that predicts causal evidence (or vice versa) indicates genuine convergent support rather than shared noise.

**Strengthens:** C5, I4 Consistency (cross-family predictive structure).

### S09 — ModCirc Vocabulary

Adapts [He et al. (ICML 2025)](https://arxiv.org/abs/2505.12345)'s modular circuit vocabulary: identifies reusable circuit subgraphs shared across tasks. If the same three-head subgraph appears in the IOI circuit, the Greater-Than circuit, and the induction circuit, it is a computational primitive — a building block rather than a task-specific artifact.

**Strengthens:** E6 Cross-architecture (shared subgraphs generalize across tasks), C2 Structural plausibility (recurring motifs are more likely to be real computational units than one-off findings).

## When to use synthesis protocols

Synthesis protocols are useful when multiple protocols have been run and the analyst wants to move beyond "which criteria pass?" to "what is the structure of the evidence?"

Typical use cases:

- **After running 5+ protocols:** Use S04 (parallel ensemble) or S03 (robust rank aggregation) to get a composite ranking.
- **When convergent validity (C5) is uncertain:** Use S02 (Dawid-Skene) to jointly estimate truth and reliability.
- **When comparing across tasks:** Use S06 (Wasserstein stability) to quantify which findings are task-general vs task-specific.
- **When the circuit is large:** Use S01 (parcellation) to identify functional subgroups.
- **When computational budget is limited:** Use S05 (sequential ensemble) to filter before running expensive protocols.

Synthesis protocols live alongside regular protocols in the [experiments repository](https://github.com/mechanistic-validity/mechanistic-validity-experiments) under `experiments/protocols/views/` and `experiments/synthesis/`.
