---
title: "Weights"
description: "Evidence from the model's persistent parameters — what can be read from the network without running it."
---

# Weights

Weight-based evidence comes from the model's persistent parameters: the matrices, biases, and embeddings that define the network's computation before any input is processed. This is the only evidence family whose observational mode requires no forward pass — the model need not run for its weight structure to be analyzed.

## Observational

Methods that read structure from weights without running the model:

- **SVD / spectral analysis** — singular-value decomposition of weight matrices reveals effective rank, principal directions, and low-rank structure. A head whose OV matrix has effective rank 1 implements a rank-1 copy operation regardless of what inputs it sees.
- **OV/QK composition scores** — inner products between the output space of one head and the query/key space of another quantify the *structural capacity* for composition, independent of whether that composition is exercised at runtime.
- **Weight norms and sparsity** — L1/L2 norms, fraction of near-zero entries, and structured sparsity patterns characterize parameter usage.
- **Minimum description length** — compression-based measures of weight complexity.
- **Embedding geometry** — distances, clustering, and subspace structure in embedding and unembedding matrices.

### What it can and cannot show

Observational weight evidence establishes *structural capacity*: this pathway exists, these matrices compose, this subspace is present. It cannot establish that the pathway is *used* at runtime — a high OV composition score between heads 3.1 and 5.4 does not prove that information flows through that connection on any specific input. That requires activation or behavioral evidence.

This asymmetry is precisely why weight evidence is valuable for convergent validity: it provides evidence from a source that is structurally independent of runtime measurements, so agreement between weight structure and activation patterns is informative in a way that agreement between two activation-based methods is not.

## Interventional

Methods that modify weights and observe consequences:

- **Weight knockout** — zeroing or masking specific weight matrices and measuring behavioral change. Analogous to ablation at the activation level but permanent: the modification persists across all inputs.
- **Weight editing** — targeted modification of specific parameters (e.g., ROME, MEMIT) to change stored associations.
- **Circuit transplant** — copying weight submatrices from one model to another and testing whether the associated behavior transfers.
- **Fine-tuning probes** — fine-tuning on a targeted task and measuring which weights change, providing evidence about which parameters encode which capabilities.

### Relationship to activation-level interventions

Weight interventions and activation interventions target different levels of the same system. A weight knockout removes a component permanently across all inputs; an activation ablation removes it for one forward pass. Agreement between the two is evidence for [C3 Convergent validity](/mechanistic-validity/framework/criteria/construct/convergent-validity/) — the component matters regardless of how you remove it. Disagreement is diagnostic: a weight knockout that breaks behavior when the corresponding activation ablation does not suggests runtime compensation that masks the activation-level intervention.

## Relevant criteria

Weight evidence is most directly relevant to:

| Criterion | How weight evidence bears on it |
|---|---|
| [C2 Structural plausibility](/mechanistic-validity/framework/criteria/construct/structural-plausibility/) | Weight structure determines what computations are architecturally possible |
| [C3 Convergent validity](/mechanistic-validity/framework/criteria/construct/convergent-validity/) | Weight-based and activation-based evidence are structurally independent |
| [I1 Necessity](/mechanistic-validity/framework/criteria/internal/necessity/) | Weight knockout establishes necessity at the parameter level |
| [I3 Minimality](/mechanistic-validity/framework/criteria/internal/minimality/) | Weight analysis can identify redundant components |
| [E4 Cross-model recurrence](/mechanistic-validity/framework/criteria/external/cross-model-recurrence/) | Weight-space similarity metrics enable cross-model comparison |
