---
title: "Geometry -- Metrics & Protocols"
description: "Concrete metrics and protocols for the geometry lens: PCA dimensionality, intrinsic dimension, subspace alignment, participation ratio, persistent homology, geodesic distance, TDA factors, Fisher-Rao distance, sheaf consistency, angular steering, symmetry equivariance, and metric space comparison."
---

# Geometry -- Metrics & Protocols

This page documents the concrete metrics and protocols that implement the [geometry lens](/framework/lenses/supporting/geometry). The geometry lens asks whether a circuit's activation manifold has the curvature, dimensionality, topology, transport, symmetry, and metric structure that the claimed computation requires. Each metric below operationalizes one aspect of that question.

The metrics fall into two groups. **Instruments** (E-prefixed) are standalone computations that produce a numeric score from model activations or weights. **Protocols** (WC/C/M-prefixed) compose multiple metrics with calibrations to produce a structured verdict. Instruments are building blocks; protocols are complete evaluations.

---

## Instruments

### E06 -- PCA Dimensionality

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Do circuit layers use fewer effective dimensions of the residual stream than non-circuit layers?

**Method.** Collect last-token residual stream activations at each layer across prompts. Center the data and compute eigenvalues via SVD. Report two quantities:

- **Effective dimensionality**: the number of PCA components needed to explain 90% of variance. Formally, the smallest $k$ such that $\sum_{i=1}^{k} \lambda_i / \sum_{j} \lambda_j \geq 0.9$.
- **Participation ratio**: $\text{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$. PR = 1 means one dimension dominates; PR = $d$ means all dimensions contribute equally.

The key output is the **dimensionality ratio**: mean effective dimensionality at circuit layers divided by mean effective dimensionality at non-circuit layers. A ratio below 1 indicates that the circuit operates in a lower-dimensional subspace -- consistent with structured, constrained computation rather than distributed processing.

**What it establishes.** Circuit layers concentrate their computation in fewer dimensions. This is a necessary (not sufficient) condition for the claim that the circuit implements a specific computation rather than a generic transformation.

**What it does not establish.** That the low-dimensional subspace encodes task-relevant features (that requires subspace alignment or curvature coherence), or that the dimensionality reduction is causally necessary (that requires ablation evidence).

**Metric IDs.**
- `E60.effective_dimensionality_ratio` -- ratio of circuit to non-circuit mean effective dimensionality
- `E60.participation_ratio_circuit` -- mean participation ratio at circuit layers

**Usage.**
```bash
uv run python 60_pca_dimensionality.py --tasks ioi sva --device cpu
uv run python 60_pca_dimensionality.py --device cuda --n-prompts 60
```

---

### E07 -- Intrinsic Dimension (Two-NN)

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Do circuit layers operate on lower-dimensional manifolds than non-circuit layers?

**Method.** Estimates the intrinsic dimension (ID) of residual stream activations at each layer using the Two-NN estimator ([Facco et al. 2017](https://doi.org/10.1038/s41598-017-11873-y)). For each data point, compute $\mu = r_2 / r_1$, the ratio of the second-nearest to first-nearest neighbor distance. The MLE of intrinsic dimension is:

$$\text{ID} = n / \sum_i \log(\mu_i)$$

Unlike PCA dimensionality, Two-NN estimates the dimension of the underlying manifold, not the linear subspace that captures variance. A dataset lying on a curved 3-dimensional manifold embedded in 768-dimensional space will have $\text{ID} \approx 3$ but potentially high PCA dimensionality (because PCA cannot capture curvature).

The key output is the **ID ratio**: mean intrinsic dimension at circuit layers divided by mean intrinsic dimension at non-circuit layers. A ratio below 1 means circuit layers constrain activations to a lower-dimensional manifold.

**What it establishes.** The activation manifold at circuit layers has fewer intrinsic degrees of freedom. The computation is geometrically constrained, not just linearly low-rank.

**What it does not establish.** The identity of the manifold's dimensions (that requires interpretive analysis), or whether the low dimension is specific to the task or an architectural property (compare against random component sets).

**Metric IDs.**
- `E62.intrinsic_dim_ratio` -- ratio of circuit to non-circuit mean intrinsic dimension
- `E62.mean_intrinsic_dim` -- overall mean intrinsic dimension across all layers

**Usage.**
```bash
uv run python 62_intrinsic_dimension.py --tasks ioi sva --device cpu
uv run python 62_intrinsic_dimension.py --device cuda --n-prompts 60
```

---

### E05 -- Subspace Alignment

**Validity layer:** Construct (C2 -- Structural plausibility, C5)

**Question.** Do circuit heads share output subspaces, and are those subspaces aligned with the answer direction?

**Method.** For each circuit head, collect its OV output (projected through $W_O$) across prompts, compute the top-$k$ singular vectors (default $k = 5$) of the centered output matrix, and represent the head's output subspace as a $k$-dimensional subspace of $\mathbb{R}^{d_\text{model}}$. Then compute:

1. **Pairwise subspace alignment** between circuit heads using principal angles. The alignment score is the mean $\cos^2(\theta_i)$ of the principal angles -- 1 means identical subspaces, 0 means orthogonal.
2. **Grassmann distance** between head subspaces: $d_G = \sqrt{\sum_i \theta_i^2}$, the geodesic distance on the Grassmann manifold $\text{Gr}(k, d_\text{model})$.
3. **Answer direction alignment**: the fraction of the correct-minus-incorrect unembedding direction that lies within each head's output subspace (norm of the projection onto the subspace).

**What it establishes.** Circuit heads whose output subspaces are aligned share a common direction in the residual stream -- they may be functionally redundant or may compose via constructive interference. High answer-direction alignment means the head's output directly contributes to the logit difference, supporting a direct logit attribution interpretation.

**What it does not establish.** Whether the shared subspace is used by downstream components (that requires path patching), or whether the alignment is specific to this circuit (compare against random head subsets).

**Metric IDs.**
- `E64.pairwise_subspace_alignment` -- mean pairwise alignment score across circuit head pairs
- `E64.mean_grassmann_distance` -- mean Grassmann distance between circuit head subspaces
- `E64.answer_direction_alignment` -- mean fraction of the answer direction captured by each head's output subspace

**Usage.**
```bash
uv run python 64_subspace_alignment.py --tasks ioi sva --device cpu
uv run python 64_subspace_alignment.py --device cuda --n-prompts 40
```

---

### E08 -- Participation Ratio (Per-Head)

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Do circuit heads project onto focused subspaces with low effective dimensionality?

**Method.** Collect the output of each attention head (in $d_\text{head}$ space) at the last token position across prompts. Compute the covariance matrix and its eigenvalues. The participation ratio is:

$$\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

PR = 1 means a single dimension dominates the head's output variance. PR = $d_\text{head}$ means all dimensions contribute equally. Circuit heads with low PR project onto focused subspaces, suggesting they encode specific features rather than distributed information.

The key output is the **PR ratio**: mean PR of circuit heads divided by mean PR of non-circuit heads. A ratio below 1 means circuit heads are more focused.

**What it establishes.** Circuit heads concentrate their output variance in fewer dimensions than non-circuit heads. This is consistent with the claim that circuit heads encode specific, identifiable features rather than serving as generic information conduits.

**What it does not establish.** What the focused dimensions encode (that requires logit lens or feature analysis), or whether the focus is causally necessary (that requires ablation).

**Metric ID.**
- `E02.participation_ratio` -- mean participation ratio of circuit heads, with non-circuit mean as baseline

**Usage.**
```bash
uv run python 65_participation_ratio.py --tasks ioi sva --device cpu
uv run python 65_participation_ratio.py --device cpu --n-prompts 60
```

---

### E09 -- Persistent Homology (H0)

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Do circuit layers have distinct topological structure in activation space?

**Method.** Build a Vietoris-Rips-inspired H0 persistence summary of the residual stream activation manifold at each layer. Compute the cosine distance matrix between last-token activations across prompts, then track how the number of connected components (0-th Betti number $\beta_0$) decreases as the distance threshold $\varepsilon$ grows -- implemented efficiently via single-linkage clustering, which gives the H0 persistence diagram directly.

Reports three quantities per layer:

- **Diameter**: the $\varepsilon$ at which full connectivity is achieved (the last merge distance in the dendrogram).
- **Mean merge distance**: average distance at which components merge.
- **Normalized Betti curve AUC**: $\int \beta_0(\varepsilon) \, d\varepsilon / (n \cdot \text{diameter})$. This summarizes how quickly the point cloud coalesces. A high AUC means the points remain fragmented until large distances -- the data has "holes" or clusters.

**What it establishes.** Circuit layers have topological features (clusters, connectivity patterns) that differ from non-circuit layers. Distinct topology at circuit layers is consistent with the claim that these layers impose structural constraints on the activation manifold.

**What it does not establish.** What the topological features represent semantically, or whether the topology is causally relevant. Persistent homology is a descriptive characterization of the data's shape, not a causal test.

**Metric ID.**
- `E06.persistent_homology_h0` -- mean diameter at circuit layers, with non-circuit mean as baseline

**Usage.**
```bash
uv run python 67_persistent_homology.py --tasks ioi induction --device cpu
uv run python 67_persistent_homology.py --device cpu --n-prompts 50
```

---

### E07b -- Geodesic Distance (Manifold Curvature)

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Do circuit layers have higher manifold curvature, indicating nonlinear computation?

**Method.** Build a $k$-nearest-neighbor graph (default $k = 10$) on residual stream activations at each layer, with Euclidean edge weights. Compute shortest-path (geodesic) distances via Dijkstra's algorithm. The **distortion ratio** for each pair of points is:

$$\text{distortion}(i, j) = \frac{d_\text{geodesic}(i, j)}{d_\text{Euclidean}(i, j)}$$

A distortion ratio of 1 means the manifold is locally flat (geodesics follow straight lines). A ratio greater than 1 indicates curvature -- the manifold bends, and the shortest path along it is longer than the straight-line distance. Higher distortion at circuit layers suggests nonlinear computation is occurring -- the data lies on a curved manifold rather than a linear subspace.

**What it establishes.** The activation manifold at circuit layers has curvature that is absent (or less pronounced) at non-circuit layers. Curvature is consistent with nonlinear computation: the circuit transforms representations in ways that cannot be captured by linear projections.

**What it does not establish.** The nature of the nonlinearity (which could be attention softmax, MLP ReLU/GELU, or their composition), or whether the curvature is functionally necessary.

**Metric ID.**
- `E08.geodesic_distance` -- mean distortion ratio at circuit layers, with non-circuit mean as baseline

**Usage.**
```bash
uv run python 68_geodesic_distance.py --tasks ioi induction --device cpu
uv run python 68_geodesic_distance.py --device cpu --n-prompts 50
```

---

### E09b -- TDA Factors (SVD of Circuit DLA)

**Validity layer:** Construct (C2 -- Structural plausibility)

**Question.** Is the circuit's behavior controlled by a small number of latent factors, or do circuit heads contribute independently?

**Method.** Compute the direct logit attribution (DLA) of each circuit head on each prompt:

$$\text{DLA}(L, H, i) = h_{L,H}^{(i)} \cdot W_O[L, H] \cdot (W_U[:, \text{correct}_i] - W_U[:, \text{incorrect}_i])$$

This produces a matrix of shape (n_prompts, n_circuit_heads). Centering and applying SVD reveals the latent factor structure. The **effective rank** at 90% variance threshold gives the number of independent modes of variation in the circuit's behavior across prompts.

A low effective rank (relative to the number of circuit heads) means the heads co-vary -- their contributions are controlled by a few shared latent factors rather than acting independently. A **rank ratio** (circuit effective rank / random subset effective rank) below 1 confirms this is a property of the circuit, not an artifact of the DLA computation.

**What it establishes.** The circuit has low-rank behavioral structure: a few latent factors explain most of the variance in how circuit heads contribute to the output. This is consistent with the claim that the circuit implements a coherent computation with shared internal variables.

**What it does not establish.** The identity of the latent factors (that requires interpretive analysis of the singular vectors), or whether the low-rank structure is causal (that requires intervention on the inferred factors).

**Metric ID.**
- `E09.tda_factors` -- effective rank of the circuit DLA matrix, with random subset effective rank as baseline

**Usage.**
```bash
uv run python 69_tda_factors.py --tasks ioi greater_than --device cpu
uv run python 69_tda_factors.py --device cpu --n-prompts 60
```

---

## Protocols

Protocols compose metrics with calibrations to produce structured verdicts. Each protocol runs a set of metric computations, compares results against thresholds, and produces a `ProtocolResult` with pass/fail status and supporting metadata.

### WC_M6 -- Information Geometry (Fisher-Rao Distance)

**Validity type:** Construct

**Question.** Are circuit components distributionally distinct or redundant, as measured by a reparameterization-invariant distance?

**Background.** Weight-space cosine similarity is not reparameterization-invariant: two components with different weights but similar activation distributions can appear dissimilar. The Fisher-Rao geodesic distance on the statistical manifold of component activation distributions is invariant under smooth reparameterizations. For two Gaussians on the Poincare half-plane:

$$d = \text{arccosh}\left(1 + \frac{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2}{2\sigma_1\sigma_2}\right)$$

This formula treats each circuit component's activation distribution as a point on a Riemannian manifold and measures the geodesic distance between points. Components that are close in Fisher-Rao distance produce similar output distributions regardless of their parameterization; components that are far apart are genuinely distributionally distinct.

**Metrics.**
- `cka` -- representational similarity (standard CKA)
- `effect_size` -- component importance via ablation effect
- `attention_clustering` -- functional grouping via attention patterns

**Thresholds.**
| Metric | Threshold |
|---|---|
| CKA | 0.5 |
| Effect size | 0.8 |
| Attention clustering | 0.5 |

**References.**
- Fisher (1925), "Theory of statistical estimation"
- Rao (1945), "Information and accuracy attainable in the estimation of statistical parameters"

**Usage.**
```bash
uv run python fisher_rao.py --tasks ioi induction --device cpu
```

---

### WC_M7 -- Sheaf Consistency Scan

**Validity type:** Measurement

**Question.** Does sheaf-theoretic consistency of the residual stream align with causal importance?

**Background.** A sheaf assigns local vector spaces to nodes (token positions) and linear maps to edges (attention connections). The sheaf Laplacian energy measures representational inconsistency -- how well information flows between positions through each attention head. This protocol cross-references two independent measurements:

- **Causal importance** (via activation patching and DAS IIA): which components matter for the task.
- **Sheaf energy** (via the sheaf Laplacian on the residual stream graph): which components transfer information consistently.

A true circuit component should have both high causal importance (IIA) AND low sheaf energy (consistent information transfer). A component with high IIA but high sheaf energy is causally important but representationally incoherent -- a red flag for the circuit hypothesis. A component with low IIA but low sheaf energy is representationally consistent but causally dispensable -- a background information channel, not a circuit component.

**Metrics.**
- `activation_patching` -- causal importance of each component
- `das_iia` -- interchange intervention accuracy (causal validation)
- `path_patching` -- path-level causal flow for structural consistency

**Thresholds.**
| Metric | Threshold |
|---|---|
| Activation patching | 0.5 |
| DAS IIA | 0.6 |
| Path patching | 0.5 |

**References.**
- Bodnar et al. (NeurIPS 2022), "Neural Sheaf Diffusion"
- Curry (2014), "Sheaves, Cosheaves and Applications"

**Usage.**
```bash
uv run python sheaf_consistency.py --tasks ioi induction --device cpu
```

---

### WC_M2 -- Angular Steering (Rotation Geometry)

**Validity type:** Construct

**Question.** Does angular steering (rotation in the activation-direction plane, preserving norm) reveal qualitatively different circuit properties than additive steering?

**Background.** Standard additive steering ($h' = h + \alpha d$, where $d$ is a steering direction) changes both direction and magnitude. Angular steering uses the Rodrigues rotation formula to rotate the activation vector $h$ toward a target direction $d$ by a specified angle, preserving $\|h\|$. The parameter space is bounded $[-90\degree, +90\degree]$ instead of unbounded $\alpha$, so angular sensitivity characterizes the circuit's directional threshold.

A circuit with a steep angular dose-response curve has a sharp directional threshold: small rotations toward the feature direction trigger a qualitative change in behavior. A shallow curve means the circuit's behavior changes gradually with direction -- there is no clean "on/off" boundary in angular space.

**Metrics.**
- `dose_response` -- angular sensitivity: slope of behavioral change vs. rotation angle (steep = toggle, shallow = gradual)
- `effect_size` -- maximum behavioral range achievable through rotation
- `das_iia` -- alignment of circuit causal structure with rotational intervention

**Thresholds.**
| Metric | Threshold |
|---|---|
| Dose response | 0.5 |
| Effect size | 0.8 |
| DAS IIA | 0.6 |

**Usage.**
```bash
uv run python angular_steering.py --tasks ioi induction --device cpu
```

---

### C13 -- Symmetry Equivariance

**Validity type:** Construct

**Question.** Does the circuit respect the symmetry group implied by the task?

**Background.** For IOI, the symmetry group is $S_2$ (swapping IO and S names). If the circuit truly computes indirect object identification, then swapping the two names in a prompt should produce correspondingly swapped circuit activations. The equivariance error measures how far the circuit deviates from this ideal:

$$\text{error} = \frac{\|W \rho_\text{in}(g) - \rho_\text{out}(g) W\|_F}{\|W\|_F}$$

This protocol has two components: a pure weight-space structural test (do the weight matrices commute with the group action?) and an activation-based functional test (do activations transform correctly under input symmetries?). The weight-space test requires no forward passes and is very fast.

**Metrics.**
- `cka` -- representational similarity under symmetry transformations
- `effect_size` -- component importance preservation under symmetry

**Thresholds.**
| Metric | Threshold |
|---|---|
| CKA | 0.5 |
| Effect size | 0.8 |

**References.**
- Cohen & Welling (2016), "Group Equivariant Convolutional Networks"
- Wang et al. (2022), "Interpretability in the Wild" -- IOI circuit with $S_2$ symmetry

**Usage.**
```bash
uv run python c13_symmetry_equivariance.py --tasks ioi --device cpu
```

---

### M12 -- Metric Space Comparison (Gromov-Hausdorff)

**Validity type:** Measurement

**Question.** How similar is the activation geometry of circuit heads across two model sizes?

**Background.** The Gromov-Hausdorff (GH) distance compares the intrinsic geometry of two metric spaces without requiring them to live in the same ambient space. Each circuit's activation vectors define a metric space (with pairwise Euclidean distances). The GH distance finds the optimal correspondence between points in the two spaces that minimizes the maximum distortion of pairwise distances:

$$d_\text{GH}(X, Y) = \inf_{\phi, \psi} \sup_{x, x'} |d_X(x, x') - d_Y(\phi(x), \phi(x'))|$$

If the normalized GH distance (divided by the diameter of either metric space) is small, the circuit's activation geometry is preserved across model scales -- evidence of a scale-invariant computational structure.

**Metrics.**
- `cka` -- representational similarity (standard CKA)
- `effect_size` -- component importance
- `activation_patching` -- component-level causal importance

**Thresholds.**
| Metric | Threshold |
|---|---|
| CKA | 0.5 |
| Effect size | 0.8 |
| Activation patching | 0.5 |

**References.**
- Gromov (1981), "Groups of polynomial growth and expanding maps"
- Memoli (2007), "On the use of Gromov-Hausdorff distances for shape matching"

**Usage.**
```bash
uv run python m12_metric_space_comparison.py --tasks ioi induction --device cpu
```

---

## How the metrics relate to each other

The geometry instruments form a progression from coarse to fine characterization of the activation manifold:

1. **Dimensionality** (E06 PCA, E07 Two-NN, E08 Participation Ratio): How many degrees of freedom does the manifold have? PCA measures linear dimensionality. Two-NN measures nonlinear (manifold) dimensionality. Participation ratio measures per-head output focus. If all three agree that circuit components are low-dimensional, the evidence is stronger than any single measure.

2. **Structure** (E05 Subspace Alignment, E09b TDA Factors): How are the low-dimensional subspaces organized? Subspace alignment asks whether circuit heads share output directions and whether those directions point toward the answer. TDA factors ask whether circuit heads co-vary as a low-rank system. Together, they characterize whether the circuit's components form a coherent geometric system or act independently.

3. **Topology** (E09 Persistent Homology, E07b Geodesic Distance): What is the shape of the activation manifold? Persistent homology detects clusters and connectivity patterns. Geodesic distance detects curvature. A manifold with distinct clusters (high Betti AUC) and high curvature (high distortion ratio) has richer structure than a flat, homogeneous one.

The protocols then compose these instruments with causal metrics:

- **Fisher-Rao** (WC_M6) adds reparameterization-invariant distance to the picture -- do components that look similar in weight space also look similar distributionally?
- **Sheaf consistency** (WC_M7) cross-references causal importance with representational coherence -- are the causally important components also the ones with consistent information flow?
- **Angular steering** (WC_M2) tests whether the geometric structure has causal consequences -- does rotating activations in the manifold's principal directions change behavior?
- **Symmetry equivariance** (C13) tests algebraic structure -- do the circuit's weights respect the task's symmetry group?
- **Metric space comparison** (M12) tests geometric universality -- is the structure preserved across models?

## Reading the scores

| Pattern | Interpretation |
|---|---|
| Low dimensionality ratio (E06/E07) + high subspace alignment (E05) | Circuit operates in a focused, shared subspace -- structurally coherent |
| Low dimensionality but low subspace alignment | Each head is focused but in different directions -- parallel, independent computations |
| Low TDA effective rank + high answer alignment | Circuit heads co-vary and point toward the answer -- coordinated, functional |
| High geodesic distortion + low intrinsic dimension | Curved but low-dimensional manifold -- nonlinear but structured computation |
| High persistent homology AUC + high geodesic distortion | Rich topological structure with curvature -- complex geometric organization |
| Low PR ratio (E08) + high subspace alignment (E05) | Heads are focused AND share output directions -- potential redundancy or constructive interference |
| Fisher-Rao distance small between circuit heads | Distributionally redundant components (may be candidates for pruning) |
| High sheaf consistency + high IIA | Representationally coherent AND causally important -- strong circuit evidence |
| Low equivariance error (C13) | Circuit structure respects task symmetries -- algebraically valid |
| Low GH distance across models (M12) | Geometric structure preserved across scale -- evidence of universality |

## Relationship to validity criteria

Each metric maps to one or more of the geometry lens criteria defined in the [geometry lens page](/framework/lenses/supporting/geometry):

| Criterion | Instruments | Protocols |
|---|---|---|
| C10 -- Curvature coherence | E07b (geodesic distance) | WC_M6 (Fisher-Rao) |
| M10 -- Parallel transport fidelity | -- | WC_M7 (sheaf consistency) |
| C11 -- Angular separability | E05 (subspace alignment) | WC_M2 (angular steering) |
| M11 -- Wasserstein stability | -- | (see [geometry lens](/framework/lenses/supporting/geometry) protocol step 5) |
| M12 -- Metric space comparison | -- | M12 (Gromov-Hausdorff) |
| C13 -- Symmetry equivariance | -- | C13 (symmetry equivariance) |
| C2 -- Structural plausibility | E06, E07, E08, E09, E09b | -- |

The dimensionality and topology instruments (E06, E07, E08, E09, E07b, E09b) all contribute to criterion C2 (structural plausibility) by characterizing whether the circuit's activation manifold has properties consistent with structured computation. They are supporting evidence that strengthens a circuit claim but do not, on their own, establish causal relevance.
