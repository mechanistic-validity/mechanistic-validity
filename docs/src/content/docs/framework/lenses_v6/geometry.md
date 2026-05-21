---
title: "Geometry"
description: "The geometric lens: does the circuit's activation manifold have the curvature, transport, symmetry, and metric structure that the claimed computation requires?"
---

# The Geometry Lens

This lens asks one question: **does the circuit's activation manifold have the geometric structure that the claimed computation requires?**

This lens draws on four mathematical traditions. **Information geometry** (Amari 2016) equips the space of activation distributions with the Fisher-Rao metric — where distance is measured by KL divergence rather than Euclidean norm. Its curvature encodes the model's sensitivity to changes in different directions: a circuit that claims to detect a specific feature should have high Fisher curvature along the direction that distinguishes instances of that feature, and low curvature along irrelevant directions. **Sheaf theory** (Bredon 1997; Curry 2014) formalizes local consistency: each circuit component carries a local representation, and the sheaf consistency condition asks whether representations agree when components share information via the residual stream. **Optimal transport** (Villani 2003) provides a different metric on distributions — the Wasserstein distance — that measures the cost of rearranging one distribution into another, capturing structural similarity that KL divergence can miss. **Representation theory** asks whether symmetries of the input (permutation of names in IOI, number agreement in SVA) are reflected as symmetries in the circuit's weight and activation structure.

Angular geometry captures a fifth property: whether task-relevant subspaces are cleanly separated or tangled in activation space. Two subspaces can be close in Euclidean distance but nearly orthogonal (well-separated), or far apart but nearly parallel (poorly separated). Because the model's linear operations — attention projections, MLP transformations — depend on direction rather than magnitude, angular separation between subspaces is the quantity that determines whether the model can distinguish the features those subspaces encode.

There is a conceptual point worth naming. The choice of metric — Euclidean, Fisher-Rao, Wasserstein — is not neutral. Each defines a different notion of "close" and "far," and therefore a different notion of what the circuit treats as similar or different. Euclidean distance in activation space is the default in most MI work (cosine similarity, L2 norms), but it is not the metric the model uses. The model's output distribution changes according to the Fisher-Rao geometry, not the Euclidean geometry. Two activations that are far apart in L2 can produce nearly identical output distributions (small Fisher-Rao distance), and two that are close in L2 can produce dramatically different outputs (large Fisher-Rao distance). Using the wrong metric leads to wrong conclusions about what the circuit distinguishes. Together — curvature, parallel transport, optimal transport, symmetry, and angular structure — these tools ask: does the circuit's geometry match its claimed function?

## Key Distinctions

### Intrinsic vs extrinsic geometry

Gauss's *Theorema Egregium* (1827) established that some geometric properties are intrinsic -- measurable by an observer living on the manifold, without reference to the ambient space -- while others are extrinsic -- depending on how the manifold is embedded. Curvature is intrinsic: a being living on a sphere can detect its curvature without knowing about three-dimensional space. The angle between two subspaces in the ambient residual stream is extrinsic: it depends on the embedding.

In MI: the Fisher information metric on a circuit's activation manifold is intrinsic -- it measures how the model itself distinguishes nearby inputs, from the model's own perspective. Angular separation between task-relevant subspaces is extrinsic -- it measures how subspaces sit relative to each other in the ambient $d_{\text{model}}$-dimensional residual stream. Both matter for circuit evaluation. Curvature tells you what the circuit treats as different. Angular separation tells you whether the circuit's representations are organized to permit downstream linear readout. A circuit with the right intrinsic geometry (high curvature along task-relevant directions) but poor extrinsic geometry (tangled subspaces) may compute the right thing internally but fail to communicate it downstream.

### Curvature as sensitivity

The Fisher information matrix at a point $\theta$ in parameter (or activation) space is:

$$\mathcal{I}(\theta)_{ij} = \mathbb{E}\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]$$

This is the Hessian of the KL divergence, and it defines a Riemannian metric on the space of distributions. High curvature along a direction means the model's output distribution changes rapidly when activations are perturbed in that direction -- the model is sensitive to that distinction. Low curvature means the model is insensitive -- perturbations in that direction do not change the output.

In MI: if the Fisher metric on a circuit's activation manifold has high curvature along the direction that separates singular from plural subject representations, the circuit distinguishes grammatical number. If the Fisher metric is flat along that direction, the circuit does not care about the distinction -- regardless of what a linear probe might find. A probe can extract information that the model does not functionally use. The Fisher metric measures what the model treats as different in terms of its output behavior, not what can be decoded from its activations by an external classifier. This is a stronger test than probing and a more geometric test than ablation.

### Local consistency vs global coherence (sheaves)

A sheaf on a topological space assigns data to each open set and requires that data on overlapping sets agree on the overlap. Bredon (1997) developed the algebraic formalism; Curry (2014) adapted it for network data analysis, where nodes carry local data and edges carry consistency conditions.

In MI: each circuit component carries a local representation of the information flowing through it. The residual stream is the overlap -- the shared medium through which components communicate. The sheaf consistency score at an edge $(u, v)$ measures whether the representation at component $u$, when projected through the residual stream, agrees with the representation at component $v$. Formally, for components $u$ and $v$ connected via the residual stream, with restriction maps $\rho_u$ and $\rho_v$ from their local representations to the shared subspace:

$$\text{consistency}(u, v) = 1 - \frac{\|\rho_u(x_u) - \rho_v(x_v)\|^2}{\|\rho_u(x_u)\|^2 + \|\rho_v(x_v)\|^2}$$

A circuit with high sheaf consistency has coherent information flow: the meaning of a representation is preserved as it moves through the circuit. A circuit with low consistency has internal contradictions -- what "plural" means at one component differs from what it means at another. High sheaf consistency is necessary for the claim that a circuit implements a unified computation rather than a sequence of unrelated transformations that happen to compose into the right answer.

### KL divergence vs Wasserstein distance

KL divergence and Wasserstein distance are both ways to measure how different two distributions are, but they capture different properties. KL divergence measures the average surprise of using one distribution to code another — it is asymmetric, can be infinite when supports don't overlap, and is sensitive to the tails. Wasserstein distance (earth mover's distance) measures the minimum cost of transporting mass from one distribution to another — it is a true metric, always finite for distributions with finite moments, and respects the geometry of the underlying space.

In MI: two circuits can produce activation distributions with small KL divergence (the log-likelihood ratio is small on average) but large Wasserstein distance (the distributions have the same total probability mass but arranged in different locations in activation space). KL divergence asks "does the circuit produce similar output likelihoods?" Wasserstein distance asks "does the circuit produce similar activation structure?" For circuit stability — does the circuit's geometric structure hold up across prompt samples, tasks, or model checkpoints? — Wasserstein distance is the more informative metric, because it captures structural rearrangement that KL divergence can miss. S06 (Wasserstein Stability) uses this to test whether a circuit's activation geometry is stable across conditions.

### Symmetry as structure (representation theory)

A computation that is equivariant with respect to a symmetry group has the group's structure baked into its weights. If swapping "Alice" and "Bob" in an IOI prompt swaps the model's predictions symmetrically, the IOI circuit's weights should reflect the permutation symmetry of the name positions. Representation theory (Serre 1977) provides the language: a group $G$ acts on the input space, and the circuit's weight matrices should intertwine the input representation with the output representation — $W \circ \rho_{\text{in}}(g) = \rho_{\text{out}}(g) \circ W$ for all $g \in G$.

In MI: most circuit claims implicitly assume symmetries. The IOI circuit should treat name positions symmetrically. The SVA circuit should be equivariant under subject-verb position shifts. The induction circuit should generalize across token identities. Testing whether the circuit's weight matrices actually commute with these symmetry transformations is a structural test that requires no forward passes — it is a pure weight-space check. A circuit whose weights break the expected symmetry either does not implement the claimed computation cleanly or implements it via a mechanism that does not respect the symmetry (which is a weaker, less general implementation). Metric geometry extends this by comparing the metric spaces of different circuits: the Gromov-Hausdorff distance between two circuits' activation geometries measures how similar their structures are, without requiring them to live in the same ambient space — enabling cross-model comparison.

### Angles vs distances

Two directions in the residual stream can be close in Euclidean distance but far in angular distance (nearly orthogonal short vectors) or far in Euclidean distance but close in angular distance (parallel vectors of different magnitude). The distinction matters because the model's linear operations -- $W_Q$, $W_K$, $W_V$, $W_O$, $W_{\text{in}}$, $W_{\text{out}}$ -- act on direction, not magnitude. A projection matrix $W$ applied to two vectors $v_1$ and $v_2$ preserves their angular relationship (up to the distortion introduced by $W$) but not their distance relationship.

In MI: angular separation between subspaces is more meaningful than Euclidean distance for representational structure. If the subspace encoding "subject is singular" and the subspace encoding "subject is plural" are nearly orthogonal, a downstream attention head can separate them with a simple linear projection. If they are nearly parallel, separation requires a nonlinear or high-dimensional operation. The angular structure of a circuit's activation space constrains what computations can be performed downstream by linear projections -- which is what attention heads and MLPs actually do. A circuit whose task-relevant subspaces are angularly tangled requires the model to use more computational resources to extract the distinctions, even if the information is present in principle.

## Analytical Constructs

### The Fisher information ellipsoid

At each point in activation space, the Fisher information matrix $\mathcal{I}(\theta)$ defines an ellipsoid whose principal axes are the eigenvectors of $\mathcal{I}$ and whose axis lengths are the square roots of the eigenvalues. The shape of this ellipsoid is the geometric fingerprint of the circuit's computation at that point.

The ellipsoid reveals structure that no single metric can:

- **Anisotropy ratio** -- the ratio of the largest to smallest eigenvalue of $\mathcal{I}$. A highly anisotropic ellipsoid (ratio $\gg 1$) means the circuit is much more sensitive to some directions than others. This is expected for a circuit that performs a specific computation: it should be sensitive to task-relevant distinctions and insensitive to irrelevant variation.
- **Principal axis alignment** -- do the principal curvature directions (eigenvectors of $\mathcal{I}$) correspond to task-relevant features? If the direction of maximum sensitivity aligns with the direction that separates correct from incorrect completions, the circuit's geometry is consistent with its claimed function.
- **Eccentricity profile** -- how the anisotropy changes across the prompt distribution. A circuit with stable anisotropy (similar ellipsoid shape across prompts) has consistent geometric structure. One whose ellipsoid shape varies wildly is geometrically unstable -- its sensitivity profile depends on the specific input.

To construct the ellipsoid: at each point (prompt, layer, circuit component set), compute the Fisher information matrix by taking the Jacobian of the circuit's output logits with respect to the activations at that point. Extract eigenvalues and eigenvectors. Report the anisotropy ratio, the alignment of principal axes with known task features, and the stability of these quantities across the prompt distribution.

A circuit that detects a specific feature should have an elongated ellipsoid aligned with that feature's direction. A circuit with a spherical ellipsoid (isotropic Fisher metric) treats all directions equally -- it has no geometric preference for the claimed feature, which contradicts any claim of feature-specific computation.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Rao, "Information and the accuracy attainable in the estimation of statistical parameters"](https://doi.org/10.1007/BF02888154) | 1945 | Statistics | **Fisher-Rao metric** -- the Fisher information matrix as a Riemannian metric on the space of probability distributions |
| [do Carmo, *Riemannian Geometry*](https://doi.org/10.1007/978-1-4757-2201-7) | 1992 | Mathematics | **Riemannian curvature** -- intrinsic geometry of manifolds; parallel transport and geodesics |
| [Amari, *Information Geometry and Its Applications*](https://doi.org/10.1007/978-4-431-55978-8) | 2016 | Information Geometry | **Amari's dualistic structure** -- the geometry of statistical manifolds, including the Fisher metric, alpha-connections, and curvature as a measure of model distinguishability |
| [Bredon, *Sheaf Theory*](https://doi.org/10.1007/978-1-4612-0647-7) | 1997 | Mathematics | **Sheaf cohomology** -- algebraic formalism for local-to-global consistency; data on overlapping regions must agree on intersections |
| [Curry, "Sheaves, cosheaves and applications"](https://arxiv.org/abs/1303.3255) | 2014 | Applied Mathematics | **Sheaves on networks** -- adaptation of sheaf theory to data on graphs and networks; consistency conditions as a measure of coherent information flow |
| [Lee, *Introduction to Smooth Manifolds*](https://doi.org/10.1007/978-1-4419-9982-5) | 2012 | Mathematics | **Smooth manifold theory** -- tangent spaces, differential forms, and the coordinate-free framework for analyzing high-dimensional structure |
| [Villani, *Topics in Optimal Transport*](https://doi.org/10.1090/gsm/058) | 2003 | Mathematics | **Optimal transport and Wasserstein distance** -- the cost of rearranging one distribution into another; a true metric that respects the geometry of the underlying space |
| [Serre, *Linear Representations of Finite Groups*](https://doi.org/10.1007/978-1-4684-9458-7) | 1977 | Mathematics | **Representation theory** -- how symmetry groups act on vector spaces; equivariance conditions on linear maps |
| [Gromov, *Metric Structures for Riemannian and Non-Riemannian Spaces*](https://doi.org/10.1007/978-0-8176-4583-0) | 1999 | Mathematics | **Gromov-Hausdorff distance** -- comparison of metric spaces without requiring a common embedding; enables cross-model geometric comparison |

## Validity type: [Construct validity](/framework/validity-types_v4/construct)

> **Fisher-Rao geometry ([Amari 2016](https://doi.org/10.1007/978-4-431-55978-8)):** The Fisher information metric is the unique Riemannian metric (up to scaling) that is invariant under sufficient statistics. It measures the intrinsic distinguishability of nearby distributions. In MI: the Fisher metric on a circuit's activation manifold measures what the circuit treats as different -- not what can be decoded from it, but what changes its output behavior.

This lens contributes primarily to construct validity. The question is not whether the circuit is causally necessary (internal validity) or whether the effect generalizes (external validity), but whether the circuit has the geometric structure that the claimed computation requires. A circuit claimed to detect grammatical number should have high Fisher curvature along the singular-plural direction. A circuit claimed to implement coherent information flow should have high sheaf consistency. A circuit claimed to distinguish between two semantic categories should have angularly separated subspaces for those categories. A circuit claimed to be equivariant under name permutation should have weights that commute with the permutation representation. These are structural predictions derived from the computational claim, testable without intervention.

The lens also contributes three criteria to measurement validity: M10 (parallel transport fidelity) characterizes whether a representation measurement at one circuit component can be meaningfully compared to a measurement at another; M11 (Wasserstein stability) characterizes whether the circuit's geometric structure is stable across conditions; and M12 (metric space comparison) enables cross-model geometric comparison without a shared ambient space.

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| C10 | Curvature coherence | Does the Fisher metric have structure matching the task -- high curvature along task-relevant directions? | Construct |
| M10 | Parallel transport fidelity | Does a representation maintain its meaning as it moves through circuit components? | Measurement |
| C11 | Angular separability | Are task-relevant subspaces angularly separated in activation space? | Construct |
| M11 | Wasserstein stability | Is the circuit's activation geometry stable across prompt samples and conditions? | Measurement |
| M12 | Metric space comparison | Can the circuit's geometric structure be meaningfully compared across models? | Measurement |
| C13 | Symmetry equivariance | Do the circuit's weights respect the symmetries implied by the computational claim? | Construct |

### C10 -- Curvature coherence

The Fisher information metric on the circuit's activation manifold should have structure matching the task: high curvature directions should align with task-relevant distinctions, and the curvature should be anisotropic rather than uniform.

A circuit that claims to detect grammatical number should be highly sensitive to perturbations along the singular-plural direction and insensitive to perturbations along orthogonal directions. The Fisher metric formalizes this: its eigenvectors define the directions of maximum and minimum sensitivity, and its eigenvalues quantify the sensitivity in each direction. Curvature coherence asks whether this sensitivity profile matches the task.

**What it establishes.** The circuit's output is differentially sensitive to input perturbations in task-relevant directions. The geometry of the circuit's activation manifold -- its intrinsic notion of "which inputs are different" -- is aligned with the task structure. This is a structural prediction: if the circuit computes what it claims to compute, it should have this geometric property. Meeting it provides convergent evidence that the circuit's internal geometry is organized around the claimed computation.

**What it does not establish.** That the circuit is causally necessary for the task (that is I1), that the geometric structure generalizes to other models (that is E6), or that the curvature is the mechanism by which the circuit performs the computation. Curvature coherence is correlational evidence at the geometric level -- it shows that the circuit's sensitivity profile is consistent with the claim, not that the sensitivity causes the behavior.

**Threshold.** The top-3 principal curvature directions (eigenvectors of $\mathcal{I}$) should predict task-relevant features with $R^2 \geq 0.5$ in a linear regression. The curvature anisotropy ratio (ratio of the largest to smallest principal curvature, i.e., the condition number of $\mathcal{I}$) should be $\geq 5$, indicating that the circuit treats some directions as substantially more important than others.

**Minimum reporting.**
- Eigenvalue spectrum of the Fisher information matrix, with the top-3 eigenvectors identified
- $R^2$ of linear regression from top-3 eigenvector projections to task-relevant labels, with bootstrap 95% CI
- Anisotropy ratio (max/min eigenvalue)
- Comparison to a size-matched random component set -- if random components show comparable anisotropy, the curvature structure is architectural rather than circuit-specific

### M10 -- Parallel transport fidelity

A representation should maintain its meaning as it moves through circuit components. If a direction in activation space encodes "plural" at one circuit component, and that direction is transported through the residual stream to a downstream component, it should still encode "plural" at the destination. The sheaf consistency score formalizes this: it measures whether local representations at connected components agree when projected onto their shared subspace.

This criterion belongs to measurement validity rather than construct validity because it characterizes whether representational measurements at different circuit locations are commensurable. If parallel transport fidelity is low, a probe finding "this direction encodes X" at layer 5 cannot be meaningfully compared to a probe finding at layer 9 -- the representational coordinate systems are incommensurable, and any comparison is a measurement artifact.

**What it establishes.** The circuit's components share a consistent representational language. Information encoded at one location arrives at another location with its meaning intact. This is a prerequisite for any claim that the circuit implements a unified computation: if each component re-encodes information in an incompatible format, the "circuit" is a sequence of independent transformations rather than a coherent computational mechanism.

**What it does not establish.** That the transported representation is causally used by downstream components (that requires intervention evidence), or that the specific meaning of the representation has been correctly identified (that requires interpretive validity). A circuit can have perfect parallel transport fidelity while the analyst misidentifies what is being transported.

**Threshold.** Mean sheaf consistency $\geq 0.8$ across all edges in the circuit graph. No individual edge drops below $0.5$. These thresholds reflect the requirement that representational coherence should be the norm across the circuit, not an occasional property of a few edges.

**Minimum reporting.**
- Mean sheaf consistency across circuit edges, with standard deviation
- Distribution of per-edge consistency scores, with the minimum edge identified
- Comparison to a baseline where the circuit components are connected in random order (shuffled circuit graph) -- if the shuffled baseline shows comparable consistency, the coherence is a property of the residual stream in general, not of this circuit's information flow
- Identification of any edges with consistency below $0.5$, with interpretation of why those edges break representational coherence

### C11 -- Angular separability

Task-relevant subspaces should be angularly separated in activation space. If a circuit is claimed to distinguish two categories -- singular vs. plural, correct name vs. incorrect name, true vs. false -- the subspaces encoding those categories should be separated by a substantial angle. Angular separation determines whether downstream linear operations (attention projections, MLP transformations) can extract the distinction without requiring nonlinear computation or high-dimensional projection.

**What it establishes.** The circuit's representations are organized so that task-relevant distinctions are geometrically accessible to downstream linear readout. This is a necessary condition for any claim that the circuit enables a specific computation via the model's standard linear-algebraic operations. If the subspaces are nearly parallel, the distinction exists only in magnitude (not direction), and separating them requires the downstream component to threshold on magnitude -- a fragile operation that is sensitive to activation scale.

**What it does not establish.** That the model actually uses the angular separation (it may use a different mechanism entirely), or that the separation is unique to this circuit (other component sets may show comparable separation). Angular separability is a geometric affordance -- it shows that the circuit's geometry permits the claimed computation, not that the computation occurs.

**Threshold.** Mean pairwise angle between task-relevant subspaces $\geq 30\degree$. The angular separation between subspaces should be $\geq 3\times$ the angular spread within each subspace (where angular spread is the standard deviation of directions within a single category's subspace). The $3\times$ ratio ensures that between-category separation exceeds within-category variability -- the geometric analog of a signal-to-noise ratio.

**Minimum reporting.**
- Mean and standard deviation of pairwise angles between task-relevant subspaces
- Angular spread within each subspace, with the $3\times$ ratio computed
- Comparison to a random-subspace baseline: sample subspaces of the same dimension from a uniform distribution on the Grassmannian and report their expected pairwise angles. In high dimensions, random subspaces tend to be nearly orthogonal, so the baseline is informative only if task-relevant subspaces are expected to be closer together than random
- Visualization of the principal angles (canonical angles) between subspaces if the subspaces are low-dimensional

### M11 -- Wasserstein stability

The circuit's activation geometry should be stable across prompt samples, tasks, and experimental conditions. Wasserstein distance (optimal transport cost) between the circuit's activation distributions under different conditions measures whether the geometric structure is a robust property of the circuit or an artifact of the specific prompt sample.

This criterion uses Wasserstein distance rather than KL divergence because Wasserstein is a true metric (symmetric, satisfies triangle inequality) and respects the geometry of activation space — it measures the cost of physically rearranging one distribution into another, capturing structural differences that KL divergence can miss (e.g., two distributions with the same entropy but different spatial arrangement).

**What it establishes.** The circuit's geometric properties — curvature, subspace angles, activation distributions — are reproducible across conditions. This is a measurement reliability criterion: if the geometry changes dramatically when you resample prompts or switch to a related task, the geometric properties measured by C10 and C11 may be artifacts of the sample rather than properties of the circuit.

**What it does not establish.** That the geometry is correct (that is C10), meaningful (that is C11), or causally relevant (that is I1). A circuit with perfect Wasserstein stability can have consistently wrong geometry.

**Threshold.** Wasserstein-1 distance between activation distributions from two independent prompt samples of size $n \geq 200$ should be $\leq 0.2 \times$ the Wasserstein distance between the circuit's activations and a size-matched random component set. This ensures that within-circuit variability is small relative to the circuit's distinctiveness from background.

**Minimum reporting.**
- Wasserstein-1 distance between bootstrap-resampled prompt sets (at least 50 resamples), with mean and 95% CI
- Wasserstein-1 distance between the circuit's activations and a random component set of equal size (the reference scale)
- Ratio of within-circuit to circuit-vs-random Wasserstein distance
- If cross-task stability is claimed, Wasserstein distance between activation distributions on different tasks

### M12 -- Metric space comparison

The circuit's geometric structure should be comparable across models without requiring a shared ambient space. Gromov-Hausdorff distance measures the similarity between two metric spaces (circuits in different models) by finding the best alignment between them — the smallest distortion needed to embed both spaces in a common metric space.

This criterion enables cross-model geometric comparison: does the IOI circuit in GPT-2 Small have the same geometric structure as the IOI circuit in GPT-2 Medium? Standard comparisons (cosine similarity, CKA) require activations to live in the same vector space or to have the same dimensionality. Gromov-Hausdorff distance requires only that each circuit's activations define a metric space — pairwise distances between activation vectors — and compares the distance structures directly.

**What it establishes.** Two circuits in different models have similar internal geometric structure — the pairwise distance relationships between their activations are preserved. This is evidence that the circuits implement similar computations at the geometric level, even if their ambient dimensions, weight magnitudes, and coordinate systems differ entirely.

**What it does not establish.** That the circuits implement the same computation functionally (that requires behavioral comparison), or that the geometric similarity is causal (that requires cross-model intervention). Geometric similarity is necessary but not sufficient for functional equivalence.

**Threshold.** Gromov-Hausdorff distance between the circuit's activation metric spaces in two models should be $\leq 0.3 \times$ the diameter of either metric space. The diameter is the maximum pairwise distance within each circuit's activations.

**Minimum reporting.**
- Gromov-Hausdorff distance between circuit activation metric spaces in two or more models
- Diameter of each metric space (for normalization)
- The optimal correspondence (which activations in model A map to which activations in model B)
- Comparison to the Gromov-Hausdorff distance between the circuits and random component sets of equal size in each model

### C13 -- Symmetry equivariance

The circuit's weight matrices should respect the symmetries implied by the computational claim. If the claimed computation is equivariant under a group $G$ — e.g., permutation of name positions in IOI, shift of subject-verb positions in SVA — then the circuit's weight matrices should commute with the group's representations: $W \circ \rho_{\text{in}}(g) = \rho_{\text{out}}(g) \circ W$ for all $g \in G$.

This is a pure weight-space test: no forward passes required. It asks whether the circuit's structure has the symmetry that its claimed computation demands. A circuit that implements IOI should treat the two name positions symmetrically in its weights. A circuit that detects grammatical number should have weights that are invariant under content-preserving transformations that preserve number. If the weights break the expected symmetry, the circuit either does not implement the claimed computation cleanly or implements it via a mechanism that is less general than the claim implies.

**What it establishes.** The circuit's weight structure has the algebraic properties consistent with the claimed computation. The symmetries of the input-output map are reflected in the circuit's internal structure. This is a structural plausibility test (strengthening C2) that is purely algebraic — it does not depend on the distribution of inputs, the choice of metric, or the activation dynamics.

**What it does not establish.** That the circuit uses the symmetry (the weights may commute with the group action accidentally), or that the symmetry is exact rather than approximate (real circuits may break symmetry slightly due to training dynamics). Approximate equivariance — weights that nearly commute with the group action — is the realistic expectation.

**Threshold.** Equivariance error $\|W \rho_{\text{in}}(g) - \rho_{\text{out}}(g) W\|_F / \|W\|_F < 0.2$ for all generators $g$ of the symmetry group $G$. The error is normalized by the weight norm to make it scale-invariant.

**Minimum reporting.**
- The symmetry group $G$ and its generators, with justification from the computational claim
- Equivariance error for each generator, with the Frobenius norm formula
- Comparison to the equivariance error of a random weight matrix of the same shape (the null distribution)
- If the symmetry is approximate, report the degree of symmetry breaking and whether it is consistent across layers

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| High curvature coherence + angular separability (C10 + C11) | Geometrically structured computation | "The circuit's activation geometry has curvature aligned with [feature] and angularly separated [category] subspaces" |
| Curvature coherence without angular separability (C10 only) | Sensitivity without clean readout | "The circuit is sensitive to [feature] but the subspaces are not angularly separated for downstream linear readout" |
| High parallel transport fidelity + causal evidence (M10 + I1) | Coherent and causally relevant information flow | "The circuit maintains representational coherence across components and is causally necessary for [behavior]" |
| Wasserstein stability + curvature coherence (M11 + C10) | Robust geometric structure | "The circuit's curvature structure is stable across prompt samples (W₁ ratio < 0.2)" |
| Symmetry equivariance + angular separability (C13 + C11) | Algebraically and geometrically structured computation | "The circuit's weights commute with [group] and its subspaces are angularly separated" |
| Cross-model metric comparison (M12) | Geometric universality | "The circuit's geometric structure is preserved across models (GH distance < 0.3× diameter)" |
| Isotropic Fisher metric, low sheaf consistency | No geometric structure | "No evidence of task-aligned geometry; the circuit's activation manifold lacks both curvature structure and representational coherence" |
| All six criteria met | Full geometric validity | "The circuit has curvature-aligned, stable, coherent, angularly separated, symmetric, and cross-model-comparable geometric structure" |

## Verdicts

- **Proposed to Causally suggestive:** The geometry lens does not gate the first transition. Geometric structure is convergent evidence, not a substitute for causal evidence from the neuroscience lens. However, an isotropic Fisher metric (C10 failed) or broken symmetry (C13 failed) is a warning: the circuit's structure does not match the claim.
- **Causally suggestive to Mechanistically supported:** C10 (curvature coherence) and C13 (symmetry equivariance) strengthen the case that the circuit is not just causally relevant but structurally organized around the claimed computation. M10 (parallel transport fidelity) and M11 (Wasserstein stability) strengthen the case that representational measurements are commensurable and reproducible.
- **Mechanistically supported to Triangulated:** C11 (angular separability) provides convergent geometric evidence from a different evidence family. M12 (metric space comparison) contributes to cross-model generalization (E6). A circuit that passes causal criteria (I1-I5), curvature coherence (C10), parallel transport fidelity (M10), angular separability (C11), Wasserstein stability (M11), and symmetry equivariance (C13) has been triangulated across methods that share no methodological assumptions.
- **Triangulated to Validated:** M12 (metric space comparison) provides cross-model geometric evidence that complements behavioral cross-model tests (E6). A circuit with preserved geometric structure across model architectures has stronger evidence for universality.

## Protocol

For a proposed circuit $C$ and behavior $B$:

1. **Fisher information computation.** For each prompt in the evaluation set (at least 200 prompts), compute the Fisher information matrix $\mathcal{I}$ at the circuit's activation manifold by taking the Jacobian of the output log-probabilities with respect to the circuit's activations. Extract eigenvalues and eigenvectors. Report the anisotropy ratio and the eigenvalue spectrum. Compare against a size-matched random component set.

2. **Curvature-feature alignment.** Project the top-3 eigenvectors of $\mathcal{I}$ onto the task-relevant feature space (e.g., the direction separating singular from plural, or correct from incorrect completions). Fit a linear regression from these projections to the task labels. Report $R^2$ with bootstrap confidence intervals.

3. **Sheaf consistency.** Define the circuit graph: nodes are circuit components (heads, MLP layers), edges connect components that communicate via the residual stream. For each edge, compute the restriction maps from the local representations to the shared residual-stream subspace. Compute the consistency score at each edge. Report the mean, standard deviation, and minimum. Compare to a shuffled-graph baseline.

4. **Angular separation.** Identify the task-relevant subspaces (e.g., by collecting activations for each category and computing the principal subspace of each category's activations). Compute pairwise angles between subspaces using canonical angles (principal angles). Report the mean pairwise angle and the angular spread within each subspace. Compute the separation-to-spread ratio.

5. **Wasserstein stability.** Split the prompt set into two independent halves. Compute the Wasserstein-1 distance between the circuit's activation distributions on each half. Repeat with bootstrap resampling (at least 50 resamples). Report the mean W₁ distance and compare to the circuit-vs-random-components W₁ distance.

6. **Symmetry equivariance.** Identify the symmetry group $G$ implied by the computational claim (e.g., $S_2$ name permutation for IOI). For each generator $g$ of $G$, construct the input and output representations $\rho_{\text{in}}(g)$ and $\rho_{\text{out}}(g)$. Compute the equivariance error $\|W\rho_{\text{in}}(g) - \rho_{\text{out}}(g)W\|_F / \|W\|_F$ for each weight matrix $W$ in the circuit. Compare to random matrices.

7. **Cross-model comparison (optional).** If the same circuit has been identified in a second model, compute the Gromov-Hausdorff distance between the two circuits' activation metric spaces. Report the distance normalized by the diameter of each space.

8. **Integration.** Assess whether the geometric evidence is consistent across all criteria. Identify discrepancies and interpret what they mean for the computational claim.

9. A skipped step must be named in the verdict.

## Case Studies

For full worked examples applying all lenses (including differential geometry) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) -- the IOI circuit's Fisher curvature can be analyzed via WC_M6 (Fisher-Rao), its parallel transport via WC_M7 (Sheaf Consistency), and its angular structure via WC_M2 (Angular Steering)
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) -- sheaf consistency is expected to be high for the two-layer composition (previous-token head to induction head)
