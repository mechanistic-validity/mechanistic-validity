---
title: "MI Structural Metrics"
description: "Structural metrics from the mechanistic interpretability lens: weight decomposition, graph analysis, and compositional structure."
---

# MI Structural Metrics

This page documents the 22 structural metrics that implement the [mechanistic interpretability lens](/framework/lenses/core/mechanistic-interpretability). These metrics operate on model weights and circuit graph structure to assess whether a claimed circuit has the weight-space and graph-theoretic properties that the claimed computation requires.

The metrics fall into five groups:

1. **Weight-space spectral analysis** -- SVD, effective rank, and spectral norms of attention weight matrices.
2. **Weight-space functional signatures** -- copying scores, composition matrices, alignment, and capacity utilization.
3. **Circuit distance and convergence** -- Jaccard overlap between independently derived circuits.
4. **Graph structure** -- path identification, edge necessity, path specificity, compositional sufficiency, minimality, and motif enrichment.
5. **Clustering and complexity** -- attention pattern clustering, polysemanticity, and local learning coefficients.

---

## Weight-Space Spectral Analysis

### B01 -- SVD / Spectral Analysis (`18_weight_extended.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** Three CPU-only metrics derived from the SVD of attention weight matrices, requiring no forward passes.

- **W_QK effective rank**: For each head, computes $W_Q W_K^T \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$, takes its SVD, and reports the effective rank $\exp(H(\hat{\sigma}))$ where $H$ is the entropy of the normalized singular values. Low effective rank means the head implements a small number of attention motifs.
- **Cosine alignment**: For each circuit head, computes the top-3 right singular vectors of $W_{OV}$, projects them through the unembedding matrix $W_U$, and reports the maximum absolute cosine similarity with any non-circuit head's projected directions. Low cosine indicates a specialized head; high cosine indicates a generic one.
- **Spectral norm ratio**: The ratio of mean spectral norm ($\sigma_1$ of $W_{OV}$) for circuit heads versus non-circuit heads.

**What it establishes.** Circuit heads occupy spectrally distinct subspaces in weight space -- they have concentrated singular value spectra and specialized output directions.

**What it does not establish.** Whether the spectral structure is causally necessary for the task, or whether the head's concentrated spectrum encodes the correct computation.

**Metric IDs.**
- `C18.wqk_effective_rank` -- mean effective rank of $W_{QK}$ for circuit heads
- `C18.cosine_alignment` -- mean max cosine similarity with non-circuit heads
- `C18.spectral_norm_ratio` -- ratio of circuit to non-circuit mean spectral norms

**Usage.**
```bash
uv run python 18_weight_extended.py --tasks ioi sva greater_than
```

---

### B05 -- Norm Trajectory (`51_norm_trajectory.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** Tracks the Frobenius norm of $W_{OV}$ for circuit heads versus non-circuit heads across layers. Reports the peak layer (layer with highest mean circuit OV norm), the linear slope of the norm trajectory across layers, and the overall norm ratio (circuit mean / non-circuit mean).

**What it establishes.** Circuit heads have distinctive norm magnitude profiles -- the circuit's contribution peaks at specific layers rather than being uniformly distributed, consistent with a staged computation.

**What it does not establish.** Whether the norm profile reflects the correct computational stages, or whether the peak layer coincides with the functionally important layer (that requires causal validation).

**Metric ID.**
- `B51.norm_trajectory` -- ratio of mean circuit OV Frobenius norm to non-circuit mean

**Usage.**
```bash
uv run python 51_norm_trajectory.py --tasks ioi sva greater_than
```

---

### B07 -- Polysemanticity Index (`52_polysemanticity.py`)

**Validity layer:** Construct (C3 -- Task specificity)

**What it computes.** Three complementary measures of how polysemantic each head's OV matrix is.

- **Effective rank of $W_{OV}$**: $\exp(H(\hat{\sigma}))$ over the normalized singular values. High effective rank indicates many active computational directions (polysemantic).
- **Participation ratio**: $(\sum \sigma_i)^2 / \sum \sigma_i^2$, ranging from 1 (single dominant direction) to $n$ (all directions equal).
- **Unembedding fan-out**: The number of unembedding columns (token directions) for which the top singular vector of $W_{OV}$ has $|\cos| > 0.1$. High fan-out means the head writes to many output tokens.

Compares circuit heads versus non-circuit heads on all three measures.

**What it establishes.** Circuit heads are less polysemantic than non-circuit heads -- they project onto fewer output directions and have more concentrated singular value spectra, consistent with task-specialized computation.

**What it does not establish.** Whether the monosemantic directions encode task-relevant features (that requires logit lens or feature analysis), or whether the specialization is necessary for the circuit's function.

**Metric IDs.**
- `B52.polysemanticity_eff_rank` -- mean effective rank for circuit heads (ratio < 1 = more monosemantic)
- `B52.polysemanticity_participation` -- mean participation ratio for circuit heads
- `B52.polysemanticity_fan_out` -- mean unembedding fan-out for circuit heads

**Usage.**
```bash
uv run python 52_polysemanticity.py --tasks ioi sva greater_than
```

---

### B08 -- Weight Decomposition / NMF (`50_weight_decomposition.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** Applies Non-negative Matrix Factorization (NMF) to the absolute values of stacked $|W_{OV}|$ matrices from circuit heads, then measures reconstruction error with half the available components. Compares against random head subsets of the same size. Also reports the number of SVD components needed to explain 90% of variance.

**What it establishes.** Circuit heads share more low-rank weight structure than random subsets -- their OV matrices can be reconstructed from fewer shared components, indicating common computational building blocks.

**What it does not establish.** What the shared components represent semantically, or whether the shared structure is specific to this task versus being a generic architectural property.

**Metric ID.**
- `B50.weight_decomposition` -- reconstruction error for circuit heads (ratio < 1 means more shared structure than random)

**Usage.**
```bash
uv run python 50_weight_decomposition.py --tasks ioi greater_than
```

---

## Weight-Space Functional Signatures

### B03 -- OV/QK Composition Scores (`49_ov_qk_composition.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** For each pair of heads $(h_1, h_2)$ where $h_1$ is in an earlier layer than $h_2$, computes the Frobenius-normalized composition score:

$$\text{score} = \frac{\|W_{OV}^{(h_1)} \cdot W_{QK}^{(h_2)}\|_F}{\|W_{OV}^{(h_1)}\|_F \cdot \|W_{QK}^{(h_2)}\|_F}$$

Compares composition scores for circuit edges (sender-receiver pathway pairs) against non-circuit edges.

**What it establishes.** The circuit's wiring is structurally privileged in weight space -- circuit edges have higher OV-QK composition scores than random head pairs, indicating direct weight-space communication channels between sender and receiver heads.

**What it does not establish.** Whether the weight-space composition is activated on task-relevant inputs (that requires activation-level analysis), or whether the composition pathway is the dominant one used by the circuit.

**Metric ID.**
- `B49.ov_qk_composition` -- mean composition score for circuit edges, with non-circuit edge mean as baseline

**Usage.**
```bash
uv run python 49_ov_qk_composition.py --tasks ioi greater_than
```

---

### B10 -- K-Composition Matrix (`B10_k_composition.py`)

**Validity layer:** Construct (C2 -- Structural plausibility, I3 -- Specificity)

**What it computes.** Measures pairwise K-composition between all head pairs: $\|W_O^{(\text{sender})} \cdot W_K^{(\text{receiver})}\|_F$. High K-composition means the sender's output subspace aligns with the receiver's key subspace -- a direct weight-space communication channel. Reports the number of "strong" edges (z-score > 1.0 versus background) and the hierarchy depth (longest chain of strong edges through the circuit).

**What it establishes.** Circuit heads communicate via direct weight composition, and the circuit has hierarchical depth consistent with a multi-stage computation.

**What it does not establish.** Whether the weight-space channel is the one actually used at inference time, or whether the hierarchy reflects the true causal ordering of the computation.

**Metric ID.**
- `B10.k_composition_mean` -- mean K-composition across circuit head pairs

**Usage.**
```bash
uv run python B10_k_composition.py --tasks ioi sva greater_than
```

---

### B11 -- Copying Score (`B11_copying_score.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** For each head, computes the largest eigenvalue of the symmetrized matrix $(W_U W_E) \cdot W_{OV}$, where $W_E$ is the embedding matrix and $W_U$ is the unembedding matrix. A large positive eigenvalue means the head's OV circuit maps input token embeddings back to the same output tokens -- token-copying behavior in weight space.

**What it establishes.** Whether specific circuit heads function as token copiers, which is a structurally expected property for heads in the "S-inhibition" or "name mover" roles of circuits like IOI.

**What it does not establish.** Whether the copying behavior is task-specific or a generic property of the head, or whether the head actually performs copying on task-relevant inputs (that requires activation analysis).

**Metric ID.**
- `B11.copying_score` -- mean copying score for circuit heads, with non-circuit mean as baseline

**Usage.**
```bash
uv run python B11_copying_score.py --tasks ioi sva greater_than
```

---

### B12 -- QK Norms and Singular Value Gap (`B12_qk_norms.py`)

**Validity layer:** Construct (C2 -- Structural plausibility, C3 -- Task specificity)

**What it computes.** Two complementary measurements on $W_Q W_K^T$ per head:

- **QK Frobenius norm**: $\|W_Q W_K^T\|_F$. High norm indicates the head has strong attention preferences in weight space. Previous-token heads (PTH) typically show norms 3--4x higher than typical heads.
- **QK singular value gap**: Ratio of the top-1 to top-2 singular values of $W_{QK}$. A high gap means attention is dominated by a single direction (focused, PTH-like). A low gap means attention is distributed across many directions.

**What it establishes.** Circuit heads have attention focus profiles consistent with their claimed functional roles -- high QK norms and gaps for position-sensitive heads, lower gaps for heads attending to content.

**What it does not establish.** The semantic content of the attention pattern, or whether the QK structure is causally necessary for the circuit.

**Metric IDs.**
- `B12.qk_frob_norm` -- mean QK Frobenius norm for circuit heads
- `B12.sv_gap` -- mean QK singular value gap for circuit heads

**Usage.**
```bash
uv run python B12_qk_norms.py --tasks ioi sva greater_than
```

---

### B13 -- Capacity Utilization Gap (`B13_capacity_utilization.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** For each head, computes the effective rank and top singular value concentration ($\sigma_1 / \sum \sigma_i$) for both $W_{QK}$ and $W_{OV}$. High concentration means the head's capacity is dominated by one computational direction (tight, specialized). Low concentration means capacity is spread across many directions (diffuse, general). Reports the ratio of circuit head concentration to non-circuit head concentration.

**What it establishes.** Circuit heads use a larger fraction of their weight-space capacity for a small number of directions -- they are more specialized than background heads.

**What it does not establish.** Whether the specialized directions are the correct ones for the claimed task, or whether the capacity utilization pattern is unique to this circuit.

**Metric ID.**
- `B13.capacity_utilization` -- ratio of circuit QK concentration to background QK concentration

**Usage.**
```bash
uv run python B13_capacity_utilization.py --tasks ioi sva
```

---

### B14 -- K-Alignment with Embedding Directions (`B14_k_alignment.py`)

**Validity layer:** Construct (C2 -- Structural plausibility)

**What it computes.** For each head, computes the top-$k$ right singular vectors of $W_{QK}$ and measures their maximum cosine similarity with the mean embedding direction from $W_E$. Also computes OV alignment: the top left singular vectors of $W_{OV}$ projected onto the unembedding matrix $W_U$.

**What it establishes.** Circuit heads that attend to specific token types show higher alignment between their QK subspace and the embedding space, since the keys they match against are functions of token embeddings. OV alignment indicates the head's output is directed toward specific token predictions.

**What it does not establish.** Which specific tokens the head attends to or produces, or whether the alignment is causally necessary for the circuit's function.

**Metric ID.**
- `B14.k_alignment` -- ratio of circuit QK-embedding alignment to background alignment

**Usage.**
```bash
uv run python B14_k_alignment.py --tasks ioi sva
```

---

## Circuit Distance and Convergence

### B06 -- Circuit Metric Distance (`26_cmd.py`)

**Validity layer:** Construct (C3 -- Task specificity)

**What it computes.** For each pair of tasks, computes the Jaccard distance between their circuit head sets: $d = 1 - |A \cap B| / |A \cup B|$. Reports the full task-by-task distance matrix, number of shared heads, number of unique heads per circuit, and normalized overlap.

**What it establishes.** Different tasks produce structurally distinct circuits. High Jaccard distance between task circuits confirms that the circuit definitions are task-specific rather than reflecting generic model structure.

**What it does not establish.** Whether the shared heads between tasks reflect genuine functional overlap (e.g., shared positional processing) or accidental overlap in the circuit discovery method.

**Metric ID.**
- `C26.cmd` -- mean pairwise Jaccard distance across all task pairs

**Usage.**
```bash
uv run python 26_cmd.py --tasks ioi sva induction
```

---

### B06 -- Edge Overlap / Jaccard (`27_edge_jaccard.py`)

**Validity layer:** Construct (C3 -- Task specificity)

**What it computes.** For each task, computes the Jaccard similarity between circuit edges derived from role-based pathway definitions and edges derived from Edge Attribution Patching (EAP). Loads EAP edge data from available data files and constructs directed edges between head pairs.

**What it establishes.** Circuit edge structure is consistent across independent discovery methods -- weight-based and activation-based circuit discovery converge on similar inter-head connections.

**What it does not establish.** That either edge set is correct in isolation, or that the overlapping edges are more important than the non-overlapping ones.

**Metric ID.**
- `C27.edge_jaccard` -- Jaccard similarity between role-based and EAP-derived edges

**Usage.**
```bash
uv run python 27_edge_jaccard.py --tasks ioi sva
```

---

### B04 -- Weight-EAP Head Jaccard (`28_weight_eap_jaccard.py`)

**Validity layer:** Construct (C2/C5 -- Convergent validity)

**What it computes.** For each task, computes the Jaccard similarity between weight-classifier-derived circuit heads and EAP-derived circuit heads. Loads EAP head sets from available data files. Reports the overlap, the heads unique to each method, and the full head lists.

**What it establishes.** Weight-derived circuits converge with activation-derived circuits -- two independent methods for identifying important heads agree on a substantial fraction of the circuit.

**What it does not establish.** That either method identifies the "true" circuit, or that heads found by only one method are unimportant. Convergence is supporting evidence, not proof.

**Metric ID.**
- `C28.weight_eap_jaccard` -- Jaccard similarity between weight-derived and EAP-derived head sets

**Usage.**
```bash
uv run python 28_weight_eap_jaccard.py --tasks ioi sva
```

---

## Graph Structure

### G01 -- Path Identification (`82_path_identification.py`)

**Validity layer:** Internal (G1 -- Path identification)

**What it computes.** For each edge (upstream-to-downstream head pair), performs path-patching: replaces the upstream head's output contribution at the downstream layer with the corrupted version (from a different prompt) and measures the change in logit diff. Computes a specificity ratio: the effect magnitude on task-relevant answer tokens divided by the effect on control (shuffled) answer tokens.

**Pass condition:** At least one path with specificity > 5x.

**What it establishes.** Specific information flow paths can be traced through the circuit -- particular edges carry task-specific signal that is substantially larger than signal for unrelated tokens.

**What it does not establish.** That the identified paths are the only ones used by the circuit, or that the path structure is stable across different input distributions.

**Metric ID.**
- `G1.path_identification` -- maximum specificity ratio across all circuit edges

**Usage.**
```bash
uv run python 82_path_identification.py --tasks ioi --n-prompts 40
```

---

### G02 -- Edge Necessity (`83_edge_necessity.py`)

**Validity layer:** Internal (G2 -- Edge necessity)

**What it computes.** For each edge in the circuit, mean-ablates the upstream head's contribution at the downstream layer position and measures the resulting drop in logit diff. An edge is "necessary" if ablating it causes > 5% drop relative to the full model.

**Pass condition:** At least 50% of edges are individually necessary.

**What it establishes.** Specific edges -- not just nodes -- are necessary for circuit function. This goes beyond node-level ablation by testing whether the inter-head connections carry essential information.

**What it does not establish.** That the edges are sufficient (a necessary edge may still require other components to function), or that the edge-level structure is minimal.

**Metric ID.**
- `G2.edge_necessity` -- fraction of edges that are individually necessary

**Usage.**
```bash
uv run python 83_edge_necessity.py --tasks ioi --n-prompts 40
```

---

### G03 -- Path Specificity (`84_path_specificity.py`)

**Validity layer:** Internal (G3 -- Path specificity)

**What it computes.** Splits prompts into two conditions: task-relevant (correct answer tokens) and control (swapped correct/incorrect tokens). Computes the edge activation pattern (vector of edge effect magnitudes across all edges) for each condition, then measures the Spearman correlation between the two patterns. Low correlation means different conditions activate different circuit paths.

**Pass condition:** Spearman $\rho < 0.5$.

**What it establishes.** The circuit uses genuinely different paths for different conditions, indicating that the graph structure encodes task-relevant distinctions rather than being a single fixed pipeline.

**What it does not establish.** What the different paths compute or whether they correspond to interpretable sub-computations.

**Metric ID.**
- `G3.path_specificity` -- Spearman correlation between task-relevant and control edge patterns

**Usage.**
```bash
uv run python 84_path_specificity.py --tasks ioi --n-prompts 40
```

---

### G04 -- Compositional Sufficiency (`85_compositional_sufficiency.py`)

**Validity layer:** Internal (G4 -- Compositional sufficiency)

**What it computes.** Tests whether the circuit's graph structure (edges between bands of heads) carries signal beyond the individual bands. Computes full-circuit faithfulness (keep all circuit heads, ablate rest), per-band faithfulness (keep one band at a time), and superadditivity: $\text{full} - \max(\text{individual bands})$. Positive superadditivity means the inter-band wiring adds value beyond what any single band provides.

**Pass condition:** Superadditivity > 0.05 AND full recovery > 0.2.

**What it establishes.** The circuit's bands compose meaningfully through their edges -- the whole is greater than its best part. This distinguishes compositional circuits from circuits that are merely collections of independently sufficient heads.

**What it does not establish.** Which specific inter-band edges are responsible for the superadditivity, or whether the band decomposition is the optimal grouping.

**Metric ID.**
- `G4.compositional_sufficiency` -- superadditivity (full circuit recovery minus best single-band recovery)

**Usage.**
```bash
uv run python 85_compositional_sufficiency.py --tasks ioi --n-prompts 40
```

---

### G05 -- Graph Minimality (`86_graph_minimality.py`)

**Validity layer:** Internal (G5 -- Graph minimality)

**What it computes.** Combines edge necessity with direction-aware testing. For each edge, ablating it must cause > 5% drop in logit diff AND the drop must be in the task-relevant direction (decrease, not increase). Reports the minimality ratio: the fraction of edges that are both magnitude-necessary and directionally correct.

**Pass condition:** Minimality ratio $\geq$ 0.8.

**What it establishes.** The circuit's edge set is minimal -- most edges contribute in the correct direction and with sufficient magnitude. A high minimality ratio indicates the circuit definition is tight, without redundant or counterproductive edges.

**What it does not establish.** That the circuit is the unique minimal set (there may be other minimal circuits), or that remaining non-minimal edges are errors in the circuit definition (they may serve redundancy or robustness functions).

**Metric ID.**
- `G5.graph_minimality` -- fraction of edges that are necessary and directionally correct

**Usage.**
```bash
uv run python 86_graph_minimality.py --tasks ioi --n-prompts 40
```

---

### G05 -- Network Motif Enrichment (`97_network_motifs.py`)

**Validity layer:** Internal (G5 -- Network motif enrichment)

**What it computes.** Inspired by Alon (2002), enumerates 3-node directed subgraph patterns (feed-forward chains, fan-in, fan-out) and the 4-node bi-fan pattern in the circuit's directed graph. Generates 1,000 Erdos-Renyi random graphs with the same node and edge counts and computes z-scores for each motif pattern. A motif is "enriched" if its z-score exceeds 2.0.

**Pass condition:** At least one motif is significantly enriched.

**What it establishes.** The circuit's directed graph contains over-represented subgraph patterns compared to random graphs of the same size -- the wiring has non-random structure consistent with specific computational motifs (e.g., hierarchical fan-in for information aggregation).

**What it does not establish.** That the enriched motifs correspond to the claimed computation, or that the ER null model is the appropriate baseline (see G7 for a degree-preserving alternative).

**Metric ID.**
- `G5.network_motif_enrichment` -- best z-score across all motif patterns

**Usage.**
```bash
uv run python 97_network_motifs.py --tasks ioi --n-random 1000
```

---

### G07 -- Motif Enrichment, Alon 2007 (`G7_motif_enrichment.py`)

**Validity layer:** Internal (G7 -- Motif enrichment, paper-faithful)

**What it computes.** A more rigorous implementation of network motif analysis following Alon (2007) methodology. Enumerates all 13 three-node directed triad census classes (the standard triad classification from graph theory) plus the bi-fan pattern. Uses degree-preserving edge swaps rather than Erdos-Renyi random graphs for the null model, preserving the in-degree and out-degree sequence. Applies Bonferroni correction for multiple testing across all motif classes.

**Pass condition:** At least one motif enriched at $p < 0.05$ after Bonferroni correction.

**What it establishes.** The circuit's graph topology has over-represented motifs even after controlling for degree distribution and multiple comparisons. This is stronger evidence than G05/97 because the degree-preserving null model eliminates the possibility that enrichment is merely a consequence of heterogeneous degree distributions.

**What it does not establish.** The functional significance of the enriched motifs, or whether the motif structure is specific to this task.

**Metric ID.**
- `G7.motif_enrichment` -- best z-score across all triad census classes and bi-fan

**Usage.**
```bash
uv run python G7_motif_enrichment.py --tasks ioi --n-random 1000
```

---

## Clustering and Complexity

### S96 -- Attention Pattern Clustering (`96_attention_clustering.py`)

**Validity layer:** Internal

**What it computes.** Runs the model on task prompts, collects last-token attention patterns from all heads, flattens each into a vector, and applies k-means clustering (with $k$ = number of circuit heads). Reports two quantities:

- **Cluster purity**: what fraction of circuit heads land in the same cluster.
- **Silhouette score**: how well the binary grouping (circuit vs. non-circuit) separates heads in attention-pattern space.

**Pass condition:** Silhouette score > 0.1.

**What it establishes.** Circuit heads form distinct clusters in attention-pattern space -- they are more similar to each other than to non-circuit heads, indicating a shared functional signature in how they distribute attention.

**What it does not establish.** What the shared attention pattern represents semantically, or whether the clustering is task-specific versus reflecting generic architectural groupings.

**Metric ID.**
- `S96.attention_clustering` -- silhouette score for circuit vs. non-circuit grouping

**Usage.**
```bash
uv run python 96_attention_clustering.py --tasks ioi --n-prompts 40
```

---

### B09 -- Local Learning Coefficient / LLC (`10_llc.py`)

**Validity layer:** Construct (C4 -- Minimality)

**What it computes.** Estimates the local learning coefficient (LLC) from Singular Learning Theory for each circuit component. Uses a Hessian-based approximation: computes gradient variance (Fisher information diagonal) across prompts, counts the number of dimensions with variance above 1% of the maximum, and derives an LLC estimate as $\text{effective\_dim} / (2 \ln n)$. Low LLC indicates a geometrically simple, specialized region of the loss landscape; high LLC indicates a complex, potentially polyfunctional region.

Optionally uses the `devinterp` library for SGLD-based LLC estimation when available.

**What it establishes.** Circuit components occupy geometrically simpler regions of the loss landscape than non-circuit components, consistent with specialized rather than general-purpose computation.

**What it does not establish.** Whether the geometric simplicity reflects the correct specialization (a head could be simple but task-irrelevant), or whether LLC differences are large enough to be functionally meaningful.

**Metric ID.**
- `C10.llc` -- mean LLC for circuit heads, with non-circuit mean as baseline

**Usage.**
```bash
uv run python 10_llc.py --tasks ioi sva --n-prompts 40
```

---

## Reading the Scores

| Pattern | Interpretation |
|---|---|
| Low effective rank + high SV gap (B01/B12) | Head implements a focused, single-direction computation (e.g., PTH or induction) |
| High OV/QK composition ratio (B03/B49) | Circuit edges are structurally privileged -- sender output aligns with receiver keys |
| High K-composition depth (B10) | Circuit has hierarchical multi-stage structure in weight space |
| High copying score for name movers (B11) | Head's OV circuit maps tokens to themselves, consistent with copying role |
| Low polysemanticity ratio (B07/B52) | Circuit heads are more monosemantic than background -- task-specialized |
| Low NMF reconstruction error ratio (B08/B50) | Circuit heads share low-rank weight structure -- common computational building blocks |
| High capacity concentration ratio (B13) | Circuit heads use their weight capacity for fewer directions than background |
| High Jaccard between weight and EAP circuits (B04/B06) | Independent discovery methods converge -- circuit is robust to methodology |
| Superadditivity > 0 (G04) | Circuit bands compose through edges -- the whole exceeds the sum of parts |
| High minimality ratio (G05) | Circuit edges are individually necessary and directionally correct -- tight definition |
| Enriched feed-forward chains (G05/G07) | Circuit has hierarchical processing structure |
| Enriched fan-in motifs (G05/G07) | Circuit aggregates information from multiple sources to single targets |
| Low LLC ratio (B09) | Circuit components occupy geometrically simple loss landscape regions |
| High silhouette score (S96) | Circuit heads cluster together in attention-pattern space |

## How the Metrics Relate

The structural metrics form three tiers of increasing specificity:

1. **Per-head weight structure** (B01, B05, B07, B09, B11, B12, B13, B14): These characterize individual heads in isolation. They answer "does this head have properties consistent with its claimed role?" Effective rank tells you whether a head is focused, copying score tells you whether it copies, QK norms tell you whether it attends sharply.

2. **Pairwise weight composition** (B03, B08, B10): These characterize head pairs and shared structure. They answer "do the circuit's heads communicate through weight space?" OV/QK composition and K-composition measure direct weight-space channels; NMF decomposition measures shared basis vectors across the circuit's heads.

3. **Graph topology** (G01--G05, G07, S96, B06): These characterize the circuit as a whole. They answer "does the circuit's wiring have non-trivial structure?" Path identification and edge necessity test individual edges; compositional sufficiency and minimality test the full graph; motif enrichment tests whether the graph's topology is non-random; circuit metric distance and convergence metrics test cross-method and cross-task consistency.

Evidence from all three tiers should align: a well-supported circuit claim has spectrally distinctive heads (tier 1) that communicate through weight-space channels (tier 2) in a non-random, minimal, compositional graph (tier 3).
