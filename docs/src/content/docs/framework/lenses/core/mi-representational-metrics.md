---
title: "MI Representational Metrics"
description: "Representational metrics from the mechanistic interpretability lens: probing, RSA, CKA, and attention analysis."
---

# MI Representational Metrics

This page documents the representational metrics that implement the [mechanistic interpretability lens](/framework/lenses/core/mechanistic-interpretability). These metrics characterize the geometric and statistical structure of neural representations at circuit-relevant layers: whether task information is linearly decodable, whether circuit layers encode task-relevant similarity structure, whether circuit subnetworks capture the full model's representational geometry, and whether attention patterns distinguish circuit from non-circuit heads. They are implemented in `mechval_v2.core.mechanistic_interpretability.representational` and can be run independently or as part of a protocol.

## Metrics

---

### E03 -- Representational Similarity Analysis (`61_rsa.py`)

**What it computes.** Computes RSA (Kriegeskorte et al., 2008) between model residual-stream representations and a task-defined target similarity structure. For each task, a target representational dissimilarity matrix (RDM) encodes which prompts should have similar representations -- prompts requiring the same correct answer are assigned distance 0, all others distance 1. A neural RDM is built from cosine distances of residual-stream activations at each layer's last token position. The RSA score is the Spearman rank correlation between the upper triangles of the target and neural RDMs.

$$
\text{RSA}(\ell) = \rho_{\text{Spearman}}\bigl(\text{vec}(\text{RDM}_{\text{target}}),\; \text{vec}(\text{RDM}_{\text{neural}}^{(\ell)})\bigr)
$$

**Evidence family.** Representational (geometric correspondence).

**Key metrics.**
| Metric | Description | Interpretation |
|---|---|---|
| `mean_circuit_rsa` - `mean_non_circuit_rsa` | Difference in RSA between circuit and non-circuit layers | Primary: circuit advantage |
| `peak_rsa` | Maximum RSA score across all layers | Peak similarity to task structure |
| `peak_layer` | Layer at which RSA peaks | Where task structure is most explicit |

**What it establishes.** Circuit layers encode task-relevant similarity structure: prompts that require the same answer are represented more similarly at circuit layers than at non-circuit layers. The RSA peak should coincide with circuit-critical layers -- indicating that those layers organize the residual stream around the task's decision boundary.

**What it does not establish.** That the similarity structure is causally used by the circuit. RSA is observational: it shows that task structure is encoded in the geometry of representations, but not that the model reads out from this geometry to produce its answer. A layer might encode task similarity as a side effect of other computations without the downstream circuit using it. Combine with probing (E02) or causal representation tests (R3) for causal evidence.

**Usage.**
```bash
uv run python 61_rsa.py --tasks ioi sva --n-prompts 40
```

---

### E02 -- Linear Probe (`66_linear_probe.py`)

**What it computes.** Trains a closed-form linear probe (OLS regression) at each layer's residual stream to predict whether the model's top prediction matches the correct answer (binary label). Measures where task-relevant information becomes linearly decodable. Then ablates all circuit heads (mean ablation) and re-probes to verify the circuit concentrates predictive signal at specific layers.

**Evidence family.** Representational (linear decodability).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `max_clean_acc` | Maximum probe accuracy across all layers (clean model) | $> 0.5$ (chance) |
| `mean_drop_at_circuit_layers` | Mean accuracy drop at circuit layers after circuit ablation | reported |
| `accuracy_per_layer_clean` | Probe accuracy at each layer | profile |
| `accuracy_per_layer_ablated` | Probe accuracy at each layer after circuit ablation | profile |

**What it establishes.** Task information is linearly decodable from the residual stream, and ablating circuit heads reduces linear decodability specifically at circuit-relevant layers. The clean probe accuracy shows where information appears; the ablation-induced drop shows where it depends on the circuit. A large drop at circuit layers means the circuit heads are responsible for making task information linearly accessible at those layers.

**What it does not establish.** That the model uses a linear readout. Linear probes can decode information that the model does not use (Hewitt & Liang, 2019). The selectivity baseline in the probe decodability metric (R1) addresses this concern. A high probe accuracy with no ablation-induced drop would indicate the circuit is not the source of the decodable information.

**Usage.**
```bash
uv run python 66_linear_probe.py --tasks ioi greater_than --n-prompts 60
```

---

### R1 -- Probe Decodability with Selectivity (`75_probe_decodability.py`)

**What it computes.** Extends the linear probe (E02) with a selectivity baseline (Hewitt & Liang, 2019). For each circuit layer, trains a logistic regression probe (gradient descent, 100 epochs) to predict correct/incorrect label from residual-stream activations. Additionally trains the same probe on random binary labels (the control task). Selectivity is the difference between task probe accuracy and control probe accuracy:

$$
\text{selectivity} = \text{acc}_{\text{task}} - \text{acc}_{\text{control}}
$$

The control task has no relationship to the activations, so any accuracy above chance reflects the probe's ability to memorize or exploit spurious structure. Subtracting it out isolates the task-specific component.

**Evidence family.** Representational (controlled probing).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `selectivity` | Best selectivity across circuit layers | $> 0.10$ |
| `task_accuracy` | Probe accuracy on the real task at each circuit layer | reported |
| `control_accuracy` | Probe accuracy on random labels at each circuit layer | reported |
| `any_layer_passes` | Whether at least one circuit layer exceeds selectivity threshold | boolean |

**What it establishes.** Task information at circuit layers is genuinely task-specific -- not an artifact of probe expressiveness. A probe that achieves $0.85$ accuracy on the real task and $0.75$ on random labels has only $0.10$ selectivity: most of its apparent "understanding" is spurious. The selectivity threshold ensures the reported decodability reflects real task structure.

**What it does not establish.** That the decodable information is causally used. Selectivity addresses the "probes memorize" concern but not the "probes decode unused information" concern. A representation can encode genuine task structure that the downstream computation ignores. Use the causal representation test (R3) to establish that the representation is load-bearing.

**Usage.**
```bash
uv run python 75_probe_decodability.py --tasks ioi sva --n-prompts 60
```

---

### R3 -- Causal Representation Test (`76_causal_representation.py`)

**What it computes.** Simplified interchange intervention accuracy (IIA) without DAS rotation. Generates counterfactual prompt pairs -- pairs with different correct answers -- then patches the residual stream from the source prompt into the base prompt at circuit layers and checks whether the model output follows the patched source. A control condition patches at a random non-circuit layer.

$$
\text{IIA}(\ell) = \frac{|\{(A, B) : \text{output}(B_{\text{patched from } A \text{ at } \ell}) = \text{answer}(A)\}|}{|\text{pairs}|}
$$

**Evidence family.** Representational + causal (interchange intervention).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `best_circuit_iia` | Highest IIA across circuit layers | $> 0.70$ |
| `control_iia` | IIA at a random non-circuit layer | $< 0.30$ |
| `passed` | Both conditions met | boolean |

**What it establishes.** The representation at circuit layers is load-bearing: patching it from a source prompt causes the model to produce the source's answer. This goes beyond decodability -- it demonstrates that the model reads from these activations to determine its output. The control condition at non-circuit layers verifies that the effect is specific to circuit layers, not a generic consequence of patching any layer.

**What it does not establish.** That the interchange intervention reflects a valid causal abstraction. Without the DAS rotation (Geiger et al., 2021), the intervention may not align with the model's internal causal variables. The raw residual stream at a layer may conflate multiple causal variables, and patching the entire residual stream intervenes on all of them simultaneously. High IIA under raw patching is a sufficient but not necessary condition for causal representation.

**Usage.**
```bash
uv run python 76_causal_representation.py --tasks ioi sva --n-prompts 40
```

---

### E92 -- Centered Kernel Alignment (`92_cka.py`)

**What it computes.** Computes linear CKA (Kornblith et al., ICML 2019) between the circuit subnetwork's representation and the full model's representation at each layer. For each layer with circuit heads, collects concatenated head outputs ($z$ at last token) for circuit heads only and for all heads, centers both matrices, and computes:

$$
\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}
$$

where $X$ is the full-model activation matrix and $Y$ is the circuit-subnetwork activation matrix (both centered, with rows = prompts and columns = concatenated head outputs).

**Evidence family.** Representational (kernel alignment).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `mean_cka_circuit_layers` | Mean CKA between circuit and full representations at circuit layers | $> 0.60$ |
| `per_layer_cka` | CKA at each layer (0 at layers without circuit heads) | profile |

**What it establishes.** The circuit subnetwork captures the representational structure of the full model at circuit-relevant layers. High CKA means the circuit heads produce a representation that is geometrically aligned with the full set of heads' representation -- the circuit is not computing something orthogonal to the rest of the model. This is evidence that the circuit is a faithful subnetwork, not a disconnected fragment.

**What it does not establish.** That the circuit captures all relevant computation. CKA measures alignment, not completeness. A circuit with CKA $= 0.8$ is well-aligned but may miss important structure captured by non-circuit heads. Low CKA at circuit layers suggests the circuit heads produce a representation that is geometrically distinct from the full model's, which may indicate the circuit is incomplete or that non-circuit heads perform related but different computations.

**Usage.**
```bash
uv run python 92_cka.py --tasks ioi sva --n-prompts 40
```

---

### E11 -- Attention Entropy (`E11_attention_entropy.py`)

**What it computes.** Computes the Shannon entropy of each head's attention pattern across prompts:

$$
H(L, H) = -\sum_{i} \text{attn}(i) \log \text{attn}(i)
$$

where $\text{attn}(i)$ is the attention weight at position $i$, averaged over the sequence dimension and across prompts. Low entropy indicates focused attention (the head attends primarily to one or a few positions); high entropy indicates diffuse attention (the head spreads attention broadly across the sequence). Compares mean entropy for circuit heads vs non-circuit heads.

**Evidence family.** Representational (attention pattern analysis).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `circuit_mean` | Mean attention entropy across circuit heads | compared to non-circuit |
| `non_circuit_mean` | Mean attention entropy across non-circuit heads | baseline |
| `ratio` | Circuit mean / non-circuit mean | reported |
| `circuit_min` | Minimum entropy in circuit (most focused head) | reported |
| `per_head` | Per-head entropy values for all circuit heads | for inspection |

**What it establishes.** Circuit heads have distinctive attention patterns compared to non-circuit heads. For circuits involving position-sensitive operations (e.g., previous-token heads in induction circuits), circuit heads should have very low entropy ($\sim 0.02$), indicating near-deterministic attention to a specific position. The ratio and per-head values reveal whether the circuit contains a mix of focused and diffuse attention heads -- suggesting different functional roles.

**What it does not establish.** That the attention pattern is causally relevant. A head can have focused attention but contribute nothing to the output (its OV circuit may project to a dead direction). Conversely, a head with high-entropy (diffuse) attention may aggregate information from many positions, which is functionally important. Attention entropy characterizes the QK circuit but says nothing about the OV circuit.

**Usage.**
```bash
uv run python E11_attention_entropy.py --tasks ioi sva greater_than
```

---

### E6b -- CKA Cross-Layer Analysis (`E6b_cka_cross_arch.py`)

**What it computes.** Computes linear CKA between residual-stream activations at different circuit layers, measuring how much representational structure is preserved as information flows through the circuit. For each pair of circuit layers, computes CKA between their last-token residual-stream activations. Additionally computes CKA between circuit and non-circuit layers for structural comparison.

**Evidence family.** Representational (cross-layer preservation).

**Key metrics.**
| Metric | Description | Pass threshold |
|---|---|---|
| `mean_consecutive_cka` | Mean CKA between consecutive circuit layers | $> 0.3$ |
| `first_last_cka` | CKA between first and last circuit layer | reported |
| `mean_circuit_vs_non_circuit` | Mean CKA between circuit and non-circuit layers | reported |
| `cka_matrix` | Full CKA matrix between all pairs of circuit layers | for visualization |

**What it establishes.** Information is preserved across circuit layers: consecutive circuit layers produce representations that share geometric structure. High consecutive CKA means the circuit processes information incrementally rather than performing a radical transformation at each stage. The first-last CKA measures how much of the initial representation survives to the final circuit layer -- a low value indicates substantial transformation (expected for circuits that compute new information), while a high value indicates the circuit mainly preserves and refines existing structure.

**What it does not establish.** That the preserved structure is task-relevant. Two layers can have high CKA because they both preserve input-level features (e.g., position embeddings) that have nothing to do with the circuit's task computation. The cross-layer CKA characterizes overall representational similarity, not task-specific similarity. Combine with RSA (E03) for task-specific representational analysis.

**Usage.**
```bash
uv run python E6b_cka_cross_arch.py --tasks ioi sva --n-prompts 20
```

---

## Reading the Scores

### Metric-level interpretation

| Metric | High score | Low score |
|---|---|---|
| E03 (RSA) | Circuit layers encode task similarity structure; geometry matches task | No preferential encoding; task structure is uniformly distributed or absent |
| E02 (Linear Probe) | Task info is linearly decodable; ablation reduces decodability at circuit layers | Decodability is uniform; ablation does not preferentially affect circuit layers |
| R1 (Probe Selectivity) | Decodability is task-specific, not probe memorization | High control accuracy; most decodability is spurious |
| R3 (Causal Representation) | Patching circuit-layer activations controls model output; representation is load-bearing | Patching does not control output; representation is decodable but not used |
| E92 (CKA) | Circuit subnetwork captures full model's representational structure | Circuit heads compute orthogonal to the rest of the model |
| E11 (Attention Entropy) | Circuit heads have distinctive attention focus (low entropy = position-specific) | Circuit and non-circuit heads have similar entropy; no attentional distinction |
| E6b (CKA Cross-Layer) | Representations preserved across circuit layers; incremental processing | Radical transformation between circuit layers; representations not preserved |

### Cross-metric triangulation

The strongest evidence comes from convergent findings across representational metrics that probe different properties:

- **E03 + E92**: RSA shows circuit layers encode task-relevant similarity, CKA shows the circuit subnetwork captures the full model's geometry. Together, they establish that the circuit both represents the task structure and does so in a way that is aligned with the full model's computation.
- **R1 + R3**: Probe decodability with selectivity (R1) shows task information is genuinely encoded; causal representation (R3) shows it is load-bearing. R1 without R3 leaves open whether the information is actually used. R3 without R1 leaves open whether the information is specific to the task.
- **E02 + E03**: Linear probe (E02) shows where information becomes decodable; RSA (E03) shows where representational geometry matches the task. If both peak at the same layers, those layers both contain and organize task information.
- **E11 + E92**: Attention entropy (E11) characterizes the QK circuit (what each head attends to); CKA (E92) characterizes the overall representational alignment. A circuit with focused attention (low entropy) and high CKA has both selective input gathering and faithful output representation.
- **E6b + R3**: Cross-layer CKA (E6b) shows information is preserved across circuit layers; causal representation (R3) shows the information is causally used. Together, they establish that the circuit incrementally processes load-bearing representations.

## Relationship to Causal and Information-Theoretic Metrics

Representational metrics occupy a middle ground between purely observational (information-theoretic) and interventional (causal) approaches. Linear probes and RSA are observational -- they characterize the geometry of representations without intervening. The causal representation test (R3) is interventional -- it patches activations and observes the effect on output.

The key question representational metrics answer is: **does the circuit create, organize, and use task-relevant representations?** Causal metrics answer whether components are necessary and sufficient. Information-theoretic metrics answer whether components share information. Representational metrics answer whether the shared information has the right geometric structure to support the claimed computation.

The ideal pattern is a chain: information-theoretic metrics show the circuit heads share information (C01), representational metrics show this information has task-relevant geometric structure (E03, R1), and causal metrics show the structure is load-bearing (R3) and necessary (activation patching).
