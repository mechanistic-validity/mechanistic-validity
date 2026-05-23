---
title: "MI Information Metrics"
description: "Information metrics from the mechanistic interpretability lens: mutual information, PID, transfer entropy, and causal discovery."
---

# MI Information Metrics

This page documents the information-theoretic metrics that implement the [mechanistic interpretability lens](/framework/lenses/core/mechanistic-interpretability). These metrics quantify information flow, shared information, and causal structure within circuits using tools from information theory -- mutual information, partial information decomposition, transfer entropy, Granger causality, and structure learning. They are implemented in `mechval_v2.core.mechanistic_interpretability.information` and can be run independently or as part of a protocol.

All metrics in this family use Direct Logit Attribution (DLA) as the per-head scalar summary: the dot product of each head's output (projected through $W_O$) with the correct-minus-incorrect unembedding direction. DLA preserves sign, which carries essential information about whether a head promotes or suppresses the correct answer.

## Metrics

---

### C01 -- Mutual Information (`54_mutual_information.py`)

**What it computes.** Estimates pairwise mutual information between circuit heads' DLA values across prompts using binned MI estimation. Each head's DLA is discretized into 10 equal-frequency quantile bins, and MI is computed from the joint histogram:

$$
I(X; Y) = \sum_{x, y} p(x, y) \log_2 \frac{p(x, y)}{p(x)\, p(y)}
$$

MI is computed for all pairs of circuit heads (within-circuit MI) and between circuit heads and a matched set of random non-circuit heads (between MI). The ratio of mean within-circuit MI to mean between MI quantifies how much more information circuit heads share with each other than with arbitrary heads.

**Evidence family.** Information-theoretic (observational).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `ratio` | Mean within-circuit MI / mean circuit-to-random MI | $> 1.0$ (random baseline) |
| `mean_within_circuit_mi` | Average MI between all circuit head pairs | reported |
| `top_mi_edges` | Highest-MI head pairs (MI-weighted graph structure) | reported |

**What it establishes.** Circuit heads share more information with each other than with random heads. High within-circuit MI indicates functional coupling: the heads' outputs co-vary in a task-relevant way. The MI-weighted graph reveals which head pairs are most informationally coupled, potentially reflecting direct information flow paths.

**What it does not establish.** Directionality or causality. MI is symmetric: $I(X; Y) = I(Y; X)$. High MI between two heads means they share information, but does not indicate which head provides information to which. Use transfer entropy (C03) or Granger causality (C07) for directional evidence. MI also does not distinguish direct from mediated dependencies -- use conditional MI (C02) for that.

**Usage.**
```bash
uv run python 54_mutual_information.py --tasks ioi sva --n-prompts 100
```

---

### C02 -- Conditional Mutual Information (`55_conditional_mi.py`)

**What it computes.** For triplets of circuit heads $(h_1, h_2, h_3)$, computes $I(h_1; h_2 \mid h_3)$ -- the MI between $h_1$ and $h_2$ conditioned on $h_3$. Uses residualization: regresses $h_3$'s DLA out of both $h_1$ and $h_2$'s DLA values, then computes binned MI on the residuals.

If $I(h_1; h_2)$ is high but $I(h_1; h_2 \mid h_3) \approx 0$, then $h_3$ mediates the $h_1$-$h_2$ dependency -- removing $h_3$'s contribution explains away the shared information. For each pair, the metric finds the best mediator among remaining circuit heads and reports the fraction of pairwise MI that is mediated vs direct.

**Evidence family.** Information-theoretic (observational, mediation analysis).

**Key metrics.**
| Metric | Description | Interpretation |
|---|---|---|
| `mean_direct_fraction` | Average fraction of MI that persists after conditioning on the best mediator | High = direct coupling |
| `mean_mediated_fraction` | Average fraction explained away by the best mediator | High = hub-mediated circuit |

**What it establishes.** Whether circuit head dependencies are direct or mediated through hub heads. High mediation ($> 0.5$) indicates a hub-and-spoke architecture where a small number of heads serve as information bottlenecks. High direct fraction indicates parallel, independent information channels.

**What it does not establish.** That conditioning on $h_3$ removes a causal pathway. Residualization removes linear association, which may not correspond to the actual causal mediation mechanism. A head that happens to correlate with both $h_1$ and $h_2$ will appear to mediate even if it plays no causal role.

**Usage.**
```bash
uv run python 55_conditional_mi.py --tasks ioi sva --n-prompts 100
```

---

### C03 -- Transfer Entropy (`53_transfer_entropy.py`)

**What it computes.** Estimates directional information flow between circuit heads across layers. For each directed pair $h_1$ (layer $L_1$) $\to$ $h_2$ (layer $L_2$, $L_2 > L_1$), estimates transfer entropy as the squared partial correlation of $h_1$'s DLA with $h_2$'s DLA, controlling for all circuit heads at layers between $L_1$ and $L_2$:

$$
\text{TE}(h_1 \to h_2) \approx r^2_{\text{partial}}(h_1, h_2 \mid \{h_k : L_1 < L_k < L_2\})
$$

Compares TE for known circuit edges versus non-edges.

**Evidence family.** Information-theoretic (directional, observational).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `ratio` | Mean TE for circuit edges / mean TE for non-circuit edges | $> 1.0$ (random baseline) |
| `mean_circuit_te` | Average TE proxy for known circuit edges | reported |
| `mean_non_circuit_te` | Average TE proxy for non-circuit head pairs | reported |

**What it establishes.** Circuit edges carry more directional information flow than non-edges. A ratio substantially above 1.0 means the circuit's claimed edge structure reflects genuine information transfer: earlier heads provide information that later heads use, beyond what intervening heads already provide.

**What it does not establish.** True causal information flow. Partial correlation is a linear proxy for transfer entropy, which is itself an observational (not interventional) measure. Two heads can show high TE because they both respond to the same input feature, not because one informs the other. Combine with path patching or activation patching for causal directional evidence.

**Usage.**
```bash
uv run python 53_transfer_entropy.py --tasks ioi sva --n-prompts 100
```

---

### C04 -- Partial Information Decomposition (`08_pid.py`)

**What it computes.** Decomposes the mutual information between pairs of circuit heads and the model output into four components:

- **Redundancy**: information that both heads provide about the output (overlapping).
- **Unique $h_1$**: information only $h_1$ provides.
- **Unique $h_2$**: information only $h_2$ provides.
- **Synergy**: information that neither head provides individually but both provide jointly.

Uses the BROJA PID implementation from the `dit` library when available, falling back to a binned approximation. DLA values are quantized into 5 equal-frequency bins.

**Evidence family.** Information-theoretic (multivariate decomposition).

**Key metrics.**
| Metric | Description | Interpretation |
|---|---|---|
| `mean_synergy` | Average synergistic information across head pairs | High = cooperative circuit |
| `mean_redundancy` | Average redundant information across head pairs | High = robust, fault-tolerant circuit |
| Per-pair decomposition | Full PID for each head pair | Identifies which pairs cooperate vs overlap |

**What it establishes.** Whether circuit heads carry complementary (synergistic) or overlapping (redundant) information about the output. High synergy means the circuit is a genuine computational unit: pairs of heads jointly encode information that neither encodes alone. High redundancy means the circuit is fault-tolerant but potentially over-specified.

**What it does not establish.** The causal structure underlying the decomposition. PID quantifies the information structure but not the mechanism by which synergy or redundancy arises. Two heads may be synergistic because they compute complementary functions (genuinely cooperative) or because they respond to different aspects of the same confound. Combine with epistasis (GN2 from the genetics lens) for a causal version of interaction analysis.

**Usage.**
```bash
uv run python 08_pid.py --tasks ioi sva --n-prompts 60
```

---

### C05 -- Information Bottleneck Analysis (`57_info_bottleneck.py`)

**What it computes.** For each layer, computes how much information the residual stream retains about the input ($I(X; T_\ell)$) versus how much it preserves about the output ($I(T_\ell; Y)$), where $X$ is the input token identity and $Y$ is the correct/incorrect binary label. Uses residual-stream activations projected onto the top 10 PCA dimensions, with MI estimated via binned approximation on PCA scores. The result is an information plane: $I(X; T)$ vs $I(T; Y)$ across layers.

**Evidence family.** Information-theoretic (information plane analysis).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `mean_circuit_I_T_Y` | Mean $I(T_\ell; Y)$ at layers containing circuit heads | $>$ `mean_non_circuit_I_T_Y` |
| `mean_non_circuit_I_T_Y` | Mean $I(T_\ell; Y)$ at non-circuit layers | baseline |
| `info_plane` | Full $I(X; T)$ and $I(T; Y)$ at each layer | for visualization |

**What it establishes.** Circuit-critical layers preserve more task-relevant information ($I(T; Y)$) than non-circuit layers. This is consistent with the information bottleneck principle: the network compresses irrelevant input information while preserving task-relevant structure, and the circuit heads are located at layers where this task-relevant information is highest.

**What it does not establish.** That circuit heads are responsible for the high $I(T; Y)$. The residual stream at a layer reflects all computations up to that point, not just the circuit heads at that layer. The layer-level analysis conflates circuit and non-circuit contributions to the residual stream.

**Usage.**
```bash
uv run python 57_info_bottleneck.py --tasks ioi sva --n-prompts 80
```

---

### C06 -- O-Information (`58_o_information.py`)

**What it computes.** Computes the O-information (Rosas et al., 2019) of circuit head DLA values, a multivariate measure that captures the overall balance between redundancy and synergy in a set of variables:

$$
\Omega = (n - 2) \cdot H(X_1, \ldots, X_n) + \sum_i H(X_i) - \sum_i H(X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_n)
$$

Positive $\Omega$ indicates redundancy-dominated: heads carry overlapping information. Negative $\Omega$ indicates synergy-dominated: heads carry information jointly that no subset carries alone. Compares $\Omega$ for the circuit head set vs random subsets of the same size to test whether the circuit is specifically synergistic or redundant.

**Evidence family.** Information-theoretic (multivariate redundancy/synergy).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `omega_circuit` | O-information for circuit heads | compared to random |
| `omega_random_mean` | Mean O-information for random head subsets of the same size | random baseline |
| `z_score` | $(\Omega_{\text{circuit}} - \mu_{\text{random}}) / \sigma_{\text{random}}$ | significance |
| `interpretation` | "redundancy" ($\Omega > 0$) or "synergy" ($\Omega < 0$) | qualitative |

**What it establishes.** Whether the circuit as a whole is organized around redundancy (fault tolerance, overlapping representations) or synergy (cooperative computation, emergent joint encoding). A $z$-score substantially different from zero means the circuit's information structure is non-random -- it is specifically more synergistic or redundant than an arbitrary subset of heads.

**What it does not establish.** Which heads drive the synergy or redundancy. O-information is a summary statistic for the entire set; it does not identify specific synergistic or redundant pairs. Use PID (C04) for pairwise decomposition.

**Usage.**
```bash
uv run python 58_o_information.py --tasks ioi sva --n-prompts 100
```

---

### C07 -- Granger Causality (`56_granger_causality.py`)

**What it computes.** Treats the sequence of head activations across layers as a "time series" (layer = time). For each pair of circuit heads $h_1$ (layer $L_1$) and $h_2$ (layer $L_2$, $L_2 > L_1$), tests whether $h_1$'s DLA Granger-causes $h_2$'s DLA: does adding $h_1$'s DLA improve prediction of $h_2$'s DLA beyond all other circuit heads at earlier layers? Uses an F-test comparing the restricted model (all earlier heads except $h_1$) to the full model (restricted + $h_1$).

**Evidence family.** Information-theoretic / causal (Granger causality).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `circuit_significance_rate` | Fraction of circuit edges that are Granger-significant at $\alpha = 0.05$ | compared to non-circuit rate |
| `non_circuit_significance_rate` | Fraction of non-circuit edges that are Granger-significant | baseline |
| `top_significant` | Top Granger-significant edges by F-statistic | reported |

**What it establishes.** Circuit edges show Granger-causal relationships at a higher rate than non-circuit edges. Granger causality -- whether adding $h_1$ improves prediction of $h_2$ beyond the other predictors -- is a well-established statistical test for directed functional coupling. If circuit edges are preferentially Granger-significant, the circuit's edge structure reflects genuine predictive information flow.

**What it does not establish.** True causality. Granger causality is a statistical (observational) concept: $h_1$ Granger-causes $h_2$ if $h_1$ contains unique predictive information about $h_2$. This can occur due to confounding (both respond to a common upstream signal with different delays). In transformers, all earlier-layer outputs are available to later layers via the residual stream, so Granger causality may reflect shared input features rather than direct information flow.

**Usage.**
```bash
uv run python 56_granger_causality.py --tasks ioi sva --n-prompts 100
```

---

### C08 -- Observational Circuit Discovery / oCSE (`07_ocse.py`)

**What it computes.** Two complementary purely observational discovery methods:

1. **Stability selection** via bootstrap LassoCV: runs 50 bootstrap LassoCV regressions predicting logit-diff from all 144 head DLAs, selects heads that appear in $> 50\%$ of bootstrap runs. Handles multicollinearity through L1 regularization and bootstrap averaging.
2. **Greedy oCSE** (observational Causal Subgraph Extraction): greedy forward selection using conditional mutual information with a permutation-calibrated threshold (95th percentile of max CMI under row permutation).

Both use DLA features (signed head contribution to logit diff) rather than norms, since sign carries essential information about head function.

**Evidence family.** Information-theoretic / statistical (observational discovery).

**Key metrics.**
| Metric | Description | Interpretation |
|---|---|---|
| `f1` | F1 score of stability selection against known circuit | primary metric |
| `precision` / `recall` | Precision and recall of discovered heads | reported |
| `ocse.f1` | F1 of oCSE forward selection | secondary metric |
| `combined.f1` | F1 of union of both methods | convergent discovery |

**What it establishes.** Circuit heads can be recovered from purely observational data -- without any interventions. Stability selection identifies heads whose DLA values are robust predictors of logit-diff across bootstrap samples. oCSE identifies heads whose DLA provides incremental conditional information. Agreement between the two methods (combined F1) provides convergent validity.

**What it does not establish.** Causal necessity. Observational discovery identifies heads that are statistically predictive, not causally necessary. A head that correlates with logit-diff but is not causally load-bearing will be discovered by both methods. Combine with activation patching or ablation for causal validation of observationally-discovered circuits.

**Usage.**
```bash
uv run python 07_ocse.py --tasks ioi sva --n-prompts 200
```

---

### C09 -- NOTEARS Structure Learning (`09_notears.py`)

**What it computes.** Learns a directed acyclic graph (DAG) over component activations using NOTEARS (Zheng et al., NeurIPS 2018), which reformulates the combinatorial DAG constraint as a continuous optimization:

$$
\min_{W} \frac{1}{2n} \|X - XW\|_F^2 + \lambda_1 \|W\|_1 \quad \text{s.t.} \quad \text{tr}(e^{W \circ W}) - d = 0
$$

where the acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ is enforced via augmented Lagrangian. The discovered DAG's parents of the output node are compared against known circuit heads via F1, precision, and recall. A permutation baseline (shuffled data) calibrates the false-positive rate.

**Evidence family.** Information-theoretic / causal (continuous structure learning).

**Key metrics.**
| Metric | Description | Baseline |
|---|---|---|
| `f1` | F1 of NOTEARS-discovered parents against known circuit | primary metric |
| `precision` / `recall` | Precision and recall of discovered causal parents | reported |
| `baseline_random` | Number of parents discovered from permuted data | false-positive calibration |

**What it establishes.** A DAG recovered by continuous optimization -- without prior assumptions about circuit structure -- identifies circuit heads as causal parents of the output. NOTEARS discovers which heads causally precede the output, enforcing acyclicity as a structural constraint. This is stronger than undirected MI or correlation: the DAG has directionality.

**What it does not establish.** That the learned DAG reflects the true causal structure. NOTEARS assumes a linear structural equation model, which may not hold for transformer computations. The DAG is learned from observational data only and is subject to the same confounding concerns as Granger causality, though the acyclicity constraint partially mitigates this by imposing structural consistency.

**Usage.**
```bash
uv run python 09_notears.py --tasks ioi sva --n-prompts 80
```

---

## Reading the Scores

### Metric-level interpretation

| Metric | High score | Low score |
|---|---|---|
| C01 (MI) | Circuit heads share more information than random heads; functional coupling | No preferential coupling; circuit heads are informationally independent |
| C02 (CMI) | High direct fraction = parallel channels; high mediated = hub architecture | Mixed or trivial MI; not enough signal to decompose |
| C03 (Transfer Entropy) | Circuit edges carry directional information; sender informs receiver | Non-circuit edges carry as much TE; circuit edges are not privileged |
| C04 (PID) | High synergy = cooperative circuit; high redundancy = fault-tolerant | Low synergy and redundancy; heads contribute independently |
| C05 (Info Bottleneck) | Circuit layers have high $I(T; Y)$; task info peaks at circuit layers | Task info is uniformly distributed or peaks outside circuit layers |
| C06 (O-Information) | Circuit is specifically synergistic or redundant vs random | Circuit has the same information structure as random head subsets |
| C07 (Granger) | Circuit edges are preferentially Granger-significant | Granger significance does not discriminate circuit from non-circuit |
| C08 (oCSE) | Observational discovery recovers circuit heads; high F1 | Discovery identifies different heads; circuit may be non-predictive |
| C09 (NOTEARS) | Structure learning recovers circuit as DAG parents | DAG parents do not match circuit; structure may be non-linear |

### Cross-metric triangulation

The strongest evidence comes from convergent findings across information-theoretic methods that probe different properties:

- **C01 + C04**: High within-circuit MI (C01) with high synergy (C04) indicates the circuit is both informationally coupled and cooperatively computational. If MI is high but synergy is low, the coupling is redundant (overlapping, not joint).
- **C03 + C07**: Transfer entropy (C03) and Granger causality (C07) both test directional information flow but use different statistical methods (partial correlation vs F-test). Convergence provides robust evidence of directed functional coupling along circuit edges.
- **C02 + C06**: Conditional MI (C02) identifies hub heads; O-information (C06) identifies overall synergy/redundancy balance. A synergy-dominated circuit (C06) with strong mediation (C02) suggests a few hub heads orchestrate cooperative computation.
- **C08 + C09**: oCSE (C08) discovers via forward selection; NOTEARS (C09) discovers via continuous DAG optimization. If both identify the same circuit heads from observational data alone, the circuit is robustly recoverable without interventions.

## Relationship to Causal Metrics

Information-theoretic metrics are observational by nature: they measure statistical relationships in activation patterns without intervening. This makes them complementary to -- not substitutes for -- causal metrics (activation patching, ablation, causal scrubbing). The information-theoretic metrics answer "do these heads share information?" while causal metrics answer "does this head's information causally affect the output?"

The ideal pattern is convergent: heads identified by MI, TE, and Granger causality should also be identified by activation patching and ablation. Divergence is informative: a head with high MI but low activation patching effect suggests its information is redundant (available elsewhere), while a head with low MI but high patching effect suggests it carries unique, sparse information.
