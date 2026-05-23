---
title: "Genetics -- Metrics & Protocols"
description: "Reference for all genetics lens metrics (GN1--GN5) and protocols (knockout hierarchy, rescue, Mendelian randomization, dose-response, epistasis)."
---

# Genetics -- Metrics & Protocols

This page documents the concrete metrics and protocols that implement the [genetics lens](/framework/lenses/supporting/genetics). The lens page covers the theoretical framework -- knockout vs knockdown, epistasis vs additivity, forward vs reverse genetics, instruments and exclusion. This page covers the code: what each metric computes, what it establishes, what it does not, and how the protocols bundle metrics into structured evaluations.

## Metrics

The genetics lens provides five standalone metrics (GN1--GN5). Each adapts a specific experimental biology technique to test whether a proposed neural circuit has properties that real genetic pathways exhibit. They are implemented in `mechval_v2.supporting.genetics` and can be run independently or as part of a protocol.

---

### GN1 -- Knock-In (Cross-Task Weight Similarity)

**Biological analog.** Gene knock-in experiments test functional conservation by inserting one organism's gene into another and checking whether it rescues function (Capecchi 1989). The question is whether the gene's role is conserved across contexts.

**What it computes.** For every pair of tasks with known circuits, GN1 extracts flattened weight vectors (concatenated $W_Q$, $W_K$, $W_V$, $W_O$) for each circuit head, then computes pairwise cosine similarity across tasks. For heads that appear in both circuits (same layer and head index), it computes direct similarity. For all other pairs, it computes cross-pair similarity. The reported value is the mean cosine similarity across all pairs.

$$
\text{GN1}(t_a, t_b) = \frac{1}{|P|} \sum_{(h_i, h_j) \in P} \cos\bigl(\mathbf{w}_{h_i},\, \mathbf{w}_{h_j}\bigr)
$$

where $P$ is the set of head pairs (shared heads compared to themselves, plus all cross-task head pairs), and $\mathbf{w}_h = [W_Q^h;\, W_K^h;\, W_V^h;\, W_O^h]$ is the flattened weight vector for head $h$.

**Pass threshold.** Mean cross-task similarity $> 0.3$.

**What it establishes.** Circuit heads performing analogous roles across different tasks share weight structure. This is evidence that the circuit implements a general-purpose computation reused across tasks, not a task-specific hack. High similarity for shared heads (same position in both circuits) is expected; high similarity for non-shared heads suggests functional analogy at the weight level.

**What it does not establish.** That the shared weight structure is causally relevant. Two heads can have similar weights but serve different functions if those functions happen to require similar linear transformations. GN1 is a structural similarity metric, not a causal one -- it should be combined with causal metrics (epistasis, knockout) to establish that the similarity reflects shared function.

**Usage.**
```bash
uv run python GN1_knock_in.py --tasks ioi sva --n-prompts 40
```

---

### GN2 -- Epistasis (Pairwise Interaction Effects)

**Biological analog.** Fisher (1918) defined epistasis as the deviation from additive effects when two genetic loci are disrupted simultaneously. Non-additive interaction reveals functional coupling between pathway components.

**What it computes.** For each task, GN2 computes the logit-difference effect of ablating each circuit head individually (single-head mean ablation), then measures the joint effect of ablating each pair simultaneously. Epistasis is the deviation from additivity:

$$
\epsilon_{ij} = \Delta_{\{i,j\}} - \Delta_i - \Delta_j
$$

where $\Delta_S = \text{LD}_{\text{clean}} - \text{LD}_{\text{ablated}(S)}$ is the logit-difference drop when the set $S$ of heads is mean-ablated. The reported value is the mean absolute epistasis $\langle |\epsilon_{ij}| \rangle$ across all tested pairs (up to 20 pairs, sampled if the circuit is large).

**Pass threshold.** Mean $|\epsilon| > 0.05$.

**What it establishes.** Circuit components interact non-additively. Positive epistasis ($\epsilon > 0$, synergy) means joint ablation is worse than the sum of individual ablations -- the heads cooperate and disrupting either alone leaves the other partially functional. Negative epistasis ($\epsilon < 0$, antagonism/buffering) means joint ablation is milder than expected -- the heads are partially redundant. Either pattern establishes that the circuit is a functional unit with internal coupling, not merely a collection of independently necessary components.

**What it does not establish.** That the interaction is task-specific (the same heads might interact on every task), or that the epistasis is mediated by direct information flow between the components (it could reflect shared dependence on a common upstream signal). GN2 does not distinguish synergy from antagonism at the aggregate level -- inspect the `pair_details` metadata for the sign of each pair's $\epsilon$.

**Reading the metadata.**
- `fraction_synergistic`: the proportion of pairs with $\epsilon > 0$. Values near 1.0 indicate a cooperative circuit; values near 0.0 indicate pervasive buffering.
- `pair_details`: the top-10 pairs by $|\epsilon|$, with individual and joint effects. Look for pairs where $|\epsilon|$ exceeds both individual effects -- these are the strongest functional couplings.

**Usage.**
```bash
uv run python GN2_epistasis.py --tasks ioi --n-prompts 40
```

---

### GN3 -- Chimera (QK/OV Cross-Layer Transplant)

**Biological analog.** Chimera experiments in developmental biology (McLaren & Wilmut 2003) transplant cells or tissues from one organism into another to test whether a component's function is modular -- whether it works in a foreign context. The question is whether sub-circuits are functionally separable.

**What it computes.** For each pair of circuit heads (preferring cross-layer pairs), GN3 creates a chimera by swapping the hook_z outputs: head A receives head B's output, and head B receives head A's output. This mixes the QK circuit (attention pattern selection) of one head with the OV circuit (value processing) of another. Task performance under the chimera is measured as the fraction of clean logit difference preserved:

$$
\text{GN3}_{\text{preservation}} = \frac{\text{LD}_{\text{chimera}}}{\text{LD}_{\text{clean}}}
$$

The reported value is the mean preservation across all tested chimeras (up to 15, sampled if the circuit is large).

**Pass threshold.** Mean chimera preservation $> 0.3$.

**What it establishes.** The QK and OV sub-circuits of different heads are functionally modular -- swapping them preserves partial task performance. High preservation indicates that the heads implement composable sub-computations (attention pattern generation and value transformation) that can be mixed across heads without catastrophic failure. This is evidence that the circuit's internal organization matches the QK/OV decomposition posited by the mathematical framework for transformer circuits (Elhage et al. 2021).

**What it does not establish.** That the QK and OV sub-circuits are fully independent. Partial preservation ($0.3$--$0.7$) is expected -- full preservation ($> 0.9$) would suggest the heads are interchangeable, which would undermine the claim that each head plays a distinct role. Very low preservation ($< 0.1$) suggests the heads' functions are not modular at the QK/OV boundary.

**Usage.**
```bash
uv run python GN3_chimera.py --tasks ioi --n-prompts 40
```

---

### GN4 -- Convergent Evolution (Cross-Task Structural Similarity)

**Biological analog.** Convergent evolution (Conway Morris 2003) occurs when unrelated organisms independently evolve similar solutions to the same environmental challenge -- eyes in vertebrates and cephalopods, wings in birds and bats. Structural similarity arising from independent optimization is evidence that the structure is functionally optimal, not accidental.

**What it computes.** For each pair of tasks, GN4 compares the structural features of their circuits without running any forward passes. It computes four sub-scores:

1. **Head Jaccard**: $|H_a \cap H_b| \;/\; |H_a \cup H_b|$ -- fraction of exact head positions shared.
2. **Layer Jaccard**: $|L_a \cap L_b| \;/\; |L_a \cup L_b|$ -- fraction of layers used by both circuits.
3. **Density cosine**: cosine similarity between layer-density vectors (fraction of heads used per layer).
4. **Size ratio**: $\min(|H_a|, |H_b|) \;/\; \max(|H_a|, |H_b|)$ -- similarity in circuit size.

The composite structural overlap is the unweighted average:

$$
\text{GN4}(t_a, t_b) = \frac{1}{4}\bigl(\text{Head}_J + \text{Layer}_J + \text{Density}_{\cos} + \text{Size}_r\bigr)
$$

**Pass threshold.** Structural overlap $> 0.5$.

**What it establishes.** Circuits for different tasks converge on similar architectural structure. This is evidence that the model reuses a common computational motif across tasks rather than implementing each task with bespoke circuitry. High overlap between unrelated tasks (e.g., IOI and SVA) is more informative than high overlap between related tasks (e.g., IOI and a permuted IOI variant), because convergence from independent functional pressures is stronger evidence of optimality.

**What it does not establish.** That the shared structure performs the same computation in both tasks. Two circuits can occupy the same heads but use them for different functions. GN4 is purely structural -- it should be paired with GN1 (weight similarity) and GN2 (epistasis) to establish that structural overlap reflects functional overlap.

**Usage.**
```bash
uv run python GN4_convergent_evolution.py --tasks ioi sva --n-prompts 40
```

---

### GN5 -- Phylogenetic Tracking (Circuit Formation Across Layers)

**Biological analog.** Developmental biology tracks how structures form over time. Haeckel's recapitulation idea -- that development mirrors evolutionary history -- is oversimplified, but the core insight stands: complex structures emerge progressively through intermediate stages. In genetics, tracking gene expression across developmental stages reveals when each gene becomes active and how the pathway assembles.

**What it computes.** GN5 treats transformer depth as a developmental timeline. For each layer cutoff $\ell \in \{0, \ldots, L-1\}$, it ablates all circuit heads in layers $> \ell$ and measures the fraction of task performance retained. This produces a cumulative formation curve: $y(\ell)$ is the fraction of the clean logit difference preserved when only circuit heads at or below layer $\ell$ are active.

A sigmoid is fit to this curve via grid search:

$$
\hat{y}(\ell) = \frac{1}{1 + \exp\bigl(-k(\ell - \ell_0)\bigr)}
$$

where $\ell_0$ is the formation midpoint (the layer at which 50% of the circuit's function has accumulated) and $k$ controls the steepness. The reported value is the $R^2$ of the sigmoid fit.

**Pass threshold.** Sigmoid $R^2 > 0.8$.

**What it establishes.** The circuit's function accumulates progressively across layers in a pattern well-described by a sigmoid. This is evidence of ordered developmental structure: early layers contribute foundational computation, a critical transition happens near $\ell_0$, and late layers refine the output. A high $R^2$ rules out erratic or non-monotonic formation, which would suggest the circuit description is incomplete or that the components interact in ways that do not respect the layer ordering.

**What it does not establish.** That the sigmoid formation pattern reflects the true causal ordering of circuit components. Ablating all heads above a cutoff is a blunt intervention -- it removes both direct contributions and any downstream amplification. A head at layer 5 might be essential not because it performs computation itself, but because it provides input that a head at layer 8 requires. The formation curve conflates these cases. Path patching (from the Mendelian randomization protocol) provides finer-grained ordering information.

**Reading the metadata.**
- `formation_midpoint`: the layer at which 50% of the circuit's function has accumulated. Early midpoints (relative to model depth) suggest the circuit's core computation happens in early-to-mid layers; late midpoints suggest it depends on later processing.
- `cumulative_curve`: the raw values at each layer. Look for sudden jumps (indicating a critical layer) or plateaus (indicating layers where no circuit head contributes).

**Usage.**
```bash
uv run python GN5_phylogenetic_tracking.py --tasks ioi --n-prompts 40
```

---

## Protocols

Protocols bundle multiple metrics and calibrations around a specific validity question. Each protocol runs its constituent metrics, applies calibrations (structural checks that contextualize the metric scores), and returns a `ProtocolResult` with scored measurements. The genetics lens provides four main protocols.

### MB_KH -- Knockout Hierarchy

**Question.** Do circuit components show graded necessity across intervention strengths?

**Biological analog.** Complete gene knockout (homozygous null), partial knockdown (RNAi, CRISPRi), and conditional knockout (Cre/lox tissue-specific deletion) are qualitatively different interventions. If all three implicate the same gene, the gene's necessity is robust. If they diverge, the gene's role depends on dosage or context.

**Metrics.**

| Metric | MI analog | Genetics analog | Threshold |
|---|---|---|---|
| `activation_patching` | Full ablation per head | Complete knockout | $> 0.7$ |
| `sigma_ablation` | Graded noise injection | Partial knockdown (RNAi) | $> 0.5$ |
| `role_ablation` | Role-specific removal | Conditional knockout (Cre/lox) | $> 0.3$ |

**Criteria strengthened.** I1 (necessity), E7 (allelic dose-response).

**What it establishes.** The circuit components are necessary across a range of intervention strengths and types. If all three metrics implicate the same components, the necessity finding is robust to methodological choice. If activation patching identifies a component but sigma ablation does not, the component may be necessary only at full removal -- its contribution is either binary or compensated at partial disruption.

**What it does not establish.** Sufficiency (these are all necessity tests), or that the three interventions probe qualitatively different aspects of the component (they may all reduce to "remove information" at different intensities).

**Usage.**
```bash
uv run python knockout_hierarchy.py --tasks ioi induction --device cuda
```

---

### MB_RE -- Rescue Experiments

**Question.** Does re-expressing a knocked-out component rescue function?

**Biological analog.** The knockout-rescue paradigm is the gold standard in molecular biology. Knockout shows the gene is necessary. Rescue -- re-introducing the gene (or a functional homolog) into the knockout background -- shows the deficit was specifically caused by the gene's absence, not by collateral developmental damage. The combination establishes that the gene is both necessary and specifically sufficient to restore function.

**Metrics.**

| Metric | MI analog | Genetics analog | Threshold |
|---|---|---|---|
| `corrupt_restore` | Corrupt then restore activations | Knockout then transgenic rescue | $> 0.7$ |
| `corrupt_restore_behavioral` | Task performance recovery | Behavioral/phenotypic rescue | $> 0.6$ |
| `das_iia` | Interchange intervention accuracy | Cross-locus complementation | $> 0.7$ |

**Criteria strengthened.** I2 (sufficiency), I7 (rescue reversibility).

**What it establishes.** The behavioral deficit caused by corrupting a circuit component is specifically reversible by restoring that component's clean activation. This rules out cascading distributional disruption as the primary cause of the observed ablation effect. If `corrupt_restore` passes but `corrupt_restore_behavioral` fails, the activation is restored but the downstream computation does not recover -- suggesting the corruption caused irreversible damage at downstream sites. If `das_iia` passes, the circuit admits a valid causal abstraction: an intervention at the proposed causal variable produces the predicted counterfactual behavior.

**What it does not establish.** That the rescued component is the unique route for the computation. Other components might also rescue function if activated appropriately. Rescue establishes specific reversibility, not exclusivity.

**Usage.**
```bash
uv run python rescue_experiment.py --tasks ioi --device cuda
```

---

### MB_MR -- Mendelian Randomization

**Question.** Can upstream variables serve as valid instruments for estimating the circuit's causal effect?

**Biological analog.** Mendelian randomization (Davey Smith & Hemani 2014) uses genetic variants as natural instruments: because genotype is assigned at conception (before postnatal confounders can act), the genotype-phenotype association is unconfounded if the exclusion restriction holds (the variant affects the outcome only through the gene it regulates). Two-sample MR estimates the effect in one population and validates in another. Cross-species generalization tests whether the causal relationship holds in a different organism.

**Metrics.**

| Metric | MI analog | Genetics analog | Threshold |
|---|---|---|---|
| `path_patching` | Position-specific causal paths | Instrument validity (exclusion restriction) | $> 0.5$ |
| `cross_task_transfer` | Train on one task, test on another | Two-sample MR (cross-population) | $> 0.4$ |
| `cross_model_invariance` | Same circuit across models | In vivo generalization (cross-species) | $> 0.6$ |

**Criteria strengthened.** I9 (instrument validity), E5 (robustness), E6 (cross-architecture).

**What it establishes.** `path_patching` tests whether upstream activations affect the output only through the proposed circuit (the exclusion restriction). A passing score means the circuit mediates the upstream signal rather than merely correlating with it. `cross_task_transfer` tests whether a circuit discovered on one task transfers to another -- the two-sample MR analog. If it transfers, the circuit captures a computation that generalizes beyond the discovery distribution. `cross_model_invariance` tests whether the same circuit structure is causal across different models -- if the causal relationship holds in multiple "organisms," it is more likely to reflect a genuine computational mechanism than a model-specific artifact.

**What it does not establish.** That the exclusion restriction holds perfectly. Path patching tests specific paths but cannot rule out all possible confounding routes through the residual stream. The F-statistic and overidentification diagnostics (when multiple instruments are available) provide quantitative bounds on instrument validity but are not proof.

**Usage.**
```bash
uv run python mendelian_randomization.py --tasks ioi sva --device cuda
```

---

### MB_DR -- Dose-Response

**Question.** Does the circuit show monotonic dose-response to graded intervention?

**Biological analog.** The Hill equation (Hill 1910) describes the relationship between ligand concentration and receptor binding. A monotonic dose-response curve -- where increasing the dose produces increasing effect -- is the basic pharmacological evidence that a drug acts through a specific target. Non-monotonic response (hormesis) suggests complex regulatory feedback. The Hill slope describes cooperativity: steep slopes indicate all-or-nothing switching.

**Metrics.**

| Metric | MI analog | Genetics analog | Threshold |
|---|---|---|---|
| `dose_response` | Graded intervention strength | EC50 / Hill curve | protocol-specific |
| `sigma_ablation` | Noise titration at multiple sigma | Allelic series (graded severity) | $> 0.5$ |
| `effect_size` | Cohen's d of intervention | Effect magnitude | protocol-specific |

**Criteria strengthened.** E2 (graded response), E7 (allelic dose-response), I1 (necessity).

**What it establishes.** The circuit's behavioral contribution degrades monotonically as intervention strength increases. A monotonic dose-response curve means the finding is not an artifact of a specific intervention strength -- it holds across the full range from mild perturbation to complete ablation. The EC50 (the intervention strength at which half the behavioral effect is observed) characterizes the circuit's sensitivity. Steep Hill slopes indicate the circuit operates as a switch; shallow slopes indicate graded, proportional contribution.

**What it does not establish.** That the dose-response curve has a pharmacological interpretation beyond analogy. Neural circuits do not have binding sites or dissociation constants in the literal sense. The Hill equation is a convenient parametric form for describing graded effects, not a mechanistic model of the circuit's internal dynamics.

**Usage.**
```bash
uv run python dose_response.py --tasks ioi --device cuda
```

---

## Reading the Scores

### Metric-level interpretation

| Metric | High score | Low score |
|---|---|---|
| GN1 (Knock-In) | Heads share weight structure across tasks; reusable computation | Task-specific weight specialization; no cross-task conservation |
| GN2 (Epistasis) | Strong non-additive interactions; circuit is a functional unit | Additive contributions; heads are independently necessary, not coupled |
| GN3 (Chimera) | QK/OV sub-circuits are modular and composable | Sub-circuits are tightly entangled; swapping destroys function |
| GN4 (Convergent Evolution) | Different tasks produce structurally similar circuits | Task-specific circuit architecture; no convergent structure |
| GN5 (Phylogenetic Tracking) | Orderly sigmoid formation across layers | Erratic or non-monotonic accumulation; circuit description may be incomplete |

### Protocol-level interpretation

| Pattern | What it means |
|---|---|
| MB_KH passes, MB_RE fails | Components are necessary at all strengths but deficits are not specifically reversible -- cascading disruption likely |
| MB_RE passes, MB_KH partial | Rescue works but the component is dispensable under partial knockdown -- graded redundancy |
| MB_MR all pass | Strong causal mediation evidence: upstream signals flow through the circuit, effect transfers across tasks and models |
| MB_MR `path_patching` passes, `cross_task_transfer` fails | Circuit mediates upstream signal but is task-specific -- not a general-purpose computation |
| MB_DR monotonic | Method-robust finding; dose-response curve can be characterized by EC50 |
| MB_DR non-monotonic | Complex dynamics: possible compensatory mechanisms, hormesis analog, or circuit description is incomplete |

### Cross-metric triangulation

The strongest evidence comes from convergent findings across metrics that probe different properties:

- **GN1 + GN4**: weight similarity and structural overlap. If both are high, the circuits share both architecture and parameterization -- strong evidence of a shared computational motif.
- **GN2 + GN3**: epistasis and modularity. If GN2 shows synergy and GN3 shows preservation, the circuit has both internal coupling and modular sub-structure -- it is organized, not monolithic.
- **GN5 + MB_KH**: developmental ordering and graded necessity. If the formation curve is sigmoidal and knockout hierarchy is consistent, the circuit assembles progressively and each stage is necessary.
- **MB_RE + MB_MR**: rescue and instrument validity. If rescue succeeds and the exclusion restriction holds, the circuit is both specifically reversible and causally mediating -- the two strongest forms of causal evidence in the genetics toolkit.

## Relationship to Other Lenses

The genetics lens extends the [neuroscience lens](/framework/lenses/core/neuroscience) (dissociation, ablation, sufficiency) with interaction structure (epistasis), reversibility (rescue), and confounding diagnostics (instruments, E-values). It complements the [pharmacology lens](/framework/lenses/core/pharmacology) (dose-response, selectivity) by adding qualitatively different intervention types (allelic series) and instrumental variable methods. It is orthogonal to the [measurement theory lens](/framework/lenses/core/measurement-theory) (reliability, validity of the metrics themselves) and the [geometry lens](/framework/lenses/supporting/geometry) (representational structure in activation space).

See the [main genetics lens page](/framework/lenses/supporting/genetics) for the full theoretical framework, criteria definitions (I6--I10, E7), and case studies.
