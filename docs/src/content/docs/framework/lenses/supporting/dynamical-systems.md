---
title: "Dynamical Systems"
description: "The structural dynamics lens: does the circuit have the geometric and spectral structure of a stable, identifiable dynamical system?"
---

# The Dynamical Systems Lens

This lens asks one question: **does the circuit have the geometric and spectral structure of a stable, identifiable dynamical system?**

When we describe a transformer circuit as "implementing" a computation, we are implicitly making a dynamical claim: information enters the circuit in early layers, is transformed through intermediate layers, and produces an output at later layers. The sequence of layers is a discrete-time dynamical system, and the activations at each layer are the state. Dynamical systems theory provides a century of tools for characterizing such systems — spectral decomposition, topological invariants, attractor identification, renormalization — that are orthogonal to the ablation-based tools of neuroscience and the dose-response curves of pharmacology. These tools characterize the *structure* of a computation without intervening on it.

The key insight is that a Koopman decomposition of layer-to-layer dynamics identifies the modes — eigenvalues and eigenfunctions — that govern how information propagates through the circuit. If those dynamics are well-captured by a low-rank linear operator, the computation is identifiable: it can be described by a small number of parameters rather than requiring a full specification of all component interactions. If the dominant modes further align with the claimed computation (e.g., a mode that tracks subject-verb number through layers), the structural evidence supports the causal evidence from other lenses. A "circuit" whose dynamics are full-rank noise does not implement a computation in any principled sense, regardless of what ablation studies show.

There is a disanalogy worth naming. Physical dynamical systems evolve in continuous time with smooth trajectories. Transformer layers are discrete, nonlinear, and include residual connections that make every layer's output a perturbation of the input rather than a wholesale state transition. The "dynamics" are better understood as iterative refinement than as a trajectory through state space. Some dynamical concepts — Lyapunov exponents, strange attractors — apply only loosely. Others — spectral decomposition, fixed-point structure, topological persistence, renormalization group flow — transfer directly, because they depend on algebraic and topological properties that do not require continuity or smoothness.

:::note
For the full metrics and protocols reference, see [Dynamical Systems -- Metrics & Protocols](/framework/lenses/supporting/dynamical-systems-metrics).
:::

## Key Distinctions

### Spectral structure vs noise

A dynamical system with identifiable structure has a spectrum — eigenvalues of its linearized dynamics — that separates into a few dominant modes and a noise floor. The dominant modes capture the system's behavior; the noise floor captures irrelevant variation. If a circuit's layer-to-layer dynamics have clear spectral separation, the computation is low-dimensional and potentially interpretable. If the spectrum is flat, the "circuit" may be an artifact of the discovery method — a random subgraph that happens to carry signal.

In MI: Koopman analysis (WC_M8) computes the eigenvalues of the linearized layer-to-layer map restricted to circuit components. A circuit with a few large eigenvalues and many near-zero eigenvalues has identifiable dynamics. A circuit with a flat spectrum does not. The spectral gap — the ratio between the $k$-th and $(k+1)$-th singular values — is a quantitative measure of identifiability. This is the same logic as the scree test in PCA, but applied to the dynamics of information flow rather than to static variance.

### Topological persistence vs fragility

Topological data analysis asks which geometric features of a dataset — connected components, loops, voids — persist across scales. A feature that appears at one filtration parameter and disappears immediately is noise. A feature with a long persistence interval is robust structure. The persistence diagram is a fingerprint of the dataset's geometry that is invariant to smooth deformations: if two point clouds have the same persistence diagram, they have the same topology regardless of differences in coordinates or magnitudes.

In MI: TDA persistence (WC_M5) computes the persistent homology of the activation manifold restricted to circuit components across prompts. Long-lived topological features — a loop in activation space that corresponds to cycling through name positions, a void that separates correct from incorrect completions — are structural invariants of the computation. Short-lived features are noise. The critical test is bootstrap stability: does the persistence diagram remain consistent when we resample prompts? A topological feature that appears in 95% of bootstrap samples is a property of the circuit. One that appears in 40% is a property of the prompt sample.

### Modes vs components

A dynamical mode is a direction in state space along which the system evolves independently (or nearly so) from other directions. A component — a head, a neuron, a layer — is a physical unit. These are different decompositions. A single mode may span many components, and a single component may participate in many modes.

In MI: most circuit analysis operates at the component level (heads, neurons). Dynamical analysis operates at the mode level (eigenvectors of the transition operator). If a circuit's claimed computation aligns with a single dominant mode, the circuit has a principled functional decomposition. If the computation is spread across many modes with similar magnitudes, the component-level description may be misleading — the real structure lives in mode space, not head space. This mirrors the argument by [Gallego et al. (2017)](https://doi.org/10.1146/annurev-neuro-092917-025811) that neural modes, not individual neurons, are the right unit of analysis for motor cortex.

### Fixed points and renormalization

A fixed point of a dynamical system is a state that maps to itself under the dynamics. Near a fixed point, the system's behavior is governed by the linearized dynamics — the Jacobian. Renormalization group (RG) theory extends this by asking which features of the dynamics survive coarse-graining: zooming out to larger scales. Directions in parameter space that grow under coarse-graining are *relevant* (they define the system's universality class); directions that shrink are *irrelevant* (they wash out at large scales). A system with few relevant directions has universal behavior that does not depend on microscopic details.

In MI: renormalization group analysis (WC_M10) asks whether the circuit's behavior is scale-invariant — does the same computational structure appear at different levels of granularity (individual neurons, heads, layers, blocks)? The beta function $\beta_i$ measures how the $i$-th coupling changes under a single coarse-graining step. $|\beta_i| \approx 0$ means the coupling is near a fixed point and the corresponding structural feature is preserved under rescaling. A circuit with many near-fixed-point directions has robust, scale-independent structure. A circuit whose couplings all flow rapidly under coarse-graining is fragile — its structure depends on the exact level of analysis.

## Analytical Constructs

### The Koopman spectrum

The signature artifact of dynamical systems evaluation is the Koopman spectrum: the eigenvalues of the (possibly infinite-dimensional) linear operator that governs the evolution of observables along a nonlinear dynamical system. In practice, this is approximated by Dynamic Mode Decomposition (DMD) applied to the layer-by-layer activations of circuit components.

The spectrum reveals structure that no single ablation can:

- **Dominant eigenvalues** -- modes with $|\lambda| \approx 1$ persist across layers. These are the computational backbone: information that survives from input to output.
- **Decaying eigenvalues** -- modes with $|\lambda| \ll 1$ die out quickly. These carry transient information that does not reach the output.
- **Reconstruction error** -- the fraction of activation variance captured by the top-$k$ modes. Low reconstruction error from few modes means the dynamics are low-dimensional and identifiable.
- **Mode alignment** -- do the dominant modes correspond to task-relevant features (e.g., a mode that tracks the indirect object in IOI)? Alignment between dynamical modes and circuit function is structural evidence for the computational claim.

To construct the spectrum: collect activations at each layer for the circuit's components across a prompt set. Stack them as a time series (layer = time step). Compute DMD. Report eigenvalues, singular value spectrum, reconstruction error, and mode interpretations. The elbow in the singular value spectrum determines $k$, the effective dimensionality of the dynamics.

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Koopman, "Hamiltonian systems and transformations in Hilbert space"](https://doi.org/10.1073/pnas.17.5.315) | 1931 | Mathematics | **Koopman operator** -- nonlinear dynamics lifted to a linear operator on observables; finite-rank approximation implies identifiable structure |
| [Wilson & Kogut, "The renormalization group and the epsilon expansion"](https://doi.org/10.1016/0370-1573(74)90023-4) | 1974 | Physics | **Renormalization group** -- universality classes, relevant vs irrelevant operators, scale invariance near fixed points |
| [Edelsbrunner et al., "Topological persistence and simplification"](https://doi.org/10.1007/s00454-002-2885-2) | 2002 | Computational Topology | **Persistent homology** -- multi-scale topological features ranked by persistence; long-lived features are structure, short-lived are noise |
| [Carlsson, "Topology and data"](https://doi.org/10.1090/S0273-0979-09-01249-X) | 2009 | Applied Mathematics | **TDA for data analysis** -- persistent homology as a robust, coordinate-free descriptor of shape in high-dimensional data |
| [Schmid, "Dynamic mode decomposition of numerical and experimental data"](https://doi.org/10.1017/S0022112010001217) | 2010 | Fluid Dynamics | **DMD** -- data-driven spectral decomposition of spatiotemporal data into coherent modes with associated growth rates and frequencies |
| [Brunton et al., "Discovering governing equations from data"](https://doi.org/10.1073/pnas.1517384113) | 2016 | Data-Driven Dynamics | **SINDy** -- sparse identification of nonlinear dynamics; low-rank structure in the dynamics implies identifiability and parsimony |
| [Mezic, "Spectrum of the Koopman operator, spectral expansions in functional spaces, and state-space geometry"](https://doi.org/10.1007/s00332-019-09598-5) | 2020 | Applied Mathematics | **Koopman spectral theory** -- eigenfunction expansion connects spectral properties to geometric invariants of the state space |

## Validity type: [Construct validity](/framework/validity-types/construct)

> **Structural identifiability (Ljung 1999):** A system is structurally identifiable if its parameters can be uniquely determined from input-output data. A circuit claim that does not specify what structural properties the circuit has -- beyond "ablating it changes the output" -- is making an unidentifiable claim.

A set of heads that collectively influence a task output is not, by itself, a circuit in any meaningful dynamical sense. It is a set. To call it a circuit is to claim that the components have coordinated structure -- that information flows through them in a specific pattern, that the pattern is stable, and that the pattern corresponds to the claimed computation. Dynamical systems theory provides the tools to test whether this structural claim holds: spectral decomposition tests whether the dynamics are low-dimensional, topological persistence tests whether the geometry is robust, and renormalization tests whether the structure survives changes in granularity.

The gap between "a set of causally relevant components" and "a structured computation" is the reason this lens adds criteria to construct validity. This lens also contributes one criterion (M7, topological persistence) to measurement validity, because the stability of the activation manifold's topology is a property of the measurement's reliability across prompt samples rather than a property of the circuit's computational role.

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| C6 | Spectral identifiability | Does the circuit's layer-to-layer dynamics have a low-rank spectral decomposition? | Construct |
| C7 | Mode-function alignment | Do the dominant dynamical modes correspond to the claimed computational function? | Construct |
| M7 | Topological persistence | Does the circuit's activation geometry have persistent topological features stable across prompts? | Measurement |
| C8 | Scale invariance | Is the circuit's computational structure preserved under coarse-graining? | Construct |

### C6 -- Spectral identifiability

The circuit's layer-to-layer dynamics, restricted to its components, should be well-approximated by a low-rank linear operator. This is the most fundamental structural test: does the computation have identifiable dynamics, or is it full-rank noise?

**What it establishes.** The computation is low-dimensional: a few modes capture most of the variance in how information transforms across layers. This makes the circuit's dynamics identifiable -- they can be described by a small number of parameters rather than requiring a full specification of all component interactions. A low-rank Koopman decomposition is evidence that the circuit has coherent internal structure, independent of whether that structure has been causally validated.

**What it does not establish.** That the identified modes correspond to the claimed computation (that is C7), that the circuit is causally necessary (that is I1), or that the low-rank structure is unique to this circuit rather than a generic property of the model's architecture.

**Threshold.** Koopman reconstruction error $\leq 0.15$ with $k \leq 5$ modes, where $k$ is chosen by the elbow method on the singular value spectrum.

**Minimum reporting.**
- Singular value spectrum with elbow marked
- Reconstruction error as a function of number of modes retained
- Eigenvalue magnitudes of the top-$k$ DMD modes
- Comparison to a size-matched random component set (to establish that low-rank structure is a property of the circuit, not of the model in general)

### C7 -- Mode-function alignment

The dominant dynamical modes should correspond to task-relevant features. C6 establishes that the dynamics are low-rank; C7 establishes that the low-rank structure is about the right thing.

**What it establishes.** The structural decomposition of the circuit's dynamics aligns with its claimed function. The modes are not just any low-rank structure -- they capture the specific information (subject identity, verb number, indirect object, token class) that the circuit is supposed to process. This is the dynamical analog of construct validity criterion C2 (structural plausibility): the internal structure of the circuit is consistent with the claimed computation.

**What it does not establish.** That the circuit is the only mechanism with this structure, or that the modes are causally responsible for the output. A mode can track a task-relevant feature without being on the causal path -- it may be an epiphenomenal correlate. Causal relevance requires intervention evidence from the neuroscience lens.

**Threshold.** The top-$k$ modes (from C6) should predict the task-relevant feature (e.g., correct vs incorrect token identity, grammatical number, name identity) with $R^2 \geq 0.5$ in a linear probe on held-out prompts.

**Minimum reporting.**
- Mode interpretations: what each dominant mode tracks, established by probing or by inspecting the mode's projection onto known features
- Linear probe $R^2$ from mode activations to task feature, with confidence interval
- Comparison to linear probe $R^2$ from raw activations (to establish that the modes are not simply re-expressing information already available without dynamical decomposition)

### M7 -- Topological persistence

The circuit's activation manifold should have persistent topological features that are stable across prompt variations. This criterion belongs to measurement validity because it characterizes the robustness of the geometric measurement itself -- whether the activation topology is a stable property of the circuit or an artifact of the particular prompt sample.

**What it establishes.** The geometric structure of the circuit's representations is robust -- not an artifact of particular prompt choices or random variation. Persistent features are structural invariants of the computation. Their stability across bootstrap resamples means the measurement of circuit geometry is reliable in the sense of classical test theory: the observed topology is close to the true topology.

**What it does not establish.** That the persistent features are causally relevant to the computation, or that they correspond to interpretable concepts. Topology is shape without semantics. A persistent loop in activation space is robust structure, but knowing that it corresponds to the circuit's claimed function requires separate evidence (C7 or intervention studies).

**Threshold.** At least 2 topological features (connected components beyond the first, 1-cycles, or higher-dimensional features) with persistence ratio $\geq 3\times$ the median persistence. Bootstrap stability: bottleneck distance between resampled persistence diagrams $\leq 0.2 \times$ maximum persistence, computed over at least 200 bootstrap samples.

**Minimum reporting.**
- Persistence diagram with persistent features highlighted
- Persistence ratios for the top features
- Bottleneck distances across bootstrap samples, with 95th percentile reported
- Comparison to a shuffled baseline (permuting prompt-component associations) to establish that persistence is a property of the circuit's geometry, not of the ambient dimensionality

### C8 -- Scale invariance

The circuit's computational structure should be preserved when components are grouped at coarser granularity. This is the strongest structural criterion: it asks whether the circuit description is robust to the analyst's choice of decomposition level.

**What it establishes.** The claimed computation is not an artifact of the level of analysis. If grouping individual neurons into heads, or heads into layers, preserves the functional relationship between circuit components, the circuit description is robust to parcellation choices. Near-fixed-point behavior under renormalization means the circuit has universal structure that does not depend on microscopic details of the decomposition.

**What it does not establish.** That the finest-grained description is correct, or that the circuit generalizes to other models. Scale invariance within a model is a weaker claim than cross-architecture generalization (E6).

**Threshold.** Renormalization-group beta function: at least 2 directions with $|\beta_i| < 0.1$ (near-fixed-point behavior). The fraction of relevant directions (those whose couplings grow under coarse-graining) should be $\leq 0.3$. These thresholds are computed across at least 2 coarse-graining steps (e.g., neuron $\to$ head $\to$ layer, or head $\to$ layer $\to$ block).

**Minimum reporting.**
- Beta function values per direction, at each coarse-graining scale
- Fraction of relevant vs irrelevant directions, with the classification threshold stated
- Comparison across at least 2 coarse-graining scales
- Description of the coarse-graining procedure (what is being grouped and how the grouped dynamics are computed)

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Low-rank Koopman + mode alignment (C6 + C7) | Structurally identifiable computation | "The circuit's dynamics are dominated by $k$ modes that track [feature]" |
| Low-rank Koopman without mode alignment (C6 only) | Identifiable but uninterpreted dynamics | "The circuit has low-dimensional dynamics; their functional role is unknown" |
| Persistent topology + causal evidence (M7 + I1) | Robust geometric structure supporting causal claims | "The circuit's activation geometry has persistent [topological feature] consistent with [computation]" |
| Scale invariance + mode alignment (C8 + C7) | Universal, interpretable computation | "The circuit's [feature]-tracking structure is preserved under coarse-graining" |
| Flat spectrum, no persistent features | No identifiable structure | "No evidence of structured dynamics; the component set may not constitute a circuit" |
| Low-rank Koopman in random set | Low-rank structure is architectural, not circuit-specific | "Low-dimensional dynamics are a generic property of this model, not specific to the proposed circuit" |

## Verdicts

- **Proposed to Causally suggestive:** C6 (spectral identifiability) alone does not gate the transition. The dynamical systems lens contributes to construct validity but does not substitute for causal evidence from the neuroscience lens. However, a flat spectrum (C6 failed) is a warning: the proposed circuit may lack coherent internal structure.
- **Causally suggestive to Mechanistically supported:** C7 (mode-function alignment) strengthens the case that the circuit is not just causally relevant but structurally organized around the claimed computation. C7 contributes to the construct validity criteria (C2, structural plausibility) required for this transition.
- **Mechanistically supported to Triangulated:** M7 (topological persistence) and C8 (scale invariance) provide convergent structural evidence from a different evidence family than causal intervention. A circuit that passes causal criteria (I1-I5), interpretive criteria (V1-V5), and dynamical criteria (C6-C8, M7) has been triangulated across lenses that share no methodological assumptions.

## Protocol

For a proposed circuit $C$ and behavior $B$:

1. **Collect activations.** Run the model on a prompt set (at least 200 prompts), recording the activations of circuit components at each layer. Stack these as a spatiotemporal matrix (layers $\times$ components $\times$ prompts).
2. **Koopman/DMD decomposition.** Compute DMD on the layer-to-layer transition. Report singular values, identify the elbow, compute reconstruction error for $k = 1, \ldots, 10$ modes, and report eigenvalue magnitudes. Compare against a size-matched random component set.
3. **Mode probing.** For each of the top-$k$ modes, compute its activation across prompts and fit a linear probe predicting the task-relevant feature. Report $R^2$ with confidence intervals. Compare to a probe on raw (non-decomposed) activations.
4. **Persistent homology.** Compute the persistent homology of the activation manifold (using Vietoris-Rips or alpha complex filtration). Report the persistence diagram, persistence ratios, and bootstrap bottleneck distances over at least 200 resamples. Compare to a shuffled baseline.
5. **Renormalization group flow.** If the circuit spans multiple layers, define at least 2 coarse-graining levels (e.g., head $\to$ layer). Compute the beta function for each coupling direction. Report the fraction of relevant directions and identify near-fixed-point behavior.
6. A skipped step must be named in the verdict.

## Case Studies

- [IOI](/framework/examples/examples/examples-ioi) -- the IOI circuit can be analyzed with this lens via WC_M5 (TDA persistence) and WC_M8 (Koopman DMD) protocols
