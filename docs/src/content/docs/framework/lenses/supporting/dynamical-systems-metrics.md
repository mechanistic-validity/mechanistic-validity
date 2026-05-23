---
title: "Dynamical Systems -- Metrics & Protocols"
description: "Metrics and protocol implementations for the dynamical systems lens: Koopman/DMD, renormalization group, TDA persistence, critical phenomena, and RG coarse-graining."
---

# Dynamical Systems -- Metrics & Protocols

These protocols implement the criteria defined in the [dynamical systems lens overview](/framework/lenses/supporting/dynamical-systems). Each protocol translates a concept from dynamical systems theory, statistical physics, or computational topology into a concrete experimental procedure for evaluating transformer circuit claims.

The dynamical systems lens contributes primarily to [construct validity](/framework/validity-types/construct) (spectral identifiability, mode-function alignment, scale invariance) with a secondary contribution to [measurement validity](/framework/validity-types/measurement) (topological persistence). The protocols below operationalize these criteria as runnable experiments.

---

## WC_M8 -- Koopman Operator / Dynamic Mode Decomposition (`koopman_dmd.py`)

**Lens:** Dynamical Systems | **Validity type:** Construct | **Framework:** Wildcard (Koopman Theory)

### Question

What are the natural dynamical modes of the residual stream as it evolves through layers? A transformer's forward pass is a discrete-time dynamical system: $h_{l+1} = F(h_l)$. The Koopman operator linearizes this nonlinear system by acting on observable functions rather than on states directly. Dynamic Mode Decomposition (DMD) approximates the Koopman operator from data, producing a spectral decomposition of the layer-to-layer dynamics.

The eigenvalues of the DMD operator reveal the circuit's dynamical structure:

- **Eigenvalues near $|\lambda| = 1$** are persistent modes -- information written once and passed through to the output. These are the computational backbone.
- **Eigenvalues with $|\lambda| < 1$** are decaying modes -- transient computations that do not survive to the output layer. These carry intermediate results that are consumed and discarded.
- **Eigenvalues with $|\lambda| > 1$** are growing modes -- information that is amplified through layers. These are dynamically unstable directions.

Comparing the eigenvalue spectra between clean and corrupted conditions reveals which dynamical modes carry task-relevant information: modes whose eigenvalues shift under corruption are task-sensitive; modes whose eigenvalues are invariant carry task-independent structure.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `cka` | Centered Kernel Alignment between activation representations across conditions (clean vs corrupted) | $\geq 0.5$ |
| `effect_size` | Overall circuit importance -- how much the dynamical modes shift between clean and corrupted runs | $\geq 0.8$ |
| `activation_patching` | Component-level causal importance within the Koopman decomposition | $\geq 0.5$ |

### Interpretation

This protocol addresses criterion **C6 (Spectral identifiability)**: does the circuit's layer-to-layer dynamics have a low-rank spectral decomposition? If the singular value spectrum has a clear elbow -- a few dominant modes followed by a noise floor -- the circuit's dynamics are low-dimensional and identifiable. The computation can be described by a small number of parameters rather than requiring full specification of all component interactions.

**High CKA + high effect size** means the circuit's representational structure is consistent across conditions and the dynamical modes are strongly task-relevant. The circuit has identifiable, task-sensitive dynamics.

**Low CKA** means the circuit's representations change substantially between conditions. If the circuit is supposed to implement the same computation regardless of condition, low CKA is a warning -- the circuit's internal structure is condition-dependent.

**Low effect size** means the circuit's dynamical modes are insensitive to the corruption, which could indicate either that the circuit is not involved in the task or that the corruption does not target the circuit's relevant modes. The latter is a limitation of the corruption method, not of the circuit.

The protocol also provides the foundation for **C7 (Mode-function alignment)**: once the top-$k$ modes are identified, their alignment with task-relevant features can be tested via probing (do the modes track subject identity, grammatical number, or other task-specific variables?). The DMD decomposition is the prerequisite for this downstream analysis.

### Theoretical grounding

| Source | Year | Contribution |
|---|---|---|
| Koopman (1931) | 1931 | The Koopman operator on observables -- linearization of nonlinear dynamics |
| Schmid (2010) | 2010 | The DMD algorithm for data-driven spectral decomposition |
| NeurIPS 2025, "Replacing nonlinear MLP layers with Koopman operators" | 2025 | Finite-dimensional Koopman for neural networks |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python koopman_dmd.py                       # all tasks, CPU
uv run python koopman_dmd.py --device cuda          # GPU
uv run python koopman_dmd.py --tasks ioi induction  # specific tasks
```

---

## WC_M10 -- Renormalization Group / Multi-Scale Coarse-Graining (`renormalization_group.py`)

**Lens:** Dynamical Systems | **Validity type:** Construct | **Framework:** Wildcard (Wilson RG)

### Question

Does the residual stream exhibit scale-invariant structure across layers? The renormalization group (RG) procedure coarse-grains adjacent layers and asks what effective dynamics emerge at each scale. This is the physicist's approach to the question "does the circuit's description depend on the level of analysis?"

The procedure works as follows. At the finest scale, the dynamics are described by the full layer-to-layer transition matrices. The RG step groups adjacent layers, computes the effective transition matrix for the grouped block, and identifies which directions in the dynamics survive the coarse-graining. Each direction is characterized by its beta function $\beta_i = \log(|\lambda_i|) / \log(2)$, which classifies it as:

- **Relevant** ($\beta_i > 0$): the coupling grows under coarse-graining. This direction becomes more important at larger scales. Relevant directions define the system's universality class.
- **Irrelevant** ($\beta_i < 0$): the coupling decays under coarse-graining. This direction washes out at larger scales. Irrelevant directions are microscopic details that do not affect macroscopic behavior.
- **Marginal** ($\beta_i \approx 0$): the coupling is near a fixed point. This direction is scale-invariant -- it looks the same regardless of the granularity of analysis.

If two different tasks share fixed-point directions, they share a common computational substrate -- a "universality class" in RG terms. This is structural evidence for shared circuitry that is independent of ablation-based evidence.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `cka` | Representational similarity between activations at different coarse-graining scales | $\geq 0.5$ |
| `cross_task_generalization` | Whether scale-invariant features (fixed-point directions) transfer across tasks | $\geq 0.5$ |
| `effect_size` | Overall circuit importance at each RG scale | $\geq 0.8$ |

### Interpretation

This protocol addresses criterion **C8 (Scale invariance)**: is the circuit's computational structure preserved under coarse-graining?

**High CKA across scales** means the circuit's representational structure is preserved when the analysis granularity changes. The circuit description is not an artifact of the level of analysis -- it captures genuine multi-scale structure.

**High cross-task generalization** means the scale-invariant directions are shared across tasks. The circuit has universal structure: computational features that are not task-specific but reflect a general organizational principle of the model. This is the strongest structural evidence the dynamical systems lens can provide -- it identifies the "universality class" of the circuit's computation.

**Many marginal directions** ($|\beta_i| < 0.1$) indicate the circuit is near a fixed point of the RG flow. Its structure is robust to changes in analysis granularity. **Many relevant directions** indicate the circuit's structure changes qualitatively at different scales -- the fine-grained and coarse-grained descriptions tell different stories. The threshold from the lens overview requires at least 2 directions with $|\beta_i| < 0.1$ and at most 30% relevant directions.

### Theoretical grounding

| Source | Year | Contribution |
|---|---|---|
| Wilson (1971) | 1971 | The RG framework (Nobel Prize 1982) |
| Kadanoff (1966) | 1966 | Block-spin renormalization that inspired Wilson |
| Mehta & Schwab (2014) | 2014 | An exact mapping between the variational RG and deep learning |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python renormalization_group.py                       # all tasks, CPU
uv run python renormalization_group.py --device cuda          # GPU
uv run python renormalization_group.py --tasks ioi induction  # specific tasks
```

---

## PH_RG -- Renormalization Group (`renormalization.py`)

**Lens:** Dynamical Systems | **Validity type:** Construct | **Framework:** Cross-discipline (Physics)

### Question

Does the circuit's description simplify under coarse-graining? Where WC_M10 focuses on the spectral analysis of RG flow (eigenvalues, beta functions, fixed-point structure), PH_RG takes a broader view: it tests whether the circuit exhibits fixed points, scale invariance, and reduced effective dimensionality under the RG transformation.

The key distinction is one of scope. WC_M10 asks "which dynamical modes survive coarse-graining?" PH_RG asks "does the circuit get simpler when you zoom out?" A circuit whose effective dimensionality decreases under coarse-graining has a hierarchical structure -- fine-grained details organize into a smaller number of coarse-grained features. A circuit whose dimensionality stays constant or increases under coarse-graining has no hierarchical organization -- every level of analysis reveals new complexity.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `renormalization` | Composite score for fixed points, scale invariance, and effective dimensionality under RG flow | $\geq 0.5$ |

### Interpretation

The single composite metric captures whether the RG procedure finds structure. A score above 0.5 indicates the circuit exhibits at least some scale-invariant properties -- the description simplifies under coarse-graining in a measurable way. A score below 0.5 indicates the circuit's structure is scale-dependent: the fine-grained description does not predict the coarse-grained one.

This protocol provides complementary evidence to WC_M10. Together, they cover both the spectral and dimensional aspects of scale invariance. When both pass, the circuit has robust multi-scale structure. When WC_M10 passes but PH_RG fails (or vice versa), the scale invariance is partial -- it holds for some aspects of the dynamics but not others.

### Theoretical grounding

| Source | Year | Contribution |
|---|---|---|
| Wilson (1971) | 1971 | Renormalization Group and Critical Phenomena |
| Kadanoff (1966) | 1966 | Scaling Laws for Ising Models near $T_c$ |
| Li et al. (2018) | 2018 | Neural Network Renormalization Group |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python renormalization.py                       # all tasks, CPU
uv run python renormalization.py --device cuda          # GPU
uv run python renormalization.py --tasks ioi induction  # specific tasks
```

---

## WC_M5 -- Topological Data Analysis / Persistence Diagrams (`tda_persistence.py`)

**Lens:** Dynamical Systems | **Validity type:** Measurement | **Framework:** Wildcard (Computational Topology)

### Question

Do circuit activation point clouds have stable topological structure? Persistent homology extracts multi-scale geometric features from activation manifolds that are invariant to smooth deformations -- they characterize the *shape* of the data, not its coordinates or magnitudes.

The key topological features are:

- **$H_0$ (connected components):** clustering structure. Multiple persistent connected components means the activation manifold has distinct clusters -- different computational regimes that do not smoothly interpolate.
- **$H_1$ (loops):** circular or ring geometry. A persistent 1-cycle means the activations form a loop in high-dimensional space -- possibly corresponding to a cyclic computation (cycling through name positions in IOI, cycling through token classes in induction).
- **Higher-dimensional features:** voids and cavities that indicate more complex geometric structure.

The persistence diagram -- a scatter plot of (birth, death) for each topological feature -- is the primary artifact. Features far from the diagonal (long persistence) are robust structure. Features near the diagonal (short persistence) are noise. The critical test is bootstrap stability: does the same persistence diagram appear when prompts are resampled?

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `cka` | Representational similarity between circuit activations across different tasks | $\geq 0.5$ |
| `cross_task_generalization` | Whether topological structure transfers across tasks | $\geq 0.5$ |
| `cross_model_invariance` | Whether topological structure is preserved across different model instances | $\geq 0.6$ |

### Interpretation

This protocol addresses criterion **M7 (Topological persistence)**: does the circuit's activation geometry have persistent topological features stable across prompts?

**High CKA** means the circuit's representational geometry is consistent across tasks. The topological structure is not task-specific -- it reflects a general property of the circuit's organization.

**High cross-task generalization** means the same topological features (loops, clusters, voids) appear when the circuit processes different tasks. This is evidence that the topology reflects the circuit's computational structure rather than the structure of any particular input distribution.

**High cross-model invariance** is the strongest topological test: the same persistent features appear in the circuit across different model instances (different random seeds, different training runs, or different model sizes). Topology that survives changes in model initialization is a genuine geometric invariant of the computation, not an artifact of a particular set of learned weights.

The thresholds from the lens overview require at least 2 topological features with persistence ratio at least 3x the median persistence, and bootstrap bottleneck distance at most 0.2x the maximum persistence over 200+ bootstrap samples.

### Theoretical grounding

| Source | Year | Contribution |
|---|---|---|
| Edelsbrunner & Harer (2008) | 2008 | Persistent Homology -- a Survey |
| Gudhi library | -- | Computational topology toolkit |
| Wasserstein distance | -- | Distance metric between persistence diagrams |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python tda_persistence.py                       # all tasks, CPU
uv run python tda_persistence.py --device cuda          # GPU
uv run python tda_persistence.py --tasks ioi induction  # specific tasks
```

---

## PH_CP -- Critical Phenomena (`critical_phenomena.py`)

**Lens:** Dynamical Systems | **Validity type:** Construct | **Framework:** Cross-discipline (Physics)

### Question

Does the circuit exhibit phase transitions -- abrupt behavioral changes as intervention strength is scaled? Does it show symmetry breaking, where equivalent representations become asymmetric under perturbation?

Phase transitions and symmetry breaking are central concepts in statistical physics. A phase transition occurs when a continuous change in a control parameter (temperature, pressure, intervention strength) produces a discontinuous change in the system's macroscopic behavior. Symmetry breaking occurs when a system that is symmetric under some transformation (e.g., permutation of equivalent components) settles into an asymmetric state -- one that picks out a preferred direction even though the underlying equations treat all directions equally.

In the context of circuit evaluation, these concepts ask whether the circuit's response to perturbation has qualitative structure -- whether there are critical thresholds where the computation changes character, and whether equivalent circuit components play asymmetric roles.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `phase_transitions` | Detection of abrupt behavioral changes under scaled interventions | $\geq 0.5$ |
| `symmetry_breaking` | Detection of asymmetric behavior among nominally equivalent representations | $\geq 0.0$ |

### Interpretation

**Phase transitions** are detected by sweeping the intervention strength continuously and looking for discontinuities in the behavioral metric. A circuit with a sharp phase transition at intervention strength $\alpha_c$ behaves qualitatively differently below and above $\alpha_c$ -- it is not simply "more perturbed" but in a different computational regime. The location of $\alpha_c$ is informative: a circuit whose phase transition occurs at low $\alpha_c$ is fragile (it changes character under weak perturbation); a circuit whose transition occurs at high $\alpha_c$ is robust (it maintains its computational regime under strong perturbation).

The phase transition threshold is distinct from the gain margin (E8 in the [control theory lens](/framework/lenses/supporting/control-theory-metrics)). The gain margin asks "how much perturbation before performance degrades by 50%?" -- a quantitative question. Phase transition detection asks "is there a perturbation strength where the behavior changes qualitatively?" -- a structural question. A circuit can have a high gain margin (performance degrades smoothly) but still exhibit a phase transition (the computational regime shifts, even though the output metric changes gradually).

**Symmetry breaking** identifies cases where components that should be interchangeable (by the circuit's symmetry) are not. For example, in a circuit with two name-mover heads, permutation symmetry predicts that swapping their roles should not change the circuit's behavior. If it does, one head has specialized and the symmetry is broken. Symmetry breaking is not inherently problematic -- it may reflect genuine functional specialization. But it should be acknowledged in the circuit description, because it means the circuit has more structure than the symmetric claim implies.

The symmetry-breaking threshold of 0.0 reflects the fact that any detected asymmetry is informative. The question is not "is there enough symmetry breaking?" but "is there any?" -- and if so, what does it tell us about the circuit's internal organization?

### Theoretical grounding

| Source | Year | Contribution |
|---|---|---|
| Wilson & Kogut (1974) | 1974 | The renormalization group and the epsilon expansion |
| Anderson (1972) | 1972 | "More Is Different" -- emergence and symmetry breaking |
| Goldenfeld (1992) | 1992 | *Lectures on Phase Transitions and the Renormalization Group* |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python critical_phenomena.py                       # all tasks, CPU
uv run python critical_phenomena.py --device cuda          # GPU
uv run python critical_phenomena.py --tasks ioi induction  # specific tasks
```

---

## Cross-protocol evidence patterns

The five dynamical systems protocols provide converging evidence when combined:

| Pattern | Protocols | What it establishes |
|---|---|---|
| Low-rank Koopman + scale invariance + persistent topology | WC_M8 + WC_M10/PH_RG + WC_M5 | Fully characterized dynamical structure: identifiable modes, robust across scales, with stable geometry |
| Low-rank Koopman without scale invariance | WC_M8 + (WC_M10 fails) | Identifiable dynamics at the current analysis granularity, but structure depends on the level of analysis |
| Persistent topology + cross-model invariance | WC_M5 | The circuit's geometry is a genuine invariant, not an artifact of training initialization |
| Phase transition + symmetry breaking | PH_CP | The circuit has critical structure -- qualitatively different computational regimes separated by a sharp threshold, with functional specialization among components |
| Scale invariance + cross-task generalization | WC_M10 + WC_M5 | The circuit has universal structure shared across tasks and robust across analysis scales -- a "universality class" |
| Flat Koopman spectrum + no persistent features | WC_M8 + WC_M5 | No identifiable dynamical structure. The proposed component set may not constitute a coherent circuit in the dynamical sense |

## Relationship to other lenses

The dynamical systems lens provides structural evidence that is orthogonal to the causal evidence from other lenses:

- **Neuroscience lens:** Ablation (I1) establishes that a component is causally necessary; Koopman decomposition (C6) establishes that the circuit's dynamics are structured. A component can be necessary without being part of a structured dynamical system (it may be a critical relay with no spectral signature), and dynamics can be structured without the components being individually necessary (the mode may span many redundant components).
- **Control theory lens:** Controllability and observability (I13, M9) characterize whether you can drive and measure the circuit's state; spectral identifiability (C6) characterizes whether the state space has low-dimensional structure. A circuit can be controllable and observable but full-rank (no identifiable modes), or low-rank but uncontrollable (the low-dimensional dynamics are in an unreachable subspace).
- **Information theory lens:** Directed information flow (I11) establishes that components communicate; mode-function alignment (C7) establishes that the communication carries structured dynamical content. Directed flow without mode structure is a pipeline without identifiable computation; mode structure without directed flow is structure without communication.
