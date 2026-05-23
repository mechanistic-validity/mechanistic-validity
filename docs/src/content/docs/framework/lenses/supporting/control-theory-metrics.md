---
title: "Control Theory -- Metrics & Protocols"
description: "Metrics and protocol implementations for the control theory lens: settling depth, stability margin, observability, engineering robustness, and PID steering."
---

# Control Theory -- Metrics & Protocols

These protocols implement the criteria defined in the [control theory lens overview](/framework/lenses/supporting/control-theory). Each protocol translates a classical control-theoretic concept into a concrete experimental procedure for evaluating transformer circuit claims.

The control theory lens contributes criteria across three validity types: [internal validity](/framework/validity-types/internal) (settling depth, controllability), [external validity](/framework/validity-types/external) (stability margin), and [measurement validity](/framework/validity-types/measurement) (observability). The protocols below operationalize these criteria as runnable experiments with defined metrics, thresholds, and calibration suites.

---

## I14 -- Settling Depth (`a14_settling_depth.py`)

**Lens:** Control Theory | **Validity type:** Internal | **Framework:** Causal (perturbation propagation and recovery)

### Question

When a circuit component is perturbed at layer $L$, how quickly does the residual stream recover? If the circuit is modular and self-contained, perturbations should either persist (the component is necessary) or settle quickly (the network compensates). Long settling depths indicate distributed processing; short settling depths suggest localized computation or self-repair.

This protocol directly measures the step response -- the central analytical construct of the control theory lens. The step response curve (perturbation magnitude vs layer depth) is the primary diagnostic artifact, revealing whether the circuit exhibits overdamped self-correction, underdamped overshoot, a persistent plateau, or divergent instability.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `mean_settling_depth` | Average number of layers until the perturbation decays below 20% of its initial magnitude | $\leq 3.0$ layers |
| `steady_state_error` | Residual perturbation at the final layer, measured as normalized $L^2$ distance from the clean run | $\leq 0.1$ |

The settling threshold is set at 20% of the initial perturbation magnitude. A component that never reaches this threshold within the model's remaining layers is classified as "non-settling."

### Interpretation

A circuit with short settling depth ($\leq 3$ layers) and low steady-state error ($\leq 0.1$) has effective self-correction -- either through its own internal stability or through backup mechanisms in the network. The distinction matters: settling that survives compound ablation (removing backup components) is circuit-intrinsic; settling that collapses under compound ablation is network-mediated.

Long settling depth combined with high steady-state error means the perturbation propagates through all remaining layers and never dissipates -- the circuit component has a lasting, distributed effect on downstream computation. This is consistent with the component being load-bearing but inconsistent with the computation being localized to the proposed circuit boundary.

### Theoretical grounding

| Source | Contribution |
|---|---|
| Ogata (2010), *Modern Control Engineering* | Settling time in LTI systems |
| Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits" | Residual stream as a dynamical system |
| McGrath et al. (2023), "The Hydra Effect" | Emergent self-repair in language model computations |

### Calibrations

Uses `CAUSAL_CALIBRATIONS` -- the standard calibration suite for perturbation-based protocols. Calibrations include random-component baselines and ablation-method comparisons.

### Usage

```bash
uv run python a14_settling_depth.py                       # all tasks, CPU
uv run python a14_settling_depth.py --device cuda          # GPU
uv run python a14_settling_depth.py --tasks ioi induction  # specific tasks
```

---

## E8 -- Stability Margin (`d08_stability_margin.py`)

**Lens:** Control Theory | **Validity type:** External | **Framework:** Behavioral (robustness under scaled perturbation)

### Question

How robust is the circuit's behavioral contribution to scaled perturbation? The gain margin is the largest multiplier on the ablation magnitude at which the model still retains more than 50% of clean performance. A large gain margin means the circuit's contribution is robust and findings from one perturbation strength are likely to hold at nearby strengths. A small gain margin means the circuit is operating near instability -- findings are intervention-strength-specific and may not transfer.

The protocol sweeps perturbation magnitudes from 0.5x to 5.0x the standard ablation strength, measuring the fraction of clean logit difference retained at each level. The resulting dose-response curve (perturbation strength vs behavioral metric) is the stability analog of pharmacological dose-response.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `gain_margin` | Largest perturbation multiplier where performance exceeds 50% of the clean baseline | $\geq 2.0$ |
| `performance_at_2x` | Fraction of clean logit difference retained at 2x perturbation strength | $\geq 0.5$ |

The perturbation magnitudes tested are: $\{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0\}$.

### Interpretation

A gain margin of 2.0 or higher means the circuit's behavioral contribution survives doubling the perturbation strength -- the finding is not balanced on a knife-edge. This supports external validity: the causal claim generalizes across intervention conditions.

A gain margin below 2.0 is a warning, not a disqualification. It means findings at the standard ablation level may not hold at other strengths, and the causal claim should be qualified as intervention-strength-specific. A gain margin below 1.0 means the circuit's behavior changes qualitatively even under a weaker-than-standard perturbation -- the finding is fragile.

The distinction between quantitative and qualitative degradation matters. Gradual performance decline (the logit difference decreases smoothly with perturbation strength) indicates a stable circuit under stress. A discontinuity -- a sharp drop at a critical perturbation strength -- indicates a phase transition, where the circuit shifts from one behavioral mode to another. The latter is more informative: it identifies the perturbation strength at which the circuit's computational regime changes.

### Theoretical grounding

| Source | Contribution |
|---|---|
| Ogata (2010), *Modern Control Engineering* | Gain margin in feedback systems |
| Cohen & Saphra (2024), "Evaluating the Faithfulness of Circuit Hypotheses" | Faithfulness under varying ablation strength |
| Wang et al. (2022), "Interpretability in the Wild" | IOI ablation studies |

### Calibrations

Uses `BEHAVIORAL_CALIBRATIONS` -- calibrations designed for behavioral (output-level) protocols, including random-circuit and shuffled-prompt baselines.

### Usage

```bash
uv run python d08_stability_margin.py                       # all tasks, CPU
uv run python d08_stability_margin.py --device cuda          # GPU
uv run python d08_stability_margin.py --tasks ioi induction  # specific tasks
```

---

## M9 -- Observability (`e09_observability.py`)

**Lens:** Control Theory | **Validity type:** Measurement | **Framework:** Representational (output-decodability of internal state)

### Question

Is the circuit's internal state decodable from model outputs? If so, the circuit is "observable" in the control-theoretic sense: its hidden state can be reconstructed from the output trajectory. The protocol tests this by training linear probes from final logits to individual circuit head activations. High $R^2$ means the output contains enough information to reconstruct what the circuit computed internally.

Observability is the measurement-validity dual of controllability. A circuit can be controllable (you can steer its internal state by intervening on its inputs) without being observable (you cannot tell what state it reached from its outputs), and vice versa. Both properties are needed for the circuit to be fully characterizable -- controllable means interventions are meaningful, observable means measurements are complete.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `mean_probe_r2` | Mean $R^2$ of linear probes from final logits to individual circuit head activations | $\geq 0.3$ |
| `observability_rank_ratio` | Rank of the joint [logits, activations] matrix divided by the number of circuit heads | $\geq 0.8$ |

### Interpretation

High `mean_probe_r2` ($\geq 0.3$) means the circuit's internal state is at least partially recoverable from the model's output. The probes are linear, so the recovered information is linearly accessible -- the circuit's internal representations are not hidden behind a nonlinear encoding that requires special decoding to access.

The `observability_rank_ratio` captures whether the mapping from outputs to internal states is full-rank. A ratio of 1.0 means every circuit head's activation is independently recoverable from the output. A ratio below 0.8 means some heads' activations are linearly dependent in the output space -- they cannot be individually distinguished from logits alone. This does not mean those heads are unimportant, but it does mean that output-based measurement cannot fully characterize their individual contributions.

Low observability ($R^2 < 0.3$, rank ratio $< 0.8$) means the circuit's internal dynamics are partially hidden from output-based evaluation. Behavioral metrics -- which measure the circuit's contribution through its effect on logits -- are seeing a projection of the circuit's state, not the full state. Any internal validity claim based solely on behavioral measurement implicitly excludes the unobservable subspace.

### Theoretical grounding

| Source | Contribution |
|---|---|
| Kalman (1960), "A New Approach to Linear Filtering and Prediction Problems" | Observability in control theory |
| Alain & Bengio (2017), "Understanding Intermediate Layers Using Linear Classifier Probes" | Linear probing methodology |
| Belinkov (2022), "Probing Classifiers: Promises, Shortcomings, and Advances" | Probe reliability and limitations |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS` -- calibrations appropriate for representational and structural protocols.

### Usage

```bash
uv run python e09_observability.py                       # all tasks, CPU
uv run python e09_observability.py --device cuda          # GPU
uv run python e09_observability.py --tasks ioi induction  # specific tasks
```

---

## EN -- Engineering Robustness (`engineering.py`)

**Lens:** Control Theory | **Validity type:** Internal | **Framework:** Cross-discipline (Engineering)

### Question

How robust is the circuit under stress? This protocol tests four complementary aspects of engineering robustness: whether performance degrades gracefully as components are removed, whether the circuit recovers from component perturbation, how performance holds under extreme conditions, and how local faults cascade through the circuit.

Engineering robustness is the practical counterpart of the gain margin (E8). Where E8 scales a single perturbation continuously, engineering robustness tests qualitatively different stress conditions -- component removal, fault injection, extreme inputs, and cascade analysis. Together, they characterize the circuit's operational envelope: the range of conditions under which the circuit's computational claims hold.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `graceful_degradation` | Smooth vs catastrophic performance decline as components are removed | $\geq 0.5$ |
| `fault_tolerance` | Recovery from component perturbation (noise injection, activation corruption) | $\geq 0.5$ |
| `stress_testing` | Performance under extreme conditions (adversarial inputs, out-of-distribution prompts) | $\geq 0.5$ |
| `error_propagation` | Fault cascade analysis -- how local faults spread through the circuit | $\leq 0.0$ (lower is better) |

### Interpretation

**Graceful degradation** measures the shape of the performance curve as components are ablated one by one. A circuit with graceful degradation loses performance smoothly -- each removed component contributes a roughly equal decrement. A circuit with catastrophic degradation has a cliff: performance is stable until a critical component is removed, then collapses. Graceful degradation suggests distributed computation; catastrophic degradation suggests a bottleneck architecture.

**Fault tolerance** measures recovery from noisy perturbation rather than clean ablation. A fault-tolerant circuit absorbs noise at individual components without propagating errors to the output. Low fault tolerance means the circuit has no error-correction mechanisms -- any perturbation at any component reaches the output.

**Stress testing** pushes the circuit beyond its normal operating conditions. The question is whether the circuit's computational structure holds under extreme inputs or whether it relies on distributional assumptions that break under stress.

**Error propagation** tracks how a localized fault cascades through the circuit's layers and components. Low error propagation means faults are contained locally. High error propagation means a single component failure disrupts the entire circuit.

### Theoretical grounding

| Source | Contribution |
|---|---|
| Leveson (2011), *Engineering a Safer World* | Systems safety and fault containment |
| Avizienis et al. (2004), "Basic Concepts and Taxonomy of Dependable Computing" | Fault tolerance taxonomy |
| Strogatz (2001), "Exploring Complex Networks" | Robustness in complex networks |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python engineering.py                       # all tasks, CPU
uv run python engineering.py --device cuda          # GPU
uv run python engineering.py --tasks ioi induction  # specific tasks
```

---

## WC_M1 -- PID Activation Steering (`pid_steering.py`)

**Lens:** Control Theory | **Validity type:** Internal | **Framework:** Wildcard (Control Theory)

### Question

Can activation steering be modeled as a PID controller? Standard steering is proportional-only (P-only): a fixed scaling factor $\alpha$ multiplied by a steering vector. PID control adds two additional terms:

- **Integral (I):** accumulated error across layers eliminates steady-state drift. If the steering effect attenuates through layers, the I-term compensates by accumulating the residual error.
- **Derivative (D):** rate-of-change damping prevents overshoot. If the circuit overreacts to steering (the effect grows before settling), the D-term suppresses the oscillation.

The dose-response curve of a circuit determines which control regime applies. A steep dose-response (small changes in $\alpha$ produce large behavioral shifts) needs D-term damping to prevent overshoot. A shallow dose-response (the circuit barely responds to steering) needs I-term accumulation to build up sufficient effect. A moderate dose-response may work with P-only steering -- standard activation engineering.

This protocol bridges the control theory lens to the pharmacology lens: the dose-response curve from pharmacological evaluation determines the control regime, and the PID analysis determines the optimal intervention strategy.

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| `dose_response` | Circuit response curve steepness -- how sensitively the output responds to steering magnitude | $\geq 0.5$ |
| `sigma_ablation` | Noise tolerance -- how much Gaussian noise the steering signal can absorb before losing effect (integral accumulation budget) | $\geq 0.5$ |
| `effect_size` | Overall control authority -- the maximum behavioral shift achievable through steering | $\geq 0.8$ |

### Interpretation

**High dose-response + high effect size** indicates a circuit that responds strongly and steeply to steering. The circuit is controllable in the strong sense: small interventions produce large, predictable behavioral changes. This is the regime where D-term damping matters -- without it, steering overshoots the target.

**Low dose-response + low sigma ablation** indicates a circuit that is hard to steer and sensitive to noise. The circuit has low control authority. I-term accumulation across layers may be needed to build sufficient steering effect, but noise sensitivity limits the accumulation budget. This regime suggests the circuit's computation is distributed or that the steering vector is not well-aligned with the circuit's functional subspace.

**High sigma ablation** indicates the steering signal is robust to noise -- the circuit's response to the steering vector survives corruption. This is consistent with the steering vector targeting a low-dimensional, robust feature of the circuit rather than relying on precise alignment with a high-dimensional representation.

The connection to Ziegler-Nichols tuning is direct: the dose-response curve determines the ultimate gain and ultimate period, from which PID gains can be computed. Whether PID steering outperforms P-only steering on a given circuit is an empirical question that this protocol is designed to answer.

### Theoretical grounding

| Source | Contribution |
|---|---|
| ICLR 2026, "Activation Steering with a Feedback Controller" | PID framework for activation steering |
| Ziegler & Nichols, tuning method | Gain selection from dose-response curves |

### Calibrations

Uses `STRUCTURAL_CALIBRATIONS`.

### Usage

```bash
uv run python pid_steering.py                       # all tasks, CPU
uv run python pid_steering.py --device cuda          # GPU
uv run python pid_steering.py --tasks ioi induction  # specific tasks
```

---

## Cross-protocol evidence patterns

The five control theory protocols provide converging evidence when combined:

| Pattern | Protocols | What it establishes |
|---|---|---|
| Short settling + high gain margin + high observability | I14 + E8 + M9 | Fully characterizable, robust control system |
| Short settling + low gain margin | I14 + E8 | Self-correcting but fragile -- findings are intervention-strength-specific |
| High observability + low controllability (via PID) | M9 + WC_M1 | The circuit is readable but not steerable -- internal state is decodable from outputs, but steering interventions have low effect size |
| Graceful degradation + high fault tolerance | EN | The circuit absorbs component loss and noise without catastrophic failure |
| Steep dose-response + high sigma ablation | WC_M1 | Strong, noise-robust control authority -- the circuit is a good target for activation engineering |
| Non-settling step response + high error propagation | I14 + EN | Perturbations cascade through all remaining layers and propagate to other components -- the circuit lacks containment |

## Relationship to other lenses

The control theory lens is complementary to other lenses in the framework:

- **Neuroscience lens:** Ablation (I1) establishes necessity; control theory establishes whether the intervention reached the circuit's full state space (controllability) and whether the measurement captured the full internal state (observability).
- **Pharmacology lens:** Dose-response curves (D1) characterize how the circuit responds to graded intervention; PID steering (WC_M1) extends this by asking which control regime (P, PI, PD, PID) produces optimal intervention outcomes.
- **Dynamical systems lens:** Koopman spectral decomposition characterizes the modes of the circuit's dynamics; control theory characterizes whether those modes are reachable (controllable) and measurable (observable).
