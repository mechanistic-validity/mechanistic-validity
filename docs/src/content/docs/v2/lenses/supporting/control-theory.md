---
title: "Control Theory"
description: "The control systems lens: is the circuit a steerable, observable, stable control system — or an uncontrollable black box?"
---

# The Control Theory Lens

This lens asks one question: **can you steer the circuit, and does it resist steering in a structured way?**

When we inject a steering vector into the residual stream, we are performing a control intervention: we apply an input to a dynamical system and observe the system's response. When we activation-patch a component and measure the downstream effect across layers, we are measuring the system's step response. Classical control theory — Kalman's state-space formulation, Nyquist's frequency-domain stability analysis, Bode's gain and phase margins — provides a century of tools for characterizing such systems. These tools ask questions that are orthogonal to the ablation-based questions of neuroscience and the dose-response questions of pharmacology. Neuroscience asks whether the component is necessary. Pharmacology asks whether the effect scales. Control theory asks whether you can *command* the circuit — drive it to a desired state, observe its internal state from its outputs, and predict how it will respond to perturbation.

The key insight is that a circuit's response to perturbation has structure. After injecting a perturbation at layer $L$, the residual stream does not simply degrade or recover — it follows a trajectory. The trajectory may overshoot (the circuit overcompensates), oscillate (it alternates between overcompensation and undercompensation), settle quickly (it has strong self-correction), or never settle (the perturbation propagates indefinitely). Each of these behaviors corresponds to a different control-theoretic property: damping, stability, bandwidth, gain margin. A circuit with well-characterized control properties is one whose behavior under intervention is *predictable*, not just measurable.

There is a disanalogy worth naming. Physical control systems operate in continuous time with continuous feedback loops. Transformer layers are discrete, feedforward, and have no explicit feedback — each layer processes the residual stream once and passes it to the next. The "control loop" is the layer-to-layer propagation through the residual stream, and the "feedback" is the residual connection that adds each layer's output back to its input. This means some control concepts (frequency-domain analysis, continuous-time stability) apply only loosely, while others (state-space controllability, observability, step response analysis, settling behavior) transfer directly to the discrete-layer setting.

## Key Distinctions

### Controllability vs observability

Kalman (1960) identified two independent properties of any linear system. A system is **controllable** if, by choosing appropriate inputs, you can drive the state from any initial condition to any desired state in finite time. A system is **observable** if you can determine the system's internal state from its outputs alone — without direct access to hidden variables.

These are independent properties. A circuit can be controllable but not observable: you can steer it by intervening on its inputs, but you cannot tell from the circuit's outputs alone where the internal state ended up. Conversely, a circuit can be observable but not controllable: you can perfectly decode the circuit's internal state from its outputs, but no intervention at its inputs can change that state — the computation is determined by upstream components outside the circuit boundary.

In MI: controllability is tested by steering interventions (activation patching, steering vectors, DAS alignment) that attempt to drive the circuit to a target state. Observability is tested by probing: can a linear or low-rank probe recover the circuit's internal activations from its output contributions to the residual stream? A circuit that is both controllable and observable is one where interventions are interpretable — you can change the state and verify the change. A circuit that is neither is a black box.

### Open-loop vs closed-loop

Open-loop control applies a fixed intervention and observes the result. Closed-loop control adjusts the intervention based on the observed result — it uses feedback. Most MI is open-loop: ablate a head, measure the effect. The intervention is fixed and the measurement is a single snapshot.

Closed-loop control in MI would mean iterative intervention: patch an activation, observe the downstream effect, adjust the patch based on what happened, and repeat. Multi-step activation patching with adjustment — or iterative steering where the multiplier is tuned to achieve a target behavioral metric — tests whether the circuit responds predictably to feedback. A circuit that is open-loop controllable (a single patch achieves the target) but closed-loop unstable (iterative adjustment diverges) has different control properties than one that converges under feedback.

In MI: closed-loop experiments are rare. When they are performed (e.g., iterative activation engineering to reach a target generation), they reveal properties invisible to open-loop ablation — particularly whether the circuit's response is monotone (more intervention always moves the output in the same direction) or non-monotone (small interventions help, large interventions overshoot, and feedback is needed to converge).

### Stability vs performance

A stable system returns to equilibrium after perturbation. A performant system reaches the desired state quickly. These can conflict — a system optimized for speed may overshoot and oscillate before settling. A heavily damped system settles without oscillation but responds slowly.

In MI: a circuit that immediately compensates for an ablation (fast settling, low steady-state error) may do so by routing through backup paths — which is functionally different from a circuit that smoothly absorbs the perturbation through its own internal structure. The distinction matters: backup-mediated recovery is a property of the network's redundancy, not the circuit's internal stability. A circuit that settles through its own dynamics (the residual stream returns to the clean trajectory within the circuit's layers) has genuine stability. A circuit that settles because other components compensate has apparent stability — the network is stable, but the circuit's contribution is not self-correcting.

The gain margin quantifies this distinction. A circuit with a gain margin of 2.0 can absorb a perturbation twice the strength of a standard ablation before qualitative behavioral change. If the stability comes from the circuit's own structure, the gain margin should hold when backup components are also ablated. If it comes from network redundancy, the gain margin collapses under compound ablation.

### Step response

The signature test of control theory is the step response: inject a step perturbation and measure the system's trajectory back to equilibrium. In a continuous-time system, the step response reveals rise time, overshoot, settling time, and steady-state error — each characterizing a different aspect of the system's control properties.

In MI: the analog is straightforward. Inject a perturbation at layer $L$ (zero-ablation, mean-ablation, or activation patch of a circuit component) and measure the residual stream distance from the clean run at each subsequent layer $L+1, L+2, \ldots, L_{\text{final}}$. The trajectory is the step response. Its shape reveals:

- **Rise time** — how quickly the perturbation effect appears in downstream layers. Instant for ablation (the effect is at layer $L$), but for steering or partial interventions, the effect may take layers to propagate.
- **Overshoot** — does the circuit overcompensate? If the residual stream distance exceeds its peak value at layer $L+1$ and then decreases, the circuit has compensation mechanisms that initially over-correct.
- **Settling depth** — how many layers until the residual stream returns to within a specified tolerance of the clean trajectory. This is I14.
- **Steady-state error** — does the circuit fully recover? If the residual stream distance at the final layer is nonzero, the perturbation has a lasting effect. This is the circuit's static gain for the perturbation.

## Analytical Constructs

### The step response curve

The primary artifact of control-theoretic evaluation is the step response curve: a plot of perturbation magnitude (y-axis, measured as residual stream cosine distance or $L^2$ distance from the clean run) against layer depth (x-axis, from the perturbation site $L$ through the model's final layer).

The curve reveals structure that no single-layer measurement can:

- **Monotone decay** — the distance decreases at every layer. The circuit has overdamped self-correction. Information lost to the perturbation is gradually restored by other components.
- **Overshoot then decay** — the distance increases at $L+1$, peaks, then decays. The circuit has underdamped self-correction — it overcompensates before settling. The overshoot magnitude characterizes how strongly the compensation mechanisms are coupled to the perturbation.
- **Plateau** — the distance reaches a constant nonzero value and stays there through remaining layers. The circuit has no self-correction for this perturbation; the effect is permanent. The plateau height is the steady-state error.
- **Divergence** — the distance grows through remaining layers. The perturbation is amplified rather than absorbed. The circuit is unstable under this perturbation strength.

To construct the curve: run the model on a prompt set with the perturbation applied at layer $L$ (e.g., zero-ablation of a circuit component). At each subsequent layer, record the $L^2$ distance between the perturbed and clean residual stream vectors, averaged across positions and prompts. Plot the layer-indexed distance. Repeat at multiple perturbation strengths (interpolating from $\alpha = 0.5$ to $\alpha = 2.0$) to obtain a family of step responses. The family reveals whether the self-correction is linear (distance scales proportionally with perturbation) or nonlinear (small perturbations settle, large ones diverge).

## Sources

| Source | Year | Field | Principle |
|---|---|---|---|
| [Nyquist, "Regeneration theory"](https://doi.org/10.1002/j.1538-7305.1932.tb02150.x) | 1932 | Electrical Engineering | **Nyquist stability criterion** — a system is stable if and only if the open-loop frequency response does not encircle the critical point; stability is a property of the loop, not any single component |
| [Bode, *Network Analysis and Feedback Amplifier Design*](https://archive.org/details/networkanalysisf0000bode) | 1945 | Control Engineering | **Gain and phase margins** — quantitative robustness measures: how much the gain can increase (or phase shift) before instability; perturbation tolerance as a number, not pass/fail |
| [Kalman, "A new approach to linear filtering and prediction problems"](https://doi.org/10.1115/1.3662552) | 1960 | Control Theory | **State-space controllability and observability** — a system is controllable if the controllability Gramian has full rank; observable if the observability Gramian has full rank; these are independent, testable properties |
| [Doyle, Glover, Khargonekar & Francis, "State-space solutions to standard $H_2$ and $H_\infty$ control problems"](https://doi.org/10.1109/9.29425) | 1989 | Robust Control | **$H_\infty$ control** — worst-case perturbation analysis; the gain margin is the smallest perturbation that destabilizes the system |
| [Astrom & Murray, *Feedback Systems: An Introduction for Scientists and Engineers*](https://doi.org/10.2307/j.ctvcm4gdk) | 2008 | Control Theory | **Step response characterization** — rise time, overshoot, settling time, and steady-state error as the four canonical descriptors of a system's transient response |
| [Skogestad & Postlethwaite, *Multivariable Feedback Control: Analysis and Design*](https://doi.org/10.1002/rnc.816) | 2005 | Control Engineering | **MIMO control and interaction analysis** — multivariable systems where inputs affect multiple outputs; the Relative Gain Array characterizes how tightly coupled the channels are |

## Validity type: [Internal validity](/v2/validity-types/internal)

> **State-space controllability ([Kalman 1960](https://doi.org/10.1115/1.3662552)):** A linear system $(A, B)$ is controllable if and only if the controllability matrix $\mathcal{C} = [B, AB, A^2B, \ldots, A^{n-1}B]$ has full rank. In MI: the circuit is a steerable system — one whose state can be driven to any target by choosing appropriate inputs — only if interventions at the circuit's inputs can reach all dimensions of the circuit's internal state. A circuit that is not controllable has internal degrees of freedom that no input intervention can reach.

The primary validity type is internal. Controllability (I13) and settling depth (I14) ask whether the circuit's internal structure supports the kind of causal interventions that internal validity criteria I1-I5 rely on. If a circuit is not controllable — if there are directions in its state space that no input perturbation can reach — then ablation and patching experiments are probing only the controllable subspace. The uncontrollable dimensions are invisible to intervention-based evaluation, and any internal validity claim implicitly excludes them.

Observability (M9) bridges to measurement validity: if the circuit's internal state cannot be fully decoded from its outputs, then behavioral metrics based on output measurements are incomplete indicators of the circuit's state. Stability margin (E8) bridges to external validity: a circuit whose behavior changes qualitatively under small perturbation increases has narrow robustness, and findings from one perturbation strength may not transfer to others.

## Criteria

| Code | Criterion | What it asks | Validity type |
|---|---|---|---|
| I13 | Controllability | Can input interventions steer the circuit to an arbitrary target state? | Internal |
| I14 | Settling depth | After perturbation, how many layers until the residual stream recovers? | Internal |
| M9 | Observability | Can the circuit's internal state be fully decoded from its outputs? | Measurement |
| E8 | Stability margin | How much perturbation before the circuit's behavior qualitatively changes? | External |

### Controllability (I13)

The circuit should be a steerable system: interventions at its inputs should be able to drive the circuit's internal state to an arbitrary target within the circuit's state space.

**What it establishes.** The circuit's internal state is reachable by input interventions. When we activation-patch or steer, we are not just perturbing the circuit — we are driving it to a specific state. If the controllability rank is high, the intervention reaches all dimensions of the circuit's processing. If it is low, there are internal dimensions that no input intervention can access, and any causal claim based on input intervention implicitly excludes those dimensions.

**What it does not establish.** That the circuit is causally necessary (I1), that the reached state produces the desired behavior, or that the circuit is the only controllable path to the target output. Controllability is a structural property of the circuit's input-state relationship, not a functional property of its computation.

**Threshold.** The controllability Gramian $\mathcal{G}_c = \sum_{k=0}^{L-1} A^k B B^T (A^T)^k$ (where $A$ is the linearized layer-to-layer map and $B$ is the input-to-state map at the intervention site) should have rank $\geq 0.8 \times \dim(\text{circuit state})$. The condition number of the controllability matrix $\mathcal{C}$ should be $< 100$; a condition number above this means some state directions require exponentially larger interventions to reach, making them controllable in theory but unreachable in practice.

**Minimum reporting.** Rank of the controllability Gramian as a fraction of circuit state dimension. Condition number of $\mathcal{C}$. If the circuit is not fully controllable, report the dimension and interpretation of the uncontrollable subspace.

### Settling depth (I14)

After a perturbation at layer $L$, the residual stream should return to near-clean-run values within a small number of subsequent layers.

**What it establishes.** The circuit (and the network around it) has self-correction or compensation mechanisms. A circuit with short settling depth absorbs perturbations quickly — the effect of an ablation or patch is localized to a few layers. A circuit that never settles propagates perturbation effects through all remaining layers, meaning every downstream computation is affected by the intervention, not just the target mechanism.

**What it does not establish.** Whether the settling is due to the circuit's own internal stability or due to backup mechanisms in other components compensating. Settling depth characterizes the network's response to the perturbation, not the circuit's isolated dynamics. To attribute settling to the circuit itself, the measurement should be repeated with backup components ablated.

**Threshold.** The residual stream $L^2$ distance from the clean run should return to within 20% of the pre-perturbation level within 3 layers of the perturbation site. Formally, if $d(l)$ is the normalized distance at layer $l$ and $d(L)$ is the distance at the perturbation layer, then $d(L+3) \leq 0.2 \cdot d(L)$. If the distance remains above $0.5 \cdot d(L)$ through all remaining layers, the component lacks compensation and the perturbation is non-settling.

**Minimum reporting.** The step response curve: $d(l)$ for $l = L, L+1, \ldots, L_{\text{final}}$. Settling depth (the first layer $l^*$ where $d(l^*) \leq 0.2 \cdot d(L)$, or "non-settling" if no such layer exists). Steady-state error $d(L_{\text{final}}) / d(L)$. Whether the curve was measured with or without backup components intact.

### Observability (M9)

The circuit's internal state should be fully decodable from its outputs.

**What it establishes.** The circuit's output contributions to the residual stream carry enough information to reconstruct the circuit's internal activations. If observability rank is high, behavioral measurements based on the circuit's output are complete — they reflect the full internal state. If observability rank is low, some internal dynamics are hidden from output-based measurement, and behavioral metrics are measuring a projection of the circuit's state rather than the state itself.

**What it does not establish.** That the internal state is interpretable, that the outputs are causally sufficient for the behavior, or that the decoding is unique. Observability is about information content, not meaning. A fully observable circuit whose internal state is uninterpretable is still a circuit whose measurements are complete.

**Threshold.** The observability Gramian $\mathcal{G}_o = \sum_{k=0}^{L-1} (A^T)^k C^T C A^k$ (where $C$ is the state-to-output map) should have rank $\geq 0.8 \times \dim(\text{circuit state})$. As a practical complement: a linear probe trained on the circuit's output to predict the circuit's internal activations should achieve $R^2 \geq 0.7$. If the Gramian rank condition is met but the probe fails, the observability is real but nonlinearly encoded — the state is present in the output but not linearly accessible.

**Minimum reporting.** Rank of the observability Gramian as a fraction of circuit state dimension. Linear probe $R^2$ from circuit output to circuit internal activations. If the circuit is not fully observable, report the dimension of the unobservable subspace and whether it corresponds to task-relevant or task-irrelevant state variables.

### Stability margin (E8)

The circuit's behavior should resist qualitative change under moderate perturbation increases.

**What it establishes.** The circuit's functional properties are robust — not balanced on a knife-edge where any increase in perturbation strength causes a qualitative shift in behavior. A high stability margin means the circuit's computational structure absorbs perturbation gracefully. Findings from one perturbation strength are likely to hold at nearby strengths, supporting external validity across intervention conditions.

**What it does not establish.** That the circuit is robust to perturbations of a different *type* (e.g., stable under zero-ablation but not under resample ablation), or that the stability margin generalizes to different prompt distributions. Stability margin is perturbation-type-specific and distribution-specific.

**Threshold.** Gain margin $\geq 2.0$: the circuit's target behavioral metric should not degrade by more than 50% when the perturbation strength is doubled from the standard ablation level. Formally, if $M(\alpha)$ is the behavioral metric at perturbation strength $\alpha$ and $\alpha_0$ is the standard ablation strength, then $M(2\alpha_0) \geq 0.5 \cdot M(\alpha_0)$. If $M(2\alpha_0) < 0.5 \cdot M(\alpha_0)$, the circuit has a gain margin below 2.0 and findings at $\alpha_0$ may not transfer to other perturbation strengths.

**Minimum reporting.** The behavioral metric at perturbation strengths $\alpha_0$ and $2\alpha_0$ (at minimum). The gain margin, computed as the largest factor $k$ such that $M(k\alpha_0) \geq 0.5 \cdot M(\alpha_0)$. Whether the behavioral change at $2\alpha_0$ is quantitative (gradual degradation) or qualitative (mode shift, e.g., from correct token to a different token class).

## Evidence Patterns

| Evidence pattern | What it establishes | Recommended language |
|---|---|---|
| Controllable + observable + stable | Fully characterizable control system | "The circuit is a steerable, observable system with gain margin $k$" |
| Controllable + observable, low stability margin | Steerable but fragile | "The circuit is steerable and observable but sensitive to perturbation strength; findings are intervention-strength-specific" |
| Controllable, not observable | Steerable but opaque | "Input interventions reach the circuit's state, but internal dynamics are not fully recoverable from outputs" |
| Observable, not controllable | Readable but not steerable | "The circuit's state is decodable from outputs, but input interventions do not reach all internal dimensions" |
| Non-settling step response | No self-correction | "Perturbation propagates through all remaining layers; the circuit lacks compensation mechanisms" |
| Short settling depth + high stability margin | Robust, self-correcting system | "The circuit absorbs perturbations within $k$ layers and tolerates $k\times$ perturbation strength" |

## Verdicts

- **Proposed --> Causally suggestive:** The control theory lens does not gate this transition. Causal evidence from the neuroscience lens (I1) remains the primary gate. However, I13 (controllability) strengthens the interpretation of positive I1 results: if the circuit is controllable, the ablation reached the circuit's full state space, not just a subspace.
- **Causally suggestive --> Mechanistically supported:** I14 (settling depth) contributes here. A circuit whose perturbation settles quickly has localized causal effects — the ablation result is about the circuit, not about cascading disruption to downstream processing. This supports confound control (I5) by ruling out propagation-mediated artifacts.
- **Mechanistically supported --> Triangulated:** E8 (stability margin) and M9 (observability) provide convergent evidence from a different evidence family (control-theoretic rather than neuroscientific or pharmacological). A circuit that is controllable, observable, stable, and causally necessary is supported by structurally independent lines of evidence.

## Protocol

For a proposed circuit $C$ with components at layers $L_1, \ldots, L_k$ and behavior $B$:

1. **Step response.** Apply the standard perturbation (zero-ablation, mean-ablation, or activation patch of $C$'s earliest component) at layer $L_1$. Record the residual stream $L^2$ distance from the clean run at each subsequent layer through $L_{\text{final}}$. Plot the step response curve.
2. **Settling depth.** From the step response curve, identify the settling depth: the first layer $l^*$ where $d(l^*) \leq 0.2 \cdot d(L_1)$. If no such layer exists, report "non-settling." Repeat with backup components ablated to determine whether settling is circuit-intrinsic or network-mediated.
3. **Stability margin.** Repeat the step response at perturbation strengths $\alpha \in \{0.5, 1.0, 1.5, 2.0\} \times \alpha_0$. For each, measure the behavioral metric $M(\alpha)$. Report the gain margin: the largest $k$ such that $M(k\alpha_0) \geq 0.5 \cdot M(\alpha_0)$.
4. **Controllability.** Linearize the layer-to-layer map at the circuit's input layer. Compute the controllability Gramian and report its rank as a fraction of circuit state dimension. Report the condition number of the controllability matrix.
5. **Observability.** Train a linear probe from the circuit's output (its contribution to the residual stream at its final layer) to predict the circuit's internal activations (at intermediate layers). Report probe $R^2$. Compute the observability Gramian and report its rank.
6. A skipped step must be named in the verdict.

## Case Studies

- [IOI](/framework/lenses_v6/examples/examples-ioi) — the IOI circuit can be evaluated with this lens via WC_M1 (PID Steering); step response analysis across the 26-head circuit reveals settling behavior after name-mover ablation
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) — two-layer induction circuit provides a minimal test case for controllability (previous-token head as input) and observability (induction head output)
