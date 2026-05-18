# Epistemic Framing Circuit: Mechanistic Validity Evaluation

## Why this document exists

The Mechanistic Validity framework defines criteria, instruments, description
modes, and verdict tiers for evaluating mechanistic interpretability claims.
But a framework on paper is not the same as a framework in practice. The only
way to know whether the criteria actually discriminate good circuits from bad
ones, whether the instruments produce meaningful measurements, and whether the
description modes capture the right distinctions is to use them on real
circuits and see what breaks.

This document records that process for one circuit: an experimental "epistemic
framing" circuit in GPT-2 small. It is not a success story. The initial circuit
fails most instruments. Two of the instruments turn out to have broken metrics.
The activation patching scan reveals the original circuit misses the most
important components entirely. These are not embarrassments to be hidden. They
are exactly what the framework is designed to surface.

The value of running a worked evaluation is threefold:

1. **It tests the circuit.** Does the claimed mechanism hold up under systematic
   probing? The framework's five validity types (construct, internal, external,
   measurement, interpretive) each ask a different question. A circuit that
   passes internal validity but fails measurement validity tells you something
   different from one that fails both.

2. **It tests the framework.** Instruments that fail on *every* circuit, strong
   and weak alike, reveal metric design problems, not circuit problems. Criteria
   that no existing instrument can evaluate reveal coverage gaps. Each evaluation
   sharpens the framework itself.

3. **It demonstrates the workflow.** Mechanistic validity is not a one-shot
   audit. It is an iterative process: evaluate, diagnose failures, determine
   whether the failure is in the claim or the methodology, fix what needs
   fixing, re-evaluate. The framework and the claims it evaluates improve
   together.

We compare a new experimental hypothesis, a hand-identified "epistemic framing"
circuit, against the well-established IOI circuit (Wang et al. 2022) as a
calibration baseline. The epistemic circuit was identified through 13 targeted
experiments and appears plausible on the surface: it has named functional roles,
directional pathways, and passes basic sanity checks. But surface plausibility
is exactly what rigorous evaluation is meant to look past. The framework
confirms the established circuit's strength while exposing where the new
hypothesis falls short.

---

## Task

**Epistemic framing** in GPT-2 small: the model processes epistemic stance markers
("I think", "I believe", "I know") prefixed to factual statements and still routes
the correct factual completion. 12 prompt pairs, e.g.:

- Clean: "I think the capital of France is" → Paris
- Bare: "The capital of France is" → Paris

The circuit should capture the components that handle the epistemic prefix
while preserving the factual prediction.

## Reference circuit: IOI (15 heads)

IOI (Indirect Object Identification, Wang et al. 2022) is used as a reference
throughout. It has 15 heads across 6 functional roles (duplicate token heads,
previous token heads, induction, S-inhibition, name movers, negative name movers)
discovered through systematic activation patching and validated extensively
in the literature.

---

## Pass 1: Initial 4-head circuit

### Circuit definition

The initial circuit was identified through targeted experiments (13 focused
probes testing specific hypotheses about epistemic processing):

| Role | Heads | Hypothesized function |
|------|-------|----------------------|
| Detector | L6H5 | Detects epistemic markers in input |
| Integrator | L9H2, L9H5 | Integrates stance with factual content |
| Executor | L10H5 | Routes final token prediction |

Pathways: detector→integrator, integrator→executor, detector→executor.

### Instrument results (40 prompts, CPU)

| Instrument | Criterion | IOI | Epistemic | Notes |
|------------|-----------|:---:|:---------:|-------|
| 70 | F1 Operation Specification | -- | -- | Metric needed recalibration (see below) |
| 71 | F2 Held-Out Prediction | PASS (0.50) | PASS (0.54) | Circuit heads' outputs predictable across prompts |
| 72 | F3 Replacement Test | PASS (1.02) | PASS (0.99) | Constant-function replacement recovers performance |
| 73 | S1 Distributional Characterization | PASS (11.4x) | FAIL (1.7x) | Circuit heads barely above background |
| 74 | S3 Distributional Stability | FAIL | FAIL | High variance, only 12 unique prompt templates |
| 75 | R1 Probe Decodability | PASS (0.58) | PASS (0.67) | Task info linearly decodable at circuit layers |
| 76 | R3 Causal Representation | PASS (0.95) | FAIL (ctrl=0.80) | Non-circuit layer 7 also passes IIA |
| 77 | A1 Procedure Specification | PASS (0.75) | FAIL (0.33) | Weak information ordering along pathways |
| 78 | A2 Composition Test | PASS (0.32) | FAIL (-0.13) | **Negative faithfulness**: circuit hurts predictions |
| 79 | A4 Intermediate State Prediction | -- | -- | Metric needed recalibration (see below) |
| 80 | CM1 Normative Account | PASS (2.25) | FAIL (1.60) | Borderline, below 2.0x threshold |
| 81 | CM2 Error Boundary | PASS (0.75) | FAIL (0.35) | Circuit doesn't track prompt difficulty |

**IOI: 8/10 pass.** (F1 and A4 excluded pending recalibration.)
**Epistemic: 3/10 pass.**

### Diagnosis

The failures cluster into two patterns:

**The circuit is too small.** The clearest signal is A2's negative faithfulness:
keeping only the 4 circuit heads and ablating everything else makes predictions
*worse* than the full model. S1's attribution ratio (1.7x vs IOI's 11.4x) confirms
the hypothesized heads don't stand out from background noise. R3 fails because
non-circuit layer 7 also passes the IIA test, meaning the task-relevant information
is distributed broadly, not concentrated in these 4 heads.

**The task has limited prompt diversity.** S3 stability and CM1 normative account
are borderline failures affected by having only 12 unique prompt templates. S3's
CV improves from 0.83 at 40 prompts to 0.54 at 200 prompts but still doesn't pass.

### Pass 1 verdict: **Proposed**

The 4-head circuit fails the most basic sufficiency test (A2). Under the framework's
verdict tiers, this places the claim at **Proposed**: the components are nominated
but not causally validated. The negative faithfulness specifically indicates the
circuit definition is incomplete rather than wrong.

---

## Metric recalibration

Two instruments (F1, A4) produced uninformative results on *both* IOI and epistemic,
indicating metric design issues rather than circuit quality issues.

### F1: Operation Specification

**Original metric:** R² of the linear OV approximation (`resid_pre @ W_V`) vs actual
`hook_z` output. This ignores the attention pattern entirely. Every attention head
fails because the function inherently depends on which tokens are attended to.

**Recalibrated metric:** Two complementary measures:
1. **Output consistency**: first principal component's variance ratio of hook_z
   across prompts. High ratio means the head's output lives in a low-dimensional
   subspace, indicating a well-specified function.
2. **Attention-weighted OV prediction**: use the actual attention pattern to
   compute the attended input, then predict via W_V. Accounts for attention.

Combined score = (consistency + max(attn_ov_r2, 0)) / 2. Pass: circuit > baseline.

### A4: Intermediate State Prediction

**Original metric:** Linear prediction of receiver hook_z from sender hook_z in the
full d_head=64 dimensional space. With ~20 training samples for 64 dimensions, the
model overfits completely (train r=1.0) and baseline heads predict almost as well
as circuit heads because residual streams share information broadly.

**Recalibrated metric:** Spearman rank correlation of scalar logit attributions
(`z @ W_O @ (W_U[correct] - W_U[incorrect])`) between sender and receiver heads
across prompts. One scalar per prompt, so no overfitting is possible.

**Open question:** After recalibration, A4 fails on *both* IOI and epistemic (IOI:
rho=0.015, epistemic: rho=-0.16). This may indicate that A4 is inappropriate for
attention head circuits. Information flows through *which tokens are attended to*,
not through correlated activation magnitudes. The criterion may only apply to
MLP-heavy or feedforward circuits.

---

## Circuit discovery: activation patching scan

To address the "circuit is too small" finding, we ran a full activation patching
scan across all 144 attention heads and 12 MLP layers.

**Method:** For each prompt pair, the clean input is the epistemic version.
The corrupted version replaces the epistemic prefix tokens ("I think" → random
tokens of same length, preserving sequence length). For each component, patch
the clean activation into the corrupted forward pass and measure logit-diff
recovery.

### Attention heads (top 20 by |effect|)

| Rank | Head | Effect | Original circuit? |
|------|------|-------:|:-----------------:|
| 1 | L0H8 | +0.425 | |
| 2 | L3H8 | +0.406 | |
| 3 | L4H6 | -0.404 | |
| 4 | L0H4 | +0.355 | |
| 5 | L5H3 | +0.333 | |
| 6 | L0H0 | +0.320 | |
| 7 | L0H9 | -0.303 | |
| 8 | L0H6 | -0.303 | |
| 9 | L2H10 | +0.262 | |
| 10 | L5H2 | -0.249 | |
| 11 | L0H7 | -0.231 | |
| 12 | L10H0 | +0.212 | |
| 13 | L2H1 | -0.207 | |
| 14 | L0H2 | -0.199 | |
| 15 | L7H9 | +0.197 | |
| 16 | L4H4 | +0.196 | |
| **17** | **L6H5** | **+0.192** | **Yes (detector)** |
| 18 | L4H9 | +0.191 | |
| 19 | L4H3 | -0.184 | |
| 20 | L4H0 | +0.176 | |
| ... | | | |
| 84 | L9H2 | +0.041 | Yes (integrator) |
| 108 | L9H5 | +0.018 | Yes (integrator) |
| 116 | L10H5 | +0.015 | Yes (executor) |

### MLP layers

| Layer | Effect |
|-------|-------:|
| MLP0 | +1.313 |
| MLP3 | -0.455 |
| MLP10 | -0.379 |
| MLP4 | -0.347 |
| MLP7 | +0.270 |
| MLP9 | +0.263 |

### Findings

1. **The original circuit captures a small fraction of the computation.** The 4
   hypothesized heads rank 17th, 84th, 108th, and 116th out of 144. Only L6H5
   has a meaningful effect (+0.19). Three of the four heads have effects smaller
   than the median head.

2. **Early layers dominate.** 8 of the top 10 heads are in layers 0-5. The
   epistemic prefix ("I think") is processed early, not in the mid-to-late
   layers the original hypothesis targeted.

3. **MLP0 has the largest effect of any component** (+1.31, 3x the top attention
   head). It transforms the epistemic token embeddings before any attention
   occurs.

4. **Negative-effect heads are critical.** L4H6 (-0.40), L0H9 (-0.30), L0H6
   (-0.30) actively suppress or redirect the epistemic signal. A complete
   circuit must include suppression as well as enhancement.

5. **51 heads have |effect| > 0.10.** The computation is highly distributed.
   Unlike IOI's clean 15-head circuit, epistemic framing may not admit a sparse
   circuit description. This is consistent with epistemic framing being a
   shallow linguistic feature (a 2-token prefix) rather than a deep syntactic
   computation.

### Proposed expanded circuit (attention heads only, |effect| > 0.15)

**Positive effect:**
L0: H0, H4, H8 / L2: H10 / L3: H8 / L4: H0, H4, H9 / L5: H3 /
L6: H5 / L7: H9 / L10: H0

**Negative effect:**
L0: H2, H6, H7, H9 / L2: H1, H5, H8 / L4: H3, H6 / L5: H2

~25 heads total. This is much larger than the original 4-head circuit but
still a fraction of the full model (25/144 = 17%).

---

## Pass 2: Recalibrated instruments on original circuit

After recalibrating F1 and A4:

| Instrument | Criterion | IOI | Epistemic | Change |
|------------|-----------|:---:|:---------:|--------|
| 70 | F1 Op Specification | PASS (0.146) | PASS (0.212) | Was excluded, now passes both |
| 79 | A4 Intermediate State | FAIL (rho=0.02) | FAIL (rho=-0.16) | Recalibrated but fails by design |

**Updated totals: IOI 9/12 pass. Epistemic 4/12 pass.**

F1 now correctly identifies that circuit heads have higher output consistency
than random heads. L10H5 in the epistemic circuit has particularly high
consistency (0.77), a focused head even if it is not a major contributor
to this specific task.

A4 reveals that attention head circuits don't exhibit correlated logit
attributions along pathways. In IOI, sender and receiver heads contribute
to the logit diff independently (rho near 0). This is architecturally expected:
composition in attention works through pattern modification, not through
correlated scalar magnitudes.

---

## Summary

| | IOI (15 heads) | Epistemic (4 heads) |
|---|:-:|:-:|
| Pass 1 (10 instruments) | 8/10 | 3/10 |
| Pass 2 (12 instruments, recalibrated) | 9/12 | 4/12 |
| Verdict tier | Mechanistically Supported | Proposed |
| Description mode | Impl-Connectomic | Impl-Topographic |

The framework successfully:

- **Discriminates** between a validated circuit (IOI, 9/12) and an underspecified
  one (epistemic, 4/12).
- **Diagnoses** the specific failure mode (circuit too small, negative faithfulness)
  and points toward remediation (expand via activation patching).
- **Identifies broken metrics** (F1 and A4) through the pattern of failing on
  *both* strong and weak circuits, distinguishing instrument problems from
  circuit problems.
- **Supports iteration.** The activation patching scan found that the top 16
  heads by effect are all outside the original circuit, directly confirming
  the framework's diagnosis.

Importantly, the framework is not a finished product. It is itself refined
through use. This evaluation exposed concrete gaps: no MLP-level criteria,
no suppression criteria, metrics calibrated for clean sequential circuits that
break on distributed computations. Each new circuit evaluated widens coverage
and sharpens existing instruments. The framework improves by the same iterative
process it prescribes: apply, find failures, diagnose whether the failure is in
the claim or the framework, fix accordingly.

### Next steps

#### Expand the circuit and re-evaluate

The activation patching scan identified ~25 heads with |effect| > 0.15 that
form the basis of an expanded circuit. Two variants to test:

- **Attention-only (25 heads):** Register the expanded head set from the patching
  scan in `circuit.py` with functional roles assigned by layer depth (early
  processors, mid composers, late routers, suppression heads). Re-run all 12
  node-level instruments. Expected improvement: S1 passes (more heads means
  stronger distributional separation), A2 composition goes positive, R3 passes
  (circuit now covers more of the information).

- **Attention+MLP variant:** Include MLP0 (effect +1.31) and MLP7 (+0.27)
  alongside the 25 heads. Keep this as a *separate* circuit definition so the
  framework evaluates them independently and reveals whether MLPs are part of
  the mechanism or just infrastructure.

More prompt diversity is also needed. The current 12 epistemic pairs limit
stability metrics (S3) and normative account coverage (CM1). Adding varied
epistemic verbs ("I suspect", "I'm confident", "I doubt", "we know that",
"scientists believe") and varying sentence structure would give the stability
instruments enough diversity to produce meaningful results.

#### Run edge-level instruments (G1-G5)

Five Graph Structure criteria test the circuit at the *edge* level, not just
which components are involved but how information flows between them. These
are necessary to advance from Implementational-Topographic (which components)
to Implementational-Connectomic (how they're wired). Scripts exist at
`src/instruments/structural/edge_analysis/`:

- **G1: Path Identification** (`82_path_identification.py`). Can specific
  information flow paths be traced through the circuit? Uses path patching:
  for each edge (upstream to downstream head), measure how much removing the
  upstream head's contribution at the downstream layer changes the output.
  Computes *specificity*: the ratio of edge effect on task-relevant prompts
  vs control prompts. Pass: at least one path with specificity > 5x. This
  distinguishes task-specific wiring from general residual stream propagation.

- **G2: Edge Necessity** (`83_edge_necessity.py`). Are specific edges
  individually necessary for the computation? For each edge, ablate just
  that edge (mean-ablate the upstream head's contribution at the downstream
  position) and measure logit diff drop. An edge is "necessary" if ablating
  it causes >5% drop. Pass: at least 50% of edges are individually necessary.
  This filters out redundant or decorative edges in the graph.

- **G3: Path Specificity** (`84_path_specificity.py`). Do different input
  conditions route through different paths? Compute edge effect patterns
  for task-relevant vs control conditions, then Spearman-correlate them.
  Pass: rho < 0.5, meaning different conditions use different edges. This
  is evidence that the circuit's wiring is functionally meaningful, not just
  an artifact of general information flow.

- **G4: Compositional Sufficiency** (`85_compositional_sufficiency.py`).
  Does the identified subgraph reproduce the full computation? Ablate all
  non-circuit heads and measure what fraction of the original logit diff is
  recovered. This is faithfulness reframed at the graph level, with
  per-band breakdowns showing which stages of the circuit carry the load.
  Pass: recovery > 30%.

- **G5: Graph Minimality** (`86_graph_minimality.py`). Is the edge set
  minimal, or does it include unnecessary connections? Combines edge
  necessity (magnitude test) with directional relevance (the drop must
  be in the task-relevant direction, not arbitrary). Pass: at least 80%
  of edges are necessary. A high minimality ratio means the graph
  description is tight, with no padding.

Running G1-G5 on both the 4-head and 25-head circuits will reveal whether
the expanded circuit has genuine structure or is just a bag of loosely
related heads.

#### Edge Attribution Patching (EAP)

The node-level activation patching scan tells us *which components* matter
but not *which connections between components* carry the signal. EAP
(Syed et al. 2023) extends activation patching to edges: for each pair
of components (sender, receiver), measure the effect of patching just the
sender's contribution to the receiver. This produces a weighted directed
graph that can be thresholded into a circuit.

EAP would replace the hand-assigned pathways in `circuit.py` with
data-driven edges, directly enabling the G1-G5 instruments. It is also the
standard methodology for circuit discovery in the literature, so running it
on epistemic framing would make the evaluation comparable to published
circuit analyses.

#### Criteria gaps exposed by this study

The current 27+5 criteria + 16 proposed criteria have coverage holes that this
case study makes visible:

**MLP-level criteria (none exist).** All current instruments operate on attention
heads. MLP0 is the single largest contributor to epistemic framing (+1.31 effect)
but no criterion tests MLP function specification, MLP necessity, or MLP-attention
composition. Proposed:
- [ ] **M-F1: MLP Operation Specification** — characterize neuron-level or
      subspace-level function of circuit MLPs (extend F1 to feedforward layers)
- [ ] **M-I1: MLP Necessity** — ablation of individual MLP layers within the circuit
- [ ] **M-A2: MLP-Attention Composition** — test whether MLPs and attention heads
      compose as the algorithmic description predicts

**Activation/neuron-level criteria (none exist).** The framework operates at the
component level (heads, layers) but not at the individual neuron or feature level.
For distributed circuits like epistemic framing where 51 heads participate:
- [ ] **N-R1: Feature Decodability** — are task-relevant features decodable from
      individual neuron activations (not just layer-level probes)?
- [ ] **N-S1: Activation Sparsity Profile** — do circuit neurons have sparser
      activation patterns than non-circuit neurons on task-relevant inputs?

**Negative/suppression criteria (none exist).** The current framework assumes circuit
components *contribute* to the computation. But 12 of the top 25 heads for epistemic
framing have *negative* effects, meaning they suppress the epistemic signal. No
criterion tests whether suppression is part of the mechanism.
- [ ] **I-S1: Suppression Specificity** — are negative-effect components specifically
      suppressing task-relevant signals or just adding noise?
- [ ] **I-S2: Suppression Necessity** — is removing a suppression head as damaging
      as removing an enhancement head?

**Algorithmic-level criteria (weak coverage).** Current criteria A1-A4 test whether
the circuit implements a step-by-step procedure, but the tests are designed for
clean sequential circuits (IOI-style). For distributed circuits:
- [ ] **A5: Parallel vs Sequential Decomposition** — can the computation be
      decomposed into parallel streams (early-layer processing vs mid-layer
      composition) rather than a single sequential pathway?
- [ ] **A6: Information Bottleneck Location** — where in the circuit does
      task-relevant information get concentrated from distributed to localized?

**Computational-level criteria (gap for shallow features).** The epistemic framing
task may be "too easy," a 2-token prefix that the model handles via shallow
pattern matching rather than deep computation. No criterion distinguishes:
- [ ] **CM3: Computational Depth** — does the circuit use multi-step composition
      or single-layer pattern matching? (Proxy: compare circuit performance to a
      bag-of-words baseline that just checks for epistemic tokens.)
- [ ] **CM4: Task Complexity Alignment** — is the circuit's complexity proportionate
      to the task's complexity? A 25-head circuit for a 2-token prefix detection
      suggests the "circuit" is really just the model's general language processing.

#### Framework methodology improvements

- [ ] **Threshold calibration protocol.** Several instruments have thresholds
      (S3: CV<0.2, CM1: separation>2.0, A4: uplift>1.3) that were set by
      intuition. Need a principled calibration: run instruments on 5+ known
      circuits, set thresholds at the point that separates validated from
      unvalidated circuits.
- [ ] **Circuit size normalization.** IOI has 15 heads, epistemic has 4. Some
      metrics (faithfulness, composition) inherently favor larger circuits.
      Instruments could additionally report size-normalized scores.
- [ ] **Negative result protocol.** When a circuit fails most instruments, the
      framework needs a structured way to recommend: (a) expand circuit,
      (b) change description mode, (c) abandon the claim. Currently this is
      implicit in the diagnosis step.
- [ ] **Multi-pass evaluation template.** This case study's Pass 1, diagnose,
      refine, Pass 2 pattern could be formalized as a standard evaluation
      protocol, not just an example.
