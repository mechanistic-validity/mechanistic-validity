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

## Pass 3: Four-circuit comparison

Following the framework's diagnosis from Pass 2 (circuit too small), we created
three alternative circuits using systematic discovery methods and evaluated all
12 instruments on each:

| Circuit | Method | Size | Rationale |
|---------|--------|------|-----------|
| Original | Hand-picked | 4 heads | Hypothesis-driven: detector, integrator, executor |
| Tight | Act-patch, \|effect\|>0.20 | 13 heads | Strongest individual contributors |
| EAP | Edge Attribution Patching, top-15 | 15 heads | Strongest edge connectivity |
| Expanded | Act-patch, \|effect\|>0.15 | 32 heads | All significant contributors |

### Results

| Instrument | Criterion | Original (4) | Tight (13) | EAP (15) | Expanded (32) |
|------------|-----------|:------------:|:----------:|:--------:|:-------------:|
| 70 | F1 Op Spec | P(0.212) | F(0.169) | F(0.152) | F(0.168) |
| 71 | F2 Held-Out | F(0.541) | F(0.474) | F(0.488) | F(0.478) |
| 72 | F3 Replace | P(0.993) | P(0.990) | P(0.990) | P(0.988) |
| 73 | S1 Dist Char | P(1.676) | P(1.005) | P(2.713) | P(1.132) |
| 74 | S3 Stability | FAIL | FAIL | FAIL | FAIL |
| 75 | R1 Probe | P(0.667) | P(0.667) | P(0.667) | P(0.667) |
| 76 | R3 Causal Rep | FAIL | FAIL | FAIL | FAIL |
| 77 | A1 Procedure | F(0.333) | F(0.571) | P(0.714) | F(0.615) |
| 78 | A2 Composition | F(-0.125) | F(-0.058) | F(-0.139) | F(0.228) |
| 79 | A4 Intermed | F(-0.164) | F(0.032) | F(-0.016) | F(0.001) |
| 80 | CM1 Normative | F(1.600) | P(25.0) | F(1.600) | F(1.500) |
| 81 | CM2 Error Bnd | F(0.350) | F(0.350) | F(0.350) | F(0.400) |

### The circuit size tradeoff

The most important finding from Pass 3 is that different validity criteria
systematically prefer different circuit sizes. This is not an artifact of
threshold calibration — it reflects a genuine tension in mechanistic
explanation between two competing desiderata:

**Specification criteria (F1, CM1) prefer small circuits.** The original
4-head circuit is the only one that passes F1 (operation specification).
The tight 13-head circuit achieves a dramatic 25x separation on CM1
(normative account). In both cases, fewer heads means more focused,
characterizable components. Adding the 19 weaker heads from the expanded
circuit dilutes the per-head specialization signal.

**Specialization and necessity are orthogonal.** The activation-patching
landscape reveals a sharp disconnect. L10H5 — unique to the original
circuit — has the highest F1 consistency of any head (0.773) but ranks
116th out of 144 heads by individual effect (+0.015). It is the most
task-specialized head in the model but one of the least individually
necessary. Conversely, L0H8 has the largest individual effect (+0.425)
but only moderate consistency (0.622). A circuit built on specialized
heads passes F1 but fails sufficiency tests; a circuit built on
necessary heads passes sufficiency but fails specification tests.

**Sufficiency criteria (A1, A2) prefer large circuits.** A2 (composition
test) shows a clear monotonic trend: Original -0.125, Tight -0.058,
Expanded +0.228. The expanded circuit still falls below the 0.3 pass
threshold, but it reverses the sign — meaning 32 heads are *almost*
sufficient to recover model performance. The original 4-head circuit
actively degrades performance when used in isolation.

**Instrument decomposition.** The 12 instruments decompose into three
behavioral categories:

1. **Discriminating** (3 instruments): F1, A1, and CM1 each pass for
   exactly one circuit (Original, EAP, and Tight respectively). These
   reveal genuine differences between discovery methods and circuit sizes.

2. **Universal pass** (3 instruments): F3, S1, and R1 pass for all
   circuits. F3 and R1 are genuinely size-invariant. S1 passes for all
   but varies dramatically by method (EAP=2.713 vs Tight=1.005).

3. **Universal fail** (6 instruments): S3, R3, A2, A4, CM2, and F2 fail
   for all circuits. Each failure has a distinct root cause — limited
   prompts (S3), insufficient selectivity (R3), distributed computation
   (A2), architectural mismatch (A4), flat difficulty gradient (CM2),
   or below-baseline predictability (F2).

**Suppression heads are computationally essential.** A2 pathway analysis
reveals that the expanded circuit's only positive-faithfulness pathways
route through "suppressor" heads (L7H8, L8H5, L8H8) — heads whose
individual ablation *improves* predictions. These heads normally
dampen the output, but in context they calibrate the circuit's
predictions. L8H8 is simultaneously the top EAP edge sender (most
connected node) and a suppressor (act-patch effect = -0.168). This
inhibitory mechanism is invisible to criteria that only consider
positive contributions, motivating the proposed suppression criteria.

### EAP vs activation patching

Two discovery methods produce largely non-overlapping circuits. The
corrected overlap matrix (verified programmatically):

```
           Original(4)  Tight(13)  EAP(15)  Expanded(32)
Original          4          0        1          1
Tight             0         13        2         13
EAP               1          2       15          5
Expanded          1         13        5         32
```

Only two heads appear in 3+ circuits: **L2H10** and **L10H0** (both in
Tight, EAP, and Expanded). No head appears in all four. The tight
circuit is a perfect subset of expanded (both from activation patching
at different thresholds). EAP finds connectivity-important heads (high
edge involvement), activation patching finds individually-important
heads (high individual ablation effect).

The two methods excel on complementary validity criteria:

| Criterion | Act-patch (Tight) | EAP | Winner |
|-----------|:-:|:-:|--------|
| CM1 Normative | **25.0x PASS** | 1.6x FAIL | Act-patch |
| A1 Procedure | 0.571 FAIL | **0.714 PASS** | EAP |
| S1 Distributional | 1.005 | **2.713** | EAP |
| A2 Composition | -0.058 | -0.139 | Act-patch |

Act-patch finds "specialists" — heads whose individual removal changes
the output. EAP finds the "backbone" — heads with high edge connectivity
that form a clean processing pipeline. Specialists produce better
normative accounts (CM1); the backbone produces better procedural
descriptions (A1). Neither is strictly better — the "right" method depends
on which validity property matters most for the claim.

The divergence is dramatic: no single head appears in all four circuits.
The original hand-picked heads (L6H5, L9H2, L9H5, L10H5) share zero
overlap with the tight circuit — the "detector" L6H5 ranks only 17th by
activation patching effect. EAP and activation patching share only 5/15
heads. Each method finds genuinely different components of the model's
epistemic processing.

**Layer distribution reveals method bias.** Activation patching is
biased toward early layers (tight: 8/13 heads in L0-L2) because early
heads have large activation magnitudes and position-based patterns.
EAP is more balanced across layers (3 early, 5 mid, 7 late) because it
measures gradient-weighted information flow between layers, naturally
surfacing multi-hop pathways. The original hand-picked circuit is
exclusively late-layer (L6-L10), reflecting the human tendency to look
at output-proximal components.

**The L10H5 anomaly explains why the Original circuit uniquely passes
F1.** L10H5 has an F1 consistency of 0.773 — 24% higher than any head
in any other circuit. It is unique to the Original circuit (not found
by either discovery method). Its high consistency but low
activation-patching effect means it doesn't change the output much, but
when it does, it's highly task-specific. This is the kind of head
hand-picking finds but automated discovery misses.

### Three orthogonal axes of circuit membership

This case study reveals that "circuit membership" is not a single
property but splits along at least three independent axes:

1. **Necessity** (activation patching): does ablating this head degrade
   task performance? High-necessity heads form the expanded circuit.
   L0H8 is the archetype (+0.425 effect).

2. **Specialization** (F1 consistency): does this head's output
   correlate with task-relevant features? High-specialization heads
   form the original circuit. L10H5 is the archetype (0.773 consistency
   but only +0.015 effect).

3. **Connectivity** (EAP edge involvement): does this head relay
   task-relevant information between layers? High-connectivity heads
   form the EAP circuit. L8H8 is the archetype (top edge sender,
   -0.168 individual effect — a suppressor that is also the main
   information broadcaster).

These axes are genuinely orthogonal: L10H5 has high specialization but
low necessity and low connectivity. L0H8 has high necessity but moderate
specialization and low connectivity. L8H8 has high connectivity but
negative necessity.

Different validity criteria test different axes: F1 tests specialization,
A2 tests necessity (ablation-based sufficiency), and A1 tests connectivity
(information flow ordering). No single circuit optimizes all three.

### What IOI has that epistemic doesn't

IOI passes 9/12 instruments. The 4 instruments where IOI passes but
all epistemic circuits fail reveal what makes a circuit "real":

| Instrument | IOI | Best epistemic | Gap |
|------------|-----|---------------|-----|
| A2 Composition | 0.320 (pass) | 0.228 (expanded) | IOI's 15 heads are sufficient |
| CM2 Error Bnd | 0.750 (pass) | 0.400 (expanded) | IOI has a difficulty gradient |
| R3 Causal Rep | 0.950 (pass) | all fail | IOI's info is concentrated |
| S3 Stability | FAIL | all fail | Both fail (not IOI-specific) |

The epistemic task's fundamental problem is distribution: task-relevant
information is spread across many heads, not concentrated in a
discoverable subset. IOI's information flow is localized (S-inhibition
heads, name mover heads) while epistemic framing appears to be a
distributed, shallow computation — possibly just a 2-token prefix
pattern that the model processes through its general language pipeline
rather than a dedicated circuit.

---

## Summary

| | IOI (15) | Original (4) | Tight (13) | EAP (15) | Expanded (32) |
|---|:-:|:-:|:-:|:-:|:-:|
| Pass count | 9/12 | 4/12 | 4/12 | 4/12 | 3/12 |
| Verdict tier | Mech. Supported | Proposed | Proposed | Proposed | Proposed |
| Best at | — | F1, F3 | CM1 | A1, S1 | A2 |

The framework successfully:

- **Discriminates** between a validated circuit (IOI, 9/12) and underspecified
  ones (epistemic, 3-4/12 across all variants).
- **Diagnoses** the specific failure mode (circuit too small, negative faithfulness)
  and points toward remediation (expand via activation patching).
- **Identifies broken metrics** (F1 and A4) through the pattern of failing on
  *both* strong and weak circuits, distinguishing instrument problems from
  circuit problems.
- **Supports iteration.** The activation patching scan found that the top 16
  heads by effect are all outside the original circuit, directly confirming
  the framework's diagnosis.
- **Reveals structural tradeoffs.** No single circuit size satisfies all validity
  criteria simultaneously. The framework surfaces this as a genuine property
  of mechanistic explanation: specification and sufficiency are in tension.
  The tight circuit (13 heads) is the best compromise, achieving the only
  non-original CM1 pass while maintaining reasonable sufficiency scores.
- **Compares discovery methods.** EAP and activation patching produce
  complementary circuits that excel on different criteria. The framework
  evaluates each independently, letting the researcher choose which validity
  properties matter most for their claim.

Importantly, the framework is not a finished product. It is itself refined
through use. This evaluation exposed concrete gaps: no MLP-level criteria,
no suppression criteria, metrics calibrated for clean sequential circuits that
break on distributed computations. Each new circuit evaluated widens coverage
and sharpens existing instruments. The framework improves by the same iterative
process it prescribes: apply, find failures, diagnose whether the failure is in
the claim or the framework, fix accordingly.

### Completed since Pass 2

- [x] **Expanded circuit (32 heads)** created and evaluated on all 12
  instruments. A2 improved from -0.125 to +0.228 (still below 0.3 threshold).
  S1 confirmed (1.132x). F1 dropped below baseline (0.168 vs 0.202).
- [x] **Tight circuit (13 heads)** at |effect|>0.20, achieving the best balance:
  CM1 passes at 25.0x separation, A1 reaches 0.571.
- [x] **EAP circuit (15 heads)** from Edge Attribution Patching. Discovers a
  largely different circuit (only 5/15 overlap with expanded). Highest S1
  (2.713x) but fails CM1 (1.6x).
- [x] **EAP discovery script** (`src/instruments/data/epistemic_eap.py`). Found
  9504 significant edges, top heads: L1H10 (163.5), L8H10 (155.6), L2H10
  (153.0). EAP vs act-patch comparison documented.
- [x] **G4 compositional sufficiency rewrite** — now measures superadditivity
  (full circuit recovery vs max single band) instead of duplicating A2.
- [x] **Four-circuit comparison** revealing the specification-vs-sufficiency
  tradeoff.

### Remaining next steps

#### Complete instrument coverage

All 12 node-level instruments are complete on all 4 circuits (48 evaluations).

**G3 path specificity (complete)**: all 4 circuits fail. Edge effects are
highly correlated between task and control prompts (rho=0.557-1.000), meaning
edges carry general processing rather than task-specific signal. Tight circuit
is closest to passing (rho=0.557 vs threshold 0.5). IOI also fails G3
(rho=0.977), suggesting this is a hard criterion for attention circuits.

**G2 edge necessity (complete)**: all 4 circuits fail. Only 1 edge out of 232
in the expanded circuit is individually necessary: L8H8→L10H0 (drop=5.0%),
confirming L8H8 as the key hub. IOI has 12/44 necessary edges (27%) vs
epistemic's 1/232 (0.4%) — a 60x difference in edge necessity density.

G1 (path identification) still running (~290 min CPU). G4 and G5 queued after.

#### Attention+MLP circuit variant

MLP0 has the largest single-component effect (+1.31, 3x the top attention
head). Creating a circuit that includes MLP0 alongside attention heads would
test whether MLPs are part of the mechanism or just infrastructure. No
current instrument supports MLP-level evaluation, so this would also drive
new instrument development.

#### More prompt diversity

The current 12 epistemic prompt templates cause S3 (stability) to fail
across all circuits. Adding varied epistemic markers ("I suspect", "I'm
confident", "I doubt", "we know that", "scientists believe") and varying
sentence structures would address this.

#### Edge-level instruments (G1-G5)

Five graph-structure instruments (G1 Path Identification through G5 Graph
Minimality) test the circuit at the edge level. With EAP edge data now
available, these instruments can measure whether the discovered edges carry
genuine task-specific signal. Running G1-G5 on all four circuits would
complete the evaluation and enable Implementational-Connectomic description.

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
