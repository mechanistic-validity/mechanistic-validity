# Epistemic Framing Circuit — Lab Notebook

This notebook documents the step-by-step process of discovering and evaluating
an epistemic framing circuit in GPT-2 small using the mechanistic validity
framework. Each section explains what was tried, why, and what was learned.
Commands are included for reproducibility.

All commands assume `cwd` is the mechanistic-validity repo root.

---

## 1. Setup

The mechanistic validity framework provides instruments (scripts that run
specific tests) organized by what they measure:

- **F1-F3**: Function specification (can we characterize what each component does?)
- **S1, S3**: Statistical properties (are measurements stable and distinctive?)
- **R1, R3**: Representational (is task information decodable and causal?)
- **A1, A2, A4**: Algorithmic (does the circuit implement a procedure?)
- **CM1, CM2**: Computational (does the circuit match a normative account?)
- **G1-G5**: Graph structure (are the edges meaningful?)

Each instrument produces a JSON file with pass/fail, numeric scores, and
metadata. The same instrument runs on any circuit definition.

```bash
# Example: run F1 (operation specification) on the IOI circuit
uv run --python 3.11 python src/instruments/causal/mdc_glennan/70_operation_specification.py \
    --tasks ioi --n-prompts 40 --device cpu
```

---

## 2. The Task

**Epistemic framing**: GPT-2 small processes sentences prefixed with epistemic
stance markers ("I think", "I believe", "I know") and still completes the
factual content correctly. 12 prompt pairs:

- "I think the capital of France is" → Paris
- "The capital of France is" → Paris

The circuit should capture the components that handle the epistemic prefix
without disrupting the factual prediction.

---

## 3. Pass 1: Original 4-Head Circuit

### Circuit definition

The initial circuit was identified through 13 targeted experiments testing
specific hypotheses about epistemic processing:

| Role | Heads | Hypothesized function |
|------|-------|----------------------|
| Detector | L6H5 | Detects epistemic markers |
| Integrator | L9H2, L9H5 | Integrates stance with factual content |
| Executor | L10H5 | Routes final token prediction |

### Running all 12 instruments

```bash
# Full instrument paths (run 3-4 at a time to avoid overloading CPU):
TASKS="epistemic_framing epistemic_tight epistemic_eap epistemic_expanded"
ARGS="--n-prompts 40 --device cpu"

# Function specification (F1-F3)
.venv/bin/python3 src/instruments/causal/mdc_glennan/70_operation_specification.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/causal/mdc_glennan/71_held_out_prediction.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/causal/mdc_glennan/72_replacement_test.py --tasks $TASKS $ARGS

# Statistical (S1, S3)
.venv/bin/python3 src/instruments/measurement/bootstrap_stability/73_distributional_characterization.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/measurement/bootstrap_stability/74_distributional_stability.py --tasks $TASKS $ARGS

# Representational (R1, R3)
.venv/bin/python3 src/instruments/representational/linear_probe/75_probe_decodability.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/representational/linear_probe/76_causal_representation.py --tasks $TASKS $ARGS

# Algorithmic (A1, A2, A4)
.venv/bin/python3 src/instruments/causal/mdc_glennan/77_procedure_specification.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/causal/mdc_glennan/78_composition_test.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/causal/counterfactual_das/79_intermediate_state_prediction.py --tasks $TASKS $ARGS

# Computational (CM1, CM2)
.venv/bin/python3 src/instruments/behavioral/generalization_gap/80_normative_account.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/behavioral/generalization_gap/81_error_boundary_analysis.py --tasks $TASKS $ARGS

# Graph structure (G1-G5)
.venv/bin/python3 src/instruments/structural/edge_analysis/82_path_identification.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/structural/edge_analysis/83_edge_necessity.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/structural/edge_analysis/84_path_specificity.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/structural/edge_analysis/85_compositional_sufficiency.py --tasks $TASKS $ARGS
.venv/bin/python3 src/instruments/structural/edge_analysis/86_graph_minimality.py --tasks $TASKS $ARGS
```

### Results

| Script | Criterion | Result | Score |
|--------|-----------|--------|-------|
| 70 | F1 Op Specification | PASS | 0.212 |
| 71 | F2 Held-Out Prediction | PASS | 0.541 |
| 72 | F3 Replacement Test | PASS | 0.993 |
| 73 | S1 Distributional Characterization | PASS | 1.676x |
| 74 | S3 Distributional Stability | FAIL | CV too high |
| 75 | R1 Probe Decodability | PASS | 0.667 selectivity |
| 76 | R3 Causal Representation | FAIL | ctrl IIA=0.80 |
| 77 | A1 Procedure Specification | FAIL | 0.333 |
| 78 | A2 Composition Test | FAIL | -0.125 faithfulness |
| 79 | A4 Intermediate State | FAIL | rho=-0.164 |
| 80 | CM1 Normative Account | FAIL | 1.6x separation |
| 81 | CM2 Error Boundary | FAIL | 0.35 |

**Score: ~5/12 pass.** The critical failure is A2 (composition test): keeping
only the 4 circuit heads and ablating everything else makes predictions *worse*
than the full model. The circuit is too small.

---

## 4. Metric Recalibration

Two instruments (F1, A4) failed on both IOI and epistemic, indicating metric
problems rather than circuit problems.

### F1: Operation Specification

**Problem:** The original R² metric (`resid_pre @ W_V` vs actual `hook_z`)
ignores the attention pattern entirely. Every attention head fails because
the function inherently depends on which tokens are attended to.

**Fix:** Two metrics: (1) output consistency (first-PC variance ratio of
hook_z across prompts), (2) attention-weighted OV prediction. Combined score.
Pass: circuit heads > random baseline.

### A4: Intermediate State Prediction

**Problem:** Linear prediction in 64-dim hook_z space overfits with ~20
training samples (train r=1.0 for both circuit and random heads).

**Fix:** Spearman rank correlation of scalar logit attributions
(`z @ W_O @ (W_U[correct] - W_U[incorrect])`) between sender and receiver.
One scalar per prompt, no overfitting possible.

**Ongoing issue:** A4 still fails on both IOI and epistemic after recalibration.
This may indicate the criterion is inappropriate for attention head circuits.
Information flows through *pattern modification*, not correlated magnitudes.

---

## 5. Circuit Discovery: Activation Patching

The "circuit too small" finding prompted a full scan.

```bash
uv run --python 3.11 python src/instruments/data/epistemic_act_patch.py \
    --n-prompts 40 --device cpu
```

### Key findings

1. **The original 4 heads rank 17th, 84th, 108th, 116th** out of 144.
   Only L6H5 has a meaningful effect (+0.19).

2. **Early layers dominate.** 8 of the top 10 heads are in layers 0-5.

3. **MLP0 has the largest effect** (+1.31, 3x the top attention head).

4. **51 heads have |effect| > 0.10.** The computation is highly distributed.

### Circuit definitions at different thresholds

| Circuit | Threshold | Heads | Description |
|---------|-----------|-------|-------------|
| Original | hand-picked | 4 | L6H5, L9H2, L9H5, L10H5 |
| Tight | |effect| > 0.20 | 13 | Top contributors, comparable to IOI (15) |
| EAP | top 15 by edge score | 15 | Edge-based discovery (different heads!) |
| Expanded | |effect| > 0.15 | 32 | All significant contributors |

```bash
# All circuits registered and runnable:
uv run --python 3.11 python -c "
import sys; sys.path.insert(0, 'src/instruments'); sys.path.insert(0, 'src')
from _common import get_circuit_info
for task in ['epistemic_framing', 'epistemic_tight', 'epistemic_eap', 'epistemic_expanded']:
    _, heads, edges = get_circuit_info(task)
    print(f'{task}: {len(heads)} heads, {len(edges)} edges')
"
```

---

## 6. EAP vs Activation Patching: Different Methods, Different Circuits

Edge Attribution Patching discovers circuits by measuring the gradient-based
contribution of each edge (sender output dot receiver gradient), while
activation patching measures the causal effect of each individual component.

```bash
uv run --python 3.11 python src/instruments/data/epistemic_eap.py \
    --n-prompts 40 --device cpu --threshold 0.005
```

### Comparison

Only **9 out of 30 heads** overlap between EAP top-30 and act-patch top-32.
EAP's top head (L1H10, score=163.5) has an activation patching effect of
only +0.057 (rank ~60). Conversely, act-patch's top head (L0H8, effect=+0.425)
doesn't appear in EAP's top 15.

This divergence is expected: activation patching measures *individual node*
importance (would removing this head change the output?) while EAP measures
*edge connectivity* (does this head's output influence downstream heads?).
A head can be important individually without being well-connected, and
vice versa.

The practical implication: running both methods gives complementary circuits,
and the framework can evaluate each independently to see which produces
a better mechanistic account.

---

## 7. The Circuit Size Tradeoff

The most striking finding from running instruments on all four circuits is
that different instruments prefer different circuit sizes. This is not a
flaw in the instruments — it reveals a genuine tension in mechanistic
explanation.

### Instruments that prefer smaller circuits

- **F1 (Operation Specification)**: Smaller circuits have more specialized heads.
  Original (4 heads): PASS at 0.212. Expanded (32 heads): FAIL at 0.168.

- **CM1 (Normative Account)**: Smaller circuits give cleaner separation between
  circuit and non-circuit behavior. Tight (13 heads): PASS at 25.0x.
  Expanded (32 heads): FAIL at 1.5x.

### Instruments that prefer larger circuits

- **A2 (Composition Test)**: Faithfulness requires sufficient coverage.
  Original (4 heads): FAIL at -0.125 (negative = circuit hurts predictions).
  Expanded (32 heads): [pending].

- **A1 (Procedure Specification)**: More heads = more pathways to trace.
  Original: FAIL at 0.333. Expanded: FAIL at 0.615 (closer to passing).

### Instruments that are size-invariant

- **R1 (Probe Decodability)**: Measures information at the layer level, not
  head level. Score = 0.667 for both original and expanded.

- **S3 (Distributional Stability)**: Depends on prompt diversity, not circuit size.
  Fails on all circuits (CV too high with only 12 prompt templates).

### What this means

The "right" circuit size depends on what validity criterion matters most.
A sufficiency claim (A2) needs enough heads to account for the computation.
A specification claim (F1, CM1) needs focused heads whose function can be
characterized. The framework surfaces this tradeoff explicitly rather than
hiding it behind a single pass/fail score.

---

## 8. Instrument Results by Circuit (complete)

All 12 node-level instruments on all 4 circuits. Each cell shows P(ass)/F(ail)
and the metric value. Instruments run with `--n-prompts 40 --device cpu`.

| Instrument | Criterion | Original (4) | Tight (13) | EAP (15) | Expanded (32) |
|------------|-----------|:------------:|:----------:|:--------:|:-------------:|
| 70 | F1 Op Spec | P(0.212) | F(0.169) | F(0.152) | F(0.168) |
| 71 | F2 Held-Out | F(0.541) | F(0.474) | F(0.488) | F(0.478) |
| 72 | F3 Replace | P(0.993) | P(0.990) | P(0.990) | P(0.988) |
| 73 | S1 Dist Char | P(1.676) | P(1.005) | P(2.713) | P(1.132) |
| 74 | S3 Stability | FAIL | FAIL | FAIL | FAIL |
| 75 | R1 Probe | P(0.667) | P(0.667) | P(0.667) | P(0.667) |
| 76 | R3 Causal Rep | FAIL | FAIL | FAIL | FAIL |
| 77 | A1 Procedure | F(0.333) | F(0.571) | **P(0.714)** | F(0.615) |
| 78 | A2 Composition | F(-0.125) | F(-0.058) | F(-0.139) | F(0.228) |
| 79 | A4 Intermed | F(-0.164) | F(0.032) | F(-0.016) | F(0.001) |
| 80 | CM1 Normative | F(1.600) | P(25.0) | F(1.600) | F(1.500) |
| 81 | CM2 Error Bnd | F(0.350) | F(0.350) | F(0.350) | F(0.400) |

**Pass count:** Original 4/12, Tight 4/12, EAP 4/12, Expanded 3/12

Note: Original F2 was initially called PASS in Pass 1 (value=0.541) but
with proper random-baseline comparison (baseline=0.692), all circuits
fail F2. Updated from 5/12 to 4/12.

Key patterns emerging:
- **EAP is the only circuit to pass A1** (procedure specification, 0.714).
  Edge-based discovery produces natural processing pipelines.
- **Tight is the only circuit to pass CM1** (normative account, 25.0x).
  Act-patch finds the heads that distinguish epistemic from non-epistemic.
- **Original is the only circuit to pass F1** (operation specification, 0.212).
  Hand-picked heads are the most individually characterizable.
- **S1 is size-invariant but method-variant**: EAP=2.713, Original=1.676,
  Expanded=1.132, Tight=1.005. EAP heads have the most distinctive
  activation patterns despite not being individually important.
- **R1** (probe decodability) is perfectly size-invariant at 0.667.
- **R3** fails uniformly because control IIA is too high — the DAS
  intervention is not selective enough to distinguish circuit from non-circuit.
- **A4** fails universally near zero — consistent with information flowing
  through attention pattern modification rather than correlated magnitudes.

---

## 9. Detailed Findings

### 9.1 F1 (Operation Specification): Smaller = More Specialized

Only the original 4-head circuit passes F1. Its heads are hand-picked for
specific roles (detector, integrator, executor), so each one has a
distinctive activation pattern. Larger circuits dilute specialization — the
32 extra heads in the expanded circuit include many with generic activation
profiles that drag the mean below random baseline.

**Original (4):** combined=0.212, baseline=~0.20 → PASS (barely)
**Tight (13):** combined=0.169, baseline=0.202 → FAIL
**EAP (15):** combined=0.152, baseline=0.170 → FAIL
**Expanded (32):** combined=0.168, baseline=0.202 → FAIL

### 9.2 CM1 (Normative Account): Tight Circuit Dominates

The tight circuit achieves a dramatic 25x separation ratio on negation
features — the 13 strongest-effect heads are precisely the ones that
differentiate epistemic from non-epistemic processing.

**Original (4):** max_sep=1.6x → FAIL (threshold: 2x)
**Tight (13):** max_sep=25.0x (negation) → PASS
**EAP (15):** max_sep=1.6x (conjunctions) → FAIL
**Expanded (32):** max_sep=1.5x (quantifiers) → FAIL

Per-feature breakdown on the negation feature across circuits:

| Circuit | Epistemic rate | Non-epistemic rate | Ratio |
|---------|:-:|:-:|:-:|
| Tight (13) | 0.25 | 0.00 | **25.0** |
| Expanded (32) | 0.10 | 0.15 | 0.67 (reversed) |
| EAP (15) | 0.05 | 0.20 | 0.25 (reversed) |

The tight circuit's heads specifically discriminate on negation —
epistemic text triggers negation-related activations but non-epistemic
text does not. The expanded and EAP circuits show *reversed* patterns,
suggesting their extra heads include non-specific components that respond
to surface negation cues regardless of epistemic content.

**Caveat:** The 25x ratio comes from 40 prompts per condition with low
base rates (10 positives, 0 negatives). A single false positive in the
non-epistemic condition would drop it to ~5x. More prompt diversity
would strengthen this finding.

### 9.3 A2 (Composition Test): Suppression Heads Are the Key

Composition test measures whether keeping only circuit heads (ablating
everything else) recovers model performance.

| Circuit | Full faithfulness | Best pathway | Worst pathway |
|---------|:-:|---|---|
| Original (4) | -0.125 | n/a | n/a |
| Tight (13) | -0.058 | -0.066 (all negative) | -0.151 |
| EAP (15) | -0.139 | -0.114 (all negative) | -0.204 |
| Expanded (32) | +0.228 | **+0.166** | -0.179 |

The expanded circuit is the only one with positive pathways, and
critically, the two positive pathways both route through
`late_suppressor` heads:

```
early_suppressor → mid_composer → late_suppressor → late_router: +0.166
early_processor  → mid_composer → late_suppressor → late_router: +0.108
```

All other pathways (including the full pipeline without suppressor) are
negative. This suggests the computation *requires* active suppression
in layers 7-8 to function. The tight and EAP circuits lack dedicated
suppression roles, which explains their negative faithfulness — they
include the processing heads but miss the suppression heads that
calibrate the output.

Including MLP0 (effect +1.31) would likely push the expanded circuit
past the 0.3 threshold, but the more interesting finding is that
suppression heads are computationally essential, not just noise.

**The suppressor paradox:** The late_suppressor heads (L7H8, L8H5, L8H8)
all have *negative* individual effects (-0.156 to -0.168): ablating them
makes the correct prediction *stronger*. They normally suppress the output.
But they're essential for circuit composition — the only positive pathways
route through them.

This makes mechanistic sense: the suppressor heads calibrate the circuit's
output by dampening competing signals. Without them (individual ablation),
the model over-predicts. With them but without the rest of the circuit
(composition test), the circuit can produce calibrated predictions.

L8H8 is particularly interesting: it's both a suppressor (act-patch
effect=-0.168) AND the top edge sender in EAP (edges to L10H0, L9H5,
L9H6 with scores 2.56, 2.54, 2.50). It suppresses the output while
being the most connected hub in the circuit. This is evidence for a
genuine inhibitory mechanism, not noise.

```bash
# Verify suppressor effects:
uv run --python 3.11 python -c "
import json
with open('src/instruments/data/epistemic_act_patch.json') as f:
    d = json.load(f)
for h in d['heads']:
    if (h['layer'], h['head']) in [(7,8), (8,5), (8,8)]:
        print(f'L{h[\"layer\"]}H{h[\"head\"]}: effect={h[\"effect\"]:+.4f}')
"
```

### 9.4 A1 (Procedure Specification): EAP Wins

A1 measures whether the circuit's roles form an ordered processing pipeline.
The EAP circuit is the **only circuit to pass** A1 (threshold: 0.7).

**Original (4):** ordering=0.333 (only 1/3 role pairs show causal ordering)
**Tight (13):** ordering=0.571, pathway_frac=0.681
**Expanded (32):** ordering=0.615, pathway_frac=0.530
**EAP (15):** ordering=0.714, pathway_frac=0.691 — **PASS**

Why EAP wins: it was designed based on edge connectivity, so its role
structure (early_relay → mid_hub → late_integrator → output) naturally
forms a sequential pipeline. Activation patching finds individually
important heads that don't necessarily form a clean processing chain.
EAP's 3 chains with high pathway fraction (0.691) show that the edge-based
roles carry most of the computation through ordered stages.

This creates a clean methodological split:
- **EAP** → best for A1 (algorithmic procedure) and S1 (distributional)
- **Act-patch tight** → best for CM1 (normative account)
- **Original hand-picked** → best for F1 (operation specification)

### 9.5 CM2 (Error Boundary): Universal Failure

All circuits fail CM2 with similar scores (0.35-0.40). The error boundary
instrument tests whether the circuit's errors align with the model's
difficulty gradient (easy prompts = high faithfulness, hard = low). But
the epistemic task has near-uniform model accuracy (0.925 across all
difficulty levels), so there's no real difficulty gradient to align with.
This is a limitation of the prompt set, not the circuit.

### 9.6 A4 (Intermediate State): Systematic Failure

A4 measures whether sender-receiver head pairs show correlated logit
attributions. All circuits fail with scores near zero. This suggests
information flows through attention pattern modification rather than
correlated activation magnitudes — the sender changes *what* the receiver
attends to, not the *magnitude* of its output. A4 may be architecturally
inappropriate for attention head circuits (it was designed for MLP neurons
where magnitude correlation is more meaningful).

### 9.7 Size-Invariant Instruments

- **R1 (Probe Decodability)**: 0.667 selectivity across all circuits.
  This measures whether task information is decodable at the layer level,
  not head level, so circuit size doesn't matter.
- **S3 (Distributional Stability)**: Fails across all circuits. With
  only 12 epistemic prompt templates, subset bootstrap has too much
  variance. More prompt diversity would fix this.
- **R3 (Causal Representation)**: Fails across all circuits. Control
  IIA is too high (1.0) — the DAS intervention affects non-circuit
  layers too. This is a known limitation of the current implementation.

---

## 10. EAP vs Activation Patching: Method Comparison

### Circuit overlap analysis

```
              Original (4)   Tight (13)   EAP (15)   Expanded (32)
Original (4)      4              0            1           1
Tight (13)        0             13            2          13 (subset)
EAP (15)          1              2           15           5
Expanded (32)     1             13            5          32
```

Key facts:
- **No head appears in all four circuits.** Each method finds different components.
- **Original ∩ Tight = 0**: hand-picked heads don't survive data-driven discovery.
  L6H5 (the "detector") ranks only 17th by activation patching effect.
- **Tight ⊂ Expanded**: all 13 tight heads appear in the expanded circuit (both
  from act-patching at different thresholds).
- **EAP ∩ Expanded = 5/15**: 10 of 15 EAP heads are completely different from
  anything found by activation patching.

### Key Divergence

EAP and activation patching discover largely non-overlapping circuits.

| Property | Activation Patching (Tight) | EAP |
|----------|---------------------------|-----|
| What it measures | Individual node importance | Edge connectivity |
| Top head | L0H8 (effect=+0.425) | L1H10 (score=163.5) |
| Circuit size | 13 heads | 15 heads |
| F1 score | 0.169 FAIL | 0.152 FAIL |
| S1 score | 1.005 | **2.713** |
| A1 procedure | 0.571 FAIL | **0.714 PASS** |
| A2 composition | -0.058 FAIL | -0.139 FAIL |
| CM1 normative | **25.0x PASS** | 1.6x FAIL |

### Interpretation

The two methods produce a clean dissociation:

- **Act-patch (tight)** finds "specialists" — heads whose individual removal
  changes the output. These produce better normative accounts (CM1=25.0x)
  because each head has a clear individual role.

- **EAP** finds the "backbone" — heads with high edge connectivity that
  form a natural processing pipeline. These produce better procedural
  descriptions (A1=0.714 PASS) because the roles reflect genuine
  information flow. EAP heads also have the strongest distributional
  signatures (S1=2.713) — they're distinctive in their activation
  patterns even though removing any single one doesn't change much.

- **Neither method wins on A2** (composition/faithfulness). Both circuits
  have negative faithfulness, meaning keeping only their heads and
  ablating everything else hurts predictions. The computation is too
  distributed for any 13-15 head subset to be sufficient.

### Edge analysis: where the methods agree

Only 2 heads appear in both circuits: **L2H10** and **L10H0**.

L10H0 is the convergence point — it receives 3 of the top 7 EAP edges
(from L8H8, L8H10, L9H3). It also has the 4th-highest individual
activation patching effect (+0.212). Both methods identify it as the
primary output head for epistemic processing.

L2H10 sends to L7H6 (an EAP mid-level hub) with edge score 2.189.
This is the only top-15 edge connecting a head found by both methods,
suggesting it's a "core" relay between the two circuit styles.

The top 15 EAP edges form a clear pipeline:
```
L1H10, L2H10, L4H7, L4H11  →  L7H6  →  (mid processing)
L8H8, L8H10                →  L9H5, L9H6, L10H0  →  (output)
```

L7H6 is the mid-level hub (receiver in 4/11 top edges) and L8H8 is the
dominant sender (5/15 top edges). Neither appears in the act-patch circuit.
The act-patch circuit's top heads (L0H8, L0H9, L2H1) don't appear in
any top-15 EAP edge — they're individually important but not well-connected.

The practical conclusion: the "right" discovery method depends on the
validity claim. Use act-patch for normative/specification claims, EAP
for algorithmic/procedural claims.

---

## 11. The Fundamental Tradeoff

```
                    F1/CM1 (Specification)
                    ↑
                    |  Original (4)          Tight (13)
                    |  ★ F1 only            ★ CM1 only
                    |
  A1 (Procedure)   |
  ←─────────────────┼──────────────────→  A2 (Sufficiency)
         EAP (15)   |                    Expanded (32)
         ★ A1 only  |                    ★ best A2
                    |
                    ↓
```

No single circuit satisfies all validity criteria simultaneously. Each
circuit occupies a distinct region of validity space:

- **Original (4 heads)**: best specification (F1, F2, F3 pass). Too small
  for sufficiency or procedure.
- **Tight (13 heads)**: best normative account (CM1=25.0x). Moderate
  procedure and sufficiency.
- **EAP (15 heads)**: best procedure (A1=0.714 PASS) and best distributional
  characterization (S1=2.713). Worst composition (A2=-0.139).
- **Expanded (32 heads)**: best sufficiency (A2=0.228, closest to passing).
  Too diluted for specification.

This is a genuine feature of mechanistic explanation, not a bug. The
framework makes it explicit rather than hiding it behind a single score.

---

## 12. Instrument Category Analysis

Grouping the 12 instruments by behavior across circuits reveals three
distinct categories:

### Category 1: Discriminating instruments (different circuits pass)

These instruments reveal genuine differences between circuits:

| Instrument | Who passes | Why |
|------------|-----------|-----|
| F1 Op Spec | Original only | Hand-picked heads are individually characterizable |
| A1 Procedure | EAP only | Edge-based roles form natural pipelines |
| CM1 Normative | Tight only | Strongest-effect heads distinguish epistemic/non-epistemic |

### Category 2: Universal passes (all circuits pass)

These instruments capture properties shared across all circuits:

| Instrument | Score range | Why universal |
|------------|-----------|---------------|
| F3 Replace | 0.988-0.993 | Individual heads are all replaceable by mean |
| S1 Dist Char | 1.005-2.713 | All circuit heads have above-average attributions |
| R1 Probe | 0.667 uniform | Layer-level property, size-independent |

### Category 3: Universal failures (no circuit passes)

These instruments identify systematic limitations:

| Instrument | Failure pattern | Root cause |
|------------|----------------|------------|
| S3 Stability | CV > 0.4 everywhere | 12 prompts too few for bootstrap |
| R3 Causal Rep | Control IIA > 0.3 | Task info is globally available |
| A2 Composition | Negative to 0.228 | 15+ heads needed, no subset sufficient |
| A4 Intermed | Near-zero rho | Attention uses pattern mod, not magnitude |
| CM2 Error Bnd | 0.35-0.40 | Uniform model accuracy, no difficulty gradient |
| F2 Held-Out | 0.474-0.541* | Below baseline predictability |

*F2 is borderline — all scores show moderate correlation, but below
random baseline. The original was initially called PASS at 0.541 in
Pass 1, but by the "beat random baseline" criterion all circuits fail.

### What this categorization reveals

The framework's 12 instruments effectively decompose into 3 that
discriminate between circuits, 3 that confirm universal properties, and
6 that identify structural limitations. The discriminating instruments
are the most valuable for the write-up because they show that different
validity claims require different circuits. The universal failures point
to concrete improvements: more prompts (S3), MLP inclusion (A2), better
control conditions (R3), and potentially retiring A4 for attention circuits.

---

## 13. Lessons for the Evaluation Protocol

This case study suggests a standard multi-pass evaluation protocol:

1. **Pass 1: Evaluate the initial circuit.** Run all instruments. Identify
   which fail and why. Focus on the pattern of failures, not individual scores.

2. **Diagnose: Instrument problem or circuit problem?** If a metric fails on
   both a known-good circuit (IOI) and the circuit under test, the metric needs
   recalibration (F1, A4 in this study). If it fails only on the circuit under
   test, the circuit needs work.

3. **Discovery: Find better circuits.** Use activation patching AND EAP to
   discover data-driven circuits. Create multiple variants at different
   thresholds. Each method finds different components.

4. **Pass 2+: Run all instruments on all variants.** Build the full comparison
   table. The pattern across circuits is more informative than any single score.
   Look for discriminating instruments (different circuits pass).

5. **Interpret the tradeoff.** The specification-vs-sufficiency tension is
   genuine. Report which circuit best supports which validity claim, rather
   than seeking a single "correct" circuit.

---

## 14. Deep Dive: Circuit Overlap and Core Heads

### Corrected overlap matrix

```
           Original(4)  Tight(13)  EAP(15)  Expanded(32)
Original          4          0        1          1
Tight             0         13        2         13
EAP               1          2       15          5
Expanded          1         13        5         32
```

Key structural facts:
- **Original ∩ Tight = 0**: hand-picked and activation-patched circuits share zero heads
- **Tight ⊂ Expanded**: tight is a perfect subset (both from act-patch, different thresholds)
- **EAP ∩ Expanded = 5/15**: edge-based discovery finds mostly different components
- **No head appears in all 4 circuits**

### Core heads (3+ circuits)

Only 2 heads appear in 3 circuits:

| Head | Circuits | F1 consistency | Notes |
|------|----------|---------------|-------|
| L2H10 | Tight, EAP, Expanded | 0.225 | Early processor, found by both methods |
| L10H0 | Tight, EAP, Expanded | 0.261 | Late router, found by both methods |

These are the only heads discovered independently by both activation patching
AND edge attribution patching. They likely represent the most robust components
of the epistemic circuit.

### Layer distribution reveals method bias

```
           early(L0-2)  mid(L3-6)  late(L7-10)
Original       0           1          3          ← late-only (hand-picked)
Tight          8           4          1          ← early-heavy (act-patch)
EAP            3           5          7          ← balanced (edge-based)
Expanded      14          12          6          ← broad coverage
```

Activation patching at a given layer measures that layer's direct effect
on the output. This biases toward early layers (L0) where attention patterns
are position-based and have large activation magnitudes. EAP measures
gradient-weighted information flow between layers, which naturally finds
multi-layer pathways through mid-to-late layers.

### The L10H5 anomaly: why Original passes F1

Per-head F1 consistency scores reveal that Original's F1 pass (0.212 vs
baseline 0.190) is driven almost entirely by L10H5:

```
Head      Consistency    Circuit(s)
L10H5     0.773          Original ONLY  ← anomalous outlier
L0H8      0.622          Tight, Expanded
L0H9      0.556          Tight, Expanded
L5H9      0.501          EAP only
L5H7      0.501          Expanded only
L7H9      0.460          Expanded only
L2H1      0.436          Tight, Expanded
L4H3      0.416          EAP, Expanded
```

L10H5 is unique to the Original circuit (not found by either discovery method)
and has dramatically higher consistency than any other head. This creates
a Simpson's-paradox-like effect: the 4-head circuit passes F1 because
L10H5's 0.773 raises the mean, while larger circuits dilute it with many
heads scoring 0.22-0.28.

**Why L10H5 wasn't rediscovered**: the full activation-patching landscape
reveals that three of the four original heads rank in the bottom half:

```
Head   Act-patch rank   Effect      F1 consistency
L6H5    17/144          +0.192      0.308
L9H2    84/144          +0.041      0.292
L9H5   108/144          +0.018      0.323
L10H5  116/144          +0.015      0.773  ← highest of ANY head
```

L10H5 has the highest F1 consistency of any measured head but ranks 116th
by activation-patching effect. It doesn't change the output when ablated
(+0.015), but when it does fire, it is extremely task-specific (consistency
0.773). This is a conceptually different kind of "circuit membership" —
high specialization but low individual necessity.

The activation-patching distribution across all 144 heads:
- 68 heads (47%) have |effect| < 0.05 (negligible)
- 32 heads (22%) have |effect| > 0.15 (in expanded circuit)
- 8 heads (5.5%) have |effect| > 0.30 (top tier)

This means the expanded circuit at threshold 0.15 already captures
all heads with meaningful individual effects. Further lowering the
threshold would add heads that don't individually contribute.

### EAP's 9 unique heads

```
L1H3, L1H10, L4H11, L5H9, L5H10, L7H6, L8H10, L9H6, L10H7
```

These heads have low individual activation-patching effects (below the
0.15 threshold for expanded) but are high-throughput intermediate nodes
in the edge graph. They serve as information relays — individually
replaceable but collectively necessary for edge-based computation paths.
This explains why EAP uniquely passes A1 (procedure specification):
these relay heads create clean sequential pipelines.

**Edge flow analysis** reveals two hub nodes:

```
Convergence hub: L7H6 (receives from early/mid layers)
  L1H10 → L7H6: 2.320
  L4H7  → L7H6: 2.240
  L2H10 → L7H6: 2.189
  L4H11 → L7H6: 2.174

Broadcasting hub: L8H8 (sends to late layers)
  L8H8 → L10H0: 2.559
  L8H8 → L9H5:  2.537
  L8H8 → L9H6:  2.496
  L8H8 → L10H7: 2.163
```

The implied pipeline: early(L1-2) → mid(L4-5) → L7H6 → L8H8/L8H10 → late(L9-10).
This is a 4-stage sequential pipeline with clear convergence-then-broadcast
topology, which is exactly what A1 (procedure specification) tests for.

L8H8's dual role is the most striking: it is simultaneously the top
EAP edge sender (broadcasting hub) AND a negative-effect suppressor
in the activation patching analysis (effect=-0.168). The head
that *inhibits* the model's output is also the head that *broadcasts*
the most task-relevant information between layers.

### CM1 deep dive: the negation signal

The tight circuit's CM1 pass (25.0x) is driven by the "negation" feature:
high-activation prompts have 25% negation words, low-activation prompts
have 0%. Comparing across all circuits:

```
Feature     | Original | Tight   | EAP     | Expanded
            | hi / lo  | hi / lo | hi / lo | hi / lo
negation    | .15/.10  | .25/.00 | .05/.20 | .10/.15
  ratio     |  1.50    | 25.00   |  0.25   |  0.67
quantifiers | .25/.25  | .35/.15 | .30/.20 | .30/.20
  ratio     |  1.00    |  2.33   |  1.50   |  1.50
```

The negation signal is circuit-specific, not universal:
- Tight: negation strongly UP in high-activation prompts (25x)
- EAP: negation strongly DOWN in high-activation prompts (0.25x)
- Original/Expanded: near-neutral

**Fragility warning**: with 40 prompts (20 high, 20 low), the
low_rate=0.00 means zero negation words in any of the 20 low-activation
prompts. A single negation word in one prompt would drop the ratio from
25x to ~5x. The second-best feature (quantifiers, 2.33x) would barely
pass the threshold. The CM1 pass is real but brittle.

**Interpretation**: the tight circuit's 8 early-layer heads (L0H0,
L0H4, L0H6, L0H7, L0H8, L0H9) are particularly sensitive to negation
tokens, which shifts the median activation and creates the high/low
split. This is consistent with early attention heads attending to
function words (negation words like "not", "never", "don't" are
high-frequency function words that appear at salient positions).

---

## 15. Deep Dive: Understanding Each Universal Failure

### S3 (Bootstrap Stability): insufficient prompt diversity

The epistemic task uses 40 prompts with only 12 unique templates.
Bootstrap resampling draws with replacement, so repeated templates
create high variance in per-head attribution rankings. With 40 prompts
and rank-based metrics, coefficient of variation > 0.4 is expected.

**Fix**: need 200+ prompts with 50+ unique templates for stable rankings.

### R3 (Causal Representation): globally distributed information

R3 measures whether circuit heads encode task-relevant features that
non-circuit heads don't. The IIA (interchange intervention accuracy)
on control heads is 0.30+, meaning non-circuit heads also have
epistemic information. This is expected for a late-emerging property
like epistemic framing — the residual stream accumulates epistemic
information across many heads, not just circuit members.

### A2 (Composition): the suppressor gateway

A2 tests whether sub-pathways through the circuit are individually
faithful. The expanded circuit (32 heads) has 13 pathways; only 3
have positive faithfulness, and ALL THREE route through late_suppressor:

```
POSITIVE pathways (expanded):
  early_suppressor→mid_composer→late_suppressor→late_router:  +0.166
  early_processor→mid_composer→late_suppressor→late_router:   +0.108
  mid_composer→late_suppressor:                               +0.012

NEGATIVE pathways (all others):
  early_suppressor→mid_suppressor:                            -0.179
  early_processor→late_router:                                -0.091
  mid_suppressor→late_router:                                 -0.091
  ... (7 more, all negative)
```

The tight circuit (13 heads, no late_suppressor role) has zero positive
pathways — every single pathway has negative faithfulness (max=-0.066).

**The suppressor is the gateway to positive composition.** Without
late_suppressor heads (L7H8, L8H5, L8H8), all pathways degrade when
isolated because the ablated suppression releases noise that the
remaining pathway can't compensate for. With suppressors in the
pathway, they provide the calibration signal that makes the pathway
self-contained.

### A4 (Intermediate State): wrong variable type

A4 measures whether intermediate activations (mid-circuit) predict
final behavior. For attention circuits, the intermediate "activations"
are attention patterns, which compute via softmax over key-query dot
products. The task-relevant information is encoded in pattern
*structure* (which positions are attended), not pattern *magnitude*.
DAS/IIA, which measures linear alignment, can't capture this.

### CM2 (Error Boundary): flat difficulty curve

CM2 checks whether the circuit's error rate increases with task
difficulty. But the model achieves 92.5% accuracy on all difficulty
levels (easy/medium/hard), creating no difficulty gradient for the
circuit to track. This is a property of the task, not the circuit.

### F2 (Held-Out): moderate but below baseline

All circuits show moderate held-out consistency (0.47-0.54) but below
the random-baseline bar. This means circuit heads are somewhat
predictable in their operation on new inputs, but not more so than
random same-size head sets. The baseline is calibrated by sampling
random head sets, and because many non-circuit heads also show moderate
consistency, the bar is high.

### The MLP elephant in the room

The activation patching scan found that MLP0 has the largest individual
effect of ANY component: +1.313, which is 3.1x the top attention head
(L0H8 at +0.425). The full MLP landscape:

```
MLP0:  +1.313  ← 3x top attention head
MLP3:  -0.455
MLP10: -0.379
MLP4:  -0.347
MLP7:  +0.270
MLP9:  +0.263
MLP8:  +0.214
MLP5:  +0.202
```

**Effect budget analysis** reveals how much of the model's computation
each circuit captures:

```
Component         Positive    Negative    Net
Attention heads    +7.08       -5.66      +1.42 (52% of total)
MLP layers         +2.59       -1.30      +1.30 (48% of total)
Total              +9.67       -6.96      +2.71

Circuit            Net effect   % of attn   % of total
Original (4)       +0.27        19%          10%
Tight (13)         +0.62        44%          23%
Expanded (32)      +1.50       106%          55%
```

Even the expanded circuit captures only 55% of the total computation.
The other 45% is in MLPs (mainly MLP0) and non-circuit attention heads.
The tight circuit captures less than a quarter of the total effect.

All current circuits are attention-only. MLP0 alone contributes more to
the epistemic task than any 3 attention heads combined. This strongly
suggests that the A2 composition failure (no circuit reaches 0.3
faithfulness) is partly because the circuits are missing their largest
component. An attention+MLP circuit including MLP0 might cross the
A2 threshold.

However, no instrument in the current framework tests MLP-level function
specification, MLP necessity, or MLP-attention composition. This is the
single largest gap exposed by this case study.

---

## 16. Lessons for the Evaluation Protocol

This case study suggests a standard multi-pass evaluation protocol:

1. **Pass 1: Evaluate the initial circuit.** Run all instruments. Identify
   which fail and why. Focus on the pattern of failures, not individual scores.

2. **Diagnose: Instrument problem or circuit problem?** If a metric fails on
   both a known-good circuit (IOI) and the circuit under test, the metric needs
   recalibration (F1, A4 in this study). If it fails only on the circuit under
   test, the circuit needs work.

3. **Discovery: Find better circuits.** Use activation patching AND EAP to
   discover data-driven circuits. Create multiple variants at different
   thresholds. Each method finds different components.

4. **Pass 2+: Run all instruments on all variants.** Build the full comparison
   table. The pattern across circuits is more informative than any single score.
   Look for discriminating instruments (different circuits pass).

5. **Interpret the tradeoff.** The specification-vs-sufficiency tension is
   genuine. Report which circuit best supports which validity claim, rather
   than seeking a single "correct" circuit.

---

## 17. Three Axes of Circuit Membership

The findings from sections 14-16 converge on a key theoretical point:
"circuit membership" is not a single concept. At least three independent
properties determine whether a head belongs to a circuit, and each
property is tested by different instruments:

```
Axis          | Measures            | Discovery method  | Best instrument
Necessity     | ablation degrades   | Act-patching      | A2 Composition
              | performance         |                   |
Specialization| output correlates   | Hand-picking /    | F1 Op Spec
              | with task features  | logit lens        |
Connectivity  | relays information  | EAP (edge-based)  | A1 Procedure
              | between layers      |                   |
```

These are genuinely orthogonal — knowing a head's rank on one axis
tells you nothing about its rank on the others:

```
Head    Necessity(rank/144)  Specialization(F1)  Connectivity(EAP rank)
L10H5    116/144 (low)         0.773 (highest)     not in top 30
L0H8       1/144 (highest)     0.622 (moderate)    not in top 30
L8H8      26/144 (moderate)    0.345 (moderate)    5/30 (high)
L7H6     122/144 (low)         0.290 (low)        12/30 (high)
```

This explains why no single circuit passes all validity criteria:
each circuit is optimized for one axis and performs poorly on the others.
It also explains why the framework needs multiple instrument categories
(Function, Algorithmic, Computational-level) — each category tests a
different axis.

**Practical implication**: circuit discovery should use at least two
methods (activation patching + EAP) and report results on both
specification and sufficiency criteria. Claiming a circuit is "the"
mechanism for a task without specifying which axis of membership is
being tested makes an incomplete claim.

---

### Proposed: composite circuit

If the three axes are genuinely complementary, a circuit built from the
top heads on each axis should outperform any single-axis circuit. Taking
the top 8 by necessity, top 4 by specialization, and top 8 by
connectivity produces a 17-head composite:

```
Axis         Heads
Necessity    L0H0, L0H4, L0H6, L0H8*, L0H9*, L3H8, L4H6, L5H3
Special.     L0H8*, L0H9*, L5H9*, L10H5
Connect.     L1H10, L2H10, L4H7, L5H9*, L8H8, L8H10, L9H5, L10H0
```

Three heads span 2 axes (*): L0H8 (N+S), L0H9 (N+S), L5H9 (S+C).
No head spans all three — the axes are genuinely independent.

**Prediction**: this 17-head circuit should pass F1 (has L10H5), come
close to CM1 (has the tight-circuit early heads), and potentially pass
A1 (has the EAP pipeline: L1H10→L2H10→L4H7→L8H8→L9H5→L10H0). It would
be the first circuit tested that draws from all three discovery methods.

Not yet implemented — would require defining a new circuit variant
and running all 12 instruments.

### Axis decomposition of the expanded circuit

Cross-referencing F1 consistency, activation-patching effect, and EAP
edge involvement for all 32 expanded-circuit heads reveals the axis
distribution:

```
Head     Cons.  Effect   EAP      Axes
L0H8     0.622  +0.425     0.0    S+N  (specialization + necessity)
L0H9     0.556  -0.303     0.0    S+N
L2H1     0.436  -0.207     0.0    S+N
L5H2     0.404  -0.249     0.0    S+N
L4H3     0.416  -0.184   124.3    S+C  (specialization + connectivity)
L8H8     0.345  -0.168   145.6    C    (connectivity only)
L0H4     0.281  +0.355     0.0    N    (necessity only)
L5H3     0.365  +0.333     0.0    N
L5H7     0.501  +0.167     0.0    S    (specialization only)
L7H9     0.460  +0.197     0.0    S
```

Quadrant counts (thresholds: consistency>0.4, |effect|>0.20, EAP>120):
- Specialization only: 3 heads
- Necessity only: 7 heads
- Connectivity only: 5 heads
- Multi-axis: 7 heads (mostly S+N)
- Below all thresholds: 10 heads

The 10 "below all" heads are the weakest members of the expanded circuit
— they were included at the 0.15 threshold but don't clearly contribute
to specialization, necessity, or connectivity. These dilution heads
are what drag the expanded circuit's F1 and CM1 scores below baseline.

---

## 18. Edge-Level Instruments (G1-G5)

### G3: Path Specificity — COMPLETE

G3 measures whether edge effects are task-specific (low correlation
between task and control effects = specific). Spearman rho close to 0
means task-specific; close to 1 means generic.

```
.venv/bin/python3 src/instruments/structural/edge_analysis/84_path_specificity.py \
    --tasks epistemic_framing epistemic_tight epistemic_eap epistemic_expanded \
    --n-prompts 40 --device cpu
```

| Circuit | Edges | Spearman rho | Pass? |
|---------|-------|-------------|-------|
| Original (4) | 5 | 1.000 | FAIL |
| Tight (13) | 32 | 0.557 | FAIL |
| EAP (15) | 75 | 0.918 | FAIL |
| Expanded (32) | 232 | 0.811 | FAIL |

All circuits fail — edge effects are correlated between task and
control prompts, meaning the edges carry general processing, not
task-specific signal. IOI also fails G3 (rho=0.977).

Tight circuit is closest to passing (rho=0.557), suggesting its
edges are the most task-specific of the four circuits.

### G2: Edge Necessity — COMPLETE

G2 measures what fraction of edges are individually necessary (removing
one edge drops performance by >5%).

```
.venv/bin/python3 src/instruments/structural/edge_analysis/83_edge_necessity.py \
    --tasks epistemic_framing epistemic_tight epistemic_eap epistemic_expanded \
    --n-prompts 40 --device cpu
```

| Circuit | Edges | Necessary | Fraction | Pass? |
|---------|-------|-----------|----------|-------|
| Original (4) | 5 | 0 | 0.00 | FAIL |
| Tight (13) | 32 | 0 | 0.00 | FAIL |
| EAP (15) | 75 | ~2 | 0.03 | FAIL |
| Expanded (32) | 232 | 1 | 0.004 | FAIL |

All circuits fail — almost no individual edge is necessary. The one
necessary edge in the expanded circuit is **L8H8→L10H0** (drop=0.050),
confirming L8H8's role as the key broadcasting hub. This is the same
edge that connects the suppressor hub to the late router.

IOI also fails G2 (27.3% necessary), so edge necessity is a hard
criterion. But IOI has 12/44 necessary edges vs epistemic's 1/232 —
a 60x difference in edge necessity density. Epistemic circuits have
highly redundant edge structures.

### G1: Path Identification — still running (~290 min CPU, heaviest instrument)

---

## 19. Next Steps

- [x] Complete all 12 node-level instruments on all 4 circuits
- [x] G3 path specificity (all circuits FAIL)
- [x] G2 edge necessity (all FAIL, only L8H8→L10H0 necessary)
- [ ] G1 path identification (running, ~290min CPU)
- [ ] G4 compositional sufficiency + G5 graph minimality (after G1-G2)
- [ ] Test composite 17-head circuit (top necessity + specialization + connectivity)
- [ ] Test attention+MLP circuit variant (include MLP0, effect=+1.31)
- [ ] Add more epistemic prompt diversity (fix S3 failure)
- [ ] Write up the three-category finding for the paper
