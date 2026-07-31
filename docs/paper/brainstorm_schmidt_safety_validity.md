# Brainstorm: Schmidt Sciences AI Interpretability RFP x Mechval Framework

Working notes, May 2026. Two RFPs on the table:
- **AI Interpretability RFP** ($300K--$1M, deadline May 26 2026): "detect and mitigate deceptive behaviors in LLMs", "move beyond academic benchmarks to address real-world risks"
- **Science of Trustworthy AI** ($1M--$5M+): broader, less constrained

Key framing decision: do NOT pitch this as "deception detection." Pitch it
as: deception is a *special case* of the general validity problem, and the
framework already provides the measurement infrastructure. Deception
detection without validity standards is security theater.

---

## 1. Deception as a Special Case of MI Validity

The core argument: every claim of the form "this feature/direction/circuit
detects deception" is a mechanistic interpretability claim, and every MI
claim is subject to the same validity requirements. Deception detection is
not a separate research agenda --- it is an *application* of MI that
inherits all of MI's validation problems.

### 1a. Finding deception features = construct validity (C1, C3, C5)

- C1 Falsifiability: What result would convince you the "deception feature"
  is *not* a deception feature? If you can't state this in advance, the
  feature is unfalsifiable and its label is decorative.
- C3 Task Specificity: Does the feature fire *only* on deception, or does
  it fire on uncertainty, hedging, politeness, compliance, any other
  correlated construct? Without C3, a "deception feature" might just be a
  "model is uncertain" feature.
- C5 Convergent Validity: Do multiple independent methods (probing, SAE
  feature identification, DAS, activation patching) converge on the same
  direction/feature/circuit? A "deception feature" found by one method and
  not confirmed by another has weak construct validity.
- The construct boundary problem: "deception" is not a natural kind in
  transformer computation. It is a human-imposed label on a cluster of
  behaviors (strategic omission, false assertion, misleading implicature,
  sandbagging, sycophancy, ...). Each sub-behavior may have a different
  mechanistic substrate. Treating them as one construct is a C3 validity
  failure unless convergent evidence supports it.

### 1b. Verifying deception features = causal sufficiency (I2, E2, E4)

- I2 Sufficiency: Does activating the "deception feature" alone produce
  deceptive outputs? If you can only show correlation (the feature fires
  during deception), you have a probe, not a mechanism.
- E2 Graded Response: Does increasing the feature's activation produce
  *more* deceptive behavior in a dose-dependent way? Monotonicity is the
  minimum bar for a causal claim.
- E4 Effect Magnitude: How large is the effect? A feature that explains
  2% of the variance in deceptive behavior is not a safety tool.
- The sufficiency gap is the biggest risk: a feature could fire reliably
  on deception (high M1) but not causally produce it (low I2). Correlation
  is the floor, not the ceiling.

### 1c. Reliability of detection = measurement reliability (M1, M2, F01, F06)

- M1 Reliability: Does the detector give consistent results across prompt
  splits, random seeds, and model checkpoints?
- M2 Measurement Invariance: Is the detection threshold stable across
  different prompt distributions, or does it need recalibration per domain?
- F01 Bootstrap Stability: What is the confidence interval on the detection
  score? A detector with CI [0.3, 0.9] is useless for deployment.
- F06 Inter-Rater Agreement: Do independent runs of the detection pipeline
  agree? The DMSAE finding (only 197/65k SAE features stable across
  training runs) is a direct cautionary tale: if 99.7% of SAE features are
  unstable, what fraction of "deception features" are training artifacts?

### 1d. "Absence of evidence != evidence of absence" = construct coverage (M6)

- M6 Construct Coverage: The metric measures its nominal target, not a
  correlated proxy. For safety: "the detector says no deception" could mean
  (a) no deception exists, (b) deception exists but is in a direction the
  detector doesn't cover, or (c) deception exists in a covered direction
  but below the threshold.
- This is the fundamental asymmetry of safety monitoring: a detector's
  silence tells you nothing unless you know its coverage. M6 forces this
  question.
- Connection to superposition: if deception is represented in superposition
  with other features, any single-direction detector has gaps by
  construction. G3 (Confound/Superposition Risk) is the precondition check.

### 1e. The "isolated demand for rigor" problem

- Current safety evaluations compare behavioral red-teaming (no validity
  standards, low reproducibility, no stated falsification criteria) against
  MI-based detection (held to the highest causal standards). This
  asymmetry makes MI look worse by construction.
- The framework levels the playing field: apply the same validity criteria
  to behavioral and MI approaches. A behavioral red-team that has no
  falsification criteria (C1), no stated coverage (M6), and no
  cross-prompt reliability (M1) is *also* invalid --- it just doesn't know it.
- Schmidt's RFP asks to "move beyond academic benchmarks." The framework
  operationalizes what "beyond" means: from behavioral adequacy to
  mechanistic validity, with explicit standards for both.

### 1f. Neel Nanda's key insight and the framework's answer

- Nanda (2024): MI provides *mechanistic evidence*, not *certainty
  guarantees*. You should not expect MI to prove a model is safe; you
  should expect it to provide a specific kind of evidence with quantifiable
  strength.
- The framework's contribution: it tells you *how much* evidence is enough.
  The verdict tiers (Proposed -> Causally Suggestive -> Mechanistically
  Supported -> Triangulated -> Validated) are graded levels of evidential
  support. A deception feature at "Proposed" should not be deployed; one at
  "Triangulated" has multiple converging lines of evidence.
- The gap: nobody has stated what verdict tier is *required* for
  safety-critical deployment. This is a policy question, but the framework
  makes it a question with a formal answer space.

---

## 2. What the Framework Already Provides for Safety Applications

### Existing metrics mapped to safety-relevant questions

| Safety question | Framework metric(s) | What it tells you |
|---|---|---|
| "Is this feature robust to adversarial prompts?" | VPD (metric 105, C18 Adversarial Parameter Decomposition) | Worst-case robustness: does the feature survive adversarially chosen ablation directions? Standard ablation is necessary-condition testing; VPD is sufficient-condition testing. |
| "Does the safety feature fire consistently?" | F01 Bootstrap Stability, F06 Inter-Rater, d08 Prompt Paraphrase | Cross-prompt and cross-run consistency. If the feature fires on "how to make a bomb" but not "describe the process of constructing an explosive device," it has low M1. |
| "Is this feature a training artifact?" | DMSAE core stability finding (197/65k), F02 Seed Variance | Only features stable across training runs are candidates for safety monitoring. 99.7% of SAE features fail this bar. |
| "Is this feature causally sufficient?" | a02 DAS-IIA, a04 Woodward Interventionism (sigma ablation), role_ablation | IIA: does swapping the feature's value across inputs swap the output? Ablation: does removing the feature remove the behavior? |
| "Does intervening on this feature actually change behavior?" | a06 Mediation, a03 CATE, E1 Intervention Reach | Mediation analysis decomposes total effects into direct and indirect paths. CATE estimates treatment effects. E1 verifies the intervention actually reached the target. |
| "Does the feature generalize across scales?" | d07 Cross-Scale Transfer, E6 Cross-Architecture Generalization | A safety feature found in GPT-2 that doesn't exist in GPT-4 is not a general safety mechanism --- it's a quirk of one model. |
| "Is the feature specific to deception, not some correlated construct?" | I3 Specificity, C3 Task Specificity, f04 Discriminant Validity | The discriminant validity test: the "deception feature" should NOT fire on non-deceptive inputs that share surface features with deceptive ones. |

### Safety subspace findings (survey Part IX, D3)

Four independent papers converge: safety-relevant information occupies a
low-rank, stable subspace that passes E2, M1, and C4 by construction.
Metric EX28 (Safety Subspace Causal Validation) operationalizes this.
Metric M14 (Safety Singular Value Entropy) quantifies compactness. Metric
EX29 (Single-Shot Safety Recovery) tests whether safety is recoverable
from a single example --- the most parsimonious form of E2.

This is the strongest positive result for the framework's applicability to
safety: MI-based safety detection works precisely when the constructs have
high validity. The framework does not just identify problems --- it
identifies solutions.

### Views and gates for safety

- V1 (Causal Effect Estimation): quantifies effect sizes, mediation paths,
  dose-response for safety features
- V2 (Causal Transportability): does the safety feature transfer across
  models? This is the scalability question.
- V3 (Counterfactual Verification): rung-3 tests --- what would have
  happened if the safety feature had a different value?
- V4 (Mechanism Adjudication): when two rival explanations of the safety
  mechanism exist, which one wins?
- G2 (Causal Identifiability): can the safety-relevant effects be estimated
  with the available interventions?
- G3 (Confound/Superposition Risk): is the safety feature in superposition
  with other features? If yes, interventions may have side effects.

### Track 3 (Causal Model Testing) for safety claims

A safety claim like "heads 9.1, 9.6, and 10.7 form a sycophancy circuit"
can be expressed as a MechanisticClaimSpec:
- Steps: each head's role in the circuit
- Edges: information flow between heads
- Predictions: ablating head X should increase sycophancy by Y
- Negative controls: ablating head X should NOT affect factual accuracy
- Rival specs: alternative sycophancy mechanisms

`mv.verify(spec)` runs all predictions and returns a verdict tier. No
one has done this for any safety-relevant circuit.

---

## 3. Gap Analysis: What's Missing for Safety Applications

### 3a. Goal representation validity

No existing metric tests whether a feature represents a *goal* vs.
correlating with goal-related tokens/contexts.

- A "deception goal" feature that fires when the model outputs deceptive
  text might just be a "deception-related token prediction" feature. These
  are different constructs: one is about internal planning, the other is
  about next-token prediction of deception-related words.
- The distinction matters for safety: a model could have a "predict
  deception-related tokens" feature without having any "deception goal"
  representation. Conflating them is a C3 validity failure.
- What's needed: a metric that discriminates goal representations from
  token-prediction correlates. Possible approach: test whether the feature
  fires *before* deceptive tokens appear (prediction) or *only during*
  deceptive planning (goal). Temporal specificity (I14 Settling Depth) is
  relevant but not sufficient.

### 3b. Adversarial robustness of MI claims

- VPD (metric 105) is the closest existing tool, but it operates on
  weight-space parameter decompositions, not on activation-space features.
- What's needed: adversarial robustness testing for activation-space
  claims. Can an adversary craft inputs that cause a valid feature to fire
  incorrectly (false positive) or fail to fire (false negative)?
- Current metrics test robustness under benign prompt variation (d08 Prompt
  Paraphrase, E5 Robustness). Adversarial robustness is strictly harder:
  the adversary is optimizing against the detector.
- Connection to ARC Evals and control evaluations: adversarial robustness
  of MI features is the MI analogue of red-teaming for behavioral evals.

### 3c. Calibrated uncertainty

- The framework assigns binary pass/fail verdicts at the criterion level
  and ordinal tiers at the verdict level. Neither provides calibrated
  probabilities.
- Safety applications need: "with probability p, this feature detects
  deception." The current framework can say "the feature passes M1
  reliability" but not "the feature has 0.92 precision and 0.78 recall for
  deception detection."
- What's needed: a calibration layer that converts criterion pass/fail into
  probabilistic detection guarantees. Bayesian extension of the verdict
  system.
- This connects to the "how much evidence is enough" question from section
  1f. The verdict tiers are qualitative answers; calibrated probabilities
  are quantitative ones.

### 3d. Compositional safety claims

- "The model is safe" requires composing many feature-level claims
  (no deception feature fires + no sandbagging feature fires + no
  goal-misgeneralization feature fires + ...).
- The composition rules are not formalized: does an AND over independent
  feature claims give a valid composite claim? What about correlated
  features? What about features that only become dangerous in combination?
- What's needed: a composition algebra for validity claims. Analogous to
  how reliability of composite tests is computed from item reliabilities
  (Cronbach's alpha, but for validity claims).
- The epistemic study's rival specs (4 rival claim specs with V4
  adjudication) are a partial model for this: composing and comparing
  multiple claims about the same behavior.

### 3e. ARC's formal verification connection

- ARC Evals and related groups (e.g., Redwood Research control evaluations)
  are interested in worst-case bounds: "under what conditions can we
  guarantee the model is safe?"
- Surprise accounting (Christiano 2022) proposes that safety arguments
  should decompose into a sequence of claims, each of which "spends"
  surprise budget. If the total surprise budget is exceeded, the safety
  argument fails.
- Can the validity framework provide formal components for surprise
  accounting? Each criterion pass could be a bounded surprise claim:
  "passing I2 Sufficiency with effect size 0.8 spends X bits of surprise
  budget." The verdict tiers would then correspond to total surprise
  budgets.
- This is speculative but could bridge the gap between MI validity and
  formal safety arguments. Worth discussing with ARC folks.

### 3f. The deception ontology problem

- There is no agreed-upon taxonomy of deceptive behaviors in LLMs.
  "Deception" is used to refer to: strategic lying, sycophancy,
  sandbagging, goal misgeneralization, faithfulness failures, hallucination
  with false confidence, and more.
- Each of these may have different mechanistic substrates. Applying the
  framework to "deception" requires first applying C1 Falsifiability to
  the construct itself: what is the falsifiable definition of "deception"
  that distinguishes it from each neighboring construct?
- The framework provides the tools for this (C1 + C3 + f04 Discriminant
  Validity) but somebody has to do the work.

---

## 4. The Schmidt-Aligned Research Agenda

Three project ideas mapped to the three RFP focus areas. Each one is
independently publishable and contributes to the framework's value
proposition.

### Project 1: Detection --- Validity evaluation of deception-adjacent features

**What**: Apply the full validity evaluation protocol to the best-known
safety-relevant features:
- Sycophancy directions (Perez et al., Sharma et al.)
- Sandbagging circuits (if any have been proposed)
- Goal misgeneralization features (Hubinger et al.'s conceptual work,
  operationalized)
- Refusal directions (Arditi et al. 2024)
- The Assistant Axis (MATS + Anthropic Fellows, 2026)

**Protocol per feature**:
1. G0 Construct Operationalization: is the construct defined?
2. G1 Measurement Calibration: are the metrics stable? (F01, F02, F06)
3. G3 Superposition Risk: is the feature in superposition?
4. I1 Necessity + I2 Sufficiency: causal tests via ablation + restoration
5. M1 Reliability: cross-prompt, cross-seed, cross-checkpoint stability
6. I3 Specificity + C3 Task Specificity: discriminant validity tests
7. E5 Robustness: prompt paraphrase generalization
8. E6 Cross-Architecture: does the feature exist in Gemma, Llama, etc.?

**Output**: A validity report card for each feature, with explicit
gaps named and verdict tiers assigned. First systematic validity
assessment of any safety feature.

**Schmidt alignment**: "detect deceptive behaviors" --- but with validity
standards, not just detection claims.

**Budget fit**: $300K--$500K. Primarily compute + 1--2 postdocs.

### Project 2: Steering --- Validity predicts steering effectiveness

**Hypothesis**: Features that pass more validity criteria (especially
I2 Sufficiency and E2 Graded Response) are more effective steering
targets than features that don't.

**What**: For each safety feature from Project 1:
1. Measure validity score (number of criteria passed, verdict tier)
2. Use the feature as a steering vector (activation addition/subtraction)
3. Measure steering effectiveness (behavioral change in target direction)
4. Measure steering specificity (unintended side effects)
5. Test: does validity score predict steering effectiveness?

**The key prediction**: a feature with high I2 (causally sufficient) and
high E2 (graded response) should be a better steering target than one
with only high M1 (reliable but not causal). If this prediction holds,
the framework provides a *selection criterion* for steering targets:
validate first, steer second.

**Counter-prediction to test**: maybe any direction that's reliably
detectable (high M1) is a good steering target regardless of causal
status. If this holds, the framework's causal criteria are over-scoped
for steering applications.

**Schmidt alignment**: "mitigate deceptive behaviors" --- steering is
the primary mitigation tool, and validity predicts which steering
interventions work.

**Budget fit**: $200K--$400K. Extension of Project 1 data.

### Project 3: Real-world deployment --- Red-team/blue-team validation gap

**What**: Measure the adversarial robustness gap between validated and
unvalidated features in a controlled red-team/blue-team exercise.

**Protocol**:
1. Blue team deploys safety monitors based on validated features
   (passing I2, M1, E5, at minimum "Mechanistically Supported" verdict)
2. Red team deploys safety monitors based on unvalidated features
   (passing only M1 or not even that)
3. Adversary generates prompts that attempt to evade detection
4. Measure: detection rate, false positive rate, adversarial evasion
   rate for each monitor class
5. Test: do validated monitors have higher adversarial robustness?

**The key prediction**: validated monitors should be harder to evade
because they target causally relevant features, not surface correlates.
An adversary can change surface features (prompting style, token
distribution) but cannot easily change the model's internal causal
structure.

**Schmidt alignment**: "move beyond academic benchmarks to address
real-world risks" --- this is a deployment-relevant test, not an
academic benchmark.

**Budget fit**: $400K--$800K. Requires adversarial ML expertise +
compute for large-scale adversarial optimization.

---

## 5. Steinhardt + Actionability Connection

### 5a. Steinhardt's evaluation hierarchy

Jacob Steinhardt's position (broadly: behavior evaluation > capability
evaluation for safety) maps directly onto the framework's E-frame:

- "Behavior evaluation" = E-family criteria (E1 Intervention Reach, E2
  Graded Response, E5 Robustness) applied to behavioral outputs
- "Capability evaluation" = I-family criteria (I1 Necessity, I2
  Sufficiency) applied to internal representations
- Steinhardt's concern: capability evaluations can be gamed because they
  test what the model *can do*, not what it *will do*
- Framework translation: capability claims without behavioral validation
  have high I-frame scores but low E-frame scores. The framework requires
  *both* for high verdict tiers, which operationalizes Steinhardt's
  preference

### 5b. ICML 2026 "Interpretability Can Be Actionable" connection

The ICML 2026 position paper's two dimensions:
1. **Concreteness**: how specific is the MI claim?
2. **Validation**: how well-tested is the MI claim?

These are a subset of the full framework:
- "Concreteness" maps to the description mode hierarchy (Computational >
  Algorithmic > Representational > Implementational). Higher modes require
  more specific claims.
- "Validation" maps to the verdict tier system (Proposed -> Validated).
  Higher tiers require more evidence.

What the framework adds beyond the ICML paper:
- The ICML paper's 2 dimensions become 5 validity types x 7 description
  modes x 5 verdict tiers --- a much richer evaluation space
- The framework provides *specific criteria* (27+) and *specific metrics*
  (84) for each evaluation point. The ICML paper calls for validation but
  doesn't say *how*. The framework does.
- The framework has 13 worked case studies showing the evaluation in
  practice. The ICML paper has conceptual arguments.

### 5c. Safety as the ultimate actionability test

- "Is MI actionable?" is equivalent to "can MI produce validated claims
  that support reliable interventions?"
- Safety is the hardest case: the cost of being wrong is highest, the
  adversary is most capable, and the construct (deception/alignment/safety)
  is least well-defined.
- If the framework can produce validated safety claims that survive
  adversarial testing, MI is actionable in the strongest possible sense.
- If it can't, we know *exactly why* (which criteria fail, at what
  verdict tier does the evidence stop) --- and that diagnostic is itself
  actionable.

### 5d. The "validated enough" operationalization

Both Steinhardt and the ICML paper are asking, implicitly: "when is an MI
claim good enough to act on?" The framework provides a formal answer:

- **Never act on Proposed claims**: no causal evidence, label might be
  wrong
- **Causally Suggestive is a yellow light**: one line of causal evidence,
  could be confounded
- **Mechanistically Supported is the minimum for monitoring**: multiple
  criteria passed, but gaps remain
- **Triangulated is the minimum for intervention/steering**: convergent
  evidence from multiple families
- **Validated is the gold standard**: all criteria passed, cross-model
  transfer confirmed

For safety specifically: the question is whether "Mechanistically
Supported" is good enough for deployment monitoring, or whether
"Triangulated" is required. This is a policy question that the framework
makes *answerable* by giving it a formal vocabulary.

---

## 6. Budget and Scope Notes

### For the $300K--$1M RFP (AI Interpretability):
- Lead with Project 1 (detection validity): most direct match to
  "detect and mitigate deceptive behaviors"
- Include Project 2 (steering) as the mitigation component
- Frame the framework as the *missing validation layer* for the field's
  safety claims --- not a new detection method, but the method for
  evaluating detection methods

### For the $1M--$5M+ RFP (Science of Trustworthy AI):
- All three projects as a unified program
- Add: cross-model validity assessment at scale (Gemma, Llama, Claude
  family if access available)
- Add: formal connection to surprise accounting / ARC-style verification
- Add: development of the calibrated uncertainty layer (section 3c)
- Frame as: establishing the measurement-theoretic foundation for AI
  safety claims, analogous to what clinical trial methodology did for
  medical claims

### What makes this competitive:
1. Not another detection method --- it's the validation layer for
   *all* detection methods
2. Already built: 84 metrics, 14 calibrations, 54 tasks, 12 claim
   specs, 13 worked case studies. This is not a proposal to build a
   framework; it's a proposal to *apply* one.
3. The framework is method-agnostic: it evaluates SAE features, circuit
   claims, probing results, and steering vectors using the same criteria.
   No competitor does this.
4. The DMSAE/SAEBench/MechEvalAgent findings (99.7% feature instability,
   93% reproducibility failure, TPP/SCR metric unreliability) are the
   motivating crisis that makes the framework timely.

### What could sink it:
1. "Too theoretical" --- the framework is measurement theory, not a
   tool you can deploy tomorrow. Counterargument: the 84 metrics are
   implementable, and Project 1 produces concrete validity report cards.
2. "Deception features don't exist yet" --- if there are no good
   deception features to evaluate, Project 1 produces null results.
   Counterargument: the null result itself is informative (the field has
   no valid safety features), and sycophancy/refusal directions do exist.
3. "Too academic for Schmidt" --- Schmidt wants deployment impact.
   Project 3 (red-team/blue-team) addresses this directly.

---

## 7. Open Questions

- Who are natural collaborators? ARC Evals (formal verification angle),
  Anthropic (NLA + safety subspace work), Goodfire (VPD + Silico platform),
  MATS (Assistant Axis)
- Which safety features are mature enough for a full validity evaluation?
  Refusal directions are probably the best candidate (Arditi et al. 2024,
  most causal evidence). Sycophancy directions are second. Pure
  "deception" features may not exist yet in the literature.
- Should the proposal include developing new safety features, or only
  evaluating existing ones? Evaluating existing ones is cleaner and more
  novel. Developing new ones is what everyone else will propose.
- What is the right model to use? GPT-2 (most tooling) is too small for
  realistic safety evaluations. Gemma 2 27B (Gemma Scope 2 artifacts) is
  the sweet spot: large enough to exhibit safety-relevant behaviors,
  comprehensively instrumented, open weights.
- How does this relate to the TMLR paper? The TMLR paper establishes the
  framework; the Schmidt project applies it to safety. The TMLR paper
  should ideally be accepted (or at least submitted) before the Schmidt
  proposal deadline. If not, the proposal can cite the preprint.
