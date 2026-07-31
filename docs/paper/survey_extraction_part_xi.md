# MI Deep Survey Part XI: Extraction

Extracted from "MI Methods Deep Survey Part XI --- The Verification
Turn: VPD Adversarial Ablation, Actionable Interpretability, Nanda
Deception Limits, ARC Heuristic Arguments, Steinhardt Behavior
Evaluation, and the Schmidt Sciences RFP" (May 2026).

Covers 6 major items from the verification and evaluation-criteria
layer (May 2025 -- May 2026): the first adversarial ablation
verification criterion for parameter-level decompositions (VPD), the
first ICML paper arguing that evaluation criteria --- not methods ---
are the missing ingredient (Orgad and Barez), the clearest statement
of interpretability's epistemic limits for deception detection (Nanda),
the formal-methods complement to empirical validity (ARC heuristic
arguments), the behavior-evaluation investment thesis (Steinhardt),
and two funding signals that frame the applied arm of the validity
framework (Schmidt Sciences RFPs). The unifying theme: the field has
converged on the thesis that verification criteria, not new methods,
are the bottleneck --- and the framework is the most complete
operationalization of that thesis.

---

## A. New Methods to Implement

### Tier 1: Critical (directly maps to framework criteria or fills gap)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| VPD Adversarial Ablation | Adversarial worst-case subset removal as gold-standard E2 verification: a subcomponent is valid iff performance survives adversarial ablation (not just random ablation) | Goodfire; goodfire.ai/research/interpreting-lm-parameters; AlignmentForum May 5 2026, 161 karma | mechanistic_interpretability | evaluation |
| Actionability Validity Dimensions | Concreteness (enables specific intervention?) + Validation (empirically tested?) as two-dimensional evaluation of interpretability claims; five application domains | Orgad, Barez et al.; arXiv:2605.11161; ICML 2026 | measurement_theory | evaluation |
| Deception Feature Causal Sufficiency | Tests whether a claimed "deception feature" is causally sufficient for strategic deception vs. mere correlation with deceptive-looking outputs; requires goal-representation E2 | Nanda; LessWrong May 2025 (curated); related: "Difficulties with Evaluating a Deception Detector" Dec 2025 | mechanistic_interpretability | evaluation |

### Tier 2: High (new methods with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| Heuristic Explanation Certification | Mechanistic account that is human-legible, has worst-case certified accuracy, and predicts extreme behaviors; formal complement to empirical validity | ARC (Alignment Research Center); alignmentforum.org; Christiano et al. heuristic arguments series | measurement_theory | evaluation |
| Surprise Accounting | Per-component uncertainty reduction: how much does each mechanistic part reduce uncertainty about model outputs? Base case solved for random MLPs | ARC; alignmentforum.org; surprise accounting posts | mechanistic_interpretability | evaluation |
| Behavior Evaluation Protocol | Behavior evaluation > capability evaluation for safety; systematic under-investment in downstream validity | Steinhardt; thegradient.pub or personal blog; "The Case for Evaluating Model Behaviors" May 20 2026 | measurement_theory | evaluation |

### Tier 3: Strategic (funding and field-level signals)

| Item | What | Source | Implication |
|---|---|---|---|
| Schmidt Sciences RFP (deceptive behaviors) | $300K--$1M grants, due May 26 2026; detect/mitigate deceptive behaviors; red team vs blue team structure; "move beyond academic benchmarks" | Schmidt Sciences; schmidtsciences.org | Near-literal description of the framework's applied arm; E2 Causal Sufficiency for deception detection is the core deliverable |
| Science of Trustworthy AI RFP | $1M--$5M+ grants; foundational research on trustworthy AI, including interpretability validation | Schmidt Sciences | Larger-scale funding signal; framework as measurement-theoretic foundation for trustworthiness claims |
| Forum-level phase transition | Phase 1 (2022--23) optimism, Phase 2 (2024) proliferation without standards, Phase 3 (2025) reckoning, Phase 4 (2026) infrastructure push | LessWrong, AlignmentForum, arXiv trajectory analysis | The framework enters at the peak of Phase 4; maximum receptivity window |

---

## B. New Metrics (suggested implementations)

### B1. VPD Adversarial Ablation Score

- **metric_id**: `EX35_vpd_adversarial_ablation`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I1 Necessity, I2 Sufficiency)
- **criteria**: E2 Causal Sufficiency (gold-standard); I1 Component Necessity
- **what it measures**: For a given subcomponent (circuit, feature, parameter subset), performs adversarial ablation: selects the worst-case subset of the subcomponent to remove (the subset whose removal maximally degrades the claimed behavior). Reports (1) adversarial_sufficiency: performance retained after removing the adversarially-selected subset (lower = the subcomponent's validity is fragile); (2) adversarial_gap: difference between random-ablation performance and adversarial-ablation performance (large gap = non-adversarial methods overestimate validity); (3) survival_threshold: minimum fraction of the subcomponent that must be retained for >50% task performance.
- **pass condition**: adversarial_sufficiency > 0.5 (retains >50% of task performance after adversarial removal of up to 30% of subcomponent); adversarial_gap < 0.2 (random and adversarial ablation results should not diverge by more than 0.2)
- **implementation notes**: The adversarial search uses greedy backwards elimination: iteratively remove the single element whose removal causes the largest performance drop, and report the cumulative curve. This is computationally O(n^2) in subcomponent size but provides the strongest E2 evidence. For large subcomponents, approximate via random subsets + local search. The VPD paper applies this to parameter-level decompositions (attention head weight matrices decomposed into interpretable rank-1 terms); the metric generalizes to any subcomponent type.
- **why critical**: This is the most stringent E2 test in the literature. Non-adversarial ablation (random subset removal, mean ablation, zero ablation) systematically overestimates causal sufficiency because it does not test worst-case robustness. Every circuit paper that reports only random ablation has an untested adversarial gap. VPD's key contribution is demonstrating that this gap exists and matters --- their parameter-level decompositions survive adversarial ablation where attribution-based methods do not.

### B2. Actionability Score (Orgad-Barez)

- **metric_id**: `EX36_actionability_score`
- **lens**: measurement_theory
- **validity_type**: External (E1 Content Validity); Construct (C5 Convergent Validity)
- **criteria**: E1 Content Validity; E2 Causal Sufficiency
- **what it measures**: Evaluates an interpretability finding along two dimensions: (1) concreteness: does the finding enable a specific intervention (steering, editing, pruning, monitoring) with a measurable outcome? (2) validation: has the intervention been empirically tested on held-out data? The product concreteness x validation gives the actionability score. Also reports the domain classification (safety, robustness, knowledge editing, capability elicitation, human-AI interaction) from the Orgad-Barez taxonomy.
- **pass condition**: actionability_score > 0.5 (both concrete and validated); findings that are concrete but unvalidated (concreteness > 0.7, validation < 0.3) are flagged as "actionable but unvalidated" --- the most dangerous category.
- **implementation notes**: Semi-automated. Concreteness is assessed by checking whether the finding includes (a) a specific model component, (b) a specific intervention type, and (c) a measurable outcome variable. Validation is assessed by checking whether the intervention was tested on data not used to discover the finding. The framework subsumes this: actionability maps to E-frame criteria, but the framework also provides M-frame reliability and C-frame construct validity, which the Orgad-Barez paper does not address.
- **why critical**: The ICML 2026 paper arrives at the same thesis as the framework (evaluation criteria are the bottleneck) but from a different direction (actionability rather than measurement theory). The convergence of two independent arguments strengthens the framework's case. The gap: Orgad-Barez have no measurement theory (M-frame) and no construct validity (C-frame). The framework is strictly more complete.

### B3. Deception Feature Validity Score

- **metric_id**: `EX37_deception_feature_validity`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency); Construct (C4 Discriminant Validity)
- **criteria**: E2 Causal Sufficiency; C4 Discriminant Validity; I3 Specificity
- **what it measures**: For a feature claimed to detect deception, tests three properties: (1) causal sufficiency: suppressing the feature reduces deceptive outputs AND activating it induces deceptive outputs on non-deceptive inputs; (2) discriminant validity: the feature fires on strategically deceptive behavior but NOT on surface-level deception-correlated outputs (e.g., hedging, uncertainty, politeness); (3) goal-representation specificity: the feature encodes goal-directed deception (requires evidence that the model represents the goal of deceiving, not just outputs that happen to be false).
- **pass condition**: causal_sufficiency > 0.5; discriminant_ratio > 2.0 (deception activation / adjacent-construct activation); goal_representation_evidence = TRUE (at least one downstream test confirms goal encoding)
- **implementation notes**: The hardest part is discriminant validity. Deception features must be distinguished from features for uncertainty, hedging, sycophancy, and confabulation --- all of which produce outputs that look deceptive but arise from different mechanisms. The goal-representation test requires showing that the feature's activation correlates with the model having an internal representation of the user's belief state (theory of mind), not just with output patterns. This is currently unsolved at scale; the metric provides the diagnostic structure even if the goal-representation test must be marked as "not yet implementable."
- **why critical**: Nanda's post makes explicit what the framework formalizes: "no deception feature" does not prove "no deception." The isolated demand for rigor (holding interpretability to a standard that no other method meets) is a field-level validity problem. But conversely, claiming to have found a deception feature requires the strongest possible validation --- E2 causal sufficiency for goal representations. This is the hardest validity test in the framework and the most safety-critical.

### B4. Heuristic Explanation Certified Accuracy

- **metric_id**: `EX38_heuristic_explanation_accuracy`
- **lens**: measurement_theory
- **validity_type**: External (E2 Causal Sufficiency); Measurement (M1 Reliability)
- **criteria**: E2 Causal Sufficiency (worst-case certified); M1 Reliability
- **what it measures**: For a mechanistic explanation of a model component, tests whether the explanation provides worst-case certified accuracy: (1) prediction_accuracy: on held-out inputs, does the explanation correctly predict the component's output? (2) extreme_behavior_prediction: on adversarially constructed inputs designed to maximally challenge the explanation, does the prediction still hold? (3) human_legibility: can a human reader use the explanation to predict the component's behavior without seeing the model weights? (assessed via human study or proxy).
- **pass condition**: prediction_accuracy > 0.8; extreme_behavior_accuracy > 0.5 (explanations that fail on extreme inputs have unverified tails); human_legibility is a diagnostic, not pass/fail
- **implementation notes**: Directly implements ARC's heuristic explanation desiderata. The worst-case certification is the key differentiator from standard faithfulness metrics: standard metrics test average-case, ARC requires worst-case. For the extreme behavior test, use adversarial input generation (gradient-based or search-based) to find inputs that maximally violate the explanation's predictions. The gap between average-case and worst-case accuracy is the "tail risk" of the explanation.
- **why critical**: ARC's heuristic arguments are the formal complement to the framework's empirical validity criteria. Where the framework tests "does this explanation work empirically?", ARC tests "can we formally certify that it works in the worst case?" The two approaches are complementary: empirical validity is necessary (ARC's formal methods are currently limited to random MLPs), and formal certification is the gold standard (currently out of reach for trained networks).

### B5. Surprise Accounting Score

- **metric_id**: `EX39_surprise_accounting`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency); External (E2 Causal Sufficiency)
- **criteria**: E2 Causal Sufficiency; I2 Compositional Sufficiency
- **what it measures**: For a decomposition of a model into components, measures how much each component reduces uncertainty about model outputs. Reports (1) total_surprise_explained: sum of per-component surprise reductions as a fraction of total model surprise; (2) surprise_gap: 1 - total_surprise_explained (unexplained surprise = missing components or interactions); (3) interaction_surprise: surprise explained by component interactions that is not captured by any single component.
- **pass condition**: total_surprise_explained > 0.8 (the decomposition accounts for >80% of model uncertainty reduction); interaction_surprise < 0.2 (interactions should not dominate)
- **implementation notes**: Surprise = negative log probability of output. Per-component surprise reduction = difference in output entropy with vs. without the component (ablated). The key challenge is that ablation effects are not additive: removing component A and component B together may have different effect than the sum of removing each separately. Interaction surprise captures this non-additivity. For the base case (random MLPs), ARC has shown that mechanistic estimation works; for trained networks, this remains an open problem. The metric provides the measurement framework even for trained networks, with the caveat that interaction terms may dominate.
- **why critical**: Surprise accounting is ARC's formalization of "does the explanation account for all of the model's behavior?" This is exactly E2 Causal Sufficiency measured in bits rather than in task performance. The information-theoretic framing is more principled than task-specific performance metrics because it is task-agnostic. The practical barrier: computing per-component surprise requires ablation studies, which are expensive and may have interference effects.

### B6. Behavior Evaluation Sufficiency

- **metric_id**: `EX40_behavior_evaluation`
- **lens**: measurement_theory
- **validity_type**: External (E2 Causal Sufficiency)
- **criteria**: E2 Causal Sufficiency (downstream); E1 Content Validity
- **what it measures**: For a model safety claim based on interpretability (e.g., "this model has no deception features"), tests whether the claim predicts actual model behavior: (1) behavior_prediction_accuracy: does the interpretability-based claim correctly predict model behavior on held-out behavioral test suites? (2) coverage: what fraction of safety-relevant behaviors are addressed by the interpretability claim? (3) calibration: for probabilistic claims, are the stated confidences well-calibrated against observed behavior rates?
- **pass condition**: behavior_prediction_accuracy > 0.7; coverage > 0.5; calibration_error < 0.15
- **implementation notes**: Implements Steinhardt's thesis: behavior evaluation is the downstream validity test for interpretability claims. An interpretability finding that does not predict behavior is either wrong or irrelevant. The behavioral test suite should include adversarial probes (red-team style) and naturalistic probes (realistic deployment scenarios). Coverage is assessed against a taxonomy of safety-relevant behaviors (from METR, AISI, or similar).
- **why critical**: Steinhardt's argument maps directly to the framework's E-frame vs. M-frame distinction. The field invests heavily in M-frame reliability (are our measurements stable?) but under-invests in E-frame downstream validity (do our measurements predict behavior?). This metric operationalizes the E-frame priority.

---

## C. New Adapter Types Needed

Part XI papers operate primarily at the evaluation-criteria level
rather than the model-representation level. No new adapter classes
are required. The key needs are evaluation protocols, not
architectural modifications.

| Adapter Enhancement | For | Why | Priority |
|---|---|---|---|
| `AdversarialAblationProtocol` (method on existing adapter) | VPD-style adversarial subset removal | Requires greedy backwards elimination over subcomponent elements with per-step performance measurement; adds adversarial search to existing ablation methods | HIGH |
| `GoalRepresentationProbe` (method on existing adapter) | Deception feature goal-representation test | Requires probing for theory-of-mind features (belief state encoding) and linking to deception feature activation; currently unsolved at scale | LOW (research frontier) |

---

## D. Key Paper Framings (strongest arguments from Part XI)

### D1. Adversarial ablation as gold-standard E2 (VPD)

> "A subcomponent is valid iff it survives adversarial ablation.
> Non-adversarial attribution methods (including CLT graphs) may
> identify causally wrong nodes because they test average-case, not
> worst-case, sufficiency."

VPD's key contribution is not the decomposition itself but the
*verification criterion*: adversarial ablation. Standard ablation
(random subset removal, mean ablation, zero ablation) tests
average-case causal sufficiency. Adversarial ablation tests
worst-case: find the subset whose removal maximally degrades
performance, and report how much of the original behavior survives.
The gap between random and adversarial ablation is the "adversarial
gap" --- papers that report only random ablation have an untested
adversarial gap that may be large.

VPD also breaks what they call the "attention barrier": prior MI
methods could interpret MLP weights (neurons have fixed directions)
and SAE features (sparse, interpretable directions), but attention
weight matrices resisted interpretation because QK and OV circuits
mix heads in non-separable ways. VPD decomposes attention weights
into interpretable rank-1 parameter terms that each have a clear
semantic role, verified by adversarial ablation. This is the first
method with interpretable parameter-level attention decompositions.

**Paper reference**: Goodfire, "Interpreting the Learned Compositions
of a Language Model's Parameters" (goodfire.ai/research/
interpreting-lm-parameters), AlignmentForum May 5 2026, 161 karma.

**Use in paper**: Gold-standard E2 test. The adversarial ablation
criterion should be adopted as the framework's highest E2 tier.
Papers that report only random ablation are testing a weaker
criterion. The VPD adversarial gap diagnostic should be included in
every E2 evaluation as a calibration: "how much worse would this
result look under adversarial ablation?"

### D2. Evaluation criteria are the bottleneck (Orgad-Barez, ICML 2026)

> "The missing ingredient is evaluation criteria, not methods. Two
> dimensions: concreteness (enables specific intervention?) and
> validation (empirically tested?)."

Orgad, Barez et al. arrive at the same thesis independently: the
field has plenty of interpretability methods but no systematic way
to evaluate whether the outputs of those methods are valid. Their
two dimensions (concreteness, validation) map to the E-frame of the
validity framework (actionability ~ E1 Content Validity +
E2 Causal Sufficiency). However, the Orgad-Barez paper stops at the
E-frame. It has no measurement theory (M-frame: are the evaluations
reliable? do they have test-retest stability?) and no construct
validity (C-frame: are we measuring what we think we're measuring?
is the construct well-defined?).

The framework subsumes the Orgad-Barez paper: their concreteness
dimension is E1, their validation dimension is E2, and the
framework adds M-frame (measurement reliability) and C-frame
(construct validity) layers that their paper does not address.

Their five application domains (safety, robustness, knowledge
editing, capability elicitation, human-AI interaction) provide a
useful taxonomy for categorizing the framework's E-frame tests by
downstream application.

**Paper reference**: Orgad, Barez et al., "Interpretability Can Be
Actionable: Evaluation Criteria for Interpretability in Practice,"
arXiv:2605.11161, ICML 2026.

**Use in paper**: Convergent evidence. Two independent groups (the
framework authors and the ICML 2026 authors) arrive at the same
thesis (evaluation criteria are the bottleneck). The framework goes
further (M-frame + C-frame). This is a C5 Convergent Validity
argument for the framework itself.

### D3. Interpretability gives evidence, not certainty (Nanda)

> "Finding 'no deception feature' doesn't prove no deception. The
> absence of evidence is not evidence of absence --- especially
> when the search space is exponential."

Nanda's deception post is the clearest statement of
interpretability's epistemic limits. Three key arguments:

1. **Absence asymmetry**: Finding a deception feature is strong
   evidence of deception capability; NOT finding one is weak evidence
   of safety. The search space (all possible feature combinations) is
   exponentially large; failure to find something in a large search
   space is uninformative.

2. **The isolated demand for rigor**: Holding interpretability to a
   standard (certainty) that no other safety method meets is
   counterproductive. Interpretability provides *mechanistic evidence*,
   which is complementary to behavioral evidence, not a replacement
   for certainty.

3. **Goal-representation problem**: Distinguishing strategic deception
   from confabulation requires evidence that the model represents the
   *goal* of deceiving --- not just that it produces deceptive outputs.
   This is an E2 Causal Sufficiency claim about goal representations,
   which is the hardest validity test in the framework.

The related post "Difficulties with Evaluating a Deception Detector"
(Dec 2025) makes this concrete: a deception detector that achieves
high accuracy on known-deceptive examples may still fail on novel
deception strategies because it detects surface correlates (hedging,
uncertainty) rather than the underlying goal representation.

**Paper reference**: Neel Nanda, LessWrong May 2025 (curated);
"Difficulties with Evaluating a Deception Detector," LessWrong
Dec 2025.

**Use in paper**: Epistemic framing. The framework does not claim to
solve deception detection --- it claims to make explicit what
evidence interpretability provides and what it does not. Nanda's
absence asymmetry is the motivation for the framework's distinction
between E2 Causal Sufficiency (what the evidence supports) and the
absence of E2 (what the evidence does not support). The framework
helps practitioners reason about the strength of their evidence
rather than making binary safe/unsafe claims.

### D4. Formal verification as complement to empirical validity (ARC)

> "A heuristic explanation is a mechanistic account that is
> human-legible, has worst-case certified accuracy, and predicts
> extreme behaviors."

ARC's heuristic argument framework is the formal-methods complement
to the empirical validity framework. Where the validity framework
asks "does this explanation work empirically on the data we have?",
ARC asks "can we formally certify that it works on ALL inputs,
including adversarial ones?"

Key concepts:

1. **Heuristic explanations**: Mechanistic accounts with three
   properties: (a) human-legible (a human can use them to predict
   behavior), (b) worst-case certified accuracy (provably correct
   within a bound), (c) extreme-behavior prediction (predict what the
   model does on the hardest inputs, not just average inputs).

2. **Surprise accounting**: Decompose the model's total output
   surprise (negative log probability) into per-component
   contributions. If the components' surprise contributions sum to
   the total, the decomposition is complete. If there is a gap,
   something is missing (interaction effects, unidentified
   components, or model behaviors not captured by the decomposition).

3. **Mechanistic estimation**: For random MLPs (the base case), ARC
   has shown that surprise accounting works --- the mechanistic
   estimate matches the true model output. For trained networks, this
   remains open. The gap between random and trained networks is the
   "training gap" in mechanistic estimation.

**Paper reference**: ARC, multiple posts on alignmentforum.org;
Christiano et al. heuristic arguments series; surprise accounting
posts.

**Use in paper**: Formal complement. The framework provides empirical
validity criteria (do the measurements work?); ARC provides formal
validity criteria (can we certify they work?). The two are
complementary, not competing. Currently, the framework is
implementable (empirical tests can be run today), while ARC's formal
certification is limited to base cases (random MLPs). The long-term
goal is convergence: empirical validity now, formal certification
when the theory catches up.

### D5. Behavior evaluation as downstream validity (Steinhardt)

> "Behavior evaluation > capability evaluation for safety. The
> under-investment in behavior evaluation is the under-investment
> in E2 and downstream validity."

Steinhardt's argument maps directly to the framework's E-frame vs.
M-frame distinction:

- **Capability evaluation** (how well does the model perform on a
  benchmark?) maps to **M-frame** (measurement reliability): are we
  measuring performance accurately?
- **Behavior evaluation** (what does the model actually do in
  deployment?) maps to **E-frame** (external/downstream validity):
  do our measurements predict real-world behavior?

The field's under-investment in behavior evaluation is isomorphic to
the framework's observation that MI research is over-indexed on
M-frame metrics (SAE reconstruction loss, feature sparsity, circuit
faithfulness on the discovery task) and under-indexed on E-frame
metrics (does the circuit predict behavior on new tasks? does the
feature cause the behavior it claims to describe?).

**Paper reference**: Jacob Steinhardt, "The Case for Evaluating
Model Behaviors," May 20 2026 (thegradient.pub or personal blog).

**Use in paper**: Strategic alignment. The framework operationalizes
Steinhardt's argument at the MI level: behavior evaluation for
interpretability claims. Every interpretability finding should be
tested against behavioral predictions, not just measurement
reliability.

### D6. Funding signals validate the framework's applied arm (Schmidt Sciences)

> "$300K--$1M to detect and mitigate deceptive behaviors. 'Move
> beyond academic benchmarks.' Red team vs. blue team structure."

The Schmidt Sciences RFP for deceptive behavior detection is a
near-literal description of the framework's applied arm:

1. **Detect deceptive behaviors** = E2 Causal Sufficiency for
   deception features (B3 metric above).
2. **Move beyond academic benchmarks** = the framework's thesis that
   benchmarks without validity criteria are insufficient.
3. **Red team vs. blue team** = adversarial ablation (B1 metric
   above) applied to deception detectors.

The larger Science of Trustworthy AI RFP ($1M--$5M+) frames
trustworthiness as requiring foundational measurement --- the
framework provides the measurement-theoretic foundation.

**Paper reference**: Schmidt Sciences, schmidtsciences.org; RFP due
May 26 2026 (deceptive behaviors); Science of Trustworthy AI RFP
(rolling).

**Use in paper**: Appendix or introduction framing. The framework is
not just academically motivated --- it addresses specific funding
priorities from major philanthropic organizations. The deceptive
behaviors RFP is the most direct external validation that the
framework addresses a recognized need.

### D7. The field-level phase transition

The forum-level trajectory across LessWrong, AlignmentForum, and
arXiv reveals a four-phase pattern:

- **Phase 1 (2022--23): Optimism.** The "interpretability is going to
  work" era. Toy models, induction heads, sparse probing. The
  implicit assumption: scale the methods and they will reveal model
  internals.

- **Phase 2 (2024): Proliferation without standards.** SAE variants
  multiply (TopK, BatchTopK, Gated, JumpReLU, Matryoshka, Adaptive).
  Circuit discovery scales to GPT-2. Feature dashboards proliferate.
  No shared evaluation criteria. Every paper uses different metrics.

- **Phase 3 (2025): Reckoning.** The negative results accumulate:
  Nanda's deception limits (May 2025), r=0.006 SAE-behavior
  correlation, CoT unfaithfulness, superposition as geometric scaling
  law (NeurIPS 2025 Best Paper Runner-Up), DMSAE's 197/65k stable
  features. The field begins to confront the gap between method
  output and validated understanding.

- **Phase 4 (2026): Infrastructure push.** The response to Phase 3:
  build the evaluation infrastructure. SAEBench audit (May 2026),
  MechEvalAgent 93% reproducibility failure (Feb 2026), NLAs (May
  2026), VPD adversarial ablation (May 2026), ICML actionability
  paper (ICML 2026), Steinhardt behavior evaluation (May 2026),
  Open Problems in MI paper (TMLR 2026). The field is actively
  building the evaluation layer that was missing in Phases 1--3.

The framework enters at Phase 4's peak: maximum receptivity, maximum
density of convergent arguments, and maximum practical urgency (Schmidt
Sciences funding, commercial MI via Goodfire, regulatory pressure).

**Use in paper**: Introduction or related work. The four-phase
narrative contextualizes the framework as the natural culmination of
the field's own trajectory, not an external critique. Phase 4 is
where the field arrived at the same conclusion the framework
formalizes.

---

## E. Gaps Identified

### E1. No adversarial ablation baseline for existing circuits

Every circuit paper in the literature reports random or targeted
ablation results. None reports adversarial ablation (worst-case subset
removal). The adversarial gap --- the difference between random and
adversarial ablation --- is unmeasured for all existing circuits
(IOI, greater-than, indirect object identification, docstring, etc.).
The B1 metric (VPD Adversarial Ablation Score) fills this gap, but
applying it retroactively to existing circuit claims is high-priority
empirical work.

### E2. No M-frame or C-frame for actionability

The Orgad-Barez ICML 2026 paper defines actionability along two
E-frame dimensions (concreteness, validation) but has no measurement
reliability criterion (M-frame: is the actionability assessment
itself reliable across evaluators? across random seeds?) and no
construct validity criterion (C-frame: is "actionability" a
well-defined construct, or does it conflate different types of
usefulness?). The framework fills both gaps.

### E3. No operational protocol for goal-representation testing

Nanda's deception argument requires distinguishing strategic deception
(goal-directed) from confabulation (output-correlated). This requires
testing whether the model represents the *goal* of deceiving, which
in turn requires a theory-of-mind probe. No operational protocol for
this exists at scale. The B3 metric (Deception Feature Validity Score)
defines the evaluation structure but marks the goal-representation
test as "not yet implementable."

### E4. Heuristic arguments limited to base cases

ARC's heuristic arguments and surprise accounting have been validated
only on random MLPs (the base case). Extension to trained networks is
an open problem. The "training gap" --- the difference between
mechanistic estimation accuracy on random vs. trained networks ---
is unmeasured and may be large. The framework's empirical approach
works for trained networks today; formal certification remains a
research frontier.

### E5. Behavior evaluation taxonomy for MI is undefined

Steinhardt argues for behavior evaluation over capability evaluation,
but does not provide a taxonomy of safety-relevant behaviors
specifically tailored to MI claims. The framework needs a behavior
taxonomy: for each type of interpretability claim (circuit,
feature, direction, decomposition), what behavioral predictions does
the claim entail, and how should those predictions be tested? The
B6 metric (Behavior Evaluation Sufficiency) provides the
measurement structure; the taxonomy of claim-to-behavior mappings
is still needed.

### E6. No validity framework for funding-driven research

The Schmidt Sciences RFPs create incentive gradients that may produce
validity-washing: research that appears to satisfy the RFP criteria
(detect deception, move beyond benchmarks) without actually achieving
validated understanding. The framework should include criteria for
evaluating whether funding-driven research meets genuine validity
standards vs. surface-level compliance.

---

## F. Cross-References to Previous Parts

| Part XI finding | Previous parts connection | Implication |
|---|---|---|
| VPD adversarial ablation (gold-standard E2) | Part X D3 (OpenAI weight-sparse circuits as highest E2) | VPD provides a more stringent E2 criterion (adversarial vs. targeted ablation); complements OpenAI's existence proof with a verification method applicable to any decomposition |
| VPD attention barrier breakthrough | Part VII B8 (CircuitLens weight-circuit recovery); Part X D3 (weight-sparse circuits) | VPD achieves interpretable attention weight decomposition that CircuitLens and weight-sparse transformers approach from different angles; three convergent methods for weight-level attention interpretation |
| ICML 2026 actionability = E-frame only | Part IX D1 (SAEBench audit derives M-frame); Part X D1 (NLA semantic validity gap = E1) | Three independent groups (framework, SAEBench auditors, ICML actionability) all arrive at "evaluation criteria are the bottleneck" but each covers different validity dimensions; full coverage requires all three |
| Nanda deception limits | Part VIII B3 (latent reasoning validity); Part IX D3 (safety subspace cluster) | Deception detection requires goal-representation E2 (harder than latent reasoning validity from Part VIII); safety subspace methods (Part IX) pass E2 for refusal but not for strategic deception |
| ARC heuristic arguments (formal complement) | Part VIII B6 (superposition regime diagnostic) | Both ARC and superposition scaling law (Part VIII) formalize properties of random vs. trained networks; the training gap is the shared open problem |
| Steinhardt behavior evaluation | Part IX D2 (93% reproducibility failure) | Steinhardt's under-investment in behavior evaluation parallels MechEvalAgent's finding that most MI research fails basic reproducibility; both are symptoms of the same E-frame under-investment |
| Schmidt Sciences RFP (deception) | Part IX D3 (safety representations most validity-passing); Part XI D3 (Nanda deception limits) | The RFP demands exactly what the framework provides: validated deception detection. Safety subspace work (Part IX) shows where validity is achievable; Nanda (Part XI) shows where it is not yet achievable |
| Phase 4 infrastructure push | Part IX D1 (SAEBench audit); Part X D7 (ecosystem gap) | The four-phase trajectory explains why SAEBench audit (Phase 4 self-correction), NLAs (Phase 4 infrastructure), and the framework (Phase 4 evaluation layer) all emerge simultaneously |

---

## G. Priority Matrix: Part XI Additions

| Item | Date | Key Contribution | Framework Zone | Priority |
|---|---|---|---|---|
| VPD Adversarial Ablation | May 2026 | Gold-standard E2; adversarial gap diagnostic | E2, I1 (verification criterion) | CRITICAL |
| ICML 2026 Actionability | May 2026 | Independent convergence on "criteria are bottleneck"; framework strictly subsumes | E1, E2 (convergent evidence for framework thesis) | CRITICAL |
| Nanda Deception Limits | May 2025 / Dec 2025 | Epistemic limits of MI for deception; goal-representation E2 requirement | E2, C4 (hardest validity test) | HIGH |
| ARC Heuristic Arguments | 2023--2026 | Formal complement to empirical validity; surprise accounting | E2, M1 (formal certification) | HIGH |
| Steinhardt Behavior Evaluation | May 2026 | E-frame vs. M-frame; under-investment in downstream validity | E2 (strategic alignment) | HIGH |
| Schmidt Sciences RFPs | May 2026 | Funding signals validating framework's applied arm | All (external validation) | MEDIUM (strategic) |
| Forum Phase Transition | 2022--2026 | Contextualizes framework as Phase 4 culmination | All (narrative framing) | MEDIUM (narrative) |

---

## H. Strategic Implications for the Paper

### H1. The verification criterion argument (VPD)

VPD's adversarial ablation provides the framework with its strongest
E2 test: a subcomponent is valid iff it survives worst-case subset
removal. This should be adopted as the framework's gold-standard E2
criterion, with random ablation downgraded to a weaker tier. The
adversarial gap (random vs. adversarial ablation performance) is a
diagnostic that should be reported for every E2 claim. This is the
single most important methodological addition from Part XI.

### H2. Convergent evidence for the framework thesis (ICML 2026)

The Orgad-Barez paper is the strongest external validation of the
framework's core thesis. Two independent groups (the framework
authors and an ICML 2026 paper) arrive at the same conclusion:
evaluation criteria, not methods, are the bottleneck. The framework
goes further (M-frame + C-frame), which means the framework is the
more complete solution. This convergence should be highlighted in
the paper's related work as C5-style evidence for the framework
itself.

### H3. The epistemic honesty argument (Nanda)

Nanda's deception posts motivate the framework's most important
meta-point: the framework helps practitioners reason about the
*strength* of their evidence, not make binary claims. "We found no
deception feature" is not a safety guarantee --- the framework
makes explicit why (absence of E2 evidence is not evidence of E2
absence). This epistemic framing is essential for the paper's safety
narrative: the framework is useful precisely because it makes
uncertainty legible, not because it eliminates it.

### H4. Empirical now, formal later (ARC)

ARC's heuristic arguments provide the long-term vision: formal
certification of mechanistic explanations. The framework is the
empirical bridge: implementable today, converging toward formal
certification as the theory matures. The paper should position the
framework as the empirical foundation on which formal methods will
eventually be built, not as a competitor to formal methods.

### H5. Maximum receptivity window (Phase 4)

The four-phase trajectory analysis shows the framework enters at the
peak of Phase 4: maximum density of "we need evaluation criteria"
arguments (VPD, ICML actionability, SAEBench audit, MechEvalAgent,
Steinhardt), maximum practical urgency (Schmidt Sciences funding,
Goodfire commercial MI, regulatory pressure), and maximum
intellectual readiness (the field has processed the Phase 3 negative
results and is actively seeking solutions). Delayed submission risks
missing this window as Phase 4 matures and specific solutions
(rather than frameworks) become the focus.

### H6. The applied arm has funding targets (Schmidt Sciences)

The Schmidt Sciences RFPs provide concrete, near-term funding
targets for the framework's applied arm. The deceptive behaviors
RFP ($300K--$1M, due May 26 2026) is nearly a literal description
of the framework applied to deception detection. The Science of
Trustworthy AI RFP ($1M--$5M+) is the larger-scale target. The paper
should be positioned to support both applications without being
narrowly tailored to either.

### H7. The attention barrier is broken (VPD)

VPD's parameter-level attention decomposition resolves a long-standing
limitation: prior MI methods could interpret MLP neurons and SAE
features but not attention weights. With VPD, all major model
components (MLPs, attention, embeddings) have at least one
interpretable decomposition method, each of which needs validity
testing. The framework is now needed for the full model, not just
the components that were previously interpretable.
