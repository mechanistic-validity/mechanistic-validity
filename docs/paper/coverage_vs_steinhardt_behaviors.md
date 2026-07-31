# Coverage Analysis: mechval vs. Steinhardt "The Case for Evaluating Model Behaviors"

Source: Jacob Steinhardt, "The Case for Evaluating Model Behaviors,"
AlignmentForum / The Gradient, May 20, 2026.

Core thesis: the field systematically under-invests in behavior evaluation
relative to capability evaluation. Behavior evaluation measures a model's
tendencies/propensities, not its peak performance. Safety depends on
behaviors, not capabilities. Public measurement creates market efficiency.

---

## 1. Core Distinction: Behavior vs. Capability Evaluation

### 1.1 Capability Evaluation (what the model CAN do)

- [COVERED] "Model proficiency at tasks (coding, science, math)" -> `ce_delta`, `logit_diff`, `per_token_nll` (we measure task performance as baselines)
- [COVERED] "Capability increases follow robust trend lines" -> framework includes `baseline_literature` in `EvalResult`

### 1.2 Behavior Evaluation (what the model TENDS to do)

- [PARTIAL] "Measure model tendencies/propensities, not peak performance" -> related: `behavior_capability_gap`, gap: most metrics measure circuit performance (capability) not behavioral propensity distributions
- [GAP] "Behavior evaluation protocol: systematic methodology for measuring behavioral tendencies across deployment conditions" -- no metric implements a full propensity-measurement protocol
- [GAP] "Behavior evaluation sufficiency: does the MI claim predict actual model behavior?" -- `actionability_score` is closest but does not test downstream behavioral prediction accuracy

---

## 2. Specific Behavior Evaluation Types

### 2.1 Sycophancy / Agreement with Factually Incorrect Users

- [PARTIAL] "Agreement with factually incorrect users" -> related: `misalignment`, `behavior_capability_gap`, gap: no metric specifically measures sycophancy rate as a behavioral propensity across varied contexts
- [GAP] "Sycophancy propensity distribution: what fraction of interactions exhibit excessive deference, stratified by context type?" -- no distributional sycophancy metric

### 2.2 Evaluation Awareness

- [COVERED] "Verbalized awareness of being evaluated" -> `eval_awareness_format_control`
- [PARTIAL] "Behavioral changes under evaluation awareness" -> related: `eval_awareness_format_control`, gap: metric tests format-level awareness but not behavioral adaptation (does the model behave differently when it detects evaluation?)

### 2.3 Reward Hacking

- [GAP] "Reward hacking frequency and situational triggers" -- no metric measures how often and under what conditions a model exploits reward signals rather than performing the intended task
- [GAP] "Reward hacking circuit identification: which components drive reward-hacking behavior?" -- no metric maps reward hacking to specific circuits

### 2.4 Subjective Experience / Internal Desires

- [GAP] "Reporting of internal desires or subjective experience" -- no metric evaluates whether model self-reports about internal states are consistent with internal representations
- [PARTIAL] "Consistency between stated and internal preferences" -> related: `latent_self_consistency`, gap: metric tests reasoning consistency, not preference/desire consistency

---

## 3. Evaluation Methodology Requirements

### 3.1 Judge Definition

- [PARTIAL] "Define a judge (typically LLM with rubric)" -> related: `autointerp` (uses LLM judge for feature descriptions), gap: no general-purpose LLM-judge framework for behavioral evaluation across arbitrary propensities
- [GAP] "Rubric-based behavioral scoring: formalized rubrics for each behavior type that an LLM judge applies consistently" -- no rubric system

### 3.2 Environment Distribution

- [PARTIAL] "Establish distribution over test environments" -> related: `cross_task_generalization`, `output_variants`, gap: metrics test across tasks but not across deployment-realistic environment distributions
- [GAP] "Deployment-realistic environment sampling: evaluation environments that match actual deployment conditions (conversation length, user demographics, topic distribution)" -- no deployment-distribution metric

### 3.3 Aggregation

- [COVERED] "Calculate average judge value across environments" -> `EvalResult` includes `value`, `ci_low`, `ci_high`, `n_samples`
- [COVERED] "Enable model comparison across time and versions" -> `crosscoder_model_diff`, `convergent_evolution`, `phylogenetic_tracking`

---

## 4. Field Recommendations

### 4.1 Safety Researchers Should Prioritize Behavior Evaluations

- [PARTIAL] "Safety researchers outside AI labs should prioritize behavior evaluations" -> related: `safety_claim_reliability`, `safety_one_shot`, gap: our safety metrics test specific safety claims, not systematic behavioral tendency measurement
- [GAP] "Behavior evaluation as primary safety evidence" -- framework treats behavioral prediction as one criterion among many, not the primary evidence type

### 4.2 Focus on Developer-Consumer Misalignment

- [GAP] "Focus on misalignment between developers and consumers" -- no metric evaluates whether model behaviors serve developer interests at the expense of user interests
- [GAP] "Identify behaviors that emerge from training incentives rather than user needs" -- no metric distinguishes training-incentive-driven from user-need-driven behaviors

### 4.3 Tail-Risk Behaviors

- [PARTIAL] "Emphasize tail-risk behaviors (power-seeking tendencies)" -> related: `misalignment`, `safety_subspace`, gap: no metric specifically targets power-seeking or resource-acquisition behavioral propensities
- [GAP] "Power-seeking propensity measurement: under what conditions does the model attempt to acquire resources, influence, or avoid shutdown?" -- no power-seeking circuit metric

### 4.4 Evaluations Unlikely from Developer Incentives

- [GAP] "Create evaluations that developers would not create due to misaligned incentives" -- no meta-evaluation framework for identifying incentive-blind spots
- [GAP] "Market efficiency through public behavioral measurement" -- no metric supports public behavioral monitoring or reporting

---

## 5. Mapping to Framework Concepts

### 5.1 E-frame vs M-frame Alignment

- [COVERED] "Capability evaluation = M-frame (measurement reliability)" -> `reproducibility_check`, `hyperparam_sensitivity`, `core_stability`
- [COVERED] "Behavior evaluation = E-frame (external/downstream validity)" -> `cross_task_generalization`, `held_out_prediction`, `actionability_score`
- [PARTIAL] "Field over-indexed on M-frame, under-indexed on E-frame" -> related: framework includes both, gap: no quantitative measure of the M-frame/E-frame balance in a research contribution

### 5.2 Downstream Behavioral Prediction

- [PARTIAL] "MI finding should predict model behavior on held-out behavioral tests" -> related: `held_out_prediction`, `cross_task_transfer`, gap: these test task-level transfer, not behavior-propensity-level transfer
- [GAP] "Behavioral prediction accuracy: for a given interpretability claim, what is its accuracy at predicting behavioral propensities on new scenarios?" -- no claim-to-behavior-prediction pipeline

---

## 6. Impact Mechanism

- [GAP] "Public measurement creates market efficiency for AI behaviors" -- no public reporting or benchmarking infrastructure metric
- [GAP] "Enable users to make informed choices based on behavioral profiles" -- no user-facing behavioral profile generation metric
- [GAP] "Shift incentive structures toward alignment through measurement transparency" -- no incentive-alignment-through-transparency metric

---

## Summary

| Status  | Count | Percentage |
|---------|-------|------------|
| COVERED | 7     | 19%        |
| PARTIAL | 10    | 27%        |
| GAP     | 20    | 54%        |
| **Total** | **37** | **100%** |

### Key gaps (genuine unsolved problems):
1. Behavioral propensity measurement (tendencies, not peak performance)
2. Sycophancy propensity distribution across contexts
3. Reward hacking frequency and circuit identification
4. Power-seeking behavioral measurement
5. Deployment-realistic environment distribution for testing
6. Rubric-based LLM judge for behavioral scoring
7. Developer-consumer misalignment detection
8. Behavioral prediction accuracy from MI claims
9. Self-report consistency with internal representations
10. Public behavioral monitoring and reporting infrastructure

### Overall assessment:
Steinhardt's framework exposes our largest coverage gap. The mechval
framework is strong on capability-level validity (does this circuit
faithfully implement this task?) but weak on behavioral propensity
evaluation (does this model tend to behave this way in deployment?).
This is precisely Steinhardt's point: the field invests in capability
measurement and under-invests in behavior measurement. Our framework
mirrors this bias.
