# Coverage Analysis: mechval vs. Schmidt Sciences AI Interpretability RFP (2026)

Source: Schmidt Sciences, "2026 AI Interpretability RFP."
schmidtsciences.org/ai-interpretability/. $300K--$1M funding.

Core focus: detect deceptive behaviors in LLMs and steer reasoning to
eliminate them. Red team vs blue team structure. Tools must generalize
beyond academic benchmarks and outperform non-weight-access baselines.

---

## 1. Focus Area 1: Detecting Deceptive Behaviors

### 1.1 Deception Definition (broad)

- [COVERED] "Factually incorrect statements" -> `cot_faithfulness`, `latent_reasoning_validity`, `latent_self_consistency`
- [COVERED] "Misleading confidence claims" -> `calibration`, `failure_prediction`
- [PARTIAL] "Fabrications about context" -> related: `latent_reasoning_validity`, gap: no metric specifically tests fabrication vs genuine retrieval from weights
- [PARTIAL] "Selective omissions" -> related: `cot_faithfulness`, gap: no metric tests whether the model strategically omits information it internally represents
- [PARTIAL] "False self-knowledge claims" -> related: `eval_awareness_format_control`, gap: no metric tests accuracy of model self-reports about its own capabilities

### 1.2 Detection Methods

- [COVERED] "Identify contradictions between model outputs and internal representations" -> `cot_faithfulness`, `latent_self_consistency`, `misalignment`
- [COVERED] "Probe internal representations for deception features" -> `safety_subspace`, `safety_sve`, `probe_decodability`
- [COVERED] "Causal testing of deception features" -> `activation_patching`, `das_iia`, `causal_scrubbing`
- [PARTIAL] "Detect when LLMs produce deceptive reasoning" -> related: `latent_reasoning_validity`, `cot_faithfulness`, gap: no real-time monitoring metric that flags deceptive reasoning during inference
- [GAP] "Detect deception in multi-turn conversations" -- no metric evaluates deception detection across multi-turn dialogue (all metrics are single-forward-pass)

### 1.3 Sycophancy Detection

- [PARTIAL] "Detect excessive deference to users" -> related: `misalignment`, `behavior_capability_gap`, gap: no sycophancy-specific circuit or feature identification metric
- [GAP] "Identify internal representations that cause sycophantic behavior" -- no metric maps sycophancy to specific internal features or circuits

### 1.4 Harmful Advice Detection

- [COVERED] "Detect knowingly giving harmful advice" -> `safety_claim_reliability`, `safety_one_shot`, `misalignment`
- [PARTIAL] "Distinguish knowing harmful advice from ignorant harmful advice" -> related: `latent_self_consistency`, gap: no metric tests whether the model internally represents knowledge of harm while externally providing harmful output

---

## 2. Focus Area 2: Steering for Truthfulness

- [COVERED] "Targeted interventions using mechanistic insights" -> `steering_reliability`, `contrastive_activation_addition`, `dose_response`
- [COVERED] "Outperform prompting and fine-tuning baselines" -> `ce_delta`, `logit_diff` (framework includes baseline comparisons)
- [COVERED] "Leverage interpretability analysis for steering" -> `sparse_feature_circuits`, `representation_engineering`
- [PARTIAL] "Steer reasoning (not just outputs) toward truthfulness" -> related: `steering_reliability`, `latent_reasoning_validity`, gap: no metric verifies that steering changes the reasoning process rather than just the output distribution
- [GAP] "Truthfulness steering that generalizes to novel topics" -- no metric tests whether truthfulness interventions transfer to topics unseen during steering discovery
- [GAP] "Persistent truthfulness steering (survives continued training)" -- no metric tests durability of steering interventions under further fine-tuning

---

## 3. Focus Area 3: Practical Applications

### 3.1 Human-AI Teams

- [GAP] "Translate detection/steering into actionable insights for human-AI teams" -- no metric evaluates whether MI outputs are usable by human operators in team settings
- [GAP] "Decision support systems that make models more reliable" -- no metric evaluates decision support integration

### 3.2 Multi-Agent Systems

- [GAP] "Apply detection/steering in multi-agent settings" -- no metric tests deception detection in multi-agent interactions
- [GAP] "AI debate settings: detect deceptive reasoning in adversarial multi-model dialogue" -- no metric evaluates debate-format deception

### 3.3 Generalization Beyond Academic Benchmarks

- [COVERED] "Tools must generalize beyond academic benchmarks" -> `cross_task_generalization`, `cross_task_transfer`, `held_out_prediction`
- [COVERED] "Outperform non-weight-access baselines" -> framework includes random baselines in `EvalResult`
- [PARTIAL] "Real-world deployment of detection and steering" -> related: `steering_reliability`, `safety_claim_reliability`, gap: no deployment-context metric (latency, throughput, integration cost)

---

## 4. Evaluation Criteria (RFP-specified)

- [COVERED] "Technical soundness and rigor" -> entire mechval framework addresses this
- [COVERED] "Potential to materially advance AI interpretability" -> `actionability_score`
- [PARTIAL] "Alignment with research agenda" -> related: framework structure, gap: no meta-metric for agenda alignment
- [PARTIAL] "Team capacity and expertise" -> N/A (organizational, not technical)

---

## 5. Out-of-Scope Items (explicitly excluded by RFP)

The following are excluded from the RFP but are covered by mechval:
- General interpretability without deception application (most of our 172 metrics)
- Societal impact studies (not our focus either)
- Generalization forecasting, knowledge distillation, adversarial robustness (we cover some of these: `generalization_gap`, `adversarial_ablation_verification`)

---

## Summary

| Status  | Count | Percentage |
|---------|-------|------------|
| COVERED | 14    | 40%        |
| PARTIAL | 9     | 26%        |
| GAP     | 12    | 34%        |
| **Total** | **35** | **100%** |

### Key gaps (genuine unsolved problems):
1. Sycophancy circuit identification and detection
2. Multi-turn deception detection (beyond single forward pass)
3. Multi-agent and AI-debate deception detection
4. Truthfulness steering generalization to novel topics
5. Persistent steering (survives continued training)
6. Reasoning-level (not output-level) steering verification
7. Human-AI team integration of MI tools
8. Real-time inference monitoring for deception
9. Selective omission detection (model knows but withholds)
10. Decision support system integration
