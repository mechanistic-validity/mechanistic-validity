---
title: "Case Study: SAE Features"
description: "Sparse autoencoder features (Cunningham et al. 2024) evaluated through the five core lenses."
---

# Case Study: SAE Features

Sparse autoencoder features ([Cunningham et al. 2024](https://arxiv.org/abs/2309.08600)) are directions in activation space extracted by training an overcomplete dictionary on residual-stream activations, evaluated by automated interpretability scoring on 150 features per method against six baselines — the default basis, random directions, PCA, ICA, top-K PCA and top-K ICA — and by activation patching on 50 IOI data points, in Pythia-70M and Pythia-410M. Each feature is given a label — "Golden Gate Bridge," "deception," "code syntax" — based on the inputs that maximally activate it, and the claim is that these features are real computational units: [representational](/mechanistic-validity/framework/modes/representational)-level entities the model uses during inference. The scaled-up dictionaries of [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) make the same claim at larger scale.

This case study evaluates SAE features *as a class*. Individual strong features (those that replicate and steer) score higher; the bulk of the dictionary scores lower. The evaluations below reflect the typical case.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Proposed. **Capped by:** M2 (baseline separation).


| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C2 Structural plausibility (partial) | C3 Convergent validity | Weak |
| Internal (Neuroscience) | I1, I2, I3, I5, I7, I9 (all partial) | I4 Specificity (inconclusive), I6, I8, I10, I11, I12 (untested) | Weak |
| External (Pharmacology) | E4 Cross-model generalization (confirmed) | E3 Cross-task generalization (untested) | Partial |
| Measurement (Measurement Theory) | M4 Calibration (partial), M6 Invariance (partial) | M5 Sensitivity (disconfirmed), M2 Baseline separation (the capping criterion) | Weak |
| Interpretive (MI) | V5 Scope declaration (confirmed) | V1, V2, V3, V4 (all partial) | Weak |

**Overall verdict: Proposed.** SAE features as a class are capped by M2 (baseline separation): SAEBench demonstrated that some evaluation metrics score higher on random models than trained ones, undermining the assumption that high scores reflect learned structure. The strongest individual features (those that replicate across seeds, respond to steering, and have coherent decoder vectors) may approach Causally Suggestive with further evidence, but the bulk of any SAE dictionary remains at Proposed — the features have been identified and labeled, but the evidence for their reality as model-intrinsic computational units is thin across all five validity types.

This is not a claim that SAE features are wrong — many may be real. It is a claim that the evidence for their validity, measured against the same standards applied to circuits, has not been marshaled. The primary gaps are baseline separation (M2 — does the dictionary separate from a random network?), convergent validity (C3 — does a different method find the same features?), specificity (I4 — single-feature ablation moves 12,000 logits, left unanalyzed), and unlicensed labeling (V4 — is the label the right one?). These three gaps share a common theme: the features may be properties of the dictionary rather than properties of the model.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Max-activating examples (feature identification) | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| Activation steering / feature clamping | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |
| Feature ablation (zeroing) | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Decoder vector projection through unembedding | [B03 OV/QK Decomposition](/mechanistic-validity/framework/metrics/#b03) | Structural |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "SAE feature $f_{42}$" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Unclear.** What observation would disconfirm the claim that feature $f_{42}$ represents "deception"? If the disconfirming condition is "the feature does not activate on deceptive text," this is circular — the feature was *defined* by its activations. A genuine falsifiability condition would be: "if steering along $f_{42}$ does not increase deceptive outputs, or if a different SAE trained with a different random seed produces a feature with $J < 0.3$ overlap." Most papers do not state such conditions.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** The decoder vector $W_{\text{dec}}[f]$ should project onto semantically coherent tokens through the unembedding matrix. Some features pass this check ("Golden Gate Bridge" projects onto bridge-related tokens). Many features lack this structural verification.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** The origin defines its construct as a dictionary feature approximating an unknown ground-truth network feature, and its case studies score against a weaker operational stand-in: whether a direction admits one human-readable explanation. The neighboring constructs it must separate from — an atomic unit, a complete unit, and a direction that is an artifact of the dictionary — are the ones the follow-up tests, and atomicity fails: meta-SAE decomposition of a 49,152-latent GPT-2 dictionary recovers further structure inside latents.

**[C6 — Complementation validity:](/mechanistic-validity/framework/criteria/construct/complementation-validity) Untested.** Does a feature correspond to one computational role, or is it a blend of multiple roles that co-occur in training data? Feature splitting is the open question and no test of atomicity is run. Meta-SAEs decompose latents further, which disconfirms the reading that SAE features are atomic units.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Weak.** SAE features are identified by one method. A different SAE with different hyperparameters or random seed may produce a different feature set. Cross-seed consistency is partially reported for strong features but not systematically measured at Jaccard level across the full dictionary.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Unclear | No pre-registered disconfirming conditions |
| C2 Structural plausibility | Partial | Some decoder vectors project coherently |
| C4 Discriminant validity | Not tested | No discriminant evaluation |
| I3 Minimality | Open question | Polysemanticity unresolved |
| C3 Convergent validity | Weak | Single method, partial cross-seed |

### Key Distinctions

- **Confirmation vs corroboration:** Max-activating examples confirm the label (the feature activates on things matching the label), but this is circular — the label was derived from those same examples. Genuine corroboration would require an independent method (weight-space analysis, causal intervention) predicting the same concept before observing activations.
- **Natural kind vs family resemblance:** A polysemantic feature that activates on "Golden Gate Bridge" and "suspension bridges" and "orange paint" may be a natural kind (bridge-related concepts) or a family resemblance (co-occurring tokens in training data). Without structural grounding, the distinction is underdetermined.
- **Operationalism vs realism:** Feature labels like "deception" imply realism (the model has a deception concept). The evidence supports only operationalism (this direction activates on texts labeled deceptive by humans). The gap between these is the core validity question.

### Nomological Network

| Prediction the construct makes | How you test it | Confirmed? |
|---|---|---|
| Activates on inputs matching the label | Max-activating examples | Circular |
| Steering along the feature changes outputs | Activation steering / clamping | Sometimes |
| Decoder vector projects onto coherent tokens | $W_{\text{dec}}[f]$ through unembedding | Sometimes |
| Does *not* activate on related non-matches | Discriminant testing | Rarely tested |
| Same feature appears under different SAE seeds | Cross-seed Jaccard comparison | Partially |
| Corresponds to one role, not co-occurrence | Polysemanticity analysis | Open |

A thin nomological network. Two rows partially confirmed, several untested. The circularity of the first row (the primary evidence) weakens the network further — one confirmed node is methodologically dependent on the discovery procedure rather than being an independent test.

---

## Neuroscience Lens — Internal Validity

*Does ablating/restoring a feature change behavior in the expected way?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Sometimes.** Ablating (zeroing) strong features degrades behavior on their associated inputs. But "necessity" for an individual SAE feature is a weaker claim than circuit necessity — many features contribute small amounts, and removing one may be compensated by others. Necessity is established for a few strong features; for the bulk of the dictionary, it is untested.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Sometimes (via steering).** Clamping a feature to a high activation value can steer model outputs — the "Golden Gate Bridge" feature reliably produces bridge-related text. This is a form of sufficiency: the feature direction alone drives the behavior. But steering is blunt (high-magnitude clamping may go off-manifold), and many features do not produce coherent effects when steered.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Inconclusive.** The off-target extent of the intervention was measured, disclosed in one clause of a figure caption, and then removed from view. Ablating a single dictionary feature moves twelve thousand logits down; the paper names the one that moves most and reads the intervention as confirming the feature's interpretation. Nothing establishes that the other movements are noise — no comparison to ablating a random direction of the same norm, no report of which tokens they are, and a display threshold chosen for clarity rather than derived from a null.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Three designed controls are run, including an α = 0 dictionary. What is absent is a bound on how strong an unmeasured confounder would need to be (I8).

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Sometimes | Strong features: yes. Bulk: untested |
| I2 Sufficiency | Sometimes | Steering works for strong features |
| I4 Specificity | Not tested | No collateral damage measured |
| M1 Reliability | Weak | Strong features partially stable |
| I7 Confound control | Not tested | Single steering method |

### Key Distinctions

- **Single vs double dissociation:** Steering demonstrates single dissociation (activating the feature produces the expected behavior). Double dissociation (activating this feature does NOT produce a different behavior, and activating a different feature does NOT produce this behavior) is untested for the vast majority of features.
- **Lesion vs stimulation:** SAE features uniquely have both lesion (zeroing) and stimulation (clamping) evidence for strong features. However, the stimulation is at supraphysiological magnitudes (5-10x typical activation), making it unclear whether the observed effects reflect normal computation or off-manifold forcing.

### Dissociation Matrix

|  | Feature-labeled task | Related but distinct task | Unrelated task |
|---|---|---|---|
| Ablate feature $f$ | ↓ (sometimes) | ? | ? |
| Clamp feature $f$ | ↑↑ (strong features) | ? | ? |
| Ablate neighboring feature $g$ | ? | ? | ? |

The matrix is extremely sparse. Even for the best-characterized features, only two cells (ablate → own task, clamp → own task) have data. Without the off-diagonal cells, we cannot distinguish "this feature specifically implements this computation" from "this direction in activation space correlates with this behavior when artificially amplified."

---

## Pharmacology Lens — External Validity

*Does intervening on a feature produce predictable downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Partial.** Steering experiments show that clamping features can shift model behavior. The "Golden Gate Bridge" feature produces bridge-related responses across varied prompts. But the reach of most features (especially abstract or behavioral ones like "deception") is not well-characterized.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Partial.** Clamping at different multipliers (1x, 5x, 10x the typical activation magnitude) produces graded effects — stronger clamping produces more extreme outputs. But the dose-response is often nonlinear and poorly characterized. At high multipliers, outputs become incoherent rather than showing more of the feature.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Unknown.** Does the Golden Gate Bridge feature work equally well on questions, stories, code prompts, and multilingual inputs? Robustness across prompt distributions is not systematically tested.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Confirmed.** The method transfers everywhere tried: Pythia, GPT-2 small and Gemma 2 2B [Leask et al. 2025].

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Partial | Works for some features, untested for most |
| I4 Specificity | Not tested | Off-target effects unmeasured |
| E5 Graded response | Variable | Some strong, most unknown |
| E2 Prompt generalization | Unknown | No cross-distribution testing |
| E4 Cross-model generalization | Not tested | Model-specific by construction |

### Key Distinctions

- **Affinity vs efficacy:** SAE features demonstrate affinity (they activate on relevant inputs) but efficacy (causal contribution to behavior) is demonstrated only for strong features under supraphysiological clamping. At normal activation magnitudes, most features have unmeasured efficacy.
- **Therapeutic window:** The dose-response breakdown at high clamping magnitudes (coherent output → feature-saturated output → incoherent output) implies a narrow therapeutic window. The useful range for steering is bounded above by off-manifold effects, but its lower bound (minimum effective dose) is uncharacterized.
- **Off-target effects as the core problem:** The pharmacology lens reveals the fundamental gap — steering interventions change outputs, but whether they change *only* the intended behavior is almost never measured. A drug that cures the disease but causes ten side effects is not well-understood.

### Dose-Response Curve

For a typical strong SAE feature (e.g., "Golden Gate Bridge"):
- **0x activation**: baseline behavior
- **1x clamping**: subtle shift toward feature-related content
- **5x clamping**: clear feature-related output (the "demo" regime)
- **10x+ clamping**: incoherent, repetitive, or degenerate output

What's missing:
- **No systematic EC₅₀** — at what magnitude does the behavioral shift become reliably detectable?
- **No off-target measurement at each dose** — fluency, factuality, and other capabilities are not tracked alongside the feature effect
- **No comparison across features** — do all features have similar dose-response shapes, or do concrete features (Golden Gate Bridge) behave differently from abstract features (deception)?
- **No characterization for weak features** — the dose-response for the bulk of the dictionary is entirely unknown

The dose-response evidence shows that *something* happens when you intervene, but the curve's shape, selectivity boundary, and generality are uncharacterized.

---

## Measurement Theory Lens — Measurement Validity

*Is the SAE decomposition a reliable metric?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Inconclusive.** The training appendix specifies the optimizer, the data volume and the epoch count and says nothing about seeds or repeated runs, and no figure caption reports variation across training runs. Figure 2's error bars are confidence intervals over the 150 scored features from a single dictionary, which measures scoring spread rather than training reliability. The follow-up supplies the missing test at one width and finds high agreement, so the picture depends on which evidence is admitted.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Partial.** All layers of Pythia-70M are covered, across five expansion ratios and two scoring regimes.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Inconclusive — the capping criterion.** The dictionary separates from random directions, but not from a random network when that control is run [Heap et al. 2025], and one SAEBench metric scores higher on a randomly initialized model than a trained one [Karvonen et al. 2025]. The origin's within-model controls and the later random-network controls disagree.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Unknown.** Can the metric distinguish between a genuine "deception" feature and a "formal language" feature that happens to co-occur with deception in the training data? The sensitivity to genuine semantic distinctions versus statistical co-occurrence is not characterized.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** The metric is a correlation, so zero has a meaning independent of the experiment: a description that predicts activations no better than chance. The paper does not hide what the scale delivers — the unselected first-five sample includes a negative score of −0.11 and a top score of 0.57. What is missing is any conversion of a score of 0.33 into a statement about how much of a feature's behavior is explained.

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Weak | Low cross-seed Jaccard for most features |
| M6 Invariance | Not tested | No cross-condition comparison |
| M2 Baseline separation | Partial | Strong features separated; boundary unclear |
| M5 Sensitivity | Unknown | Co-occurrence vs. semantics not distinguished |
| M4 Calibration | Not reported | Post-hoc thresholds |
| C3 Convergent validity | Weak | Max-activating tail only |

### Key Distinctions

- **Reliability vs validity:** Low cross-seed reliability (M1) places a ceiling on validity — if the metric does not produce the same result twice, the result cannot be valid regardless of how compelling any single run appears. For SAE features, the reliability ceiling is low for most of the dictionary.
- **Convergent vs discriminant validity:** SAE features lack both. Convergent: does a different decomposition method (NMF, ICA, probing) find the same features? Discriminant: do features that should be distinct (deception vs. sarcasm) actually have low overlap? Neither is systematically tested.
- **The metric creates the object:** Unlike probes or circuits (which measure pre-existing model properties), SAEs *construct* the feature set. The measurement and the measured object are not independent — a core measurement-theoretic concern.

### MTMM Matrix

| | SAE seed A (feature $f$) | SAE seed B (feature $f'$) | Probing (concept $c$) | Weight analysis (direction $d$) |
|---|---|---|---|---|
| **SAE seed A** | — | low-moderate Jaccard | ? | ? |
| **SAE seed B** | low-moderate | — | ? | ? |
| **Probing** | ? | ? | — | ? |
| **Weight analysis** | ? | ? | ? | — |

The only filled cell (cross-seed SAE comparison) shows low-moderate agreement for most features, with higher agreement for strong features. No cross-method comparisons exist — we do not know if SAE features, probing directions, and weight-space analyses converge on the same representational structure. Without this cross-method comparison, SAE features cannot be validated as model-intrinsic rather than method-specific.

---

## MI Lens — Interpretive Validity

*Are the feature labels warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Partial.** The formal object and the metric are declared exactly: rows of a learned matrix, and a correlation between simulated and actual activations. The word carrying the claim is less disciplined — across four sentences "feature" denotes a unit of the network to be reverse engineered, a row of the dictionary, and a real-world property that text corresponds to, and the last sentence uses two of those senses at once.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Weak.** The primary evidence for feature identity is behavioral (max-activating examples, steering). But the claim is representational — it asserts that the model *encodes* this information, not just that manipulating the direction changes behavior. Behavioral evidence (steering) underdetermines representational claims: a direction can produce deceptive outputs when steered without being "the deception representation."

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Not done.** For most features, alternative explanations are not considered. A "deception" feature might equally be a "formal language + negation" feature, a "long-sentence" feature, or a "training-data-artifact" feature. Without discriminant testing (C4), alternatives are not excluded.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Confirmed.** Both the origin and the follow-up declare their limits and quantify them [Leask et al. 2025].

**[V4 — Unlicensed labeling:](/mechanistic-validity/framework/criteria/interpretive/unlicensed-labeling) Partial.** "Monosemantic" projects semantics onto a sparsity constraint. The origin hedges consistently; downstream labels such as "deception" do not.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Pass | Representational level stated |
| V2 Level-evidence match | Weak | Behavioral evidence for representational claim |
| V3 Alternative level | Not done | No discriminant testing |
| V5 Scope declaration | Often missing | Labels exceed evidence scope |

### Key Distinctions

- **Description vs explanation:** SAE features are descriptive (they identify directions that correlate with concepts) but not explanatory (they do not specify the algorithm that produces or uses the representation). The label names the content but not the computation.
- **Component identity vs component role:** A feature's identity (its decoder direction) is precisely specified. Its role (how the model uses this direction during inference) is almost entirely uncharacterized. We know what the feature "looks like" but not what it "does."
- **Faithfulness vs understanding:** Even features with high steering faithfulness (Golden Gate Bridge) may not represent genuine understanding of the model's computation — the direction may be exploitable without being the model's actual representational strategy.

### Evidence Convergence Map

- **Implementational → Interpretation:** Weak. No weight-space evidence identifies features independently. The decoder vectors are products of the SAE training, not independent structural analysis.
- **Algorithmic → Interpretation:** Very weak. How features interact during inference — which features compose with which, what algorithm they jointly implement — is almost entirely uncharacterized.
- **Computational → Interpretation:** Moderate for strong features. Steering shows that the direction is functionally relevant (it can shift computation). But "functionally relevant when artificially amplified" is weaker than "used by the model during normal inference."

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Zeroing (ablation) | partial (strong feat.) | — | ∅ | ∅ | ∅ |
| Clamping (steering) | — | partial (strong feat.) | ∅ | ∅ | partial |
| Cross-seed comparison | — | — | partial | — | — |
| Max-activating examples | — | — | circular | — | — |
| Decoder projection | — | — | partial | — | — |

Most cells empty or structurally invalid (∅). The two interventional rows (zeroing, clamping) provide partial evidence for strong features only. The observational rows (max-activating, decoder projection) provide representational evidence that is either circular or partial. No algorithmic evidence exists for any feature.

### Causal Sufficiency Graph

- Input → feature activation: **dashed** (correlation observed via max-activating examples; causal direction not established)
- Feature activation → model behavior: **dashed** (demonstrated only under supraphysiological clamping; normal-regime causal contribution uncharacterized)
- Feature → downstream features: **absent** (feature interaction and composition is not mapped)
- Feature → output logits: **dashed** (decoder vector projects onto logits, but whether this pathway is causally active during normal inference is untested)

No solid edges. The entire causal graph for SAE features operates in the "suggestive but unconfirmed" regime. This is the fundamental interpretive gap: features are identified and labeled, but their causal role in the model's computation is inferred rather than demonstrated.

---
