---
title: "Case Study: SAE Features"
description: "Sparse autoencoder features (Bricken et al. 2023, Templeton et al. 2024) evaluated through all five validity lenses."
---

# Case Study: SAE Features

Sparse autoencoder features ([Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html), [Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)) are directions in activation space extracted by training an overcomplete dictionary. Each feature is given a label — "Golden Gate Bridge," "deception," "code syntax" — based on the inputs that maximally activate it. The claim is that these features are real computational units: [representational](/framework/modes/representational)-level entities that the model uses during inference.

This case study evaluates SAE features *as a class*. Individual strong features (those that replicate and steer) score higher; the bulk of the dictionary scores lower. The evaluations below reflect the typical case.

## Composite Verdict

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C2 Structural plausibility (partial) | C5 Convergent validity | Weak |
| Internal (Neuroscience) | I2 Sufficiency (strong features) | I3/I4/I5 Most criteria | Weak |
| External (Pharmacology) | E1 Intervention reach (partial) | E3/E5/E6 Most criteria | Weak |
| Measurement (Measurement Theory) | M3 Baseline separation (partial) | M1 Reliability | Weak |
| Interpretive (MI) | V1 Level declaration | V4/V5 Alternatives + scope | Weak |

**Overall verdict: Proposed to Causally suggestive.** SAE features as a class sit at the lowest verdict tiers. The strongest individual features (those that replicate across seeds, respond to steering, and have coherent decoder vectors) approach *Causally suggestive*. The bulk of any SAE dictionary remains at *Proposed* — the features have been identified and labeled, but the evidence for their reality as model-intrinsic computational units is thin across all five lenses.

This is not a claim that SAE features are wrong — many may be real. It is a claim that the evidence for their validity, measured against the same standards applied to circuits, has not been marshaled. The primary gaps are convergent validity (C5 — does a different method find the same features?), measurement reliability (M1 — does a different training run find the same features?), and alternative exclusion (V4 — is the label the right one?). These three gaps share a common theme: the features may be properties of the dictionary rather than properties of the model.

---

## Philosophy of Science Lens — Construct Validity

*Is "SAE feature $f_{42}$" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Unclear.** What observation would disconfirm the claim that feature $f_{42}$ represents "deception"? If the disconfirming condition is "the feature does not activate on deceptive text," this is circular — the feature was *defined* by its activations. A genuine falsifiability condition would be: "if steering along $f_{42}$ does not increase deceptive outputs, or if a different SAE trained with a different random seed produces a feature with $J < 0.3$ overlap." Most papers do not state such conditions.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Partial.** The decoder vector $W_{\text{dec}}[f]$ should project onto semantically coherent tokens through the unembedding matrix. Some features pass this check ("Golden Gate Bridge" projects onto bridge-related tokens). Many features lack this structural verification.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Not tested.** Features are evaluated on their *maximally activating examples* — a discovery-set evaluation. Specificity would require showing that a "deception" feature activates on deception and *does not* activate on closely related non-deception (sarcasm, fiction, hypotheticals). This discriminant testing is rarely performed.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Open question.** Does a feature correspond to one computational role, or is it a blend of multiple roles that co-occur in training data? Polysemantic features — those that activate on apparently unrelated concepts — fail this criterion. The extent of polysemanticity in typical SAE dictionaries is debated.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Weak.** SAE features are identified by one method. A different SAE with different hyperparameters or random seed may produce a different feature set. Cross-seed consistency is partially reported for strong features but not systematically measured at Jaccard level across the full dictionary.

| Criterion | Verdict | Key evidence |
|---|---|---|
| C1 Falsifiability | Unclear | No pre-registered disconfirming conditions |
| C2 Structural plausibility | Partial | Some decoder vectors project coherently |
| C3 Task specificity | Not tested | No discriminant evaluation |
| C4 Minimality | Open question | Polysemanticity unresolved |
| C5 Convergent validity | Weak | Single method, partial cross-seed |

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

**[I1 — Necessity:](/framework/criteria/internal/necessity) Sometimes.** Ablating (zeroing) strong features degrades behavior on their associated inputs. But "necessity" for an individual SAE feature is a weaker claim than circuit necessity — many features contribute small amounts, and removing one may be compensated by others. Necessity is established for a few strong features; for the bulk of the dictionary, it is untested.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Sometimes (via steering).** Clamping a feature to a high activation value can steer model outputs — the "Golden Gate Bridge" feature reliably produces bridge-related text. This is a form of sufficiency: the feature direction alone drives the behavior. But steering is blunt (high-magnitude clamping may go off-manifold), and many features do not produce coherent effects when steered.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Not tested.** Does ablating a "deception" feature selectively impair deception-related outputs without affecting other capabilities? This requires measuring collateral damage, which is rarely done for individual features.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Weak.** Cross-seed replication shows that strong features (high-frequency, high-magnitude) are relatively stable. Weaker features may not replicate. No systematic cross-checkpoint or cross-model consistency has been reported.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Not tested.** Steering typically uses a single method (activation addition at a fixed scale). Multi-method comparison (clamping at different layers, steering via different feature dictionaries) is not performed.

| Criterion | Verdict | Key evidence |
|---|---|---|
| I1 Necessity | Sometimes | Strong features: yes. Bulk: untested |
| I2 Sufficiency | Sometimes | Steering works for strong features |
| I3 Specificity | Not tested | No collateral damage measured |
| I4 Consistency | Weak | Strong features partially stable |
| I5 Confound control | Not tested | Single steering method |

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

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Partial.** Steering experiments show that clamping features can shift model behavior. The "Golden Gate Bridge" feature produces bridge-related responses across varied prompts. But the reach of most features (especially abstract or behavioral ones like "deception") is not well-characterized.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Partial.** Clamping at different multipliers (1x, 5x, 10x the typical activation magnitude) produces graded effects — stronger clamping produces more extreme outputs. But the dose-response is often nonlinear and poorly characterized. At high multipliers, outputs become incoherent rather than showing more of the feature.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Not tested.** Does steering along "deception" selectively increase deception without affecting fluency, factuality, or other behaviors? Off-target effects are rarely measured. The intervention may be producing general distributional shift rather than targeted behavioral change.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Variable.** Some features produce large, clear effects (Golden Gate Bridge). Others produce weak or incoherent effects. The distribution of effect magnitudes across the dictionary is not systematically reported.

**[E5 — Robustness:](/framework/criteria/external/robustness) Unknown.** Does the Golden Gate Bridge feature work equally well on questions, stories, code prompts, and multilingual inputs? Robustness across prompt distributions is not systematically tested.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** SAE features are model-specific by construction — each dictionary is trained on one model's activations. Whether "the same feature" exists across models requires a separate alignment step that is not standard.

| Criterion | Verdict | Key evidence |
|---|---|---|
| E1 Intervention reach | Partial | Works for some features, untested for most |
| E2 Graded response | Partial | Nonlinear, breaks at high magnitudes |
| E3 Selectivity | Not tested | Off-target effects unmeasured |
| E4 Effect magnitude | Variable | Some strong, most unknown |
| E5 Robustness | Unknown | No cross-distribution testing |
| E6 Cross-architecture | Not tested | Model-specific by construction |

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

*Is the SAE decomposition a reliable instrument?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Weak.** Different SAE training runs (different seeds, hyperparameters) produce different dictionaries. The Jaccard overlap between features identified by two independent SAEs is low for most of the dictionary. The instrument's test-retest reliability is poor.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Not tested.** Do SAE features show the same properties when the dictionary is trained on different data subsets? When applied to different layers? Measurement invariance across conditions is not reported.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Partial.** Strong features (high activation, clear semantic coherence) are clearly separated from noise. But the boundary between "real features" and "dictionary artifacts" is not well-defined. How many of the 16,384 features in a typical SAE are real?

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Unknown.** Can the instrument distinguish between a genuine "deception" feature and a "formal language" feature that happens to co-occur with deception in the training data? The sensitivity to genuine semantic distinctions versus statistical co-occurrence is not characterized.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.** What activation level constitutes "the feature is on"? Thresholds are typically chosen post-hoc. Without calibration, activation magnitudes are hard to interpret.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Weak.** Max-activating examples capture the top-activating tail. They do not capture: the feature's behavior at moderate activations, its interactions with other features, its role in downstream computation, or its boundary cases (what it *almost* fires on but doesn't).

| Criterion | Verdict | Key evidence |
|---|---|---|
| M1 Reliability | Weak | Low cross-seed Jaccard for most features |
| M2 Invariance | Not tested | No cross-condition comparison |
| M3 Baseline separation | Partial | Strong features separated; boundary unclear |
| M4 Sensitivity | Unknown | Co-occurrence vs. semantics not distinguished |
| M5 Calibration | Not reported | Post-hoc thresholds |
| M6 Construct coverage | Weak | Max-activating tail only |

### Key Distinctions

- **Reliability vs validity:** Low cross-seed reliability (M1) places a ceiling on validity — if the instrument does not produce the same result twice, the result cannot be valid regardless of how compelling any single run appears. For SAE features, the reliability ceiling is low for most of the dictionary.
- **Convergent vs discriminant validity:** SAE features lack both. Convergent: does a different decomposition method (NMF, ICA, probing) find the same features? Discriminant: do features that should be distinct (deception vs. sarcasm) actually have low overlap? Neither is systematically tested.
- **The instrument creates the object:** Unlike probes or circuits (which measure pre-existing model properties), SAEs *construct* the feature set. The measurement and the measured object are not independent — a core measurement-theoretic concern.

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

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [representational](/framework/modes/representational) level — features are directions in activation space that encode information about inputs.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Weak.** The primary evidence for feature identity is behavioral (max-activating examples, steering). But the claim is representational — it asserts that the model *encodes* this information, not just that manipulating the direction changes behavior. Behavioral evidence (steering) underdetermines representational claims: a direction can produce deceptive outputs when steered without being "the deception representation."

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Variable.** "Golden Gate Bridge" is narratively coherent — the feature fires on bridge-related content and steers toward bridge-related output. "Deception" is less coherent — what exactly is the model encoding? Intent to deceive? Surface patterns associated with deceptive text? The narrative coherence varies by feature.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Not done.** For most features, alternative explanations are not considered. A "deception" feature might equally be a "formal language + negation" feature, a "long-sentence" feature, or a "training-data-artifact" feature. Without discriminant testing (C3), alternatives are not excluded.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Often missing.** Feature labels like "deception" imply a broad, abstract semantic concept. The evidence (max-activating examples from one model, one layer) supports only a narrow scope — "this direction in this layer activates on these inputs." The label exceeds the evidence.

| Criterion | Verdict | Key evidence |
|---|---|---|
| V1 Level declaration | Pass | Representational level stated |
| V2 Level-evidence match | Weak | Behavioral evidence for representational claim |
| V3 Narrative coherence | Variable | Strong for concrete, weak for abstract features |
| V4 Alternative exclusion | Not done | No discriminant testing |
| V5 Scope honesty | Often missing | Labels exceed evidence scope |

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
