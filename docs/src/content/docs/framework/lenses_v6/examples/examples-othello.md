---
title: "Case Study: Othello World Model"
description: "The Othello-GPT world model (Li et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Othello World Model

[Li et al. (2023)](https://arxiv.org/abs/2210.13382) train a GPT on Othello game transcripts and claim the model develops an internal **world model** — a linear representation of the board state that tracks which squares are occupied by black, white, or empty. The claim is that the model does not merely memorize move sequences but represents the underlying game state, and that this representation is causally used during move prediction.

This is a [representational](/framework/modes_v3/representational)-level claim with algorithmic implications: it asserts not just that board state information is present in activations but that the model constructs and uses a world model for prediction.

## Composite Verdict

| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C1 Falsifiability | C4 Minimality | Partial |
| Internal (Neuroscience) | I1 Necessity (partial) | I5 Confound control | Weak–Partial |
| External (Pharmacology) | E1 Intervention reach | E2/E3/E6 Most criteria | Partial |
| Measurement (Measurement Theory) | M2 Invariance (partial) | M4/M5 Sensitivity + Calibration | Weak |
| Interpretive (MI) | V2 Level-evidence match | V4 Alternative exclusion | Partial |

**Overall verdict: Causally suggestive.** The Othello world model has genuine evidence for board-state representation (probe results are real and causal interventions work), but the interpretive framing ("world model") exceeds what the evidence establishes. The primary gaps are confound control (I5 — is the probe direction causally real or an artifact?), alternative exclusion (V4 — heuristics vs. genuine spatial reasoning), and scope honesty (V5 — "world model" carries implications beyond "linear board-state decodability").

This case study illustrates a pattern worth naming: **interpretive inflation** — a mechanistic finding (linear probe recovers board state) is described using a term (world model) that implies more structure, compositionality, and causal role than the evidence supports. The finding is real; the label is aspirational. The framework's contribution here is not to dismiss the finding but to name precisely where the label exceeds the evidence and what additional tests would close the gap.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Linear probing (board state recovery) | [E02 Linear Probe](/framework/metrics/representational/e02-linear-probe) | Representational |
| Causal intervention / activation patching (board state) | [A02 Counterfactual DAS](/framework/metrics/causal/a02-counterfactual-das) | Causal |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "world model" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Pass.** The claim generates testable predictions: a linear probe trained on residual stream activations should recover the board state with high accuracy. If board state is not linearly decodable from the residual stream, the world model claim is disconfirmed. The authors also test a stronger prediction: intervening on the representation (patching board-state information) should change the model's move predictions in the way a world model would predict.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Partial.** The representation is linearly decodable — a probe recovers board state at high accuracy. But linear decodability does not establish structural plausibility in the weight-space sense. A world model should correspond to some identifiable structure in the model's parameters (attention patterns that track spatial relationships, MLP neurons that compute legal moves). The probe result is an activation-space finding, not a weight-space one.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Pass.** The Othello model is trained on a single task, so cross-task evaluation in the traditional sense does not apply. However, the world model claim is specific — it claims board-state representation, not generic sequence memorization. The probe results show that the model tracks board state rather than superficial sequence statistics, which is a form of discriminant validity.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Not tested.** Is the full residual stream necessary to represent the board state, or could a smaller subspace suffice? The probe identifies a linear subspace, but whether this subspace is minimal (no further compression possible) is not systematically tested.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Partial.** Two methods are used: linear probing (a representational method) and causal intervention (patching board-state information). These have partially overlapping assumptions — both rely on the activation space being the right level of analysis. A weight-space method (identifying which attention heads or MLP layers implement the world model) would provide stronger convergence.

### Key Distinctions

- **Operationalism vs realism:** "World model" is a realist theoretical term applied to an operational finding (linear decodability of board state). The operational evidence supports only "linear board-state representation." The realist label implies structured spatial reasoning, counterfactual prediction, and compositional game understanding — none of which are directly tested. This is the clearest case of label-reality mismatch among the case studies.
- **Confirmation vs corroboration:** The work uses two methods (probing and patching), providing partial corroboration — stronger than pure confirmation. However, both methods operate in the same activation space and share the assumption that the linear probe direction is meaningful. A genuinely independent corroboration (e.g., weight-space identification of board-tracking circuits, or training-dynamic predictions) would substantially strengthen the claim.
- **Observable vs theoretical:** Board-state decodability is observable (the probe works). "World model" is theoretical (it implies structured, compositional representation used for planning). The gap between the observable and the theoretical label is the central interpretive tension.

### Nomological Network

The Othello world model connects to:
- **Linear decodability** — probe recovers board state from residual stream (representational, confirmed)
- **Causal intervention** — patching board-state information shifts predictions (causal, confirmed)
- **Spatial structure** — does the representation encode spatial adjacency relationships? (partially explored by Nanda)
- **Legal move computation** — does the model use board state to determine legal moves, or are legal moves computed separately? (untested)
- **Training dynamics** — does board-state representation emerge at a specific training phase? (untested)
- **Weight-space implementation** — which attention heads or MLP layers construct the representation? (partially explored)
- **Counterfactual reasoning** — does the model use the world model to reason about hypothetical board states? (untested)

Two nodes confirmed, one partially explored, four untested. A thin network for such a strong claim — "world model" implies rich compositional structure, but most of the network nodes that would confirm that richness remain unconnected.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that the model implements a world model?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Partial.** The causal evidence is indirect. Li et al. show that intervening on the board-state representation (patching activations to reflect a different board state) changes predictions. This establishes that the representation is causally relevant. But it does not establish necessity in the ablation sense — removing the representation was not tested (it is unclear how to ablate a distributed linear representation).

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Partial.** Patching board-state information into the residual stream changes predictions in the expected direction. This is a form of sufficiency — the representation contains enough information to shift behavior. But sufficiency in isolation (can the board-state representation alone drive correct move prediction?) is not tested.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Partial.** The probe specifically decodes board state, not other features (move legality, piece count, game phase). This establishes that the representation is specific to board state. But whether intervening on the board-state representation *only* changes board-state-relevant predictions (not general prediction quality) is not systematically measured.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Partial.** The representation is tested across game positions (not just one board state). The probe accuracy is high across diverse positions. But cross-model consistency (does a differently-trained Othello-GPT develop the same representation?) is not reported.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Weak.** The patching intervention assumes that the linear probe direction is the causally relevant direction. But patching along a probe direction may work because it moves activations in a way that incidentally helps prediction, not because the direction is the model's internal representation. Neel Nanda's follow-up work raises this concern: the "world model" may be an artifact of the probe rather than a property of the model.

### Key Distinctions

- **Single vs double dissociation:** Only single dissociation is shown: the probe decodes board state from the trained model. The necessary control — showing that task-irrelevant information (e.g., move parity, sequence position statistics) is NOT decodable at the same accuracy from the same layer — is incompletely addressed. Without this, high probe accuracy may reflect general representational richness rather than specific world-model structure.
- **Lesion vs stimulation:** The paper includes both correlational observation (probing, analogous to recording) and stimulation (patching board-state information). This is a genuine strength — the combination of passive measurement and active intervention provides stronger evidence than either alone. However, the lesion complement (removing the board-state subspace entirely) is absent.

### Dissociation Matrix

|  | Move prediction (legal) | Move prediction (strategic) | Sequence statistics | Board-state probe accuracy |
|---|---|---|---|---|
| Patch board-state info | **Shifts (confirmed)** | ? | ? | **Changes (by design)** |
| Ablate board-state subspace | ? | ? | ? | ? |
| Patch sequence statistics | ? | ? | ? | ? |
| Train on shuffled games | ? | ? | ? | ? |

Two cells filled (both in the patching row). The ablation row and the control conditions (sequence statistics, shuffled training) are empty. Without the ablation complement, we cannot confirm that the board-state representation is necessary (not just sufficient to shift predictions). The shuffled-training control would test whether an equivalent probe accuracy emerges even without genuine board-state structure in the training data.

---

## Pharmacology Lens — External Validity

*Does intervening on the world model produce expected effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Pass.** Patching board-state information changes model predictions across many game positions. The intervention has broad reach within the task.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Not tested.** Does patching a "stronger" board-state signal (further from the actual state) produce a proportionally larger prediction change? A parametric dose-response is not reported.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Not tested.** Does board-state intervention only change move predictions, or does it also affect the model's confidence, attention patterns, or other outputs? Off-target effects are not measured.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Moderate.** The linear probe achieves high accuracy (~90%+ on board-state recovery), and patching produces measurable prediction changes. The effect is real but the magnitude of the causal intervention (how much of prediction quality it explains) is not precisely quantified.

**[E5 — Robustness:](/framework/criteria/external/robustness) Partial.** Works across diverse game positions. Not tested across different game phases (opening vs. endgame) or board sizes.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Not tested.** The result is specific to one model architecture trained on Othello. Whether similar world models emerge in different architectures (e.g., Mamba, state-space models) is unknown.

### Key Distinctions

- **Affinity vs efficacy:** The probe demonstrates affinity (board-state information is present and decodable), but efficacy is weaker — patching shifts predictions in the expected direction without achieving full control. The gap between "information is there" and "information causally drives output" is precisely the affinity-efficacy distinction, and this work sits closer to the affinity end.
- **The system compensates:** The network uses distributed representations, making compensation likely but untested. If you ablate the board-state subspace, do other representational dimensions partially recover the function? This question is never asked, leaving open whether the "world model" is a fragile single pathway or a robust distributed computation with redundancy.
- **The metric is part of the finding:** Probe accuracy is the primary metric and simultaneously the primary evidence. The finding IS high probe accuracy — there is no independent behavioral measure that the "world model" improves. This circularity means the result cannot distinguish a genuinely used representation from a linearly-decodable epiphenomenon.

### Dose-Response Curve

The Othello world model's dose-response is minimally characterized:
- **Patching at full strength**: predictions shift in the expected direction (confirmed)
- **Probe accuracy by layer**: representation builds up across layers (a form of spatial dose-response)

What's missing:
- **No parametric patching sweep** — varying the magnitude of the board-state patch from 0 to 1 to see if prediction changes scale linearly
- **No off-target measurement** — does patching board state also change non-board-state-relevant outputs?
- **No threshold detection** — at what patch magnitude does the prediction begin to change?
- **No saturation test** — does patching beyond the "correct" board state produce paradoxical effects?

The curve has one confirmed point (full-strength patching works) and a layer-wise buildup trajectory. No interior of the dose-response is characterized. We know the intervention works but cannot characterize its sensitivity, linearity, or selectivity boundary.

---

## Measurement Theory Lens — Measurement Validity

*Is the linear probe a reliable metric?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Not reported.** Are the probe results stable across different probe training runs? Different probe architectures? No test-retest or bootstrap stability is reported.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Partial.** The probe works across game positions, providing some invariance. But invariance across layers (does the representation look the same at every layer?) is partially explored — the representation builds up across layers, which is informative but complicates invariance claims.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Partial.** The probe accuracy is high, but what is the baseline? A probe trained on random activations (not from an Othello-trained model) should fail. This baseline is partially addressed — probes on untrained models recover little board state. But the critical baseline is: what accuracy does a probe achieve on a model that memorizes sequences without a world model? This is harder to construct and not reported.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Unknown.** Can the probe distinguish between a genuine world model and a model that uses heuristic shortcut features (e.g., "this square was recently played, so it's probably occupied")? The probe's sensitivity to genuine spatial reasoning versus statistical shortcuts is unclear.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Not reported.** What does 90% probe accuracy mean? Is it "the model has a strong world model" or "90% of board state is linearly decodable, which is expected even without a world model"? Without calibration against models of known capability, the number is hard to interpret.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Partial.** The probe measures one aspect of the world model (static board state). A complete world model would also track legal moves, strategic evaluation, and game dynamics. Only board-state recovery is tested.

### Key Distinctions

- **Reliability vs validity:** Probe accuracy on the trained model is high (~90%+) vs. low on untrained models, establishing a baseline separation that partially addresses reliability. But validity — does high probe accuracy actually indicate a "world model" rather than exploitable statistical regularities? — is the deeper unresolved question. The measurement is reliable but its interpretation is underdetermined.
- **True score vs observed score:** The probe gives an observed score (accuracy on board-state recovery). The true score (degree to which the model genuinely represents and uses a world model) is unknown. The gap between observed and true score is the fundamental measurement problem — the metric measures something, but what it measures may not be the construct of interest.

### MTMM Matrix

| | Linear probe (board state) | Causal patching (board state) | Linear probe (move legality) | Causal patching (move legality) |
|---|---|---|---|---|
| **Linear probe (board state)** | — | Moderate (partially converge) | ? | ? |
| **Causal patching (board state)** | Moderate | — | ? | ? |
| **Linear probe (move legality)** | ? | ? | — | ? |
| **Causal patching (move legality)** | ? | ? | ? | — |

One convergent cell partially filled (probing and patching partially agree on board-state representation — high probe accuracy correlates with successful patching). No discriminant cells filled — we do not know whether the methods agree *more* about board state than about other decodable features (move legality, piece count). Without discriminant comparison, the convergent evidence could reflect that both methods pick up on general representational richness rather than specific world-model structure.

---

## MI Lens — Interpretive Validity

*Is "world model" the right interpretation?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [representational](/framework/modes_v3/representational) level — it asserts that the model encodes board state as a linear representation in activation space.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Pass.** Representational evidence (linear probing) supports a representational claim. The causal intervention adds algorithmic-level evidence. The evidence matches or exceeds the claim level.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Moderate.** "The model builds a world model and uses it for prediction" is a coherent story. But "world model" is a strong term that implies a structured, compositional representation of game state. The evidence shows that board state is linearly decodable — this is consistent with a world model but also consistent with a set of independent features that happen to correlate with board state. The narrative slightly overstates the structural implications.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Weak.** The key alternative is that the model uses heuristic features (recency of play, local board patterns) that happen to correlate with board state, rather than constructing a genuine spatial representation. This alternative would also produce high probe accuracy and partially successful patching. The authors do not fully exclude it. Nanda's replication work suggests the truth may be between the two — some genuine board-state tracking, but less than "world model" implies.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Partial.** "World model" implies more than what is demonstrated. The evidence supports "linear board-state representation." Whether this constitutes a "world model" in the computational theory of mind sense — a structured representation used for planning and counterfactual reasoning — is not established. The label carries philosophical weight that the evidence does not fully bear.

### Key Distinctions

- **Description vs explanation:** "Linear board-state representation" is descriptive (the probe works). "World model" is explanatory (it implies the model constructs and reasons over a structured game state). The evidence supports the description; the explanation is aspirational.
- **Faithfulness vs understanding:** The probe is faithful to the activation data (it correctly decodes board state). But understanding (does the model USE this representation as a world model for planning?) is not established. Faithful measurement with uncertain mechanistic interpretation.
- **Activation evidence vs weight evidence:** All evidence is activation-based (probing, patching). Weight-space evidence (which parameters implement the world model, how board-state information flows through the architecture) would be needed to confirm implementation rather than mere presence.

### Evidence Convergence Map

- **Implementational → Interpretation:** Weak. No weight-space analysis identifies which components construct the board-state representation. The claim is about what is represented, without specifying how it is implemented.
- **Representational → Interpretation:** Moderate. The probe directly addresses the representational claim (board state is encoded). But "encoded" is weaker than "world model" — the convergence supports the weaker claim better.
- **Computational → Interpretation:** Weak. We know the model predicts legal moves. Whether it does so VIA the board-state representation (supporting "world model") or via other features (supporting "heuristic shortcuts that correlate with board state") is unresolved.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Linear probing | — | — | ✓ | ∅ | ∅ |
| Board-state patching | — | Partial | ✓ | ∅ | ∅ |
| Board-state ablation | — | — | — | — | — |
| Attention analysis | — | — | — | — | — |
| Training dynamics | — | — | — | — | — |

Cells cluster in the representational column — the evidence establishes that board state is represented but not how it is algorithmically used. The necessity row is empty (no ablation of the representation), and the algorithmic/computational columns are empty (no evidence for how the representation drives prediction). The "world model" interpretation requires algorithmic-column evidence that does not exist.

### Causal Sufficiency Graph

- Game transcript → residual stream activations: **solid** (the model processes the transcript)
- Residual stream → linear board-state decodability: **solid** (probe succeeds)
- Board-state patch → prediction shift: **solid** (patching works)
- Board-state representation → move selection algorithm: **dashed** (inferred, not causally isolated)
- Heuristic features → move prediction: **dashed** (alternative pathway, not excluded)
- Board-state representation → counterfactual reasoning: **absent** (no evidence)

The solid edges confirm that board-state information exists and is causally relevant. The dashed edges represent the interpretive gap: we cannot determine whether the board-state representation is the primary algorithmic pathway for prediction (supporting "world model") or a correlated byproduct of features that drive prediction through other pathways (supporting "heuristic features"). The absent edge (counterfactual reasoning) represents the strongest implication of "world model" that has no supporting evidence.

---
