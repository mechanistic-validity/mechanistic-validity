---
title: "Case Study: Othello World Model"
description: "The Othello-GPT world model (Li et al. 2023) evaluated through the five core lenses."
---

# Case Study: Othello World Model

[Li et al. (2023)](https://arxiv.org/abs/2210.13382) train a GPT on Othello game transcripts and claim the model develops an internal **world model** — a representation of the board state that tracks which squares are occupied by black, white, or empty, decoded by nonlinear probes across all 64 tiles where linear probes never dip below 20% error. The claim is that the model does not merely memorize move sequences but represents the underlying game state, and that this representation is causally used during move prediction.

This is a [representational](/mechanistic-validity/framework/modes/representational)-level claim with algorithmic implications: it asserts not just that board state information is present in activations but that the model constructs and uses a world model for prediction.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Causally Suggestive. **Capped by:** E1 (intervention reach).


| Lens | Strongest criterion | Weakest criterion | Overall |
|---|---|---|---|
| Construct (Phil. Sci.) | C1 Falsifiability, C3 Convergent validity (both confirmed) | C6 Complementation validity (untested) | Partial |
| Internal (Neuroscience) | I1 Necessity (partial) | I7 Confound control | Weak–Partial |
| External (Pharmacology) | E4 Cross-model generalization, E2 Prompt generalization, E6 Novel prediction | E1 Intervention reach (the capping criterion) | Partial |
| Measurement (Measurement Theory) | M6 Invariance (partial) | M4/M5 Sensitivity + Calibration | Weak |
| Interpretive (MI) | V2 Level-evidence match | V3 Alternative level | Partial |

**Overall verdict: Causally suggestive.** The Othello world model has genuine evidence for board-state representation — probe results are real and causal interventions work — but the interpretive framing ("world model") exceeds what the evidence establishes. The claim is capped by intervention reach (E1): one intervention operator, reported under three outcome metrics. Above that it is blocked by double dissociation (I6), and the label question is carried by unlicensed labeling (V4), since "world model" is defined and only a state summary is ever measured.

This case study illustrates a pattern worth naming: **interpretive inflation** — a mechanistic finding (linear probe recovers board state) is described using a term (world model) that implies more structure, compositionality, and causal role than the evidence supports. The finding is real; the label is aspirational. The framework's contribution here is not to dismiss the finding but to name precisely where the label exceeds the evidence and what additional tests would close the gap.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Linear probing (board state recovery) | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| Causal intervention / activation patching (board state) | [A02 Counterfactual DAS](/mechanistic-validity/framework/metrics/#a02) | Causal |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "world model" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Pass.** The claim generates testable predictions: a linear probe trained on residual stream activations should recover the board state with high accuracy. If board state is not linearly decodable from the residual stream, the world model claim is disconfirmed. The authors also test a stronger prediction: intervening on the representation (patching board-state information) should change the model's move predictions in the way a world model would predict.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Partial.** The representation is linearly decodable — a probe recovers board state at high accuracy. But linear decodability does not establish structural plausibility in the weight-space sense. A world model should correspond to some identifiable structure in the model's parameters (attention patterns that track spatial relationships, MLP neurons that compute legal moves). The probe result is an activation-space finding, not a weight-space one.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) **Partial.** Two constructs the board representation might be confused with are excluded by measurement. Removing a quarter of the game tree leaves the error rate at 0.02%, which separates the representation from memorized transcripts, and the randomized network fixes how much a probe recovers from an untrained network at the same capacity. The construct never separated from is the weaker reading of the paper's own claim: a decodable state summary that the next-move computation reads.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Not applicable.** One activation vector holds all 64 tiles, so there are no components to prune.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) **Confirmed.** Three readouts point the same way: a nonlinear probe decodes the board, editing the decoded state moves the predictions, and error, F1 and KL divergence agree on the size of that effect. The agreement is weaker than it looks, because the intervention is gradient descent on the same probe's class score, so the observational and causal instruments share their definition of the representation.

### Key Distinctions

- **Operationalism vs realism:** "World model" is a realist theoretical term applied to an operational finding (linear decodability of board state). The operational evidence supports only "linear board-state representation." The realist label implies structured spatial reasoning, counterfactual prediction, and compositional game understanding — none of which are directly tested. This is the clearest case of label-reality mismatch among the case studies.
- **Confirmation vs corroboration:** The work uses two methods (probing and patching), providing partial corroboration — stronger than pure confirmation. However, both methods operate in the same activation space and share the assumption that the linear probe direction is meaningful. A genuinely independent corroboration (e.g., weight-space identification of board-tracking circuits, or training-dynamic predictions) would substantially strengthen the claim.
- **Observable vs theoretical:** Board-state decodability is observable (the probe works). "World model" is theoretical (it implies structured, compositional representation used for planning). The gap between the observable and the theoretical label is the central interpretive tension.

### Nomological Network

The Othello world model connects to:
- **Linear decodability** — probe recovers board state from residual stream (representational, confirmed)
- **Causal intervention** — patching board-state information shifts predictions (causal, confirmed)
- **Spatial structure** — Nanda's follow-up recovers a linear representation under a mine/theirs basis rather than black/white, which strengthens the linearity result and shows the probe target was never varied at origin (scored under I5 and M3)
- **Legal move computation** — does the model use board state to determine legal moves, or are legal moves computed separately? (untested)
- **Training dynamics** — does board-state representation emerge at a specific training phase? (untested)
- **Weight-space implementation** — which attention heads or MLP layers construct the representation? (partially explored)
- **Counterfactual reasoning** — does the model use the world model to reason about hypothetical board states? (untested)

Two nodes confirmed, one partially explored, four untested. A thin network for such a strong claim — "world model" implies rich compositional structure, but most of the network nodes that would confirm that richness remain unconnected.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish that the model implements a world model?*

### Criteria

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Partial.** The causal evidence is indirect. Li et al. show that intervening on the board-state representation (patching activations to reflect a different board state) changes predictions. This establishes that the representation is causally relevant. But it does not establish necessity in the ablation sense — removing the representation was not tested (it is unclear how to ablate a distributed linear representation).

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Partial.** Patching board-state information into the residual stream changes predictions in the expected direction. This is a form of sufficiency — the representation contains enough information to shift behavior. But sufficiency in isolation (can the board-state representation alone drive correct move prediction?) is not tested.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) Partial.** The probe specifically decodes board state, not other features (move legality, piece count, game phase). This establishes that the representation is specific to board state. But whether intervening on the board-state representation *only* changes board-state-relevant predictions (not general prediction quality) is not systematically measured.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Three confounds are handled by design. Training-set overlap is removed by truncating the game tree, and by the fact that the network never sees a board state as input. Distributional support is removed by the unnatural benchmark. Probe capacity is held fixed against a randomized network, so decoding power is subtracted from decoded content. The confound left uncontrolled is the probe target itself, which fixes what counts as the board state before any measurement begins.

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

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Untested — the capping criterion.** One intervention operator is used, reported under three outcome metrics. Reproducing the result under a second intervention family is what would lift the claim to Mechanistically Supported.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) **Partial.** Two intervention-strength knobs are swept and the response is graded in the shape a mechanism predicts: too many layers reaches back into layers whose representations are unreliable, too few leaves the network no computation to propagate the edit, and the error rises on both sides. The knob never swept is the size of the edit itself against a graded behavioral outcome, so the dose-response curve the criterion asks for is absent.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) **Confirmed.** The causal result is measured on a thousand cases per subset rather than on selected examples, and the unnatural subset is off-distribution by construction, since those boards cannot arise from legal play. Probing runs on two training distributions that differ in kind — strategic human play against uniformly sampled legal moves. Generalization across inputs is demonstrated at scale and in the hardest available direction.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Confirmed on post-origin evidence.** Absent at origin, which trains one architecture twice; seven architectures were tested later [Yuan and Søgaard, 2025].

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

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Confirmed.** Probe accuracies are re-run 100 times and the deviations are reported in Tables 4 and 5 of the origin paper.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) **Confirmed.** The result is reported over every axis the design makes available rather than at one favorable setting: eight layers, two training distributions, a randomized control network, game progression, probe capacity, natural and unnatural boards, and three outcome metrics. The layerwise profile is interpretable and matches what probing studies of natural language report.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Confirmed.** Three floors are reported: an untrained network, a constant guess, and null intervention baselines of 2.68 and 2.59 against measured errors of 0.12 and 0.06.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Unknown.** Can the probe distinguish between a genuine world model and a model that uses heuristic shortcut features (e.g., "this square was recently played, so it's probably occupied")? The probe's sensitivity to genuine spatial reasoning versus statistical shortcuts is unclear.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) **Partial.** The causal measure is calibrated: an error is a count of false positives and false negatives against the post-intervention legal-move set, it has a null floor, and three metrics that weight errors differently agree. The observational measure is not — footnote 5 concedes that occupancy alone is a linear function of the input, so part of the 3-way probe's accuracy is available without any board representation, and no analysis says how much.

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) **Partial.** No sentence declares what level of description the claim is pitched at. The level has to be read off the vocabulary, which moves: the introduction and the contribution list say world model, the experimental sections say representation of the board state, and the conclusion puts the two together by glossing the board as the Othello world.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) **Partial.** Against the authors' own definition — an understandable model of the process producing the sequences — the evidence covers the state and stops short of the process. What is measured is that the board state is decodable and that editing it moves the predictions. The rules that generate the sequences are touched once, in Appendix E, which reports that single-tile attribution recovers the AND inside one line and fails on the OR across lines.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** One alternative is stated as a hypothesis and killed with an experiment: memorized transcripts cannot explain a 0.02% error rate on a game tree a quarter of which was withheld. The alternative at the level immediately below the claim is never stated. §4 poses the open question as decodable versus causally decodable, and once causality is established the paper treats the world-model reading as settled, with no intermediate reading — a causally-used state summary — considered.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Confirmed.** The synthetic scope is declared in the title, in §1.1 and in the conclusion.

**[V4 — Unlicensed labeling:](/mechanistic-validity/framework/criteria/interpretive/unlicensed-labeling) Partial.** "World model" is defined in the paper, and what is measured is a decodable, causally-used state summary. Whether that constitutes a world model in the computational-theory-of-mind sense — a structured representation used for planning and counterfactual reasoning — is not established.

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
