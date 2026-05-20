---
title: "Case Study: Grokking / Modular Addition"
description: "The modular addition circuit with Fourier features (Nanda et al. 2023) evaluated through all five validity lenses."
---

# Case Study: Grokking / Modular Addition

[Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) analyze a small transformer trained on **modular addition** ($a + b \mod p$) that undergoes "grokking" — sudden generalization long after memorizing the training set. They claim the model learns a **Fourier-based algorithm**: inputs are embedded into Fourier components (sinusoidal representations of position mod $p$), attention computes trigonometric identities to combine them, and the output reads off the result from the Fourier representation.

This is the strongest structural evidence in published MI — the weight matrices are fully reverse-engineered and the algorithm is mathematically specified. The catch: it is a toy model (1-layer, 113 parameters of interest, mod-113 arithmetic). The construct validity question is whether this tells us anything about real models.

## Composite Verdict

| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1/C2/C5 (all strong) | — | Validated |
| Internal | I2/I5 (sufficiency + confound) | — | Validated |
| External | E1–E5 (all pass) | E6 Cross-architecture | Strong (within scope) |
| Measurement | All criteria pass | — | Validated |
| Interpretive | V4 Alternative exclusion | — | Validated |

**Overall verdict: Validated (within scope).** The modular addition circuit reaches the highest verdict tier — *Validated* — within its toy-model scope. Every criterion across all five lenses is satisfied. This makes it the gold standard for what a complete mechanistic explanation looks like.

The limitation is scope: E6 (cross-architecture) is the only weak point, and it is a fundamental one. A toy model with perfect internal validity but unknown external validity establishes a *proof of concept* rather than a *general finding*. The framework's contribution here is to name precisely what is achieved (complete explanation of one model) and what is not (evidence that real models work this way). The grokking result is the ceiling of MI — it shows what "fully understood" looks like. The gap between this and any real-model circuit is the gap the field is working to close.

---

## Philosophy of Science Lens — Construct Validity

*Is "the Fourier algorithm" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/framework/criteria/construct/falsifiability) Strong pass.** The claim specifies the exact algorithm: the model computes $\cos(2\pi k(a+b)/p)$ via trigonometric identities applied in the attention layer. This generates precise quantitative predictions about every weight matrix entry. Any deviation from the predicted Fourier structure would disconfirm the claim.

**[C2 — Structural plausibility:](/framework/criteria/construct/structural-plausibility) Strong pass.** This is the paper's primary achievement. The embedding matrix entries are verified to approximate $\cos(2\pi k a/p)$ and $\sin(2\pi k a/p)$ for specific frequencies $k$. The attention pattern implements the trigonometric addition formula. The unembedding reads off the result. Every weight matrix is accounted for — not just "consistent with" but "mathematically predicted by" the Fourier algorithm.

**[C3 — Task specificity:](/framework/criteria/construct/task-specificity) Pass (single task).** The model is trained on one task. Specificity is trivially satisfied — there is no off-task to test.

**[C4 — Minimality:](/framework/criteria/construct/minimality) Pass.** The circuit uses the full model (it is a 1-layer transformer). But every component is accounted for — no redundant parameters. The circuit is minimal in the sense that removing any Fourier frequency degrades performance on the corresponding input pairs.

**[C5 — Convergent validity:](/framework/criteria/construct/convergent-validity) Strong pass.** The Fourier structure is identified through: (1) weight-space analysis (Fourier decomposition of $W_E$), (2) activation-space analysis (probing for Fourier components), (3) mechanistic prediction (computing exact predicted outputs from the algorithm and comparing to actual outputs), and (4) training dynamics analysis (watching Fourier components emerge during grokking). Four independent lines of evidence converge.

### Key Distinctions

- **Confirmation vs corroboration:** The training dynamics provide genuine corroboration: watching Fourier components emerge during grokking was not predicted by the static weight analysis but independently confirms the same mechanistic story. This temporal dimension elevates the evidence beyond mere confirmation.
- **Operationalism vs realism:** The construct is fully operationalized — "Fourier features" refers to specific measurable weight-matrix entries, not an abstract theoretical posit. The operational definition exhausts the phenomenon, dissolving the question of whether this is "real understanding."
- **Observable vs theoretical:** There is no gap between the observable and theoretical here. The claimed entities (Fourier components in weight matrices) are directly observable in the parameters — no inference chain is required. This eliminates the underdetermination problem.

### Nomological Network

The Fourier algorithm construct connects to:
- **Weight structure** — embedding entries approximate $\cos(2\pi k a/p)$ and $\sin(2\pi k a/p)$ (structural, confirmed)
- **Attention mechanism** — implements trigonometric addition formula (structural, confirmed)
- **Output prediction** — algorithm reproduces model outputs to numerical precision (behavioral, confirmed)
- **Training dynamics** — Fourier components emerge during grokking phase transition (temporal, confirmed)
- **Per-frequency ablation** — removing a frequency degrades specific input pairs (causal, confirmed)
- **Cross-seed replication** — same algorithm type emerges across random seeds (consistency, confirmed)
- **Cross-architecture transfer** — does the algorithm appear in multi-layer or larger models? (untested)

Six nodes confirmed, one unconnected. The thickest nomological network of any MI result — every internal prediction is verified, with only the external generalization edge remaining open.

---

## Neuroscience Lens — Internal Validity

*Does the evidence establish implementation?*

### Criteria

**[I1 — Necessity:](/framework/criteria/internal/necessity) Pass.** Ablating specific Fourier frequencies (zeroing the corresponding components in the embedding) degrades performance on input pairs involving those frequencies. The ablation is at the *feature* level rather than the component level, which is more precise.

**[I2 — Sufficiency:](/framework/criteria/internal/sufficiency) Pass.** The Fourier algorithm, when executed manually on the model's weights, reproduces the model's outputs to high precision. This is the strongest possible sufficiency: the algorithm *is* the model, not just a description of it.

**[I3 — Specificity:](/framework/criteria/internal/specificity) Pass (trivial).** Single-task model — no off-task to test.

**[I4 — Consistency:](/framework/criteria/internal/consistency) Pass.** The Fourier structure emerges consistently across different random seeds (the specific frequencies chosen may vary, but the algorithm class is the same). The grokking transition reliably produces the same kind of structure.

**[I5 — Confound control:](/framework/criteria/internal/confound-control) Pass.** The mechanistic account is so complete that confounds are ruled out — the algorithm predicts outputs from weights alone, with no unexplained variance.

### Key Distinctions

- **Lesion vs stimulation:** Both directions are demonstrated: ablating individual Fourier components degrades specific inputs (lesion), and the complete algorithm reproduces outputs from weights alone (equivalent to showing sufficiency without stimulation artifacts).
- **Structural vs functional connectivity:** Structural connectivity (weight-space Fourier patterns) and functional connectivity (which inputs activate which components during inference) are both fully characterized and perfectly aligned. In real models, these often dissociate.
- **Single vs double dissociation:** Trivially satisfied. Per-frequency ablation shows that each frequency is necessary for its corresponding input pairs and unnecessary for others — this is a complete dissociation matrix within the single-task domain.

### Dissociation Matrix

|  | Input pairs using frequency $k_1$ | Input pairs using frequency $k_2$ | Input pairs using frequency $k_3$ |
|---|---|---|---|
| Ablate frequency $k_1$ | **↓↓ (strong)** | No effect | No effect |
| Ablate frequency $k_2$ | No effect | **↓↓ (strong)** | No effect |
| Ablate frequency $k_3$ | No effect | No effect | **↓↓ (strong)** |

A clean diagonal matrix — each frequency is necessary and sufficient for its corresponding inputs, and unnecessary for others. This is the textbook double-dissociation pattern, achieved because the mechanism is fully decomposable into independent frequency channels. No other MI result has this level of dissociation evidence.

---

## Pharmacology Lens — External Validity

*Does intervening on the mechanism produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/framework/criteria/external/intervention-reach) Pass.** You can manipulate specific Fourier components and predict the exact change in outputs. Complete intervention control.

**[E2 — Graded response:](/framework/criteria/external/graded-response) Pass.** Partially ablating a Fourier component (scaling it down) produces graded degradation proportional to the scaling factor. Perfect dose-response.

**[E3 — Selectivity:](/framework/criteria/external/selectivity) Pass.** Intervening on one frequency affects only the input pairs that rely on that frequency. Clean selectivity.

**[E4 — Effect magnitude:](/framework/criteria/external/effect-magnitude) Complete.** The algorithm accounts for 100% of the model's computation. No unexplained variance.

**[E5 — Robustness:](/framework/criteria/external/robustness) Pass (within scope).** The algorithm works for all inputs in the modular arithmetic domain.

**[E6 — Cross-architecture:](/framework/criteria/external/cross-architecture) Weak — the critical gap.** This is a 1-layer toy transformer. Whether real models (GPT-2, Pythia) use Fourier-like representations for arithmetic is unknown. The algorithmic insight may not transfer to models with multiple layers, larger vocabularies, and diverse training data.

### Key Distinctions

- **The system compensates:** Because this is a toy model trained on a single task, system compensation is essentially absent — there are no alternative pathways or redundant mechanisms that could mask intervention effects. This is why ablation results are so clean, and why this clarity may not transfer to real models.
- **Affinity vs efficacy:** Both are maximally demonstrated. The Fourier structure shows the mechanism has the capacity (affinity) and the complete output reproduction shows it exercises that capacity (efficacy). There is no gap between structural potential and functional reality.
- **Naming requires criteria:** "Fourier features" is perhaps the most rigorously operationalized name in MI — it refers to specific measurable mathematical structure in weight matrices with a precise functional interpretation.

### Dose-Response Curve

The grokking/modular addition circuit provides the ideal dose-response:
- **Dose axis:** Scaling factor applied to a Fourier frequency component (0 = fully ablated, 1 = normal)
- **Response axis:** Accuracy on input pairs relying on that frequency
- **Observed relationship:** Perfectly proportional — graded degradation tracks scaling factor linearly
- **EC₅₀:** Approximately 0.5 (half-strength component produces half the accuracy benefit)
- **Selectivity:** Perfect — intervening on frequency $k$ affects only input pairs involving $k$, with zero off-target effects
- **Therapeutic window:** Infinite — any intervention strength between 0 and 1 is "safe" (produces only the intended, predicted effect)

This is the pharmacological ideal: a perfectly linear dose-response with perfect selectivity and no off-target effects. It serves as a reference standard against which real-model dose-response curves should be compared.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/framework/criteria/measurement/reliability) Pass.** The Fourier decomposition is deterministic — same model, same result every time. Replication across seeds confirms the finding.

**[M2 — Invariance:](/framework/criteria/measurement/invariance) Pass.** The measurement works regardless of which specific frequencies the model chose — the *type* of algorithm is invariant across training runs.

**[M3 — Baseline separation:](/framework/criteria/measurement/baseline-separation) Pass.** Random models show no Fourier structure. The signal is clearly above noise.

**[M4 — Sensitivity:](/framework/criteria/measurement/sensitivity) Pass.** The Fourier decomposition precisely identifies which components contribute and which do not.

**[M5 — Calibration:](/framework/criteria/measurement/calibration) Pass.** The algorithm's output matches the model's output to numerical precision — perfect calibration.

**[M6 — Construct coverage:](/framework/criteria/measurement/construct-coverage) Complete.** Every parameter is explained. Nothing is left unmeasured.

### Key Distinctions

- **Reliability vs validity:** Both are maximally satisfied. The Fourier decomposition is perfectly reliable (deterministic, reproducible across seeds at the algorithm-type level) and perfectly valid (it predicts outputs exactly). This is the measurement ideal that real-model studies aspire to.
- **Convergent vs discriminant validity:** Four independent evidence lines (weight decomposition, activation probing, mechanistic prediction, training dynamics) all converge on the same Fourier structure. Discriminant validity is trivially satisfied — random untrained models show no Fourier structure whatsoever.

### MTMM Matrix

| | Weight decomposition (Fourier) | Activation probing (Fourier) | Mechanistic prediction (Fourier) | Training dynamics (Fourier) |
|---|---|---|---|---|
| **Weight decomposition** | — | High (convergent) | High (convergent) | High (convergent) |
| **Activation probing** | High | — | High (convergent) | High (convergent) |
| **Mechanistic prediction** | High | High | — | High (convergent) |
| **Training dynamics** | High | High | High | — |

All convergent cells are high — four independent methods identify the same structure. Discriminant validity is trivially satisfied (untrained models show zero Fourier structure by any method). This is a maximally well-behaved MTMM pattern: all methods agree on the presence of the construct, and all methods agree on its absence in controls.

---

## MI Lens — Interpretive Validity

*Is the interpretation warranted by the evidence?*

### Criteria

**[V1 — Level declaration:](/framework/criteria/interpretive/level-declaration) Pass.** The claim is at the [structural](/framework/modes_v3/structural) level — it fully specifies the algorithm in terms of weight matrices.

**[V2 — Level-evidence match:](/framework/criteria/interpretive/level-evidence-match) Strong pass.** The evidence *is* the structure — weight matrices are directly decoded into the algorithm. Evidence and claim are at the same level.

**[V3 — Narrative coherence:](/framework/criteria/interpretive/narrative-coherence) Strong.** The story (embed as Fourier → combine via trig identities → decode result) is mathematically precise and mechanistically complete.

**[V4 — Alternative exclusion:](/framework/criteria/interpretive/alternative-exclusion) Pass.** The mechanistic account is so complete that alternative explanations are effectively excluded — you cannot explain why the weights have Fourier structure if the model is not computing via Fourier.

**[V5 — Scope honesty:](/framework/criteria/interpretive/scope-honesty) Pass — with a caveat.** The claim is honest about scope (modular addition in a toy model). The caveat is that readers may over-generalize: "transformers learn Fourier algorithms" is not what the paper shows. The paper shows that *this* toy model learns *this* Fourier algorithm.

### Key Distinctions

- **Description vs explanation:** This is the strongest example of genuine explanation in MI. The mechanistic account does not merely describe which components are active — it specifies the exact mathematical algorithm and explains why the weights take their specific values. The explanation is complete: given the task and the Fourier algorithm, every weight matrix entry is predicted.
- **Component identity vs component role:** There is no gap between identity and role here. Each weight matrix entry has a precise functional interpretation derived from the Fourier algorithm. The "role" is not an interpretive label applied post hoc — it is a mathematical prediction verified against observations.
- **Faithfulness vs understanding:** Both are maximally satisfied. The algorithm is faithful (it reproduces 100% of outputs) and understood (the mathematical basis is completely specified). This is the only MI result where faithfulness and understanding are both at ceiling.

### Evidence Convergence Map

- **Implementational → Interpretation:** Perfect. Every weight matrix is decoded. No unexplained parameters.
- **Algorithmic → Interpretation:** Perfect. The algorithm (Fourier-based trigonometric combination) is mathematically specified and verified.
- **Computational → Interpretation:** Perfect (within scope). The computation (modular addition) is exactly what the algorithm produces.

All three levels converge perfectly — this is the only MI result with complete convergence across all evidence modes.

### Intervention-Interpretation Matrix

| | Necessity | Sufficiency | Representational | Algorithmic | Computational |
|---|---|---|---|---|---|
| Per-frequency ablation | ✓ | — | ✓ | ✓ | ✓ |
| Weight decomposition | — | ✓ | ✓ | ✓ | ✓ |
| Mechanistic prediction | — | ✓ | ✓ | ✓ | ✓ |
| Training dynamics | — | — | ✓ | ✓ | — |

Nearly all cells filled. The only systematic gap is that ablation provides necessity but not sufficiency (you remove something and performance drops, but you cannot "add" a new frequency). The weight decomposition and mechanistic prediction rows provide sufficiency (the algorithm reproduces outputs without any intervention). This is the most complete intervention-interpretation matrix in MI.

### Causal Sufficiency Graph

- Input tokens → Fourier embedding: **solid** (embedding matrix entries are verified as $\cos$/$\sin$ of input position)
- Fourier embedding → trigonometric combination (attention): **solid** (attention weights implement the addition formula)
- Trigonometric combination → output decoding: **solid** (unembedding reads off the result from the combined representation)
- Full path (input → embedding → attention → output): **solid** (complete end-to-end causal chain verified by exact output reproduction)

All edges solid. Every step in the causal chain is independently verified AND the complete chain reproduces outputs exactly. This is the only MI result with a fully verified causal sufficiency graph — no dashed edges, no unknown interactions, no gaps.

---

