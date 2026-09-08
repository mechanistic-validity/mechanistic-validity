---
title: "Case Study: Grokking / Modular Addition"
description: "The modular addition circuit with Fourier features (Nanda et al. 2023) evaluated through the five core lenses."
---

# Case Study: Grokking / Modular Addition

[Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) analyze a small transformer trained on **modular addition** ($a + b \mod p$) that undergoes "grokking" — sudden generalization long after memorizing the training set. They claim the model learns a **Fourier-based algorithm**: inputs are embedded into Fourier components (sinusoidal representations of position mod $p$), attention computes trigonometric identities to combine them, and the output reads off the result from the Fourier representation.

This is the strongest structural evidence in published MI — the weight matrices are fully reverse-engineered and the algorithm is mathematically specified. The catch: it is a toy model (1-layer transformer, mod-113 arithmetic). The construct validity question is whether this tells us anything about real models.

## Composite Verdict

> **Verdict (framework paper, Table 6):** Mechanistically Supported. **Capped by:** I6 (double dissociation).


| Lens | Strongest | Weakest | Overall |
|---|---|---|---|
| Construct | C1/C2/C3 (all strong) | — | Strong |
| Internal | I2/I7 (sufficiency + confound) | I6 (double dissociation) | Mechanistically Supported |
| External | E1/E2 Intervention reach + Prompt generalization | E5 Graded response | Partial |
| Measurement | M2/M6 Baseline separation + Invariance | M5/M7 Sensitivity + Selection correction | Partial–Strong |
| Interpretive | V3 Alternative level | — | Strong |

**Overall verdict: Mechanistically Supported.** The modular addition circuit has strong structural and causal evidence — the algorithm is mathematically specified and every weight matrix is accounted for. The capping criterion is I6 (double dissociation): no study has tested a second behavior that the Fourier circuit spares while ablating a matched control circuit impairs. Despite the completeness of the reverse engineering, this crossed-design test has not been performed.

The scope limitation is narrower than it looks: E4 is partial rather than untested — one- and two-layer transformers are run at origin, and nothing larger. The capping criterion remains I6.

## Metrics used in original work

| Method | Our metric | Family |
|---|---|---|
| Weight-space Fourier decomposition | [B01 SVD/Spectral](/mechanistic-validity/framework/metrics/#b01) | Structural |
| Per-frequency ablation | [A01 Pearl SCM](/mechanistic-validity/framework/metrics/#a01) | Causal |
| Activation probing (Fourier components) | [E02 Linear Probe](/mechanistic-validity/framework/metrics/#e02) | Representational |
| Mechanistic prediction (exact output reproduction) | [D04 CE Delta](/mechanistic-validity/framework/metrics/#d04) | Behavioral |
| Training dynamics analysis (phase transition) | [D04 CE Delta](/mechanistic-validity/framework/metrics/#d04) | Behavioral |

> To run these metrics yourself, see [Experiment 10: Published Circuit Evaluation](https://github.com/mechanistic-validity/mechanistic-validity-experiments/tree/main/experiments/10_published_circuit_evaluation).

---

## Philosophy of Science Lens — Construct Validity

*Is "the Fourier algorithm" a coherent construct?*

### Criteria

**[C1 — Falsifiability:](/mechanistic-validity/framework/criteria/construct/falsifiability) Strong pass.** The claim specifies the exact algorithm: the model computes $\cos(2\pi k(a+b)/p)$ via trigonometric identities applied in the attention layer. This generates precise quantitative predictions about every weight matrix entry. Any deviation from the predicted Fourier structure would disconfirm the claim.

**[C2 — Structural plausibility:](/mechanistic-validity/framework/criteria/construct/structural-plausibility) Strong pass.** This is the paper's primary achievement. The embedding matrix entries are verified to approximate $\cos(2\pi k a/p)$ and $\sin(2\pi k a/p)$ for specific frequencies $k$. The attention pattern implements the trigonometric addition formula. The unembedding reads off the result. Every weight matrix is accounted for — not just "consistent with" but "mathematically predicted by" the Fourier algorithm.

**[C4 — Discriminant validity:](/mechanistic-validity/framework/criteria/construct/discriminant-validity) Partial.** Excluded loss rises during circuit formation while train loss stays flat, which separates the Fourier mechanism from memorization. No neighboring construct is localized to separate it from.

**[I3 — Minimality:](/mechanistic-validity/framework/criteria/internal/minimality) Partial.** Every key frequency is load-bearing on a leave-one-out at frequency granularity. At neuron granularity it does not hold: 79 of 512 neurons fail the polynomial fit the account predicts.

**[C3 — Convergent validity:](/mechanistic-validity/framework/criteria/construct/convergent-validity) Partial.** The Fourier structure is identified through weight-space analysis (Fourier decomposition of $W_E$), activation-space analysis (probing for Fourier components), and mechanistic prediction (computing exact predicted outputs from the algorithm and comparing to actual outputs). Three evidence families at origin, but all three descend from Fourier-space ablation, so their agreement is partly the same primitive with itself.

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

**[I1 — Necessity:](/mechanistic-validity/framework/criteria/internal/necessity) Pass.** Ablating specific Fourier frequencies (zeroing the corresponding components in the embedding) degrades performance on input pairs involving those frequencies. The ablation is at the *feature* level rather than the component level, which is more precise.

**[I2 — Sufficiency:](/mechanistic-validity/framework/criteria/internal/sufficiency) Pass.** The Fourier algorithm, when executed manually on the model's weights, reproduces the model's outputs to high precision. This is the strongest possible sufficiency: the algorithm *is* the model, not just a description of it.

**[I4 — Specificity:](/mechanistic-validity/framework/criteria/internal/specificity) N/A.** One task; the model has no off-target behavior to spare. This is a statement about the setting, not a pass — a single-task toy model licenses less than a model doing many things.

**[I7 — Confound control:](/mechanistic-validity/framework/criteria/internal/confound-control) Partial.** Training the models is what buys the controls. Data fraction, modulus, depth, seed, regularizer type and weight-decay strength are all varied rather than found, and the λ = 0 run is the sharpest: with the regularizer removed the excluded loss stays flat, so the circuit does not form and grokking does not occur. What remains uncontrolled is on the analysis side, where thresholds are fixed, and in the architecture, since every mainline conclusion rests on one width and one depth.

### Key Distinctions

- **Lesion vs stimulation:** Both directions are demonstrated: ablating individual Fourier components degrades specific inputs (lesion), and the complete algorithm reproduces outputs from weights alone (equivalent to showing sufficiency without stimulation artifacts).
- **Structural vs functional connectivity:** Structural connectivity (weight-space Fourier patterns) and functional connectivity (which inputs activate which components during inference) are both fully characterized and perfectly aligned. In real models, these often dissociate.
- **Single vs double dissociation:** Not demonstrated. Per-frequency ablation is a within-mechanism decomposition, not a crossed design: no second mechanism has been shown intact under an ablation that breaks the Fourier circuit. This is the criterion that caps the claim.

### Dissociation Matrix

|  | Input pairs using frequency $k_1$ | Input pairs using frequency $k_2$ | Input pairs using frequency $k_3$ |
|---|---|---|---|
| Ablate frequency $k_1$ | **↓↓ (strong)** | No effect | No effect |
| Ablate frequency $k_2$ | No effect | **↓↓ (strong)** | No effect |
| Ablate frequency $k_3$ | No effect | No effect | **↓↓ (strong)** |

A clean diagonal within a single mechanism — each frequency is load-bearing for its own input pairs. This is frequency-level decomposability, not double dissociation: a crossed design needs a second mechanism that survives what breaks this one, and none has been run.

---

## Pharmacology Lens — External Validity

*Does intervening on the mechanism produce expected downstream effects?*

### Criteria

**[E1 — Intervention reach:](/mechanistic-validity/framework/criteria/external/intervention-reach) Pass.** You can manipulate specific Fourier components and predict the exact change in outputs. Complete intervention control.

**[E5 — Graded response:](/mechanistic-validity/framework/criteria/external/graded-response) Not tested.** Ablation is binary per component. No intervention interpolates between full and zero strength, so no dose-response curve exists.

**[E2 — Prompt generalization:](/mechanistic-validity/framework/criteria/external/prompt-generalization) Pass (within scope).** The algorithm works for all inputs in the modular arithmetic domain.

**[E4 — Cross-model generalization:](/mechanistic-validity/framework/criteria/external/cross-model-recurrence) Weak — the critical gap.** This is a 1-layer toy transformer. Whether real models (GPT-2, Pythia) use Fourier-like representations for arithmetic is unknown. The algorithmic insight may not transfer to models with multiple layers, larger vocabularies, and diverse training data.

### Key Distinctions

- **The system compensates:** Because this is a toy model trained on a single task, system compensation is essentially absent — there are no alternative pathways or redundant mechanisms that could mask intervention effects. This is why ablation results are so clean, and why this clarity may not transfer to real models.
- **Affinity vs efficacy:** Both are maximally demonstrated. The Fourier structure shows the mechanism has the capacity (affinity) and the complete output reproduction shows it exercises that capacity (efficacy). There is no gap between structural potential and functional reality.
- **Naming requires criteria:** "Fourier features" is perhaps the most rigorously operationalized name in MI — it refers to specific measurable mathematical structure in weight matrices with a precise functional interpretation.

### Dose-Response Curve

The grokking/modular addition circuit provides the ideal dose-response:
- **Dose axis:** Scaling factor applied to a Fourier frequency component (0 = fully ablated, 1 = normal)
- **Response axis:** Accuracy on input pairs relying on that frequency
- **Observed relationship:** not measured — the dose axis here is circuit *extent* (which frequencies are removed), not intervention *strength*
- **EC₅₀:** undefined — no intervention interpolates between full and zero strength
- **Selectivity:** intervening on frequency $k$ affects the input pairs involving $k$; the measurement is at frequency granularity, not intervention strength
- **Therapeutic window:** undefined — a window requires a strength axis, and none was swept

This is the pharmacological ideal: a perfectly linear dose-response with perfect selectivity and no off-target effects. It serves as a reference standard against which real-model dose-response curves should be compared.

---

## Measurement Theory Lens — Measurement Validity

*Are the metrics reliable and well-calibrated?*

### Criteria

**[M1 — Reliability:](/mechanistic-validity/framework/criteria/measurement/reliability) Partial.** Repetition is real at the level of training: five seeds at the mainline configuration, all five carried through the mechanism analysis, and a standard deviation of loss over runs reported. Repetition is absent at the level of the measurements that carry the mechanism claim — the fraction of variance explained is quoted as one number per direction with no interval, so the reader cannot tell how much of a 93.2% would survive resampling.

**[M6 — Invariance:](/mechanistic-validity/framework/criteria/measurement/invariance) Pass.** The measurement works regardless of which specific frequencies the model chose — the *type* of algorithm is invariant across training runs.

**[M2 — Baseline separation:](/mechanistic-validity/framework/criteria/measurement/baseline-separation) Pass.** Random models show no Fourier structure. The signal is clearly above noise.

**[M5 — Sensitivity:](/mechanistic-validity/framework/criteria/measurement/sensitivity) Not tested.** The controls are all known-negatives (untrained models, non-key frequencies). Nothing with a known answer is planted and recovered, so the instrument's floor is unmeasured.

**[M4 — Calibration:](/mechanistic-validity/framework/criteria/measurement/calibration) Partial.** Loss is reported in nats on a task whose chance level is fixed and known, so a number like 5.27 can be read directly as worse than a model predicting uniformly — more calibration than most interpretability results carry. The reference itself is never computed: "worse than uniform" is asserted and log 113 ≈ 4.73 appears nowhere, so the margin above chance has to be reconstructed by the reader. The fraction-of-variance-explained instrument has no false-positive reference at all.

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

**[V1 — Level declaration:](/mechanistic-validity/framework/criteria/interpretive/level-declaration) Partial.** The level is legible from usage and never declared. The abstract puts an algorithm-level claim and an implementation-level method in consecutive sentences, and §3.1 states the algorithm as a four-step procedure without saying whether the claim is about the function computed, the algorithm used, or the weights that realize it. A reader can reconstruct the answer; the paper does not supply it.

**[V2 — Level-evidence match:](/mechanistic-validity/framework/criteria/interpretive/level-evidence-match) Partial.** The evidence sits mostly at the weight level for an algorithm-level claim, which is a close match and unusual in this audit set. The overreach is in the word *fully*: by the paper's own numbers 79 of 512 neurons fall outside the single-frequency polynomial account, two of four attention heads receive a role introduced with "We speculate", and the headline logit reconstruction explains 95% of variance rather than all of it.

**[V3 — Alternative level:](/mechanistic-validity/framework/criteria/interpretive/alternative-level) Partial.** Memorization is addressed and refuted. The basis alternative is not: Zhong et al. (2023) give two distinct algorithms over the same five frequencies, and the metrics that would separate them are not run here.

**[V5 — Scope declaration:](/mechanistic-validity/framework/criteria/interpretive/scope-declaration) Pass — with a caveat.** The claim is honest about scope (modular addition in a toy model). The caveat is that readers may over-generalize: "transformers learn Fourier algorithms" is not what the paper shows. The paper shows that *this* toy model learns *this* Fourier algorithm.

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

