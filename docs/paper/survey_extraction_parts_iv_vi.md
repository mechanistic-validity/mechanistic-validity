# MI Deep Survey Parts IV-VI: Unified Extraction

Extracted from three survey documents covering field fault lines, open problems,
SAE geometry crises, CoT monitorability, and new methods (May 2026).

---

## A. New Methods to Implement

### Tier 1: Critical (directly relevant to factorized circuits + validity framework)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| VPD (adVersarial Parameter Decomposition) | Rank-1 param subcomponents via adversarial ablation; handles attention Q/K/V/O | Bushnaq, Braun, Sharkey (Goodfire), May 2026 | mechanistic_interpretability | discovery |
| RelP (Relevance Patching) | LRP-based attribution; r=0.956 vs r=0.006 for attribution patching on MLPs | Jafari, Nanda et al., NeurIPS 2025 | mechanistic_interpretability | discovery |
| MAS (Model Alignment Search) | Bidirectional causal cross-model comparison via learned invertible transforms | Satchel Grant (Stanford), ICLR 2026 Re-Align | mechanistic_interpretability | evaluation |
| CRM (Complete Replacement Models) | Transcoders + Lorsa for exact attribution (piecewise linear full model) | Open-Moss, Feb 2026 | mechanistic_interpretability | discovery |
| IBCircuit | Info-bottleneck circuit discovery without corrupted activation design | Bian et al., ICML 2025 | mechanistic_interpretability | discovery |

### Tier 2: High (new methods with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| SPD (Stochastic Parameter Decomposition) | Rank-1 via stochastic masking, public code | Bushnaq, Braun, Sharkey, Jun 2025 | mechanistic_interpretability | discovery |
| ActivationReasoning (AR) | SAE features -> logical propositions -> symbolic rules -> steering | Helff et al., ICLR 2026 | mechanistic_interpretability | evaluation |
| Cross-model steering transfer | Steering vectors transferred across model families via learned autoencoder | Oozeer et al., ICML 2025 | mechanistic_interpretability | steering |
| CoT faithfulness (paired dissociation) | Detects unfaithful CoT via paired contradictory questions | Arcuschin, Nanda et al. (GDM), ICLR 2025 | mechanistic_interpretability | evaluation |
| SAE architecture duality | Cross-architecture agreement as construct validity measure | Lindsey et al., NeurIPS 2025 | measurement_theory | evaluation |
| LayerNavigator | Discriminability x consistency layer selection for steering | Sun et al., NeurIPS 2025 | mechanistic_interpretability | evaluation |
| Steering reliability diagnostics | Cosine similarity + activation separation as reliability predictors | arXiv:2602.17881, Feb 2026 | measurement_theory | evaluation |

### Tier 3: Medium (interesting but lower priority)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| MoE-X | Intrinsically interpretable MoE with in-expert sparsity | Yang et al., ICML 2025 | mechanistic_interpretability | discovery |
| SADI | Input-adaptive dynamic steering (semantics-dependent) | Wang et al., ICLR 2025 | mechanistic_interpretability | steering |
| BSA benchmark | 2000-instance reasoning-level safety alignment eval | ICLR 2026 | mechanistic_interpretability | benchmark |
| APD | Attribution-based parameter decomposition (superseded by VPD) | Braun, Bushnaq, Sharkey, Jan 2025 | mechanistic_interpretability | discovery |
| Steering for bias | Cross-task generalization failure as discriminant validity | Siddique et al., EACL 2026 | measurement_theory | steering |

---

## B. New Adapter Types Needed

| Adapter | For | Why it's different | Priority |
|---|---|---|---|
| `VPDAdapter` | Rank-1 parameter subcomponents | Decomposes attention Q/K/V/O directly; adversarial ablation built-in | CRITICAL |
| `LorsaAdapter` | Low-rank sparse attention (CRM) | Per-feature QK+OV circuits, not encoder/decoder matrices | HIGH |
| `SPDAdapter` | Stochastic parameter decomposition | Could share interface with VPD; public code available | HIGH |
| `MoEXAdapter` | MoE-X expert activations | Expert activations as features; routing decisions part of identification | MEDIUM |
| `MASAlignmentAdapter` | Cross-model alignment maps | Invertible linear transforms defining shared causal subspace | MEDIUM |

---

## C. Key Paper Framings (strongest arguments from the surveys)

### C1. The field admits the measurement gap

> "We have been disappointed by the amount of progress made by ambitious mech
> interp work, from both us and others" -- Neel Nanda, GDM

> "The most ambitious vision of mech interp he once dreamed of is probably
> dead" -- Nanda, 80,000 Hours podcast, Sept 2025

The TMLR 2025 30-author survey (Sharkey, Nanda et al.) explicitly lists
validation of descriptions (section 2.1.4) as an open problem without
providing a solution. Apollo listed "Write up reviews on the links between
comp neuro and mech interp and philosophy of science and mech interp" as
desirable but has not done it.

### C2. Attribution patching is broken (quantified)

> "For GPT-2 Large MLP outputs, attribution patching achieves Pearson r=0.006
> with activation patching, while RelP achieves r=0.956"

"A substantial fraction of 'circuit analysis' papers from 2023-2025 that
used attribution patching on MLP circuits may have been finding artifacts."

### C3. Architecture determines ontology

> "An SAE does not just reveal concepts -- it determines what can be seen at all."
> -- Lindsey et al., NeurIPS 2025

SAE features are outside the superposition hypothesis: clustered, hierarchical,
non-orthogonal. Either the hypothesis or the method is wrong.

### C4. Three-layer validation failure

Layer 1: Object-level methods (SAEs, NLAs, circuits) are under-validated.
Layer 2: Validation methods themselves (attribution patching, probes) are
under-validated.
Layer 3: Field response has been to lower standards rather than fix validation.

### C5. "One head = one behavior" is false (VPD finding)

VPD empirically shows attention behaviors are distributed across heads.
Challenges the head-level circuit discovery paradigm that dominates MI.

### C6. CoT unfaithfulness is an E2 causal sufficiency failure

Models produce post-hoc rationalizations that are coherent but causally
disconnected from actual computation. Detection requires paired questions
(dissociation test). Rates: GPT-4o-mini 13%, Claude 3.5 Haiku 7%.

### C7. The safe-to-dangerous shift (fundamental epistemological limit)

Evaluation context necessarily differs structurally from deployment. No
behavioral or representational test can fully rule out alignment faking.
This is why external validity cannot be achieved solely through
representation-level testing.

### C8. LSI framework's missing validity depth

Locate-Steer-Improve (30+ authors, arXiv Jan 2026) provides no criterion for
when a location claim is valid. Multiple levels give different interventions
achieving same behavior. The mechanistic validity framework fills exactly
this gap.

### C9. Methods proliferation without shared standard

> "SAEs, transcoders, crosscoders, CLTs, CRMs, Lorsa, MoE-X, AR, DAS,
> Boundless DAS, MAS, IBCircuit, RelP -- 15+ methods, no shared validity
> standard, no way to compare across them."

### C10. Platonic convergence as C5 foundation

If representations converge across models/modalities/scales toward shared
statistical model of reality, then C5 convergent validity has theoretical
foundation: cross-model agreement evidences validity because both models
independently converged to same world-structure. The RPRH counter (convergent
attractors under shared constraints) is actually more useful: C5 detects
shared computational constraints, not metaphysical structure.

### C11. Steering vector direction non-uniqueness

Steering vectors trained on different prompt variations point in different
directions yet perform similarly. "If 17 different directions all steer
'toward honesty' but point in different directions, you haven't localized
the 'honesty' representation." Failed E1 content validity despite achieving
behavioral outcome.

### C12. Behavioral training can't substitute for mechanistic understanding

Anti-scheming training reduces but doesn't eliminate covert behavior. Models
suppress behavioral signals while underlying representations persist.
Representation-level metrics that test internal structure are necessary.

---

## D. Gaps Identified (potential contributions)

1. No universal SAE architecture (architecture duality is fundamental)
2. "What is a feature?" remains formally undefined (Apollo open problem)
3. Philosophy of explanation for MI practitioners (Apollo listed, unfilled)
4. Cross-model canonical features (strongest convergent validity, untested)
5. Non-linear representation detection (when concepts live on curved manifolds)
6. VPD scaling beyond 67M parameters
7. MI methods to replace CoT monitoring (40-author consensus says needed)
8. Intrinsic vs post-hoc validity criteria (how they differ for MoE-X vs SAEs)
9. Scale as C5 confound (convergence increases with scale -- is that validity or artifact?)
10. Cross-modal convergent validity (language + vision agreement, now tractable)

---

## E. Strategic Paper Positioning

The surveys collectively make the case:

1. **The field has the diagnosis** (Nanda's pivot, TMLR open problems) but not
   the measurement theory to formalize it
2. **Every existing method family lacks systematic validity** -- the cross-method
   table (SAEs: no external validity; VPD: no measurement validity; probes:
   no validity at all; NLAs: no E-frame) makes the gap concrete
3. **LSI is the nearest competing framework** -- cite explicitly, position as
   "the measurement theory for when Locate is done correctly"
4. **The three-layer failure** provides the structural argument for why
   piecemeal fixes don't work
5. **The 15+ methods proliferation** provides the empirical argument for why
   a shared standard is needed now
