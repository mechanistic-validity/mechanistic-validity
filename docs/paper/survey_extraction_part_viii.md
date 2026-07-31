# MI Deep Survey Part VIII: Extraction

Extracted from "MI Methods Deep Survey Part VIII --- Four Theoretical
Earthquakes: Superposition as Scaling Law, COCONUT's Failure, Semantic
Hubs, LLM Language Networks, and ModCirc Global Interpretability"
(May 2026).

Covers 7 major findings from 9 papers (2024--2026), primarily theoretical
rather than methodological. The unifying theme: the conceptual landscape
underpinning MI has shifted --- superposition is geometrically inevitable,
latent reasoning fails validity tests, cross-modal hubs are causally real,
and circuits need cross-task modularity testing. Each finding has concrete
implications for specific validity framework criteria.

---

## A. New Methods to Implement

### Tier 1: Critical (directly maps to framework criteria or fills gap)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| Functional Localizer | Neuroscience-style domain-selective localization: contrast conditions identify units selective for a domain, ablation confirms causal necessity | AlKhamissi, Tuckute et al. (EPFL + MIT); EMNLP 2025 (llm-language-network.epfl.ch) | mechanistic_interpretability | evaluation |
| ModCirc Cross-Task Modularity | Measures whether circuit components reuse across tasks (task-agnosticity, composability, faithfulness); 5 modularity criteria | He, Zheng et al.; ICML 2025 (PMLR 267:22865) | mechanistic_interpretability | evaluation |
| Latent Reasoning Validity | Tests whether latent/continuous-thought tokens encode steerable, causally necessary reasoning (steering sensitivity + shortcut exploitation detection) | Zhang, Tang et al. (2025, arXiv:2512.21711); evaluating Hao et al. COCONUT (COLM 2025, arXiv:2412.06769) | mechanistic_interpretability | evaluation |

### Tier 2: High (new tests with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| Semantic Hub Convergence | Cross-modal/cross-lingual semantic agreement at intermediate layers; causal cross-modal intervention test | Wu, Yu, Yogatama, Lu, Kim; ICLR 2025 (arXiv:2411.04986) | mechanistic_interpretability | evaluation |
| Dual Mechanism Discriminant Validity | Decomposes value/safety directions into intrinsic vs. prompted mechanisms; measures discriminant validity between overlapping but distinct pathways | Han, Lim, Kong, Jo; NeurIPS 2025 Workshop / ICML 2026 (arXiv:2509.24319) | mechanistic_interpretability | evaluation |
| Superposition Regime Diagnostic | Measures feature interference in residual stream to determine whether model operates in weak vs. strong superposition regime; L ~ 1/m scaling test | Liu, Liu, Gore; NeurIPS 2025 Best Paper Runner-Up (arXiv:2505.10465) | measurement_theory | evaluation |

### Tier 3: Theoretical (no direct metric, but framing implications)

| Finding | What | Source | Implication |
|---|---|---|---|
| Alternative Models of Superposition | TMS toy model inaccurate for real networks; no competing theory | LessWrong Aug 2025 (pCJXa3DbEfGjcZAgZ) | Motivates empirical validation framework over theory-derived criteria |
| Brain Alignment Saturation | Brain alignment saturates at 4B tokens; invalid for frontier models | Hosseini et al.; EMNLP 2025 | Brain alignment is not a valid C5 criterion for frontier models |

---

## B. New Metrics (suggested implementations)

### B1. Functional Localizer (Domain-Selective Causal Units)

- **metric_id**: `EX19_functional_localizer`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I1 Necessity, I3 Specificity)
- **criteria**: I1 Component Necessity; I3 Specificity; C5 Convergent Validity
- **what it measures**: Applies the neuroscience functional localizer paradigm to LLMs. Runs standardized contrast conditions (linguistic vs. non-linguistic stimuli) to identify units that respond selectively to the target domain. Then ablates those units to confirm causal necessity. The selectivity index (d-prime between domain and control activations) and the causal deficit (performance drop from selective ablation vs. random ablation) are the primary diagnostics.
- **pass condition**: causal_deficit > 0.2 (selective ablation hurts domain performance significantly more than random ablation of equal count); selectivity_dprime > 1.0
- **implementation notes**: Requires domain-specific and control stimuli. For language: sentences vs. scrambled/reversed tokens. For reasoning: logic problems vs. pattern matching. The method generalizes to any domain contrast. The EPFL implementation tests 18 LLMs; our implementation focuses on the per-model diagnostic.
- **why critical**: Provides a neuroscience-validated protocol for testing whether circuit components are domain-specific (M-frame criterion). A circuit that fires equally to domain and control stimuli has no domain selectivity, regardless of attribution patching scores.

### B2. ModCirc Cross-Task Modularity Score

- **metric_id**: `EX20_modcirc_modularity`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity (cross-task); E2 Causal Sufficiency
- **what it measures**: For a set of circuits discovered on different tasks, measures (1) node overlap between related tasks, (2) faithfulness preservation when a circuit discovered on task A is applied to task B, and (3) compositional coverage (fraction of model behavior explained by the circuit vocabulary). The cross-task faithfulness score is the primary diagnostic.
- **pass condition**: cross_task_faithfulness > 0.4 (circuit discovered on task A maintains >40% faithfulness on related task B)
- **implementation notes**: Requires circuits for multiple tasks (from the task registry). Tests each circuit on every other task's prompts. The node overlap and cross-task faithfulness together operationalize ModCirc's first two criteria (task-agnosticity and composability). This extends the existing `cross_task_generalization` metric with circuit-level specificity.
- **why critical**: Every existing circuit paper tests faithfulness on the discovery task only. This metric tests whether circuits generalize, which is the minimum bar for them being model properties rather than task artifacts.

### B3. Latent Reasoning Validity Score

- **metric_id**: `EX21_latent_reasoning_validity`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency); External (E1 Content Validity)
- **criteria**: E1 Content Validity; I2 Compositional Sufficiency; C5 Convergent Validity
- **what it measures**: For intermediate representations claimed to encode reasoning states, tests (1) steering sensitivity: whether perturbing specific dimensions of the representation predictably changes the output (not just noise); (2) shortcut detection: whether the representation's predictive success depends on dataset artifacts rather than genuine reasoning. The steering_sensitivity score and shortcut_exploitation_rate are the two diagnostics.
- **pass condition**: steering_sensitivity > 0.3; shortcut_exploitation_rate < 0.2
- **implementation notes**: Generalizes the "Do Latent Tokens Think?" methodology beyond COCONUT to any representation claimed to encode task-relevant information (SAE features, circuit activations, hidden states). The steering test perturbs the representation and checks if output changes are semantically consistent. The shortcut test compares performance on biased vs. unbiased versions of the same task.
- **why critical**: As models move toward latent reasoning (COCONUT, o1-style), interpretability claims about internal representations must be validated. This metric provides the validation protocol.

### B4. Semantic Hub Convergence Score

- **metric_id**: `EX22_semantic_hub_convergence`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity (cross-modal)
- **what it measures**: For semantically equivalent inputs in different surface forms (languages, code vs. math, formal vs. informal), measures representation similarity at intermediate layers. The hub_convergence_score is the mean cosine similarity of semantically equivalent inputs at the layer of maximum convergence. The causal_cross_modal_effect measures whether intervening on the hub representation in one modality predictably changes outputs in another.
- **pass condition**: hub_convergence_score > 0.5; causal_cross_modal_effect > 0.1
- **implementation notes**: Requires paired inputs with equivalent semantics in different surface forms. For language models: English/French/German translations of the same sentences. For code: equivalent algorithms in different languages. The layer sweep identifies the "hub layer" (peak convergence). This is the strongest possible C5 test: two completely different input forms producing the same internal representation.
- **why critical**: Features or circuits that pass the hub test are semantically grounded, not surface-form artifacts. This is a stronger validity criterion than any single-modality test.

### B5. Dual Mechanism Discriminant Validity

- **metric_id**: `EX23_dual_mechanism_discriminant`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C4 Discriminant Validity)
- **criteria**: C4 Discriminant Validity
- **what it measures**: For a given behavioral construct (e.g., "model values", "safety behavior"), extracts two representation directions: (1) intrinsic direction from baseline behavior and (2) prompted direction from instruction-following behavior. Measures the overlap (cosine similarity) and the independent steering effect of each direction's residual after removing the shared component. The discriminant_separation and independent_steering_effect are the primary diagnostics.
- **pass condition**: discriminant_separation > 0.2 (the two directions are not identical); independent_steering_effect > 0.1 (the residual of each still steers independently)
- **implementation notes**: Generalizes the Han et al. dual mechanism finding. For each construct: collect activations under baseline conditions (intrinsic) and under system-prompt conditions (prompted). Extract mean difference directions. Compute shared vs. independent components. Test steering with each independently. Safety-relevant because conflating intrinsic and prompted mechanisms leads to false confidence in alignment interventions.
- **why critical**: Most direct empirical instantiation of C4 Discriminant Validity applied to safety-relevant constructs. Without this test, safety measures that only target the prompted mechanism leave the intrinsic mechanism unchanged.

### B6. Superposition Regime Diagnostic

- **metric_id**: `M13_superposition_regime`
- **lens**: measurement_theory
- **validity_type**: Measurement (M6 Construct Coverage)
- **criteria**: M6 Construct Coverage
- **what it measures**: Estimates whether a model layer operates in the weak or strong superposition regime by measuring feature interference. Computes (1) feature packing ratio: number of active directions vs. residual stream dimension, (2) interference score: mean absolute pairwise cosine similarity between active feature directions, (3) scaling exponent: fit of loss ~ d_model^alpha. Strong superposition is indicated by packing_ratio >> 1, high interference, and alpha ~ -1.
- **pass condition**: This is a diagnostic, not pass/fail. Reports regime classification (weak/strong/transition) and the quantitative indicators.
- **implementation notes**: Uses SVD of activation matrices and pairwise cosine similarity of top principal components. The packing ratio comes from comparing effective rank to embedding dimension. The scaling exponent requires comparing models of different sizes from the same family (Pythia, OPT). Can be approximated for a single model by comparing effective dimensionality across layers.
- **why critical**: If a model operates in strong superposition, SAE features cannot be "the true features" --- they are one of many valid decompositions. This diagnostic informs how to interpret all downstream validity results.

---

## C. New Adapter Types Needed

This part introduces primarily theoretical findings, so adapter needs are
minimal compared to Parts IV--VII.

| Adapter | For | Why it's different | Priority |
|---|---|---|---|
| `LatentReasoningAdapter` | COCONUT-style continuous thought representations | Non-token-aligned; variable-length latent sequences; no discrete tokenization | LOW (research frontier) |
| `MultimodalHubAdapter` | Cross-modal representations at hub layers | Must align representations from different input modalities; requires paired inputs | LOW (requires multimodal models) |

Note: Most Part VIII metrics operate on standard HookedTransformer hook
points and do not require new adapter types. The functional localizer,
ModCirc modularity, dual mechanism, and superposition regime diagnostics
all work with existing activation capture infrastructure.

---

## D. Key Paper Framings (strongest arguments from Part VIII)

### D1. Superposition is the mechanism, not the failure mode

> "Scaling laws are geometrically derived from superposition. Removing
> superposition would break the scaling properties that make LLMs effective.
> No SAE can recover 'the true features' because no such unique set exists
> in the strong-superposition regime."

The NeurIPS 2025 Best Paper Runner-Up. The single most important theoretical
finding: the theoretical basis for SAEs (recover features from superposition
via sparsity) is contradicted by the geometric reality (superposition is
irreducible, not recoverable). The framework is needed precisely because
no decomposition can claim to be uniquely correct.

**Use in paper**: Lead theoretical motivation. Cite as: "If features are
not uniquely recoverable (Liu et al. 2025), then any particular
decomposition's features require external validity criteria to determine
whether they are useful --- which is what the framework provides."

### D2. COCONUT fails all three validity tests (worked example)

> "COCONUT continuous thoughts fail E1 (no steerable reasoning), E2
> (shortcut exploitation), and C5 (OOD collapse). The 'Do Latent Tokens
> Think?' paper literally implements the validity framework and finds a
> famous method fails."

Perfect worked example of the framework applied to a non-SAE method.
The paper tests steering sensitivity (E1), causal sufficiency (E2),
and OOD generalization (C5) --- and finds all three fail.

**Use in paper**: Extended example in the framework application section.
Shows the framework generalizes beyond SAE features to any representation
claim.

### D3. The semantic hub is causally active (strongest C5 evidence)

> "Intervening in the semantic hub in one modality predictably affects
> outputs in other modalities. Cross-modal causal generalization is by
> definition a robust convergent validity finding."

The Semantic Hub Hypothesis (ICLR 2025) provides the strongest possible
C5 Convergent Validity evidence: the same representation is found across
completely different input modalities, and it is causally active.

**Use in paper**: Example of what strong C5 evidence looks like. Sets the
standard that single-modality C5 tests should aspire to.

### D4. Functional localizers provide ready-made M-frame tests

> "A circuit that fires selectively to language vs. non-linguistic stimuli
> and whose ablation specifically hurts language performance is a valid
> circuit. A circuit that fires equally to both is a failed M-frame result."

The EPFL/MIT functional localizer paradigm translates directly to the
validity framework's domain-specificity criterion.

**Use in paper**: Concrete protocol component. The functional localizer is
a neuroscience method that already does what the M-frame requires, tested
across 18 LLMs.

### D5. Brain alignment is invalid for frontier models

> "Brain alignment saturates at ~4B tokens of training. For frontier models
> which have surpassed human language proficiency, brain alignment is no
> longer a meaningful benchmark. Model-internal validity criteria --- the
> framework --- are the only remaining option."

**Use in paper**: Motivates the framework's model-internal criteria. Brain
alignment was the field's best external validity criterion; it stops working
at scale. The framework fills the gap.

### D6. ModCirc independently converges on framework criteria

> "ModCirc's five criteria (task-agnosticity, composability,
> interpretability, faithfulness, coverage) are a restricted version of the
> general validity criteria. The field independently converged on the need
> for the framework's measurement approach."

**Use in paper**: Convergent evidence that the framework's criteria are not
idiosyncratic. When an ICML 2025 paper independently derives 5 criteria
that map directly onto the framework's validity dimensions, the framework
is not over-specified --- it is the formalization of what the field already
needs.

### D7. Dual mechanisms create a C4 requirement for safety

> "Intrinsic and prompted value mechanisms are partially overlapping but
> genuinely distinct. Safety measures that target only the prompted
> mechanism leave intrinsic values unchanged."

**Use in paper**: Most precise empirical instantiation of C4 Discriminant
Validity applied to safety. Without discriminant validity testing, safety
interventions may be ineffective against the intrinsic mechanism while
appearing successful against the prompted one.

---

## E. Gaps Identified

### E1. No metric for "decomposition utility despite non-uniqueness"

The superposition scaling law paper implies that no decomposition is
uniquely correct. The framework needs a criterion for "useful despite
non-unique" --- a decomposition is valid not because it recovers the true
features, but because it satisfies a set of utility criteria (causal
influence, stability, interpretability) to a sufficient degree.

### E2. No metric for latent reasoning interpretability

If models move to COCONUT-style latent reasoning, the entire CoT
monitoring paradigm fails. The framework needs criteria for what
"valid interpretation of latent tokens" requires. The B3 metric
(Latent Reasoning Validity) is a first step.

### E3. No cross-modal validity test in current framework

The Semantic Hub finding enables a new class of C5 tests (cross-modal
agreement) that is strictly stronger than within-modality convergent
validity. The B4 metric (Semantic Hub Convergence) addresses this, but
requires multimodal model support.

### E4. Brain alignment transition zone needs formalization

The framework should specify at what model scale brain alignment
transitions from useful to invalid as a C5 criterion. The EMNLP 2025
finding suggests ~4B training tokens as the transition point, but this
needs operationalization.
