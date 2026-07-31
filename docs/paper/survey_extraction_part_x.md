# MI Deep Survey Part X: Extraction

Extracted from "MI Methods Deep Survey Part X --- The Infrastructure
Layer: NLAs, Gemma Scope 2, OpenAI Weight-Sparse Transformers, Goodfire
Silico, Assistant Axis, and TransformerLens v3" (May 2026).

Covers 6 major items from the tooling and platform layer (Nov 2025 --
May 2026): the first unsupervised activation verbalization method
(NLAs), the most comprehensively instrumented frontier model (Gemma
Scope 2), the highest-validated circuit paper in the literature
(weight-sparse transformers), the first production MI platform
(Goodfire Silico), a safety-critical representational direction
(Assistant Axis), and the multimodal expansion of the standard MI
tooling stack (TransformerLens v3 / SAELens v6). The unifying theme:
the MI ecosystem now has production-grade tooling, but no validation
layer for the claims those tools produce.

---

## A. New Methods to Implement

### Tier 1: Critical (directly maps to framework criteria or fills gap)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| NLA Semantic Validity Gap | Tests whether NLA round-trip reconstruction accuracy implies semantic validity of the natural language description --- the core gap identified by Anthropic's own release | Anthropic; transformer-circuits.pub/2026/nla/ (May 2026) | mechanistic_interpretability | evaluation |
| Weight-Sparse Circuit Completeness | Tests necessity + sufficiency of circuits extracted from weight-sparse transformers; the highest E2 Causal Sufficiency score in the literature | Gao, Rajaram, Coxon, Govande, Baker, Mossing (OpenAI); arXiv:2511.13653 (Nov 2025) | mechanistic_interpretability | evaluation |
| Assistant Axis Causal Stability | Tests whether a dominant PCA direction in persona space is causally sufficient for assistant character, reliable across persona samples, and discriminant from adjacent constructs | MATS + Anthropic Fellows; github.com/safety-research/assistant-axis (Jan 2026) | mechanistic_interpretability | evaluation |

### Tier 2: High (new tests with clear validity framework mapping)

| Method | What | Source | Lens | Metric type |
|---|---|---|---|---|
| NLA-SAE Convergent Validity | MTMM test: agreement between NLA descriptions and SAE-based feature descriptions for the same activations --- convergent validity across description methods | Anthropic NLA + SAELens; derived from Part X MTMM analysis | measurement_theory | evaluation |
| Goodfire Hallucination Reduction Validation | Tests whether feature-level interventions that reduce hallucinations generalize out-of-distribution and are due to correct feature identification vs. perturbation side effects | Goodfire; goodfire.ai (Apr 2026) | mechanistic_interpretability | evaluation |
| Gemma Scope 2 Cross-Method MTMM | Multi-trait multi-method comparison across SAE, transcoder, crosscoder, and CLT on Gemma 3 using Gemma Scope 2 artifacts | GDM; HuggingFace google/gemma-scope-2-27b-it (Dec 2025) | measurement_theory | evaluation |

### Tier 3: Infrastructure (tooling updates, no direct metric)

| Item | What | Source | Implication |
|---|---|---|---|
| TransformerLens v3 | Gemma 3 multimodal hooks, mT5 support, MPS CI | TransformerLensOrg (May 2026) | All protocols can run on Gemma 3 multimodal models; cross-modal C5 testing now available |
| SAELens v6 | Separate SAE architecture classes, any-model training, unified from_pretrained() | decoderesearch/SAELens (May 2026) | Any SAE architecture loadable via uniform API; backwards-incompatible migration |
| Awesome-MI repos | 5+ actively maintained curated lists; none include validity analysis | Various GitHub repos (2024--2026) | Framework fills the evaluation gap in every existing resource |

---

## B. New Metrics (suggested implementations)

### B1. NLA Semantic Validity Gap

- **metric_id**: `EX31_nla_semantic_validity`
- **lens**: mechanistic_interpretability
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: E1 Content Validity; C5 Convergent Validity
- **what it measures**: For each feature, compares the NLA-style round-trip reconstruction (does the activation survive compression into text and back?) with an independent semantic validity test (does the text description predict the feature's behavior on held-out examples?). The gap between reconstruction accuracy and predictive accuracy is the semantic validity gap: high reconstruction + low prediction = the text encodes activation information non-semantically.
- **pass condition**: semantic_validity_gap < 0.3 (reconstruction accuracy should not exceed predictive accuracy by more than 0.3). Features with gap > 0.3 have high informational validity but low semantic validity.
- **implementation notes**: Proxy implementation (full NLA requires Claude-scale models). Uses PCA-based reconstruction as proxy for NLA round-trip, and held-out top-k prediction accuracy as proxy for semantic validity. The gap between the two scores is the diagnostic. Extends EX5 (NLA Reconstruction Fidelity) with the semantic validity dimension.
- **why critical**: NLAs are Anthropic's flagship new method (May 2026). The validation gap (reconstruction != semantics) is the exact gap the framework fills. This is the highest-impact case study for the framework's utility.

### B2. Weight-Sparse Circuit Completeness

- **metric_id**: `EX32_weight_sparse_circuit`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I1 Necessity, I2 Sufficiency)
- **criteria**: E2 Causal Sufficiency; I1 Component Necessity
- **what it measures**: For a given task, identifies the minimal circuit via weight-magnitude pruning (zero out smallest-magnitude weights until task performance degrades). Tests necessity (removing any circuit component breaks the task) and sufficiency (the circuit alone performs the task). Reports circuit size ratio (circuit weights / total weights), necessity score (mean performance drop from single-component ablation), and sufficiency score (circuit-alone performance / full-model performance).
- **pass condition**: sufficiency_score > 0.8; necessity_score > 0.2; circuit_size_ratio < 0.1
- **implementation notes**: Weight-magnitude pruning is a standard technique applicable to any model (not just sparse-trained ones). The OpenAI paper shows circuits are ~16x smaller in sparse models vs. dense at matched loss. This metric applies the same pruning protocol to standard models as a baseline, measuring how close we get to the sparse-model ideal.
- **why critical**: Highest validated E2 Causal Sufficiency in the literature. Provides the existence proof and reference implementation for what a fully validated circuit looks like.

### B3. Assistant Axis Causal Stability

- **metric_id**: `EX33_assistant_axis`
- **lens**: mechanistic_interpretability
- **validity_type**: Internal (I2 Sufficiency); Measurement (M1 Reliability)
- **criteria**: E2 Causal Sufficiency; M1 Reliability; C4 Discriminant Validity
- **what it measures**: Tests three properties of a dominant representational direction: (1) Causal sufficiency --- suppressing the direction degrades target behavior (e.g., assistant character); (2) Reliability --- the direction is stable across different persona sample sets used to extract it; (3) Discriminant validity --- the direction is specific to the target construct and does not fire on adjacent but distinct constructs.
- **pass condition**: causal_deficit > 0.3 (suppression causes >30% behavior degradation); direction_stability > 0.8 (cosine similarity across extraction runs); discriminant_ratio > 2.0 (target activation / adjacent activation)
- **implementation notes**: Uses PCA on contrast activations (target persona vs. baseline) to extract the dominant direction. Causal test via activation addition/suppression. Reliability test via bootstrap resampling of the persona set. Discriminant test using adjacent-construct prompts (e.g., "helpful" vs. "harmless" vs. "honest"). The Assistant Axis paper tests E2 but not M1 or C4 --- this metric fills those gaps.
- **why critical**: The Assistant Axis is the clearest case study for applying all five validity lenses to a safety-critical claim. The paper validates E2 but leaves M1 and C4 untested --- the framework provides exactly those missing tests.

### B4. NLA-SAE Convergent Validity

- **metric_id**: `EX34_nla_sae_convergence`
- **lens**: measurement_theory
- **validity_type**: Construct (C5 Convergent Validity)
- **criteria**: C5 Convergent Validity
- **what it measures**: For features described by both NLA-style methods and SAE-based methods, measures agreement between the two descriptions. Agreement is computed via (1) top-k token overlap between NLA-predicted top tokens and SAE feature's max-activating tokens, and (2) direction cosine similarity between the NLA-reconstructed direction and the SAE decoder direction. The MTMM logic: two different measurement methods (NLA and SAE) should converge on the same feature description if the description is valid.
- **pass condition**: token_overlap > 0.3; direction_cosine > 0.5
- **implementation notes**: Proxy implementation. Uses PCA-based feature directions as NLA proxy and SAE decoder directions directly. Token overlap computed from top-k activating positions for each method. This directly implements the MTMM test suggested by Part X's analysis of NLAs.
- **why critical**: The most direct C5 test available for NLA features. Two completely independent description methods (verbalization-based and sparse-coding-based) pointing to the same feature description is strong convergent validity evidence.

---

## C. New Adapter Types Needed

Part X papers primarily operate on standard model representations and
the existing adapter infrastructure. No new adapter classes are needed.

| Adapter Enhancement | For | Why | Priority |
|---|---|---|---|
| `PersonaContrastAdapter` (method on existing adapter) | Assistant Axis extraction | Requires paired persona/baseline prompts and PCA on contrast vectors; adds dominant direction extraction as a reusable method | MEDIUM |
| `WeightPruningAdapter` (method on existing adapter) | Weight-sparse circuit extraction | Requires iterative weight-magnitude pruning with task-performance monitoring; adds circuit extraction via pruning | MEDIUM |

Note: Gemma Scope 2 artifacts load via standard `SAE.from_pretrained()`
from SAELens v6. The existing `SAEAdapter` handles this without
modification.

---

## D. Key Paper Framings (strongest arguments from Part X)

### D1. Round-trip accuracy != semantic validity (NLAs)

> "Round-trip accuracy can be high even if the text is nonsemantic
> (the AR learns to encode the activation in any textual form, not
> necessarily a human-meaningful one)."

NLAs conflate informational validity (does the text preserve enough
bits?) with semantic validity (does the text describe the activation
meaningfully?). The framework's E1 Content Validity criterion is
precisely the gap NLA validation doesn't close. This is the clearest
single example of why the framework matters: the field's most
high-profile new method has a known validation gap that maps exactly
to an existing framework criterion.

**Use in paper**: Lead case study. NLAs + framework = the gap is
identifiable and testable. Without the framework, the gap is merely
noted as a limitation in the release blog post.

### D2. Gemma Scope 2 as the reference MTMM substrate

> "Every layer, every sublayer, every method (SAE / transcoder /
> crosscoder / CLT) is publicly available for Gemma 3 27B-IT."

Gemma Scope 2 is the ideal substrate for MTMM testing because all
four method types (SAE, transcoder, crosscoder, CLT) are available
on the same model. Running the same validity protocol across all four
methods and comparing results gives convergent/discriminant validity
evidence that is impossible to obtain on any other model.

**Use in paper**: Reference implementation for all protocol examples.
Every worked example should include a Gemma 3 + Gemma Scope 2 variant.

### D3. Highest E2 Causal Sufficiency in the literature (OpenAI)

> "Each circuit is: verified necessary by ablation, verified sufficient
> by isolation, described at the weight level. Five residual channels,
> two neurons, one attention head --- fully sufficient and necessary."

The OpenAI sparse transformer paper achieves what no other circuit
paper has: both necessity and sufficiency validated at the weight level,
with circuits small enough to be fully human-readable. This is the
existence proof that high E2 is achievable --- but only at toy scale
with intrinsic sparsity.

**Use in paper**: Positive existence proof. The framework's E2
criterion is achievable, as demonstrated by this paper. The hard
question: what does E2 validation look like at production scale
without pre-baked sparsity?

### D4. Commercial MI needs validity testing (Goodfire)

> "At $1.25B valuation, Goodfire's interpretability claims will be
> scrutinized by customers, regulators, and safety researchers. A
> validity framework that can assess whether Goodfire's 'features'
> are valid descriptions of model behavior is commercially valuable."

Goodfire's 58% hallucination reduction claim is exactly the type of
downstream application the framework must evaluate. Without validity
testing: Is the improvement due to correctly identified features, or
perturbation side effects? Does it generalize OOD? Is the cost
advantage robust?

**Use in paper**: Industry motivation. The framework is not just
academically interesting --- it's commercially necessary as MI
enters production.

### D5. Assistant Axis: strongest E2, weakest M1/C4

> "The Assistant Axis passes E2 Causal Sufficiency (the strongest
> criterion) but is under-tested on M1, C4, and full E1."

The Assistant Axis is the clearest multi-lens case study: strong on
one criterion (E2), untested on others (M1, C4). The framework
reveals the specific gaps. Without it, the paper would be assessed
only on its E2 results and the M1/C4 gaps would go unnoticed.

**Use in paper**: Five-lens worked example showing how the framework
reveals gaps invisible from any single lens.

### D6. Cross-modal C5 testing is now available (TransformerLens v3)

> "TransformerLens v3.2's Gemma 3 multimodal support means the
> Semantic Hub Hypothesis test is now implementable: run the same
> protocol on text inputs and equivalent image inputs, compare
> activation patterns."

Cross-modal convergent validity testing requires hookable multimodal
models. TransformerLens v3 provides this for Gemma 3. The Semantic
Hub Hypothesis (cross-modal convergence at intermediate layers) is
now empirically testable as a C5 criterion.

**Use in paper**: Infrastructure readiness. The framework's most
ambitious C5 test (cross-modal convergence) is now implementable.

### D7. The ecosystem gap

> "Every tool above produces interpretability claims. None provides
> a principled framework for evaluating whether those claims are
> valid."

Across all ten survey parts, the MI ecosystem has: reference models
(Gemma 3, Llama 3.3, Claude), training libraries (SAELens, TL),
evaluation infrastructure (SAEBench, AxBench, MechEvalAgent), and
platforms (Neuronpedia, Goodfire, Anthropic tools). The framework is
the missing evaluation layer for the entire ecosystem.

**Use in paper**: This is the meta-argument across all ten parts.
Every layer of the ecosystem produces claims; none validates them.

---

## E. Gaps Identified

### E1. No validation protocol for NLA descriptions

NLA descriptions are validated only by round-trip reconstruction
accuracy. No protocol tests whether the natural language text is
semantically meaningful (E1), whether it is stable across different
activation samples (M1), or whether it discriminates from descriptions
of adjacent features (C4). The B1 metric (NLA Semantic Validity Gap)
addresses E1; M1 and C4 remain open.

### E2. No cross-method MTMM on same model

Gemma Scope 2 provides all four method types (SAE, transcoder,
crosscoder, CLT) on the same model. No paper runs the same validity
protocol across all four and compares results. This is the most
accessible MTMM test in the ecosystem and would provide the strongest
C5 evidence to date.

### E3. No production-scale circuit validation

The OpenAI weight-sparse transformer paper validates circuits at toy
scale. No comparable validation exists at production model scale.
The framework's E2 criterion needs a scaling analysis: how does
validation quality degrade as model size and circuit complexity
increase?

### E4. No validity framework for commercial MI claims

Goodfire's 58% hallucination reduction claim is unvalidated by any
independent framework. As MI enters production ($1.25B valuation),
the need for independent validity assessment of commercial claims
becomes urgent. The framework should include a "commercial claim
validation" protocol.

### E5. Assistant Axis M1 and C4 gaps

The Assistant Axis paper validates E2 but does not test M1
(reliability across different persona sample sets) or C4 (discriminant
validity from adjacent constructs like capability vs. character).
These are the highest-priority gaps for a safety-critical claim.

### E6. No cross-modal convergent validity baseline

TransformerLens v3 enables cross-modal C5 testing on Gemma 3, but
no baseline exists for what level of cross-modal convergence should
be expected. The Semantic Hub Hypothesis provides a theoretical
prediction; the framework needs an empirical calibration.

---

## F. Cross-References to Previous Parts

| Part X finding | Previous parts connection | Implication |
|---|---|---|
| NLA semantic validity gap | Part VII D4 (weight/activation description agreement) | NLA adds a third description modality (verbalization); MTMM now has three methods to triangulate |
| Gemma Scope 2 MTMM substrate | Part VII B7 (Matryoshka cross-scale consistency) | Gemma Scope 2 includes Matryoshka SAEs; cross-scale consistency testable on same artifacts |
| Weight-sparse circuits (OpenAI) | Part VII B8 (CircuitLens weight-circuit recovery) | Both are weight-based circuit methods; CircuitLens works post-hoc, OpenAI requires intrinsic sparsity |
| Goodfire commercial claims | Part IX D1 (SAEBench audit) | Commercial MI claims built on unaudited metrics inherit the SAEBench reliability crisis |
| Assistant Axis | Part VIII B6 (dual mechanism discriminant) | Assistant Axis is a single dominant direction; dual mechanism framework tests whether it is actually unitary or decomposable |
| TransformerLens v3 multimodal | Part VIII B4 (semantic hub convergence) | Multimodal hooks enable the cross-modal semantic hub test that Part VIII defined |
| MI as 2026 Breakthrough Technology | Part IX D2 (93% reproducibility failure) | Mainstream recognition increases scrutiny; the 93% failure rate becomes a public problem |

---

## G. Priority Matrix: Part X Additions

| Item | Release Date | Key Contribution | Framework Zone | Priority |
|---|---|---|---|---|
| NLA Semantic Validity Gap | May 2026 | Round-trip != semantic; E1 Content Validity gap | E1, C5 (highest-impact case study) | CRITICAL |
| Weight-Sparse Circuit Completeness | Nov 2025 | Existence proof for E2; necessity + sufficiency both validated | E2, I1, I2 | HIGH |
| Assistant Axis Causal Stability | Jan 2026 | Multi-lens case study; E2 strong, M1/C4 untested | E2, M1, C4 | HIGH |
| NLA-SAE Convergent Validity | May 2026 (derived) | MTMM across description methods | C5 | HIGH |
| Gemma Scope 2 Cross-Method MTMM | Dec 2025 | Reference dataset for all protocols | C5 (reference implementation) | HIGH (infrastructure) |
| Goodfire Hallucination Validation | Apr 2026 | Commercial MI needs validity testing | All lenses | MEDIUM (industry case study) |
| TransformerLens v3 / SAELens v6 | May 2026 | Cross-modal C5 testing now available | All protocols | MEDIUM (infrastructure) |

---

## H. Strategic Implications for the Paper

### H1. The ecosystem completion argument

Part X completes the ecosystem picture across all ten survey parts.
The MI field now has: methods (Parts I--III), validated and
invalidated claims (Parts IV--VI), architectural remediations
(Part VII), theoretical shifts (Part VIII), evaluation infrastructure
in crisis (Part IX), and production tooling (Part X). The framework
is the missing layer that connects all of these.

### H2. NLA as the lead case study

NLAs are the highest-profile recent MI method (Anthropic, May 2026,
featured on Neuronpedia). The validation gap (reconstruction !=
semantic validity) maps exactly to E1 Content Validity. Adding an
NLA case study to the framework demonstration is the highest-impact
single addition for the paper.

### H3. The commercial urgency argument

Goodfire's $1.25B valuation and Silico launch (Apr 2026) make MI
validity a commercial necessity, not just an academic concern.
Customers, regulators, and safety researchers will demand independent
validation of MI-based claims. The framework provides this.

### H4. The positive existence proof (OpenAI)

The weight-sparse transformer paper is the strongest positive result
in the survey: circuits that are necessary, sufficient, and fully
human-readable. This proves that the framework's highest criterion
(E2 Causal Sufficiency) is achievable. The challenge the framework
formalizes: extending this to production scale.

### H5. The five-lens case study (Assistant Axis)

The Assistant Axis is the ideal worked example for the paper:
strong E2, untested M1/C4, partial E1/C5. Running all five lenses
reveals specific actionable gaps. Without the framework, the paper
would be assessed only on its strongest result (E2), hiding the
weakest ones (M1, C4).

### H6. Infrastructure readiness

TransformerLens v3 + SAELens v6 + Gemma Scope 2 mean that every
protocol in the framework can be implemented against a standard
API and run on a comprehensively instrumented frontier model.
The infrastructure barrier is gone; the only missing piece is the
evaluation framework itself.
