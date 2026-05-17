---
title: "Mechanistic Interpretability"
description: "The interpretive validity lens: does the claim match the strength and scope of the evidence?"
---

# The Mechanistic Interpretability Lens

This lens asks one question: **does the claim match the strength and scope of the evidence?**

Every MI result pairs a measurement with an interpretation. The measurement might be an IIA score, a faithfulness percentage, a logit difference, an ablation effect, or an alignment score. The interpretation is the claim attached to that measurement: "the model *computes* indirect object identification through a circuit of 26 heads," or "induction heads *implement* in-context learning." Whether the interpretation is justified depends on whether the claim is stated at a level the evidence can actually support.

This is the only lens in the framework not imported from another field. The pharmacological, measurement-theoretic, neuroscientific, and philosophical lenses each bring criteria developed and validated in their home disciplines. The criteria here are derived from the empirical track record of MI itself — from published cases where the number was real but the sentence outran it, and from analysis of what a corrected sentence would have required.

The positive formulation: these are standards to meet, not failures to avoid. Meeting them distinguishes a contribution ready for scrutiny from one that requires qualification. MI is a young field. Most published work does not yet meet all of them. That is expected and not a criticism — it is the specification of what would strengthen each claim.

## Key Distinctions

### Description vs explanation

"We described the circuit" and "we explained the computation" are different achievements. A description says what is there — these heads are active, this path carries signal, this component has high attribution. An explanation says why it works — the algorithm that necessitates each step, the structural property that makes the computation possible, what would have to change for the behavior to disappear.

In MI: most published "explanations" are descriptions in explanatory language. "The name-mover head copies the IO name to the output" sounds like an explanation but is a description of observed behavior on specific inputs. The explanation would answer: what structural property of the W_OV matrix forces name-copying? Under what conditions would this head fail to copy? What is the minimal perturbation that breaks it? Descriptions are bound to the cases you observed. Explanations predict what happens in cases you haven't tested. The gap between them is the gap between reporting and understanding.

### Faithfulness vs understanding

A circuit can be maximally faithful — 100% performance recovery when isolated, total degradation when ablated — without being understood at all. Faithfulness is a property of the circuit relative to the task. Understanding is a property of the human relative to the circuit. They are independent axes.

In MI: a maximally faithful circuit you cannot interpret is an honest mystery — causally confirmed but mechanistically opaque. A beautifully interpreted circuit that fails faithfulness tests is a just-so story — narratively compelling but empirically unsupported. The field often treats "faithful and interpretable" as a single achievement, but you can have either without the other. A paper that reports high faithfulness has established causal relevance. It has not established understanding unless it also explains why the circuit works, what it computes, and what would break it.

### Component identity vs component role

"Head 9.9 is in the IOI circuit" (identity) and "Head 9.9 is a name-mover" (role) are different claims requiring different evidence. Identity requires only causal evidence — ablating the component changes the output. Role requires mechanistic evidence — the weight structure matches the claimed function, the component doesn't perform this function on non-target inputs, and the role label makes predictions that can be tested independently.

In MI: the slide from identity to role happens in one sentence in most papers and is a common overclaim. A head identified by activation patching receives a functional label ("name-mover," "induction head," "inhibition head") that implies a richer mechanistic story than the patching result establishes. The label is a hypothesis about role, not a conclusion from identity. Testing the label requires additional evidence: does the W_OV structure match? Does the head perform this function and not others? Would a different head with the same structural properties also earn the label?

### Activation evidence vs weight evidence

"We observed this during inference" and "this is structurally encoded in the weights" are different epistemic types that are often treated interchangeably. Activation-based evidence is dynamic and context-dependent — it tells you what happens on specific inputs. Weight-based evidence is static and context-free — it tells you what is structurally possible regardless of input.

In MI: activations and weights can disagree. Weights may encode capabilities the model never uses in practice — structural potential that is never activated on naturalistic inputs. Activations may show computations that are not legible from weight inspection alone — distributed or superposed representations that no single weight matrix reveals. A claim supported by both activation and weight evidence is qualitatively stronger than one supported by either alone, because the two evidence types are sensitive to different failure modes. Activation-only claims miss structural context; weight-only claims miss dynamic behavior.

## Analytical Constructs

MI's analytical toolkit consists of three complementary constructs, each probing a different validity dimension.

### The evidence convergence map

A triangular diagram with three vertices — Computational, Algorithmic, Implementational (Marr's three levels) — and arrows from each pointing toward a central interpretation node. For each circuit claim, color the arrows: green (evidence exists and converges), red (evidence contradicts), gray (not yet tested).

The map answers: do all three levels of description point to the same interpretation?

For the IOI circuit:
- **Implementational → Interpretation**: Strong (green). Ablation and patching identify specific heads; weight-space signatures confirm structural roles.
- **Algorithmic → Interpretation**: Moderate (yellow-green). "Detect-inhibit-copy" is a specified multi-step process, but whether it is the *unique* algorithm or one of several compatible algorithms is not established.
- **Computational → Interpretation**: Weak (gray). We know the circuit performs IOI, but whether IOI is the right computational description — or whether it is a special case of "contextual coreference resolution" — is untested.

The map makes visible exactly where the interpretive story needs more evidence.

### The intervention-interpretation matrix

A grid with intervention types on rows and claim types on columns. Each cell records whether evidence of that intervention type exists supporting (or contradicting) that claim type.

**Rows (intervention types):**
- Ablation (mean, resample, zero)
- Activation patching (clean → corrupt, corrupt → clean)
- Interchange intervention (IIA / DAS)
- Activation addition (steering vectors)
- Weight-space analysis (no intervention — structural)
- Naturalistic observation (no intervention — correlational)

**Columns (claim types):**
- Necessity ("this component is required")
- Sufficiency ("this component is enough")
- Representational ("this component encodes X")
- Algorithmic ("this component computes procedure Y")
- Computational ("the model solves problem Z via this mechanism")

Each cell: ✓ (evidence confirms), ✗ (evidence contradicts), ? (not tested), or ∅ (this evidence type cannot in principle support this claim type).

The ∅ cells are the key insight. Ablation *cannot* support a representational claim (it tests necessity, not what is represented). Naturalistic observation *cannot* support a causal claim. Unconstrained nonlinear IIA *cannot* support a representational claim (Sutter et al. 2025). These are not missing experiments — they are category errors. The matrix makes them visible.

A common gap: having only cells filled in the ablation row but making claims in the algorithmic or computational columns — where ablation cells are structurally ∅.

### The causal sufficiency graph

A directed graph where nodes are circuit components and edges are information-flow paths, with edge style encoding evidence strength:

- **Solid line** — IIA-confirmed or path-patching-confirmed causal connection
- **Dashed line** — correlation or single-method patching only
- **Double line** — confirmed by multiple independent methods (double dissociation equivalent)
- **Red ✗** — tested and found NOT causal (the edge was hypothesized but patching shows no information flow)

The central question: does the solid-edge subgraph form a connected path from input to output for the target task?

- **Connected solid path exists**: The causal chain is established. Each link has been independently confirmed. This is internal validity at its strongest.
- **Gaps in the solid path**: We know the start and end components but haven't confirmed the intermediate connections. The algorithm is hypothesized but the wiring is incomplete.
- **Only dashed edges**: The circuit is associational — components are identified but causal connections between them are not established.

For the IOI circuit: nodes are the S-inhibition heads, name-mover heads, and backup name-movers. Edges from S-inhibition → name-movers are solid (path patching confirms information flow). Edges to backup heads are dashed (we know they activate compensatorily but the causal pathway is less precisely characterized). The red ✗ edges are the tested-and-rejected hypotheses — equally informative for understanding what the circuit is NOT.

A random-initialization model should produce an empty graph (no solid edges). If solid edges appear in a random model, the instrument is unreliable. This provides a natural baseline comparison.

## Sources

| Source | Year | Concept imported |
|---|---|---|
| [Marr, *Vision*](https://doi.org/10.7551/mitpress/9780262514620.001.0001) | 1982 | **Three levels of description** — implementational, algorithmic, computational; evidence at a lower level does not license a claim at a higher level without additional evidence specific to the higher level |
| [Olsson et al.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 2022 | **Nomological network for circuits** — induction heads earn their name by connecting to attention patterns, compositional mechanisms, behavioral predictions, and training dynamics predictions, each independently testable |
| [McDougall et al.](https://arxiv.org/abs/2310.04625) | NeurIPS 2023 | **Coverage quantification** — L10H7's impact on 76.9% of a curated distribution; the unreported 23.1% is the scope of the claim, not a footnote |
| [Sutter et al.](https://arxiv.org/abs/2507.08802) | NeurIPS 2025 | **Level declaration** — unconstrained nonlinear IIA achieves near-perfect scores on random-init models; high IIA is an implementational measurement, not a representational one, unless the map architecture is constrained |
| [Wang et al.](https://arxiv.org/abs/2211.00593) | 2022 | **Circuit non-uniqueness** — activation patching returns "a" circuit, not "the" circuit; the circuit is relative to the procedure, the prompt distribution, and the ablation method |
| [Conmy et al.](https://arxiv.org/abs/2304.14997) | NeurIPS 2023 | **Automated circuit discovery** — ACDC and EAP find circuits by a specific procedure; different procedures find different but comparably faithful circuits |

## Validity type: [Interpretive validity](/framework/validity-types/interpretive)

> **Level assignment:** For a claim $\mathcal{C}$, define $L(\mathcal{C}) \in \{I, A, C\}$ where $I$ = implementational (names components, weights, or activations), $A$ = algorithmic (names a procedure), $C$ = computational (names a problem and asserts a solution). Interpretive validity requires $L(\mathcal{E}) \geq L(\mathcal{C})$: evidence at a lower level does not license a claim at a higher level.

The reason a level framework is useful in MI is that the field has a systematic upward drift tendency. An activation patching result (implementational) produces a list of heads. The paper names the heads by their apparent function: "name-mover heads," "S-inhibition heads." A name is already an algorithmic claim — it asserts a procedure. By the next paragraph, the paper often says the model "computes indirect object identification" through the circuit, which is a computational claim. The drift from $I$ to $A$ to $C$ happens in three sentences without the additional evidence each upgrade requires.

Identifying the drift is not a refutation of the result. The implementational finding may be real, well-supported, and significant. The issue is the framing: algorithmic and computational language creates interpretability claims that go beyond what the evidence establishes, and those claims are what get cited, extended, and built upon.

We extend Marr's three levels with four implementational sub-modes to capture the actual range of MI claims: (1) **topographic** — which components are involved; (2) **connectomic** — how they are wired; (3) **activation-statistical** — what firing patterns look like; (4) **functional** — what transformation a component performs. Each sub-mode requires its own characteristic evidence.

## The criteria

### Level declaration

Every principal quantitative claim must name the level at which it is stated before the evidence is collected.

An IIA score is an implementational-representational measurement: it tells us that some transformation of activations at a specific location tracks a causal variable. "The model *computes* a function" is a computational claim. "The model *represents* the variable linearly at that location" is a representational claim, which linear IIA can support because the linearity constraint on the alignment map is itself representational evidence. "The activation at that location *tracks* the causal variable" is an implementational claim that any IIA score, linear or not, supports.

[Sutter et al. (NeurIPS 2025)](https://arxiv.org/abs/2501.07615) show why this matters precisely. Unconstrained nonlinear IIA achieves near-perfect scores on random-initialization models. The score is a correct measurement — there does exist a nonlinear transformation that maps the random activations onto the target variable. But the claim "the model encodes X" is a representational claim that the score does not support, because the same score would appear in a model with no encoding at all. The level mismatch is the problem, not the score.

**What to declare.** The level ($I$, $A$, or $C$, and within implementational, the sub-mode) of each principal quantitative claim. If the planned narrative will use a higher-level sentence than the evidence supports, the additional evidence required for the upgrade should be named before the experiment.

![Description Levels and Evidence Requirements — Marr's three levels with implementational sub-modes and description creep arrow](/figures/marr_levels.svg)

### Circuit non-uniqueness

Activation patching returns *a* circuit that is faithful to a behavior under a specific procedure on a specific prompt distribution. It does not return *the* circuit.

This is not a limitation of current methods — it is a mathematical fact about the search problem. Given that multiple circuits can achieve comparable faithfulness on any finite prompt distribution, any single procedure returns one member of a set of candidates. Whether that member is "the" circuit depends on whether the others are somehow ruled out, which requires evidence beyond the single-procedure result.

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) found the IOI circuit using activation patching. [Conmy et al. (2023)](https://arxiv.org/abs/2304.14997) re-found a comparable circuit using ACDC, which operates by a different algorithmic principle. The two circuits overlap substantially at the component level, which is convergent evidence that their shared components are genuinely important. The components in one circuit but not the other are the locus of the remaining uncertainty.

The recommended language is "a circuit faithful to $B$ under procedure $P$" rather than "the circuit for $B$," unless at least two procedures with different assumptions have been applied and their overlap characterized. A single-procedure result is a valid starting point. It becomes a more specific claim when a second procedure confirms it.

**What to report.** The discovery procedure named. The circuit described as "a circuit faithful to $B$ under $P$." If multi-procedure agreement has been tested, the Jaccard similarity at the component level $J(C_A, C_B) = |C_A \cap C_B| / |C_A \cup C_B|$.

<details class="worked-example">
<summary>Worked example: weight-based circuit signatures vs. activation patching</summary>

Consider GPT-2 Small. We identify a set of weight-space directions (via SVD of the OV matrices) that load heavily on layer 8 and layer 9, with OV cosine similarity to the IOI name-mover head signatures above 0.7. This is a structural (connectomic) finding: the weight-space geometry resembles the known IOI circuit topology.

We also run activation patching on the same model, using the Wang et al. IOI prompt set. This returns a set of components with high patching attribution at layers 3, 8, and 9, broadly overlapping with the weight-space finding.

The Jaccard similarity between the two circuits (weight-based vs. activation-patching) is $J = 0.61$. This is substantial convergent evidence: two methods with genuinely different assumptions — one purely weight-space, one activation-based — agree on the majority of components. The 39% disagreement is not a contradiction; it characterizes the current precision of the comparison. We report the circuit as "identified by both weight-signature analysis and activation patching ($J = 0.61$); the shared components are the more robust part of the claim."

If we had reported only the weight-space result, the Jaccard of 0.61 is potential future evidence. If we reported only the activation patching result, the weight-space structure would be uncharacterized. Reporting both gives a more informative picture than either alone.
</details>

### Level-evidence separation

After analysis, audit every sentence in the narrative against the level of the evidence supporting it. A sentence that uses algorithmic or computational language must be traced to algorithmic or computational evidence.

The most common failure mode is not deliberate overclaiming — it is drift. A paper discovers a set of heads (implementational), names them by apparent function (sliding into algorithmic), and then describes the model's behavior as "computing" or "implementing" that function (reaching computational). Each step feels natural. The aggregate drift is substantial.

The partial evidence table provides reference for commonly seen patterns:

| Pattern | Level mismatch | Recommended language |
|---|---|---|
| IIA reported; "model encodes X" claimed | $L(\mathcal{E}) = I$, $L(\mathcal{C}) = R$ | "IIA $= S$ at location $L$ (implementational); representational claim requires a linear map constraint" |
| Activation patching; "model computes X" claimed | $L(\mathcal{E}) = I$, $L(\mathcal{C}) = C$ | "Components $C$ are causally involved in behavior $B$ (implementational)" |
| Single-procedure circuit; called "the circuit" | Non-uniqueness untested | "A circuit faithful to $B$ under procedure $P$" |
| Faithfulness reported without ablation method | Method-dependent number | "Faithfulness $= F\%$ under [method] on [distribution]" |
| "Main role" claimed with $\kappa < 0.9$ | Scope overclaim | "On the tested distribution ($\kappa = X$), consistent with role $R$" |
| Single-model mechanism claim | Architecture-specific | "In [model family], a mechanism consistent with…" |

### Coverage quantification and scope honesty

Claims should be stated at the scope the evidence licenses, not the scope the narrative implies.

[McDougall et al. (NeurIPS 2023)](https://arxiv.org/abs/2310.04625) measured head L10H7's impact on 76.9% of a curated prompt distribution and described its "main role" as copy suppression. The 76.9% is an implementational measurement of impact on a specific distribution. The "main role" language implies something about the full training distribution. The 23.1% of cases where copy suppression is not the operating mechanism are uncharacterized.

We define the coverage ratio:

$$\kappa = \frac{N_{\text{covered}}}{N_{\text{total}}}$$

where $N_{\text{covered}}$ is the number of evaluated instances for which the claim holds and $N_{\text{total}}$ is the total evaluation size. Scope-claiming language — "main role," "primary function," "the mechanism" — is warranted when $\kappa > 0.9$ on a representative distribution. Below that, the recommended language is explicit: "On the tested distribution ($\kappa = 0.77$), L10H7's activation is consistent with copy suppression. The remaining 23% are not characterized."

This is not pessimism. A finding with $\kappa = 0.77$ on a well-designed distribution is a strong result — it means the mechanism operates in the large majority of tested cases. Stating the coverage makes the strength visible without claiming into the untested fraction.

**What to report.** $\kappa$ for the primary claim. If $\kappa < 0.9$, the uncharacterized fraction acknowledged rather than subsumed into a scope-claiming summary.

### Cross-architecture evidence requirement

Claims about mechanisms — as opposed to claims about components in a specific model — should include evidence from at least one other model family, or be explicitly bounded to the tested family.

A mechanism is an algorithmic or computational claim: it asserts that a procedure or computation is present in the model. A mechanism found only in GPT-2 Small may be a property of GPT-2's training trajectory, initialization, or architecture, not a property of language models in general. [Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) strengthened the induction head claim precisely by showing that induction heads appear across model families, making the claim about a class of models rather than a single instance.

Cross-architecture evidence does not require the same circuit. It requires an analogous mechanism — performing the same function through possibly different components — with a matching criterion stated before testing.

**What to report.** If cross-architecture evidence is absent, the claim bounded to the tested family: "in GPT-2 Small, a circuit consistent with IOI…" If cross-architecture evidence is present, the matching criterion stated and the results reported for each tested family.

## Verdicts

- **Proposed → Causally suggestive:** Requires I1 (level declaration at the implementational level). A claim stated at the algorithmic or computational level without level declaration cannot be upgraded.
- **Causally suggestive → Mechanistically supported:** Requires I3 (level-evidence separation audited) and I4 (coverage $\kappa$ reported). A finding where $\kappa$ is high and no upward drift is present is ready for this upgrade.
- **Mechanistically supported → Triangulated:** Requires I2 (non-uniqueness addressed via at least two procedures) and, ideally, I5 (cross-architecture evidence or explicit bound).
- **Triangulated → Validated:** Requires all five criteria met and $\kappa > 0.9$ on a representative distribution.

## Protocol

For any reported finding:

1. **Level declaration.** Label every principal quantitative claim with its level before collecting evidence. If the narrative will use higher-level language, name the additional evidence required for the upgrade.
2. **Circuit non-uniqueness.** Name the discovery procedure. Report the circuit as "a circuit faithful to $B$ under procedure $P$." Apply at least two procedures with different assumptions if uniqueness is central to the claim.
3. **Level-evidence separation.** After analysis, audit every sentence in the narrative. Flag upward drift from $I$ to $A$ or $A$ to $C$. Downgrade or add evidence.
4. **Coverage quantification.** Compute $\kappa$. State the scope of the claim at the level $\kappa$ licenses.
5. **Cross-architecture evidence.** State a matching criterion. Report findings in at least one other model family, or explicitly bound the claim.

A skipped step must be named in the verdict.

## Case studies

For full worked examples applying all five lenses (including interpretive validity) to published claims:

- [IOI Circuit](/framework/lenses_v6/examples/examples-ioi) — "the circuit" language for a single-procedure result
- [Induction Heads](/framework/lenses_v6/examples/examples-induction-heads) — level declarations match evidence; full nomological network
- [Docstring Circuit](/framework/lenses_v6/examples/examples-docstring) — label risk: "variable binding" vs. simpler "positional copying"
- [Othello World Model](/framework/lenses_v6/examples/examples-othello) — interpretive inflation: "world model" exceeds evidence
- [Copy Suppression](/framework/lenses_v6/examples/examples-copy-suppression) — coverage quantification ($\kappa = 0.77$) done explicitly
- [SAE Features](/framework/lenses_v6/examples/examples-sae-features) — level mismatch: implementational features, computational claims
- [Gender Bias](/framework/lenses_v6/examples/examples-gender-bias) — scope honesty failure: "bias circuit" implies separability
