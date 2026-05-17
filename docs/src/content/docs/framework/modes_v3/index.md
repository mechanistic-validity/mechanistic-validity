---
title: "Description Modes"
description: "Seven levels at which a mechanistic claim can be stated — each a commitment about the kind of explanation being offered."
---

# Description Modes

A description mode is a commitment about **what kind of explanation you are offering**. It is not a label applied after analysis — it is a constraint declared before analysis that determines what counts as evidence, what counts as a gap, and what the finding means if it succeeds.

The distinction matters because mechanistic interpretability routinely conflates levels. A paper that discovers *which components* are active (implementational) writes a narrative about *what the model computes* (computational). Each level upgrade requires additional evidence that the lower level cannot provide. The modes framework makes these upgrades explicit: you declare a mode, you collect evidence appropriate to that mode, and the verdict is issued at that mode. If you want to claim at a higher mode, you state what additional evidence the upgrade requires.

## Intellectual origins

The seven-mode system extends Marr's (1982) tripartite distinction and incorporates subsequent critiques.

| Source | Year | Contribution |
|---|---|---|
| [Marr, *Vision*](https://doi.org/10.7551/mitpress/9780262514620.001.0001) | 1982 | Three-level framework: computational (what and why), algorithmic (how, as a procedure), implementational (physical substrate). The levels are defined by *independence*: a computational-level description constrains but does not determine the algorithm; an algorithm constrains but does not determine the implementation. |
| [Pylyshyn, *Computation and Cognition*](https://doi.org/10.7551/mitpress/2004.001.0001) | 1984 | Added the distinction between *functional architecture* (the fixed substrate that constrains algorithms) and *representations* (the content that algorithms manipulate). A description of what is represented is not the same as a description of how it is processed. |
| [Dennett, *The Intentional Stance*](https://doi.org/10.7551/mitpress/3820.001.0001) | 1987 | The stance hierarchy (physical, design, intentional) — each stance yields valid predictions, but predictions at one stance do not constitute evidence at another. A system accurately described from the intentional stance may have no internal structure corresponding to the attributed beliefs. |
| [Craver, *Explaining the Brain*](https://doi.org/10.1093/acprof:oso/9780199299317.001.0001) | 2007 | Mechanistic explanation requires specifying *entities* (what), *activities* (how they produce changes), and *organization* (spatial, temporal, and causal ordering). A list of entities is not a mechanism. A mechanism requires showing how the entities' activities, organized in a specific way, produce the phenomenon. |
| [Bechtel & Richardson, *Discovering Complexity*](https://doi.org/10.7551/mitpress/8328.001.0001) | 2010 | Decomposition (identifying parts) and localization (assigning functions to parts) are distinct strategies, each with characteristic failure modes. Localization fails when functions are distributed; decomposition fails when parts interact nonlinearly. |
| [Kaplan & Craver](https://doi.org/10.1093/bjps/axr010) | 2011 | The "model-to-mechanism mapping" (3M) constraint: a model explains a phenomenon only if (a) the variables in the model correspond to identifiable components or activities in the mechanism, and (b) the dependencies in the model correspond to causal relations in the mechanism. Purely predictive models — even perfectly accurate ones — do not explain unless 3M is satisfied. |

## Formal structure

Define a claim $\mathcal{C}$ as a triple $(\phi, M, \mathcal{E})$ where $\phi$ is the phenomenon (a behavioral regularity), $M$ is the proposed explanation (the model or description), and $\mathcal{E}$ is the evidence. A mode assignment $\mu(\mathcal{C}) \in \{C, A, R, I_{\text{top}}, I_{\text{con}}, I_{\text{stat}}, I_{\text{fun}}\}$ is a declaration of the *kind* of explanation $M$ purports to be.

The modes form a partial order by *explanatory commitment*:

$$C > A > R > I_{\text{fun}} > I_{\text{con}} > I_{\text{top}}$$

where $>$ means "makes strictly more commitments about the target system." A computational claim commits to a function being computed and a reason it is adaptive. An algorithmic claim commits to a specific procedure. A representational claim commits to what information is encoded but not how it is processed. An implementational claim commits only to physical/structural facts about the substrate.

The independence principle: evidence at mode $\mu_1$ does not constitute evidence at mode $\mu_2 > \mu_1$ without additional bridging evidence specific to $\mu_2$. This is not a heuristic — it follows from the logical structure of the claims. An implementational finding (these heads are active) is consistent with multiple algorithmic accounts; selecting among them requires algorithmic-level evidence (demonstrating the procedure the components jointly execute).

## The seven modes

### 1. Computational

**Question:** What function does the system compute and why is it the right function?

**Commitment:** The explanation specifies an input-output mapping $f: X \to Y$ and a normative account of why $f$ is the correct or optimal solution to an environmental problem. Marr's example: early vision computes edge detection because edges correspond to depth discontinuities, object boundaries, and illumination changes — the explanation is why edges are the *right* thing to compute.

**In MI:** A computational-mode claim says the model solves a specific problem (indirect object identification, ordinal comparison, in-context sequence completion) and that this solution is *the* solution to an identifiable subproblem of language modeling. The claim is not just that the model produces the right outputs — any sufficiently trained model does that. The claim is that the model's internal organization reflects the structure of the problem in a way that makes the solution intelligible.

**Evidence required:**
- Specification of the input-output function $f$ with domain and range characterized
- Error analysis showing that the model's failure modes correspond to cases where the problem specification is genuinely ambiguous or ill-defined (not arbitrary failures)
- Edge-case behavior consistent with the normative theory: if the theory predicts graceful degradation in a specific regime, the model should degrade gracefully there and not elsewhere
- Ideally, a formal optimality argument: the model's solution is in some sense efficient or rational given its constraints

**Characteristic failure:** Naming a function without providing the normative account. "The model computes IOI" is not a computational-mode claim unless accompanied by an account of why IOI is a well-posed subproblem and what structural properties of language make it separable.

<details class="worked-example">
<summary>Worked example: IOI as computational vs. algorithmic</summary>

Wang et al. (2022) identify a circuit for indirect object identification. Is the claim computational or algorithmic?

**Computational version:** "The model solves the coreference resolution subproblem of identifying which named entity fills the indirect object role, treating this as a constraint-satisfaction problem over syntactic roles and entity mentions. This is a well-posed subproblem because indirect objects in English are systematically predictable from the verb's argument structure and prior entity mentions."

**Algorithmic version:** "The model identifies the indirect object through a specific procedure: duplicate-token heads mark repeated names, S-inhibition heads suppress the subject, and name-mover heads copy the remaining name to the output position."

The first makes a commitment about *what problem is being solved and why*. The second makes a commitment about *how it is solved step by step*. The evidence for the second (activation patching, path patching) does not establish the first — the first requires additionally showing that the problem decomposition is correct (that IOI is genuinely separable from broader coreference) and that the solution structure reflects the problem structure (not just that it produces correct outputs).

Most published IOI work operates at the algorithmic level, using computational-level language.
</details>

### 2. Algorithmic

**Question:** What procedure does the system execute to produce the output?

**Commitment:** The explanation specifies a sequence of operations, their ordering, and how intermediate representations flow between them. The algorithm is stated with enough precision that it could be re-implemented — it is a description of a procedure, not just a naming of components.

**In MI:** An algorithmic claim says the model follows a specific step-by-step procedure: "first, previous-token heads copy position information backward; then, induction heads use the Q-K composition to attend to the token following the previous occurrence; then, the OV circuit copies the attended token to the output." This is more than a list of components — it specifies the *order of operations* and the *information flow* between steps.

**Evidence required:**
- Path-level causal tracing demonstrating the claimed information flow (path patching, not just activation patching)
- Timing consistency: if the algorithm claims step A feeds step B, interventions on A at the correct position should affect B's output and interventions at other positions should not
- The algorithm must be *sufficient* in the sense of Craver (2007): executing the claimed procedure on the claimed inputs must produce the claimed outputs. This is tested by constructing the minimal circuit and verifying it reproduces the behavior without the rest of the model.
- Ideally, prediction of novel intermediate states: if the algorithm passes through a specific intermediate representation at step $k$, that representation should be detectable at the corresponding hook point

**Characteristic failure:** Naming components and calling their sequential activation an "algorithm." An algorithm must specify *what operation each step performs on its input to produce its output*, not just *which components are active in which order*.

### 3. Representational

**Question:** What information does the system encode, in what format, and where?

**Commitment:** The explanation specifies *content* — what variables or features are represented — and *format* — how that content is encoded (linearly, as a subspace, distributed across neurons, etc.). A representational claim does not commit to how the representation is *used* — that would be algorithmic. It commits only to what is *there*.

**In MI:** A representational claim says "the model linearly encodes syntactic number at layer $L$, position $p$, in a direction $\hat{v}$ with separation $d' > 2$ between singular and plural contexts." This is a specific, falsifiable claim about what information is decodable, where, and in what geometric form.

**Formal characterization:** Let $h_l^p \in \mathbb{R}^{d_{\text{model}}}$ be the residual-stream activation at layer $l$, position $p$. A linear representational claim asserts the existence of a direction $\hat{v}$ such that $\langle h_l^p, \hat{v} \rangle$ tracks a causal variable $Z$ with $d' = (\mu_+ - \mu_-) / \sigma_{\text{pooled}} > \tau$ for some threshold $\tau$.

A *causal* representational claim (stronger) additionally requires that intervening on $h_l^p$ along $\hat{v}$ changes the model's behavior in a manner consistent with changing $Z$ — i.e., the representation is not merely decodable but *used*. This is the distinction between DAS/IIA (which demonstrates causal relevance of a direction) and probing (which demonstrates only decodability).

**Evidence required:**
- The geometric form stated explicitly (linear direction, subspace, distributed)
- Decodability demonstrated with appropriate baselines (Hewitt & Liang 2019 selectivity for probes; random-vector and untrained-model baselines for IIA)
- If the claim is *causal* representational: interchange intervention (IIA/DAS) demonstrating that the representation is load-bearing, not merely present
- The alignment map's capacity stated and varied (Sutter et al. 2025): if IIA remains high only with unconstrained nonlinear maps, the finding is about map flexibility, not linear geometry

**Characteristic failure:** Reporting probe accuracy without a control task. Reporting IIA without stating or varying the alignment map architecture. Conflating "decodable" with "encoded" — the first is a measurement, the second is a representational claim requiring causal evidence.

### 4. Implementational (topographic)

**Question:** Which components are involved?

**Commitment:** The explanation identifies a set of components (heads, neurons, layers, SAE features) that participate in producing the behavior. No commitment to what they do, how they are connected, or what procedure they execute.

**In MI:** "Heads L5H1, L5H5, L6H9, L7H3, L7H10, L8H6, L8H10, L8H11, L9H6, and L9H9 are causally involved in IOI behavior." This is a topographic claim — a map of *where* the computation happens without explaining *what happens there*.

**Evidence required:**
- Causal evidence of involvement: ablation, activation patching, or attribution demonstrating that removing or corrupting these components changes the target behavior
- Specificity relative to a size-matched random set: the claimed components should have more impact than randomly selected components of the same type and count
- The discovery procedure named, since different procedures can return different component sets for the same behavior (Conmy et al. 2023)

**Characteristic failure:** Treating a topographic finding as if it were algorithmic. "We identified the IOI circuit" (topographic) vs. "we identified how the model performs IOI" (algorithmic) — the first is a map, the second is an explanation.

### 5. Implementational (connectomic)

**Question:** How are the components wired to each other?

**Commitment:** The explanation identifies directed connections between components — which feeds into which, through what pathway. A connection claim is stronger than a topographic claim because it asserts *structure* (a graph) rather than just *membership* (a set).

**In MI:** "Head L9H9 receives input from L7H3 via the residual stream; the QK composition between L7H3's output and L9H9's query vectors produces the attention pattern that targets the indirect object." This is a wiring claim.

**Evidence required:**
- Path-level causal evidence (path patching, edge attribution) demonstrating that the claimed pathway is load-bearing
- The connection should be *specific*: patching along the claimed path should have substantially more effect than patching along alternative paths of the same length
- Weight-space evidence (QK/OV composition scores, virtual weight analysis) provides convergent structural evidence when available

### 6. Implementational (activation-statistical)

**Question:** What are the distributional properties of activations at these components?

**Commitment:** The explanation characterizes *what the activations look like* — their distribution, sparsity, clustering, or geometric properties — without committing to what computation produces them or what they represent.

**In MI:** "SAE feature 1247 has mean activation 0.3 across the corpus, fires above threshold on 12% of tokens, has a bimodal activation distribution with peaks at $-0.5$ and $+1.2$, and its top-activating contexts are predominantly tokens following opening parentheses." This is a statistical characterization of a component's behavior.

**Evidence required:**
- Summary statistics (mean, variance, sparsity, distribution shape) computed on a representative corpus
- The corpus and its coverage stated — statistics on the Pile-10k are statistics on the Pile-10k, not "the component's behavior in general"
- Stability: the statistics should be consistent across random subsamples of the corpus (bootstrap CI reported)

### 7. Implementational (functional)

**Question:** What input-output transformation does this specific component perform?

**Commitment:** The explanation specifies what a single component does to its input to produce its output — the *function* it implements, at the component level rather than the system level. This is the strongest implementational sub-mode: it commits to *what the part does*, not just *where it is* or *what it's connected to* or *what its activations look like*.

**In MI:** "Head L10H7 implements a linear suppression function: for token $t$ at position $p$, if $t$ has high probability in the output distribution before this head acts, the head subtracts a vector proportional to $t$'s unembedding direction from the residual stream, reducing $t$'s logit by approximately $\Delta \propto -\text{attn}(p, q) \cdot \langle W_{OV} h_q, W_U[t] \rangle$." This specifies the input-output function of a single component.

**Evidence required:**
- The function stated with enough precision to generate quantitative predictions
- Predictions tested: the function's predicted output compared to the component's actual output on held-out inputs
- The function should be *necessary*: replacing the component with its claimed function should preserve the behavior (sufficiency), and the claimed function should not be achievable by a simpler description (minimality)

## Mode as a constraint on interpretation

The mode is not a post-hoc label — it is a pre-analysis commitment that constrains what the analysis means. Before running an experiment, a researcher should state: "This experiment will provide evidence at mode $\mu$. If it succeeds, the claim is: [specific claim at mode $\mu$]. The evidence required is: [specific list]. If I wish to make a claim at mode $\mu' > \mu$, I will additionally need: [specific bridging evidence]."

This discipline prevents the drift that characterizes much current MI writing, where implementational experiments are narrated in computational language. The drift is not always wrong — sometimes algorithmic language is warranted by implementational evidence when the implementation is sufficiently constrained. But the warrant must be made explicit.

## The upgrade problem

Moving from a lower mode to a higher mode requires *bridging evidence* — evidence that is specific to the higher mode and cannot be derived from the lower mode alone.

| Upgrade | Bridging evidence required |
|---|---|
| $I_{\text{top}} \to I_{\text{con}}$ | Path-level causal evidence of directed connections |
| $I_{\text{con}} \to I_{\text{fun}}$ | Input-output function of each component, tested on held-out inputs |
| $I_{\text{fun}} \to R$ | Demonstration that the component's function is *representational* — encoding a specific variable — rather than just performing a mathematical operation |
| $R \to A$ | Ordering, composition, and sufficiency: the representations flow through a specific procedure that jointly produces the behavior |
| $A \to C$ | Normative account: why this algorithm solves the right problem, with error analysis showing failures correspond to problem boundaries |

Each upgrade is independently falsifiable. A finding can be strong at $I_{\text{con}}$ but fail the $I_{\text{con}} \to I_{\text{fun}}$ upgrade — the wiring is established but the function of each component is not. This is informative, not a failure. The mode tells you what you have.

## Decision procedure

Given a completed analysis, assign the mode by asking:

1. Did you identify which components are involved? → $I_{\text{top}}$ minimum
2. Did you establish directed connections between them? → $I_{\text{con}}$ minimum
3. Did you characterize the input-output function of individual components? → $I_{\text{fun}}$ minimum
4. Did you demonstrate what information is encoded and in what format? → $R$ minimum
5. Did you demonstrate a step-by-step procedure with ordering and information flow? → $A$ minimum
6. Did you provide a normative account of what problem is solved and why? → $C$ minimum

The mode is the highest level for which all required evidence is present. Partial evidence at a higher level does not upgrade the mode — it is reported as "suggestive of $\mu'$" while the verdict is issued at the established mode.
