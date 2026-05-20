---
title: "Computational Mode"
description: "What function does the system compute and why is it the right function? — Marr's highest level of explanation."
---


# Computational Mode

| | |
|---|---|
| Origin | [Marr (1982)](https://doi.org/10.7551/mitpress/9780262514620.001.0001), Level 1 |
| Question | What function does the system compute, and why is it the *right* function? |
| Licensing evidence | Task specification + normative account + behavioral coverage across the full variation space |
| Interpretive-validity risk | Level inflation — claiming computational understanding when the evidence licenses only algorithmic or implementational |
| Position in partial order | $C > A > R > I_{\text{fun}} > I_{\text{con}} > I_{\text{top}}$ — highest commitment |

## What this mode claims

A verdict tagged `[computational]` specifies an input-output mapping $f: \mathcal{X} \to \mathcal{Y}$ *and* a normative account of why $f$ is the correct or optimal solution to an environmental problem. Marr's canonical example: early vision computes edge detection because edges correspond to depth discontinuities, object boundaries, and illumination changes. The explanation is why edges are the *right* thing to compute — not just that the system computes them.

In MI, a computational-mode claim says the model solves a specific problem (indirect object identification, ordinal comparison, in-context sequence completion) and that this solution is *the* solution to an identifiable subproblem of language modeling. The claim is not just that the model produces the right outputs — any sufficiently trained model does that. The claim is that the model's internal organization reflects the structure of the problem in a way that makes the solution intelligible.

## Formal characterization

Let $\mathcal{T}$ denote a task (a behavioral regularity in the data). A computational-mode claim asserts:

1. There exists a function $f_C: \mathcal{X} \to \mathcal{Y}$ that characterizes the task
2. The circuit $C$ approximates $f_C$ across the relevant variation: $\Pr_{x \sim \mathcal{D}_{\text{test}}}[m(C, x) = f_C(x)] \geq 1 - \epsilon$
3. $f_C$ is the *right* function to compute given the structure of language — there is a normative account of why this subproblem is separable and why the model should solve it

Condition (3) is what distinguishes computational from algorithmic. An algorithm can be stated without justifying *why* it solves the right problem.

## What licenses a `[computational]` tag

Three requirements, all mandatory:

1. **Specification of $f$** with domain and range characterized — precise enough that a reader could build a lookup table implementing it without knowing transformers exist
2. **Normative account** — why this function is the correct solution to a genuine subproblem of language modeling. What structural properties of language make the problem separable?
3. **Error analysis** — the model's failure modes correspond to cases where the problem specification is genuinely ambiguous or ill-defined, not arbitrary failures. Edge-case behavior consistent with the normative theory.

Optionally: a formal optimality argument showing the model's solution is efficient or rational given its constraints.

## What does NOT license a `[computational]` tag

- **Naming a function without the normative account.** "The model computes IOI" is not computational unless accompanied by an account of why IOI is a well-posed subproblem and what structural properties of language make it separable from broader coreference.
- **High faithfulness on one prompt distribution.** Faithfulness is a behavioral metric on tested prompts, not a task-level characterization.
- **Showing ablation degrades performance.** That is implementational (locus of causal necessity). It does not establish what problem the circuit solves.
- **Teleological inflation.** "The model *needs* this circuit to perform the task" — need is implementational, not computational.

<details class="worked-example">
<summary>Worked example: IOI as computational vs. algorithmic</summary>

[Wang et al. (2022)](https://arxiv.org/abs/2211.00593) identify a circuit for indirect object identification. Is the claim computational or algorithmic?

**Computational version:** "The model solves the coreference resolution subproblem of identifying which named entity fills the indirect object role, treating this as a constraint-satisfaction problem over syntactic roles and entity mentions. This is a well-posed subproblem because indirect objects in English are systematically predictable from the verb's argument structure and prior entity mentions."

**Algorithmic version:** "The model identifies the indirect object through a specific procedure: duplicate-token heads mark repeated names, S-inhibition heads suppress the subject, and name-mover heads copy the remaining name to the output position."

The first makes a commitment about *what problem is being solved and why*. The second makes a commitment about *how it is solved step by step*. The evidence for the second (activation patching, path patching) does not establish the first — the first requires additionally showing that the problem decomposition is correct (that IOI is genuinely separable from broader coreference) and that the solution structure reflects the problem structure.

Most published IOI work operates at the algorithmic level, using computational-level language.
</details>

<details class="worked-example">
<summary>Worked example: Greater-Than as a valid computational claim</summary>

[Hanna et al. (2023)](https://arxiv.org/abs/2305.00586) characterize the Greater-Than circuit:

**Task specification.** $f_{\text{GT}}(\text{"The war lasted from } y_1 \text{ to } y_2\text{"}) = \text{valid iff } y_2 > y_1$

**Normative account.** Temporal ordering is a genuine subproblem of language modeling because English systematically constrains year sequences in "from X to Y" constructions. The model should solve this because violating temporal order would produce low-probability continuations across a wide class of natural text.

**Behavioral coverage.** Tested across 11 sentence frames and year pairs spanning 1000-2000, with accuracy $\geq 0.92$ on all frames including those not used during discovery.

**Error analysis.** Failures cluster at the boundary ($y_2 \approx y_1$) where the problem is genuinely ambiguous, consistent with the normative theory.

This satisfies all three requirements: specification, normative account, and error analysis.
</details>

## Upgrade and downgrade

| Direction | What's required |
|---|---|
| $A \to C$ (upgrade from algorithmic) | Normative account: why this algorithm solves the *right* problem. Error analysis showing failures correspond to problem boundaries, not arbitrary implementation limits. |
| $C \to A$ (downgrade to algorithmic) | If the normative account is missing or the error analysis shows failures unrelated to problem structure, the claim is algorithmic at best. |

## Metrics that provide computational-level evidence

- **D01 (Behavioral: logit attribution)** — measures output quality across conditions
- **D07 (Generalization gap)** — tests whether the circuit generalizes beyond its discovery distribution
- **A06 (Probabilistic specificity)** — tests whether the circuit is task-specific or a general bottleneck
- **F03 (Nomological validity)** — whether the circuit obeys theoretical predictions about the task

## Key references

- Marr, D. (1982). [*Vision.*](https://doi.org/10.7551/mitpress/9780262514620.001.0001) MIT Press. — Computational level as problem specification.
- Hanna, M., Liu, O., & Variengien, A. (2023). ["How does GPT-2 compute greater-than over the number line?"](https://arxiv.org/abs/2305.00586) *ICLR 2024.* — Computational + algorithmic with broad behavioral coverage.
- Wang, K., et al. (2022). ["Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small."](https://arxiv.org/abs/2211.00593) *ICLR 2023.* — IOI; computational framing with partial coverage.
- Kaplan, D. M. & Craver, C. F. (2011). ["The explanatory force of dynamical and mathematical models in neuroscience."](https://doi.org/10.1093/bjps/axr010) *British Journal for the Philosophy of Science.* — 3M constraint: models explain only when variables map to mechanism components.
