---
title: "Superposition"
description: "Criterion audit of the superposition claim, as submitted."
---

# Superposition

**Source.** *Toy Models of Superposition* ([Elhage et al., 2022](https://arxiv.org/abs/2209.10652)).

**Description.** A ReLU network trained to reconstruct sparse features stores more of them than it has dimensions, giving each feature a non-orthogonal direction and absorbing the interference those directions create; the linear model differing only in activation function shows no superposition at any sparsity. Where the tradeoff falls is mapped over an importance $×$ sparsity grid against closed-form losses for each candidate weight configuration, and the geometries the trained models settle into are uniform polytopes solving a generalized Thomson problem. Evaluated on: toy autoencoders on synthetic features whose sparsity, importance and correlation are set by hand, across four architectures and sizes from $(2,1)$ to 400 features in 30 dimensions, plus an absolute-value model that computes through a compressed layer rather than storing. Description mode: representational.

## Readings of the claim

**Superposition.** Readings of the claim, from the version the authors state to the version the field cites.

| **Reading** | **Verdict** | **Missing** |
|---|---|---|
| **Primary.** A ReLU network on sparse features represents more features than it has dimensions, in geometries set by sparsity and importance | Mechanistically Supported | A converse arm, with some property shown intact while superposition is removed (I6); correction for the lowest-loss selection the reported geometries are drawn from (M7); an input distribution outside the synthetic family (E2) |
| The reported geometries are properties of the optimum rather than of the optimizer | Underdetermined | The $m=2$ case is reported as much harder for gradient descent and its solutions are selected by loss (I7, M7); an invited comment in the same article shows the global minimum can have a smaller basin of attraction than nearby local minima, and the paper leaves it open (V3) |
| The phase diagram is fixed by sparsity and importance alone | Disconfirmed | Nothing — tested and failed: the paper holds the activation function fixed across every one of its own experiments, and a replication carried in the same article finds the phase diagrams look quite different under other activation functions (M3, M6) |
| Language models represent features in superposition | Insufficient | Every measurement is on a toy model, and the paper labels its real-model section validation by consistency with existing reports rather than measurement taken here (E4); no natural-data distribution appears anywhere, and Open Questions asks whether real importance and sparsity curves can be estimated at all (E2) |

## Verdict

**Superposition.** Verdict: **Mechanistically Supported**. The claim stops at the rung marked Blocked; rungs below it are Reached and rungs above it are blocked by that rung. Criteria in bold are absent or adverse at that rung; only those the rung requires can block it.

| **Tier** | **Requires** | **Missing** |
|---|---|---|
| Proposed | construct (C1–C2) | Reached. |
| Causally Suggestive | measurement (M2); internal (I1) | Reached. |
| Mechanistically Supported | internal (I2, I4); external (E1) | Reached. |
| Triangulated | construct (C3–C4); internal (I5–I7); external (E2, E4) | Blocked. **I6** double dissociation — Importance and sparsity are crossed, but on one outcome; no converse arm is run |
| Validated | construct (C5–C6); measurement (M1–M6); internal (I3, I10–I12); external (E2–E6); interpretive (V1–V5) | Blocked at Triangulated. **C6** complementation validity — Ground truth is available by construction and the test is not run; **I10** rescue reversibility — Adversarial training reduces superposition; nothing is corrupted and then restored; **I12** offset coupling — Offset never run; the learning-dynamics section is limited by the authors' own account |

## 36-criterion audit

**Superposition.** Full 36-criterion audit. Status: **C** = Confirmed, **PC** = Partially confirmed, **U** = Untested, **I** = Inconclusive, **D** = Disconfirmed, **N/A** = Not applicable.

| **ID** | **Criterion** | **Status** | **Evidence** |
|---|---|---|---|
| ***Construct Validity*** |   |   |   |
| C1 | Falsifiability | C | Point prediction against a linear control, plus a two-parameter phase diagram |
| C2 | Structural plausibility | C | Mechanism derived in closed form, then matched against the trained weights |
| C3 | Convergent validity | C | Theory and experiment agree, and three outside groups replicated before publication |
| C4 | Discriminant validity | PC | Three-way discrimination against planted features; PCA separated as a limiting case |
| C5 | Nomological validity | C | Formal links to compressed sensing, the Thomson problem and distributed codes |
| C6 | Complementation validity | U | Ground truth is available by construction and the test is not run |
| ***Measurement Validity*** |   |   |   |
| M1 | Reliability | PC | Repetition run and reported; aggregation is selection by loss, with no dispersion |
| M2 | Baseline separation | C | Linear control cannot superpose, runs at every sparsity, and separates completely |
| M3 | Stability | PC | Sparsity, importance, correlation and size all swept; activation function held fixed |
| M4 | Calibration | PC | Two purpose-built measures with stated readings and no external calibration |
| M5 | Sensitivity | C | Planted ground-truth features give a known positive at every point |
| M6 | Invariance | PC | Holds across sizes and data structures; one activation function and one loss |
| M7 | Selection correction | U | Best-of-1000 and worst-discarded selection documented, with no correction applied |
| ***Internal Validity*** |   |   |   |
| I1 | Necessity | C | Output nonlinearity, sparsity and negative bias each shown to be required |
| I2 | Sufficiency | C | Built from the hypothesized ingredients alone, and superposition appears |
| I3 | Minimality | PC | Load-bearing parts named and the smallest case solved; data assumptions untested |
| I4 | Specificity | N/A | One task and one loss, and the paper says cross-superposition loss comparison fails |
| I5 | Rival mechanism exclusion | PC | PCA named as a limiting case; incidental polysemanticity untested ([Lecomte et al., 2023](https://arxiv.org/abs/2312.03096)) |
| I6 | Double dissociation | U | Importance and sparsity are crossed, but on one outcome; no converse arm is run |
| I7 | Confound control | PC | Data confounds set by construction; optimization left as an uncontrolled one |
| I8 | Confounding sensitivity | U | Unattempted; nothing bounds an unmeasured confounder in the synthetic setup |
| I9 | Epistatic interaction | C | Feature interactions are the subject once uniformity is dropped, and each is measured |
| I10 | Rescue reversibility | U | Adversarial training reduces superposition; nothing is corrupted and then restored |
| I11 | Onset coupling | PC | Onset tracked across training; feature dimensionality jumps as the loss drops |
| I12 | Offset coupling | U | Offset never run; the learning-dynamics section is limited by the authors' own account |
| ***External Validity*** |   |   |   |
| E1 | Intervention reach | PC | Four intervention forms including two that remove superposition, cost reported |
| E2 | Prompt generalization | PC | Input distribution swept along every axis the theory names, all synthetic |
| E3 | Cross-task generalization | PC | Two tasks, with computation carrying the weight; representativeness left open |
| E4 | Cross-model recurrence | PC | Four architectures inside the toy family; real-model support is consistency only |
| E5 | Graded response | C | Dose-response is the core object: sparsity swept, response monotone and located |
| E6 | Novel prediction | C | Two predictions that could have failed: polytope geometry and adversarial fragility |
| ***Interpretive Validity*** |   |   |   |
| V1 | Level declaration | C | Levels separated before any experiment; the chosen definition flagged as circular |
| V2 | Level-evidence match | C | Claims stay at the toy-model level, and every step up from it is marked |
| V3 | Alternative level | PC | PCA examined as a limiting case; the optimization alternative raised and left open |
| V4 | Unlicensed labeling | C | Metaphors are physical and cashed out; agentive verbs appear but stay scare-quoted |
| V5 | Scope declaration | C | Scope declared in the title, in Key Results, in the Discussion and in Open Questions |
| **Total:** 15 Confirmed, 14 Partially confirmed, 6 Untested, 1 Not applicable |   |   |   |
| **Verdict: Mechanistically Supported** |   |   |   |


## Exploratory Lens Analysis

An exploratory reading of this claim through the framework's five lenses, written for this site and not part of the paper, is at [Superposition — exploratory lens analysis](/mechanistic-validity/framework/examples/examples-superposition/).
