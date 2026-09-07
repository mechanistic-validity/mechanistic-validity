---
title: Glossary
---

# Glossary

Quick-reference definitions for MI methods and scientific concepts referenced across this site. For the validity types, criteria, and verdict tiers, see the [Framework](/mechanistic-validity/framework/).

---

## Interpretability methods

<span id="activation-patching"></span>**Activation patching.** Replacing a component's activations from one forward pass into another to test causal relevance. The most common interventional method in circuit discovery. Establishes necessity (does removing this component matter?) but not sufficiency or specificity without further controls. See [activation patching on learnmechinterp](https://learnmechinterp.com/topics/activation-patching/).

<span id="path-patching"></span>**Path patching.** Variant of activation patching that tests specific information-flow paths rather than individual components. Establishes directed dependency between steps in a proposed circuit, not just node importance. See [path patching on learnmechinterp](https://learnmechinterp.com/topics/activation-patching/#path-patching).

<span id="ablation"></span>**Ablation.** Removing or zeroing a component (head, neuron, direction) to test necessity. A component is necessary if ablation degrades performance on the target task. Different ablation methods (zero, mean, resample) can give different answers; the method should be stated. See [activation patching on learnmechinterp](https://learnmechinterp.com/topics/activation-patching/).

<span id="attribution-patching"></span>**Attribution patching (EAP, ACDC).** Gradient-based approximations to activation patching that scale to full circuits. Edge attribution patching (EAP) estimates each edge's causal contribution; ACDC uses iterative patching to prune a circuit graph. See [attribution patching on learnmechinterp](https://learnmechinterp.com/topics/attribution-patching/).

<span id="causal-abstraction"></span>**Causal abstraction.** A framework for testing whether a high-level causal model is faithfully implemented by a neural network. [DAS](#das) and [IIA](#iia) are the primary tools; [causal scrubbing](#causal-scrubbing) extends this to full computational graphs. See [causal abstraction on learnmechinterp](https://learnmechinterp.com/topics/causal-abstraction/).

<span id="das"></span>**DAS (Distributed Alignment Search).** A method that searches over subspaces to find one whose swap transfers a causal variable. Evaluated by [IIA](#iia). The linearity of the alignment map constrains what the result establishes — unrestricted nonlinear maps can achieve high IIA on random models. See [causal abstraction on learnmechinterp](https://learnmechinterp.com/topics/causal-abstraction/).

<span id="iia"></span>**IIA (Interchange Intervention Accuracy).** The fraction of inputs on which swapping a subspace's projection successfully transfers the target variable's value. Tests surgical intervention quality. Low IIA is ambiguous: the swap may be non-surgical, or the causal graph may be wrong.

<span id="causal-scrubbing"></span>**Causal scrubbing.** A method that tests whether a proposed computational graph fully accounts for a model's behavior by resampling all activations not explained by the graph. The result depends on the pre-specified causal graph. See [causal abstraction on learnmechinterp](https://learnmechinterp.com/topics/causal-abstraction/).

<span id="sae"></span>**SAE (Sparse Autoencoder).** Learns an overcomplete dictionary of directions from a model's activations. Each direction is a candidate feature. The sparsity criterion is reconstruction-based, not causal — causal validation (steering, ablation) is needed to establish that a direction is a mechanism. See [sparse autoencoders on learnmechinterp](https://learnmechinterp.com/topics/sparse-autoencoders/).

<span id="linear-probing"></span>**Linear probing.** Trains a linear classifier on intermediate activations to test whether a concept is linearly represented. Observational, not causal — high probing accuracy does not establish that the model uses the representation. See [probing classifiers on learnmechinterp](https://learnmechinterp.com/topics/probing-classifiers/).

<span id="logit-lens"></span>**Logit lens / tuned lens.** Applies the unembedding matrix (or a learned affine transform) at intermediate layers to read off vocabulary-level predictions. Observational layer-by-layer readout. See [logit lens on learnmechinterp](https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/).

<span id="composition-score"></span>**Composition score.** $\|W^{OV}_u \cdot W^{QK}_v\|_F$ — a distribution-free upper bound on how much head $u$'s output influences head $v$'s attention pattern. Invariant under head permutations but not under orthogonal rotations. See [composition and virtual heads on learnmechinterp](https://learnmechinterp.com/topics/composition-and-virtual-heads/).

<span id="qk-ov-circuits"></span>**QK/OV circuits.** The two functional circuits within each attention head. The QK circuit determines where attention is directed; the OV circuit determines what information is moved. SVD of these matrices reveals the head's computational structure. See [QK/OV circuits on learnmechinterp](https://learnmechinterp.com/topics/qk-ov-circuits/).

## Scientific foundations

<span id="construct-validity"></span>**Construct validity.** Whether the measurement actually measures the theoretical construct it claims to measure. Adapted from psychometrics (Cronbach & Meehl, 1955). In mechanistic interpretability: does the circuit actually implement the computation the label claims? See the [Stanford Encyclopedia of Philosophy entry on construct validity](https://plato.stanford.edu/entries/reliabilism/) and the [construct validity type](/mechanistic-validity/framework/validity-types/construct/).

<span id="interventionism"></span>**Interventionism.** The philosophical framework (Woodward, 2003) that defines causal claims in terms of interventions: $X$ causes $Y$ if intervening on $X$ (while holding other variables fixed) changes $Y$. Activation patching and ablation are interventionist methods. See the [Stanford Encyclopedia of Philosophy entry on causation and manipulability](https://plato.stanford.edu/entries/causation-mani/).

<span id="mechanisms-in-science"></span>**Mechanisms in science.** The philosophical literature on what constitutes a mechanism — from Machamer, Darden & Craver's "entities and activities" to the new mechanist philosophy. See the [Stanford Encyclopedia of Philosophy entry on mechanisms in science](https://plato.stanford.edu/entries/science-mechanisms/).

<span id="double-dissociation"></span>**Double dissociation.** A crossed experimental design showing that component A is necessary for task X but not Y, and component B is necessary for Y but not X. Borrowed from neuropsychology. The strongest form of specificity evidence. See criterion [I6](/mechanistic-validity/framework/criteria/internal/double-dissociation/).

<span id="dose-response"></span>**Dose-response.** A graded manipulation showing that increasing the strength of an intervention produces a monotonic change in the outcome. Borrowed from pharmacology. In MI: steering with increasing coefficients, partial ablation, graded activation clamping.

<span id="do-calculus"></span>**do-calculus.** Pearl's formal language for distinguishing observation ($P(Y \mid X)$) from intervention ($P(Y \mid \text{do}(X))$). Every ablation study performs a do-operation. Stating it as one makes the conditioning set and exclusion restriction explicit. See [Pearl (2009), *Causality*](https://doi.org/10.1017/CBO9780511803161) and the [causal inference lens](/mechanistic-validity/framework/lenses/supporting/causal-inference/).

<span id="potential-outcomes"></span>**Potential outcomes.** Rubin's framework for treatment effects: each unit (prompt) has outcomes under treatment (ablated) and control (intact). The average treatment effect hides heterogeneity across prompts — a circuit can be necessary on some prompts and irrelevant on others. See [Rubin (1974)](https://doi.org/10.1037/h0037350) and the [causal inference lens](/mechanistic-validity/framework/lenses/supporting/causal-inference/).

<span id="transportability"></span>**Transportability.** Pearl & Bareinboim's formal conditions for when causal effects transfer across populations (models). A circuit discovered in GPT-2 Small transfers to GPT-2 Medium only if the differences between models do not open a confounding path. See [Pearl & Bareinboim (2011)](https://doi.org/10.1609/aaai.v25i1.8018) and the [causal inference lens](/mechanistic-validity/framework/lenses/supporting/causal-inference/).

<span id="kochs-postulates"></span>**Koch's postulates.** The classical criteria (1890) for establishing that an organism causes a disease: presence in all cases, isolation in pure culture, induction in a new host, re-isolation. Revised three times as unculturable organisms broke the isolation requirement — Rivers (1937) for viruses, [Falkow (1988)](https://doi.org/10.1093/cid/10.Supplement_2.S274) for genes, [Fredricks & Relman (1996)](https://doi.org/10.1128/cmr.9.1.18) for sequence data. What transfers to MI is the response to non-isolability. See [Koch's postulates on Wikipedia](https://en.wikipedia.org/wiki/Koch%27s_postulates) and the [medical microbiology lens](/mechanistic-validity/framework/lenses/supporting/medical-microbiology/).

<span id="hills-criteria"></span>**Hill's criteria.** Bradford Hill's (1965) nine "viewpoints" for evaluating whether an observed association is causal: strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, and analogy. Hill never called them criteria and denied they were sufficient or necessary conditions — [Phillips & Goodman (2004)](https://doi.org/10.1136/jech.2003.011502) trace the misreading. Several map to validity criteria in this framework: temporality → I11 onset coupling, biological gradient → E5 graded response, experiment → I1 necessity.

## Companion frameworks

<span id="mechanistic-views"></span>**Mechanistic Views.** A framework defining what a mechanism is — what kind of object is it, when two mechanisms are the same, and what formalism expresses the claim. Nine views spanning existing interpretability claims, ordered by ontological commitment. See [Mechanistic Views](https://mechanistic-views.github.io/mechanistic-views/).
