# Levels of Description — Perplexity Research Notes (saved 2026-05-22)

User saved this for later analysis. Not yet incorporated.

## The level-of-description claim

Most interpretability work operates at two levels: **activations** (what fires on a prompt) and **features/factors** (what directions encode meaning). Your weights paper already operates at a third level — **static weight geometry** — but it could be sharper if you explicitly name and defend a hierarchy.

| Level | Object | Method examples | What it answers |
|---|---|---|---|
| L0 Activations | Per-prompt residuals | ACDC, EAP, patching | What fired on this distribution |
| L1 Features | Learned dictionary directions | SAE, factor bank, DAS | What concepts the model uses |
| L2 Composed weights | OV·U, WQ·WK, K-composition, OV eigenspectra | Your paper's spectral/behavioral features, K-comp graph, an-neuron prediction | What the network is structurally built to do |
| L3 Functional roles | Tier classifications (backbone/detector/copier/readout) | Bootstrap stability classifier | What computational primitive each component implements |

Your contribution is making **L2 and L3** explicit and showing they have predictive content that L0/L1 methods miss. Frame the paper around that hierarchy rather than only "we found a circuit from weights."

## Pull from mech-val repo

From the 22 experiments you listed, these directly upgrade the weights paper:

| Mech-val experiment | Weights paper section it strengthens | What it adds |
|---|---|---|
| Blind prediction | Method intro | Pre-registered prediction is the strongest possible validity claim. |
| Circuit topology from weights | K-composition section | Promotes the K-comp graph from descriptive to predictive. |
| Concentrated-distributed ratio | Minimality-composition paradox | Free quantification of when EAP will fail. |
| Cross-circuit feature reuse | Cross-task transfer | Backbone/readout universality becomes a measured quantity. |
| Redundancy fingerprint | Copier tier discussion | Predicts backup heads — direct answer to Wang's backup name movers. |
| Prompt-distribution sensitivity | Discussion / complementarity | Operationalizes "structural vs distributional." |
| Sutter analogue | Discussion | Shows your method is Sutter-proof in a measurable sense. |
| Steering comparison 2x2 | Headline result | This is the headline — weight prediction → steering success. |
| Dark matter fraction | EAP failure analysis | Quantifies how much of computation EAP misses. |
| Weight-informed ACDC | Complementarity | Makes the "complementary not competitive" framing operational. |
| Information decomposition (bits per tier) | Tier validation | Quantitative grounding for the four-tier taxonomy. |

## Beyond factors — composed weight objects

| Composed object | Definition | What it reveals | Status in your work |
|---|---|---|---|
| OV·U (output-to-vocab) | W_O W_V W_U per head | Direct logit effect of writing in head's OV subspace; reveals copier vs suppressor | Already used; promote to its own section. |
| QK·E (query-key over embeddings) | W_Q W_K W_E per head | Which tokens a head attends to from which; reveals PTH, DTH, induction. | Partially used; formalize. |
| K-composition tensor | ⟨W_O[u], W_K[v]⟩_F | Information-routing graph between heads | Already in paper; promote to "weight-space wiring diagram". |
| Q-composition / V-composition | Same with W_Q, W_V | Distinguishes routing vs content composition | Probably underused — could differentiate copier subtypes. |
| OV eigenspectrum | Eigenvalues of W_O W_V | Copy vs suppress vs rotate behavior | Already used; expose distribution per layer. |
| MLP neuron direct effect | W_out[i] W_U | Per-neuron promotion of vocab tokens (your an-neuron) | Cross-model success — generalize to a "neuron atlas" appendix. |
| MLP·U full matrix | W_out W_U | Layer-level vocab effect; reveals which neurons jointly promote what | Likely worth adding. |
| Readers/writers of a factor | Heads/neurons whose OV/MLP·U projects onto a factor direction | Connects L1 factors to L2 composed weights | This is the bridge to the factor paper. |

## New high-level claims this enables

| Claim | Evidence path | Why it's new |
|---|---|---|
| Composed-weight geometry predicts circuit membership before any causal test | Blind prediction experiment + bootstrap stability + 19-experiment validation | Pre-registration with held-out tasks. |
| There exist universal computational primitives (backbone/readout) and task-specific ones (detector/copier), identifiable from weights alone | Cross-task transfer + cross-model transfer + Pythia transfer | Empirical taxonomy of weight signatures across models. |
| Marginal-attribution methods systematically miss distributed circuits, and the miss is predictable from concentrated/distributed ratio | Dark matter fraction + concentrated-distributed ratio + 8-method census | Theoretical bridge between L0 and L2: when L0 fails is predictable from L2 geometry. |

## Connecting weights paper <-> factor paper

| Shared component | Weights paper use | Factor paper use |
|---|---|---|
| Composed weight extractor | OV·U, K-comp graph, MLP·U | Reader/writer identification for factors |
| Bootstrap stability classifier | Functional role tagging | Could classify factor types (token-predictor, suppressor, etc.) |
| Cross-model transfer harness | GPT-2 -> Medium/Large/XL -> Pythia | Same harness applied to factorized checkpoints |
| Tier taxonomy | Backbone/detector/copier/readout | Could test whether factor-bank checkpoints preserve the same tiers |
