# Coverage Comparison: mechval vs Nanda "200 Concrete Open Problems in Mechanistic Interpretability" (2022)

Source: Neel Nanda (2022). "200 Concrete Open Problems in Mechanistic Interpretability."
https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability

Nanda's problems are spread across 9 thematic posts. Each problem below is classified as:
- **COVERED**: mechval has metrics that directly address the problem
- **PARTIAL**: mechval has related metrics but does not fully address it
- **GAP**: mechval has nothing relevant

---

## 1. The Case for Analysing Toy Language Models (Problems 1.1--1.23)

- [PARTIAL] Problem 1.1: "How far can you get with deeply reverse engineering a neuron in a 1L model?" -> related: `autointerp`, `rule_based_descriptions`; gap: no deep single-neuron reverse engineering metric for toy models
- [PARTIAL] Problem 1.2: "Find an interesting neuron and fully reverse engineer which direction should activate that feature" -> related: `autointerp`, `output_centric_description`; gap: no feature-direction alignment metric
- [GAP] Problem 1.3: "Look for trigram neurons (e.g., 'ice cream -> sundae')" -- no n-gram neuron detection metric
- [GAP] Problem 1.4: "Check the SoLU paper for ideas like finding a base64 neuron" -- no specific neuron type search metric
- [GAP] Problem 1.5: "Rigorously reverse engineer a neuron in 2L or larger models" -- no multi-layer neuron reverse engineering metric
- [GAP] Problem 1.6: "Hunt through Neuroscope for toy models and look for interesting neurons" -- exploratory, no metric needed
- [PARTIAL] Problem 1.7: "Find polysemantic neurons and explore what's occurring" -> related: `prism_polysemanticity`, `superposition_regime`; gap: no per-neuron polysemanticity decomposition
- [GAP] Problem 1.8: "Find neurons whose behavior matches regex or code patterns" -- no regex-matching neuron detector
- [PARTIAL] Problem 1.9: "How do 3-layer and 4-layer attention-only models differ from 2L?" -> related: `cross_model_invariance`, `cka_cross_arch`; gap: no layer-count comparison metric for toy models
- [PARTIAL] Problem 1.10: "Look for composition scores and identify pairs of heads that compose" -> related: `k_composition`, `composition_test`; gap: these exist but composition scores are known to have limitations
- [COVERED] Problem 1.11: "Look for evidence of composition in head interactions" -> `k_composition`, `composition_test`, `path_patching`
- [COVERED] Problem 1.12: "Ablate a single head and analyze performance changes across text" -> `activation_patching`, `sigma_ablation`, `role_ablation`
- [PARTIAL] Problem 1.13: "Look for tasks an nL model cannot do but (n+1)L model can" -> related: `cross_model_invariance`; gap: no explicit capability emergence threshold metric
- [GAP] Problem 1.14: "How do 1L SoLU/GELU models differ from 1L attention-only?" -- no activation-function comparison metric
- [GAP] Problem 1.15: "How do 2L SoLU models differ from 1L?" -- no SoLU-specific analysis
- [GAP] Problem 1.16: "How does 1L GELU differ from 1L SoLU?" -- no activation function interpretability comparison
- [GAP] Problem 1.17: "Analyse how a larger model 'fixes the bugs' of a smaller model" -- no bug-fixing circuit detection metric
- [GAP] Problem 1.18: "Does a 1L MLP transformer fix skip trigram bugs of 1L Attn Only?" -- no skip-trigram bug analysis
- [GAP] Problem 1.19: "Does a 3L attn-only model handle split-token induction challenges?" -- no split-token induction metric
- [GAP] Problem 1.20: "Address misfiring when previous token appears multiple times" -- no misfiring analysis metric
- [GAP] Problem 1.21: "Stopping induction on tokens likely showing repeated string ends" -- no induction suppression metric
- [GAP] Problem 1.22: "Does a 2L model with MLPs fix induction-related bugs?" -- no induction bug analysis
- [GAP] Problem 1.23: "Choose your own adventure analyzing interesting text patterns across models" -- open-ended, not a metric

## 2. Looking for Circuits in the Wild (Problems 2.1--2.36)

- [PARTIAL] Problem 2.1: "Reverse engineer induction heads with pointer arithmetic in GPT-2 Small" -> related: `eap`, `automatic_circuit_discovery`; gap: no pointer-arithmetic-specific induction head metric
- [GAP] Problem 2.2: "Interpret how models continue common sequences (e.g., 1 2 3 4 -> 5)" -- no sequence continuation circuit metric
- [GAP] Problem 2.3: "Analyze numbers at the start of lines of arbitrary length" -- no numbered-list continuation metric
- [GAP] Problem 2.4: "Reverse engineer how GPT-2 Small generates 3-letter acronyms" -- no acronym generation circuit metric
- [GAP] Problem 2.5: "Interpret how models convert names to email formats" -- no name-to-email circuit metric
- [COVERED] Problem 2.6: "Interpret factual recall using causal tracing" -> `activation_patching`, `mediation`, `mediation_v2`
- [GAP] Problem 2.7: "Analyze learning that words after full stops are capitalized" -- no capitalization circuit metric
- [GAP] Problem 2.8: "Reverse engineer how models count objects" -- no counting circuit metric
- [GAP] Problem 2.9: "Interpret how memorization occurs in models" -- no memorization circuit metric
- [PARTIAL] Problem 2.10: "Reverse engineer an induction head in a non-toy model" -> related: `eap`, `automatic_circuit_discovery`, `path_patching`; gap: no induction-head-specific reverse engineering metric
- [PARTIAL] Problem 2.11: "Analyze how models choose correct pronouns based on antecedents" -> related: `das_iia`, `iia_variants` (IOI is related); gap: no pronoun resolution circuit metric
- [COVERED] Problem 2.12: "Find and interpret behaviors of your own choosing in language models" -> `automatic_circuit_discovery`, `eap`, `sparse_feature_circuits` (general circuit finding)
- [GAP] Problem 2.13: "Interpret how code models predict closing brackets" -- no bracket-matching circuit metric
- [GAP] Problem 2.14: "Analyze how models close HTML tags correctly" -- no tag-matching circuit metric
- [GAP] Problem 2.15: "Reverse engineer how models determine methods depend on object type in code" -- no code type dispatch circuit metric
- [GAP] Problem 2.16: "Discover and interpret interesting patterns in code model behavior" -- no code model circuit discovery metric
- [COVERED] Problem 2.17: "Understand IOI in Stanford mistral models across different random seeds to test circuit universality" -> `convergent_evolution`, `cross_model_invariance`, `cross_model_transfer`
- [COVERED] Problem 2.18: "Determine whether earlier IOI circuit heads exhibit backup-style behavior when ablated" -> `role_ablation`, `activation_patching`, `corrupt_restore`
- [PARTIAL] Problem 2.19: "Identify general patterns explaining when backup head behavior occurs" -> related: `role_ablation`; gap: no backup-head emergence condition metric
- [GAP] Problem 2.20: "Reverse engineer how duplicate token heads recognize token copies without false positives" -- no duplicate token head precision metric
- [PARTIAL] Problem 2.21: "Understand how GPT-Neo implements IOI through MLP composition" -> related: `path_patching`, `eap`; gap: no MLP-specific composition metric for IOI
- [PARTIAL] Problem 2.22: "Analyze the role of Negative/Backup Name Mover Heads outside the IOI task" -> related: `cross_task_generalization`, `cross_task_transfer`; gap: no negative-name-mover role analysis outside IOI
- [PARTIAL] Problem 2.23: "Determine conditions for compensation mechanisms where ablation doesn't reduce performance" -> related: `adversarial_ablation_verification`, `corrupt_restore`; gap: no compensation/robustness condition classifier
- [GAP] Problem 2.24: "Study whether GPT-Neo (trained without dropout) exhibits backup heads" -- no dropout-vs-backup-head correlation metric
- [GAP] Problem 2.25: "Reverse engineer '4.11' (sharp previous token heads) at the parameter level" -- no per-head parameter-level reverse engineering metric
- [PARTIAL] Problem 2.26: "Understand what occurs in adversarial IOI examples (S-Inhibition Head patterns)" -> related: `adversarial_ablation_verification`, `boundary_sweep`; gap: no S-inhibition-specific adversarial analysis
- [PARTIAL] Problem 2.27: "Explain why models contain many induction heads and how they specialize" -> related: `attention_clustering`, `functional_localizer`; gap: no induction head specialization taxonomy metric
- [PARTIAL] Problem 2.28: "Determine why GPT-2 Small's performance degrades when MLP0 is ablated" -> related: `activation_patching`, `sigma_ablation`; gap: no MLP0-specific analysis
- [PARTIAL] Problem 2.29: "Find evidence supporting the residual stream as shared bandwidth hypothesis" -> related: `channel_capacity`, `information_bottleneck_circuit`, `spectral_svd`; gap: no explicit bandwidth measurement
- [GAP] Problem 2.30: "Identify neurons showing parameter dedication to memory management and cleanup" -- no memory-management neuron detection
- [GAP] Problem 2.31: "Analyze what happens to memory in induction circuits after use" -- no post-use memory trace metric
- [GAP] Problem 2.32: "Interpret how translation heads in GPT-J function" -- no translation head circuit metric
- [GAP] Problem 2.33: "Find and reverse engineer advanced induction heads like pattern matching heads" -- no pattern-matching head metric
- [GAP] Problem 2.34: "Explain how few-shot learning mechanisms operate in larger models" -- no few-shot mechanism circuit metric
- [GAP] Problem 2.35: "Reverse engineer how addition is computed, focusing on two-digit addition" -- no arithmetic circuit metric
- [PARTIAL] Problem 2.36: "Analyze emergent features in the residual stream that appear systematically larger than others" -> related: `spectral_svd`, `qk_norms`; gap: no emergent-feature magnitude ranking metric

## 3. Interpreting Algorithmic Problems (Problems 3.1--3.33)

- [GAP] Problem 3.1: "Sorting fixed-length lists" -- no sorting algorithm circuit metric
- [GAP] Problem 3.2: "Sorting variable-length lists" -- no variable-length sorting metric
- [GAP] Problem 3.3: "Interpret a 2-layer MLP trained on modular addition" -- no modular arithmetic circuit metric
- [GAP] Problem 3.4: "Interpret a 1-layer transformer on modular subtraction" -- no modular arithmetic metric
- [GAP] Problem 3.5: "Finding the minimum or maximum of two integers" -- no min/max circuit metric
- [GAP] Problem 3.6: "Permuting lists" -- no permutation circuit metric
- [GAP] Problem 3.7: "Calculating sequences with Fibonacci-style recurrence relations" -- no recurrence circuit metric
- [GAP] Problem 3.8: "Five-digit addition/subtraction with digit order variations" -- no multi-digit arithmetic metric
- [GAP] Problem 3.9: "Predicting simple code function outputs" -- no code evaluation circuit metric
- [GAP] Problem 3.10: "Graph theory problems including path-finding tasks" -- no graph algorithm circuit metric
- [GAP] Problem 3.11: "Training models on multiple algorithmic tasks simultaneously" -- no multi-task algorithmic circuit metric
- [GAP] Problem 3.12: "Training and interpreting models for automata tasks" -- no automata circuit metric
- [GAP] Problem 3.13: "In-context linear regression (Garg et al.)" -- no in-context learning algorithm metric
- [GAP] Problem 3.14: "Other in-context learning problems: sparse linear functions, 2-layer networks, decision trees" -- no in-context meta-learning circuit metric
- [GAP] Problem 3.15: "Five-digit or binary multiplication" -- no multiplication circuit metric
- [GAP] Problem 3.16: "Predicting repeated subsequences and reverse-engineering induction heads" -- no subsequence prediction metric
- [GAP] Problem 3.17: "Finding and interpreting custom algorithmic problems" -- open-ended
- [GAP] Problem 3.18: "Building toy model of Indirect Object Identification task" -- training task, not a metric
- [GAP] Problem 3.19: "Follow-up IOI questions regarding consistency, layer variations, architectural changes" -- no toy-IOI variant metric
- [GAP] Problem 3.20: "Reverse-engineering Othello-GPT algorithms and circuit features" -- no board-game circuit metric
- [GAP] Problem 3.21: "Training one-layer attention transformer to predict previous tokens" -- training task, not a metric
- [GAP] Problem 3.22: "Three-layer attention-only transformer for IOI" -- training task, not a metric
- [GAP] Problem 3.23: "Repeating modular addition analysis using GELU activation" -- no activation-function-specific circuit metric
- [GAP] Problem 3.24: "Understanding memorization mechanisms in neural networks" -- no memorization mechanism metric
- [PARTIAL] Problem 3.25: "Comparing dimensionality reduction techniques on interpretable tasks" -> related: `spectral_svd`, `cka`; gap: no systematic dimensionality reduction comparison metric
- [GAP] Problem 3.26: "Analyzing weight matrices and neuron clusters in modular addition" -- no weight-cluster metric for algorithmic tasks
- [PARTIAL] Problem 3.27: "Evaluating direct logit attribution usefulness and limitations" -> related: `mean_centered_logit`, `logit_diff`; gap: no meta-evaluation of logit attribution reliability
- [GAP] Problem 3.28: "Building toy models to understand the Lottery Ticket Hypothesis" -- no lottery ticket metric
- [GAP] Problem 3.29: "Creating algorithmic models to explore Deep Double Descent" -- no double descent metric
- [GAP] Problem 3.30: "Attempting concrete Othello-GPT starter projects" -- no Othello-GPT metric
- [GAP] Problem 3.31: "Finding and understanding modular circuits in Othello-GPT" -- no board-game modular circuit metric
- [GAP] Problem 3.32: "Studying neuron interpretability and superposition in Othello-GPT" -- no board-game superposition metric
- [GAP] Problem 3.33: "Using Othello-GPT as a transformer circuit laboratory for testing conjectures" -- no board-game test-bed metric

## 4. Exploring Polysemanticity and Superposition (Problems 4.1--4.45)

- [GAP] Problem 4.1: "Does dropout create a privileged basis?" -- no dropout-vs-basis metric
- [GAP] Problem 4.2: "Replicate absolute value model and study variants of the ReLU output models" -- no toy superposition model metric
- [GAP] Problem 4.3: "Explore neuron superposition by training absolute value model on x -> x^2" -- no toy model metric
- [GAP] Problem 4.4: "What happens to the ReLU output model with non-uniform sparsity?" -- no non-uniform sparsity toy metric
- [GAP] Problem 4.5: "Make inputs binary and look at AND or OR of pairs" -- no boolean superposition metric
- [GAP] Problem 4.6: "Keep inputs as uniform reals and look at max(x, y)" -- no max-function superposition metric
- [GAP] Problem 4.7: "Make the features binary (exactly two possible values)" -- no binary feature superposition metric
- [GAP] Problem 4.8: "Make the features discrete (e.g., 1, 2, or 3)" -- no discrete feature superposition metric
- [GAP] Problem 4.9: "Make the features uniform [0.5, 1]" -- no bounded feature superposition metric
- [GAP] Problem 4.10: "What happens if you replace ReLUs with GELUs in toy models?" -- no activation function comparison for toy models
- [GAP] Problem 4.11: "Can you find a toy model where GELU acts significantly differently from ReLU?" -- no GELU-vs-ReLU divergence metric
- [GAP] Problem 4.12: "Build a toy model of classification with cross-entropy loss" -- training task, not a metric
- [GAP] Problem 4.13: "Build a toy model with many more hidden features than output features" -- training task, not a metric
- [GAP] Problem 4.14: "Build a toy model needing multiple hidden layers of ReLUs" -- training task, not a metric
- [PARTIAL] Problem 4.15: "Build a toy model of attention head superposition/polysemanticity" -> related: `superposition_regime`, `prism_polysemanticity`; gap: no attention-head-level superposition toy model metric
- [GAP] Problem 4.16: "Build a toy model dealing with simultaneous interference" -- no interference resolution metric
- [GAP] Problem 4.17: "A learned example of a network with a 'non-linear representation'" -- no non-linear representation detection metric
- [GAP] Problem 4.18: "A network that doesn't have a discrete number of features" -- no continuous-feature representation metric
- [GAP] Problem 4.19: "A neural network with a 'non-decomposable' representation" -- no non-decomposability detection metric
- [GAP] Problem 4.20: "A task where networks can learn multiple different sets of features" -- no feature-set multiplicity metric
- [PARTIAL] Problem 4.21: "How are token identities stored in the 64-dimensional space of induction heads?" -> related: `spectral_svd`, `cka`; gap: no token-identity subspace geometry metric
- [GAP] Problem 4.22: "How does the previous token head communicate the value of the previous token?" -- no inter-head communication encoding metric
- [PARTIAL] Problem 4.23: "How is name/position information represented in IOI circuit residual stream?" -> related: `das_iia`, `causal_representation`; gap: no name/position encoding geometry metric
- [PARTIAL] Problem 4.24: "In models with absolute positional embeddings, does positional information get dedicated dimensions?" -> related: `spectral_svd`; gap: no positional dimension dedication metric
- [PARTIAL] Problem 4.25: "Can you find geometric superposition configurations from the ReLU output model in the residual stream?" -> related: `superposition_regime`; gap: no empirical geometric superposition detector for real models
- [GAP] Problem 4.26: "Can you find examples of locally almost-orthogonal bases?" -- no local orthogonality metric
- [GAP] Problem 4.27: "Can you find evidence for 'genre directions' enabling bottleneck superposition?" -- no genre-direction metric
- [GAP] Problem 4.28: "Can you find examples of a model learning to deal with simultaneous interference?" -- no interference resolution metric in real models
- [PARTIAL] Problem 4.29: "Look at a polysemantic neuron in a 1L model and figure out how the model disambiguates" -> related: `prism_polysemanticity`, `contextual_decomposition`; gap: no disambiguation-mechanism metric
- [GAP] Problem 4.30: "Do this on a two layer language model (studying polysemantic neurons)" -- no multi-layer polysemantic neuron analysis
- [PARTIAL] Problem 4.31: "Identify every neuron that represents a specific feature from a polysemantic neuron in a 1L model" -> related: `feature_absorption`; gap: no feature-to-neuron-set mapping metric
- [GAP] Problem 4.32: "Try to fully reverse engineer that feature" -- no complete feature reverse engineering metric
- [GAP] Problem 4.33: "Can you use superposition to create an adversarial example?" -- no superposition-based adversarial metric
- [GAP] Problem 4.34: "Can you find examples of asymmetric superposition in the MLP layer?" -- no asymmetric superposition metric
- [PARTIAL] Problem 4.35: "Pick a simple feature and train a linear probe to detect it in MLP activations" -> related: `probe_decodability`; gap: no MLP-specific probing metric
- [PARTIAL] Problem 4.36: "Look for features in neuroscope represented by various neurons; train a probe" -> related: `probe_decodability`, `autointerp`; gap: no neuroscope-guided probe metric
- [PARTIAL] Problem 4.37: "How do SoLU and GELU models compare under the polysemanticity metric?" -> related: `prism_polysemanticity`; gap: no SoLU-vs-GELU comparative metric
- [PARTIAL] Problem 4.38: "Can you find any better metrics for polysemanticity than those in the SoLU paper?" -> related: `prism_polysemanticity`, `superposition_regime`; gap: this IS a metric design problem we partially address
- [GAP] Problem 4.39: "Can you find evidence of LayerNorm smuggling through superposition in SoLU models?" -- no LayerNorm superposition smuggling metric
- [PARTIAL] Problem 4.40: "How do SoLU and GELU models with same initialization differ?" -> related: `cross_model_invariance`, `cka_cross_arch`; gap: no activation-function-controlled comparison
- [GAP] Problem 4.41: "How does GELU vs ReLU compare re polysemanticity?" -- no cross-activation-function polysemanticity comparison
- [GAP] Problem 4.42: "If you train a 1L or 2L model with d_mlp = 100 * d_model, what happens?" -- no overcomplete MLP analysis metric
- [GAP] Problem 4.43: "Study the original T5 XXL model regarding superposition" -- no T5-specific superposition metric
- [GAP] Problem 4.44: "Can you freeze all weights apart from a single MLP layer, then make the MLP layer 10x width?" -- no width-expanded frozen-model metric
- [GAP] Problem 4.45: "Pick one open question from Toy Models and try to make progress on it" -- open-ended

## 5. Analysing Training Dynamics (Problems 5.1--5.37)

- [GAP] Problem 5.1: "Understand why 5-digit addition has a phase change per digit" -- no phase-change-per-digit metric
- [GAP] Problem 5.2: "Determine the order of phase changes across digits" -- no phase change ordering metric
- [GAP] Problem 5.3: "Analyze PCA of logits or flattened weights; interpret components and predict grokking timing" -- no grokking prediction metric
- [GAP] Problem 5.4: "Develop a metric predicting grokking onset without future information" -- no grokking onset prediction metric
- [GAP] Problem 5.5: "Explain why models select specific frequencies and mid-training switches" -- no frequency selection metric
- [GAP] Problem 5.6: "Test if including progress measures in loss accelerates or stops grokking" -- no progress-measure loss augmentation metric
- [GAP] Problem 5.7: "Find evidence of phase transitions in complex models" -- no phase transition detection metric
- [GAP] Problem 5.8: "Refine analytical arguments with more faithful transformer toy models" -- theoretical, not a metric
- [GAP] Problem 5.9: "Investigate lottery ticket dynamics in induction head formation" -- no lottery ticket dynamics metric
- [PARTIAL] Problem 5.10: "Explain why certain heads appear consistently across similarly-trained models" -> related: `convergent_evolution`, `cross_model_invariance`; gap: no head-emergence consistency predictor
- [GAP] Problem 5.11: "Measure how knocking out circuit parameters delays generalization" -- no ablation-delay-generalization metric
- [GAP] Problem 5.12: "Find progress measures predicting head composition development" -- no composition progress metric
- [GAP] Problem 5.13: "Predict which heads compose first at initialization" -- no initialization-based composition predictor
- [GAP] Problem 5.14: "Determine if composition develops as a phase transition" -- no composition phase transition metric
- [GAP] Problem 5.15: "Build toy fine-tuning models and identify internal motifs" -- no fine-tuning motif metric
- [GAP] Problem 5.16: "Explore performance changes on original training distribution post fine-tuning" -- no fine-tuning degradation metric
- [GAP] Problem 5.17: "Analyze mechanistic differences in fine-tuned models via attribution" -- no fine-tuning attribution difference metric
- [GAP] Problem 5.18: "Compare neuron activation patterns before and after fine-tuning" -- no pre/post fine-tuning neuron comparison
- [GAP] Problem 5.19: "Conduct broad mechanistic exploration of fine-tuning processes" -- open-ended
- [GAP] Problem 5.20: "Identify phase transitions during fine-tuning checkpoints" -- no fine-tuning phase transition metric
- [GAP] Problem 5.21: "Replicate induction head phase transition across checkpointed models" -- no training-checkpoint phase transition metric
- [GAP] Problem 5.22: "Analyze neuron formation as phase transitions in SoLU models" -- no neuron formation phase metric
- [GAP] Problem 5.23: "Apply per-token loss analysis to discover additional phase changes" -- no per-token phase change metric
- [GAP] Problem 5.24: "Track recognizable attention patterns across training" -- no attention pattern tracking metric
- [GAP] Problem 5.25: "Investigate IOI task dynamics during training" -- no IOI training dynamics metric
- [GAP] Problem 5.26: "Analyze name-mover heads via direct logit attribution during training" -- no training-time DLA tracking
- [GAP] Problem 5.27: "Study attention patterns across head categories during training" -- no cross-training attention tracking
- [GAP] Problem 5.28: "Examine algorithmic tasks like few-shot learning and sorting during training" -- no training-time algorithmic task metric
- [GAP] Problem 5.29: "Investigate soft induction heads like translation during training" -- no soft-induction training metric
- [GAP] Problem 5.30: "Analyze phase changes in benchmark performance" -- no benchmark phase change metric
- [GAP] Problem 5.31: "Test hypothesis that scaling laws result from numerous tiny phase changes" -- no micro-phase-change metric
- [COVERED] Problem 5.32: "Measure output consistency across models trained with different seeds" -> `convergent_evolution`, `cross_model_invariance`
- [COVERED] Problem 5.33: "Test consistency on algorithmic tasks across models" -> `convergent_evolution`, `cross_model_transfer`
- [COVERED] Problem 5.34: "Compare IOI circuit implementation across similarly-sized models" -> `cross_model_invariance`, `cross_model_transfer`, `convergent_evolution`
- [GAP] Problem 5.35: "Identify capabilities smaller models possess that larger ones lack" -- no inverse-scaling capability metric
- [GAP] Problem 5.36: "Apply Git Re-Basin techniques to modular addition models" -- no model merging metric
- [GAP] Problem 5.37: "Find domains where Git Re-Basin successfully interpolates circuits" -- no circuit interpolation metric

## 6. Techniques, Tooling and Automation (Problems 6.1--6.59)

- [COVERED] Problem 6.1: "Breaking current techniques - find concrete edge cases where techniques break" -> `error_boundary`, `boundary_sweep`, `hyperparam_sensitivity`
- [COVERED] Problem 6.2: "Direct logit attribution - look at GPT-Neo Small where logit lens works badly" -> `mean_centered_logit`, `logit_diff`
- [GAP] Problem 6.3: "Can you fix DLA in GPT-Neo Small, e.g., by finding a linear approximation to the final layer?" -- no DLA fix metric
- [GAP] Problem 6.4: "Linearising LayerNorm - look at the scale factor for each layernorm across data" -- no LayerNorm linearization metric
- [COVERED] Problem 6.5: "Activation patching - explore when it breaks due to dependence on multiple variables" -> `activation_patching`, `boundary_sweep`, `error_boundary`
- [COVERED] Problem 6.6: "Causal scrubbing - explore limitations and edge cases" -> `causal_scrubbing`
- [COVERED] Problem 6.7: "Ablations - start with backup name movers in IOI where zero ablations break" -> `sigma_ablation`, `role_ablation`, `adversarial_ablation_verification`
- [COVERED] Problem 6.8: "Can you find places where one ablation method breaks but others don't?" -> `sigma_ablation` (compares ablation types)
- [PARTIAL] Problem 6.9: "Composition scores don't work well for IOI circuit; investigate why" -> related: `k_composition`, `composition_test`; gap: no composition score failure analysis metric
- [GAP] Problem 6.10: "Eigenvalue copying score - explore limitations" -- no eigenvalue copying score metric
- [PARTIAL] Problem 6.11: "Automate ways to identify heads that compose, validated on IOI" -> related: `k_composition`, `composition_test`, `eap`; gap: no automated composition discovery validated against ground truth
- [PARTIAL] Problem 6.12: "Look for composition on specific inputs by decomposing the residual stream" -> related: `contextual_decomposition`, `path_patching`; gap: no input-specific composition decomposition metric
- [COVERED] Problem 6.13: "Can you do head composition detection with direct path patching?" -> `path_patching`, `path_specificity`, `path_identification`
- [COVERED] Problem 6.14: "Compare causal tracing to activation patching; do they give the same outputs?" -> `activation_patching`, `mediation`, `mediation_v2`
- [COVERED] Problem 6.15: "How do activation patching results change when you patch single layers instead of 10 adjacent ones?" -> `activation_patching` (supports per-layer)
- [COVERED] Problem 6.16: "Can you get anywhere when patching specific neurons?" -> `activation_patching`
- [COVERED] Problem 6.17: "Can you get results when patching sets of neurons?" -> `activation_patching`, `sparse_feature_circuits`
- [PARTIAL] Problem 6.18: "Automated ways to analyse attention patterns to find previous token heads" -> related: `attention_clustering`, `functional_localizer`; gap: no specific previous-token-head detector metric
- [PARTIAL] Problem 6.19: "Automated ways to find duplicate token heads" -> related: `attention_clustering`, `functional_localizer`; gap: no duplicate-token-head detector metric
- [PARTIAL] Problem 6.20: "Automated ways to find induction heads via repeated random tokens" -> related: `attention_clustering`, `functional_localizer`; gap: no automated induction head detection score
- [GAP] Problem 6.21: "Automated detection of translation heads" -- no translation head detection metric
- [GAP] Problem 6.22: "Automated detection of few-shot learning heads" -- no few-shot head detection metric
- [GAP] Problem 6.23: "Find automated way to detect pointer arithmetic induction heads" -- no pointer arithmetic head metric
- [PARTIAL] Problem 6.24: "Detecting IOI circuit heads (S-Inhibition, name mover, etc.)" -> related: `functional_localizer`, `attention_clustering`; gap: no IOI-role-specific head detector
- [PARTIAL] Problem 6.25: "Detecting factual recall heads" -> related: `mediation`, `mediation_v2`, `functional_localizer`; gap: no specific factual-recall-head detector
- [GAP] Problem 6.26: "Combine head detectors to make a 'wiki' for a range of models" -- no cross-model head wiki metric
- [PARTIAL] Problem 6.27: "Can you do a similar thing for neuron interpretability, e.g., finding trigram neurons?" -> related: `autointerp`; gap: no automated neuron type classification metric
- [PARTIAL] Problem 6.28: "Finding good ways to find max activating dataset examples for attention heads" -> related: `autointerp`; gap: no attention-head max-activation metric
- [PARTIAL] Problem 6.29: "Refining max activating dataset examples technique for neuron interpretability" -> related: `autointerp`, `rule_based_descriptions`; gap: no max-activation refinement metric
- [GAP] Problem 6.30: "Corrupt different token embeddings to see which matter for neuron activation" -- no token-corruption neuron analysis metric
- [GAP] Problem 6.31: "Compare corrupted tokens to randomly chosen directions in neuron activation space" -- no directional corruption comparison metric
- [PARTIAL] Problem 6.32: "Validate max activating examples by comparing to direct effect on logits" -> related: `autointerp`, `mean_centered_logit`; gap: no max-act-vs-logit-effect validation metric
- [GAP] Problem 6.33: "Using models like RoBERTa or GPT-3 to find similar text and replacing specific tokens" -- no model-assisted counterfactual generation metric
- [GAP] Problem 6.34: "Look at dataset examples at different quantiles for neuron activations" -- no quantile activation analysis metric
- [GAP] Problem 6.35: "Add refinements to Neuroscope infrastructure" -- tooling, not a metric
- [GAP] Problem 6.36: "Finding the minimal example to activate a neuron by truncating text" -- no minimal activation example metric
- [GAP] Problem 6.37: "Replicate interpretability illusion results by finding polysemantic neurons across datasets" -- no interpretability illusion replication metric
- [GAP] Problem 6.38: "In SoLU models, compare max activating results for pre, mid, and post SoLU activations" -- no SoLU-stage comparison metric
- [COVERED] Problem 6.39: "Can GPT-3 figure out trends in max activating examples for a neuron?" -> `autointerp`
- [GAP] Problem 6.40: "Can you use GPT-3 to generate counterfactual prompts for activation patching on novel problems?" -- no LLM-generated counterfactual metric
- [GAP] Problem 6.41: "Choose your own adventure - find a way to usefully use an LLM for interpretability" -- open-ended
- [PARTIAL] Problem 6.42: "Feature attribution - compare integrated gradients to max activation or neuron attribution" -> related: `relp`, `contextual_decomposition`; gap: no integrated gradients comparison metric
- [COVERED] Problem 6.43: "Probing - get evidence for or against predictions in Toy Models of Superposition" -> `probe_decodability`, `superposition_regime`
- [GAP] Problem 6.44: "Pick anything interesting from Rauker et al's survey" -- open-ended
- [GAP] Problem 6.45: "Adapt Wiles et al's automated bug analysis from image models to language models" -- no automated bug analysis metric
- [COVERED] Problem 6.46: "Taking existing well-understood circuits and explore quantitative ways to characterise them as true circuits" -> `graph_minimality`, `edge_necessity`, `edge_jaccard`, `compositional_sufficiency`
- [COVERED] Problem 6.47: "Build on Arthur Conmy's work to automatically find circuits via recursive path patching" -> `automatic_circuit_discovery`, `eap`, `sparse_feature_circuits`
- [GAP] Problem 6.48: "Resolve open issues and feature requests for TransformerLens" -- tooling, not a metric
- [COVERED] Problem 6.49: "Build tooling to take the 'diff' of two models with different internal structures" -> `crosscoder_model_diff`, `cka_cross_arch`
- [PARTIAL] Problem 6.50: "Run models on text and look at biggest per-token log probability difference" -> related: `per_token_nll`; gap: no cross-model per-token difference metric
- [GAP] Problem 6.51: "Run models on various benchmarks and compare performance" -- benchmarking, not a mech-interp metric
- [GAP] Problem 6.52: "Try 'benchmarks' of algorithmic task ability like IOI, acronyms, emails" -- benchmarking
- [GAP] Problem 6.53: "Try qualitative exploration by generating text from models" -- qualitative, not a metric
- [COVERED] Problem 6.54: "Build tooling to take the 'diff' of two models with the same internal structure" -> `crosscoder_model_diff`, `cka`
- [COVERED] Problem 6.55: "Look at weight differences and find the largest difference" -> `crosscoder_model_diff`
- [PARTIAL] Problem 6.56: "Run models on text and compare activations - look for biggest differences" -> related: `cka`, `crosscoder_model_diff`; gap: no per-activation biggest-difference metric
- [PARTIAL] Problem 6.57: "Look at direct logit attribution differences across layers and heads on various texts" -> related: `mean_centered_logit`, `logit_diff`; gap: no cross-model DLA comparison metric
- [COVERED] Problem 6.58: "Do activation patching where one model performs much better than the other" -> `activation_patching`, `crosscoder_model_diff`
- [GAP] Problem 6.59: "Find principled alternatives to QK matrix analysis for rotary attention models" -- no RoPE-specific QK analysis metric

## 7. Image Model Interpretability (Problems 7.1--7.18)

- [GAP] Problem 7.1: "Reverse engineer ResNets using feature visualization" -- no vision model metric (framework is LLM-focused)
- [GAP] Problem 7.2: "Apply transformer circuits methods to Vision Transformers" -- no ViT circuit metric
- [GAP] Problem 7.3: "Investigate ConvNeXt" -- no ConvNeXt metric
- [GAP] Problem 7.4: "Improve hand-coded curve detectors by adding color" -- no curve detector metric
- [GAP] Problem 7.5: "Attempt hand-coding circuits beyond curve detectors" -- no hand-coded circuit metric
- [GAP] Problem 7.6: "Apply Causal Scrubbing to validate claimed curve circuits" -- no image causal scrubbing metric
- [GAP] Problem 7.7: "Search for equivariance patterns -- families of analogous neurons" -- no equivariance metric
- [GAP] Problem 7.8: "Analyze polysemantic neurons in image models" -- no image polysemanticity metric
- [GAP] Problem 7.9: "Explore diverse circuits using the weight explorer" -- no weight explorer metric
- [GAP] Problem 7.10: "Examine weight sparsity between adjacent layers in multimodal models" -- no multimodal weight sparsity metric
- [GAP] Problem 7.11: "Rigorously reverse-engineer multimodal circuits" -- no multimodal circuit metric
- [GAP] Problem 7.12: "Apply transformer circuits techniques to attention heads in image portions of multimodal models" -- no multimodal attention metric
- [GAP] Problem 7.13: "Refine max-activating text string generation techniques" -- no max-activating string generation metric
- [GAP] Problem 7.14: "Train checkpointed Inception runs to detect phase transitions in curve detector formation" -- no vision training dynamics metric
- [GAP] Problem 7.15: "Test activation patching effectiveness on Inception-scale models" -- no vision activation patching metric
- [GAP] Problem 7.16: "Apply feature visualization to diffusion model neurons" -- no diffusion model metric
- [GAP] Problem 7.17: "Identify style-transfer neurons in diffusion models" -- no diffusion style neuron metric
- [GAP] Problem 7.18: "Analyze circuit activation variations across different noise input levels in diffusion models" -- no diffusion noise-level metric

## 8. Interpreting Reinforcement Learning (Problems 8.1--8.21)

- [GAP] Problem 8.1: "Replicate Tom McGrath's AlphaZero work with LeelaChessZero" -- no RL/chess metric (framework is LLM-focused)
- [GAP] Problem 8.2: "Try applying this to an AlphaZero-style Go agent" -- no Go agent metric
- [GAP] Problem 8.3: "Train a small AlphaZero model on Tic-Tac-Toe and interpret it" -- no toy RL metric
- [GAP] Problem 8.4: "Can you extend the work on LeelaZero?" -- no chess feature computation metric
- [GAP] Problem 8.5: "Interpret one of the examples in the goal misgeneralisation papers" -- no goal misgeneralization metric
- [GAP] Problem 8.6: "Tree Gridworld and Monster Gridworld from Shah et al" -- no gridworld metric
- [GAP] Problem 8.7: "CoinRun misgeneralisation prediction" -- no CoinRun metric
- [GAP] Problem 8.8: "Can you apply transformer circuits techniques to a decision transformer?" -- no decision transformer metric
- [GAP] Problem 8.9: "Train and interpret a model from In-Context RL and Algorithmic Distillation" -- no in-context RL metric
- [GAP] Problem 8.10: "Interpret CarperAI's RLHF model" -- no RLHF circuit metric
- [GAP] Problem 8.11: "Can you find circuits corresponding to longer term planning?" -- no planning circuit metric
- [GAP] Problem 8.12: "Can you get traction on interpreting a reward model?" -- no reward model metric
- [GAP] Problem 8.13: "Train a toy RLHF model to do a simple task" -- no toy RLHF metric
- [GAP] Problem 8.14: "Try training and interpreting a small model from Guez et al" -- no RL planning metric
- [GAP] Problem 8.15: "Can you interpret a small model trained with policy gradients on a gridworld?" -- no policy gradient circuit metric
- [GAP] Problem 8.16: "Can you interpret a small model trained with policy gradients on an OpenAI gym task?" -- no gym task circuit metric
- [GAP] Problem 8.17: "Can you interpret a small model trained with policy gradients on Atari?" -- no Atari circuit metric
- [GAP] Problem 8.18: "Try any of the above, training with Q-Learning instead" -- no Q-learning circuit metric
- [GAP] Problem 8.19: "Train another network to copy an RL agent's output logits and interpret that" -- no distilled RL metric
- [GAP] Problem 8.20: "Extend understanding to study the agent during training" -- no RL training dynamics metric
- [GAP] Problem 8.21: "Choose your own adventure interpreting an agent" -- open-ended

## 9. Studying Learned Features in Language Models (Problems 9.1--9.62)

- [PARTIAL] Problem 9.1: "Explore random neurons using interactive neuroscope" -> related: `autointerp`; gap: no interactive exploration metric
- [PARTIAL] Problem 9.2: "Look for interesting conceptual neurons in middle layers of larger models" -> related: `autointerp`, `rule_based_descriptions`; gap: no conceptual neuron discovery metric
- [GAP] Problem 9.3: "Look for examples of detokenization neurons" -- no detokenization neuron metric
- [GAP] Problem 9.4: "Look for examples of trigram neurons" -- no trigram neuron metric
- [GAP] Problem 9.5: "Look for examples of retokenization neurons" -- no retokenization neuron metric
- [GAP] Problem 9.6: "Look for examples of context neurons (e.g., base64)" -- no context neuron metric
- [GAP] Problem 9.7: "Look for neurons that align with feature ideas" -- no feature-neuron alignment metric
- [GAP] Problem 9.8: "Look for neurons with a naive but incorrect initial story that is simpler after investigation" -- no interpretability refinement metric
- [GAP] Problem 9.9: "Neurons with a naive but incorrect initial story that is more complex after investigation" -- no interpretability complexity metric
- [PARTIAL] Problem 9.10: "How much does the logit attribution of a neuron align with dataset example patterns?" -> related: `mean_centered_logit`, `autointerp`; gap: no logit-vs-activation alignment metric
- [GAP] Problem 9.11: "If logit attribution seems inconsistent, figure out what's going on" -- no logit attribution inconsistency diagnosis metric
- [GAP] Problem 9.12: "For dataset examples for neurons in a 1L network, measure how much pre-activation comes from each attention head vs embedding" -- no neuron activation source decomposition metric
- [GAP] Problem 9.13: "Basic syntax features: start of line, end of sentence, proper nouns, numbers, dates..." -- no specific syntax feature neuron metric
- [GAP] Problem 9.14: "Proper noun features: Name, Firstname, Surname, Country, City..." -- no proper noun neuron metric
- [GAP] Problem 9.15: "Python code features: variable in function definition, disambiguation features for commas, colons..." -- no code feature neuron metric
- [GAP] Problem 9.16: "Level of indent for a line" -- no indentation level neuron metric
- [GAP] Problem 9.17: "Level of bracket nesting" -- no bracket nesting neuron metric
- [GAP] Problem 9.18: "General code features: Base64, hexadecimal, HTML tags, programming language detection" -- no code feature neuron metric
- [GAP] Problem 9.19: "LaTeX features: common commands, section titles" -- no LaTeX neuron metric
- [GAP] Problem 9.20: "Features in compiled LaTeX, e.g., paper citations" -- no LaTeX citation neuron metric
- [GAP] Problem 9.21: "Abstract neurons (Christmas, sadness, teenager, anime, Pokemon, etc.)" -- no abstract concept neuron metric
- [GAP] Problem 9.22: "Foreign language disambiguation (e.g., 'die' in Dutch vs German vs Afrikaans)" -- no language disambiguation neuron metric
- [GAP] Problem 9.23: "Words with multiple meanings (e.g., bat = animal or sports equipment)" -- no word sense disambiguation neuron metric
- [GAP] Problem 9.24: "Search for memory management neurons (high negative cosine sim between w_in and w_out)" -- no memory management neuron metric
- [GAP] Problem 9.25: "Search for signal boosting neurons (high positive cosine sim between w_in and w_out)" -- no signal boosting neuron metric
- [GAP] Problem 9.26: "Search for neurons that clean up superposition interference" -- no interference cleanup neuron metric
- [GAP] Problem 9.27: "Can you find split-token neurons?" -- no split-token neuron metric
- [PARTIAL] Problem 9.28: "Can you find examples of neuron families/equivariance?" -> related: `convergent_evolution`; gap: no intra-model neuron family detection metric
- [GAP] Problem 9.29: "Induction should not trigger when current token is repeated but previous is not" -- no induction false positive metric
- [GAP] Problem 9.30: "Fixing a skip trigram bug" -- no skip trigram bug metric
- [GAP] Problem 9.31: "This token is duplicated" -- no token duplication detection neuron metric
- [GAP] Problem 9.32: "Splitting 'token X is duplicated' for many common tokens" -- no per-token duplication neuron metric
- [GAP] Problem 9.33: "Neurons which represent positional information" -- no position-encoding neuron metric
- [GAP] Problem 9.34: "What's the longest n-gram you can find that seems represented?" -- no max n-gram length metric
- [COVERED] Problem 9.35: "Try training linear probes for any of the above features" -> `probe_decodability`
- [PARTIAL] Problem 9.36: "How does ability to recover features from residual stream compare to MLP vs attention layer outputs?" -> related: `probe_decodability`; gap: no layer-type-specific probe comparison metric
- [GAP] Problem 9.37: "Are there features that can only be recovered from certain MLP layers?" -- no layer-specific feature exclusivity metric
- [GAP] Problem 9.38: "Are there features significantly easier to recover from early layer residual streams?" -- no layer-dependent recoverability metric
- [GAP] Problem 9.39: "Is a neuron the most activated neuron on its own max activating text?" -- no max-activation reciprocity metric
- [GAP] Problem 9.40: "Look at distributions of neuron activations (pre and post activation)" -- no neuron activation distribution metric
- [GAP] Problem 9.41: "Do neurons vary in how heavy-tailed their distributions are?" -- no neuron distribution heavy-tailedness metric
- [GAP] Problem 9.42: "How similar are distributions between SoLU and GELU?" -- no cross-activation-function distribution comparison
- [GAP] Problem 9.43: "What does the distribution of LayerNorm scale and softmax denominator in SoLU look like?" -- no SoLU-specific distribution metric
- [PARTIAL] Problem 9.44: "Can you find any genuinely monosemantic neurons?" -> related: `prism_polysemanticity`; gap: no monosemanticity certification metric (inverse of polysemanticity)
- [GAP] Problem 9.45: "Find a feature where GELU is used to calculate it in a way ReLU couldn't" -- no GELU-specific feature computation metric
- [PARTIAL] Problem 9.46: "Can you find a feature represented by several neurons?" -> related: `feature_absorption`; gap: no multi-neuron feature representation metric
- [PARTIAL] Problem 9.47: "What happens to the model if you ablate some of these neurons?" -> related: `sigma_ablation`, `activation_patching`; gap: no feature-specific ablation metric
- [GAP] Problem 9.48: "Can you find a feature that is highly diffuse across neurons?" -- no feature diffuseness metric
- [COVERED] Problem 9.49: "Look at direct logit attribution of neurons, and find max dataset examples" -> `mean_centered_logit`, `autointerp`
- [GAP] Problem 9.50: "Look at max negative DLA. Are there neurons that systematically suppress the correct next token?" -- no negative DLA suppression metric
- [PARTIAL] Problem 9.51: "Try comparing how monosemantic neurons in GELU vs SoLU models are" -> related: `prism_polysemanticity`; gap: no cross-activation monosemanticity comparison
- [PARTIAL] Problem 9.52: "Can you find a better and more robust metric for monosemanticity?" -> related: `prism_polysemanticity`, `autointerp`; gap: this IS a metric design problem
- [COVERED] Problem 9.53: "Can you see any correspondence between what neurons represent in each model?" -> `convergent_evolution`, `cross_model_invariance`
- [COVERED] Problem 9.54: "If a feature is represented in one model, how likely is it to be represented in the other?" -> `convergent_evolution`, `cross_model_transfer`
- [GAP] Problem 9.55: "Can you find a neuron whose activation isn't significantly affected by the current token?" -- no context-only neuron metric
- [GAP] Problem 9.56: "Can you find evidence that models are doing something more sophisticated regarding clause/sentence boundaries?" -- no sentence boundary sophistication metric
- [GAP] Problem 9.57: "Replicate Knowledge Neurons in Pretrained Transformers on a generative model" -- no knowledge neuron metric
- [GAP] Problem 9.58: "Can you replicate interpretability illusion results on SoLU models?" -- no interpretability illusion metric
- [GAP] Problem 9.59: "Try doing dimensionality reduction over neuron activations across text" -- no neuron dimensionality reduction metric
- [GAP] Problem 9.60: "Pick a BERTology paper and try to replicate it on GPT-2" -- open-ended
- [GAP] Problem 9.61: "Make a PR to Neuroscope with some feature you wish it had" -- tooling, not a metric
- [GAP] Problem 9.62: "Replicate the part of Conjecture's Polytopes paper about top dataset examples for a neuron" -- no polytope metric

---

## Summary

| Category | COVERED | PARTIAL | GAP | Total |
|----------|---------|---------|-----|-------|
| 1. Toy Language Models | 2 | 4 | 17 | 23 |
| 2. Circuits in the Wild | 5 | 12 | 19 | 36 |
| 3. Algorithmic Problems | 0 | 2 | 31 | 33 |
| 4. Polysemanticity & Superposition | 0 | 12 | 33 | 45 |
| 5. Training Dynamics | 3 | 1 | 33 | 37 |
| 6. Techniques, Tooling & Automation | 18 | 12 | 29 | 59 |
| 7. Image Model Interpretability | 0 | 0 | 18 | 18 |
| 8. Reinforcement Learning | 0 | 0 | 21 | 21 |
| 9. Learned Features in LMs | 5 | 10 | 47 | 62 |
| **Total** | **33** | **53** | **248** | **334** |

Note: Nanda's list as published contains ~334 individual problems across 9 posts, exceeding the "200" title. Many are sub-variants or progressive extensions of a core problem.

### Coverage rate
- **Direct coverage**: 33/334 = 9.9%
- **Partial + covered**: 86/334 = 25.7%
- **True gaps**: 248/334 = 74.3%

### Key gap themes (relevant to validity evaluation)

1. **Algorithmic task circuits** (Category 3): mechval has zero coverage. These are about interpreting how models solve well-defined algorithmic tasks (modular arithmetic, sorting, counting). Important for validity because they provide ground-truth circuits to validate against.

2. **Training dynamics** (Category 5): Almost zero coverage. Phase transitions, grokking, circuit formation during training. Important for understanding whether circuits are artifacts of training or genuine computational structure.

3. **Feature universality / cross-model consistency** (Categories 4, 9): Partial coverage via `convergent_evolution` and `cross_model_invariance`, but no metrics for specific universality predictions (same features across architectures, activation functions, initialization seeds).

4. **Neuron-level feature cataloguing** (Category 9): mechval has `autointerp` and `prism_polysemanticity` but lacks metrics for specific neuron types (trigram, detokenization, memory management, signal boosting, context neurons).

5. **Superposition geometry** (Category 4): `superposition_regime` exists but does not measure geometric configurations (orthogonal, antipodal, polytope) or interference resolution in real models.

6. **Non-LLM architectures** (Categories 7, 8): Complete gap. mechval is LLM-focused by design. Image models, RL agents, diffusion models, and decision transformers are entirely out of scope.

7. **Head type taxonomy** (Category 6): No automated detectors for specific head types (induction, previous token, duplicate token, translation, few-shot). `functional_localizer` and `attention_clustering` are related but don't produce head-type classifications.

### Most impactful gaps for validity evaluation

The following gaps are most relevant to the mechval mission of evaluating mechanistic interpretability claims:

1. **Superposition geometry in real models** (4.25, 4.26, 4.28) -- Can claimed features actually be found with predicted geometric relationships?
2. **Feature universality across seeds/architectures** (5.32-5.34, 9.53-9.54) -- Partially covered but needs deeper cross-architecture metrics
3. **Circuit formation dynamics** (5.7, 5.14, 5.21-5.23) -- Do circuits form as phase transitions? Critical for understanding whether circuits are real computational units
4. **Head type detection** (6.18-6.25) -- Automated head classification is foundational for validating circuit claims
5. **Neuron type detection** (9.3-9.6, 9.24-9.25) -- Memory management, signal boosting, and cleanup neurons are testable claims about MLP function
6. **Ablation method comparison** (6.7-6.8) -- Covered, but ablation method disagreement remains under-measured
7. **Interpretability illusions** (6.37, 9.58) -- No metric for detecting when interpretability methods give misleading results
8. **Algorithmic circuit ground truth** (3.3-3.4, 3.8, 3.15) -- Modular arithmetic circuits provide the closest thing to ground-truth validation
