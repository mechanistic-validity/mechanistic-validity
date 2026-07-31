# Mechanistic Validity — deep-research findings (digested)

Evidence base from the Perplexity deep-research pass, organized by where each finding lands
in v6. Citations to fill into `references.bib`. Gaps flagged for a targeted second pass.

> **Reconciliation with v5 (grep-verified — 44 citation hits).** ~80% of what follows is ALREADY
> cited in v5 and is a PORT job, not new work: Miller, Meloux, MIB (mueller), SAEBench (karvonen),
> Bean, Freiesleben, Hewitt, TracR (lindner), grokking + clock/pizza (nanda), and the whole
> motivation section (salmon/Bennett, ORBITA, candidate genes/Border, COVID/DeGrave, emergent
> abilities/Schaeffer). The v6 `\PORT` markers already point at these — do NOT re-cite them.
> **GENUINELY NEW (0 hits in v5 — the only things to actually ADD):** the two non-circuit cases'
> evidence — SAE canonicity (Leask ICLR 2025, seed-dependence, Braun 2024) and steering fragility
> (Arditi 2024 + CoT-disrupts, adversarial-robustness, inverse-scaling, COSMIC) — plus the 2024–26
> follow-up wave that ENRICHES existing cases (ACDC, "Demystifying Variance," "Have Faith,"
> Adaptive/Adversarial Circuit, "Induction Heads Across Architectures," "Dissociating Decodability").
> Net: the search paid off only for the new cases + fresh follow-ups; the rest was already yours.

## §4 Audit — per-claim evidence

### IOI circuit (Mechanistically Supported)
- **Miller, Chughtai & Saunders 2024**, "Transformer circuit faithfulness metrics are not robust" — the 87% is node-wise; the circuit was specified edge-level; faithfulness swings with ablation methodology. *Strengthens E1 (intervention reach) — the headline number is method-conditional.*
- **"Demystifying Variance in Circuit Discovery"** — circuit-finding varies across runs/templates. *Supports non-uniqueness + prompt-specificity.*
- **ACDC (NeurIPS 2023)** — automated discovery gives overlapping-but-not-identical circuits. *Supports I4/C3 non-uniqueness.*
- **"Have Faith in Faithfulness" (2024)** — circuit-overlap metrics are insufficient. *Foil for the audit's own faithfulness reliance.*
- **Adaptive Circuit Behavior (2024)** — IOI-style circuits shift under distributional generalization. *E2 external validity.*
- **Adversarial Circuit Evaluation (ICLR 2024)** — circuits can be attacked to break claimed faithfulness.
- Meloux et al. 2025 (already cited) — alternative head sets, comparable faithfulness.

### Induction heads (Triangulated)
- **"Induction Head Implementation Across Diverse Architectures"** — mechanism recurs across model families. *Direct E4 cross-model corroboration — backs the Triangulated verdict via the corroboration route.*

### Probing classifiers (Proposed)
- **Hewitt & Liang 2019** — control tasks + selectivity (linguistic minus control accuracy); many ELMo probes not selective. *Anchor.*
- **"Dissociating Decodability and Causal Use in Bracket…"** — title maps directly onto "decodable ≠ used." *Pull in full; it is the cleanest single cite for the Probing verdict.*
- **Pimentel et al.** — info-theoretic critique of control-task design. *Nuance for measurement-validity discussion.*

### Gender-bias circuits (Disconfirmed) — ⚠ EVIDENCE GAP
- No 2024–2026 follow-up found that directly tests whether "bias" is separable from grammatical-gender knowledge. The Disconfirmed verdict currently rests on Vig et al. 2020 + the structural argument. **Decision needed:** either (a) run a direct separability test (candidate for §6), or (b) soften the verdict from a confident "Disconfirmed" to "construct-questionable / Underdetermined pending a separability test." This is the audit's most provocative verdict and its thinnest evidence base — R1 will push here.

## §1 + Related Work — the contribution delta (R2)
- **Bean et al., NeurIPS 2025**, "Measuring what Matters: Construct Validity in LLM benchmarks" — 29-expert review of 445 benchmarks; construct-validity failures; 8 recommendations.
- **ICML 2025 oral position** — medical-LLM benchmarks fail against real clinical data.
- **KEY:** all of this targets benchmarks of *capability*, NOT causal mechanistic claims about *how* a model computes. None applies construct-validity machinery to circuit/feature-level causal claims. **This is external confirmation that your delta is real and untaken — cite it as the delta, not just related work.**

## §6 Self-validation + §5 Standards — preregistration precedent
- **"A Two-Sided Discussion of Preregistration of NLP Research"** — the debate, in-field.
- **"Pre-registration for Predictive Modeling" (2023)**; **2024 dissertation** on preregistration for AI/ML (overfitting, underspecification, shortcut learning).
- **Toth 2021** (registered reports); **RSOS registered report** — preregistration increases trust in findings. *Grounds the prescriptive "standards for the field" in ML-specific precedent, not just CONSORT.*

## §4 Non-circuit — SAE feature
Pivot the case to the claim **"SAE features are canonical, monosemantic units corresponding to concepts."**
- **Leask et al., ICLR 2025** (SAE stitching / meta-SAEs) — SAEs are *incomplete* (bigger SAEs find latents smaller ones miss) and *not atomic* (an "Einstein" latent decomposes into "scientist" + "Germany" + "famous person"). *Textbook C4 discriminant-validity failure — the Einstein example is the case's centerpiece.*
- **Seed-dependence in SAEs** — latent identity varies by training seed. *M1/M3 reliability/stability failure.*
- **Braun et al. 2024** (end-to-end dictionary learning) — targets *functionally important* features, implicitly conceding standard SAE training doesn't guarantee causal use. *Supports "decodable ≠ used" for features.*
- Predicted verdict: **Proposed / Causally Suggestive**, capped by C4 (not atomic) + M1/M3 (seed-dependent).

## §4 Non-circuit — steering vector (refusal direction, Arditi et al. 2024)
- **"Beyond a Single Direction: CoT Disrupts Simple Steering of Refusal" (2026)** — challenges the single-direction narrative under reasoning.
- **"Adversarial Robustness of Activation Steering"**, **"Predicting Where Steering Vectors Succeed"** — distribution-shift / adversarial failure modes (E5/E6).
- **"Inverse Scaling in Activation Steering"** — effectiveness degrades with scale in some regimes (E5 dose-response boundary).
- **COSMIC (2025)** — more general refusal-direction identification; implies the original wasn't fully general (E4).
- **KEY:** the field increasingly narrates refusal as a *behavioral lever*, not a *representation of the concept* — directly your V2 mode-match point, confirmed by the literature's own drift.
- Predicted verdict: **Mechanistically Supported** on the behavioral axis (necessity + sufficiency + dose-response), capped by V2 (lever narrated as mechanism) + E5/E6 fragility.

## §2 Motivation — cross-science primary citations
- **Dead salmon:** Bennett, Baird, Miller & Wolford 2009/2011 (uncorrected p<0.001 voxels in a dead fish; gone under FWE/FDR). Pair with **Vul et al.** "voodoo correlations" (double-dipping).
- **Candidate-gene psychiatry:** *Am. J. Psychiatry* critical review of first 10 years of cGxE — "the more closely a replication matched the original, the less likely it replicated" + publication bias.
- **ORBITA:** Al-Lamee et al., *Lancet* 2017 — first blinded sham-controlled PCI trial; no significant benefit over placebo in stable angina.
- Cold fusion / COVID-ML: use existing citations (not re-verified this pass).

## §6 Ground-truth calibration — RESOLVED
- **Clock/Pizza (Nanda 2023 + "Clock and the Pizza," NeurIPS 2023)** — PRIMARY. Two distinct,
  fully-known ground-truth mechanisms for modular addition, with a Fourier diagnostic (mode at
  2k mod p = Pizza, k = Clock). Lets §6 test tier *discrimination* (claim matches A vs B vs
  neither), not just reachability — a stronger design than a single ground truth.
- **Tracr (Lindner et al. 2023, RASP→weights)** + **Neural Decompiling of Tracr (2024)** —
  SECONDARY. Ground truth by construction; the decompiling inverse is a reference/decoding-fidelity case.

## Related Work — MIB (foil #2) — RESOLVED
- **MIB (ICML 2026)** — two tracks (circuit localization; causal variable localization: SAEs, DAS),
  4 tasks, 5 models. Evaluates method *performance* vs tasks, NOT claim-level validity tiers →
  the delta holds. **Key finding, direct SAE-case ammunition: SAE features perform NO BETTER than
  raw neurons on causal variable localization; supervised DAS performs best.**

## Gender-bias verdict — RESOLVED as a genuine gap
- No 2024–2026 paper re-tests Vig et al. 2020 separability. Closest: 2025 "Universal Patterns of
  Grammatical Gender in Multilingual Models" and a 2026 disentanglement preprint — both frame
  grammatical-vs-social-gender separation as STILL OPEN. **Reframe:** Disconfirmed on C4 — the
  construct presupposes a separation Vig's own mediation contradicts and no work has established;
  do not claim to have proven non-separability. **Option:** make the separability test a NEW §6
  experiment (converts thinnest verdict to empirical + shows the framework generating a test).

## Cold fusion / COVID-ML — RESOLVED (primary sourcing)
- **Cold fusion:** Fleischmann & Pons 1989 (press conference, bypassed peer review) → collapse via
  (1) Coulomb barrier, (2) missing neutrons/gammas vs claimed heat, (3) replication failure
  (MIT/Caltech/Harwell); **Google 2019** rigorous replication still null.
- **COVID-ML:** **DeGrave, Janizek & Lee 2020** — classifiers rely on dataset artifacts, not lung
  pathology; **external validation on new-hospital data PASSES while the causal claim is false**.
  The cleanest single example of external-validity-passes-while-internal-validity-fails — feature it.

## Package 3 — IOI follow-ups + cross-cutting (⚠ spans TWO papers)

**Belongs to the Φ / cross-view-invariance paper, NOT MechVal** (do NOT put in v6): the Φ
recalculation (Merullo pushing Φ above 1.5–2.1), the invariance-depth ladder, the
`cross_view_invariance_v4` worked example. Route to that paper.

**MechVal — genuinely new (ADD):**
- **Adhikari et al. 2026** ("Emergence of Minimal Circuits for IOI," ACL SRW) → IOI non-uniqueness / minimality.
- **Conditional Co-Ablation 2026** + **Self-Repair survey 2026** → MI-NATIVE evidence that backup
  name-mover claims can be ablation-protocol artifacts. Grounds v5's EXISTING "system reserve /
  backup circuits compensate" theme (pharma/genetics lenses, lines 717/731) in MI evidence, not
  just analogy; reinforces E1 method-conditionality. Worth a short field-wide-confound paragraph.
- **⚠ "Causally Grounded Mechanistic Interpretability for LLMs" 2026** → POSSIBLE COMPETITOR for the
  contribution delta. READ IT; distinguish in related work or risk an "already done" reject (R2).
- **Self-Influence (NAACL Findings 2025)** → optional new independent discovery method (evidence-family diversity).

**MechVal — already in v5 (reuse, don't re-cite as new):**
- **Merullo 2024** — already in the bib as a synthesis method; NEW is its evidential USE for IOI
  E4 (reproduces on a larger GPT-2) + E3 (~78% head reuse on Colored Objects). Add to the IOI
  scoring trace (E4 untested → partial).
- **Copy Suppression (mcdougall)** — already case study #4; BlackboxNLP 2024 is the same motif, not a new case.
- **Adaptive Circuit Behavior** — already stubbed (E2).
