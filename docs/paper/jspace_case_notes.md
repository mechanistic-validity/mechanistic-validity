# J-space case study — notes (real detail; lean scaffold lives in v6 §4)

Source: your two audits — `Anthropic J audit mechval.pdf` (full MechVal + cross-framework) and
`Anthropic J critique .pdf` (mech-interp vs philosophy). This file holds the **MechVal slice**.
Cross-framework material (invariance-depth Δ, unification, MechViews view-slide, Grassmannian) is
flagged at the bottom for your OTHER papers — keep it OUT of MechVal (one thesis per paper).

## The claim
Anthropic 2026, "Verbalizable Representations Form a Global Workspace in Language Models"
(transformer-circuits.pub/2026/workspace). Method = **Jacobian lens (J-lens)**: top-k directions of
the average Jacobian of output logits w.r.t. activations, across a context distribution → the
**J-space** (<10% of variance/layer), narrated as a "global workspace / center of Claude's mind."

## MechVal verdict: **Mechanistically Supported** (same tier as IOI)
Necessity + sufficiency established via THREE converging interventions (swap / ablate / inject).
Caps below Triangulated on: (a) single evidence family [C3], (b) I3 specificity untested,
(c) I4 double dissociation untested, (d) V4 interpretive inflation.

## Criterion trace (MechVal slice)
- **Description mode — MODE CONFLATION (the reason this case matters):** implementational-statistical/
  topographic (a named subspace) → representational (encodes "reportable" state) → computational
  ("workspace doing deliberate reasoning"). Evidence ceiling = representational; narrative =
  computational. THE flagship V1/V2 + V4 example — better than Othello (timely, high-profile, safety-relevant).
- **C1** partial (operationally defined + intervention predictions tested; but the GWT narrative is
  not independently falsifiable — any high-influence subspace could be called a "workspace").
- **C2** confirmed (a causal residual-stream subspace is architecturally plausible).
- **C3** partial — Nanda's Qwen replication adds a MODEL, not an independent evidence FAMILY; still
  mostly interventional-activation single-cell; no weight-space (SVD/composition) evidence.
- **C4 UNTESTED — the key gap.** Is J-space separable from the model's general high-influence
  subspace? No size-matched random-high-influence baseline. (= the cardiac-stent problem.)
- **C5** weak — GWT contested; **COGITATE 2023/2025** adversarial collaboration: GNW's frontal-ignition
  prediction FAILED. Dehaene & Naccache themselves: "ignition remains to be demonstrated"; "J-space
  capacity seems high" vs GWT's ~3–4-item bottleneck. Transformer is feedforward (no recurrence) — IIT
  would assign near-zero Φ by construction.
- **M1** untested; **M2** partial (interventions beat correlation, but no null model for "workspace vs
  high-gradient subspace"); M3–M6 mostly untested.
- **I1** confirmed (ablation collapses multi-hop reasoning, fluency survives). **I2** confirmed
  (swap/inject change what the model reasons/reports).
- **I3 UNTESTED — most actionable hole.** Does ablation hurt ONLY deliberate tasks, or everything that
  coordinates many representations? If the latter, it's a general high-influence bottleneck, not a "workspace."
- **I4** untested; **I5** partial (task-difficulty confound not formally ruled out).
- **E1** partial (swap/ablate/inject converge, but same family); **E2** partial; **E4** partial —
  Nanda Qwen = ROLE-level transport at best, not subspace-identity transport.
- **V4 FAILED** — "mental workspace," "center of its mind," "access consciousness" import GWT
  commitments (Object→Role→Computational slide). **V3** not excluded (simpler "high-eigenvalue subspace
  of the average Jacobian" explanation not ruled out).

## Sharpest single point (novel — worth foregrounding in the C-level discussion, not just this case)
**"The measurement operationalizes the construct."** J-space is DEFINED as the reportable
(max-output-influence) subspace, so its correlation with deliberate reasoning is guaranteed by
construction — a C4/C1 circularity. A clean, general pattern the framework catches.

## Safety tie-in (feed §5)
The deception/misalignment results (J-space leaks "fake"/"secretly"/"fraud"; Sonnet 4.5 blackmail
scenario shows strategic good behavior) are the STRONGEST part, sit closer to Triangulated, and need
NO GWT narrative. Live, dated instance of the minimum-deployment argument: Nanda says it needs
independent replication before being trusted as a safety monitor → "don't deploy a Mechanistically-
Supported monitor as if it were Validated."

## Citations to add (bib)
- Anthropic 2026, J-space paper (transformer-circuits.pub/2026/workspace).
- Nanda (DeepMind) — Qwen open-weight cross-lab replication.
- Baars 1988 / Dehaene & Naccache — Global Workspace Theory (+ their J-space commentary).
- COGITATE adversarial collaboration 2023/2025 — GNW vs IIT; GNW frontal-ignition prediction failed (C5 anchor).

## BONUS — SAE-case enrichment (from the critique doc; strengthens §4 SAE beyond Leask)
- **"Sanity Checks for Sparse Autoencoders" (2026):** SAEs recover only ~9% of true features on
  synthetic ground truth despite 71% explained variance; **RANDOM frozen baselines match trained SAEs**
  on interpretability / sparse-probing / causal-editing metrics → devastating M2 baseline-separation failure.
- **GDM mech-interp (Lewis Smith, Nanda et al.):** SAEs underperform LINEAR PROBES on harmful-intent
  detection OOD; team "deprioritized fundamental SAE research," "field over-invested in SAEs."
- **Ma et al., NeurIPS 2025, "Revising and Falsifying SAE Feature Explanations":** autointerp labels
  overly broad, miss polysemanticity, eval biases inflate apparent performance.
- Together with Leask (not-atomic) + seed-dependence, the SAE case becomes very strong. Nuance the J-space
  case is NOT SAE-based (J-lens has real interventional validation) — the SAE critique targets the PRIOR
  Anthropic paradigm, not J-space.

## Cross-framework — DO NOT put in MechVal v6 (route to the other papers)
- Invariance-depth Δ ≈ 1.5, 4-family ladder (object/subspace/structural/process) → `cross_view_invariance`.
- Unification (v ≈ 0.3–0.4, Cov unknown, Reference role-level) → `mechanistic_unification`.
- MechViews view-slide (Object→Role→Computational), Subspace-view + Grassmannian stability → `mechanistic_views` / `mechanistic_knowledge`.
- EAP-vs-J-space comparison (same gradient-based hybrid cell) → views / reference papers.
