---
title: "Interpretive Validity"
description: "Is the story told about the mechanism the right story, told at the right level of abstraction? — formal specification of V1–V5."
---

# Interpretive Validity — Formal Specification

| | |
|---|---|
| Question | Is the narrative about the mechanism the right narrative, told at the right level of abstraction? |
| Lens | [Mechanistic Interpretability](/framework/lenses_v6/mechanistic_interpretability) |
| Criteria | V1–V5 |
| Foundational framing | [Marr (1982)](https://doi.org/10.7551/mitpress/9780262514620.001.0001) three levels; [Geiger et al. (2021)](https://arxiv.org/abs/2106.02997) causal abstraction; [Méloux et al. (2026)](https://openreview.net/forum?id=vERnMGBqxJ) non-identifiability |
| Dependency | Downstream of all four other validity types — interpretive validity cannot be evaluated until the mechanism is real, generalizable, coherent, and well-measured |
| Status in MI | Only recently addressed; driven by formal non-identifiability results |

Interpretive validity asks whether the narrative told about a mechanism is licensed by the evidence for it. Every MI paper offers some story about what the circuit *does* and why it does it. That story is an interpretive claim. Interpretive validity is the standard by which such claims can be evaluated and distinguished from evidence.

## The Marr level tag

Every verdict in this framework must carry a **Marr level tag** specifying the level at which the claim operates:

- **[computational]** — what problem the mechanism solves (e.g., "identifies the indirect object position")
- **[algorithmic]** — what procedure the mechanism implements (e.g., "copies the IO name from the S-inhibited position to the output")
- **[implementational]** — which specific weights or activations carry the mechanism (e.g., "L9H9's $W_{OV}$ matrix with rank-2 structure implementing the copy")

A verdict without a Marr tag is interpretively incomplete. It makes an implicit level claim that cannot be evaluated. The tag forces commitment to a level, which in turn specifies the minimum evidence required for the claim to be valid.

## V1 — Level Declaration

> [Full criterion page →](/framework/criteria/interpretive/level-declaration)

Every claim must explicitly state which Marr level it operates at.

**Pass condition:** The claim contains an explicit tag — `[computational]`, `[algorithmic]`, or `[implementational]` — or an equivalent explicit statement.

**Why this is non-trivial:** Many MI claims are superficially ambiguous. "L9H9 is a name-mover" could mean:
- *Implementationally:* L9H9's activations causally mediate IOI behavior
- *Algorithmically:* L9H9 implements a copying sub-procedure in the IOI algorithm
- *Computationally:* L9H9's function is to identify and copy indirect objects

These three claims have different evidence requirements. Without declaration, the ablation evidence that licenses the implementational claim is silently presented as if it licensed the algorithmic and computational claims too.

**Common failure — undeclared scope:** A narrative is presented without a level tag. "This head is a name-mover" — is that implementational or algorithmic? Both seem licensed by the same ablation result, and neither has been specifically validated.

## V2 — Level-Evidence Match

> [Full criterion page →](/framework/criteria/interpretive/level-evidence-match)

The evidence presented must be the right type for the level claimed.

**Pass condition:** For each Marr level, the required evidence class is present.

| Level claimed | Required evidence class |
|---|---|
| [computational] | Behavioral characterization across prompt distributions (what problem is solved?) |
| [algorithmic] | IIA / DAS with a stated causal model of the algorithm ([Geiger et al. 2021](https://arxiv.org/abs/2106.02997)) |
| [implementational] | Weight-space signatures + ablation necessity |

**The most common failure — level-evidence mismatch:** Ablation shows L9H9 causally mediates IOI. Claim: "L9H9 implements the name-moving algorithm." The evidence licenses "L9H9 is causally necessary for IOI at the implementational level." The algorithmic claim — that it *implements* a named algorithm — requires causal abstraction (IIA) showing the subspace behaves like the algorithmic variable when swapped across inputs. Without it, the algorithmic story is an interpretation pasted onto implementational evidence, not a finding.

**The upgrade protocol:**

$$I_{\text{imp}} \xrightarrow{\text{weight-space signature}} I_{\text{fun}} \xrightarrow{\text{IIA with causal model}} A \xrightarrow{\text{cross-architecture behavioral}} C$$

Each arrow requires additional evidence beyond what the previous level established.

## V3 — Narrative Coherence

> [Full criterion page →](/framework/criteria/interpretive/narrative-coherence)

The story told about the mechanism must be internally consistent across evidence types.

**Pass condition:** The narrative is consistent across (a) what the weights say, (b) what interventions show, and (c) what the subspace geometry implies. Any single evidence family allows a story that the others might contradict.

**Formal requirement:** For each evidence family $F_i$ used in the narrative, the interpretation $I_i$ derived from $F_i$ must be consistent with the interpretations $I_j$ derived from all other families:

$$\forall i, j: \quad I_i \not\perp I_j$$

**Narrative incoherence failure:** Structural evidence (weight analysis) suggests one mechanism; causal evidence (ablation) suggests another; representational evidence (IIA) is consistent with neither. Each evidence stream is separately valid, but the story constructed from all three is incoherent. This failure is the hardest to notice because each individual stream looks clean.

**MI-specific example:** Weight analysis of L9H9 shows a rank-2 $W_{OV}$ consistent with copying. Ablation shows necessity. IIA shows the subspace does not align with the IO variable when swapped. The weight and ablation evidence suggest a name-mover; the IIA evidence contradicts it. A coherent narrative must either (a) explain the IIA failure or (b) revise the functional label.

## V4 — Alternative Exclusion

> [Full criterion page →](/framework/criteria/interpretive/alternative-exclusion)

Alternative circuits, mechanisms, and interpretations must be tested rather than dismissed.

**Pass condition:** At least one alternative circuit and one alternative interpretation have been tested empirically and either rejected or reported as underdetermined.

**The Méloux result:** [Méloux et al. (2026)](https://openreview.net/forum?id=vERnMGBqxJ) prove that on Boolean MLPs small enough to enumerate exhaustively, multiple circuits replicate the same behavior, multiple interpretations exist for the same circuit, and multiple algorithms can be causally aligned with the same network. This is not a pathological case — it is the generic case. The field's standard practice of presenting one circuit as *the* circuit for a behavior fails alternative exclusion by default.

**The fix is not to find the unique circuit** (which may not exist) but to state explicitly:
1. What alternative circuits were considered
2. How they were tested
3. Whether the finding is underdetermined (multiple circuits replicate the behavior with similar scores)

**Formal requirement:** Let $\mathcal{C}$ be the set of circuits that achieve faithfulness $F \geq F(C^*) - \epsilon$ for the discovered circuit $C^*$. Alternative exclusion requires $|\mathcal{C}|$ to be reported. If $|\mathcal{C}| > 1$, the finding is underdetermined.

**Calibration:** No published MI paper reports $|\mathcal{C}|$ explicitly. Most present a single circuit. The ACDC and EAP results often differ substantially from manual circuit discovery — this is alternative exclusion failure made visible.

## V5 — Scope Honesty

> [Full criterion page →](/framework/criteria/interpretive/scope-honesty)

The narrative must be scoped to what the evidence actually licenses.

**Pass condition:** No claim in the narrative exceeds the evidence level of the weakest evidence type supporting it.

**Formal requirement:** Let $L(E)$ be the Marr level licensed by evidence set $E$, and $L(N)$ be the Marr level claimed by narrative $N$. Scope honesty requires:

$$L(N) \leq L(E)$$

| Evidence available | Maximum licensed claim | Example |
|---|---|---|
| Ablation necessity only | [implementational] | "L9H9 is causally necessary for IOI" |
| Necessity + $W_{OV}$ signature | [implementational-functional] | "L9H9 implements a copying operation at the implementational level" |
| Necessity + $W_{OV}$ + IIA with causal model | [algorithmic] | "L9H9 implements the name-moving step of the IOI algorithm" |
| All above + cross-architecture behavioral | [computational] | "L9H9 solves the indirect object identification sub-problem" |

**The scope creep failure:** A claim migrates upward from implementational evidence to computational conclusions without the additional evidence required. "This head copies tokens" (implementational, licensed by $W_{OV}$ + ablation) vs. "this head implements the induction algorithm" (algorithmic, requires IIA evidence) vs. "this head solves the prefix-matching problem" (computational, requires cross-architecture behavioral evidence). The last two require progressively stronger evidence. Scope creep presents the last two as if licensed by the first.

## Which evidence families bear on interpretive validity

Unlike the other four types, interpretive validity requires cross-family consistency by design:

| Criterion | Evidence families required | Why cross-family is mandatory |
|---|---|---|
| V1 — Level declaration | None — this is a labeling act | No instrument can tell you what level you're claiming; the researcher must declare it |
| V2 — Level-evidence match | Representational (for algorithmic) + Structural (for implementational) | Algorithmic claims require IIA; implementational claims require weight signatures; one cannot substitute for the other |
| V3 — Narrative coherence | Structural + Causal + Representational together | The story must be consistent across all three families simultaneously |
| V4 — Alternative exclusion | Behavioral (test alternatives on new distributions) + Causal (test alternative circuits causally) | Alternatives must be tested empirically, not dismissed conceptually |
| V5 — Scope honesty | Measurement-theoretic (baseline separation determines the strength floor) | The strength of the claim must not exceed what the measurement validity supports |

## Partial-pass interpretation

| Pattern | Criteria met | Interpretation | Recommended language |
|---|---|---|
| Level declared, evidence mismatch | V1 | Honest about level; wrong evidence type for the claim | "Implementational evidence presented for algorithmic claim; IIA needed" |
| Level match, no alternative exclusion | V1, V2 | Evidence type is correct; uniqueness assumed rather than tested | "Algorithmically licensed; alternative circuits not excluded" |
| Coherent narrative, scope creep | V1, V2, V3 | Self-consistent but overclaimed | "Coherent at implementational level; computational claim requires cross-architecture behavioral evidence" |
| All five met | V1–V5 | Full interpretive validity | Claim is licensed at the declared level with alternatives considered |

## Protocol

For claim $N$ at Marr level $L$:

1. **V1.** Add the explicit Marr tag to the claim. State which of the three levels is intended.
2. **V2.** Check that the evidence type matches the level. If algorithmic, confirm IIA with a stated causal model is present. If implementational, confirm weight-space signature + ablation are present.
3. **V3.** Check that structural, causal, and representational evidence streams tell consistent stories. If any stream contradicts, address it explicitly.
4. **V4.** Enumerate at least one alternative circuit and one alternative interpretation. Test them empirically or report that the finding is underdetermined.
5. **V5.** Audit each sentence of the narrative. Does any sentence claim more than the weakest evidence type supports? If so, revise the sentence or add the required evidence.

[^1]: Méloux et al. (2026): multiple circuits replicate the same behavior on Boolean MLPs.
[^2]: The Méloux result does not show that circuit discovery is impossible; it shows that the uniqueness assumption underlying most circuit papers is false. The appropriate response is to report $|\mathcal{C}|$ rather than to abandon circuit discovery.
