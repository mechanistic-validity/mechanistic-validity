# ADR-005: Linguistics-Based Task Taxonomy

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Migrate task domains from ad-hoc labels to a linguistics-grounded taxonomy with 10 domains and 9 experiment groups.

### Domain mapping
| Old domain | New domain | Rationale |
|------------|-----------|-----------|
| entity_tracking | linguistics_coreference | IOI, centering theory, resumptive pronouns are coreference phenomena |
| agreement | linguistics_agreement | SVA, gendered pronouns are agreement phenomena |
| repetition | patterns | RTI and induction are pattern-matching, not linguistic phenomena per se |
| numerical | math | Greater-than is mathematical comparison |
| structural | patterns or linguistics_syntax | Depends on the specific task |
| discourse | linguistics_pragmatics or linguistics_semantics | Epistemic tasks are pragmatic; negation/sentiment are semantic |

### New domains
- `linguistics_binding` — reflexive anaphora, BLiMP binding phenomena
- `linguistics_morphology` — BLiMP irregular forms
- `linguistics_phonology` — phonetic composition tasks
- `linguistics_syntax` — filler-gap, NPI licensing, bracket matching, BLiMP syntax tasks

### Experiment groups
published, rti_discovery, ioi_ablations, repetition_taxonomy, epistemic_study, linguistic_probes, blimp, phonetic_composition, roadmap

## Alternatives Considered

1. **Keep ad-hoc labels** — No alignment with linguistic theory, hard to communicate in papers
2. **Chomskyan syntactic hierarchy** — Too syntax-focused, misses semantics/pragmatics/phonology
3. **BLiMP-only taxonomy** — Too narrow, only covers some tasks

## Justification

- Aligns with standard linguistics subfield boundaries
- Makes the benchmark legible to linguists and cognitive scientists
- Each domain corresponds to a distinct type of linguistic knowledge the model must possess
- Experiment groups capture provenance (where did these tasks come from?) while domains capture content (what linguistic phenomenon do they test?)

## Impact

- `_builtins.py`: all 36 existing tasks re-tagged + 18 new task classes added
- Total: 36 -> 54 tasks, 6 -> 10 domains, 7 -> 9 experiment groups
