---
name: mechval
description: Load the Mechanistic Validity framework for evaluating mechanistic interpretability claims. Use when designing validation experiments, grading circuit evidence, choosing instruments, auditing claims, or mapping criteria to description modes.
disable-model-invocation: true
---

# Mechanistic Validity Framework

Reference repo: `/Users/elliottower/Documents/GitHub/mechanistic-validity/`

## Project context (live)

```!
cat /Users/elliottower/Documents/GitHub/mechanistic-validity/CLAUDE.md
```

## Current repo state (dynamic)

```!
cd /Users/elliottower/Documents/GitHub/mechanistic-validity && git log --oneline -5 2>/dev/null && echo "---" && echo "Branch: $(git branch --show-current 2>/dev/null)" && echo "---" && echo "Skills:" && ls skills/*/SKILL.md 2>/dev/null && echo "---" && echo "Instrument count:" && find src/instruments -name "*.py" -not -name "_*" -not -name "__*" 2>/dev/null | wc -l && echo "---" && echo "Task count:" && ls src/lib/tasks/ 2>/dev/null | grep -v __pycache__ | wc -l && echo "---" && echo "Criteria docs:" && ls docs/src/content/docs/framework/criteria/*/  2>/dev/null | grep ".md" | wc -l
```

## Criteria checklist

```!
cat /Users/elliottower/Documents/GitHub/mechanistic-validity/skills/mechval/references/criteria-checklist.md
```

## How to use this skill

Once loaded, you have the full framework. Common workflows:

### Audit a circuit claim
1. Read the criteria checklist above — check each criterion against available evidence
2. For specific criteria details: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/criteria/{type}/{criterion}.md`
3. Assign a verdict tier based on evidence profile

### Find the right instrument
1. Read the instrument-to-criteria mapping: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/skills/mechval/references/instruments-by-criteria.md`
2. For instrument docs: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/instruments_v2/{family}/{id}.md`
3. Run scripts with: `uv run python /Users/elliottower/Documents/GitHub/mechanistic-validity/src/instruments/{family}/{instrument}/{script}.py --tasks <task> --device cpu`

### Determine description mode
1. Read modes: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/modes_v3/index.md`
2. Use the decision procedure (in order): I_top → I_con → I_fun → R → A → C
3. Mode = highest level where ALL required evidence is present

### Write a verdict
1. Read verdict tiers: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/verdicts_v3/index.md`
2. Read how-to: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/how-to/write-a-verdict.md`
3. Format: `<component> <computation> in <model> on <task> — <tier> [mode-tag]; <unaddressed types>`

### Check a worked example
Available case studies: ioi, induction-heads, greater-than, grokking, copy-suppression,
successor-heads, docstring, knowledge-neurons, othello, sae-features, superposition,
probing, gender-bias

Read with: `Read /Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/lenses_v6/examples/examples-{name}.md`

## Key paths

| What | Path (relative to repo root) |
|------|------|
| Criteria (all) | `docs/src/content/docs/framework/criteria/` |
| Description modes | `docs/src/content/docs/framework/modes_v3/` |
| Validity types | `docs/src/content/docs/framework/validity-types_v4/` |
| Evidence families | `docs/src/content/docs/framework/evidence-families_v3/` |
| Instrument docs | `docs/src/content/docs/framework/instruments_v2/` |
| Instrument code | `src/instruments/` |
| Shared utilities | `src/instruments/_common.py` |
| Task definitions | `src/lib/tasks/` |
| Verdict tiers | `docs/src/content/docs/framework/verdicts_v3/` |
| Worked examples | `docs/src/content/docs/framework/lenses_v6/examples/` |
| How-to guides | `docs/src/content/docs/how-to/` |
| Taxonomy overview | `docs/src/content/docs/framework/taxonomy/index.md` |

All absolute paths: prefix with `/Users/elliottower/Documents/GitHub/mechanistic-validity/`

## Quick reference

### 7 description modes (partial order)
```
Computational > Algorithmic > Representational > Impl-Functional > Impl-Connectomic > Impl-Topographic
(Impl-Activation-Statistical is orthogonal)
```

### 27 criteria (by validity type)
**Construct (C1-C5):** Falsifiability, Structural Plausibility, Task Specificity, Minimality, Convergent Validity
**Internal (I1-I5):** Necessity, Sufficiency, Specificity, Consistency, Confound Control
**External (E1-E6):** Intervention Reach, Graded Response, Selectivity, Effect Magnitude, Robustness, Cross-Architecture
**Measurement (M1-M6):** Reliability, Invariance, Baseline Separation, Sensitivity, Calibration, Construct Coverage
**Interpretive (V1-V5):** Level Declaration, Level-Evidence Match, Narrative Coherence, Alternative Exclusion, Scope Honesty

### Verdict tiers
1. Proposed → 2. Causally Suggestive → 3. Mechanistically Supported → 4. Triangulated → 5. Validated

### Validity type dependency
Construct → Measurement → Internal → External → Interpretive
