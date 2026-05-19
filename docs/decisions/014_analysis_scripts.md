# ADR-014: Coverage and Comparison Analysis Scripts

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Add two analysis scripts under `scripts/`:

1. **`track3_coverage_report.py`** — Comprehensive gap analysis showing which circuits have specs, what's tested, what's missing
2. **`spec_comparison.py`** — Pairwise circuit comparison for mechanism adjudication (V4)

## Justification

### Coverage report
Without a coverage report, it's hard to know:
- Which circuits still need claim specs
- Which computational steps are never ablated or measured
- Whether attention-only specs are sufficient given superposition risk levels
- Which generator-only tasks need circuit discovery before Track 3

The report identifies concrete gaps like "step X is never measured as an intervention target" or "this spec is attention-only but has medium superposition risk."

### Spec comparison
The epistemic framing task has 4 circuit variants (core, expanded, tight, EAP) discovered by different methods. Comparing them structurally (Jaccard similarity on head sets, role overlap) is the first step toward mechanism adjudication (V4).

This also applies to IOI variants (IOI vs centering_theory vs resumptive — same circuit, different prompt distributions) and RTI variants (rti vs rti_pattern).

## Impact

- `scripts/track3_coverage_report.py`: runs without GPU, introspects all tasks/specs
- `scripts/spec_comparison.py`: pairwise structural comparisons + summary table
- Both are standalone scripts, not part of the library API
