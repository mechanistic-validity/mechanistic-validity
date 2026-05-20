# Changelog

All notable changes to mechanistic-validity.

## [Unreleased]

### Changed
- **Metric folder restructure**: all core metrics now live under `metrics/core/` with 6 sub-categories matching the docs site: `behavioral`, `causal`, `information`, `measurement`, `representational`, `structural`. `cognitive_science/` and `cross_discipline/` remain at the `metrics/` level.
- **Registry extracted to `metric_registry.py`**: `METRIC_REGISTRY`, `CALIBRATION_REGISTRY`, `METRIC_FAMILIES`, `list_families()`, `list_metrics()`, `list_calibrations()`, `dispatch()` moved out of `__init__.py`.
- **Spec verification extracted to `spec_verification.py`**: `verify()`, `extract_value()`, `evaluate_prediction()`, `compute_mode_verdicts()`, `compute_claim_ceiling()` moved out of `__init__.py`.
- **Plugin discovery via entry points**: mechval-lab registers itself through `importlib.metadata` entry points (`mechval.plugins` group) instead of a `try: import mechval_lab` in `__init__.py`.
- **`__init__.py` slimmed down**: now just imports + thin `run()`/`calibrate()`/`status()` wrappers.

### Added
- `metrics/core/measurement/` placeholder directory (matches docs site category).
- 21 cross-discipline metrics: ecology (EC1-EC3), physics (PH1-PH3), engineering (EN1-EN4), genetics (GN1-GN5), information theory (IT1-IT3), computational biology (CB1), economics (ECON1-ECON2).
- `origin` and `subcategory` fields on `InstrumentInfo` dataclass. `origin` is `established` (paper-backed) or `experimental` (our contributions). `subcategory` supports e.g. `meta_cognitive` within `cognitive_science`.
- Cognitive science metrics reorganized: EX1-EX11 under `cognitive_science/` subdirectories, SM01-SM10 under `cognitive_science/meta_cognitive/`.

### Removed
- Empty `wildcard/` directory (contents previously moved to `cognitive_science/meta_cognitive/`).

### Fixed
- Hardcoded `importlib.import_module` paths in 3 causal metric files updated after folder move.

### Metrics
- **134 total** (84 core + 21 cognitive science + 29 cross-discipline)
- **14 calibrations** (unchanged)
- **7 families**: behavioral, causal, cognitive_science, cross_discipline, information, representational, structural

## [0.1.0] — 2025-05-19

Initial benchmark infrastructure build. See `CHANGELOG_benchmark_infra.md` for full details.

- 3 Tracks + 4 Views + 4 Gates architecture
- Pydantic v2 models, Cyclopts CLI, pydantic-settings config
- Track 3 (Causal Model Testing) operational end-to-end
- 84 metrics, 14 calibrations, 54 tasks, 9 experiment groups
- W&B Weave tracing shim, artifact adapter pattern (SAE, FactorBank)
