# ADR-001: Pydantic v2 for all models

**Date:** 2025-05-19
**Status:** Implemented

## Decision

Use Pydantic v2 BaseModel for all data models: enums, I/O types, MechanisticClaimSpec, verification results, artifact manifests, gate results, view results, settings.

## Alternatives Considered

1. **Plain dataclasses** (existing approach) — No validation, no JSON schema, no computed fields
2. **attrs** — Good validation but less ecosystem support, no JSON schema generation
3. **Pydantic v1** — Would work but v2 is 5-17x faster and has better computed_field support

## Justification

- JSON schema generation for the CLI `verify` command (load spec from JSON file)
- `computed_field` for `confirmation_rate` and `negative_control_rate` — these should update automatically as predictions are tested
- `model_validate_json()` for deserializing specs from disk
- Pydantic v2 is the standard for Python data validation in 2025
- Needed eventually for pydantic-settings (CLI config)

## Impact

- `pyproject.toml`: added `pydantic>=2.0` to dependencies
- 11 enums defined in `models.py`
- All Track 3 models use BaseModel with Field defaults
- Existing EvalResult dataclass kept as-is (metrics layer) — Pydantic only for the framework layer
