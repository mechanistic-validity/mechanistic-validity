# ADR-012: Weave Tracing as Optional Zero-Cost Shim

**Date:** 2025-05-19
**Status:** Implemented

## Decision

`@mv.op` is a decorator shim: when `mv.init_tracing()` has been called and `weave` is installed, it wraps functions with `@weave.op` for W&B Weave tracing. Otherwise it's a no-op identity decorator.

## Justification

- Zero hard dependency: `wandb` and `weave` are optional (`pip install mechanistic-validity[weave]`)
- Zero runtime cost when not initialized: `@mv.op` returns the unwrapped function
- When active, all `mv.run()` / `mv.verify()` / `mv.calibrate()` calls are traced with inputs, outputs, and timing
- Users can decorate their own custom metrics with `@mv.op`

## Alternatives Considered

1. **Hard wandb dependency** — Forces all users to install wandb even for local-only use
2. **Custom logging** — Reinvents what Weave already does
3. **No tracing** — Makes debugging and reproducibility harder in team settings

## Implementation

```python
def op(fn=None):
    if fn is None:
        return op
    if _is_initialized():
        import weave
        return weave.op(fn)
    return fn
```

## Impact

- `tracing.py`: ~25 lines
- `__init__.py`: `mv.op`, `mv.init_tracing()` exposed
- `pyproject.toml`: `[weave]` optional dependency group
