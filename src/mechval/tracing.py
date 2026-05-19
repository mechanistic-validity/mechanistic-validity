"""Weave tracing shim — @mv.op that no-ops without Weave installed.

Usage:
    import mechval as mv

    mv.init_tracing("my-project")   # optional — activates Weave logging

    @mv.op
    def my_metric(model, task):
        ...
"""
from __future__ import annotations

import os

_weave_initialized = False


def _is_initialized() -> bool:
    return _weave_initialized


def init(project: str | None = None) -> None:
    global _weave_initialized
    try:
        import weave

        proj = project or os.environ.get("MV_WANDB_PROJECT", "mechanistic-validity")
        weave.init(proj)
        _weave_initialized = True
    except ImportError:
        pass


def op(fn=None):
    if fn is None:
        return op
    if _is_initialized():
        import weave

        return weave.op(fn)
    return fn
