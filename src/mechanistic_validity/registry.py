"""Task registry — Gymnasium-style load_task() / list_tasks() API.

Built-in tasks are registered at import time. External tasks are
discovered via the ``mechanistic_validity.tasks`` entry-point group.
"""
from __future__ import annotations

import importlib.metadata
import warnings

from mechanistic_validity.lib.tasks.task import CircuitTask

_BUILTIN_TASKS: dict[str, type[CircuitTask]] = {}
_PLUGIN_TASKS: dict[str, type[CircuitTask]] = {}
_PLUGINS_LOADED = False


def _register_builtins() -> None:
    if _BUILTIN_TASKS:
        return
    from mechanistic_validity.lib.tasks._builtins import BUILTIN_TASK_CLASSES
    for cls in BUILTIN_TASK_CLASSES:
        _BUILTIN_TASKS[cls.task_id] = cls


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in importlib.metadata.entry_points(group="mechanistic_validity.tasks"):
        try:
            cls = ep.load()
            if ep.name not in _BUILTIN_TASKS:
                _PLUGIN_TASKS[ep.name] = cls
            else:
                warnings.warn(
                    f"Plugin task {ep.name!r} conflicts with builtin — skipped",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(f"Plugin {ep.name!r} failed to load: {e}", stacklevel=2)


def _all_tasks() -> dict[str, type[CircuitTask]]:
    _register_builtins()
    _load_plugins()
    return {**_PLUGIN_TASKS, **_BUILTIN_TASKS}


def load_task(task_id: str) -> CircuitTask:
    registry = _all_tasks()
    if task_id not in registry:
        raise ValueError(f"Unknown task: {task_id!r}. Available: {sorted(registry)}")
    return registry[task_id]()


def list_tasks(source: str | None = None, model_family: str | None = None) -> list[str]:
    result = []
    for task_id, cls in sorted(_all_tasks().items()):
        if source is not None and cls.source != source:
            continue
        if model_family is not None and cls.model_family != model_family:
            continue
        result.append(task_id)
    return result
