"""Task registry — Gymnasium-style load_task() / list_tasks() API.

Built-in tasks are registered at import time. External tasks are
discovered via the ``mechval.tasks`` entry-point group.
"""
from __future__ import annotations

import importlib.metadata
import warnings

from mechval.lib.tasks.task import CircuitTask

_BUILTIN_TASKS: dict[str, type[CircuitTask]] = {}
_PLUGIN_TASKS: dict[str, type[CircuitTask]] = {}
_PLUGINS_LOADED = False


def _register_builtins() -> None:
    if _BUILTIN_TASKS:
        return
    from mechval.lib.tasks._builtins import BUILTIN_TASK_CLASSES
    for cls in BUILTIN_TASK_CLASSES:
        _BUILTIN_TASKS[cls.task_id] = cls


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in importlib.metadata.entry_points(group="mechval.tasks"):
        try:
            cls = ep.load()
            if ep.name not in _BUILTIN_TASKS:
                _PLUGIN_TASKS[ep.name] = cls
            elif _BUILTIN_TASKS[ep.name].source == "published":
                warnings.warn(
                    f"Plugin task {ep.name!r} conflicts with published builtin — skipped",
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


def list_tasks(
    source: str | None = None,
    model_family: str | None = None,
    domain: str | None = None,
    experiment_group: str | None = None,
    circuit_status: str | None = None,
    has_circuit: bool | None = None,
) -> list[str]:
    result = []
    for task_id, cls in sorted(_all_tasks().items()):
        if source is not None and cls.source != source:
            continue
        if model_family is not None and cls.model_family != model_family:
            continue
        if domain is not None and cls.domain != domain:
            continue
        if experiment_group is not None and cls.experiment_group != experiment_group:
            continue
        if circuit_status is not None and cls.circuit_status != circuit_status:
            continue
        if has_circuit is True and cls.circuit_status not in ("full_circuit", "proxy_circuit"):
            continue
        if has_circuit is False and cls.circuit_status in ("full_circuit", "proxy_circuit"):
            continue
        result.append(task_id)
    return result


def list_experiment_groups() -> list[str]:
    return sorted({cls.experiment_group for cls in _all_tasks().values() if cls.experiment_group})


def list_domains() -> list[str]:
    return sorted({cls.domain for cls in _all_tasks().values() if cls.domain})
