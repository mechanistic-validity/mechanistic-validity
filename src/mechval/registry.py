"""Task registry — Gymnasium-style load_task() / list_tasks() API.

Built-in tasks are registered at import time. External tasks are
discovered via the ``mechval.tasks`` entry-point group or registered
programmatically via ``register_task()``.

External packages can register tasks, metrics, and protocols in two ways:

1. **Programmatic** (simplest)::

       from mechval.registry import register_task
       register_task(MyTask)

2. **Entry points** (auto-discovery on install)::

       # In your pyproject.toml:
       [project.entry-points."mechval.tasks"]
       my_task = "my_package.tasks:MyTask"

       [project.entry-points."mechval.metrics"]
       my_metric = "my_package.metrics:run_my_metric"

       [project.entry-points."mechval.protocols"]
       my_protocol = "my_package.protocols:run_protocol"
"""
from __future__ import annotations

import importlib
import importlib.metadata
import warnings

from mechval.lib.tasks.task import CircuitTask

# ── Task registries ───────────────────────────────────────────────────────────

_BUILTIN_TASKS: dict[str, type[CircuitTask]] = {}
_PLUGIN_TASKS: dict[str, type[CircuitTask]] = {}
_EXTERNAL_TASKS: dict[str, type[CircuitTask]] = {}
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

    # Task entry points
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

    # Metric entry points
    for ep in importlib.metadata.entry_points(group="mechval.metrics"):
        try:
            obj = ep.load()
            # obj can be a callable — store as (module_path, fn_name)
            mod_path = obj.__module__
            fn_name = obj.__name__
            if ep.name not in _EXTERNAL_METRICS:
                _EXTERNAL_METRICS[ep.name] = (mod_path, fn_name)
        except Exception as e:
            warnings.warn(f"Metric plugin {ep.name!r} failed to load: {e}", stacklevel=2)

    # Protocol entry points
    for ep in importlib.metadata.entry_points(group="mechval.protocols"):
        try:
            obj = ep.load()
            mod_path = obj.__module__
            fn_name = obj.__name__
            if ep.name not in _PROTOCOL_REGISTRY:
                _PROTOCOL_REGISTRY[ep.name] = (mod_path, fn_name)
        except Exception as e:
            warnings.warn(f"Protocol plugin {ep.name!r} failed to load: {e}", stacklevel=2)


def _all_tasks() -> dict[str, type[CircuitTask]]:
    _register_builtins()
    _load_plugins()
    # Priority: builtins > external (programmatic) > plugins (entry-points)
    return {**_PLUGIN_TASKS, **_EXTERNAL_TASKS, **_BUILTIN_TASKS}


# ── Metric and protocol registries ────────────────────────────────────────────

_EXTERNAL_METRICS: dict[str, tuple[str, str]] = {}
_PROTOCOL_REGISTRY: dict[str, tuple[str, str]] = {}


# ── Public registration API ───────────────────────────────────────────────────

def register_task(task_cls: type[CircuitTask]) -> type[CircuitTask]:
    """Register an external task class.

    Usage::

        from mechval.registry import register_task
        register_task(MyCustomTask)

    Or as a decorator::

        @register_task
        class MyCustomTask(CircuitTask):
            task_id = "my_custom"
            ...

    The task is immediately available via ``load_task()`` and ``list_tasks()``.
    Built-in tasks with the same ``task_id`` take precedence.
    """
    _EXTERNAL_TASKS[task_cls.task_id] = task_cls
    return task_cls


def register_metric(metric_id: str, module_path: str, fn_name: str) -> None:
    """Register an external metric.

    Usage::

        from mechval.registry import register_metric
        register_metric("my_metric", "my_package.metrics", "run_my_metric")

    The metric becomes available via ``mechval.run("my_metric", ...)``.
    Built-in metrics with the same ID take precedence.
    """
    _EXTERNAL_METRICS[metric_id] = (module_path, fn_name)


def register_protocol(protocol_id: str, module_path: str, fn_name: str = "run_protocol") -> None:
    """Register an external protocol.

    Usage::

        from mechval.registry import register_protocol
        register_protocol("my_protocol", "my_package.protocols", "run_protocol")
    """
    _PROTOCOL_REGISTRY[protocol_id] = (module_path, fn_name)


# ── Query API ─────────────────────────────────────────────────────────────────

def get_task(task_id: str) -> type[CircuitTask]:
    """Get a task class by ID (without instantiating)."""
    registry = _all_tasks()
    if task_id not in registry:
        raise ValueError(f"Unknown task: {task_id!r}. Available: {sorted(registry)}")
    return registry[task_id]


def load_task(task_id: str) -> CircuitTask:
    """Get a task instance by ID."""
    return get_task(task_id)()


def list_tasks(
    source: str | None = None,
    model_family: str | None = None,
    domain: str | None = None,
    experiment_group: str | None = None,
    circuit_status: str | None = None,
    has_circuit: bool | None = None,
) -> list[str]:
    """List all registered tasks (built-in + external + plugins) with optional filtering."""
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


def get_external_metrics() -> dict[str, tuple[str, str]]:
    """Return all externally registered metrics (programmatic + entry-point)."""
    _load_plugins()
    return dict(_EXTERNAL_METRICS)


def list_protocols() -> list[str]:
    """List all registered protocols."""
    _load_plugins()
    return sorted(_PROTOCOL_REGISTRY)


def get_protocol(protocol_id: str) -> tuple[str, str]:
    """Get protocol (module_path, fn_name) by ID."""
    _load_plugins()
    if protocol_id not in _PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown protocol: {protocol_id!r}. Available: {list_protocols()}")
    return _PROTOCOL_REGISTRY[protocol_id]


def dispatch_protocol(protocol_id: str, **kwargs):
    """Import and call a registered protocol function."""
    mod_path, fn_name = get_protocol(protocol_id)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)


def list_experiment_groups() -> list[str]:
    return sorted({cls.experiment_group for cls in _all_tasks().values() if cls.experiment_group})


def list_domains() -> list[str]:
    return sorted({cls.domain for cls in _all_tasks().values() if cls.domain})
