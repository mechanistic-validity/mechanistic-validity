---
title: "Contributing"
description: "How to contribute cases, metrics, or corrections."
---

# Contributing

Contributions are welcome across all three repositories.

## Adding a new metric

1. Create a Python file under `src/mechval/metrics/core/<family>/` matching the naming convention of existing metrics
2. Register the metric in `src/mechval/metric_registry.py`
3. Add a docs page under `docs/src/content/docs/framework/metrics/<family>/`
4. Add tests

## Adding a new task

Tasks live in `src/mechval/lib/tasks/_builtins.py`. A task defines a circuit (set of heads/MLPs), a dataset, and evaluation prompts.

## Adding a new experiment

Experiments live in the [mechanistic-validity-experiments](https://github.com/mechanistic-validity/mechanistic-validity-experiments) repository. Create a numbered directory under `experiments/` with a README describing the goal, methods, and TODOs.

## Reporting corrections

If a published baseline value, circuit definition, or citation is incorrect, open an issue on the main repository with the correct value and source.
