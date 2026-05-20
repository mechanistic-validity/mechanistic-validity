---
title: "Quickstart"
description: "The five-minute version of the framework."
---

# Quickstart

## Install

```bash
pip install mechval
```

For experiment infrastructure (sweep runner, result tracking):

```bash
pip install mechval-lab
```

## Run a metric

```python
import mechval as mv

mv.set_output_dir("./results")
results = mv.run("k_composition", tasks=["ioi"])
```

## List available metrics and tasks

```python
mv.list_metrics()
mv.list_metrics(family="causal")
mv.list_tasks()
mv.list_families()
mv.list_calibrations()
```

## Verify a claim spec

A claim spec is a pre-registered set of predictions about a circuit. `verify()` runs the relevant metrics and checks whether the predictions hold.

```python
task = mv.load_task("ioi")
spec = task.get_claim_spec()
result = mv.verify(spec, device="cpu")
```

## Run a calibration

Calibrations are reference measurements (random baseline, untrained model, etc.) used to contextualize metric scores.

```python
results = mv.calibrate("bootstrap", tasks=["ioi"])
```

## Check status

```python
mv.status()
```
