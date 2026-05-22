"""Role-targeted ablation — Track 3 interventional metric.

Given a MechanisticClaimSpec prediction, ablates the intervention_target
role's heads and measures the effect at the measurement_target (another
role or "output"). Returns the normalized effect size.

This is the metric that makes mv.verify() execute real causal claims.

Usage:
    Typically called by mv.verify() rather than mv.run() directly.
    For standalone use:
        mv run role_ablation --tasks ioi
"""

import numpy as np
import torch

from mechval.metrics.common import (
    EvalResult,
    calibrate_mean_z,
    generate_prompts,
    get_circuit_heads,
    get_token_ids,
    heads_to_layer_dict,
    load_model,
    log,
    logit_diff_from_logits,
    make_ablation_hook,
)


def _get_role_heads(task: str, role_name: str) -> set[tuple[int, int]]:
    """Get heads for a specific role from the circuit spec."""
    from mechval.registry import load_task
    t = load_task(task)
    circuit = t.get_circuit()
    roles = circuit.roles
    if role_name in roles:
        return set(tuple(h) for h in roles[role_name])
    for step_name, step_role in _step_to_role_mapping(task).items():
        if step_name == role_name and step_role in roles:
            return set(tuple(h) for h in roles[step_role])
    return set()


def _step_to_role_mapping(task: str) -> dict[str, str]:
    """Map computational step names to circuit role names via claim spec."""
    from mechval.registry import load_task
    t = load_task(task)
    spec = t.get_claim_spec()
    if spec is None:
        return {}
    return {step.name: step.maps_to_role for step in spec.steps if step.maps_to_role}


def _get_step_components(task: str, step_name: str) -> dict:
    """Get all component types (heads, mlps, neurons) for a step."""
    from mechval.registry import load_task
    t = load_task(task)
    spec = t.get_claim_spec()
    if spec is None:
        return {"heads": set(), "mlps": [], "neurons": []}
    for step in spec.steps:
        if step.name == step_name:
            heads = set(tuple(h) for h in step.maps_to_heads)
            if not heads and step.maps_to_role:
                circuit = t.get_circuit()
                if step.maps_to_role in circuit.roles:
                    heads = set(tuple(h) for h in circuit.roles[step.maps_to_role])
            return {
                "heads": heads,
                "mlps": list(step.maps_to_mlps),
                "neurons": [tuple(n) for n in step.maps_to_neurons],
            }
    role_map = _step_to_role_mapping(task)
    if step_name in role_map:
        circuit = t.get_circuit()
        role = role_map[step_name]
        heads = set(tuple(h) for h in circuit.roles.get(role, []))
        mlps = list(circuit.mlp_nodes.get(role, []))
        return {"heads": heads, "mlps": mlps, "neurons": []}
    return {"heads": set(), "mlps": [], "neurons": []}


def _make_mlp_ablation_hooks(mlp_layers: list[int], ablation_type: str = "zero"):
    """Create hooks to ablate full MLP layers."""
    hooks = []
    for layer in mlp_layers:
        def _hook(mlp_out, hook, _atype=ablation_type):
            if _atype == "zero":
                mlp_out[:] = 0.0
            elif _atype == "mean":
                mlp_out[:] = mlp_out.mean(dim=1, keepdim=True)
            return mlp_out
        hooks.append((f"blocks.{layer}.hook_mlp_out", _hook))
    return hooks


def _make_neuron_ablation_hooks(neurons: list[tuple[int, int]], ablation_type: str = "zero"):
    """Create hooks to ablate individual MLP neurons."""
    by_layer: dict[int, list[int]] = {}
    for layer, neuron_idx in neurons:
        by_layer.setdefault(layer, []).append(neuron_idx)
    hooks = []
    for layer, neuron_list in by_layer.items():
        def _hook(pre_act, hook, _neurons=neuron_list, _atype=ablation_type):
            for n in _neurons:
                if _atype == "zero":
                    pre_act[:, :, n] = 0.0
                elif _atype == "mean":
                    pre_act[:, :, n] = pre_act[:, :, n].mean()
            return pre_act
        hooks.append((f"blocks.{layer}.mlp.hook_pre", _hook))
    return hooks


@torch.no_grad()
def _measure_role_output(model, tokens, correct_id, incorrect_id,
                         target_heads: set[tuple[int, int]]) -> float:
    """Measure the total attention output (hook_z) contribution from target heads to logit diff."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
    total = 0.0
    for L, H in target_heads:
        z = cache[f"blocks.{L}.attn.hook_z"]
        total += z[0, -1, H].sum().item()
    return total


@torch.no_grad()
def run_role_ablation(model, tasks: list[str], n_prompts: int = 40,
                      intervention_target: str | None = None,
                      measurement_target: str | None = None,
                      ablation_type: str = "zero") -> list[EvalResult]:
    """Ablate one role's heads, measure effect on another role or output.

    If intervention_target/measurement_target are not provided, runs a
    comprehensive all-role ablation scan.
    """
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit, skipping")
            continue

        prompts = generate_prompts(task, tokenizer, n_prompts)
        if not prompts:
            log(f"  {task}: no prompts, skipping")
            continue

        correct_ids, incorrect_ids = get_token_ids(prompts, tokenizer)
        if not correct_ids:
            continue

        mean_z = calibrate_mean_z(model, prompts)
        step_role_map = _step_to_role_mapping(task)
        from mechval.registry import load_task
        circuit = load_task(task).get_circuit()
        roles = circuit.roles

        if intervention_target and measurement_target:
            result = _run_targeted(
                model, task, prompts, correct_ids, incorrect_ids,
                mean_z, roles, step_role_map,
                intervention_target, measurement_target, ablation_type,
                n_layers, n_heads,
            )
            if result:
                results.append(result)
        else:
            scan_results = _run_full_scan(
                model, task, prompts, correct_ids, incorrect_ids,
                mean_z, roles, step_role_map, ablation_type,
                n_layers, n_heads,
            )
            results.extend(scan_results)

    return results


@torch.no_grad()
def _run_targeted(model, task, prompts, correct_ids, incorrect_ids,
                  mean_z, roles, step_role_map,
                  intervention_target, measurement_target, ablation_type,
                  n_layers, n_heads) -> EvalResult | None:
    """Ablate intervention_target, measure effect at measurement_target."""
    components = _get_step_components(task, intervention_target)
    int_heads = components["heads"]
    int_mlps = components["mlps"]
    int_neurons = components["neurons"]

    int_role = step_role_map.get(intervention_target, intervention_target)
    if not int_heads:
        if int_role in roles:
            int_heads = set(tuple(h) for h in roles[int_role])

    if not int_heads and not int_mlps and not int_neurons:
        log(f"  {task}: no components found for {intervention_target!r}")
        return None

    hooks = []
    if int_heads:
        int_by_layer = heads_to_layer_dict(int_heads)
        hooks.extend(make_ablation_hook(int_by_layer, mean_z, ablation_type))
    if int_mlps:
        hooks.extend(_make_mlp_ablation_hooks(int_mlps, ablation_type))
    if int_neurons:
        hooks.extend(_make_neuron_ablation_hooks(int_neurons, ablation_type))

    if measurement_target == "output":
        clean_diffs, ablated_diffs = [], []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_logits = model(tokens)
            clean_diffs.append(logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i]))
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_diffs.append(logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i]))

        clean_mean = np.mean(clean_diffs)
        ablated_mean = np.mean(ablated_diffs)
        if abs(clean_mean) < 1e-8:
            effect = 0.0
        else:
            effect = (ablated_mean - clean_mean) / abs(clean_mean)

        log(f"  {task}: ablate({intervention_target})→output: "
            f"clean={clean_mean:.3f} ablated={ablated_mean:.3f} effect={effect:.3f}")

        return EvalResult(
            metric_id="role_ablation",
            value=effect,
            n_samples=len(clean_diffs),
            metadata={
                "task": task,
                "intervention_target": intervention_target,
                "intervention_role": int_role,
                "measurement_target": measurement_target,
                "clean_mean": float(clean_mean),
                "ablated_mean": float(ablated_mean),
                "ablation_type": ablation_type,
            },
        )
    else:
        meas_components = _get_step_components(task, measurement_target)
        meas_heads = meas_components["heads"]
        meas_mlps = meas_components["mlps"]
        meas_neurons = meas_components["neurons"]

        meas_role = step_role_map.get(measurement_target, measurement_target)
        if not meas_heads:
            if meas_role in roles:
                meas_heads = set(tuple(h) for h in roles[meas_role])

        if not meas_heads and not meas_mlps and not meas_neurons:
            log(f"  {task}: no measurement components for {measurement_target!r}")
            return None

        hook_names_to_capture = set()
        for L, H in meas_heads:
            hook_names_to_capture.add(f"blocks.{L}.attn.hook_z")
        for layer in meas_mlps:
            hook_names_to_capture.add(f"blocks.{layer}.hook_mlp_out")
        for layer, _ in meas_neurons:
            hook_names_to_capture.add(f"blocks.{layer}.mlp.hook_post")

        clean_effects, ablated_effects = [], []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)

            _, clean_cache = model.run_with_cache(
                tokens, names_filter=lambda n, _names=hook_names_to_capture: n in _names,
            )
            clean_val = _measure_components(clean_cache, meas_heads, meas_mlps, meas_neurons)
            clean_effects.append(clean_val)

            captured = {}
            def _capture_hook(act, hook, _captured=captured):
                _captured[hook.name] = act.detach().clone()
                return act
            capture_hooks = [(name, _capture_hook) for name in hook_names_to_capture]
            all_hooks = hooks + capture_hooks
            model.run_with_hooks(tokens, fwd_hooks=all_hooks)
            ablated_val = _measure_components(captured, meas_heads, meas_mlps, meas_neurons)
            ablated_effects.append(ablated_val)

        clean_mean = np.mean(clean_effects)
        ablated_mean = np.mean(ablated_effects)
        if abs(clean_mean) < 1e-8:
            effect = 0.0
        else:
            effect = (ablated_mean - clean_mean) / abs(clean_mean)

        log(f"  {task}: ablate({intervention_target})→{measurement_target}: "
            f"clean={clean_mean:.3f} ablated={ablated_mean:.3f} effect={effect:.3f}")

        return EvalResult(
            metric_id="role_ablation",
            value=effect,
            n_samples=len(clean_effects),
            metadata={
                "task": task,
                "intervention_target": intervention_target,
                "intervention_role": int_role,
                "measurement_target": measurement_target,
                "measurement_role": meas_role,
                "clean_mean": float(clean_mean),
                "ablated_mean": float(ablated_mean),
                "ablation_type": ablation_type,
            },
        )


def _measure_components(cache_or_dict, heads: set[tuple[int, int]],
                        mlps: list[int], neurons: list[tuple[int, int]]) -> float:
    """Sum activation norms from heads, MLPs, and neurons at the last position."""
    total = 0.0
    for L, H in heads:
        key = f"blocks.{L}.attn.hook_z"
        if key in cache_or_dict:
            total += cache_or_dict[key][0, -1, H].norm().item()
    for layer in mlps:
        key = f"blocks.{layer}.hook_mlp_out"
        if key in cache_or_dict:
            total += cache_or_dict[key][0, -1].norm().item()
    for layer, neuron_idx in neurons:
        key = f"blocks.{layer}.mlp.hook_post"
        if key in cache_or_dict:
            total += abs(cache_or_dict[key][0, -1, neuron_idx].item())
    return total


@torch.no_grad()
def _run_full_scan(model, task, prompts, correct_ids, incorrect_ids,
                   mean_z, roles, step_role_map, ablation_type,
                   n_layers, n_heads) -> list[EvalResult]:
    """Ablate each role, measure logit diff effect."""
    results = []
    for role_name, heads in roles.items():
        role_heads = set(tuple(h) for h in heads)
        by_layer = heads_to_layer_dict(role_heads)
        hooks = make_ablation_hook(by_layer, mean_z, ablation_type)

        clean_diffs, ablated_diffs = [], []
        for i, p in enumerate(prompts):
            if i >= len(correct_ids):
                break
            tokens = model.to_tokens(p.text)
            clean_logits = model(tokens)
            clean_diffs.append(logit_diff_from_logits(clean_logits, correct_ids[i], incorrect_ids[i]))
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_diffs.append(logit_diff_from_logits(ablated_logits, correct_ids[i], incorrect_ids[i]))

        clean_mean = np.mean(clean_diffs)
        ablated_mean = np.mean(ablated_diffs)
        if abs(clean_mean) < 1e-8:
            effect = 0.0
        else:
            effect = (ablated_mean - clean_mean) / abs(clean_mean)

        log(f"  {task}/{role_name}: clean={clean_mean:.3f} ablated={ablated_mean:.3f} effect={effect:.3f}")

        results.append(EvalResult(
            metric_id="role_ablation",
            value=effect,
            n_samples=len(clean_diffs),
            metadata={
                "task": task,
                "role": role_name,
                "n_heads": len(role_heads),
                "heads": sorted(role_heads),
                "clean_mean": float(clean_mean),
                "ablated_mean": float(ablated_mean),
                "ablation_type": ablation_type,
            },
        ))

    return results
