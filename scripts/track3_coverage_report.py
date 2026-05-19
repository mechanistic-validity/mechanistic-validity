"""Track 3 coverage report — shows which circuits have claim specs,
what predictions exist, and identifies gaps.

Usage:
    python scripts/track3_coverage_report.py
"""
import sys
sys.path.insert(0, "src")

import mechval


def main():
    task_ids = mechval.list_tasks()

    circuits_with_specs = []
    circuits_without_specs = []
    generator_only = []

    for tid in task_ids:
        task = mechval.load_task(tid)
        if task.circuit_status == "generator_only":
            generator_only.append(task)
            continue
        spec = task.get_claim_spec()
        if spec:
            circuits_with_specs.append((task, spec))
        else:
            circuits_without_specs.append(task)

    print("=" * 72)
    print("TRACK 3 COVERAGE REPORT")
    print("=" * 72)

    print(f"\nTotal tasks: {len(task_ids)}")
    print(f"  With claim specs:     {len(circuits_with_specs)}")
    print(f"  Without claim specs:  {len(circuits_without_specs)}")
    print(f"  Generator-only:       {len(generator_only)}")

    print("\n" + "-" * 72)
    print("CIRCUITS WITH CLAIM SPECS (Track 3 ready)")
    print("-" * 72)

    total_predictions = 0
    total_neg_controls = 0
    total_steps = 0
    total_edges = 0

    for task, spec in circuits_with_specs:
        n_pred = len(spec.predictions)
        n_neg = len(spec.negative_controls)
        n_steps = len(spec.steps)
        n_edges = len(spec.edges)
        total_predictions += n_pred
        total_neg_controls += n_neg
        total_steps += n_steps
        total_edges += n_edges

        component_types = set()
        for step in spec.steps:
            component_types.update(step.component_types)

        print(f"\n  {task.task_id} ({spec.author})")
        print(f"    Steps: {n_steps} | Edges: {n_edges} | "
              f"Predictions: {n_pred} | Neg Controls: {n_neg}")
        print(f"    Component types: {', '.join(component_types) or 'none'}")
        print(f"    Superposition risk: {spec.superposition_risk.polysemanticity_risk}")
        print(f"    Identifiability: {spec.identifiability.status.value}")

        print(f"    Steps:")
        for step in spec.steps:
            heads_str = f"{len(step.maps_to_heads)} heads" if step.maps_to_heads else ""
            mlps_str = f"{len(step.maps_to_mlps)} MLPs" if step.maps_to_mlps else ""
            neurons_str = f"{len(step.maps_to_neurons)} neurons" if step.maps_to_neurons else ""
            parts = [p for p in [heads_str, mlps_str, neurons_str] if p]
            print(f"      - {step.name} ({step.category}): {', '.join(parts)}")

        print(f"    Predictions:")
        for pred in spec.predictions:
            print(f"      + {pred.name}: ablate({pred.intervention_target})→"
                  f"{pred.measurement_target} [{pred.expected_direction.value}] "
                  f"threshold={pred.expected_threshold}")
        for pred in spec.negative_controls:
            print(f"      - {pred.name}: ablate({pred.intervention_target})→"
                  f"{pred.measurement_target} [invariant] (negative control)")

    print(f"\n    TOTALS: {total_steps} steps, {total_edges} edges, "
          f"{total_predictions} predictions, {total_neg_controls} negative controls")

    print("\n" + "-" * 72)
    print("CIRCUITS WITHOUT CLAIM SPECS (could add Track 3 specs)")
    print("-" * 72)

    for task in circuits_without_specs:
        try:
            circuit = task.get_circuit()
            roles = circuit.roles
            n_heads = sum(len(v) for v in roles.values())
            role_names = list(roles.keys())
            print(f"\n  {task.task_id}")
            print(f"    Domain: {task.domain} | Group: {task.experiment_group}")
            print(f"    Roles ({len(role_names)}): {', '.join(role_names)}")
            print(f"    Total heads: {n_heads}")
        except (NotImplementedError, AttributeError):
            print(f"\n  {task.task_id}")
            print(f"    Domain: {task.domain} | Group: {task.experiment_group}")
            print(f"    [no circuit available]")

    print("\n" + "-" * 72)
    print("GENERATOR-ONLY TASKS (need circuits before Track 3)")
    print("-" * 72)

    by_group = {}
    for task in generator_only:
        by_group.setdefault(task.experiment_group, []).append(task)
    for group, group_tasks in sorted(by_group.items()):
        task_ids = [t.task_id for t in group_tasks]
        print(f"\n  {group} ({len(group_tasks)} tasks):")
        for tid in task_ids:
            print(f"    - {tid}")

    print("\n" + "-" * 72)
    print("GAP ANALYSIS")
    print("-" * 72)

    gaps = []

    for task, spec in circuits_with_specs:
        has_mlp = any(step.maps_to_mlps for step in spec.steps)
        has_neurons = any(step.maps_to_neurons for step in spec.steps)
        has_features = any(step.maps_to_features for step in spec.steps)
        all_head_only = not has_mlp and not has_neurons and not has_features

        if all_head_only and spec.superposition_risk.polysemanticity_risk != "low":
            gaps.append(f"  {task.task_id}: attention-only spec but "
                        f"{spec.superposition_risk.polysemanticity_risk} superposition risk "
                        f"— consider adding MLP/neuron components")

        pred_targets = {p.measurement_target for p in spec.predictions}
        neg_targets = {p.measurement_target for p in spec.negative_controls}
        step_names = set(spec.step_names())
        untested_as_measurement = step_names - pred_targets - neg_targets - {"output"}
        if untested_as_measurement:
            gaps.append(f"  {task.task_id}: steps never measured: "
                        f"{', '.join(untested_as_measurement)}")

        intervention_targets = {p.intervention_target for p in spec.all_predictions()}
        untested_as_intervention = step_names - intervention_targets
        if untested_as_intervention:
            gaps.append(f"  {task.task_id}: steps never ablated: "
                        f"{', '.join(untested_as_intervention)}")

    for task in circuits_without_specs:
        if task.circuit_status == "full_circuit":
            gaps.append(f"  {task.task_id}: has full circuit but no claim spec")

    if gaps:
        for gap in gaps:
            print(gap)
    else:
        print("  No gaps found!")

    print("\n" + "-" * 72)
    print("CROSS-CIRCUIT PREDICTION MATRIX")
    print("-" * 72)
    print("\n  Which prediction types appear across specs:\n")

    pred_types = {}
    for task, spec in circuits_with_specs:
        for pred in spec.all_predictions():
            key = (pred.intervention_target.split("_")[-1] if "_" in pred.intervention_target
                   else pred.intervention_target,
                   pred.measurement_target,
                   pred.expected_direction.value)
            pred_types.setdefault(key, []).append(task.task_id)

    print(f"  {'Pattern':<45} {'Circuits'}")
    print(f"  {'─' * 45} {'─' * 25}")
    for (int_t, meas_t, direction), task_ids in sorted(pred_types.items()):
        pattern = f"ablate({int_t})→{meas_t} [{direction}]"
        print(f"  {pattern:<45} {', '.join(task_ids)}")

    print()


if __name__ == "__main__":
    main()
