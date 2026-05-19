"""Compare circuit specs for the same task — Track 3 adjudication.

When multiple circuit hypotheses exist for the same task (e.g., different
discovery methods), this script compares them side-by-side: component overlap,
prediction overlap, structural differences.

Usage:
    python scripts/spec_comparison.py
"""
import sys
sys.path.insert(0, "src")

import mechval


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_circuits(task_id_a: str, task_id_b: str):
    """Compare two circuits structurally."""
    task_a = mechval.load_task(task_id_a)
    task_b = mechval.load_task(task_id_b)
    circuit_a = task_a.get_circuit()
    circuit_b = task_b.get_circuit()

    heads_a = {h for heads in circuit_a.roles.values() for h in heads}
    heads_b = {h for heads in circuit_b.roles.values() for h in heads}

    overlap = heads_a & heads_b
    only_a = heads_a - heads_b
    only_b = heads_b - heads_a
    j = jaccard(heads_a, heads_b)

    return {
        "task_a": task_id_a,
        "task_b": task_id_b,
        "heads_a": len(heads_a),
        "heads_b": len(heads_b),
        "overlap": len(overlap),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "jaccard": j,
        "overlap_heads": sorted(overlap),
        "roles_a": list(circuit_a.roles.keys()),
        "roles_b": list(circuit_b.roles.keys()),
    }


def main():
    print("=" * 72)
    print("CIRCUIT COMPARISON — Alternative Hypotheses for Same Tasks")
    print("=" * 72)

    comparison_groups = {
        "Epistemic Framing": [
            "epistemic_framing",
            "epistemic_expanded",
            "epistemic_tight",
            "epistemic_eap",
        ],
        "IOI Variants": [
            "ioi",
            "centering_theory",
            "resumptive",
        ],
        "RTI Variants": [
            "rti",
            "rti_pattern",
        ],
    }

    for group_name, task_ids in comparison_groups.items():
        print(f"\n{'─' * 72}")
        print(f"  {group_name}")
        print(f"{'─' * 72}")

        valid_tasks = []
        for tid in task_ids:
            try:
                task = mechval.load_task(tid)
                circuit = task.get_circuit()
                heads = {h for hs in circuit.roles.values() for h in hs}
                roles = list(circuit.roles.keys())
                valid_tasks.append((tid, len(heads), roles))
                print(f"\n  {tid}: {len(heads)} heads, roles={roles}")
            except Exception as e:
                print(f"\n  {tid}: [error: {e}]")

        if len(valid_tasks) < 2:
            print("  (need at least 2 circuits to compare)")
            continue

        print(f"\n  Pairwise comparisons:")
        for i in range(len(valid_tasks)):
            for j in range(i + 1, len(valid_tasks)):
                tid_a = valid_tasks[i][0]
                tid_b = valid_tasks[j][0]
                result = compare_circuits(tid_a, tid_b)
                print(f"\n    {tid_a} vs {tid_b}:")
                print(f"      Jaccard similarity: {result['jaccard']:.3f}")
                print(f"      Overlap: {result['overlap']} heads")
                print(f"      Only in {tid_a}: {result['only_a']} heads")
                print(f"      Only in {tid_b}: {result['only_b']} heads")
                if result['overlap_heads']:
                    print(f"      Shared heads: {result['overlap_heads']}")

    print("\n" + "=" * 72)
    print("CLAIM SPEC STRUCTURAL SUMMARY")
    print("=" * 72)

    task_ids = mechval.list_tasks()
    specs = []
    for tid in task_ids:
        task = mechval.load_task(tid)
        spec = task.get_claim_spec()
        if spec:
            specs.append((tid, spec))

    if not specs:
        print("  No claim specs found.")
        return

    header = f"  {'Task':<20} {'Steps':>5} {'Edges':>5} {'Pred':>5} {'Neg':>4} " \
             f"{'Comp Types':<25} {'Risk':<8}"
    print(f"\n{header}")
    print(f"  {'─' * 20} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 4} {'─' * 25} {'─' * 8}")

    for task_id, spec in specs:
        comp_types = set()
        for step in spec.steps:
            comp_types.update(step.component_types)
        ct_str = ", ".join(sorted(comp_types))
        risk = spec.superposition_risk.polysemanticity_risk
        print(f"  {task_id:<20} {len(spec.steps):>5} {len(spec.edges):>5} "
              f"{len(spec.predictions):>5} {len(spec.negative_controls):>4} "
              f"{ct_str:<25} {risk:<8}")

    print()


if __name__ == "__main__":
    main()
