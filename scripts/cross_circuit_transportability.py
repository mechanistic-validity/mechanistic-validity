"""Cross-circuit transportability analysis — which heads are shared across circuits?

Computes pairwise Jaccard similarity and identifies universal, shared, and
unique heads across all circuits with full definitions. Highlights the
"hub heads" that appear in many circuits.

Usage:
    python scripts/cross_circuit_transportability.py
"""
import sys
sys.path.insert(0, "src")

import mechval
from collections import Counter


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    all_task_ids = mechval.list_tasks()

    circuits = {}
    for tid in all_task_ids:
        task = mechval.load_task(tid)
        if task.circuit_status in ("generator_only", "planned"):
            continue
        circuit = task.get_circuit()
        if not circuit:
            continue
        heads = frozenset(h for hs in circuit.roles.values() for h in hs)
        if not heads:
            continue
        circuits[tid] = {
            "heads": heads,
            "roles": circuit.roles,
            "n_heads": len(heads),
        }

    deduped = {}
    seen_head_sets = {}
    for tid, info in circuits.items():
        key = info["heads"]
        if key in seen_head_sets:
            seen_head_sets[key].append(tid)
        else:
            seen_head_sets[key] = [tid]
            deduped[tid] = info

    print("=" * 72)
    print("CROSS-CIRCUIT TRANSPORTABILITY ANALYSIS")
    print("=" * 72)

    print(f"\n{len(circuits)} tasks with circuits, {len(deduped)} unique circuits\n")

    print("Circuit families (tasks sharing identical head sets):")
    for head_set, tids in seen_head_sets.items():
        if len(tids) > 1:
            print(f"  {tids[0]} family ({len(tids)} tasks): {', '.join(tids)}")
    print()

    names = sorted(deduped.keys())

    print("-" * 72)
    print("PAIRWISE JACCARD SIMILARITY")
    print("-" * 72)
    print()

    header = f"{'':>22}"
    for name in names:
        short = name[:10]
        header += f" {short:>10}"
    print(header)

    for i, n1 in enumerate(names):
        row = f"{n1:>22}"
        for j, n2 in enumerate(names):
            j_val = jaccard(set(deduped[n1]["heads"]), set(deduped[n2]["heads"]))
            if i == j:
                row += f" {'---':>10}"
            elif j_val == 0:
                row += f" {'·':>10}"
            else:
                row += f" {j_val:>10.3f}"
        print(row)
    print()

    print("-" * 72)
    print("HUB HEADS — heads appearing in multiple circuits")
    print("-" * 72)

    head_counter = Counter()
    head_circuits = {}
    for tid, info in deduped.items():
        for h in info["heads"]:
            head_counter[h] += 1
            head_circuits.setdefault(h, []).append(tid)

    multi = [(h, count) for h, count in head_counter.items() if count > 1]
    multi.sort(key=lambda x: (-x[1], x[0]))

    if multi:
        print()
        for head, count in multi:
            circuits_str = ", ".join(head_circuits[head])
            roles_str = []
            for tid in head_circuits[head]:
                role = next(
                    (r for r, hs in deduped[tid]["roles"].items() if head in hs),
                    "?",
                )
                roles_str.append(f"{tid}={role}")
            print(f"  Head {head}: in {count} circuits")
            print(f"    Roles: {', '.join(roles_str)}")
    else:
        print("\n  No heads shared across circuits.")

    print()
    print("-" * 72)
    print("CIRCUIT SIZE COMPARISON")
    print("-" * 72)
    print()

    for tid in sorted(deduped.keys(), key=lambda t: deduped[t]["n_heads"]):
        info = deduped[tid]
        layers = sorted(set(h[0] for h in info["heads"]))
        n_roles = len(info["roles"])
        family_count = len(seen_head_sets[info["heads"]])
        family_str = f" (+{family_count - 1} variants)" if family_count > 1 else ""
        print(
            f"  {tid:>22}: {info['n_heads']:>2} heads, "
            f"{n_roles} roles, layers {min(layers)}-{max(layers)}{family_str}"
        )

    print()
    print("-" * 72)
    print("INDUCTION-FAMILY UNIVERSAL COMPONENTS")
    print("-" * 72)

    pth_ind_tasks = ["ioi", "induction", "copy_suppression"]
    available = [t for t in pth_ind_tasks if t in deduped]

    if len(available) >= 2:
        print()
        pth_sets = []
        ind_sets = []
        for tid in available:
            pth = set(tuple(h) for h in deduped[tid]["roles"].get("PTH", []))
            ind = set(tuple(h) for h in deduped[tid]["roles"].get("IND", []))
            pth_sets.append(pth)
            ind_sets.append(ind)
            print(f"  {tid}: PTH={sorted(pth)}, IND={sorted(ind)}")

        universal_pth = set.intersection(*pth_sets) if pth_sets else set()
        universal_ind = set.intersection(*ind_sets) if ind_sets else set()
        print(f"\n  Universal PTH (all {len(available)}): {sorted(universal_pth)}")
        print(f"  Universal IND (all {len(available)}): {sorted(universal_ind)}")

    print()
    print("-" * 72)
    print("RTI FAMILY — same circuit, different prompts")
    print("-" * 72)

    rti_family = seen_head_sets.get(deduped.get("rti", {}).get("heads", frozenset()), [])
    if rti_family:
        print(f"\n  {len(rti_family)} tasks share the same 15-head circuit: {', '.join(rti_family)}")
        print(f"  All marked proxy_circuit except 'rti' (full_circuit)")
        print(f"  Question: do behavioral variants actually recruit the same heads?")
        print(f"  Status: UNTESTED — needs per-variant activation patching")
        print()

        rti_heads = set(deduped["rti"]["heads"])
        print(f"  Cross-task overlap with RTI circuit:")
        for tid in sorted(deduped.keys()):
            if tid in rti_family or tid == "rti":
                continue
            other_heads = set(deduped[tid]["heads"])
            overlap = rti_heads & other_heads
            if overlap:
                j = jaccard(rti_heads, other_heads)
                overlap_roles = []
                for h in sorted(overlap):
                    rti_role = next((r for r, hs in deduped["rti"]["roles"].items() if h in hs), "?")
                    other_role = next((r for r, hs in deduped[tid]["roles"].items() if h in hs), "?")
                    overlap_roles.append(f"{h}: rti={rti_role}, {tid}={other_role}")
                print(f"    {tid} (J={j:.3f}): {', '.join(overlap_roles)}")
    print()

    print("-" * 72)
    print("EPISTEMIC RIVALRY — 4 specs for same task")
    print("-" * 72)

    ep_tasks = ["epistemic_framing", "epistemic_tight", "epistemic_eap", "epistemic_expanded"]
    ep_available = [t for t in ep_tasks if t in deduped]

    if len(ep_available) >= 2:
        print()
        for i, t1 in enumerate(ep_available):
            for t2 in ep_available[i + 1:]:
                j = jaccard(set(deduped[t1]["heads"]), set(deduped[t2]["heads"]))
                overlap = set(deduped[t1]["heads"]) & set(deduped[t2]["heads"])
                print(f"  {t1} vs {t2}: J={j:.3f}, {len(overlap)} shared heads")

        all_ep_heads = [set(deduped[t]["heads"]) for t in ep_available]
        universal = set.intersection(*all_ep_heads) if all_ep_heads else set()
        any_heads = set.union(*all_ep_heads) if all_ep_heads else set()
        print(f"\n  Universal (in all {len(ep_available)}): {sorted(universal) if universal else 'none'}")
        print(f"  Total unique heads across all variants: {len(any_heads)}")
        print(f"  Core heads (present in all circuits) would be the strongest candidates")

    print()


if __name__ == "__main__":
    main()
