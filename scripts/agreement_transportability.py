"""Agreement transportability analysis — do agreement circuits share components?

Compares the gendered pronoun (gender agreement) and SVA (number agreement)
circuits to test whether agreement mechanisms generalize across linguistic
features. This is a V2 transportability analysis.

Usage:
    python scripts/agreement_transportability.py
"""
import sys
sys.path.insert(0, "src")

import mechval


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    print("=" * 72)
    print("AGREEMENT TRANSPORTABILITY ANALYSIS")
    print("=" * 72)
    print("\nDo gender agreement and number agreement circuits share components?")
    print("If yes: agreement may be a unified mechanism in GPT-2.")
    print("If no: agreement is feature-specific (separate gender vs number circuits).\n")

    agreement_tasks = {
        "gendered_pronoun": "Gender agreement (Mathwin 2023)",
        "sva": "Subject-verb number agreement (Lazo et al. 2025)",
    }

    circuits = {}
    for tid, desc in agreement_tasks.items():
        task = mechval.load_task(tid)
        circuit = task.get_circuit()
        heads = {h for hs in circuit.roles.values() for h in hs}
        circuits[tid] = {
            "task": task,
            "circuit": circuit,
            "heads": heads,
            "roles": circuit.roles,
            "desc": desc,
        }
        print(f"  {tid}: {desc}")
        print(f"    Heads ({len(heads)}): {sorted(heads)}")
        print(f"    Roles: {list(circuit.roles.keys())}")
        for role, role_heads in circuit.roles.items():
            layers = sorted(set(h[0] for h in role_heads))
            print(f"      {role}: {role_heads} (layers {layers})")
        print()

    print("-" * 72)
    print("HEAD OVERLAP")
    print("-" * 72)

    gp_heads = circuits["gendered_pronoun"]["heads"]
    sva_heads = circuits["sva"]["heads"]

    overlap = gp_heads & sva_heads
    only_gp = gp_heads - sva_heads
    only_sva = sva_heads - gp_heads
    j = jaccard(gp_heads, sva_heads)

    print(f"\n  Jaccard similarity: {j:.3f}")
    print(f"  Shared heads ({len(overlap)}): {sorted(overlap)}")
    print(f"  Only in gendered_pronoun ({len(only_gp)}): {sorted(only_gp)}")
    print(f"  Only in SVA ({len(only_sva)}): {sorted(only_sva)}")

    if overlap:
        print(f"\n  Shared head role assignments:")
        for head in sorted(overlap):
            gp_role = None
            sva_role = None
            for role, heads in circuits["gendered_pronoun"]["roles"].items():
                if head in heads:
                    gp_role = role
            for role, heads in circuits["sva"]["roles"].items():
                if head in heads:
                    sva_role = role
            same = "SAME" if gp_role == sva_role else "DIFFERENT"
            print(f"    {head}: gendered_pronoun={gp_role}, sva={sva_role} [{same}]")

    print("\n" + "-" * 72)
    print("LAYER DISTRIBUTION")
    print("-" * 72)

    for tid in ["gendered_pronoun", "sva"]:
        layers = sorted(set(h[0] for h in circuits[tid]["heads"]))
        print(f"\n  {tid}: layers {layers} (span {min(layers)}-{max(layers)})")

    print("\n" + "-" * 72)
    print("STRUCTURAL COMPARISON")
    print("-" * 72)

    gp_roles = circuits["gendered_pronoun"]["roles"]
    sva_roles = circuits["sva"]["roles"]

    print(f"\n  Gendered pronoun: {len(gp_roles)} roles, "
          f"{sum(len(v) for v in gp_roles.values())} heads")
    print(f"  SVA: {len(sva_roles)} roles, "
          f"{sum(len(v) for v in sva_roles.values())} heads")

    print("\n  Role-by-role layer comparison:")
    print(f"  {'GP Role':<20} {'Layers':<15} {'SVA Role':<20} {'Layers':<15}")
    print(f"  {'─' * 20} {'─' * 15} {'─' * 20} {'─' * 15}")

    gp_sorted = sorted(gp_roles.items(), key=lambda x: min(h[0] for h in x[1]))
    sva_sorted = sorted(sva_roles.items(), key=lambda x: min(h[0] for h in x[1]))

    for i in range(max(len(gp_sorted), len(sva_sorted))):
        gp_name = gp_sorted[i][0] if i < len(gp_sorted) else ""
        gp_layers = str(sorted(set(h[0] for h in gp_sorted[i][1]))) if i < len(gp_sorted) else ""
        sva_name = sva_sorted[i][0] if i < len(sva_sorted) else ""
        sva_layers = str(sorted(set(h[0] for h in sva_sorted[i][1]))) if i < len(sva_sorted) else ""
        print(f"  {gp_name:<20} {gp_layers:<15} {sva_name:<20} {sva_layers:<15}")

    print("\n" + "-" * 72)
    print("ALSO COMPARING: Induction-family circuits")
    print("-" * 72)
    print("\n  Circuits that share PTH+IND structure:\n")

    pth_ind_tasks = ["ioi", "induction", "copy_suppression"]
    for tid in pth_ind_tasks:
        task = mechval.load_task(tid)
        circuit = task.get_circuit()
        pth = circuit.roles.get("PTH", [])
        ind = circuit.roles.get("IND", [])
        print(f"  {tid}:")
        print(f"    PTH: {pth}")
        print(f"    IND: {ind}")

    all_pth = set()
    all_ind = set()
    for tid in pth_ind_tasks:
        circuit = mechval.load_task(tid).get_circuit()
        all_pth.update(tuple(h) for h in circuit.roles.get("PTH", []))
        all_ind.update(tuple(h) for h in circuit.roles.get("IND", []))

    print(f"\n  Universal PTH heads (in all 3): ", end="")
    universal_pth = set.intersection(
        *[set(tuple(h) for h in mechval.load_task(t).get_circuit().roles.get("PTH", []))
          for t in pth_ind_tasks]
    )
    print(sorted(universal_pth) if universal_pth else "none")

    print(f"  Universal IND heads (in all 3): ", end="")
    universal_ind = set.intersection(
        *[set(tuple(h) for h in mechval.load_task(t).get_circuit().roles.get("IND", []))
          for t in pth_ind_tasks]
    )
    print(sorted(universal_ind) if universal_ind else "none")

    print("\n" + "-" * 72)
    print("INTERPRETATION")
    print("-" * 72)

    if j > 0.3:
        print(f"\n  HIGH overlap (Jaccard={j:.3f}): Agreement mechanisms may share")
        print(f"  components across gender and number features. This supports a")
        print(f"  unified agreement mechanism hypothesis.")
    elif j > 0.1:
        print(f"\n  MODERATE overlap (Jaccard={j:.3f}): Some shared infrastructure")
        print(f"  but substantial feature-specific components. Partial unification.")
    else:
        print(f"\n  LOW overlap (Jaccard={j:.3f}): Gender and number agreement use")
        print(f"  largely separate circuits. Agreement is feature-specific in GPT-2,")
        print(f"  not a unified mechanism.")
    print()


if __name__ == "__main__":
    main()
