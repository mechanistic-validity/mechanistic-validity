"""Proxy circuit analysis — which tasks inherit circuits and how valid is that?

Identifies all proxy circuits, computes how they relate to their parent
circuits, and proposes a validation protocol for each family.

Usage:
    python scripts/proxy_circuit_analysis.py
"""
import sys
sys.path.insert(0, "src")

import mechval
from collections import defaultdict


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    all_task_ids = mechval.list_tasks()

    tasks_by_status = defaultdict(list)
    for tid in all_task_ids:
        task = mechval.load_task(tid)
        tasks_by_status[task.circuit_status].append(tid)

    print("=" * 72)
    print("PROXY CIRCUIT ANALYSIS")
    print("=" * 72)
    print(f"\nTask counts by circuit status:")
    for status, tids in sorted(tasks_by_status.items()):
        print(f"  {status}: {len(tids)}")

    print("\n" + "=" * 72)
    print("1. PROXY CIRCUIT FAMILIES")
    print("=" * 72)

    head_sets = {}
    for tid in all_task_ids:
        task = mechval.load_task(tid)
        try:
            circuit = task.get_circuit()
        except (NotImplementedError, AttributeError):
            continue
        if circuit:
            heads = frozenset(h for hs in circuit.roles.values() for h in hs)
            head_sets[tid] = heads

    families = defaultdict(list)
    for tid, heads in head_sets.items():
        families[heads].append(tid)

    multi_families = {k: v for k, v in families.items() if len(v) > 1}
    print(f"\n{len(multi_families)} circuit families (tasks sharing identical heads):\n")

    for heads, tids in multi_families.items():
        statuses = {}
        for tid in tids:
            task = mechval.load_task(tid)
            statuses[tid] = task.circuit_status
        parent = next((t for t, s in statuses.items() if s == "full_circuit"), tids[0])
        proxies = [t for t in tids if t != parent]

        print(f"  Family: {parent} ({len(list(heads))} heads)")
        print(f"    Parent: {parent} ({statuses[parent]})")
        for p in proxies:
            print(f"    Proxy:  {p} ({statuses[p]})")

        task = mechval.load_task(parent)
        circuit = task.get_circuit()
        print(f"    Roles: {list(circuit.roles.keys())}")
        spec = task.get_claim_spec()
        print(f"    Has claim spec: {spec is not None}")
        print()

    print("=" * 72)
    print("2. THE PROXY ASSUMPTION")
    print("=" * 72)

    print("""
  When a task has circuit_status="proxy_circuit", it means:

  ASSUMPTION: "The same set of heads that implements task X also
  implements task Y, because Y is a behavioral variant of X."

  This is testable. The validation protocol is:

  1. STRUCTURAL TEST (no GPU needed):
     - Do the proxy tasks use the same prompt structure?
     - Do they target the same linguistic phenomenon?
     - Are the expected outputs the same type (next-token, logit diff)?

  2. BEHAVIORAL TEST (needs GPU):
     - Run activation patching per task variant
     - Compare which heads have significant causal effect
     - Compute Jaccard similarity between parent and proxy circuits

  3. VERDICT:
     - Jaccard > 0.7: circuit transfers (promote to full_circuit)
     - Jaccard 0.3-0.7: partial transfer (needs its own circuit)
     - Jaccard < 0.3: circuit doesn't transfer (needs independent discovery)
""")

    print("=" * 72)
    print("3. PER-FAMILY STRUCTURAL ASSESSMENT")
    print("=" * 72)

    print("""
  RTI FAMILY (5 tasks, same 15-head circuit)
  ──────────────────────────────────────────
  Parent: rti (repeated token identification)
  Proxies: rti_pattern, token_flood, buffalo, mib_rti

  Structural similarity:
    - rti: "A B C D A" → predict second A
    - rti_pattern: "X Y X Y X" → predict repeated pattern
    - token_flood: "the the the the" → predict flood continuation
    - buffalo: "buffalo buffalo buffalo" → predict recursive repetition
    - mib_rti: same as rti, MIB-formatted prompts

  Assessment: HIGH structural similarity. All test "has this token
  appeared before?" The backbone→detector→copier→readout pipeline
  is plausibly the same mechanism. But token_flood (many repetitions)
  may weight the copier heads differently than rti (single repetition).

  Confidence: MEDIUM — likely transfers but head importance may differ.
  Priority for GPU testing: MEDIUM.
""")

    print("""  IOI FAMILY (4 tasks, same 15-head circuit)
  ──────────────────────────────────────────
  Parent: ioi (indirect object identification)
  Proxies: centering_theory, resumptive, self_allo

  Structural similarity:
    - ioi: "When Mary and John went to the store, John gave a drink to"
    - centering_theory: center-embedded clauses testing coreference
    - resumptive: resumptive pronouns in relative clauses
    - self_allo: self-referential allocentric constructions

  Assessment: MODERATE structural similarity. All test coreference/
  name binding, but the syntactic structures differ significantly.
  S-inhibition may not be needed for centering_theory if there's no
  duplicate name to suppress.

  Confidence: LOW-MEDIUM — the DTH→S-Inh→NM pipeline is specific to
  the "suppress repeated name" pattern. Centering theory and resumptive
  pronouns have different syntactic triggers.
  Priority for GPU testing: HIGH.
""")

    print("""  INDUCTION FAMILY (4 tasks, same 7-head circuit)
  ────────────────────────────────────────────────
  Parent: induction (sequence completion)
  Proxies: sequence_internal, alternating_pair, novel_song

  Structural similarity:
    - induction: "A B ... A B" → predict B after second A
    - sequence_internal: internal sequence completion
    - alternating_pair: "A B A B" → predict alternating pattern
    - novel_song: novel sequence memorization

  Assessment: HIGH structural similarity. All test "complete a repeated
  sequence." The PTH→IND pipeline (previous token → induction) is the
  core mechanism for all. Alternating pairs are a direct specialization.
  Novel songs test whether the circuit generalizes to unseen sequences.

  Confidence: HIGH — induction is a well-characterized, general mechanism.
  Priority for GPU testing: LOW (most likely to transfer).
""")

    print("=" * 72)
    print("4. HOW THIS CONNECTS TO THE FRAMEWORK")
    print("=" * 72)

    print("""
  Track 1 (Circuit Localization):
    Proxy circuits skip Track 1 — they inherit a circuit without running
    circuit discovery. The implicit claim is "discovery would find the
    same circuit." This is testable.

  Track 3 (Causal Model Testing):
    Currently only parent tasks have claim specs. Proxy tasks could get
    their own specs OR the parent spec could be run on proxy prompts
    to test transportability.

  View V2 (Causal Transportability):
    This is THE view for proxy circuit validation. V2 metrics directly
    test whether a circuit's causal structure transfers across:
    - Different prompt types (RTI family)
    - Different syntactic constructions (IOI family)
    - Different sequence types (induction family)

  Gate G3 (Confound/Superposition Risk):
    Proxy circuits inherit the parent's superposition risk assessment.
    But different prompts may activate different polysemantic heads,
    changing the confound profile.

  View V4 (Mechanism Adjudication):
    If a proxy task gets its own circuit via Track 1, V4 can compare
    the parent circuit vs the independently-discovered circuit.
""")

    print("=" * 72)
    print("5. CONCRETE NEXT STEPS")
    print("=" * 72)

    print("""
  CPU (no GPU needed):
    ✓ Structural assessment of each family (done above)
    ✓ Cross-circuit head overlap analysis (done in cross_circuit_transportability.py)
    □ Add V2 transportability metrics to the metric registry
    □ Write proxy validation claim specs (test parent circuit on proxy prompts)

  GPU (needs RunPod):
    □ Run activation patching per RTI variant (5 tasks × ~1h each)
    □ Run activation patching per IOI variant (4 tasks × ~1h each)
    □ Compare: does the same set of heads matter for each variant?
    □ Run Track 3 verify() on parent specs using proxy prompts
    □ Compute V2 transportability scores for each family

  Analysis:
    □ For each family, produce a transportability verdict:
      - "Circuit transfers" → promote proxy to full_circuit
      - "Partial transfer" → discover variant-specific circuit
      - "No transfer" → proxy_circuit status is misleading, needs own circuit
""")


if __name__ == "__main__":
    main()
