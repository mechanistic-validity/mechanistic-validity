"""Attention Schema Extraction (Attention-Over-Attention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric:         SM-05 — Attention Schema Extraction
Categories:     wildcard, self_model
Evidence family: structural
Description mode: implementational-functional

Tests whether the model contains an "attention schema" — heads that
attend to what other heads are doing, rather than to input tokens.

Background:
    Graziano's Attention Schema Theory (2011, "Human Consciousness and
    its Relationship to Social Neuroscience", Cognitive Neuroscience
    2:98-113; 2013, "Consciousness and the Attention Schema", Cognitive
    Neuropsychology 30:325-333) proposes that conscious systems maintain
    an internal model of their own attention — a simplified schematic
    of where attention is deployed.

    In transformer terms: are there attention heads that specifically
    compose with other heads' outputs (via K-composition or V-composition)
    rather than attending to input token embeddings? These would be
    "meta-attention" heads — monitoring and integrating the attention
    patterns of other heads.

    This is a purely weight-space analysis. It examines the structure
    of Q/K/V/O weight matrices to identify heads whose "receptive
    field" is other heads' outputs rather than the input embedding.

    Connections:
    - Graziano & Kastner (2011) — attention schema theory foundations
    - Graziano (2013) — consciousness and attention schema
    - Webb & Graziano (2015) "The Attention Schema Theory: A
      Mechanistic Account of Subjective Awareness", Frontiers in
      Psychology 6:500
    - Elhage et al. (2021) "A Mathematical Framework for Transformer
      Circuits", Anthropic — K-composition and V-composition

    The metric uses K-composition score from Elhage et al.:
    For heads A (layer l_A) and B (layer l_B > l_A):
        K-composition(A→B) measures how much head B's key circuit
        uses head A's output as input, rather than the token
        embedding directly.

Method:
    1. For each pair of heads (A at layer l_A, B at layer l_B)
       where l_B > l_A:
       - Compute K-composition score: ||W_K[B] @ W_O[A]||_F /
         (||W_K[B]||_F * ||W_O[A]||_F)
       - High score = B's keys strongly attend to A's outputs
    2. Build directed graph: edge A→B if K-composition(A→B) exceeds
       threshold
    3. Compute in-degree for each head: heads with high in-degree are
       integrating across many other heads' outputs
    4. Identify candidate "attention schema" heads: those with
       in-degree > mean + 1.5 * std
    5. Report schema candidates, their in-degree distribution, and
       the composition graph structure

    This runs entirely from weight matrices — no forward pass needed.

Pass condition: at least one head with in-degree > 2 * mean_in_degree.

Usage:
    mechval.run("attention_schema", tasks=["ioi"], device="cpu")
"""

import numpy as np
import torch

from mechval.metrics.common import (
    CIRCUIT_TASKS,
    EvalResult,
    InstrumentInfo,
    get_circuit_heads,
    load_model,
    log,
    parse_common_args,
    save_incremental,
    save_results,
)

INSTRUMENT_INFO = InstrumentInfo(
    name="Attention Schema Extraction",
    paper_ref="Graziano & Kastner 2011; Elhage et al. 2021",
    paper_cite="Graziano 2013 (attention schema theory); Elhage et al. 2021 (composition scores)",
    description="Identifies meta-attention heads that compose with other heads' outputs (attention-over-attention)",
    category="wildcard",
    tier="cogsci",
    origin="established",
    subcategory="meta_cognitive",
)

COMPOSITION_THRESHOLD = 0.1


@torch.no_grad()
def run_attention_schema(model, tasks: list[str],
                         n_prompts: int = 40) -> list[EvalResult]:
    results = []

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model

    for task in tasks:
        circuit_heads = get_circuit_heads(task)
        if not circuit_heads:
            log(f"  {task}: no circuit heads, skipping")
            continue

        heads = sorted(circuit_heads)
        n_circuit = len(heads)
        log(f"  {task}: {n_circuit} circuit heads, computing K-composition graph")

        composition_scores = {}
        edges = []

        for i, (la, ha) in enumerate(heads):
            W_O_a = model.W_O[la, ha].detach().float()

            for j, (lb, hb) in enumerate(heads):
                if lb <= la:
                    continue

                W_K_b = model.W_K[lb, hb].detach().float()

                composition = W_K_b @ W_O_a
                comp_norm = torch.norm(composition, p='fro').item()
                wk_norm = torch.norm(W_K_b, p='fro').item()
                wo_norm = torch.norm(W_O_a, p='fro').item()

                if wk_norm > 0 and wo_norm > 0:
                    k_comp = comp_norm / (wk_norm * wo_norm)
                else:
                    k_comp = 0.0

                key_a = f"L{la}H{ha}"
                key_b = f"L{lb}H{hb}"
                composition_scores[f"{key_a}->{key_b}"] = float(k_comp)

                if k_comp > COMPOSITION_THRESHOLD:
                    edges.append({
                        "source": key_a,
                        "target": key_b,
                        "k_composition": float(k_comp),
                    })

        in_degree: dict[str, int] = {f"L{l}H{h}": 0 for l, h in heads}
        out_degree: dict[str, int] = {f"L{l}H{h}": 0 for l, h in heads}
        weighted_in: dict[str, float] = {f"L{l}H{h}": 0.0 for l, h in heads}

        for edge in edges:
            in_degree[edge["target"]] += 1
            out_degree[edge["source"]] += 1
            weighted_in[edge["target"]] += edge["k_composition"]

        in_values = np.array(list(in_degree.values()), dtype=float)
        mean_in = float(in_values.mean())
        std_in = float(in_values.std()) if len(in_values) > 1 else 0.0

        schema_threshold = mean_in + 1.5 * std_in if std_in > 0 else mean_in * 2
        schema_candidates = [
            k for k, v in in_degree.items()
            if v > schema_threshold and v > 0
        ]

        has_schema = any(v > 2 * mean_in for v in in_degree.values() if mean_in > 0)
        passed = has_schema

        head_stats = {}
        for key in in_degree:
            head_stats[key] = {
                "in_degree": in_degree[key],
                "out_degree": out_degree[key],
                "weighted_in_composition": float(weighted_in[key]),
                "is_schema_candidate": key in schema_candidates,
            }

        ranking = sorted(head_stats.items(),
                        key=lambda kv: kv[1]["in_degree"], reverse=True)

        log(f"    {len(edges)} composition edges (threshold={COMPOSITION_THRESHOLD})")
        log(f"    mean_in_degree={mean_in:.2f}  std={std_in:.2f}")
        log(f"    Schema candidates: {schema_candidates if schema_candidates else 'none'}")
        for k, v in ranking[:5]:
            log(f"      {k}: in={v['in_degree']}  out={v['out_degree']}  "
                f"weighted={v['weighted_in_composition']:.4f}"
                f"{'  [SCHEMA]' if v['is_schema_candidate'] else ''}")
        log(f"    [{'PASS (schema detected)' if passed else 'FAIL (no schema)'}]")

        results.append(EvalResult(
            metric_id="SM05.attention_schema",
            value=float(max(in_degree.values())) if in_degree else 0.0,
            n_samples=0,
            instrument_info=INSTRUMENT_INFO,
            metadata={
                "task": task,
                "n_circuit_heads": n_circuit,
                "n_composition_edges": len(edges),
                "composition_threshold": COMPOSITION_THRESHOLD,
                "mean_in_degree": mean_in,
                "std_in_degree": std_in,
                "schema_candidates": schema_candidates,
                "head_stats": head_stats,
                "top_edges": sorted(edges, key=lambda e: e["k_composition"], reverse=True)[:20],
                "passed": passed,
            },
        ))

    return results


def main():
    parser = parse_common_args("SM-05: Attention Schema Extraction")
    args = parser.parse_args()

    tasks = args.tasks or CIRCUIT_TASKS
    model = load_model(args.model, args.device)

    log("=" * 60)
    log("SM-05: ATTENTION SCHEMA EXTRACTION")
    log("=" * 60)

    out = args.out or "SM05_attention_schema.json"
    jsonl_out = out.replace(".json", ".jsonl")
    results = []

    for task in tasks:
        task_results = run_attention_schema(model, [task], args.n_prompts)
        results.extend(task_results)
        for r in task_results:
            save_incremental(r, jsonl_out)

    save_results(results, out, args=args)
    log(f"\nDone. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
