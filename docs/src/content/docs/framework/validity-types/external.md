---
title: "External Validity"
---

# External Validity

Does the mechanism generalize? Causal evidence on one prompt set, with one ablation method, in one model, establishes internal validity at best. External validity asks whether the mechanism generalizes across prompts, ablation methods, related tasks, and ideally across models. A result that survives rigorous causal testing under one set of conditions is a local result. External validity is what separates a local result from a finding about the model.

## Position in the Dependency Chain

External validity occupies the fourth position in the chain: **Construct → Measurement → Internal → External → Interpretive**.

We place external validity after internal validity because a finding that has not been established causally within its discovery conditions cannot be said to generalize. Internal validity asks whether the evidence supports the causal claim in the tested setting. External validity asks whether that causal claim extends to new settings — different prompts, different intervention methods, different tasks, different models. A claim that fails internal validity has nothing to generalize. A claim that passes internal validity but fails external validity is real but local: it holds under the conditions tested and we do not know whether it holds elsewhere.

Interpretive validity comes after external validity because the scope of a mechanistic narrative depends on knowing where the mechanism does and does not hold. A mechanism that generalizes across models licenses broader interpretive claims than one confined to a single architecture.

## Theoretical Lineage

The intellectual foundations of external validity come from pharmacology, causal inference, and philosophy of science.

**Pharmacology** provides the most developed framework for generalization testing. The Phase III clinical trial (Rang et al. 2006) exists to answer the external validity question: a drug that works under controlled conditions in Phase II must be shown to work across diverse patient populations, dosing regimens, and clinical sites before it counts as effective. We draw three specific transfers from this tradition. Dose-response curves (E5) formalize graded response — the requirement that partial intervention produce partial effects. Therapeutic windows formalize the distinction between an intervention strength that produces the target effect and one that produces general degradation. Cross-population generalization maps to cross-model recurrence (E4).

**Pearl (2011)** provides the formal language for cross-model transfer through transportability theory. Transportability asks under what conditions a causal effect estimated in one population (model) can be transported to another. The key insight is that recurrence of a behavioral pattern across models does not establish that the same causal organization produces it. Different causal structures can generate identical input–output relations. E4 requires a declared correspondence criterion — a stated basis for claiming that the causal organization, not just the behavior, recurs.

**Steel (2008)** articulates the limits of extrapolation from recurrence alone. Steel argues that recurrence supports only limited induction: stronger extrapolation requires evidence that the causally relevant process and its supporting conditions are preserved in the target. We use this to distinguish E4 (cross-model recurrence with a correspondence criterion) from the weaker claim that the same behavior appears in a second model.

**Hill (1965)** contributes two of his nine viewpoints for causal inference that bear directly on external validity. Graded response — the expectation that stronger exposure produces stronger effect — becomes E5. Specificity — the expectation that the cause is preferentially associated with the effect rather than with unrelated outcomes — informs E3 (cross-task generalization) and E1 (intervention reach).

## Criteria

| ID | Name | Question |
|---|---|---|
| E1 | Intervention reach | Has the result been reproduced under at least two intervention families, and do they agree? |
| E2 | Prompt generalization | Does the mechanism hold on diverse prompts beyond the discovery distribution? |
| E3 | Cross-task generalization | Does the mechanism transfer to related tasks? |
| E4 | Cross-model recurrence | Does the corresponding causal organization recur across independently trained models under a declared correspondence criterion? |
| E5 | Graded response | Does partial ablation produce partial effects? |
| E6 | Novel prediction | Does the mechanism predict new, untested behaviors? |

E1–E3 test the breadth of the finding across methods, inputs, and tasks. E4 tests whether the mechanism is a property of the computational problem rather than of a single trained instance. E5 tests the quantitative structure of the causal relationship. E6 tests whether the mechanism has predictive content beyond the observations it was built to explain.

**On E4:** Cross-model recurrence requires more than observing the same behavior in a second model. Different causal organizations can produce the same input–output relation, so E4 asks whether a *corresponding causal organization* recurs under a declared criterion of correspondence. A criterion might be Jaccard overlap of circuit components under a stated alignment, cosine similarity of weight-space signatures, or IIA of the same causal abstraction. The criterion must be stated before the comparison is attempted.

## Failure Examples

**IOI faithfulness across methods (E1).** Miller, Chughtai & Saunders showed that the faithfulness of the IOI circuit — the same circuit, the same model, the same prompts — spans below 0% to over 100% across six methodological choices (ablation type, metric, complement definition). A result that changes sign depending on methodological choices that the original paper did not vary has not been reproduced across intervention families. This is an E1 failure: the finding is conditional on a specific methodological configuration that was not identified as load-bearing at the time of publication.

**Shortcut learning in CNNs (E2/C4).** Geirhos et al. demonstrated that CNNs trained on ImageNet classify images by texture rather than shape, contrary to the assumed construct. Models that appeared to generalize on the training distribution failed on texture-shape conflict stimuli. The mechanism that drove accuracy on standard benchmarks did not transfer to inputs where texture and shape disagreed. This is both an external validity failure (the mechanism does not generalize to the broader task the benchmark was meant to sample) and a construct validity failure (the construct "object recognition" was not distinguished from "texture matching").

## Cross-Disciplinary Foundations

The table below shows which of the eight theoretical foundations contribute to external validity and what they contribute.

| Discipline | Transfer to external validity |
|---|---|
| Pharmacology | Dose-response curves (E5), therapeutic windows, cross-population generalization (E4), affinity vs efficacy distinction |
| Causal inference | Transportability theory for cross-model transfer (E4); formal conditions under which causal conclusions transport across settings |
| Philosophy of science | Severe testing — a result that has survived tests it could have failed across diverse conditions carries more evidential weight than one tested in a single configuration |
| Medical microbiology | Graded presence as evidence for causal involvement (E5); Koch's postulates require the agent to produce the disease in a new host, not just be found in the original |
| Genetics | Cross-species conservation as evidence for functional importance; the distinction between sequence conservation (recurrence) and functional conservation (preserved causal role) parallels E4 |
