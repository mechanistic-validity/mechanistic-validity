# Mechanistic Validity

> **Under active development.** Feedback welcome.

A framework for evaluating the validity of mechanistic interpretability claims about neural networks, drawing on philosophy of science, neuroscience, pharmacology, measurement theory, and MI itself.

## Why this exists

Mechanistic interpretability makes claims like "this circuit implements indirect object identification" or "this neuron represents the concept of dogs." But how do we know these claims are true? What standards of evidence should we apply?

Other sciences solved this decades ago. This project adapts their validity frameworks into a unified evaluation lens for circuit-level claims.

## The framework

### Five validity lenses

| Lens | Tradition | Core question |
|---|---|---|
| Construct | Philosophy of science | Is the claim falsifiable and well-defined? |
| Internal | Neuroscience | Is the causal evidence sound? |
| External | Pharmacology | Does it generalize beyond the test conditions? |
| Measurement | Measurement theory | Are the instruments reliable and calibrated? |
| Interpretive | MI | Is the description level declared and consistent? |

### Six evidence families

| Family | Code | Count | What it asks |
|---|---|---|---|
| Causal | A01–A13 | 13 | Does X causally produce Y? |
| Structural | B01–B09 | 9 | Do the weights encode the claimed computation? |
| Information-theoretic | C01–C09 | 9 | What information flows where? |
| Behavioral | D01–D09 | 9 | Does the circuit reproduce model behavior? |
| Representational | E01–E10 | 10 | What geometric structure do activations have? |
| Measurement-theoretic | F01–F08 | 8 | Are the instruments themselves reliable? |

### Verdict tiers

| Tier | Name | What it means |
|---|---|---|
| 1 | Proposed | Structural alignment only, no causal evidence |
| 2 | Causally suggestive | Necessity established (ablation degrades behavior) |
| 3 | Mechanistically supported | Necessity + sufficiency |
| 4 | Triangulated | Multiple independent instruments converge |
| 5 | Validated | All five lenses pass |

### Description modes

Seven levels of mechanistic description, from *computational* ("what is computed and why") down through *algorithmic* and four *implementational* sub-modes (topographic, connectomic, activation-statistical, functional).

### 58 evaluation instruments

Concrete measurement procedures drawn from causal inference, linear algebra, information theory, behavioral testing, representation geometry, and measurement-theoretic reliability analysis.

### 27 validity criteria

Specific pass/fail criteria across five validity types: construct (C1–C5), internal (I1–I5), external (E1–E6), measurement (M1–M6), and interpretive (V1–V5).

## 13 worked case studies

The full framework applied to published MI results:

| Case study | Verdict |
|---|---|
| IOI Circuit (Wang et al. 2022) | Triangulated |
| Induction Heads (Olsson et al. 2022) | Mechanistically supported |
| Greater-Than (Hanna et al. 2023) | Mechanistically supported |
| Grokking (Nanda et al. 2023) | Causally suggestive |
| Copy Suppression (McDougall et al. 2023) | Mechanistically supported |
| Successor Heads (Gould et al. 2023) | Causally suggestive |
| Docstring Circuit (Heimersheim & Janiak 2023) | Causally suggestive |
| Knowledge Neurons (Dai et al. 2022) | Proposed |
| Othello World Model (Li et al. 2023) | Triangulated |
| SAE Features (Cunningham et al. 2023) | Causally suggestive |
| Superposition (Elhage et al. 2022) | Proposed |
| Probing Classifiers (Belinkov 2022) | Proposed |
| Gender Bias Circuits (Vig et al. 2020) | Proposed |

## Quick start

```bash
cd docs && npm install && npm run dev
```

## Citation

If you use this framework in your research, see [`CITATION.cff`](CITATION.cff) or click the "Cite this repository" button on GitHub.

## License

[MIT](LICENSE)
