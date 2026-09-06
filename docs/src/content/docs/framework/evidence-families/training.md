---
title: "Training"
description: "Evidence from the model's developmental history — how the network came to be as it is."
---

# Training

Training-based evidence comes from the model's developmental history: how its parameters changed during optimization, when capabilities appeared, and what happens when the training process is modified. This is the least-used evidence family in current mechanistic interpretability, yet it addresses questions that no post-training measurement can answer — particularly about whether a mechanism is a stable product of the learning process or a fragile artifact of a specific training run.

## Observational

Methods that analyze training trajectories without modifying them:

- **Loss curves** — training and validation loss over time, including structure in the curve (plateaus, phase transitions, sudden drops).
- **Checkpoint comparison** — comparing weight structure, activation patterns, or behavioral profiles across training checkpoints to track when and how a mechanism forms.
- **Phase transitions** — identifying points in training where qualitative changes in behavior or internal structure occur (e.g., grokking, induction head formation).
- **Learning dynamics** — tracking gradient norms, parameter velocities, effective learning rates, and other optimization-level quantities across training.

### Onset coupling

Observational training evidence is the primary source for [I11 Onset coupling](/mechanistic-validity/framework/criteria/internal/onset-coupling/): does the mechanism appear when the capability appears? If a circuit is claimed to implement a capability, and that capability emerges at step 10,000, the circuit's weight structure and activation patterns should undergo a corresponding transition at the same point. Onset coupling that holds across multiple training seeds strengthens the claim; onset coupling that fails — the capability appears but the proposed circuit does not — is evidence against the mechanistic claim.

## Interventional

Methods that modify the training process and observe consequences:

- **Ablate-then-retrain** — removing a component (by zeroing or masking weights) and continuing training. Tests whether the model recovers the capability through alternative pathways, which bears on [I5 Rival mechanism exclusion](/mechanistic-validity/framework/criteria/internal/rival-mechanism-exclusion/).
- **Fine-tuning** — targeted training on specific data and measuring which components change. Components that change during task-specific fine-tuning are implicated in that task.
- **Curriculum manipulation** — modifying the training data distribution and observing effects on mechanism formation. Tests whether the mechanism depends on specific data patterns or emerges from general optimization pressure.
- **Knockout training** — training with specific components held fixed (frozen) and measuring whether the capability still develops.

### Offset coupling

Interventional training evidence addresses [I12 Offset coupling](/mechanistic-validity/framework/criteria/internal/offset-coupling/): does the mechanism go when the capability is removed? If fine-tuning on data that removes a capability also degrades the proposed circuit, this is evidence for a causal link between the circuit and the capability. The neuroscience analogue is Craver's "top-down leg" of mutual manipulability: manipulating the system-level phenomenon (the capability) affects the component-level mechanism (the circuit).

## Relevant criteria

Training evidence is most directly relevant to:

| Criterion | How training evidence bears on it |
|---|---|
| [I11 Onset coupling](/mechanistic-validity/framework/criteria/internal/onset-coupling/) | Does the mechanism form when the capability develops? |
| [I12 Offset coupling](/mechanistic-validity/framework/criteria/internal/offset-coupling/) | Does the mechanism degrade when the capability is removed? |
| [I5 Rival mechanism exclusion](/mechanistic-validity/framework/criteria/internal/rival-mechanism-exclusion/) | Does ablate-then-retrain recover via alternative pathways? |
| [M3 Stability](/mechanistic-validity/framework/criteria/measurement/stability/) | Is the mechanism stable across training seeds? |
| [I10 Rescue reversibility](/mechanistic-validity/framework/criteria/internal/rescue-reversibility/) | Does restoring a component after ablation-then-retrain recover the original mechanism? |
