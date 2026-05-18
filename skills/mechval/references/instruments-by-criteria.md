# Instruments by Criteria

Which instrument scripts to run for each criterion.
All paths relative to /Users/elliottower/Documents/GitHub/mechanistic-validity/src/instruments/

## I1 Necessity
- causal/scm_pearl/02_activation_patching.py
- causal/woodward/03_sigma_ablation.py (multi-method robustness)
- causal/woodward/35_resample_complement.py (resample variant)
- causal/mediation/05_mediation_v2.py (NDE/NIE decomposition)
- causal/mediation/24_pse.py (path-specific effect)
- causal/regularity_inus/39_inus_conditions.py (INUS conditions)
- causal/actual_cause/40_actual_causation.py (Halpern-Pearl)
- behavioral/ce_delta/43_ce_delta.py (CE loss change)
- behavioral/per_token_nll/44_per_token_nll.py (position-specific)

## I2 Sufficiency
- causal/counterfactual_das/01_das_iia.py (DAS-IIA primary)
- causal/counterfactual_das/20_corrupt_restore.py (reverse patching)
- causal/counterfactual_das/31_multi_axis_iia.py (multi-variable)
- causal/scm_pearl/04_causal_scrubbing.py
- behavioral/logit_diff_recovery/21_output_variants.py (5 metrics)
- behavioral/subnetwork_probe/48_subnetwork_probe.py
- representational/linear_probe/66_linear_probe.py

## I3 Specificity
- causal/rubin_cate/06_cate.py (heterogeneous effects)
- causal/rubin_cate/25_intervention_specificity.py
- causal/pid/08_pid.py (information decomposition)
- information/mutual_info/54_mutual_information.py
- information/conditional_mi/55_conditional_mi.py
- information/synergistic_info/58_o_information.py

## I4 Consistency
- causal/granger_te/32_cross_task_iia_transfer.py
- causal/causal_discovery/09_notears.py (DAG recovery)
- causal/causal_discovery/42_pc_algorithm.py
- information/transfer_entropy/53_transfer_entropy.py
- information/granger/56_granger_causality.py
- representational/rsa/61_rsa.py

## I5 Confound Control
- causal/woodward/37_misalignment_score.py (noising vs denoising asymmetry)
- causal/rubin_cate/25_intervention_specificity.py

## C2 Structural Plausibility
- causal/mdc_glennan/18_weight_extended.py (weight metrics)
- causal/mdc_glennan/19_logic_gates.py (gate detection)
- structural/spectral_svd/18_weight_extended.py
- structural/effective_rank/18_weight_extended.py
- structural/ov_qk_analysis/49_ov_qk_composition.py
- structural/norm_trajectory/51_norm_trajectory.py
- structural/ica_nmf/50_weight_decomposition.py
- representational/pca_dimensionality/60_pca_dimensionality.py
- representational/intrinsic_dimension/62_intrinsic_dimension.py
- representational/participation_ratio/65_participation_ratio.py
- representational/persistent_homology/67_persistent_homology.py
- representational/geodesic_distance/68_geodesic_distance.py

## C3 Task Specificity
- structural/template_distance/26_cmd.py (circuit metric distance)
- structural/polysemanticity/52_polysemanticity.py
- measurement/discriminant_validity/17_discriminant_validity.py

## C4 Minimality
- causal/mdl_slt/10_llc.py (local learning coefficient)
- causal/mdl_slt/29_hyperparam_sensitivity.py
- behavioral/mdl_compression/47_mdl_compression.py
- information/info_bottleneck/57_info_bottleneck.py

## C5 Convergent Validity
- measurement/convergent_validity/12_convergent_validity.py
- measurement/convergent_validity/23_nomological_validity.py
- measurement/convergent_validity/36_incremental_validity.py
- structural/weight_alignment/28_weight_eap_jaccard.py
- representational/subspace_alignment/64_subspace_alignment.py

## E1-E6 External
- behavioral/generalization_gap/46_generalization_gap.py (E1)
- causal/transportability/38_cross_model_invariance.py (E5/E6)
- causal/transportability/41_transportability.py (E5/E6)
- representational/cross_task_overlap/63_cross_task_overlap.py (E5)

## M1-M6 Measurement
- measurement/bootstrap_stability/11_bootstrap.py (M1)
- measurement/bootstrap_stability/30_seed_variance.py (M1)
- measurement/internal_consistency/16_reliability_suite.py (M1)
- measurement/test_retest/16_reliability_suite.py (M1)
- measurement/inter_rater/59_inter_rater.py (M1)
- measurement/measurement_invariance/13_measurement_invariance.py (M2)
- measurement/discriminant_validity/17_discriminant_validity.py (M3)
- measurement/sensitivity/14_derived_metrics.py (M4)
- behavioral/calibration/45_calibration.py (M5)
