# Test Documentation Summary: test_belief.py

**Original File:** `/home/kobi/Documents/github/POMDPPlanners/POMDPPlanners/tests/test_core/test_belief.py`  
**Total Tests:** 133  
**Documented Tests:** 4  
**Documentation Coverage:** 3.0%

## Test Distribution by Type

- **Undocumented:** 129 tests
- **Unit:** 4 tests

---

## Test Documentation Details

### 1. `test_belief_config_id_identical_values_produces_same_hash`

**Line:** 8

**Purpose:** Validates that identical belief objects produce the same config_id hash for caching

**Given:** Two TestBelief instances initialized with identical values (42)

**When:** Config IDs are generated for both belief objects

**Then:** Both beliefs produce identical config_id values for proper cache functionality

**Test Type:** unit

---

### 2. `test_belief_config_id_different_values_produces_different_hash`

**Line:** 42

**Purpose:** Ensures belief objects with different values generate unique config_id hashes

**Given:** Two TestBelief instances with different values (42 and 43)

**When:** Config IDs are computed for both belief objects

**Then:** The beliefs produce different config_id values to prevent cache collisions

**Test Type:** unit

---

### 3. `test_belief_config_id_numpy_arrays_handles_identical_content`

**Line:** 76

**Purpose:** Verifies config_id generation works correctly with numpy array belief values

**Given:** Two TestBelief instances containing identical numpy arrays [1, 2, 3]

**When:** Config IDs are generated for both numpy-based beliefs

**Then:** Both beliefs produce identical config_id values despite being separate array objects

**Test Type:** unit

---

### 4. `test_belief_objects_usable_as_dictionary_keys_and_set_members`

**Line:** 110

**Purpose:** Validates belief objects can be used as dictionary keys and set members for caching

**Given:** Two identical TestBelief instances with same value (42)

**When:** Beliefs are used in sets and as dictionary keys

**Then:** Set deduplicates identical beliefs and dictionary access works with both objects

**Test Type:** unit

---

### 5. `test_belief_equality`

**Line:** 156

**Description:** Test belief equality.

---

### 6. `test_belief_equality_with_different_types`

**Line:** 180

**Description:** Test equality comparison with different types.

---

### 7. `test_weighted_particle_belief_config_id_deterministic`

**Line:** 198

**Description:** *No documentation available*

---

### 8. `test_weighted_particle_belief_config_id_changes_with_particles`

**Line:** 211

**Description:** *No documentation available*

---

### 9. `test_weighted_particle_belief_config_id_changes_with_weights`

**Line:** 224

**Description:** *No documentation available*

---

### 10. `test_weighted_particle_belief_config_id_with_numpy_particles`

**Line:** 237

**Description:** *No documentation available*

---

### 11. `test_weighted_particle_belief_config_id_with_mixed_particles`

**Line:** 250

**Description:** *No documentation available*

---

### 12. `test_weighted_particle_belief_config_id_with_resampling`

**Line:** 263

**Description:** *No documentation available*

---

### 13. `test_weighted_particle_belief_config_id_with_different_number_order`

**Line:** 274

**Description:** *No documentation available*

---

### 14. `test_weighted_particle_belief_config_id_with_different_numpy_order`

**Line:** 288

**Description:** *No documentation available*

---

### 15. `test_weighted_particle_belief_hashable`

**Line:** 302

**Description:** *No documentation available*

---

### 16. `test_weighted_particle_belief_equality`

**Line:** 320

**Description:** *No documentation available*

---

### 17. `test_weighted_particle_belief_equality_with_different_order`

**Line:** 344

**Description:** *No documentation available*

---

### 18. `test_weighted_particle_belief_equality_with_numpy_particles`

**Line:** 359

**Description:** *No documentation available*

---

### 19. `test_weighted_particle_belief_equality_with_resampling`

**Line:** 378

**Description:** *No documentation available*

---

### 20. `test_weighted_particle_belief_equality_with_different_types`

**Line:** 389

**Description:** *No documentation available*

---

### 21. `test_to_DiscreteDistribution_basic`

**Line:** 399

**Description:** Test basic conversion to DiscreteDistribution with simple particles.

---

### 22. `test_to_DiscreteDistribution_duplicate_particles`

**Line:** 414

**Description:** Test conversion with duplicate particles that should be combined.

---

### 23. `test_to_DiscreteDistribution_numpy_particles`

**Line:** 440

**Description:** Test conversion with numpy array particles.

---

### 24. `test_to_DiscreteDistribution_mixed_particles`

**Line:** 470

**Description:** Test conversion with mixed type particles.

---

### 25. `test_to_DiscreteDistribution_weight_normalization`

**Line:** 508

**Description:** Test that weights are properly normalized in the output distribution.

---

### 26. `test_create_belief_from_config_basic`

**Line:** 523

**Description:** *No documentation available*

---

### 27. `test_create_belief_particles_and_weights`

**Line:** 541

**Description:** *No documentation available*

---

### 28. `test_reinvigoration_discrete_light_dark`

**Line:** 558

**Description:** *No documentation available*

---

### 29. `test_reinvigoration_discrete_light_dark_full_coverage`

**Line:** 575

**Description:** *No documentation available*

---

### 30. `test_reinvigoration_continuous_light_dark_full_coverage`

**Line:** 591

**Description:** *No documentation available*

---

### 31. `test_reinvigoration_sanity_pomdp`

**Line:** 608

**Description:** *No documentation available*

---

### 32. `test_belief_update_basic`

**Line:** 626

**Description:** Test basic belief update functionality.

---

### 33. `test_belief_update_with_resampling`

**Line:** 651

**Description:** Test belief update with resampling enabled.

---

### 34. `test_belief_update_state_transitions`

**Line:** 675

**Description:** Test that belief update correctly handles state transitions.

---

### 35. `test_belief_update_observation_probabilities`

**Line:** 692

**Description:** Test that belief update correctly computes observation probabilities.

---

### 36. `test_belief_update_with_tiger_pomdp`

**Line:** 709

**Description:** Test belief update with TigerPOMDP environment.

---

### 37. `test_belief_update_preserves_particle_count`

**Line:** 727

**Description:** Test that belief update preserves the number of particles.

---

### 38. `test_belief_update_weight_normalization`

**Line:** 743

**Description:** Test that belief update properly handles weight normalization.

---

### 39. `test_belief_update_with_extreme_weights`

**Line:** 762

**Description:** Test belief update with extreme weight values.

---

### 40. `test_belief_update_consistency`

**Line:** 779

**Description:** Test that belief update produces consistent results.

---

### 41. `test_belief_update_with_different_actions`

**Line:** 799

**Description:** Test that belief update behaves differently for different actions.

---

### 42. `test_weighted_particle_belief_state_update_initialization_empty`

**Line:** 823

**Description:** Test WeightedParticleBeliefStateUpdate initialization with empty lists.

---

### 43. `test_weighted_particle_belief_state_update_initialization_with_data`

**Line:** 832

**Description:** Test WeightedParticleBeliefStateUpdate initialization with particles and weights.

---

### 44. `test_weighted_particle_belief_state_update_initialization_weight_sum`

**Line:** 844

**Description:** Test that weights_sum is calculated correctly during initialization.

---

### 45. `test_weighted_particle_belief_state_update_inplace_update`

**Line:** 854

**Description:** Test inplace_update method adds state and weight correctly.

---

### 46. `test_weighted_particle_belief_state_update_inplace_update_observation_probability`

**Line:** 879

**Description:** Test that inplace_update correctly computes observation probability.

---

### 47. `test_weighted_particle_belief_state_update_multiple_inplace_updates`

**Line:** 900

**Description:** Test multiple inplace_updates accumulate correctly.

---

### 48. `test_weighted_particle_belief_state_update_update_returns_new_instance`

**Line:** 920

**Description:** Test that update method returns a new instance.

---

### 49. `test_weighted_particle_belief_state_update_update_preserves_original_data`

**Line:** 946

**Description:** Test that update method preserves original particles and weights.

---

### 50. `test_weighted_particle_belief_state_update_inplace_vs_update_comparison`

**Line:** 971

**Description:** Test that inplace_update modifies the belief in-place while update returns a new belief.

---

### 51. `test_weighted_particle_belief_state_update_sample_basic`

**Line:** 1050

**Description:** Test basic sampling functionality.

---

### 52. `test_weighted_particle_belief_state_update_sample_uniform_weights`

**Line:** 1069

**Description:** Test sampling with uniform weights.

---

### 53. `test_weighted_particle_belief_state_update_sample_single_particle`

**Line:** 1089

**Description:** Test sampling with a single particle.

---

### 54. `test_weighted_particle_belief_state_update_sample_empty_belief_raises_error`

**Line:** 1102

**Description:** Test that sampling from empty belief raises ValueError.

---

### 55. `test_weighted_particle_belief_state_update_sample_zero_weights_raises_error`

**Line:** 1110

**Description:** Test that sampling with zero weights raises ValueError.

---

### 56. `test_weighted_particle_belief_state_update_sample_empty_particles_raises_error`

**Line:** 1121

**Description:** Test that creating belief with mismatched particles and weights raises ValueError.

---

### 57. `test_weighted_particle_belief_state_update_sample_normalization`

**Line:** 1131

**Description:** Test that sampling works correctly with unnormalized weights.

---

### 58. `test_weighted_particle_belief_state_update_with_tiger_pomdp`

**Line:** 1154

**Description:** Test WeightedParticleBeliefStateUpdate integration with TigerPOMDP.

---

### 59. `test_weighted_particle_belief_state_update_with_tiger_pomdp_different_observations`

**Line:** 1181

**Description:** Test WeightedParticleBeliefStateUpdate with different observations in TigerPOMDP.

---

### 60. `test_weighted_particle_belief_state_update_inheritance`

**Line:** 1202

**Description:** Test that WeightedParticleBeliefStateUpdate inherits from Belief.

---

### 61. `test_weighted_particle_belief_state_update_config_id`

**Line:** 1209

**Description:** Test that WeightedParticleBeliefStateUpdate has a config_id property.

---

### 62. `test_weighted_particle_belief_state_update_config_id_deterministic`

**Line:** 1221

**Description:** Test that identical beliefs have identical config_ids.

---

### 63. `test_weighted_particle_belief_state_update_config_id_changes_with_particles`

**Line:** 1234

**Description:** Test that config_id changes when particles change.

---

### 64. `test_weighted_particle_belief_state_update_config_id_changes_with_weights`

**Line:** 1246

**Description:** Test that config_id changes when weights change.

---

### 65. `test_weighted_particle_belief_state_update_config_id_with_numpy_particles`

**Line:** 1258

**Description:** Test config_id with numpy array particles.

---

### 66. `test_weighted_particle_belief_state_update_config_id_with_mixed_particles`

**Line:** 1272

**Description:** Test config_id with mixed type particles.

---

### 67. `test_weighted_particle_belief_state_update_config_id_with_different_order`

**Line:** 1286

**Description:** Test config_id with particles and weights in different order.

---

### 68. `test_weighted_particle_belief_state_update_config_id_with_different_numpy_order`

**Line:** 1301

**Description:** Test config_id with numpy array particles in different order.

---

### 69. `test_weighted_particle_belief_state_update_config_id_with_empty_belief`

**Line:** 1316

**Description:** Test config_id with empty belief.

---

### 70. `test_weighted_particle_belief_state_update_config_id_with_single_particle`

**Line:** 1325

**Description:** Test config_id with single particle.

---

### 71. `test_weighted_particle_belief_state_update_config_id_with_duplicate_particles`

**Line:** 1339

**Description:** Test config_id with duplicate particles.

---

### 72. `test_weighted_particle_belief_state_update_config_id_with_extreme_weights`

**Line:** 1353

**Description:** Test config_id with extreme weight values.

---

### 73. `test_weighted_particle_belief_state_update_config_id_uniqueness`

**Line:** 1367

**Description:** Test that config_id is unique for different beliefs.

---

### 74. `test_weighted_particle_belief_state_update_config_id_consistency`

**Line:** 1381

**Description:** Test that config_id is consistent across multiple calls.

---

### 75. `test_weighted_particle_belief_state_update_equality`

**Line:** 1395

**Description:** Test equality comparison between WeightedParticleBeliefStateUpdate instances.

---

### 76. `test_weighted_particle_belief_state_update_inequality`

**Line:** 1410

**Description:** Test inequality comparison between WeightedParticleBeliefStateUpdate instances.

---

### 77. `test_weighted_particle_belief_state_update_hashable`

**Line:** 1428

**Description:** Test that WeightedParticleBeliefStateUpdate instances are hashable.

---

### 78. `test_weighted_particle_belief_state_update_edge_cases`

**Line:** 1447

**Description:** Test edge cases for WeightedParticleBeliefStateUpdate.

---

### 79. `test_weighted_particle_belief_state_update_comprehensive_usage_example`

**Line:** 1474

**Description:** Test the comprehensive WeightedParticleBeliefStateUpdate usage example from the class docstring.

---

### 80. `test_weighted_particle_belief_state_update_update_method_example`

**Line:** 1525

**Description:** Test the update method usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 81. `test_weighted_particle_belief_state_update_inplace_update_example`

**Line:** 1560

**Description:** Test the inplace_update method usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 82. `test_weighted_particle_belief_state_update_sample_method_example`

**Line:** 1586

**Description:** Test the sample method usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 83. `test_unweighted_particle_belief_state_update_initialization_empty`

**Line:** 1614

**Description:** Test UnweightedParticleBeliefStateUpdate initialization with empty list.

---

### 84. `test_unweighted_particle_belief_state_update_initialization_with_data`

**Line:** 1622

**Description:** Test UnweightedParticleBeliefStateUpdate initialization with particles.

---

### 85. `test_unweighted_particle_belief_state_update_initialization_weight_sum`

**Line:** 1632

**Description:** Test that weights_sum is calculated correctly during initialization.

---

### 86. `test_unweighted_particle_belief_state_update_inplace_update`

**Line:** 1641

**Description:** Test inplace_update method adds state correctly.

---

### 87. `test_unweighted_particle_belief_state_update_inplace_update_ignores_observation`

**Line:** 1665

**Description:** Test that inplace_update doesn't use observation probability (unweighted).

---

### 88. `test_unweighted_particle_belief_state_update_multiple_inplace_updates`

**Line:** 1684

**Description:** Test multiple inplace_updates accumulate correctly.

---

### 89. `test_unweighted_particle_belief_state_update_update_returns_new_instance`

**Line:** 1703

**Description:** Test that update method returns a new instance.

---

### 90. `test_unweighted_particle_belief_state_update_update_preserves_original_data`

**Line:** 1729

**Description:** Test that update method preserves original particles.

---

### 91. `test_unweighted_particle_belief_state_update_inplace_vs_update_comparison`

**Line:** 1752

**Description:** Test that inplace_update modifies the belief in-place while update returns a new belief.

---

### 92. `test_unweighted_particle_belief_state_update_sample_basic`

**Line:** 1791

**Description:** Test basic sampling functionality.

---

### 93. `test_unweighted_particle_belief_state_update_sample_uniform_distribution`

**Line:** 1809

**Description:** Test sampling produces roughly uniform distribution.

---

### 94. `test_unweighted_particle_belief_state_update_sample_single_particle`

**Line:** 1828

**Description:** Test sampling with a single particle.

---

### 95. `test_unweighted_particle_belief_state_update_sample_duplicate_particles`

**Line:** 1840

**Description:** Test sampling with duplicate particles.

---

### 96. `test_unweighted_particle_belief_state_update_sample_empty_belief_raises_error`

**Line:** 1862

**Description:** Test that sampling from empty belief raises ValueError.

---

### 97. `test_unweighted_particle_belief_state_update_with_tiger_pomdp`

**Line:** 1870

**Description:** Test UnweightedParticleBeliefStateUpdate integration with TigerPOMDP.

---

### 98. `test_unweighted_particle_belief_state_update_inheritance`

**Line:** 1895

**Description:** Test that UnweightedParticleBeliefStateUpdate inherits from Belief.

---

### 99. `test_unweighted_particle_belief_state_update_config_id`

**Line:** 1902

**Description:** Test that UnweightedParticleBeliefStateUpdate has a config_id property.

---

### 100. `test_unweighted_particle_belief_state_update_config_id_deterministic`

**Line:** 1913

**Description:** Test that identical beliefs have identical config_ids.

---

### 101. `test_unweighted_particle_belief_state_update_config_id_changes_with_particles`

**Line:** 1924

**Description:** Test that config_id changes when particles change.

---

### 102. `test_unweighted_particle_belief_state_update_config_id_with_numpy_particles`

**Line:** 1935

**Description:** Test config_id with numpy array particles.

---

### 103. `test_unweighted_particle_belief_state_update_config_id_with_mixed_particles`

**Line:** 1947

**Description:** Test config_id with mixed type particles.

---

### 104. `test_unweighted_particle_belief_state_update_config_id_with_different_order`

**Line:** 1959

**Description:** Test config_id with particles in different order.

---

### 105. `test_unweighted_particle_belief_state_update_config_id_with_different_numpy_order`

**Line:** 1972

**Description:** Test config_id with numpy array particles in different order.

---

### 106. `test_unweighted_particle_belief_state_update_config_id_with_empty_belief`

**Line:** 1985

**Description:** Test config_id with empty belief.

---

### 107. `test_unweighted_particle_belief_state_update_config_id_with_single_particle`

**Line:** 1994

**Description:** Test config_id with single particle.

---

### 108. `test_unweighted_particle_belief_state_update_config_id_with_duplicate_particles`

**Line:** 2006

**Description:** Test config_id with duplicate particles.

---

### 109. `test_unweighted_particle_belief_state_update_config_id_different_duplicates`

**Line:** 2018

**Description:** Test config_id with different numbers of duplicate particles.

---

### 110. `test_unweighted_particle_belief_state_update_config_id_uniqueness`

**Line:** 2030

**Description:** Test that config_id is unique for different beliefs.

---

### 111. `test_unweighted_particle_belief_state_update_config_id_consistency`

**Line:** 2044

**Description:** Test that config_id is consistent across multiple calls.

---

### 112. `test_unweighted_particle_belief_state_update_equality`

**Line:** 2057

**Description:** Test equality comparison between UnweightedParticleBeliefStateUpdate instances.

---

### 113. `test_unweighted_particle_belief_state_update_inequality`

**Line:** 2070

**Description:** Test inequality comparison between UnweightedParticleBeliefStateUpdate instances.

---

### 114. `test_unweighted_particle_belief_state_update_hashable`

**Line:** 2087

**Description:** Test that UnweightedParticleBeliefStateUpdate instances are hashable.

---

### 115. `test_unweighted_particle_belief_state_update_edge_cases`

**Line:** 2104

**Description:** Test edge cases for UnweightedParticleBeliefStateUpdate.

---

### 116. `test_unweighted_particle_belief_state_update_comprehensive_usage_example`

**Line:** 2123

**Description:** Test comprehensive usage example similar to the weighted version.

---

### 117. `test_unweighted_particle_belief_state_update_differences_from_weighted`

**Line:** 2171

**Description:** Test that UnweightedParticleBeliefStateUpdate behaves differently from WeightedParticleBeliefStateUpdate.

---

### 118. `test_unweighted_particle_belief_state_update_sample_with_extreme_cases`

**Line:** 2207

**Description:** Test sampling behavior with extreme cases.

---

### 119. `test_weighted_particle_belief_basic_incremental_construction_usage_example`

**Line:** 2238

**Description:** Test the basic incremental belief construction usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 120. `test_weighted_particle_belief_immutable_updates_usage_example`

**Line:** 2265

**Description:** Test the immutable belief updates for tree search usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 121. `test_weighted_particle_belief_update_strategies_comparison_usage_example`

**Line:** 2308

**Description:** Test the comparing belief update strategies usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 122. `test_weighted_particle_belief_mcts_integration_usage_example`

**Line:** 2338

**Description:** Test the Monte Carlo Tree Search integration usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 123. `test_weighted_particle_belief_weighted_sampling_usage_example`

**Line:** 2386

**Description:** Test the weighted sampling and state estimation usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 124. `test_weighted_particle_belief_config_id_caching_usage_example`

**Line:** 2423

**Description:** Test the configuration ID and caching usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 125. `test_weighted_particle_belief_custom_particle_types_usage_example`

**Line:** 2448

**Description:** Test the custom particle types usage example from WeightedParticleBeliefStateUpdate docstring.

---

### 126. `test_unweighted_particle_belief_state_update_basic_uniform_belief_construction_usage_example`

**Line:** 2491

**Description:** Test the basic uniform belief construction usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 127. `test_unweighted_particle_belief_state_update_mcts_with_uniform_beliefs_usage_example`

**Line:** 2513

**Description:** Test the Monte Carlo Tree Search with uniform beliefs usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 128. `test_unweighted_particle_belief_state_update_comparing_weighted_vs_unweighted_usage_example`

**Line:** 2549

**Description:** Test the comparing weighted vs unweighted belief updates usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 129. `test_unweighted_particle_belief_state_update_discrete_observation_filtering_usage_example`

**Line:** 2588

**Description:** Test the discrete observation filtering usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 130. `test_unweighted_particle_belief_state_update_immutable_belief_trees_usage_example`

**Line:** 2625

**Description:** Test the immutable belief trees for planning usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 131. `test_unweighted_particle_belief_state_update_memory_efficient_accumulation_usage_example`

**Line:** 2668

**Description:** Test the memory-efficient particle accumulation usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 132. `test_unweighted_particle_belief_state_update_configuration_caching_usage_example`

**Line:** 2694

**Description:** Test the configuration caching and equality usage example from UnweightedParticleBeliefStateUpdate docstring.

---

### 133. `test_unweighted_particle_belief_state_update_large_scale_accumulation_usage_example`

**Line:** 2720

**Description:** Test the large-scale particle accumulation usage example from UnweightedParticleBeliefStateUpdate docstring.

---

