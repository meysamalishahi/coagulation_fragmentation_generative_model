# GPU synthetic campaign

All intervals are nonparametric 95% bootstrap intervals over seeds.
Training objectives are model-specific: path NLL for coagulation/birth-only and epsilon-MSE plus count NLL for diffusion, so they must not be compared across model families.

## paired: base / birth_only

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 1.84 | [1.8004, 1.8721] |
| test_path_nll | 1.84 | [1.8004, 1.8721] |
| cardinality_tv | 0.0506 | [0.0386, 0.064] |
| component_mass_w1 | 0.0038573 | [0.0028411, 0.0056241] |
| pair_distance_w1 | 0.021569 | [0.016163, 0.02951] |
| summary_mmd | 0.0064878 | [0.0055919, 0.007259] |
| summary_energy_distance | 0.0059797 | [-0.00087064, 0.01283] |
| classifier_two_sample_accuracy | 0.52175 | [0.512, 0.5315] |
| empty_sample_rate | 0.0002 | [0, 0.0006] |
| milliseconds_per_sample | 10.538 | [10.415, 10.661] |
| mass_residual_max | 0 | [0, 0] |

## paired: base / cardinality_diffusion

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 2.4334 | [2.4225, 2.4449] |
| cardinality_tv | 0.0826 | [0.0674, 0.0978] |
| component_mass_w1 | 0.012504 | [0.011006, 0.01413] |
| pair_distance_w1 | 0.034493 | [0.013873, 0.062193] |
| summary_mmd | 0.015823 | [0.0083635, 0.028151] |
| summary_energy_distance | 0.019326 | [0.013572, 0.025024] |
| classifier_two_sample_accuracy | 0.569 | [0.5475, 0.606] |
| empty_sample_rate | 0 | [0, 0] |
| milliseconds_per_sample | 0.27175 | [0.26432, 0.28211] |
| mass_residual_max | 1.153e-07 | [1.1101e-07, 1.1977e-07] |

## paired: base / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 3.7658 | [3.6913, 3.8376] |
| test_path_nll | 3.7658 | [3.6913, 3.8376] |
| cardinality_tv | 0.105 | [0.084, 0.126] |
| component_mass_w1 | 0.0075038 | [0.002903, 0.012933] |
| pair_distance_w1 | 0.033389 | [0.022995, 0.048825] |
| summary_mmd | 0.0068364 | [0.005552, 0.0081209] |
| summary_energy_distance | 0.010488 | [0.0051754, 0.015801] |
| classifier_two_sample_accuracy | 0.521 | [0.49825, 0.5385] |
| empty_sample_rate | 0.0002 | [0, 0.0006] |
| milliseconds_per_sample | 88.249 | [85.039, 91.426] |
| mass_residual_max | 0 | [0, 0] |

## paired: base / coagulation_calibrated

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 3.7658 | [3.6913, 3.8376] |
| test_path_nll | 3.7658 | [3.6913, 3.8376] |
| cardinality_tv | 0.1024 | [0.0806, 0.1252] |
| component_mass_w1 | 0.010848 | [0.0048812, 0.018218] |
| pair_distance_w1 | 0.040552 | [0.028723, 0.055315] |
| summary_mmd | 0.0080745 | [0.0053222, 0.01199] |
| summary_energy_distance | 0.010006 | [0.0034618, 0.016551] |
| classifier_two_sample_accuracy | 0.539 | [0.52275, 0.55375] |
| empty_sample_rate | 0 | [0, 0] |
| milliseconds_per_sample | 91.1 | [83.363, 100.06] |
| mass_residual_max | 0 | [0, 0] |

## sweep: cardinality_high / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 5.0938 | [4.873, 5.3147] |
| test_path_nll | 5.0938 | [4.873, 5.3147] |
| cardinality_tv | 0.17 | [0.1592, 0.178] |
| component_mass_w1 | 0.018427 | [0.016012, 0.021499] |
| pair_distance_w1 | 0.13205 | [0.11097, 0.15478] |
| summary_mmd | 0.027896 | [0.021333, 0.035824] |
| summary_energy_distance | 0.040996 | [0.029227, 0.053003] |
| classifier_two_sample_accuracy | 0.6315 | [0.614, 0.649] |
| empty_sample_rate | 0.014 | [0.0092, 0.0184] |
| milliseconds_per_sample | 147.92 | [144.34, 151.96] |
| mass_residual_max | 0 | [0, 0] |

## sweep: cardinality_low / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 5.9172 | [5.8362, 5.9981] |
| test_path_nll | 5.9172 | [5.8362, 5.9981] |
| cardinality_tv | 0.2088 | [0.1884, 0.2292] |
| component_mass_w1 | 0.055777 | [0.051325, 0.060229] |
| pair_distance_w1 | 0.17124 | [0.14225, 0.20176] |
| summary_mmd | 0.0422 | [0.038035, 0.047266] |
| summary_energy_distance | 0.035329 | [0.020419, 0.058174] |
| classifier_two_sample_accuracy | 0.6285 | [0.598, 0.6605] |
| empty_sample_rate | 0.0144 | [0.0124, 0.0172] |
| milliseconds_per_sample | 61.581 | [56.04, 68.852] |
| mass_residual_max | 0 | [0, 0] |

## sweep: imbalance_strong / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 5.1775 | [4.912, 5.443] |
| test_path_nll | 5.1775 | [4.912, 5.443] |
| cardinality_tv | 0.146 | [0.1272, 0.1636] |
| component_mass_w1 | 0.024797 | [0.02325, 0.026344] |
| pair_distance_w1 | 0.14284 | [0.1341, 0.15191] |
| summary_mmd | 0.023774 | [0.019182, 0.02775] |
| summary_energy_distance | 0.028268 | [0.018023, 0.039557] |
| classifier_two_sample_accuracy | 0.5835 | [0.5525, 0.6145] |
| empty_sample_rate | 0.012 | [0.0084, 0.0164] |
| milliseconds_per_sample | 92.094 | [88.736, 94.329] |
| mass_residual_max | 0 | [0, 0] |

## sweep: imbalance_weak / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 5.8341 | [5.6615, 6.045] |
| test_path_nll | 5.8341 | [5.6615, 6.045] |
| cardinality_tv | 0.1528 | [0.1396, 0.1684] |
| component_mass_w1 | 0.03059 | [0.024808, 0.036524] |
| pair_distance_w1 | 0.11701 | [0.10345, 0.13] |
| summary_mmd | 0.029044 | [0.021111, 0.036976] |
| summary_energy_distance | 0.035221 | [0.02, 0.055458] |
| classifier_two_sample_accuracy | 0.636 | [0.613, 0.6545] |
| empty_sample_rate | 0.012 | [0.0076, 0.0172] |
| milliseconds_per_sample | 91.254 | [86.293, 95.558] |
| mass_residual_max | 0 | [0, 0] |

## sweep: interaction_strong / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 6.0139 | [5.9511, 6.0677] |
| test_path_nll | 6.0139 | [5.9511, 6.0677] |
| cardinality_tv | 0.1436 | [0.1236, 0.1544] |
| component_mass_w1 | 0.031112 | [0.02759, 0.035413] |
| pair_distance_w1 | 0.14774 | [0.13868, 0.15966] |
| summary_mmd | 0.029985 | [0.026681, 0.033543] |
| summary_energy_distance | 0.030827 | [0.022295, 0.038538] |
| classifier_two_sample_accuracy | 0.6575 | [0.642, 0.6735] |
| empty_sample_rate | 0.006 | [0.0048, 0.0072] |
| milliseconds_per_sample | 93.774 | [90.817, 96.732] |
| mass_residual_max | 0 | [0, 0] |

## sweep: interaction_weak / coagulation

| Metric | Mean | 95% CI |
|---|---:|---:|
| test_training_objective | 5.9024 | [5.8727, 5.9269] |
| test_path_nll | 5.9024 | [5.8727, 5.9269] |
| cardinality_tv | 0.146 | [0.1336, 0.1584] |
| component_mass_w1 | 0.023382 | [0.019588, 0.026783] |
| pair_distance_w1 | 0.14109 | [0.12369, 0.15857] |
| summary_mmd | 0.027676 | [0.023649, 0.031702] |
| summary_energy_distance | 0.04034 | [0.026586, 0.05798] |
| classifier_two_sample_accuracy | 0.6 | [0.5785, 0.619] |
| empty_sample_rate | 0.0124 | [0.0088, 0.016] |
| milliseconds_per_sample | 94.016 | [89.727, 99.385] |
| mass_residual_max | 0 | [0, 0] |

