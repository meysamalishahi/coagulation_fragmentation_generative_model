# CaloChallenge sampler calibration

Selection used 200 held-out validation events; test was evaluated once on 1000 events.

Selected leaf temperature: 0.7
Selected merge-rate scale: 1.5

## Test metrics

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| cardinality_tv | 0.347 | 0.212 |
| retained_response_w1 | 0.35557 | 0.377586 |
| longitudinal_centroid_w1 | 0.0434902 | 0.0417942 |
| transverse_centroid_x_w1 | 0.0590371 | 0.0642564 |
| transverse_centroid_y_w1 | 0.063527 | 0.0616883 |
| spatial_spread_w1 | 0.189621 | 0.180517 |
| summary_mmd | 0.649859 | 0.664505 |
| summary_energy_distance | 0.932454 | 1.01448 |
| classifier_two_sample_accuracy | 0.9775 | 0.97125 |
| coordinate_out_of_geometry_fraction | 0.017379 | 0.0143985 |
| mass_residual_mean | 0 | 2.22045e-19 |
| mass_residual_max | 0 | 1.11022e-16 |
| empty_sample_rate | 0.001 | 0 |
| overflow_rate | — | 0.082 |
| milliseconds_per_sample | 2700.84 | 1836.75 |
| peak_gpu_memory_mb | 37.3481 | 18.2334 |
