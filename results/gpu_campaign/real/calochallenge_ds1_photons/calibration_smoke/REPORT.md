# CaloChallenge sampler calibration

Selection used 8 held-out validation events; test was evaluated once on 8 events.

Selected leaf temperature: 1
Selected merge-rate scale: 1

## Test metrics

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| cardinality_tv | 0.347 | 0.75 |
| retained_response_w1 | 0.35557 | 0.336597 |
| longitudinal_centroid_w1 | 0.0434902 | 0.0419672 |
| transverse_centroid_x_w1 | 0.0590371 | 0.0213405 |
| transverse_centroid_y_w1 | 0.063527 | 0.0319249 |
| spatial_spread_w1 | 0.189621 | 0.151506 |
| summary_mmd | 0.649859 | 0.513859 |
| summary_energy_distance | 0.932454 | 0.877974 |
| classifier_two_sample_accuracy | 0.9775 | 0.857143 |
| coordinate_out_of_geometry_fraction | 0.017379 | 0.0157978 |
| mass_residual_mean | 0 | 0 |
| mass_residual_max | 0 | 0 |
| empty_sample_rate | 0.001 | 0 |
| overflow_rate | — | 0 |
| milliseconds_per_sample | 2700.84 | 1215.62 |
| peak_gpu_memory_mb | 37.3481 | 17.2261 |
