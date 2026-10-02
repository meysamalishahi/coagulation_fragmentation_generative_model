# Dataset-1 photon shape-model report

Trainable parameters: 25,944,964 (-0.143% versus ViT-CFM shape network).

Final-step validation loss: 2.02375
Best validation loss (diagnostic only): 2.02375
Training seconds: 36.774

## Shape-only validation metrics

Layer energies are supplied from the held-out truth here to isolate shape quality.
The final official sample instead uses independently generated layer energies.

| Metric | Value |
|---|---:|
| mean_profile_l1 | 0.882284 |
| reference_active_cells_mean | 224.031 |
| generated_active_cells_mean | 160.875 |
| active_cells_mean_error | 63.1562 |
| layer_conservation_max_mev | 0.3125 |
| milliseconds_per_shower | 14.8598 |
| peak_gpu_memory_mb | 792.592 |
| incident_response_mean | 0.964039 |
