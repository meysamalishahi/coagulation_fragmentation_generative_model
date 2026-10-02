# Neural GPU model

`gpu_model.py` implements the full trainable reverse process from the paper:

- permutation-equivariant Transformer set encoder;
- categorical leaf-budget model;
- coupled beta/Gaussian-mixture birth marks, including the `1 / reservoir`
  mass-coordinate Jacobian;
- symmetric pair probabilities and a separately learned total merge rate;
- dense or symmetric k-nearest-neighbor candidate pairs;
- the marked point-process path loss, including a stratified Monte Carlo
  estimate of the merge survival integral;
- exact forward fragmentation/deletion history construction; and
- conservative event-driven reverse sampling with thinning.

Run the correctness and CUDA smoke tests in the available GPU environment:

```bash
conda run -n tabkde python -m unittest -v test_gpu_model.py
```

The module is dataset-independent. The continuous synthetic benchmark is a
separate driver and imports this implementation rather than duplicating it:

```bash
conda run -n tabkde python gpu_synthetic_benchmark.py \
  --output-dir results/gpu_synthetic
```

Use `--split-alpha`, `--interaction-strength`, `--cardinality-mean`, and
`--max-cardinality` to create the controlled regimes required by the paper.
The driver writes a checkpoint, raw training history, sample/cardinality
metrics, plots, environment metadata, and the full run configuration.

## Detached full run

Start the 4,000-example, 30-epoch run as the transient user service
`coagulation-gpu-synthetic-full`. It survives terminal closure and is not a
child of the launching shell:

```bash
sh run_gpu_synthetic_full_detached.sh
```

Check its PID and recent log output without attaching to it:

```bash
sh check_gpu_synthetic_full.sh
```

Outputs, the PID file, and `run.log` are kept under
`results/gpu_synthetic_full/`.

## Complete experiment campaign

`gpu_experiment_campaign.py` runs the requested experiments sequentially on a
single GPU:

- five matched seeds of the coagulation model and equal-parameter birth-only
  ablation;
- five seeds of a cardinality-conditioned, permutation-equivariant DDPM;
- validation-only count calibration for every coagulation seed;
- five-seed sweeps over split imbalance, interaction strength, and
  cardinality; and
- the conditional CaloChallenge Dataset 1 photon experiment.

The campaign is resumable: a directory with a completed `summary.json` is
skipped. Start it detached from the terminal and inspect it with:

```bash
sh run_gpu_campaign_detached.sh
sh check_gpu_campaign.sh
```

The aggregate CSV files, bootstrap intervals, report, individual logs, and
progress record are written below `results/gpu_campaign/`.

## Real calorimeter data

Download and verify the official CaloChallenge Dataset 1 photon train and
evaluation files (DOI `10.5281/zenodo.8099322`) with:

```bash
sh download_calorimeter_data.sh
```

`calorimeter_data.py` reads the official XML geometry and converts each shower
to a conservative weighted set. The component mass is deposited energy divided
by incident energy; the reservoir contains unrepresented response and energy
from cells omitted by the 1%-of-incident-energy threshold/top-k reduction.
This threshold gives a non-saturated sparse-set cardinality distribution on
the official training sample. Features are normalized
`(layer, x, y)` cell centers, and standardized log incident energy is supplied
through the model's conditioning input. `gpu_calorimeter_benchmark.py` trains
against the official training file and evaluates only on the official held-out
file.

Calorimeter generation is batched across independent event trajectories while
preserving a separate event clock and thinning decision for each shower. It
prints the completed count, throughput, and ETA, and atomically writes
`sampling_checkpoint.pt` every 25 samples by default. That checkpoint includes
partial CPU samples, Python and PyTorch RNG states, elapsed time, peak GPU
memory, and data/model fingerprints, so an interrupted invocation resumes
without repeating completed samples or silently mixing configurations. Adjust
the controls with `--sample-batch-size` and `--sample-checkpoint-every`.

Calibrate a completed real-data model without tuning on the official test file:

```bash
sh run_gpu_calorimeter_calibration_detached.sh
sh check_gpu_calorimeter_calibration.sh
```

The grid is selected on the held-out validation slice. Only the selected
temperature and merge-rate scale are then evaluated on the official test
file. Every grid point and the final test generation have separate resumable
checkpoints below the source run's `calibration/` directory.

The real-data driver also supports the equal-parameter birth-only ablation via
`--model-type birth_only`; it automatically disables forward splitting. A
small engineering pilot can be run, after the calibration gate, with:

```bash
sh run_gpu_calorimeter_birth_pilot_detached.sh
sh check_gpu_calorimeter_birth_pilot.sh
```

After that pilot passes, run the scientifically matched 4,000/800-event,
25-epoch birth-only baseline with:

```bash
sh run_gpu_calorimeter_birth_full_detached.sh
sh check_gpu_calorimeter_birth_full.sh
```

## Official Dataset-1 comparison

The completed 4,000-event experiments above are engineering experiments, not
direct CaloChallenge leaderboard comparisons. The official comparison path
must use the complete 368-voxel representation, the full first photon file for
training, an independently generated shower response, and the public
low-/high-level classifier AUC evaluation.

The reference ViT-CFM configuration contains 25,982,005 trainable parameters
in its shape network and 1,956,225 in its energy network (27,938,230 total).
The parameter-matched condensation experiment will count every learned module
used during inference and target this combined budget within one percent. The
existing 794,875-parameter model remains a separately reported compact model.

`calorimeter_data.py` now provides the protocol primitives for this path:

- dense HDF5 loading with the official 1 MeV evaluation threshold;
- the canonical Dataset-1 incident-energy schedule;
- strict weighted-set-to-368-voxel rasterization; and
- atomic export in the official `showers`/`incident_energies` HDF5 format.

`VoxelBirthOnlyModel` constrains every generated core component to an official
detector cell and masks occupied cells, preventing invalid coordinates and
duplicate core hits. A full-file audit shows why the official model cannot be
implemented by merely increasing `max_components`: after the 1 MeV cut, the
median shower has 259 active cells and the 99th percentile has 367. The fair
implementation therefore uses the sequential condensation process for the
energetic core and a dense stochastic residual decoder for the remaining
low-energy cells. This keeps the official output intact without requiring
hundreds of sequential birth decisions per shower.

The independently trained five-layer response model is implemented by
`calorimeter_energy.py` and `gpu_calorimeter_energy.py`. It has 1,951,789
trainable parameters, 0.227% below the ViT-CFM energy-network budget, and does
not read deposited energy from test showers. Run the complete published-budget
training as a detached user service and check it with:

```bash
sh run_gpu_calorimeter_official_energy_detached.sh
sh check_gpu_calorimeter_official_energy.sh
```

The run uses the first 119,790 showers for training, the last 1,210 for
validation, AdamW with learning rate `1e-4` and weight decay `0.1`, batch size
256, cosine decay, and 250,000 optimizer steps. Outputs are stored below
`results/gpu_official/energy_full/`.

For parity with the published `es_load_best_model: false` setting, the primary
`model.pt`, reported metrics, and generated validation sample always use the
final 250,000-step weights. The best-validation state is retained only as the
explicitly labelled `best_model_diagnostic.pt` and is not used in the fair
comparison.

The full-geometry shape component is implemented in `calorimeter_shape.py`
and trained by `gpu_calorimeter_shape.py`. It uses a sequential categorical
birth process for the 48-cell energetic core and a conditional Dirichlet
decoder for all remaining official cells. Generated core energy is projected
only when necessary to respect the independently generated layer budgets; the
residual decoder then fills the exact remaining energy in each layer. The
result is nonnegative, uses all 368 cells, and conserves every generated layer
energy by construction.

The shape model contains 25,944,964 trainable parameters, 0.143% below the
25,982,005-parameter ViT-CFM shape network. Together with the 1,951,789
parameter energy model, the complete generator contains 27,896,753 parameters,
0.148% below the 27,938,230 reference total. Every learned module used for
generation is included in this count.

The shape training protocol matches the published comparison on the main
resource axes: the 119,790/1,210 train/validation split, batch size 64, 800,000
optimizer steps, AdamW at `1e-4` with weight decay `0.1`, cosine decay, six
blocks, six attention heads, and final-step checkpoint selection. First run
the full-architecture smoke test; only its successful completion unlocks the
long detached run:

```bash
sh run_gpu_calorimeter_official_shape_smoke_detached.sh
sh check_gpu_calorimeter_official_shape.sh
sh run_gpu_calorimeter_official_shape_detached.sh
```

The final official HDF5 generator is
`gpu_calorimeter_generate_official.py`. It samples the trained energy model,
conditions the shape model only on those generated energies, writes 121,000
showers in the public 368-voxel format, logs progress, saves restart state, and
uses generation batches of 100 for the timing comparison. No deposited energy
or layer information is read from the test file during generation.

The 121,000 count follows the reference code's executable Dataset-1 helper
(1,000 complete 121-condition schedules). The reference YAML still says
`n_samples: 100000`; this inconsistency is recorded, and a 100,000-event
sensitivity evaluation will also be reported rather than silently choosing
one interpretation.

The final report will keep two evaluation labels separate: (1) the exact
ViT-CFM paper pipeline and its classifier settings, and (2) the public
CaloChallenge evaluator. This avoids silently mixing the paper repository's
pre-evaluation cut with the public evaluator's 1 MeV cell threshold. The
headline SOTA comparison is valid only after the full shape training,
independent 121,000-shower generation, and low-/high-level classifier AUC,
KPD, FPD, speed, and memory evaluations have completed.

For unattended execution, `run_gpu_calorimeter_official_end_to_end_detached.sh`
starts a separate supervisor service. It monitors the shape service, retries or
resumes it up to three times, validates the final-step checkpoint, resumes
partial HDF5 generation, runs both evaluation protocols, and writes
`results/gpu_official/fair_comparison/FINAL_REPORT.md`. Its machine-readable
heartbeat is `supervisor_status.json`; unrecoverable errors are written to
`supervisor_failure.json`. Check every stage with:

```bash
sh check_gpu_calorimeter_official_comparison.sh
```
