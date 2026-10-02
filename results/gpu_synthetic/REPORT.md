# Continuous synthetic GPU benchmark

## Scope

This is the first end-to-end neural run, not the final five-seed comparison.
It verifies that the complete reverse model can be optimized and sampled on a
continuous hierarchical weighted-set distribution. The hidden data genealogy
is used only for evaluation.

The run used 1,024 training sets, 256 validation sets, 512 test sets, 12
epochs, and 512 generated samples. The model has 150,977 parameters and was
trained on an NVIDIA RTX A5000 with PyTorch 2.7.0/CUDA 12.6.

## Optimization

Training path NLL decreased from 29.8699 to 8.2997. Best validation path NLL
was 7.4136, and held-out test path NLL was 7.8116. Total training time was
166.36 seconds.

## Held-out event diagnostics

| Metric | Result |
|---|---:|
| Merge events | 1,846 |
| Merge-pair accuracy | 0.2465 |
| Target-pair probability | 0.1338 |
| Uniform target probability | 0.0568 |
| Candidate recall | 1.0000 |
| Terminal hidden-sibling top-pair accuracy | 0.5137 |
| Terminal hidden-sibling probability | 0.4072 |
| Terminal hidden-sibling uniform probability | 0.3339 |

The learned merge kernel assigns the observed reverse-corruption pair about
2.35 times the uniform probability. On the independently generated data tree,
the highest-scored terminal pair is a true leaf-sibling pair in 51.4% of test
sets. The data genealogy is not provided during training.

## Generated-set metrics

| Metric | Result |
|---|---:|
| Final-cardinality TV | 0.2168 |
| Component-mass Wasserstein-1 | 0.0426 |
| Pair-distance Wasserstein-1 | 0.2119 |
| Invariant-summary MMD | 0.0451 |
| Invariant-summary energy distance | 0.0596 |
| Classifier two-sample accuracy | 0.6707 |
| Empty-sample rate | 0.0410 |
| Mean/max conservation residual | 0.0 / 0.0 |
| Sampling time | 97.05 ms/set |

The cardinality metric includes an explicit overflow bin for generated counts
above the data cap of eight. The model produces 7.4% such samples, so count
calibration is a visible weakness rather than being silently discarded.

## Interpretation

This run establishes an end-to-end GPU implementation and meaningful learning
on continuous sets. It does not establish superiority over birth-only or
published baselines, and it is not yet a paper-level result. The next
synthetic stage should run five seeds and controlled sweeps over split
imbalance, interaction strength, and cardinality, followed by the planned
capacity-matched birth-only comparison.

