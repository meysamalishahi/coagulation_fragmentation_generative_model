# Consolidated CPU validation report

## Executive conclusion

The CPU work supports the **mathematical implementation** and the
**small-scale viability** of the coagulation mechanism. It does not yet
support the claim that the proposed model beats modern variable-dimensional
generative models.

The strongest results are:

- all 13 mathematical/numerical checks pass;
- exact finite-state probability-flux reversal holds to machine precision;
- a tabular marked-event likelihood recovers analytical reverse rates with
  0.54% aggregate relative error;
- a learned reverse merge kernel improves held-out event likelihood and
  conservative parent reconstruction over uniform and distance-only kernels;
- a matched four-parameter structured density strongly outperforms independent
  insertion on a controlled shared-parent distribution;
- sparse pair scoring is about 79x faster than full enumeration at
  cardinality 1024 in the reported microbenchmark.

These results make the project ready for a neural GPU implementation. They do
not make the current draft empirically ready for ICML.

## 1. Mathematical and implementation checks

The dependency-free correctness suite was run with seed 20260930. It also
passed under two additional seeds during robustness testing.

| Check | Observed error | Tolerance | Result |
|---|---:|---:|---|
| Event maps, inversion, and conservation | 5.55e-16 | 2e-12 | Pass |
| Closed-form flow and inverse | 3.73e-16 | 2e-12 | Pass |
| Normalized encoder/decoder | 2.22e-16 | 2e-12 | Pass |
| Split Jacobian | 4.65e-10 | 2e-7 | Pass |
| Dequantization Jacobian | 4.05e-10 | 1e-6 | Pass |
| Flow Jacobian | 1.40e-9 | 1e-6 | Pass |
| Continuous split change of variables | 0.268% | 0.8% | Pass |
| Finite exact time reversal | 2.22e-16 | 2e-10 | Pass |
| Reverse-rate tabular MLE | 0.541% | 1.2% | Pass |
| Uniform birth bridge | 0.100% | 0.25% | Pass |
| Survival-integral estimator | 0.029% | 0.25% | Pass |
| Constant-rate path likelihood | 0.00119 absolute | 0.004 | Pass |
| Constant-rate thinning | 0.205% | 0.5% | Pass |

The finite benchmark enumerated 321 unordered states and tested 23 positive
reverse edges. The exact flux and reverse Kolmogorov derivative residuals were
at floating-point precision.

Eight core unit tests and five extended-experiment unit tests also pass. The
extended tests explicitly check conservation and centroid preservation for
synthetic fragmentation, one reverse merge per parent, invariance to the
array order used to represent an unordered set, and the structured-coordinate
inverse and Jacobian.

## 2. Learned reverse merge experiment

Five independent seeds were run. For each seed, 1,000 training sets and 400
test sets were generated. Parent components were conservatively split into
two leaves, and the reverse marked-event likelihood was trained on the
resulting histories. Values below are mean plus or minus a 95% t-interval over
the five seeds.

| Model | Event NLL | Sibling accuracy | Exact reconstruction | Reconstruction error |
|---|---:|---:|---:|---:|
| Uniform | 2.9830 ± 0.0354 | 0.4166 ± 0.0127 | 0.0840 ± 0.0142 | 0.7085 ± 0.0157 |
| Distance-only | 2.1588 ± 0.0694 | 0.4071 ± 0.0144 | 0.1720 ± 0.0200 | 0.5718 ± 0.0189 |
| Learned full kernel | **1.7164 ± 0.0736** | **0.6047 ± 0.0237** | **0.2855 ± 0.0315** | **0.4442 ± 0.0248** |

The full kernel uses distance, mass-ratio, combined-mass, and centroid
features. It improves event NLL and reconstruction, showing that the proposed
marked reverse likelihood is trainable and that pair information beyond
distance is useful on this synthetic process.

This is not an end-to-end neural generation result. The absolute exact
reconstruction rate of 28.6% also shows substantial room for improvement.

![Learned reverse merge results](cpu_extended/merge_recovery.png)

## 3. Independent birth versus structured coordinates

This is a controlled positive-control experiment. Data pairs share a latent
parent center and have lognormal separation. Both fitted models have four
scalar parameters:

- independent birth: separate Gaussian mean and standard deviation for the
  two insertions;
- structured coordinates: Gaussian parent center and Gaussian log-separation,
  including the exact coordinate Jacobian in the likelihood.

Each seed used 12,000 training and 20,000 test pairs.

| Model | Held-out NLL | Separation W1 | Covariance error | Energy distance |
|---|---:|---:|---:|---:|
| Independent birth | 2.2194 ± 0.0148 | 0.8210 ± 0.0152 | 0.6895 ± 0.0101 | 0.2334 ± 0.0140 |
| Structured coordinates | **0.5630 ± 0.0044** | **0.0019 ± 0.0014** | **0.0090 ± 0.0077** | **0.0028 ± 0.0057** |

The structured model wins decisively, as it should when the data truly have a
shared-parent mechanism. This confirms the intended inductive bias and the
likelihood implementation. It does not prove an advantage on natural data,
because the benchmark was deliberately generated from that bias.

![Independent versus structured](cpu_extended/birth_vs_structured.png)

## 4. CPU scaling

Full pair scoring and a one-dimensional k=8 neighborhood rule were measured
from cardinality 16 through 1024. At N=1024:

- full enumeration: 523,776 candidate pairs and 6.18 ms per evaluation;
- sparse k=8 rule: 8,156 candidate pairs and 0.078 ms per evaluation;
- measured speedup: approximately 79x.

![Pair-scoring scaling](cpu_extended/pair_scaling.png)

Sparse scoring is useful only if the correct merge remains in the candidate
set. A real-data study must therefore report candidate recall as well as
speed.

## 5. What is now supported

| Claim | Status |
|---|---|
| Event maps conserve scalar mass | Numerically verified to machine precision and proved in the paper |
| Split, dequantization, and flow Jacobians are implemented correctly | Verified by finite differences and Monte Carlo integration |
| The continuous split map produces the reverse 1/m factor | Verified by independent Monte Carlo integrals |
| Finite probability-flux reversal is correct | Verified exactly on the enumerated process |
| Birth bridge, survival likelihood, and thinning are implemented correctly | Verified against analytic answers |
| Reverse merge rates can be fitted through a marked-event likelihood | Supported by tabular and synthetic learned experiments |
| Structured coordinates can beat independent insertion | Supported only on the controlled positive-control distribution |
| Coagulation improves a real generative model | Not established |
| The method beats published baselines | Not tested |
| The empirical evidence is sufficient for ICML | No |

## 6. Remaining work

The remaining decisive experiments are:

1. implement the complete neural birth-and-merge sampler;
2. compare with a capacity-matched learned birth-only model;
3. run Jump Diffusion and at least one continuous set-generation baseline;
4. evaluate on a real domain with meaningful scalar mass;
5. report held-out likelihood or ELBO, set quality, cardinality calibration,
   conservation, runtime, and five-seed confidence intervals;
6. test sparse-pair candidate recall and quality-speed tradeoffs.

These experiments could technically be made to run very slowly on a CPU, but
GPU access is now practically justified. Further CPU-only toy experiments are
unlikely to change the paper's central empirical uncertainty.

## Reproduction

Run the complete CPU workflow from the project directory:

```bash
sh run_all_cpu_experiments.sh
```

The extended run took 98.81 seconds on the recorded machine. Exact package
versions are in `requirements-cpu.txt`; environment information and raw
results are in `cpu_extended/summary.json`.
