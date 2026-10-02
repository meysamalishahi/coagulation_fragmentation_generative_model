# CPU validation suite

This directory contains a dependency-free implementation of the small
experiments and implementation checks that do not require a GPU.

Run the full suite:

```bash
python3 cpu_validation.py
```

Run all correctness and trainable CPU experiments (about two minutes on the
machine used for the report):

```bash
sh run_all_cpu_experiments.sh
```

Run the quick smoke test and unit tests:

```bash
python3 cpu_validation.py --quick
python3 -m unittest -v test_cpu_validation.py
```

The full run writes:

- `results/cpu_validation/cpu_validation_summary.json`
- `results/cpu_validation/cpu_validation_checks.csv`

## What it validates

1. Split/merge and deletion/birth are inverse maps to numerical precision.
2. Every event and the closed-form flow conserve total mass.
3. The normalized-weight encoder and decoder agree.
4. Finite-difference determinants agree with the split, dequantization, and
   flow Jacobian formulas in the paper.
5. Independent Monte Carlo integrals agree under
   `d(children) = m d(parent, rho, z, u)`, which is the source of the reverse
   `1/m` factor.
6. An enumerated integer-mass process satisfies exact probability-flux
   reversal and the reverse Kolmogorov derivative identity. A tabular
   marked-point-process maximum-likelihood estimate also recovers the
   analytical reverse rates from simulated censored exposures.
7. Uniform birth times have the stated order-statistic means.
8. The uniform-time estimator of a survival integral and the constant-rate
   point-process likelihood agree with analytic answers, and thinning
   produces the correct exponential waiting-time mean.
9. A small synthetic diagnostic checks whether a parent/separation
   parameterization captures pair structure better than independent births.

## What it does not validate

The code does not train the proposed neural model, compare with published
baselines, establish real-data sample quality, or demonstrate scalability.
The synthetic pair experiment is an implementation/representation diagnostic,
not evidence sufficient for an ICML empirical section.

The finite benchmark uses counting measure, so it checks discrete flux and
unordered multiplicities but cannot by itself check the continuous `1/m`
factor. That is why the suite includes the separate continuous-coordinate
Monte Carlo identity.

The extended results and figures are written to `results/cpu_extended/`.
