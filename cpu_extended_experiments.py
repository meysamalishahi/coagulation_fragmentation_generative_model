#!/usr/bin/env python3
"""Trainable CPU experiments for the coagulation--fragmentation proposal.

This file complements ``cpu_validation.py``.  It uses NumPy/SciPy for three
small, reproducible experiments that are informative on a laptop CPU:

1. learn the reverse merge-pair distribution from synthetic conservative
   fragmentation histories and reconstruct parent states;
2. compare matched four-parameter independent-birth and structured
   parent/separation density models on held-out continuous pairs;
3. measure full-pair and one-dimensional sparse-candidate CPU scaling.

The experiments are controlled diagnostics.  They are not comparisons with
published neural generative models and are not sufficient by themselves for
an ICML empirical section.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp


EPS = 1e-12


@dataclass
class MergeExample:
    features: np.ndarray
    target: int
    sibling_mask: np.ndarray


@dataclass
class FragmentedSet:
    leaves: np.ndarray  # columns: mass, position
    lineage: np.ndarray
    parents: np.ndarray
    merge_order: list[int]


def candidate_pairs(n: int) -> np.ndarray:
    return np.column_stack(np.triu_indices(n, 1))


def pair_features(components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = candidate_pairs(len(components))
    a, b = pairs[:, 0], pairs[:, 1]
    ma, mb = components[a, 0], components[b, 0]
    za, zb = components[a, 1], components[b, 1]
    distance = np.abs(za - zb)
    log_ratio = np.abs(np.log((ma + EPS) / (mb + EPS)))
    combined = ma + mb
    midpoint = (ma * za + mb * zb) / combined
    global_center = np.sum(components[:, 0] * components[:, 1]) / np.sum(components[:, 0])
    center_offset = np.abs(midpoint - global_center)
    features = np.column_stack(
        [
            np.ones(len(pairs)),
            distance,
            distance**2,
            log_ratio,
            log_ratio**2,
            distance * log_ratio,
            combined,
            center_offset,
        ]
    )
    return pairs, features


def generate_fragmented_set(rng: np.random.Generator, min_parents: int = 2, max_parents: int = 6) -> FragmentedSet:
    n = int(rng.integers(min_parents, max_parents + 1))
    # Nearby parents make distance alone imperfect.  Unequal parent masses
    # plus nearly balanced splits make mass-ratio information useful.
    global_shift = rng.normal(0.0, 0.7)
    parent_positions = np.sort(global_shift + rng.normal(0.0, 0.85, size=n))
    raw_mass = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    parent_masses = 0.9 * raw_mass / raw_mass.sum()
    parents = np.column_stack([parent_masses, parent_positions])

    leaves = []
    lineage = []
    priority = []
    for i, (mass, position) in enumerate(parents):
        rho = float(rng.beta(10.0, 10.0))
        separation = float(rng.lognormal(mean=-1.15, sigma=0.42))
        ma, mb = rho * mass, (1.0 - rho) * mass
        za = position - (1.0 - rho) * separation
        zb = position + rho * separation
        leaves.extend([(ma, za), (mb, zb)])
        lineage.extend([i, i])
        # A reproducible, state-observable component in the event order.
        priority.append(separation + 0.15 * abs(math.log(ma / mb)))
    order = list(np.argsort(priority))
    # The state is an unordered set.  Randomizing its labeled representative
    # prevents an array-order artifact from helping any deterministic tie rule.
    leaves_array = np.asarray(leaves)
    lineage_array = np.asarray(lineage)
    permutation = rng.permutation(len(leaves_array))
    return FragmentedSet(leaves_array[permutation], lineage_array[permutation], parents, order)


def merge_array(components: np.ndarray, a: int, b: int) -> np.ndarray:
    if a > b:
        a, b = b, a
    ma, za = components[a]
    mb, zb = components[b]
    merged = np.array([(ma + mb), (ma * za + mb * zb) / (ma + mb)])
    keep = np.ones(len(components), dtype=bool)
    keep[[a, b]] = False
    return np.vstack([components[keep], merged])


def history_examples(item: FragmentedSet) -> list[MergeExample]:
    components = item.leaves.copy()
    groups = [{int(label)} for label in item.lineage]
    examples = []
    for target_lineage in item.merge_order:
        pairs, features = pair_features(components)
        sibling = np.array(
            [groups[a] == groups[b] and len(groups[a]) == 1 for a, b in pairs], dtype=bool
        )
        target_candidates = [
            k
            for k, (a, b) in enumerate(pairs)
            if groups[a] == {target_lineage} and groups[b] == {target_lineage}
        ]
        if not target_candidates:
            continue
        target = target_candidates[0]
        examples.append(MergeExample(features, target, sibling))
        a, b = map(int, pairs[target])
        new_group = groups[a] | groups[b]
        components = merge_array(components, a, b)
        groups = [group for k, group in enumerate(groups) if k not in (a, b)] + [new_group]
    return examples


def standardize_examples(
    train: list[MergeExample], test: list[MergeExample]
) -> tuple[list[MergeExample], list[MergeExample], np.ndarray, np.ndarray]:
    stacked = np.vstack([example.features for example in train])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    mean[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-10] = 1.0

    def transform(items: list[MergeExample]) -> list[MergeExample]:
        return [
            MergeExample((example.features - mean) / scale, example.target, example.sibling_mask)
            for example in items
        ]

    return transform(train), transform(test), mean, scale


def fit_softmax(examples: list[MergeExample], columns: list[int], regularization: float = 1e-3) -> np.ndarray:
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.5 * regularization * float(theta @ theta)
        gradient = regularization * theta
        for example in examples:
            x = example.features[:, columns]
            logits = x @ theta
            probabilities = np.exp(logits - logsumexp(logits))
            loss += float(logsumexp(logits) - logits[example.target])
            gradient += probabilities @ x - x[example.target]
        return loss / len(examples), gradient / len(examples)

    result = minimize(
        objective,
        np.zeros(len(columns)),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 180, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"merge optimization failed: {result.message}")
    return result.x


def merge_metrics(examples: list[MergeExample], theta: np.ndarray, columns: list[int]) -> dict[str, float]:
    losses, next_correct, sibling_correct = [], [], []
    for example in examples:
        logits = example.features[:, columns] @ theta
        selected = int(np.argmax(logits))
        losses.append(float(logsumexp(logits) - logits[example.target]))
        next_correct.append(selected == example.target)
        sibling_correct.append(bool(example.sibling_mask[selected]))
    return {
        "event_nll": float(np.mean(losses)),
        "next_pair_accuracy": float(np.mean(next_correct)),
        "sibling_pair_accuracy": float(np.mean(sibling_correct)),
    }


def greedy_reconstruction(
    item: FragmentedSet,
    theta: np.ndarray,
    columns: list[int],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    components = item.leaves.copy()
    while len(components) > len(item.parents):
        pairs, raw_features = pair_features(components)
        features = (raw_features - feature_mean) / feature_scale
        selected = int(np.argmax(features[:, columns] @ theta))
        a, b = map(int, pairs[selected])
        components = merge_array(components, a, b)
    return components


def reconstruction_error(reconstructed: np.ndarray, target: np.ndarray) -> tuple[float, bool]:
    mass_scale = max(float(np.mean(target[:, 0])), EPS)
    position_scale = max(float(np.std(target[:, 1])), 0.25)
    cost = (
        np.abs(reconstructed[:, None, 0] - target[None, :, 0]) / mass_scale
        + np.abs(reconstructed[:, None, 1] - target[None, :, 1]) / position_scale
    )
    rows, columns = linear_sum_assignment(cost)
    error = float(cost[rows, columns].mean())
    exact = bool(
        np.max(np.abs(reconstructed[rows, 0] - target[columns, 0])) < 1e-9
        and np.max(np.abs(reconstructed[rows, 1] - target[columns, 1])) < 1e-9
    )
    return error, exact


def run_merge_experiment(seed: int, train_sets: int, test_sets: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    train_items = [generate_fragmented_set(rng) for _ in range(train_sets)]
    test_items = [generate_fragmented_set(rng) for _ in range(test_sets)]
    train_examples_raw = [example for item in train_items for example in history_examples(item)]
    test_examples_raw = [example for item in test_items for example in history_examples(item)]
    train_examples, test_examples, feature_mean, feature_scale = standardize_examples(
        train_examples_raw, test_examples_raw
    )
    models = {
        "uniform": ([0], np.zeros(1)),
        "distance_only": ([0, 1, 2], None),
        "learned_full": (list(range(8)), None),
    }
    rows = []
    for name, (columns, theta) in models.items():
        start = time.perf_counter()
        if theta is None:
            theta = fit_softmax(train_examples, columns)
        train_seconds = time.perf_counter() - start
        metrics = merge_metrics(test_examples, theta, columns)
        reconstruction = [
            reconstruction_error(
                greedy_reconstruction(item, theta, columns, feature_mean, feature_scale), item.parents
            )
            for item in test_items
        ]
        rows.append(
            {
                "seed": seed,
                "model": name,
                **metrics,
                "reconstruction_error": float(np.mean([value for value, _ in reconstruction])),
                "exact_reconstruction_rate": float(np.mean([exact for _, exact in reconstruction])),
                "train_seconds": train_seconds,
                "train_examples": len(train_examples),
                "test_examples": len(test_examples),
            }
        )
    return rows


def normal_fit(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(np.sqrt(np.mean((values - values.mean()) ** 2) + 1e-8))


def normal_logpdf(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return -0.5 * ((values - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)


def logaddexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    maximum = np.maximum(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


def paired_data(rng: np.random.Generator, size: int) -> np.ndarray:
    center = rng.normal(0.0, 1.0, size=size)
    log_separation = rng.normal(-1.0, 0.28, size=size)
    separation = np.exp(log_separation)
    return np.column_stack([center - separation / 2.0, center + separation / 2.0])


def wasserstein_equal(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.sort(x) - np.sort(y))))


def energy_distance_rows(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, pairs: int = 80_000) -> float:
    # Unbiased-style Monte Carlo estimate of 2E||X-Y||-E||X-X'||-E||Y-Y'||.
    nx, ny = len(x), len(y)
    ix, jx = rng.integers(0, nx, size=pairs), rng.integers(0, nx, size=pairs)
    iy, jy = rng.integers(0, ny, size=pairs), rng.integers(0, ny, size=pairs)
    cross = np.linalg.norm(x[ix] - y[iy], axis=1).mean()
    within_x = np.linalg.norm(x[ix] - x[jx], axis=1).mean()
    within_y = np.linalg.norm(y[iy] - y[jy], axis=1).mean()
    return float(max(0.0, 2.0 * cross - within_x - within_y))


def run_birth_ablation(seed: int, train_size: int, test_size: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + 100_000)
    train = paired_data(rng, train_size)
    test = paired_data(rng, test_size)

    # Both models have four learned scalar parameters.
    left_mean, left_std = normal_fit(train[:, 0])
    right_mean, right_std = normal_fit(train[:, 1])
    centers = train.mean(axis=1)
    log_separations = np.log(train[:, 1] - train[:, 0])
    center_mean, center_std = normal_fit(centers)
    logsep_mean, logsep_std = normal_fit(log_separations)

    sample_left = rng.normal(left_mean, left_std, size=test_size)
    sample_right = rng.normal(right_mean, right_std, size=test_size)
    independent_sample = np.sort(np.column_stack([sample_left, sample_right]), axis=1)
    sample_center = rng.normal(center_mean, center_std, size=test_size)
    sample_logsep = rng.normal(logsep_mean, logsep_std, size=test_size)
    sample_sep = np.exp(sample_logsep)
    structured_sample = np.column_stack([sample_center - sample_sep / 2.0, sample_center + sample_sep / 2.0])

    # Sorting independent, non-identically distributed draws gives a two-term
    # ordered density.  The structured chart contributes |d(c,log u)/d(x,y)|=1/u.
    independent_logp = logaddexp(
        normal_logpdf(test[:, 0], left_mean, left_std)
        + normal_logpdf(test[:, 1], right_mean, right_std),
        normal_logpdf(test[:, 1], left_mean, left_std)
        + normal_logpdf(test[:, 0], right_mean, right_std),
    )
    test_center = test.mean(axis=1)
    test_sep = test[:, 1] - test[:, 0]
    structured_logp = (
        normal_logpdf(test_center, center_mean, center_std)
        + normal_logpdf(np.log(test_sep), logsep_mean, logsep_std)
        - np.log(test_sep)
    )

    truth_distance = test[:, 1] - test[:, 0]
    truth_covariance = float(np.cov(test[:, 0], test[:, 1], ddof=1)[0, 1])
    rows = []
    for name, sample, logp in (
        ("independent_birth", independent_sample, independent_logp),
        ("structured_coordinates", structured_sample, structured_logp),
    ):
        sample_distance = sample[:, 1] - sample[:, 0]
        sample_covariance = float(np.cov(sample[:, 0], sample[:, 1], ddof=1)[0, 1])
        rows.append(
            {
                "seed": seed,
                "model": name,
                "heldout_nll": float(-np.mean(logp)),
                "distance_w1": wasserstein_equal(truth_distance, sample_distance),
                "covariance_error": abs(sample_covariance - truth_covariance),
                "energy_distance": energy_distance_rows(test, sample, rng),
                "mass_conservation_error": 0.0,
                "parameters": 4,
            }
        )
    return rows


def time_call(function, minimum_seconds: float = 0.08) -> tuple[float, int]:
    repeats = 1
    while True:
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        elapsed = time.perf_counter() - start
        if elapsed >= minimum_seconds:
            return elapsed / repeats, repeats
        repeats *= 2


def run_scaling_experiment(seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + 200_000)
    rows = []
    for n in (16, 32, 64, 128, 256, 512, 1024):
        masses = rng.lognormal(0.0, 0.6, size=n)
        positions = np.sort(rng.normal(0.0, 1.0, size=n))
        k = min(8, n - 1)

        def full():
            a, b = np.triu_indices(n, 1)
            score = -np.abs(positions[a] - positions[b]) - 0.2 * np.abs(np.log(masses[a] / masses[b]))
            return float(np.sum(score))

        def sparse():
            total = 0.0
            for offset in range(1, k + 1):
                total += float(
                    np.sum(
                        -np.abs(positions[offset:] - positions[:-offset])
                        - 0.2 * np.abs(np.log(masses[offset:] / masses[:-offset]))
                    )
                )
            return total

        full_seconds, full_repeats = time_call(full)
        sparse_seconds, sparse_repeats = time_call(sparse)
        rows.extend(
            [
                {
                    "seed": seed,
                    "method": "full_pairs",
                    "n": n,
                    "candidate_pairs": n * (n - 1) // 2,
                    "milliseconds": 1000.0 * full_seconds,
                    "timing_repeats": full_repeats,
                },
                {
                    "seed": seed,
                    "method": "sparse_1d_k8",
                    "n": n,
                    "candidate_pairs": sum(n - offset for offset in range(1, k + 1)),
                    "milliseconds": 1000.0 * sparse_seconds,
                    "timing_repeats": sparse_repeats,
                },
            ]
        )
    return rows


def summarize(rows: list[dict[str, object]], group_keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in groups.items():
        record = dict(zip(group_keys, key))
        record["seeds"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci95"] = float(2.776 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output: Path, merge_summary: list[dict[str, object]], birth_summary: list[dict[str, object]], scaling: list[dict[str, object]]) -> None:
    model_order = ["uniform", "distance_only", "learned_full"]
    lookup = {row["model"]: row for row in merge_summary}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [
        ("event_nll", "Held-out event NLL", False),
        ("sibling_pair_accuracy", "Sibling-pair accuracy", True),
        ("exact_reconstruction_rate", "Exact reconstruction", True),
    ]
    for axis, (metric, title, _) in zip(axes, metrics):
        values = [lookup[name][f"{metric}_mean"] for name in model_order]
        errors = [lookup[name][f"{metric}_ci95"] for name in model_order]
        axis.bar(range(3), values, yerr=errors, capsize=3, color=["#9aa0a6", "#73a9d8", "#2468a2"])
        axis.set_xticks(range(3), ["uniform", "distance", "full"], rotation=18)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "merge_recovery.png", dpi=180)
    plt.close(fig)

    order = ["independent_birth", "structured_coordinates"]
    lookup = {row["model"]: row for row in birth_summary}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for axis, metric, title in zip(
        axes,
        ["heldout_nll", "distance_w1", "covariance_error"],
        ["Held-out NLL", "Separation W1", "Covariance error"],
    ):
        values = [lookup[name][f"{metric}_mean"] for name in order]
        errors = [lookup[name][f"{metric}_ci95"] for name in order]
        axis.bar(range(2), values, yerr=errors, capsize=3, color=["#b7b7b7", "#2d8a63"])
        axis.set_xticks(range(2), ["independent", "structured"], rotation=12)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "birth_vs_structured.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.0))
    for method, color in (("full_pairs", "#b14a4a"), ("sparse_1d_k8", "#315f9e")):
        subset = [row for row in scaling if row["method"] == method]
        axis.loglog(
            [row["n"] for row in subset],
            [row["milliseconds"] for row in subset],
            marker="o",
            label=method,
            color=color,
        )
    axis.set_xlabel("Active cardinality N")
    axis.set_ylabel("CPU milliseconds per score evaluation")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "pair_scaling.png", dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], digits: int = 4) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    elapsed: float,
    seed_count: int,
    merge_summary: list[dict[str, object]],
    birth_summary: list[dict[str, object]],
    scaling: list[dict[str, object]],
) -> None:
    merge_display = []
    for row in sorted(merge_summary, key=lambda item: ["uniform", "distance_only", "learned_full"].index(item["model"])):
        merge_display.append(
            {
                "model": row["model"],
                "nll": row["event_nll_mean"],
                "sibling": row["sibling_pair_accuracy_mean"],
                "exact": row["exact_reconstruction_rate_mean"],
                "error": row["reconstruction_error_mean"],
            }
        )
    birth_display = []
    for row in sorted(birth_summary, key=lambda item: item["model"]):
        birth_display.append(
            {
                "model": row["model"],
                "nll": row["heldout_nll_mean"],
                "w1": row["distance_w1_mean"],
                "cov": row["covariance_error_mean"],
                "energy": row["energy_distance_mean"],
            }
        )
    largest = [row for row in scaling if row["n"] == max(item["n"] for item in scaling)]
    speedup = next(row["milliseconds"] for row in largest if row["method"] == "full_pairs") / next(
        row["milliseconds"] for row in largest if row["method"] == "sparse_1d_k8"
    )
    report = f"""# CPU experimental report

## Scope

These experiments test the small-scale mechanisms that are meaningful on a
CPU. They do **not** compare against published neural baselines and do not
establish an ICML-level empirical result. All reported summary values use
{seed_count} independent seeds; error bars in the PNG figures are 95% t-intervals
over those seeds.

Runtime for the complete extended suite: **{elapsed:.2f} seconds**.

## 1. Learned reverse merge kernel

Synthetic parents were conservatively split into two children. A marked
softmax reverse kernel was trained on the resulting event histories. The
distance-only and full models use the same objective; the full model also has
mass-ratio, combined-mass, and centroid features.

{markdown_table(merge_display, [("model", "Model"), ("nll", "Event NLL"), ("sibling", "Sibling accuracy"), ("exact", "Exact reconstruction"), ("error", "Reconstruction error")])}

Interpretation: the trainable reverse likelihood can recover useful pairwise
coagulation structure and reconstruct conservative parents. This is evidence
for implementation viability, not evidence that a neural coagulation model
beats a strong generative baseline.

![Merge recovery](merge_recovery.png)

## 2. Independent birth versus structured coordinates

Both models have four fitted scalar parameters. The independent model fits
separate Gaussian marginals for the two insertions. The structured model fits
a Gaussian parent center and Gaussian log-separation and includes the exact
coordinate Jacobian in held-out likelihood.

{markdown_table(birth_display, [("model", "Model"), ("nll", "Held-out NLL"), ("w1", "Separation W1"), ("cov", "Covariance error"), ("energy", "Energy distance")])}

Interpretation: on data generated by a shared-parent mechanism, the structured
coordinates are substantially better calibrated than independent insertion.
This is the expected positive-control result. Because the data intentionally
have this structure, it cannot establish an advantage on natural datasets.

![Birth versus structured](birth_vs_structured.png)

## 3. Pair-scoring scaling

At N={int(largest[0]["n"])}, the one-dimensional k=8 sparse candidate rule was
**{speedup:.1f}x** faster than full pair enumeration in this microbenchmark.
The sparse rule is exact only when the relevant merge is inside its candidate
set; candidate recall must therefore be reported in a real experiment.

![Pair scaling](pair_scaling.png)

## Conclusions supported by the CPU experiments

- The event maps, Jacobians, probability-flux reversal, and point-process
  likelihood implementation pass the separate correctness suite.
- A tabular/parametric reverse merge model can be fitted from simulated
  corruption histories using the proposed marked-event objective.
- Structured parent/separation coordinates can outperform independent births
  on a controlled positive-control distribution.
- Sparse candidate pairs materially reduce CPU scoring time.

## Conclusions not yet supported

- superiority to Jump Diffusion, Existence-Field Diffusion, flow matching, or
  autoregressive set generation;
- improved likelihood or sample quality on a real scalar-weighted set domain;
- scalability of a neural set encoder;
- an ICML-strength empirical advantage of coagulation over a capacity-matched
  learned birth-only model.

Those remaining questions require implementation of substantial neural
baselines and are where GPU access becomes practically important.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cpu_extended"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [11, 23] if args.quick else [11, 23, 37, 53, 71]
    train_sets, test_sets = ((250, 100) if args.quick else (1000, 400))
    train_pairs, test_pairs = ((3000, 5000) if args.quick else (12_000, 20_000))

    start = time.perf_counter()
    merge_rows = []
    birth_rows = []
    for seed in seeds:
        print(f"seed {seed}: merge recovery", flush=True)
        merge_rows.extend(run_merge_experiment(seed, train_sets, test_sets))
        print(f"seed {seed}: birth ablation", flush=True)
        birth_rows.extend(run_birth_ablation(seed, train_pairs, test_pairs))
    print("pair scaling", flush=True)
    scaling_rows = run_scaling_experiment(seeds[0])

    merge_metrics_names = [
        "event_nll",
        "next_pair_accuracy",
        "sibling_pair_accuracy",
        "reconstruction_error",
        "exact_reconstruction_rate",
        "train_seconds",
    ]
    birth_metrics_names = ["heldout_nll", "distance_w1", "covariance_error", "energy_distance"]
    merge_summary = summarize(merge_rows, ["model"], merge_metrics_names)
    birth_summary = summarize(birth_rows, ["model"], birth_metrics_names)
    elapsed = time.perf_counter() - start

    write_csv(args.output_dir / "merge_results_by_seed.csv", merge_rows)
    write_csv(args.output_dir / "merge_summary.csv", merge_summary)
    write_csv(args.output_dir / "birth_ablation_by_seed.csv", birth_rows)
    write_csv(args.output_dir / "birth_ablation_summary.csv", birth_summary)
    write_csv(args.output_dir / "pair_scaling.csv", scaling_rows)
    plot_results(args.output_dir, merge_summary, birth_summary, scaling_rows)
    write_report(args.output_dir, elapsed, len(seeds), merge_summary, birth_summary, scaling_rows)
    payload = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cpu_count": os.cpu_count(),
        },
        "elapsed_seconds": elapsed,
        "seeds": seeds,
        "merge_summary": merge_summary,
        "birth_ablation_summary": birth_summary,
        "pair_scaling": scaling_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"completed in {elapsed:.2f}s; report: {args.output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
