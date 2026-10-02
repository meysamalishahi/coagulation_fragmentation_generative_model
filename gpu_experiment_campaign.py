#!/usr/bin/env python3
"""Run and aggregate the paired multi-seed synthetic GPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parent
BENCHMARK = PROJECT / "gpu_synthetic_benchmark.py"
CALIBRATION = PROJECT / "gpu_count_calibration.py"
DIFFUSION = PROJECT / "gpu_diffusion_baseline.py"
CALORIMETER = PROJECT / "gpu_calorimeter_benchmark.py"
SEEDS = (11, 23, 37, 53, 71)


REGIMES = {
    "imbalance_strong": ["--split-alpha", "0.6"],
    "imbalance_weak": ["--split-alpha", "6.0"],
    "interaction_weak": ["--interaction-strength", "0.5"],
    "interaction_strong": ["--interaction-strength", "2.0"],
    "cardinality_low": ["--cardinality-mean", "3.0", "--max-cardinality", "6"],
    "cardinality_high": ["--cardinality-mean", "6.5", "--max-cardinality", "10"],
}


def write_progress(root: Path, stage: str, detail: str) -> None:
    payload = {"stage": stage, "detail": detail, "updated_unix": time.time()}
    (root / "campaign_progress.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], output: Path, root: Path, stage: str) -> None:
    if (output / "summary.json").exists():
        print(f"skip completed: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    write_progress(root, stage, str(output))
    print("run:", " ".join(command), flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_campaign(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    full = [
        "--train-size", "4000", "--validation-size", "800", "--test-size", "1000",
        "--epochs", "30", "--batch-size", "64", "--samples", "1000",
        "--hidden-dim", "128", "--transformer-layers", "3", "--max-components", "20",
    ]
    for seed in SEEDS:
        for model_type in ("coagulation", "birth_only"):
            output = root / "paired" / f"seed_{seed}" / model_type
            command = [
                sys.executable, str(BENCHMARK), "--model-type", model_type,
                "--output-dir", str(output), "--seed", str(seed), *full,
            ]
            run_command(command, output, root, "paired")
        diffusion_output = root / "paired" / f"seed_{seed}" / "cardinality_diffusion"
        diffusion_command = [
            sys.executable, str(DIFFUSION),
            "--output-dir", str(diffusion_output), "--seed", str(seed),
            "--train-size", "4000", "--validation-size", "800", "--test-size", "1000",
            "--epochs", "30", "--batch-size", "64", "--samples", "1000",
            "--hidden-dim", "128", "--transformer-layers", "3", "--diffusion-steps", "100",
        ]
        run_command(diffusion_command, diffusion_output, root, "paired")
        coagulation = root / "paired" / f"seed_{seed}" / "coagulation"
        if not (coagulation / "count_calibration.json").exists():
            write_progress(root, "calibration", str(coagulation))
            with (coagulation / "calibration.log").open("a", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, str(CALIBRATION), str(coagulation)],
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    sweep = [
        "--train-size", "1500", "--validation-size", "300", "--test-size", "500",
        "--epochs", "15", "--batch-size", "64", "--samples", "500",
        "--hidden-dim", "64", "--transformer-layers", "2", "--max-components", "24",
    ]
    for regime, arguments in REGIMES.items():
        for seed in SEEDS:
            output = root / "controlled_sweeps" / regime / f"seed_{seed}"
            command = [
                sys.executable, str(BENCHMARK), "--model-type", "coagulation",
                "--output-dir", str(output), "--seed", str(seed), *sweep, *arguments,
            ]
            run_command(command, output, root, "controlled_sweeps")
    calorimeter_output = root / "real" / "calochallenge_ds1_photons"
    calorimeter_command = [
        sys.executable, str(CALORIMETER), "--output-dir", str(calorimeter_output),
        "--seed", "20261001", "--train-size", "4000", "--validation-size", "800",
        "--test-size", "1000", "--samples", "1000", "--epochs", "25",
        "--batch-size", "32", "--data-components", "32", "--max-components", "48",
        "--hidden-dim", "128", "--transformer-layers", "3",
    ]
    run_command(calorimeter_command, calorimeter_output, root, "real_calorimeter")
    aggregate(root)
    write_progress(root, "complete", "synthetic campaign and real calorimeter benchmark complete")


def flatten_summary(path: Path, group: str, seed: int, model: str, regime: str = "base") -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    sample = summary["sample_metrics"]
    row = {
        "group": group,
        "regime": regime,
        "seed": seed,
        "model": model,
        "objective_type": summary.get("objective_type", "path_nll"),
        "cardinality_tv": sample["cardinality_tv"],
        "component_mass_w1": sample["component_mass_w1"],
        "pair_distance_w1": sample["pair_distance_w1"],
        "summary_mmd": sample["summary_mmd"],
        "summary_energy_distance": sample["summary_energy_distance"],
        "classifier_two_sample_accuracy": sample["classifier_two_sample_accuracy"],
        "empty_sample_rate": sample["empty_sample_rate"],
        "milliseconds_per_sample": sample["milliseconds_per_sample"],
        "mass_residual_max": sample["mass_residual_max"],
    }
    if "test_path" in summary:
        row["test_training_objective"] = summary["test_path"]["loss"]
        row["test_path_nll"] = summary["test_path"]["loss"]
    else:
        row["test_training_objective"] = summary["test_objective"]["loss"]
    if model == "coagulation":
        row["merge_pair_accuracy"] = summary["merge_metrics"]["merge_pair_accuracy"]
        row["merge_target_probability"] = summary["merge_metrics"]["merge_target_probability"]
    return row


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20261001)
    means = np.mean(rng.choice(values, size=(20_000, len(values)), replace=True), axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(root: Path) -> None:
    rows = []
    for seed in SEEDS:
        for model in ("coagulation", "birth_only", "cardinality_diffusion"):
            path = root / "paired" / f"seed_{seed}" / model / "summary.json"
            if path.exists():
                rows.append(flatten_summary(path, "paired", seed, model))
        coagulation_path = root / "paired" / f"seed_{seed}" / "coagulation" / "summary.json"
        calibration_path = root / "paired" / f"seed_{seed}" / "coagulation" / "count_calibration.json"
        if coagulation_path.exists() and calibration_path.exists():
            row = flatten_summary(coagulation_path, "paired", seed, "coagulation_calibrated")
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            row.update(calibration["calibrated_test_metrics"])
            row["leaf_temperature"] = calibration["selected"]["leaf_temperature"]
            row["merge_rate_scale"] = calibration["selected"]["merge_rate_scale"]
            rows.append(row)
    for regime in REGIMES:
        for seed in SEEDS:
            path = root / "controlled_sweeps" / regime / f"seed_{seed}" / "summary.json"
            if path.exists():
                rows.append(flatten_summary(path, "sweep", seed, "coagulation", regime))
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with (root / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "test_training_objective", "test_path_nll", "cardinality_tv", "component_mass_w1", "pair_distance_w1",
        "summary_mmd", "summary_energy_distance", "classifier_two_sample_accuracy",
        "empty_sample_rate", "milliseconds_per_sample", "mass_residual_max",
    ]
    summaries = []
    groups = sorted({(row["group"], row["regime"], row["model"]) for row in rows})
    for group, regime, model in groups:
        selected = [row for row in rows if (row["group"], row["regime"], row["model"]) == (group, regime, model)]
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected if metric in row], dtype=float)
            if len(values):
                mean, low, high = ci95(values)
                summaries.append(
                    {"group": group, "regime": regime, "model": model, "metric": metric,
                     "seeds": len(values), "mean": mean, "ci95_low": low, "ci95_high": high}
                )
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# GPU synthetic campaign", "",
        "All intervals are nonparametric 95% bootstrap intervals over seeds.",
        "Training objectives are model-specific: path NLL for coagulation/birth-only and "
        "epsilon-MSE plus count NLL for diffusion, so they must not be compared across model families.", "",
    ]
    for group, regime, model in groups:
        lines.extend((f"## {group}: {regime} / {model}", "", "| Metric | Mean | 95% CI |", "|---|---:|---:|"))
        for row in summaries:
            if (row["group"], row["regime"], row["model"]) == (group, regime, model):
                lines.append(f"| {row['metric']} | {row['mean']:.5g} | [{row['ci95_low']:.5g}, {row['ci95_high']:.5g}] |")
        lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--root", type=Path, default=PROJECT / "results" / "gpu_campaign")
    args = parser.parse_args()
    if args.command == "run":
        run_campaign(args.root)
    else:
        aggregate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
