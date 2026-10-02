#!/usr/bin/env python3
"""Validation-only sampler calibration for a trained CaloChallenge model.

The temperature and coagulation-rate scale are selected using the held-out
validation split.  The official test file is generated only once, after the
selection has been fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import calorimeter_data as calo
import gpu_calorimeter_benchmark as benchmark
import gpu_model as gm
import gpu_synthetic_benchmark as synthetic_benchmark


PROJECT = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT / "results/gpu_campaign/real/calochallenge_ds1_photons"


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _load_records(summary: dict, calibration_samples: int, test_samples: int):
    dataset = summary["dataset"]
    run = summary["run_config"]
    geometry = calo.load_geometry(Path(dataset["geometry_file"]))
    loader = {
        "geometry": geometry,
        "max_components": int(run["data_components"]),
        "relative_threshold": float(run["relative_threshold"]),
        "log_energy_mean": float(dataset["incident_log_mean"]),
        "log_energy_std": float(dataset["incident_log_std"]),
    }
    validation_count = min(calibration_samples, int(run["validation_size"]))
    test_count = min(test_samples, int(run["test_size"]))
    validation = calo.load_records(
        Path(dataset["train_file"]),
        start=int(run["train_size"]),
        count=validation_count,
        **loader,
    )
    test = calo.load_records(Path(dataset["test_file"]), start=0, count=test_count, **loader)
    return validation, test


def _load_model(run_dir: Path, device: torch.device):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    neural = gm.NeuralConfig(**checkpoint["neural_config"])
    corruption = gm.CorruptionConfig(**checkpoint["corruption_config"])
    model = gm.CoagulationModel(neural).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return summary, model, corruption


def _metric_row(
    reference,
    generated,
    maximum: int,
    seed: int,
    temperature: float,
    rate_scale: float,
    milliseconds: float,
) -> dict:
    metrics, _ = benchmark.evaluate_samples(reference, generated, maximum, seed)
    return {
        "leaf_temperature": temperature,
        "merge_rate_scale": rate_scale,
        "validation_cardinality_tv": metrics["cardinality_tv"],
        "validation_overflow_rate": float(np.mean([state.n > maximum for state in generated])),
        "validation_retained_response_w1": metrics["retained_response_w1"],
        "validation_spatial_spread_w1": metrics["spatial_spread_w1"],
        "validation_summary_mmd": metrics["summary_mmd"],
        "milliseconds_per_sample": milliseconds,
    }


def calibrate(args: argparse.Namespace) -> dict:
    if args.calibration_samples < 1 or args.test_samples < 1:
        raise ValueError("calibration and test sample counts must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summary, model, corruption = _load_model(args.run_dir, device)
    validation, test = _load_records(summary, args.calibration_samples, args.test_samples)
    maximum = int(summary["run_config"]["data_components"])
    reference_validation = [record.state for record in validation]

    rows = []
    for temperature in args.temperatures:
        for rate_scale in args.rate_scales:
            name = f"validation_t{_slug(temperature)}_r{_slug(rate_scale)}.pt"
            generated, milliseconds, _ = benchmark.generate(
                model,
                validation,
                corruption,
                args.seed,
                output,
                args.sample_batch_size,
                args.checkpoint_every,
                leaf_temperature=temperature,
                merge_rate_scale=rate_scale,
                checkpoint_name=name,
            )
            row = _metric_row(
                reference_validation,
                generated,
                maximum,
                args.seed + 1,
                temperature,
                rate_scale,
                milliseconds,
            )
            rows.append(row)
            print(
                f"calibration temperature={temperature:g} rate_scale={rate_scale:g} "
                f"cardinality_tv={row['validation_cardinality_tv']:.4f} "
                f"overflow={row['validation_overflow_rate']:.4f} "
                f"response_w1={row['validation_retained_response_w1']:.4f}",
                flush=True,
            )

    # Cardinality is the diagnosed failure.  Response W1 breaks ties without
    # using any information from the official test file.
    selected = min(
        rows,
        key=lambda row: (
            row["validation_cardinality_tv"],
            row["validation_overflow_rate"],
            row["validation_retained_response_w1"],
            abs(row["leaf_temperature"] - 1.0) + abs(row["merge_rate_scale"] - 1.0),
        ),
    )
    temperature = float(selected["leaf_temperature"])
    rate_scale = float(selected["merge_rate_scale"])
    print(
        f"selected on validation: temperature={temperature:g} rate_scale={rate_scale:g}; "
        "starting the single test evaluation",
        flush=True,
    )
    generated_test, milliseconds, peak_memory = benchmark.generate(
        model,
        test,
        corruption,
        args.seed + 10_000,
        output,
        args.sample_batch_size,
        args.checkpoint_every,
        leaf_temperature=temperature,
        merge_rate_scale=rate_scale,
        checkpoint_name="selected_test.pt",
    )
    test_metrics, cardinality_rows = benchmark.evaluate_samples(
        [record.state for record in test], generated_test, maximum, args.seed + 20_000
    )
    test_metrics.update(
        {
            "overflow_rate": float(np.mean([state.n > maximum for state in generated_test])),
            "milliseconds_per_sample": milliseconds,
            "peak_gpu_memory_mb": peak_memory,
        }
    )
    result = {
        "source_run": str(args.run_dir.resolve()),
        "selection_data": "held-out training-file validation split only",
        "test_policy": "one evaluation after fixing calibration",
        "validation_samples": len(validation),
        "test_samples": len(test),
        "seed": args.seed,
        "grid": rows,
        "selected": {
            "leaf_temperature": temperature,
            "merge_rate_scale": rate_scale,
            "validation_cardinality_tv": selected["validation_cardinality_tv"],
            "validation_retained_response_w1": selected["validation_retained_response_w1"],
        },
        "uncalibrated_test_metrics": summary["sample_metrics"],
        "calibrated_test_metrics": test_metrics,
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "calibration_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    synthetic_benchmark.write_csv(output / "cardinality_selected.csv", cardinality_rows)
    synthetic_benchmark.plot_cardinality(output / "cardinality_selected.png", cardinality_rows)
    report = [
        "# CaloChallenge sampler calibration",
        "",
        f"Selection used {len(validation)} held-out validation events; test was evaluated once on {len(test)} events.",
        "",
        f"Selected leaf temperature: {temperature:g}",
        f"Selected merge-rate scale: {rate_scale:g}",
        "",
        "## Test metrics",
        "",
        "| Metric | Uncalibrated | Calibrated |",
        "|---|---:|---:|",
    ]
    for key, calibrated in test_metrics.items():
        uncalibrated = summary["sample_metrics"].get(key)
        old = "—" if uncalibrated is None else f"{uncalibrated:.6g}"
        report.append(f"| {key} | {old} | {calibrated:.6g} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN / "calibration")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--rate-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20271001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    result = calibrate(parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
