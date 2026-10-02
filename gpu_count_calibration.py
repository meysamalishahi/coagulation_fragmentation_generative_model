#!/usr/bin/env python3
"""Validation-only cardinality calibration for a trained coagulation model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch

import gpu_model as gm
import gpu_synthetic_benchmark as bench


def load_run(run_dir: Path, device: torch.device):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    synthetic = bench.SyntheticConfig(**summary["synthetic_config"])
    corruption = gm.CorruptionConfig(**summary["corruption_config"])
    run = bench.RunConfig(**summary["run_config"])
    neural = gm.NeuralConfig(**checkpoint["neural_config"])
    model = bench.build_model(neural, run.model_type).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, synthetic, corruption, run


def calibrate(
    run_dir: Path,
    device: torch.device,
    calibration_samples: int,
    temperatures: list[float],
    rate_scales: list[float],
) -> dict:
    model, synthetic, corruption, run = load_run(run_dir, device)
    if run.model_type != "coagulation":
        raise ValueError("merge-rate calibration applies only to the coagulation model")
    validation = bench.generate_dataset(run.validation_size, run.seed + 2, synthetic)
    validation_states = [example.state for example in validation[:calibration_samples]]
    rows = []
    for temperature in temperatures:
        for rate_scale in rate_scales:
            generated, milliseconds, _ = bench.sample_model(
                model,
                len(validation_states),
                corruption,
                run.seed + 60_000 + len(rows),
                leaf_temperature=temperature,
                merge_rate_scale=rate_scale,
            )
            tv, _ = bench.cardinality_tv(validation_states, generated, synthetic.max_cardinality)
            rows.append(
                {
                    "leaf_temperature": temperature,
                    "merge_rate_scale": rate_scale,
                    "validation_cardinality_tv": tv,
                    "milliseconds_per_sample": milliseconds,
                }
            )
            print(
                f"temperature={temperature:.3f} rate_scale={rate_scale:.3f} val_tv={tv:.4f}",
                flush=True,
            )
    best = min(
        rows,
        key=lambda row: (
            row["validation_cardinality_tv"],
            abs(row["leaf_temperature"] - 1.0) + abs(row["merge_rate_scale"] - 1.0),
        ),
    )
    test_examples = bench.generate_dataset(run.test_size, run.seed + 3, synthetic)
    generated, milliseconds, peak_memory = bench.sample_model(
        model,
        run.samples,
        corruption,
        run.seed + 70_000,
        leaf_temperature=best["leaf_temperature"],
        merge_rate_scale=best["merge_rate_scale"],
    )
    reference = [example.state for example in test_examples[: run.samples]]
    metrics, cardinality_rows = bench.evaluate_samples(
        reference, generated, synthetic.max_cardinality, run.seed + 80_000
    )
    metrics["milliseconds_per_sample"] = milliseconds
    metrics["peak_gpu_memory_mb"] = peak_memory
    result = {
        "run_dir": str(run_dir),
        "selection_data": "validation_only",
        "calibration_samples": len(validation_states),
        "grid": rows,
        "selected": {
            "leaf_temperature": best["leaf_temperature"],
            "merge_rate_scale": best["merge_rate_scale"],
        },
        "uncalibrated_test_metrics": json.loads((run_dir / "summary.json").read_text())["sample_metrics"],
        "calibrated_test_metrics": metrics,
        "synthetic_config": asdict(synthetic),
        "run_config": asdict(run),
    }
    (run_dir / "count_calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "count_calibration_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bench.write_csv(run_dir / "cardinality_calibrated.csv", cardinality_rows)
    bench.plot_cardinality(run_dir / "cardinality_calibrated.png", cardinality_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=120)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--rate-scales", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = calibrate(
        args.run_dir,
        torch.device(args.device),
        args.calibration_samples,
        args.temperatures,
        args.rate_scales,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

