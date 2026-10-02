#!/usr/bin/env python3
"""Run the pinned ViT-CFM Dataset-1 evaluation and summarize paper metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import torch

import calorimeter_metrics


PROJECT = Path(__file__).resolve().parent
REFERENCE_ROOT = PROJECT / "external/vit4hep_reference"
DEFAULT_SAMPLE = PROJECT / "results/gpu_official/fair_comparison/generated_ds1_photons.hdf5"
DEFAULT_REFERENCE = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_2.hdf5"
DEFAULT_XML = PROJECT / "external/calochallenge-homepage/code/binning_dataset_1_photons.xml"
DEFAULT_OUTPUT = PROJECT / "results/gpu_official/fair_comparison/paper_evaluation"


def evaluation_config(args: argparse.Namespace, mode: str, run_index: int):
    evaluation = SimpleNamespace(
        eval_dataset="1-photons",
        eval_mode=mode,
        eval_cut=args.cut_mev,
        eval_hdf5_file=str(args.reference_file),
        eval_p_label="",
        eval_cls_resnet_layers=18,
        eval_cls_resnet_lr=2e-5,
        eval_cls_resnet_n_epochs=48,
        eval_cls_n_layer=2,
        eval_cls_n_hidden=2048,
        eval_cls_dropout=0.0,
        eval_cls_lr=2e-4,
        eval_cls_batch_size=1000,
        eval_cls_n_epochs=100,
        eval_cls_save_mem=True,
    )
    return SimpleNamespace(
        run_dir=str(args.output_dir),
        run_idx=run_index,
        evaluation=evaluation,
        data=SimpleNamespace(xml_filename=str(args.geometry)),
    )


def parse_classifier(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"Final result of classifier test \(AUC / JSD\):\s*\n([0-9.]+) / ([0-9.]+)", text)
    if not matches:
        raise ValueError(f"could not parse classifier metrics from {path}")
    auc, jsd = matches[-1]
    return {"auc": float(auc), "jsd": float(jsd)}


def parse_physics_distances(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    values = re.search(
        r"FPD \(x10\^3\): ([0-9.eE+-]+) ± ([0-9.eE+-]+).*"
        r"KPD \(x10\^3\): ([0-9.eE+-]+) ± ([0-9.eE+-]+)",
        text,
        flags=re.DOTALL,
    )
    if values is None:
        raise ValueError(f"could not parse physics distances from {path}")
    return {
        "fpd_x1e3": float(values.group(1)),
        "fpd_error_x1e3": float(values.group(2)),
        "kpd_x1e3": float(values.group(3)),
        "kpd_error_x1e3": float(values.group(4)),
    }


def run(args: argparse.Namespace) -> dict:
    if not args.sample_file.exists() or not args.reference_file.exists():
        raise FileNotFoundError("generated and reference HDF5 files are required")
    with h5py.File(args.sample_file, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("generated HDF5 is not marked complete")
        samples = np.asarray(handle["showers"], dtype=np.float32)
        incidents = np.asarray(handle["incident_energies"], dtype=np.float32)
    if len(samples) != 121000 and not args.allow_nonreference_count:
        raise ValueError(
            "paper-exact evaluation requires 121,000 samples from the executable reference schedule"
        )
    if not (REFERENCE_ROOT / "SOURCE.md").exists():
        raise FileNotFoundError("pinned ViT-CFM evaluation source is missing")
    sys.path.insert(0, str(REFERENCE_ROOT))
    import experiments.calo_utils.ugr_evaluation.evaluate as reference_evaluation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    classifier_cfg = evaluation_config(args, "all-cls", 0)
    reference_evaluation.run_from_py(samples.copy(), incidents.copy(), classifier_cfg)

    physics_distances = None
    if not args.skip_physics_distances:
        cut_samples = samples.copy()
        cut_samples[cut_samples < args.cut_mev] = 0.0
        with h5py.File(args.reference_file, "r") as handle:
            reference_showers = np.asarray(handle["showers"][: len(samples)], dtype=np.float32)
            reference_incidents = np.asarray(
                handle["incident_energies"][: len(samples)], dtype=np.float32
            )
        reference_showers[reference_showers < args.cut_mev] = 0.0
        generated_hlf = reference_evaluation.HLF.HighLevelFeatures(
            "photon", filename=str(args.geometry)
        )
        reference_hlf = reference_evaluation.HLF.HighLevelFeatures(
            "photon", filename=str(args.geometry)
        )
        generated_hlf.CalculateFeatures(cut_samples)
        generated_hlf.Einc = incidents
        reference_hlf.CalculateFeatures(reference_showers)
        reference_hlf.Einc = reference_incidents
        generated_features = reference_evaluation.prepare_high_data_for_classifier(
            cut_samples, incidents, generated_hlf, 0.0, cut=args.cut_mev
        )[:, :-1]
        reference_features = reference_evaluation.prepare_high_data_for_classifier(
            reference_showers,
            reference_incidents,
            reference_hlf,
            1.0,
            cut=args.cut_mev,
        )[:, :-1]
        fpd_value, fpd_error = calorimeter_metrics.fpd(
            reference_features, generated_features, min_samples=10_000
        )
        kpd_value, kpd_error = calorimeter_metrics.kpd(
            reference_features, generated_features, batch_size=10_000
        )
        physics_distances = {
            "fpd_x1e3": 1000.0 * fpd_value,
            "fpd_error_x1e3": 1000.0 * fpd_error,
            "kpd_x1e3": 1000.0 * kpd_value,
            "kpd_error_x1e3": 1000.0 * kpd_error,
            "implementation": "JetNet 0.2.5 formulas pinned in calorimeter_metrics.py",
        }

    classifier_dir = args.output_dir / "eval_0"
    result = {
        "protocol": {
            "reference_source": str(REFERENCE_ROOT / "SOURCE.md"),
            "sample_count": len(samples),
            "cut_mev": args.cut_mev,
            "classifier_hidden_layers": 3,
            "classifier_hidden_width": 2048,
            "classifier_epochs": 100,
            "classifier_batch_size": 1000,
            "classifier_learning_rate": 2e-4,
            "split": [0.6, 0.2, 0.2],
            "seed": args.seed,
        },
        "low_level": parse_classifier(
            classifier_dir / "classifier_all-cls_cls-low_1-photons.txt"
        ),
        "high_level": parse_classifier(
            classifier_dir / "classifier_all-cls_cls-high_1-photons.txt"
        ),
        "vit_cfm_reported_auc": {"low_level": 0.518, "high_level": 0.504},
    }
    if physics_distances is not None:
        result["physics_distances"] = physics_distances
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Paper-exact Dataset-1 photon evaluation",
        "",
        "| Metric | Condensation model | ViT-CFM reported |",
        "|---|---:|---:|",
        f"| Low-level classifier AUC | {result['low_level']['auc']:.4f} | 0.518 |",
        f"| High-level classifier AUC | {result['high_level']['auc']:.4f} | 0.504 |",
    ]
    if "physics_distances" in result:
        distances = result["physics_distances"]
        report.extend(
            (
                f"| FPD x10^3 | {distances['fpd_x1e3']:.4f} ± {distances['fpd_error_x1e3']:.4f} | — |",
                f"| KPD x10^3 | {distances['kpd_x1e3']:.4f} ± {distances['kpd_error_x1e3']:.4f} | — |",
            )
        )
    report.extend(
        (
            "",
            "Lower classifier AUC is better; 0.5 is indistinguishable from the reference sample.",
            "The public CaloChallenge evaluator is reported separately because its cell cut and split differ.",
        )
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cut-mev", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--skip-physics-distances", action="store_true")
    parser.add_argument("--allow-nonreference-count", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
