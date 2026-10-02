#!/usr/bin/env python3
"""Detached end-to-end supervisor for the official calorimeter comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np


PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results/gpu_official"
SHAPE = RESULTS / "shape_full"
COMPARISON = RESULTS / "fair_comparison"
SAMPLE = COMPARISON / "generated_ds1_photons.hdf5"
REFERENCE = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_2.hdf5"
PUBLIC_CODE = PROJECT / "external/calochallenge-homepage/code"
STATUS = COMPARISON / "supervisor_status.json"
FAILURE = COMPARISON / "supervisor_failure.json"
FINAL_SUMMARY = COMPARISON / "final_summary.json"
FINAL_REPORT = COMPARISON / "FINAL_REPORT.md"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_status(stage: str, message: str, **extra) -> None:
    value = {
        "stage": stage,
        "message": message,
        "updated_unix": time.time(),
        **extra,
    }
    write_json(STATUS, value)
    print(f"supervisor stage={stage} message={message}", flush=True)


def run_command(command: list[str], *, cwd: Path = PROJECT) -> None:
    print("running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def service_active(name: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", name],
        check=False,
    )
    return result.returncode == 0


def validate_shape_result() -> dict:
    summary_path = SHAPE / "summary.json"
    model_path = SHAPE / "model.pt"
    if not summary_path.exists() or not model_path.exists():
        raise FileNotFoundError("shape result is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("checkpoint_selection") != "final_step":
        raise ValueError("shape result did not use final-step checkpoint selection")
    if summary.get("history", [{}])[-1].get("step") != 800000:
        raise ValueError("shape training did not reach 800,000 optimizer steps")
    parameters = int(summary["parameter_count"])
    if abs(parameters - 25_982_005) / 25_982_005 >= 0.01:
        raise ValueError("shape model is outside the one-percent parameter budget")
    validation = float(summary["final_validation"]["loss"])
    if not math.isfinite(validation):
        raise ValueError("shape validation loss is non-finite")
    return summary


def wait_for_shape() -> dict:
    restart_count = 0
    last_heartbeat = 0.0
    while True:
        if (SHAPE / "summary.json").exists():
            result = validate_shape_result()
            update_status("shape_complete", "full 800,000-step shape result validated")
            return result
        if not service_active("calochallenge-official-shape"):
            if restart_count >= 3:
                raise RuntimeError("shape service stopped repeatedly without a complete result")
            restart_count += 1
            update_status(
                "shape_restart",
                "shape service stopped; starting or resuming from its latest checkpoint",
                restart_count=restart_count,
            )
            run_command(["sh", str(PROJECT / "run_gpu_calorimeter_official_shape_detached.sh")])
            time.sleep(10)
            if not service_active("calochallenge-official-shape"):
                continue
        if time.time() - last_heartbeat >= 600:
            latest_step = None
            log_path = SHAPE / "run.log"
            if log_path.exists():
                try:
                    matches = re.findall(
                        r"step=(\d+)", log_path.read_text(encoding="utf-8", errors="replace")
                    )
                    latest_step = int(matches[-1]) if matches else None
                except Exception:
                    latest_step = None
            update_status(
                "shape_training",
                "waiting for detached full shape training",
                latest_checkpoint_step=latest_step,
                restart_count=restart_count,
            )
            last_heartbeat = time.time()
        time.sleep(60)


def validate_generation() -> dict:
    summary_path = COMPARISON / "generation_summary.json"
    if not summary_path.exists() or not SAMPLE.exists():
        raise FileNotFoundError("official generation is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["samples"]) != 121000:
        raise ValueError("reference-executable generation must contain 121,000 showers")
    if abs(int(summary["combined_parameter_count"]) - 27_938_230) / 27_938_230 >= 0.01:
        raise ValueError("combined generator is outside the one-percent parameter budget")
    with h5py.File(SAMPLE, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("generated HDF5 is not marked complete")
        if handle["showers"].shape != (121000, 368):
            raise ValueError("generated HDF5 has the wrong shower shape")
        for start in range(0, 121000, 4096):
            values = np.asarray(handle["showers"][start : start + 4096])
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError("generated HDF5 contains invalid voxel energies")
    return summary


def generate_showers() -> dict:
    if (COMPARISON / "generation_summary.json").exists():
        result = validate_generation()
        update_status("generation_complete", "existing official generation validated")
        return result
    for attempt in range(1, 4):
        update_status(
            "generation",
            "generating 121,000 independent showers",
            attempt=attempt,
        )
        try:
            run_command(
                [
                    sys.executable,
                    str(PROJECT / "gpu_calorimeter_generate_official.py"),
                    "--output-file",
                    str(SAMPLE),
                    "--energy-checkpoint",
                    str(RESULTS / "energy_full/model.pt"),
                    "--shape-checkpoint",
                    str(SHAPE / "model.pt"),
                    "--samples",
                    "121000",
                    "--batch-size",
                    "100",
                    "--checkpoint-every",
                    "1000",
                    "--seed",
                    "20261004",
                    "--device",
                    "cuda",
                ]
            )
            result = validate_generation()
            update_status("generation_complete", "official HDF5 passed integrity checks")
            return result
        except Exception as error:
            update_status("generation_retry", str(error), attempt=attempt)
            if attempt == 3:
                raise
            time.sleep(60)
    raise AssertionError("unreachable")


def paper_evaluation() -> dict:
    output = COMPARISON / "paper_evaluation"
    summary_path = output / "summary.json"
    if not summary_path.exists():
        for attempt in range(1, 4):
            update_status(
                "paper_evaluation",
                "running pinned ViT-CFM AUC, FPD, and KPD protocol",
                attempt=attempt,
            )
            try:
                run_command(
                    [
                        sys.executable,
                        str(PROJECT / "gpu_calorimeter_evaluate_paper.py"),
                        "--sample-file",
                        str(SAMPLE),
                        "--reference-file",
                        str(REFERENCE),
                        "--geometry",
                        str(PUBLIC_CODE / "binning_dataset_1_photons.xml"),
                        "--output-dir",
                        str(output),
                        "--cut-mev",
                        "0.015",
                        "--seed",
                        "20261005",
                    ]
                )
                break
            except Exception as error:
                update_status("paper_evaluation_retry", str(error), attempt=attempt)
                if attempt == 3:
                    raise
                time.sleep(60)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    for level in ("low_level", "high_level"):
        auc = float(result[level]["auc"])
        if not 0.0 <= auc <= 1.0:
            raise ValueError(f"invalid {level} classifier AUC")
    update_status("paper_evaluation_complete", "paper-exact metrics validated")
    return result


def parse_public_classifier(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"Final result of classifier test \(AUC / JSD\):\s*\n([0-9.]+) / ([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError(f"could not parse public evaluator output {path}")
    auc, jsd = matches[-1]
    return {"auc": float(auc), "jsd": float(jsd)}


def public_evaluation() -> dict:
    result = {}
    for mode in ("cls-low", "cls-high"):
        output = COMPARISON / "public_evaluation" / mode
        metric_file = output / f"classifier_{mode}_1-photons.txt"
        if not metric_file.exists():
            output.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, 4):
                update_status(
                    "public_evaluation",
                    f"running public CaloChallenge {mode} classifier",
                    attempt=attempt,
                )
                try:
                    run_command(
                        [
                            sys.executable,
                            str(PUBLIC_CODE / "evaluate.py"),
                            "--input_file",
                            str(SAMPLE),
                            "--reference_file",
                            str(REFERENCE),
                            "--mode",
                            mode,
                            "--dataset",
                            "1-photons",
                            "--output_dir",
                            str(output),
                            "--cls_n_layer",
                            "2",
                            "--cls_n_hidden",
                            "2048",
                            "--cls_batch_size",
                            "1000",
                            "--cls_n_epochs",
                            "100",
                            "--cls_lr",
                            "0.0002",
                            "--save_mem",
                        ],
                        cwd=PUBLIC_CODE,
                    )
                    break
                except Exception as error:
                    update_status("public_evaluation_retry", str(error), mode=mode, attempt=attempt)
                    if attempt == 3:
                        raise
                    time.sleep(60)
        result[mode.replace("-", "_")] = parse_public_classifier(metric_file)
    write_json(COMPARISON / "public_evaluation/summary.json", result)
    update_status("public_evaluation_complete", "public evaluator metrics validated")
    return result


def assemble_final_report(
    shape_result: dict,
    generation: dict,
    paper: dict,
    public: dict,
) -> dict:
    energy = json.loads((RESULTS / "energy_full/summary.json").read_text(encoding="utf-8"))
    result = {
        "status": "complete",
        "model": {
            "energy_parameters": energy["parameter_count"],
            "shape_parameters": shape_result["parameter_count"],
            "combined_parameters": generation["combined_parameter_count"],
            "vit_cfm_parameters": 27_938_230,
            "difference_fraction": generation["parameter_difference_fraction"],
        },
        "training": {
            "energy_iterations": energy["run_config"]["iterations"],
            "shape_iterations": shape_result["run_config"]["iterations"],
            "energy_seconds": energy["training_seconds"],
            "shape_seconds": shape_result["training_seconds"],
            "shape_final_validation": shape_result["final_validation"],
        },
        "generation": generation,
        "paper_protocol": paper,
        "public_protocol": public,
    }
    write_json(FINAL_SUMMARY, result)
    distances = paper.get("physics_distances", {})
    lines = [
        "# Final fair Dataset-1 photon comparison",
        "",
        "The complete energy-plus-shape generator was trained on the official first photon file and",
        "evaluated against the independent second file. Test-shower deposited energies were never",
        "provided to generation.",
        "",
        "## Headline paper-protocol results",
        "",
        "| Metric | Condensation model | ViT-CFM reported |",
        "|---|---:|---:|",
        f"| Low-level classifier AUC | {paper['low_level']['auc']:.4f} | 0.518 |",
        f"| High-level classifier AUC | {paper['high_level']['auc']:.4f} | 0.504 |",
    ]
    if distances:
        lines.extend(
            (
                f"| FPD x10^3 | {distances['fpd_x1e3']:.4f} ± {distances['fpd_error_x1e3']:.4f} | — |",
                f"| KPD x10^3 | {distances['kpd_x1e3']:.4f} ± {distances['kpd_error_x1e3']:.4f} | — |",
            )
        )
    lines.extend(
        (
            "",
            "Lower classifier AUC is better; 0.5 is indistinguishable from Geant4.",
            "",
            "## Public CaloChallenge evaluator",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Low-level classifier AUC | {public['cls_low']['auc']:.4f} |",
            f"| High-level classifier AUC | {public['cls_high']['auc']:.4f} |",
            "",
            "## Fairness and resources",
            "",
            f"- Combined parameters: {generation['combined_parameter_count']:,} "
            f"({100.0 * generation['parameter_difference_fraction']:+.3f}% versus ViT-CFM).",
            "- Shape training: 119,790/1,210 split, batch 64, 800,000 steps, AdamW 1e-4/0.1, cosine decay.",
            "- Energy training: batch 256, 250,000 steps, identical AdamW schedule family.",
            f"- Generation: {generation['samples']:,} showers, batch 100, "
            f"{generation['milliseconds_per_shower']:.4f} ms/shower on {generation['device_name']}.",
            "- Primary checkpoints are final-step weights, matching `es_load_best_model: false`.",
            "- Paper-exact and public-challenge protocols are kept separate because their cuts and splits differ.",
        )
    )
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_status("complete", "all training, generation, evaluation, and reporting completed")
    return result


def main() -> int:
    COMPARISON.mkdir(parents=True, exist_ok=True)
    if FAILURE.exists():
        FAILURE.unlink()
    try:
        if FINAL_SUMMARY.exists() and FINAL_REPORT.exists():
            update_status("complete", "existing final comparison validated by artifact presence")
            return 0
        shape_result = wait_for_shape()
        generation = generate_showers()
        paper = paper_evaluation()
        public = public_evaluation()
        assemble_final_report(shape_result, generation, paper, public)
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_unix": time.time(),
        }
        write_json(FAILURE, failure)
        update_status("failed", str(error), failure_file=str(FAILURE))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
