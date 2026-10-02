#!/usr/bin/env python3
"""Generate a resumable official Dataset-1 HDF5 sample from both trained models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch

import calorimeter_data as calo
import calorimeter_energy as energy
import calorimeter_shape as shape
import gpu_model as gm


PROJECT = Path(__file__).resolve().parent
DEFAULT_XML = PROJECT / "external/calochallenge-homepage/code/binning_dataset_1_photons.xml"
DEFAULT_ENERGY = PROJECT / "results/gpu_official/energy_full/model.pt"
DEFAULT_SHAPE = PROJECT / "results/gpu_official/shape_full/model.pt"
DEFAULT_OUTPUT = PROJECT / "results/gpu_official/fair_comparison/generated_ds1_photons.hdf5"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_energy_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = energy.EnergyModelConfig(**checkpoint["model_config"])
    model = energy.LayerEnergyModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    transform = energy.EnergyTransform.from_dict(checkpoint["energy_transform"])
    return model, transform


def load_shape_model(
    path: Path,
    geometry: calo.CalorimeterGeometry,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = shape.ShapeModelConfig(**checkpoint["model_config"])
    layer_indices, layer_ids = shape.contiguous_layer_indices(geometry.layer_ids)
    model = shape.HybridVoxelShapeModel(config, geometry.features, layer_indices)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    corruption = gm.CorruptionConfig(**checkpoint["corruption_config"])
    return model, corruption, layer_ids


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if args.samples < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("sample count, batch size, and checkpoint interval must be positive")
    if not args.energy_checkpoint.exists() or not args.shape_checkpoint.exists():
        raise FileNotFoundError("both completed final-step model checkpoints are required")
    device = torch.device(args.device)
    geometry = calo.load_geometry(args.geometry)
    energy_model, transform = load_energy_model(args.energy_checkpoint, device)
    shape_model, corruption, layer_ids = load_shape_model(args.shape_checkpoint, geometry, device)
    if tuple(int(value) for value in transform.layer_ids) != layer_ids:
        raise ValueError("energy model and shape model use different calorimeter layers")
    energy_parameters = sum(value.numel() for value in energy_model.parameters() if value.requires_grad)
    shape_parameters = sum(value.numel() for value in shape_model.parameters() if value.requires_grad)
    reference_total = 27_938_230
    combined_parameters = energy_parameters + shape_parameters
    if abs(combined_parameters - reference_total) / reference_total >= 0.01:
        raise ValueError("combined model is not within one percent of the reference parameter count")

    repeats = math.ceil(args.samples / 121)
    # ViT-CFM's Dataset-1 helper shuffles 1,000 complete 121-event schedules
    # and therefore produces 121,000 conditions even though its YAML retains
    # the older nominal ``n_samples: 100000`` value.  The default below follows
    # the executable reference pipeline; alternative counts remain explicit.
    incidents = calo.dataset1_incident_energies(
        repeats=repeats, seed=args.seed + 17
    )[: args.samples].reshape(-1)
    signature = {
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "energy_checkpoint": str(args.energy_checkpoint.resolve()),
        "energy_sha256": file_sha256(args.energy_checkpoint),
        "shape_checkpoint": str(args.shape_checkpoint.resolve()),
        "shape_sha256": file_sha256(args.shape_checkpoint),
        "geometry": str(args.geometry.resolve()),
        "corruption": asdict(corruption),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output_file.with_suffix(args.output_file.suffix + ".state.pt")
    completed = 0
    elapsed_prior = 0.0
    python_rng = random.Random(args.seed + 1)
    if state_path.exists():
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError(f"generation checkpoint configuration does not match: {state_path}")
        completed = int(checkpoint["completed"])
        elapsed_prior = float(checkpoint["elapsed_seconds"])
        python_rng.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        print(f"resuming official generation at {completed}/{args.samples}", flush=True)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    mode = "r+" if args.output_file.exists() else "w"
    with h5py.File(args.output_file, mode) as handle:
        if mode == "w":
            handle.create_dataset(
                "showers", shape=(args.samples, geometry.voxel_count), dtype=np.float32
            )
            handle.create_dataset(
                "incident_energies", data=incidents[:, None].astype(np.float32)
            )
            handle.attrs["completed_samples"] = 0
            handle.attrs["complete"] = False
        else:
            if handle["showers"].shape != (args.samples, geometry.voxel_count):
                raise ValueError("partial output HDF5 has the wrong shape")
            np.testing.assert_array_equal(
                np.asarray(handle["incident_energies"]).reshape(-1), incidents.astype(np.float32)
            )
        if completed > args.samples:
            raise ValueError("generation checkpoint is beyond the requested sample count")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        next_checkpoint = ((completed // args.checkpoint_every) + 1) * args.checkpoint_every
        while completed < args.samples:
            stop = min(args.samples, completed + args.batch_size)
            incident_tensor = torch.from_numpy(incidents[completed:stop].astype(np.float32)).to(device)
            total, layers, _ = energy_model.sample(incident_tensor, transform)
            standardized, _ = transform.encode_targets(incident_tensor, layers)
            conditions = torch.cat((transform.condition(incident_tensor), standardized), dim=-1)
            layer_fractions = layers / total[:, None].clamp_min(1e-12)
            dense_fractions, _ = shape_model.sample_fractions(
                conditions, layer_fractions, corruption, python_rng
            )
            showers = (dense_fractions * total[:, None]).float().cpu().numpy()
            handle["showers"][completed:stop] = showers
            completed = stop
            if completed >= next_checkpoint or completed == args.samples:
                handle.flush()
                elapsed = elapsed_prior + time.perf_counter() - start
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                )
                atomic_save(
                    {
                        "signature": signature,
                        "completed": completed,
                        "elapsed_seconds": elapsed,
                        "python_rng_state": python_rng.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": (
                            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                        ),
                        "peak_gpu_memory_mb": peak,
                    },
                    state_path,
                )
                handle.attrs["completed_samples"] = completed
                handle.flush()
                rate = completed / max(elapsed, 1e-12)
                eta = (args.samples - completed) / max(rate, 1e-12)
                print(
                    f"generation progress={completed}/{args.samples} "
                    f"rate={rate:.3f}_showers/s eta_seconds={eta:.1f}",
                    flush=True,
                )
                while next_checkpoint <= completed:
                    next_checkpoint += args.checkpoint_every
        handle.attrs["complete"] = True
        handle.attrs["completed_samples"] = completed
        handle.attrs["combined_parameter_count"] = combined_parameters
        handle.flush()

    elapsed = elapsed_prior + time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    result = {
        "sample_file": str(args.output_file),
        "samples": args.samples,
        "voxel_count": geometry.voxel_count,
        "energy_parameter_count": energy_parameters,
        "shape_parameter_count": shape_parameters,
        "combined_parameter_count": combined_parameters,
        "reference_parameter_count": reference_total,
        "parameter_difference_fraction": (combined_parameters - reference_total) / reference_total,
        "elapsed_seconds": elapsed,
        "milliseconds_per_shower": 1000.0 * elapsed / args.samples,
        "peak_gpu_memory_mb": peak,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "signature": signature,
    }
    summary_path = args.output_file.parent / "generation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--energy-checkpoint", type=Path, default=DEFAULT_ENERGY)
    parser.add_argument("--shape-checkpoint", type=Path, default=DEFAULT_SHAPE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_XML)
    parser.add_argument("--samples", type=int, default=121000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
