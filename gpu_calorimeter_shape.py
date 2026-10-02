#!/usr/bin/env python3
"""Train the parameter-matched Dataset-1 hybrid shower-shape model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

import calorimeter_data as calo
import calorimeter_energy as energy
import calorimeter_shape as shape
import gpu_model as gm


PROJECT = Path(__file__).resolve().parent
DEFAULT_XML = PROJECT / "external/calochallenge-homepage/code/binning_dataset_1_photons.xml"
DEFAULT_TRAIN = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_1.hdf5"
DEFAULT_ENERGY = PROJECT / "results/gpu_official/energy_full/model.pt"


@dataclass(frozen=True)
class ShapeRunConfig:
    seed: int = 20261003
    train_size: int = 119790
    validation_size: int = 1210
    iterations: int = 800000
    batch_size: int = 64
    validation_batch_size: int = 64
    validation_every: int = 4000
    validation_samples: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    core_components: int = 48
    hidden_dim: int = 480
    transformer_layers: int = 6
    attention_heads: int = 6
    mixture_components: int = 16
    residual_hidden_dim: int = 1536
    residual_layers: int = 4
    residual_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        integer_values = (
            self.train_size,
            self.validation_size,
            self.iterations,
            self.batch_size,
            self.validation_batch_size,
            self.validation_every,
            self.validation_samples,
            self.core_components,
            self.hidden_dim,
            self.transformer_layers,
            self.attention_heads,
            self.mixture_components,
            self.residual_hidden_dim,
            self.residual_layers,
        )
        if min(integer_values) < 1:
            raise ValueError("sizes, iterations, and model dimensions must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if self.residual_loss_weight <= 0.0:
            raise ValueError("residual_loss_weight must be positive")


def batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield slice(start, min(count, start + batch_size))


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_energy_transform(path: Path) -> energy.EnergyTransform:
    if not path.exists():
        raise FileNotFoundError(f"completed energy checkpoint is required: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return energy.EnergyTransform.from_dict(checkpoint["energy_transform"])


def make_conditions(
    transform: energy.EnergyTransform,
    incidents_mev: np.ndarray,
    layer_energies_mev: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    incidents = torch.from_numpy(np.asarray(incidents_mev, dtype=np.float32)).reshape(-1)
    layers = torch.from_numpy(np.asarray(layer_energies_mev, dtype=np.float32))
    targets, nonempty = transform.encode_targets(incidents, layers)
    conditions = torch.cat((transform.condition(incidents), targets), dim=-1).float()
    return conditions, nonempty


def loss_on_batch(
    model: shape.HybridVoxelShapeModel,
    showers_mev: np.ndarray,
    conditions: torch.Tensor,
    geometry: calo.CalorimeterGeometry,
    corruption: gm.CorruptionConfig,
    objective_generator: torch.Generator,
    residual_weight: float,
) -> dict[str, torch.Tensor]:
    states, fractions = shape.dense_to_core_states(
        showers_mev, geometry.features, model.config.core_components
    )
    core = model.sampled_core_nll(states, conditions, corruption, objective_generator)
    residual = model.residual_nll(states, conditions, fractions)
    return {
        "loss": core["loss"] + residual_weight * residual["loss"],
        "count_nll": core["count_nll"],
        "birth_nll": core["birth_nll"],
        "residual_nll": residual["residual_nll"],
        "residual_cross_entropy": residual["residual_cross_entropy"],
    }


@torch.no_grad()
def evaluate(
    model: shape.HybridVoxelShapeModel,
    showers_mev: np.ndarray,
    conditions: torch.Tensor,
    geometry: calo.CalorimeterGeometry,
    corruption: gm.CorruptionConfig,
    batch_size: int,
    seed: int,
    residual_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "count_nll": 0.0,
        "birth_nll": 0.0,
        "residual_nll": 0.0,
        "residual_cross_entropy": 0.0,
    }
    seen = 0
    generator = torch.Generator().manual_seed(seed)
    for indices in batches(len(showers_mev), batch_size):
        count = indices.stop - indices.start
        terms = loss_on_batch(
            model,
            showers_mev[indices],
            conditions[indices],
            geometry,
            corruption,
            generator,
            residual_weight,
        )
        for name in totals:
            totals[name] += float(terms[name].detach().cpu()) * count
        seen += count
    return {name: value / seen for name, value in totals.items()}


@torch.no_grad()
def sample_shape_metrics(
    model: shape.HybridVoxelShapeModel,
    showers_mev: np.ndarray,
    incidents_mev: np.ndarray,
    layer_energies_mev: np.ndarray,
    conditions: torch.Tensor,
    corruption: gm.CorruptionConfig,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    totals = np.asarray(layer_energies_mev, dtype=np.float32).sum(axis=-1)
    layer_fractions = np.asarray(layer_energies_mev, dtype=np.float32) / np.maximum(
        totals[:, None], 1e-12
    )
    generated = []
    rng = random.Random(seed)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    for indices in batches(len(showers_mev), batch_size):
        fractions, _ = model.sample_fractions(
            conditions[indices],
            torch.from_numpy(layer_fractions[indices]),
            corruption,
            rng,
        )
        generated.append(fractions.cpu().numpy() * totals[indices, None])
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        peak = torch.cuda.max_memory_allocated(model.device) / 1024**2
    else:
        peak = 0.0
    elapsed = time.perf_counter() - start
    generated_mev = np.concatenate(generated)
    reference_total = np.maximum(showers_mev.sum(axis=-1, keepdims=True), 1e-12)
    generated_total = np.maximum(generated_mev.sum(axis=-1, keepdims=True), 1e-12)
    reference_fraction = showers_mev / reference_total
    generated_fraction = generated_mev / generated_total
    mean_profile_l1 = float(
        np.abs(reference_fraction.mean(axis=0) - generated_fraction.mean(axis=0)).sum()
    )
    reference_active = (showers_mev >= 1.0).sum(axis=-1)
    generated_active = (generated_mev >= 1.0).sum(axis=-1)
    metrics = {
        "mean_profile_l1": mean_profile_l1,
        "reference_active_cells_mean": float(reference_active.mean()),
        "generated_active_cells_mean": float(generated_active.mean()),
        "active_cells_mean_error": float(abs(reference_active.mean() - generated_active.mean())),
        "layer_conservation_max_mev": float(
            np.max(
                np.abs(
                    np.stack(
                        [
                            generated_mev[:, model.layer_indices.cpu().numpy() == layer].sum(axis=1)
                            for layer in range(model.config.layer_count)
                        ],
                        axis=1,
                    )
                    - layer_energies_mev
                )
            )
        ),
        "milliseconds_per_shower": 1000.0 * elapsed / len(showers_mev),
        "peak_gpu_memory_mb": peak,
        "incident_response_mean": float(np.mean(generated_mev.sum(axis=1) / incidents_mev)),
    }
    return metrics, generated_mev


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = ShapeRunConfig(
        **{name: getattr(args, name) for name in ShapeRunConfig.__dataclass_fields__}
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    metadata = calo.inspect_hdf5(args.train_file)
    requested = config.train_size + config.validation_size
    if requested > metadata["events"]:
        raise ValueError("training and validation splits exceed the official file")
    geometry = calo.load_geometry(args.geometry)
    if geometry.voxel_count != metadata["voxels"]:
        raise ValueError("training file and detector geometry disagree")
    dense = calo.load_dense_batch(args.train_file, count=requested)
    layer_indices, layer_ids = shape.contiguous_layer_indices(geometry.layer_ids)
    layer_energies = energy.layer_energies(dense.showers_mev, geometry).astype(np.float32)
    transform = load_energy_transform(args.energy_checkpoint)
    if tuple(int(value) for value in transform.layer_ids) != layer_ids:
        raise ValueError("energy checkpoint and shape geometry use different layers")
    conditions, _ = make_conditions(transform, dense.incident_energies_mev, layer_energies)

    train_showers = dense.showers_mev[: config.train_size]
    train_conditions = conditions[: config.train_size]
    validation_showers = dense.showers_mev[-config.validation_size :]
    validation_incidents = dense.incident_energies_mev[-config.validation_size :]
    validation_layers = layer_energies[-config.validation_size :]
    validation_conditions = conditions[-config.validation_size :]

    model_config = shape.ShapeModelConfig(
        voxel_count=geometry.voxel_count,
        layer_count=len(layer_ids),
        condition_dim=conditions.shape[1],
        core_components=config.core_components,
        hidden_dim=config.hidden_dim,
        transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
        mixture_components=config.mixture_components,
        residual_hidden_dim=config.residual_hidden_dim,
        residual_layers=config.residual_layers,
    )
    model = shape.HybridVoxelShapeModel(model_config, geometry.features, layer_indices).to(device)
    corruption = gm.CorruptionConfig(
        max_components=config.core_components,
        split_rate=0.0,
        gamma=0.0,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    reference_parameters = 25_982_005
    official_architecture = (
        config.core_components == 48
        and config.hidden_dim == 480
        and config.transformer_layers == 6
        and config.attention_heads == 6
        and config.mixture_components == 16
        and config.residual_hidden_dim == 1536
        and config.residual_layers == 4
    )
    if official_architecture and abs(parameter_count - reference_parameters) / reference_parameters >= 0.01:
        raise ValueError("shape model is not within one percent of the reference parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=0.0
    )
    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    objective_generator = torch.Generator().manual_seed(config.seed + 2)
    start_step = 1
    best_validation = float("inf")
    history: list[dict] = []
    prior_seconds = 0.0
    latest = output / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["run_config"] != asdict(config):
            raise ValueError(f"resume checkpoint configuration does not match: {latest}")
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError(f"resume model configuration does not match: {latest}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        batch_generator.set_state(checkpoint["batch_generator_state"].cpu())
        objective_generator.set_state(checkpoint["objective_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        start_step = checkpoint["step"] + 1
        best_validation = checkpoint["best_validation"]
        history = checkpoint["history"]
        prior_seconds = checkpoint["training_seconds"]
        print(f"resuming shape training at step {start_step} from {latest}", flush=True)

    start = time.perf_counter()
    window = {name: 0.0 for name in (
        "loss", "count_nll", "birth_nll", "residual_nll", "residual_cross_entropy"
    )}
    window_count = 0
    for step in range(start_step, config.iterations + 1):
        model.train()
        indices = torch.randint(
            config.train_size, (config.batch_size,), generator=batch_generator
        ).numpy()
        optimizer.zero_grad(set_to_none=True)
        terms = loss_on_batch(
            model,
            train_showers[indices],
            train_conditions[indices],
            geometry,
            corruption,
            objective_generator,
            config.residual_loss_weight,
        )
        if not bool(torch.isfinite(terms["loss"])):
            diagnostic = {
                name: float(value.detach().cpu()) for name, value in terms.items()
            }
            print(
                f"nonfinite step={step} terms={diagnostic} "
                f"batch_index_min={int(indices.min())} batch_index_max={int(indices.max())} "
                f"shower_min={float(train_showers[indices].min())} "
                f"shower_max={float(train_showers[indices].max())} "
                f"condition_finite={bool(torch.isfinite(train_conditions[indices]).all())}",
                flush=True,
            )
            raise FloatingPointError("non-finite shape-model loss")
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        for name in window:
            window[name] += float(terms[name].detach().cpu())
        window_count += 1

        if step % config.validation_every == 0 or step == config.iterations:
            validation = evaluate(
                model,
                validation_showers,
                validation_conditions,
                geometry,
                corruption,
                config.validation_batch_size,
                config.seed + 100_000,
                config.residual_loss_weight,
            )
            elapsed = prior_seconds + time.perf_counter() - start
            row = {
                "step": step,
                **{f"train_{name}": value / window_count for name, value in window.items()},
                **{f"validation_{name}": value for name, value in validation.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_seconds": elapsed,
            }
            history.append(row)
            print(
                f"step={step:06d} train_loss={row['train_loss']:.5f} "
                f"val_loss={row['validation_loss']:.5f} "
                f"birth={row['validation_birth_nll']:.5f} "
                f"residual={row['validation_residual_nll']:.5f} "
                f"lr={row['learning_rate']:.3g} seconds={elapsed:.1f}",
                flush=True,
            )
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": model_config.to_dict(),
                        "energy_transform": transform.to_dict(),
                        "run_config": asdict(config),
                        "selection": "best_validation_diagnostic_only",
                        "step": step,
                        "validation_loss": best_validation,
                    },
                    output / "best_model_diagnostic.pt",
                )
            atomic_torch_save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "batch_generator_state": batch_generator.get_state(),
                    "objective_generator_state": objective_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "best_validation": best_validation,
                    "history": history,
                    "training_seconds": elapsed,
                    "run_config": asdict(config),
                    "model_config": model_config.to_dict(),
                },
                latest,
            )
            window = {name: 0.0 for name in window}
            window_count = 0

    training_seconds = prior_seconds + time.perf_counter() - start
    final_validation = evaluate(
        model,
        validation_showers,
        validation_conditions,
        geometry,
        corruption,
        config.validation_batch_size,
        config.seed + 100_000,
        config.residual_loss_weight,
    )
    sample_count = min(config.validation_samples, config.validation_size)
    torch.manual_seed(config.seed + 3)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 3)
    sample_metrics, generated = sample_shape_metrics(
        model,
        validation_showers[:sample_count],
        validation_incidents[:sample_count],
        validation_layers[:sample_count],
        validation_conditions[:sample_count],
        corruption,
        min(config.validation_batch_size, 100),
        config.seed + 4,
    )
    np.savez_compressed(
        output / "validation_shape_samples.npz",
        generated_showers_mev=generated,
        reference_showers_mev=validation_showers[:sample_count],
        incident_energies_mev=validation_incidents[:sample_count],
    )
    result = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "dataset": {
            "name": "CaloChallenge Dataset 1 photons",
            "train_file": str(args.train_file),
            "geometry_file": str(args.geometry),
            "layer_ids": list(layer_ids),
            "split": [config.train_size, config.validation_size],
            "full_voxel_count": geometry.voxel_count,
        },
        "run_config": asdict(config),
        "model_config": model_config.to_dict(),
        "corruption_config": asdict(corruption),
        "energy_transform": transform.to_dict(),
        "parameter_count": parameter_count,
        "reference_parameter_count": reference_parameters,
        "parameter_difference_fraction": (parameter_count - reference_parameters) / reference_parameters,
        "checkpoint_selection": "final_step",
        "final_validation": final_validation,
        "best_validation_loss": best_validation,
        "training_seconds": training_seconds,
        "sample_metrics": sample_metrics,
        "history": history,
    }
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "energy_transform": transform.to_dict(),
            "run_config": asdict(config),
            "corruption_config": asdict(corruption),
        },
        output / "model.pt",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Dataset-1 photon shape-model report",
        "",
        f"Trainable parameters: {parameter_count:,} "
        f"({100.0 * result['parameter_difference_fraction']:+.3f}% versus ViT-CFM shape network).",
        "",
        f"Final-step validation loss: {final_validation['loss']:.6g}",
        f"Best validation loss (diagnostic only): {best_validation:.6g}",
        f"Training seconds: {training_seconds:.3f}",
        "",
        "## Shape-only validation metrics",
        "",
        "Layer energies are supplied from the held-out truth here to isolate shape quality.",
        "The final official sample instead uses independently generated layer energies.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report.extend(f"| {name} | {value:.6g} |" for name, value in sample_metrics.items())
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/gpu_official/shape")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_XML)
    parser.add_argument("--energy-checkpoint", type=Path, default=DEFAULT_ENERGY)
    for name, field in ShapeRunConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)
