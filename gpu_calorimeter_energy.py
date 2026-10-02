#!/usr/bin/env python3
"""Train the size-matched Dataset-1 layer-energy generator on GPU."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import calorimeter_data as calo
import calorimeter_energy as energy


PROJECT = Path(__file__).resolve().parent
DEFAULT_XML = PROJECT / "external/calochallenge-homepage/code/binning_dataset_1_photons.xml"
DEFAULT_TRAIN = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_1.hdf5"


@dataclass(frozen=True)
class EnergyRunConfig:
    seed: int = 20261002
    train_size: int = 119790
    validation_size: int = 1210
    iterations: int = 250000
    batch_size: int = 256
    validation_batch_size: int = 1024
    validation_every: int = 4000
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    hidden_dim: int = 676
    hidden_layers: int = 5
    mixture_components: int = 16

    def __post_init__(self) -> None:
        integer_values = (
            self.train_size, self.validation_size, self.iterations, self.batch_size,
            self.validation_batch_size, self.validation_every, self.hidden_dim,
            self.hidden_layers, self.mixture_components,
        )
        if min(integer_values) < 1:
            raise ValueError("sizes, iterations, and model dimensions must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")


def batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield slice(start, min(count, start + batch_size))


@torch.no_grad()
def evaluate(
    model: energy.LayerEnergyModel,
    conditions: torch.Tensor,
    targets: torch.Tensor,
    nonempty: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "empty_nll": 0.0, "continuous_nll": 0.0}
    seen = 0
    for indices in batches(len(conditions), batch_size):
        count = indices.stop - indices.start
        terms = model.nll(
            conditions[indices].to(device), targets[indices].to(device), nonempty[indices].to(device)
        )
        for name in totals:
            totals[name] += float(terms[name].cpu()) * count
        seen += count
    return {name: value / seen for name, value in totals.items()}


def wasserstein_equal(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) != len(second):
        raise ValueError("equal-size samples are required")
    return float(np.mean(np.abs(np.sort(first) - np.sort(second))))


@torch.no_grad()
def sample_metrics(
    model: energy.LayerEnergyModel,
    transform: energy.EnergyTransform,
    incidents: torch.Tensor,
    reference_layers: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    generated_total = []
    generated_layers = []
    for indices in batches(len(incidents), batch_size):
        total, layers, _ = model.sample(incidents[indices].to(device), transform)
        generated_total.append(total.cpu())
        generated_layers.append(layers.cpu())
    generated_total_tensor = torch.cat(generated_total)
    generated_layer_tensor = torch.cat(generated_layers)
    reference_total = reference_layers.sum(dim=-1)
    reference_response = (reference_total / incidents).numpy()
    generated_response = (generated_total_tensor / incidents).numpy()
    reference_fraction = reference_layers / reference_total[:, None].clamp_min(1e-12)
    generated_fraction = generated_layer_tensor / generated_total_tensor[:, None].clamp_min(1e-12)
    layer_w1 = [
        wasserstein_equal(reference_fraction[:, index].numpy(), generated_fraction[:, index].numpy())
        for index in range(reference_layers.shape[1])
    ]
    metrics = {
        "response_w1": wasserstein_equal(reference_response, generated_response),
        "mean_layer_fraction_w1": float(np.mean(layer_w1)),
        "reference_empty_rate": float((reference_total <= 1e-6).float().mean()),
        "generated_empty_rate": float((generated_total_tensor <= 1e-6).float().mean()),
        "reference_response_mean": float(np.mean(reference_response)),
        "generated_response_mean": float(np.mean(generated_response)),
    }
    metrics.update({f"layer_{index}_fraction_w1": value for index, value in enumerate(layer_w1)})
    return metrics, generated_total_tensor.numpy(), generated_layer_tensor.numpy()


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = EnergyRunConfig(
        **{name: getattr(args, name) for name in EnergyRunConfig.__dataclass_fields__}
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    metadata = calo.inspect_hdf5(args.train_file)
    if config.train_size + config.validation_size > metadata["events"]:
        raise ValueError("training and validation splits exceed the official file")
    geometry = calo.load_geometry(args.geometry)
    incidents_np, layers_np = energy.load_layer_energy_arrays(
        args.train_file,
        geometry,
        count=config.train_size + config.validation_size,
    )
    train_incidents_np = incidents_np[: config.train_size]
    train_layers_np = layers_np[: config.train_size]
    validation_incidents_np = incidents_np[-config.validation_size :]
    validation_layers_np = layers_np[-config.validation_size :]
    layer_ids = tuple(int(value) for value in np.unique(geometry.layer_ids))
    transform = energy.fit_energy_transform(train_incidents_np, train_layers_np, layer_ids)

    train_incidents = torch.from_numpy(train_incidents_np).float()
    train_layers = torch.from_numpy(train_layers_np).float()
    validation_incidents = torch.from_numpy(validation_incidents_np).float()
    validation_layers = torch.from_numpy(validation_layers_np).float()
    train_conditions = transform.condition(train_incidents)
    train_targets, train_nonempty = transform.encode_targets(train_incidents, train_layers)
    validation_conditions = transform.condition(validation_incidents)
    validation_targets, validation_nonempty = transform.encode_targets(
        validation_incidents, validation_layers
    )

    model_config = energy.EnergyModelConfig(
        dimension=len(layer_ids),
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        mixture_components=config.mixture_components,
    )
    model = energy.LayerEnergyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=0.0
    )
    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    start_step = 1
    best_validation = float("inf")
    best_state = None
    history = []
    prior_seconds = 0.0
    latest = output / "latest.pt"
    if latest.exists():
        # RNG states are CPU ByteTensors even when model training runs on CUDA.
        # Load the container on CPU; model/optimizer restoration moves learned
        # tensors to their parameter devices as needed.
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["run_config"] != asdict(config):
            raise ValueError(f"resume checkpoint configuration does not match: {latest}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        batch_generator.set_state(checkpoint["batch_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        start_step = checkpoint["step"] + 1
        best_validation = checkpoint["best_validation"]
        best_state = checkpoint["best_state"]
        history = checkpoint["history"]
        prior_seconds = checkpoint["training_seconds"]
        print(f"resuming energy training at step {start_step} from {latest}", flush=True)

    start = time.perf_counter()
    window_loss = 0.0
    window_count = 0
    for step in range(start_step, config.iterations + 1):
        model.train()
        indices = torch.randint(
            len(train_conditions), (config.batch_size,), generator=batch_generator
        )
        optimizer.zero_grad(set_to_none=True)
        terms = model.nll(
            train_conditions[indices].to(device),
            train_targets[indices].to(device),
            train_nonempty[indices].to(device),
        )
        if not bool(torch.isfinite(terms["loss"])):
            raise FloatingPointError("non-finite energy-model loss")
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        scheduler.step()
        window_loss += float(terms["loss"].detach().cpu())
        window_count += 1

        if step % config.validation_every == 0 or step == config.iterations:
            validation = evaluate(
                model,
                validation_conditions,
                validation_targets,
                validation_nonempty,
                config.validation_batch_size,
                device,
            )
            elapsed = prior_seconds + time.perf_counter() - start
            row = {
                "step": step,
                "train_nll": window_loss / window_count,
                "validation_nll": validation["loss"],
                "empty_nll": validation["empty_nll"],
                "continuous_nll": validation["continuous_nll"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_seconds": elapsed,
            }
            history.append(row)
            print(
                f"step={step:06d} train_nll={row['train_nll']:.5f} "
                f"val_nll={row['validation_nll']:.5f} lr={row['learning_rate']:.3g} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                best_state = copy.deepcopy(model.state_dict())
            atomic_torch_save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "batch_generator_state": batch_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "best_validation": best_validation,
                    "best_state": best_state,
                    "history": history,
                    "training_seconds": elapsed,
                    "run_config": asdict(config),
                },
                latest,
            )
            window_loss = 0.0
            window_count = 0

    training_seconds = prior_seconds + time.perf_counter() - start
    # The published ViT-CFM training configuration uses
    # ``es_load_best_model: false``. Keep the final optimizer-step weights as
    # the primary fair-comparison artifact; the best-validation checkpoint is
    # saved separately for diagnostics and is never used for the reported
    # samples or metrics.
    final_validation = evaluate(
        model,
        validation_conditions,
        validation_targets,
        validation_nonempty,
        config.validation_batch_size,
        device,
    )
    torch.manual_seed(config.seed + 2)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 2)
    metrics, generated_total, generated_layers = sample_metrics(
        model,
        transform,
        validation_incidents,
        validation_layers,
        config.validation_batch_size,
        device,
    )
    np.savez_compressed(
        output / "validation_samples.npz",
        incident_energies_mev=validation_incidents_np,
        total_energies_mev=generated_total,
        layer_energies_mev=generated_layers,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
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
        },
        "run_config": asdict(config),
        "model_config": asdict(model_config),
        "energy_transform": transform.to_dict(),
        "parameter_count": parameter_count,
        "reference_parameter_count": 1_956_225,
        "parameter_difference_fraction": (parameter_count - 1_956_225) / 1_956_225,
        "checkpoint_selection": "final_step",
        "final_validation_nll": final_validation["loss"],
        "best_validation_nll": best_validation,
        "training_seconds": training_seconds,
        "sample_metrics": metrics,
        "history": history,
    }
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "energy_transform": transform.to_dict(),
            "run_config": asdict(config),
        },
        output / "model.pt",
    )
    if best_state is not None:
        atomic_torch_save(
            {
                "model_state": best_state,
                "model_config": asdict(model_config),
                "energy_transform": transform.to_dict(),
                "run_config": asdict(config),
                "selection": "best_validation_diagnostic_only",
            },
            output / "best_model_diagnostic.pt",
        )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Dataset-1 photon energy-model report",
        "",
        f"Trainable parameters: {parameter_count:,} "
        f"({100.0 * result['parameter_difference_fraction']:+.3f}% versus ViT-CFM energy network).",
        "",
        f"Final-step validation NLL: {final_validation['loss']:.6g}",
        f"Best validation NLL (diagnostic only): {best_validation:.6g}",
        f"Training seconds: {training_seconds:.3f}",
        "",
        "## Validation-sample metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report.extend(f"| {name} | {value:.6g} |" for name, value in metrics.items())
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/gpu_official/energy")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_XML)
    for name, field in EnergyRunConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)
