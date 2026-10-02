#!/usr/bin/env python3
"""Conditional GPU benchmark on CaloChallenge Dataset 1 photon showers."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

import calorimeter_data as calo
import gpu_model as gm
import gpu_synthetic_benchmark as bench


PROJECT = Path(__file__).resolve().parent
DEFAULT_XML = PROJECT / "external/calochallenge-homepage/code/binning_dataset_1_photons.xml"
DEFAULT_TRAIN = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_1.hdf5"
DEFAULT_TEST = PROJECT / "data/calochallenge/dataset_1_photons/dataset_1_photons_2.hdf5"


@dataclass(frozen=True)
class CaloRunConfig:
    model_type: str = "coagulation"
    seed: int = 20261001
    train_size: int = 4000
    validation_size: int = 800
    test_size: int = 1000
    epochs: int = 25
    batch_size: int = 32
    samples: int = 1000
    data_components: int = 32
    max_components: int = 48
    # A 1% incident-energy cell cut yields a non-saturated sparse-set count
    # distribution on the official photon training file (median 24 cells).
    relative_threshold: float = 1e-2
    hidden_dim: int = 128
    transformer_layers: int = 3
    mixture_components: int = 8
    learning_rate: float = 2e-4
    sample_batch_size: int = 8
    sample_checkpoint_every: int = 25

    def __post_init__(self) -> None:
        if self.model_type not in {"coagulation", "birth_only"}:
            raise ValueError("model_type must be 'coagulation' or 'birth_only'")


def make_histories(records, corruption: gm.CorruptionConfig, seed: int):
    rng = random.Random(seed)
    return [
        gm.simulate_forward_history(record.state, corruption, rng, condition=record.condition)
        for record in records
    ]


def state_summary(state: gm.WeightedSetState, maximum: int) -> np.ndarray:
    masses = state.masses.detach().float().cpu().numpy()
    features = state.features.detach().float().cpu().numpy()
    response = float(masses.sum())
    if not len(masses) or response <= 0.0:
        return np.asarray([0.0, response, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, state.reservoir])
    weights = masses / response
    centroid = np.sum(weights[:, None] * features, axis=0)
    spread = float(np.sqrt(np.sum(weights * np.sum((features - centroid) ** 2, axis=1))))
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    if len(features) > 1:
        delta = features[:, None] - features[None, :]
        distances = np.sqrt(np.sum(delta * delta, axis=-1))
        pair_mean = float(distances[np.triu_indices(len(features), 1)].mean())
    else:
        pair_mean = 0.0
    return np.asarray(
        [len(masses) / maximum, response, weights.max(), entropy, *centroid.tolist(), spread, pair_mean, state.reservoir],
        dtype=np.float64,
    )


def evaluate_samples(reference, generated, maximum: int, seed: int):
    tv, cardinality_rows = bench.cardinality_tv(reference, generated, maximum)
    first = np.stack([state_summary(state, maximum) for state in reference])
    second = np.stack([state_summary(state, maximum) for state in generated])
    residuals = np.asarray([state.conservation_residual() for state in generated])
    generated_features = [state.features.detach().cpu().numpy() for state in generated if state.n]
    out_of_geometry = (
        float(np.mean(np.abs(np.concatenate(generated_features, axis=0)) > 1.0)) if generated_features else 0.0
    )
    metrics = {
        "cardinality_tv": tv,
        "retained_response_w1": bench.empirical_wasserstein(first[:, 1], second[:, 1]),
        "longitudinal_centroid_w1": bench.empirical_wasserstein(first[:, 4], second[:, 4]),
        "transverse_centroid_x_w1": bench.empirical_wasserstein(first[:, 5], second[:, 5]),
        "transverse_centroid_y_w1": bench.empirical_wasserstein(first[:, 6], second[:, 6]),
        "spatial_spread_w1": bench.empirical_wasserstein(first[:, 7], second[:, 7]),
        "summary_mmd": bench.mmd_rbf(first, second),
        "summary_energy_distance": bench.energy_distance(first, second, np.random.default_rng(seed)),
        "classifier_two_sample_accuracy": bench.classifier_two_sample(first, second, seed),
        "coordinate_out_of_geometry_fraction": out_of_geometry,
        "mass_residual_mean": float(residuals.mean()),
        "mass_residual_max": float(residuals.max()),
        "empty_sample_rate": float(np.mean([state.n == 0 for state in generated])),
    }
    return metrics, cardinality_rows


@torch.no_grad()
def generate(
    model,
    records,
    corruption,
    seed: int,
    output: Path,
    batch_size: int,
    checkpoint_every: int,
    leaf_temperature: float = 1.0,
    merge_rate_scale: float = 1.0,
    checkpoint_name: str = "sampling_checkpoint.pt",
):
    """Generate with batched neural calls and resumable atomic checkpoints."""
    if batch_size < 1 or checkpoint_every < 1:
        raise ValueError("sampling batch size and checkpoint interval must be positive")
    model.eval()
    condition_hash = hashlib.sha256()
    for record in records:
        condition_hash.update(record.condition.detach().cpu().numpy().tobytes())
        condition_hash.update(np.asarray([record.state.total_mass], dtype=np.float64).tobytes())
    model_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        model_hash.update(name.encode("utf-8"))
        model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    signature = {
        "count": len(records),
        "seed": seed,
        "batch_size": batch_size,
        "leaf_temperature": leaf_temperature,
        "merge_rate_scale": merge_rate_scale,
        "corruption": asdict(corruption),
        "condition_sha256": condition_hash.hexdigest(),
        "model_sha256": model_hash.hexdigest(),
    }
    checkpoint_path = output / checkpoint_name
    samples = []
    prior_seconds = 0.0
    prior_peak = 0.0
    rng = random.Random(seed)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError(f"sampling checkpoint configuration does not match: {checkpoint_path}")
        samples = checkpoint["samples"]
        prior_seconds = float(checkpoint["elapsed_seconds"])
        prior_peak = float(checkpoint.get("peak_gpu_memory_mb", 0.0))
        rng.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if model.device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        print(
            f"resuming sampling at {len(samples)}/{len(records)} from {checkpoint_path}",
            flush=True,
        )
    else:
        random.seed(seed)
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    next_checkpoint = ((len(samples) // checkpoint_every) + 1) * checkpoint_every

    def save_sampling_checkpoint(complete: bool) -> tuple[float, float]:
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            current_peak = torch.cuda.max_memory_allocated(model.device) / 1024**2
            cuda_states = torch.cuda.get_rng_state_all()
        else:
            current_peak = 0.0
            cuda_states = None
        elapsed = prior_seconds + time.perf_counter() - start
        peak = max(prior_peak, current_peak)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "signature": signature,
                "samples": samples,
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_mb": peak,
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": cuda_states,
                "complete": complete,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        return elapsed, peak

    while len(samples) < len(records):
        stop = min(len(records), len(samples) + batch_size)
        chunk = records[len(samples) : stop]
        generated = model.sample_batch(
            [record.state.total_mass for record in chunk],
            corruption,
            conditions=[record.condition for record in chunk],
            rng=rng,
            leaf_temperature=leaf_temperature,
            merge_rate_scale=merge_rate_scale,
        )
        samples.extend(state.to("cpu") for state in generated)
        if len(samples) >= next_checkpoint or len(samples) == len(records):
            elapsed, peak = save_sampling_checkpoint(len(samples) == len(records))
            rate = len(samples) / max(elapsed, 1e-9)
            eta = (len(records) - len(samples)) / max(rate, 1e-9)
            print(
                f"sampling progress={len(samples)}/{len(records)} "
                f"rate={rate:.3f}_samples/s eta_seconds={eta:.1f} "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
            while next_checkpoint <= len(samples):
                next_checkpoint += checkpoint_every
    elapsed, peak = save_sampling_checkpoint(True)
    return samples, 1000.0 * elapsed / len(samples), peak


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    run_config = CaloRunConfig(**{name: getattr(args, name) for name in CaloRunConfig.__dataclass_fields__})
    geometry = calo.load_geometry(args.geometry)
    train_metadata = calo.inspect_hdf5(args.train_file)
    test_metadata = calo.inspect_hdf5(args.test_file)
    log_mean, log_std = calo.incident_log_stats(args.train_file)
    total_needed = args.train_size + args.validation_size
    if total_needed > train_metadata["events"] or args.test_size > test_metadata["events"]:
        raise ValueError("requested split is larger than the available official file")
    loader = dict(
        geometry=geometry, max_components=args.data_components,
        relative_threshold=args.relative_threshold, log_energy_mean=log_mean, log_energy_std=log_std,
    )
    train_records = calo.load_records(args.train_file, start=0, count=args.train_size, **loader)
    validation_records = calo.load_records(
        args.train_file, start=args.train_size, count=args.validation_size, **loader
    )
    test_records = calo.load_records(args.test_file, start=0, count=args.test_size, **loader)
    corruption = gm.CorruptionConfig(
        max_components=args.max_components,
        split_rate=0.0 if args.model_type == "birth_only" else gm.CorruptionConfig.split_rate,
    )
    validation_histories = make_histories(validation_records, corruption, args.seed + 4)
    test_histories = make_histories(test_records, corruption, args.seed + 5)
    neural = gm.NeuralConfig(
        feature_dim=3, condition_dim=1, max_components=args.max_components,
        hidden_dim=args.hidden_dim, embedding_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers, attention_heads=4,
        mixture_components=args.mixture_components,
    )
    model = (
        gm.CoagulationModel(neural)
        if args.model_type == "coagulation"
        else gm.BirthOnlyModel(neural)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    best_loss = float("inf")
    best_state = None
    order_rng = random.Random(args.seed + 6)
    start_epoch = 1
    prior_training_seconds = 0.0
    latest_checkpoint = output / "latest.pt"
    if latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        saved_run_config = checkpoint.get("run_config", {})
        current_run_config = asdict(run_config)
        training_keys = set(current_run_config) - {"sample_batch_size", "sample_checkpoint_every"}
        saved_run_config.setdefault("model_type", "coagulation")
        if any(saved_run_config.get(key) != current_run_config[key] for key in training_keys):
            raise ValueError(f"resume checkpoint configuration does not match: {latest_checkpoint}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        best_state = checkpoint["best_state"]
        order_rng.setstate(checkpoint["order_rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        prior_training_seconds = checkpoint["training_seconds"]
        print(f"resuming from epoch {start_epoch:03d} using {latest_checkpoint}", flush=True)
    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = list(range(len(train_records)))
        order_rng.shuffle(order)
        total = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for batch_number, indices in enumerate(bench.batches(order, args.batch_size)):
            records = [train_records[index] for index in indices]
            history_seed = args.seed * 1_000_003 + epoch * 10_007 + batch_number
            histories = make_histories(records, corruption, history_seed)
            optimizer.zero_grad(set_to_none=True)
            terms = model.path_loss(histories, random.Random(history_seed + 1))
            if not bool(torch.isfinite(terms["loss"])):
                raise FloatingPointError("non-finite calorimeter path loss")
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(terms["loss"].detach().cpu()) * len(records)
            seen += len(records)
        validation = bench.evaluate_path_loss(
            model, validation_histories, args.batch_size, args.seed + 10_000 + epoch
        )
        row = {
            "epoch": epoch, "train_loss": total / seen,
            "validation_loss": validation["loss"], "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_nll={row['train_loss']:.4f} "
            f"val_nll={row['validation_loss']:.4f} seconds={row['epoch_seconds']:.2f}", flush=True,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
        elapsed_training = prior_training_seconds + time.perf_counter() - training_start
        temporary_checkpoint = output / "latest.pt.tmp"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_loss": best_loss,
                "best_state": best_state,
                "order_rng_state": order_rng.getstate(),
                "training_seconds": elapsed_training,
                "run_config": asdict(run_config),
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(latest_checkpoint)
        bench.write_csv(output / "training_history.csv", history)
    training_seconds = prior_training_seconds + time.perf_counter() - training_start
    if best_state is not None:
        model.load_state_dict(best_state)
    test_path = bench.evaluate_path_loss(model, test_histories, args.batch_size, args.seed + 20_000)
    generated, milliseconds, peak = generate(
        model,
        test_records[: args.samples],
        corruption,
        args.seed + 30_000,
        output,
        args.sample_batch_size,
        args.sample_checkpoint_every,
    )
    sample_metrics, cardinality_rows = evaluate_samples(
        [record.state for record in test_records[: args.samples]], generated, args.data_components, args.seed + 40_000
    )
    sample_metrics.update(milliseconds_per_sample=milliseconds, peak_gpu_memory_mb=peak)
    response_clipped = sum(record.retained_energy_mev > record.incident_energy_mev for record in train_records)
    result = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "dataset": {
            "name": "CaloChallenge Dataset 1 photons", "doi": "10.5281/zenodo.8099322",
            "train_file": str(args.train_file), "test_file": str(args.test_file),
            "geometry_file": str(args.geometry), "voxel_count": geometry.voxel_count,
            "incident_log_mean": log_mean, "incident_log_std": log_std,
            "train_over_response_clipped": response_clipped,
        },
        "run_config": asdict(run_config), "neural_config": asdict(neural),
        "model_type": args.model_type,
        "corruption_config": asdict(corruption),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds, "best_validation_nll": best_loss,
        "test_path": test_path, "sample_metrics": sample_metrics,
    }
    torch.save(
        {"model_state": model.state_dict(), "neural_config": asdict(neural),
         "corruption_config": asdict(corruption), "dataset": result["dataset"]}, output / "model.pt"
    )
    serializable = json.loads(json.dumps(result, default=str))
    (output / "summary.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    bench.write_csv(output / "training_history.csv", history)
    bench.write_csv(output / "cardinality.csv", cardinality_rows)
    bench.plot_training(output / "training.png", history)
    bench.plot_cardinality(output / "cardinality.png", cardinality_rows)
    report = [
        f"# CaloChallenge Dataset 1 photon benchmark: {args.model_type}", "",
        f"Official train/evaluation files: `{args.train_file.name}` / `{args.test_file.name}`.", "",
        f"Best validation path NLL: {best_loss:.6g}",
        f"Test path NLL: {test_path['loss']:.6g}", "",
        "## Sample metrics", "", "| Metric | Value |", "|---|---:|",
    ]
    report.extend(f"| {key} | {value:.6g} |" for key, value in sample_metrics.items())
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return serializable


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/gpu_calorimeter")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_XML)
    for name, field in CaloRunConfig.__dataclass_fields__.items():
        option = "--" + name.replace("_", "-")
        choices = ["coagulation", "birth_only"] if name == "model_type" else None
        parser.add_argument(option, type=type(field.default), default=field.default, choices=choices)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
