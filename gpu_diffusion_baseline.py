#!/usr/bin/env python3
"""Cardinality-conditioned permutation-equivariant DDPM baseline."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import gpu_model as gm
import gpu_synthetic_benchmark as bench


@dataclass(frozen=True)
class DiffusionConfig:
    feature_dim: int = 2
    min_cardinality: int = 2
    max_cardinality: int = 8
    hidden_dim: int = 128
    transformer_layers: int = 3
    attention_heads: int = 4
    time_embedding_dim: int = 32
    diffusion_steps: int = 100
    beta_min: float = 1e-4
    beta_max: float = 2e-2


def sinusoidal_embedding(time: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=time.device, dtype=time.dtype) / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class SetDenoiser(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        data_dim = 1 + config.feature_dim
        context_dim = config.time_embedding_dim + 1
        self.input = nn.Sequential(
            nn.Linear(data_dim + context_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.transformer_layers, norm=nn.LayerNorm(config.hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, data_dim),
        )

    def forward(self, noisy: Tensor, mask: Tensor, steps: Tensor, counts: Tensor) -> Tensor:
        normalized_time = steps.to(noisy.dtype) / max(self.config.diffusion_steps - 1, 1)
        time_embedding = sinusoidal_embedding(normalized_time, self.config.time_embedding_dim)
        count_feature = (counts.to(noisy.dtype) / self.config.max_cardinality)[:, None]
        context = torch.cat((time_embedding, count_feature), dim=-1)
        tokens = self.input(torch.cat((noisy, context[:, None, :].expand(-1, noisy.shape[1], -1)), dim=-1))
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        return self.output(tokens) * mask[..., None]


class CardinalityDiffusion(nn.Module):
    def __init__(self, config: DiffusionConfig, channel_mean: Tensor, channel_std: Tensor):
        super().__init__()
        self.config = config
        self.denoiser = SetDenoiser(config)
        if config.min_cardinality < 1 or config.min_cardinality > config.max_cardinality:
            raise ValueError("invalid cardinality support")
        self.count_logits = nn.Parameter(
            torch.zeros(config.max_cardinality - config.min_cardinality + 1)
        )
        self.register_buffer("channel_mean", channel_mean)
        self.register_buffer("channel_std", channel_std)
        betas = torch.linspace(config.beta_min, config.beta_max, config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def device(self) -> torch.device:
        return self.count_logits.device

    def loss(self, states: Sequence[gm.WeightedSetState], generator: torch.Generator | None = None) -> dict[str, Tensor]:
        clean, mask, counts = pack_states(states, self.config.max_cardinality, self.device)
        clean = (clean - self.channel_mean) / self.channel_std
        steps = torch.randint(
            0, self.config.diffusion_steps, (len(states),), device=self.device, generator=generator
        )
        noise = torch.randn(clean.shape, device=self.device, dtype=clean.dtype, generator=generator) * mask[..., None]
        alpha_bar = self.alpha_bars[steps][:, None, None]
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        prediction = self.denoiser(noisy, mask, steps, counts)
        squared = (prediction - noise).square().sum(-1)
        denoising = (squared * mask).sum() / (mask.sum() * clean.shape[-1])
        count_targets = counts - self.config.min_cardinality
        if bool(((count_targets < 0) | (count_targets >= self.count_logits.numel())).any()):
            raise ValueError("state cardinality is outside the configured support")
        count_nll = F.cross_entropy(
            self.count_logits[None, :].expand(len(states), -1), count_targets
        )
        return {"loss": denoising + count_nll, "denoising_mse": denoising, "count_nll": count_nll}

    @torch.no_grad()
    def sample(self, count: int, seed: int) -> list[gm.WeightedSetState]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        count_indices = torch.multinomial(
            F.softmax(self.count_logits, dim=0), count, replacement=True, generator=generator
        )
        counts = count_indices + self.config.min_cardinality
        max_n = max(int(counts.max()), 1)
        mask = torch.arange(max_n, device=self.device)[None, :] < counts[:, None]
        data_dim = 1 + self.config.feature_dim
        current = torch.randn((count, max_n, data_dim), device=self.device, generator=generator) * mask[..., None]
        for step in reversed(range(self.config.diffusion_steps)):
            steps = torch.full((count,), step, device=self.device, dtype=torch.long)
            prediction = self.denoiser(current, mask, steps, counts)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (current - self.betas[step] / torch.sqrt(1.0 - alpha_bar) * prediction) / torch.sqrt(alpha)
            if step:
                noise = torch.randn(current.shape, device=self.device, generator=generator)
                current = mean + torch.sqrt(self.betas[step]) * noise
            else:
                current = mean
            current = current * mask[..., None]
        current = current * self.channel_std + self.channel_mean
        states = []
        for row, cardinality in enumerate(counts.tolist()):
            if cardinality == 0:
                states.append(gm.empty_state(1.0, self.config.feature_dim, device=self.device, dtype=current.dtype))
                continue
            values = current[row, :cardinality]
            masses = F.softmax(values[:, 0], dim=0)
            states.append(
                gm.WeightedSetState(
                    masses,
                    values[:, 1:],
                    0.0,
                    1.0,
                    tuple(range(cardinality)),
                )
            )
        return states


def state_coordinates(state: gm.WeightedSetState) -> Tensor:
    masses = state.masses.float()
    normalized = masses / masses.sum()
    log_mass = normalized.log()
    log_mass = log_mass - log_mass.mean()
    return torch.cat((log_mass[:, None], state.features.float()), dim=-1)


def compute_normalization(states: Sequence[gm.WeightedSetState]) -> tuple[Tensor, Tensor]:
    coordinates = torch.cat([state_coordinates(state) for state in states], dim=0)
    mean = coordinates.mean(dim=0)
    std = coordinates.std(dim=0).clamp_min(1e-3)
    return mean, std


def pack_states(states: Sequence[gm.WeightedSetState], maximum: int, device: torch.device):
    max_n = max(state.n for state in states)
    if max_n > maximum:
        raise ValueError("state cardinality exceeds diffusion maximum")
    data_dim = 1 + states[0].feature_dim
    values = torch.zeros((len(states), max_n, data_dim), device=device)
    mask = torch.zeros((len(states), max_n), device=device, dtype=torch.bool)
    counts = torch.tensor([state.n for state in states], device=device, dtype=torch.long)
    for row, state in enumerate(states):
        values[row, : state.n] = state_coordinates(state).to(device)
        mask[row, : state.n] = True
    return values, mask, counts


def evaluate_loss(model: CardinalityDiffusion, states, batch_size: int) -> dict[str, float]:
    model.eval()
    totals = {}
    seen = 0
    with torch.no_grad():
        for batch in bench.batches(states, batch_size):
            terms = model.loss(batch)
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.cpu()) * len(batch)
            seen += len(batch)
    return {name: value / seen for name, value in totals.items()}


def train(args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    synthetic = bench.SyntheticConfig(
        min_cardinality=args.min_cardinality,
        max_cardinality=args.max_cardinality,
        cardinality_mean=args.cardinality_mean,
        split_alpha=args.split_alpha,
        interaction_strength=args.interaction_strength,
        separation_scale=args.separation_scale,
    )
    train_examples = bench.generate_dataset(args.train_size, args.seed + 1, synthetic)
    validation_examples = bench.generate_dataset(args.validation_size, args.seed + 2, synthetic)
    test_examples = bench.generate_dataset(args.test_size, args.seed + 3, synthetic)
    train_states = [example.state for example in train_examples]
    validation_states = [example.state for example in validation_examples]
    test_states = [example.state for example in test_examples]
    mean, std = compute_normalization(train_states)
    config = DiffusionConfig(
        min_cardinality=args.min_cardinality,
        max_cardinality=args.max_cardinality,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        diffusion_steps=args.diffusion_steps,
    )
    model = CardinalityDiffusion(config, mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    order_rng = random.Random(args.seed + 6)
    history = []
    best_loss = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_states)))
        order_rng.shuffle(order)
        total = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for indices in bench.batches(order, args.batch_size):
            batch = [train_states[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            terms = model.loss(batch)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(terms["loss"].detach().cpu()) * len(batch)
            seen += len(batch)
        validation = evaluate_loss(model, validation_states, args.batch_size)
        row = {
            "epoch": epoch,
            "train_objective": total / seen,
            "validation_objective": validation["loss"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_objective={row['train_objective']:.4f} "
            f"val_objective={row['validation_objective']:.4f} seconds={row['seconds']:.2f}",
            flush=True,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
    training_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"model_state": model.state_dict(), "diffusion_config": asdict(config),
         "synthetic_config": asdict(synthetic), "args": vars(args)},
        output / "model.pt",
    )
    test_objective = evaluate_loss(model, test_states, args.batch_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    sample_start = time.perf_counter()
    generated = model.sample(args.samples, args.seed + 40_000)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory = 0.0
    milliseconds = 1000.0 * (time.perf_counter() - sample_start) / args.samples
    metrics, cardinality_rows = bench.evaluate_samples(
        test_states[: args.samples], generated, synthetic.max_cardinality, args.seed + 50_000
    )
    metrics["milliseconds_per_sample"] = milliseconds
    metrics["peak_gpu_memory_mb"] = peak_memory
    result = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "model_type": "cardinality_diffusion",
        "objective_type": "ddpm_epsilon_mse_plus_count_nll",
        "synthetic_config": asdict(synthetic),
        "diffusion_config": asdict(config),
        "run_config": vars(args),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "best_validation_objective": best_loss,
        "test_objective": test_objective,
        "sample_metrics": metrics,
    }
    serializable = json.loads(json.dumps(result, default=str))
    (output / "summary.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    bench.write_csv(output / "cardinality.csv", cardinality_rows)
    bench.plot_cardinality(output / "cardinality.png", cardinality_rows)
    return serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gpu_diffusion"))
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-cardinality", type=int, default=2)
    parser.add_argument("--max-cardinality", type=int, default=8)
    parser.add_argument("--cardinality-mean", type=float, default=4.5)
    parser.add_argument("--split-alpha", type=float, default=2.0)
    parser.add_argument("--interaction-strength", type=float, default=1.0)
    parser.add_argument("--separation-scale", type=float, default=0.65)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
