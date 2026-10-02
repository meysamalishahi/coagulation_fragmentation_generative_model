#!/usr/bin/env python3
"""Continuous hierarchical weighted-set benchmark for the neural model.

The data distribution is independent of the model's corruption process.  It
recursively fragments a unit-mass parent into a variable-cardinality spatial
set and retains the latent genealogy only for diagnostics.  Training uses new
forward corruption histories each epoch and the exact marked-event objective
implemented in :mod:`gpu_model`.
"""

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

import gpu_model as gm


@dataclass(frozen=True)
class SyntheticConfig:
    min_cardinality: int = 2
    max_cardinality: int = 8
    cardinality_mean: float = 4.5
    split_alpha: float = 2.0
    interaction_strength: float = 1.0
    separation_scale: float = 0.65
    root_scale: float = 0.8
    reservoir_min: float = 0.03
    reservoir_max: float = 0.12
    feature_dim: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.min_cardinality <= self.max_cardinality:
            raise ValueError("invalid cardinality range")
        if self.split_alpha <= 0.0 or self.interaction_strength <= 0.0:
            raise ValueError("split_alpha and interaction_strength must be positive")
        if self.feature_dim != 2:
            raise ValueError("this spatial benchmark currently uses two dimensions")
        if not 0.0 < self.reservoir_min <= self.reservoir_max < 1.0:
            raise ValueError("reservoir range must lie inside (0, 1)")


@dataclass(frozen=True)
class HierarchicalExample:
    state: gm.WeightedSetState
    target_cardinality: int
    genealogy: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class RunConfig:
    model_type: str = "coagulation"
    seed: int = 20261001
    train_size: int = 4000
    validation_size: int = 800
    test_size: int = 1000
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    samples: int = 1000
    hidden_dim: int = 128
    transformer_layers: int = 3
    mixture_components: int = 6
    validation_every: int = 1

    def __post_init__(self) -> None:
        if self.model_type not in {"coagulation", "birth_only"}:
            raise ValueError("model_type must be 'coagulation' or 'birth_only'")


def _sample_cardinality(rng: np.random.Generator, config: SyntheticConfig) -> int:
    # A truncated over-dispersed count keeps all sizes represented while
    # avoiding a synthetic benchmark tied to one fixed cardinality.
    shape = 4.0
    rate = shape / max(config.cardinality_mean - config.min_cardinality, 0.25)
    extra_rate = rng.gamma(shape, 1.0 / rate)
    value = config.min_cardinality + int(rng.poisson(extra_rate))
    return int(np.clip(value, config.min_cardinality, config.max_cardinality))


def generate_hierarchical_example(rng: np.random.Generator, config: SyntheticConfig) -> HierarchicalExample:
    """Generate one exact mass- and centroid-preserving fragmentation tree."""
    target = _sample_cardinality(rng, config)
    root = rng.normal(0.0, config.root_scale, size=2)
    # Each leaf is [mass, x, y, lineage, depth].
    leaves: list[list[float]] = [[1.0, float(root[0]), float(root[1]), 0.0, 0.0]]
    genealogy: list[tuple[int, int, int]] = []
    next_lineage = 1
    while len(leaves) < target:
        masses = np.asarray([leaf[0] for leaf in leaves])
        # Interaction strength >1 preferentially refines massive clusters;
        # values below 1 spread refinement more uniformly across leaves.
        probabilities = masses ** config.interaction_strength
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(leaves), p=probabilities))
        mass, x, y, parent_lineage, depth = leaves.pop(index)
        rho = float(rng.beta(config.split_alpha, config.split_alpha))
        rho = float(np.clip(rho, 0.02, 0.98))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        depth_scale = 1.0 / math.sqrt(depth + 1.0)
        magnitude = float(
            rng.lognormal(
                mean=math.log(config.separation_scale * math.sqrt(mass) * depth_scale),
                sigma=0.25,
            )
        )
        separation = magnitude * np.asarray([math.cos(angle), math.sin(angle)])
        first_lineage, second_lineage = next_lineage, next_lineage + 1
        next_lineage += 2
        first_position = np.asarray([x, y]) - (1.0 - rho) * separation
        second_position = np.asarray([x, y]) + rho * separation
        leaves.extend(
            (
                [rho * mass, float(first_position[0]), float(first_position[1]), float(first_lineage), depth + 1.0],
                [
                    (1.0 - rho) * mass,
                    float(second_position[0]),
                    float(second_position[1]),
                    float(second_lineage),
                    depth + 1.0,
                ],
            )
        )
        genealogy.append((int(parent_lineage), first_lineage, second_lineage))

    rng.shuffle(leaves)
    normalized_masses = np.asarray([leaf[0] for leaf in leaves], dtype=np.float64)
    features = np.asarray([[leaf[1], leaf[2]] for leaf in leaves], dtype=np.float64)
    lineages = [int(leaf[3]) for leaf in leaves]
    reservoir = float(rng.uniform(config.reservoir_min, config.reservoir_max))
    state = gm.make_state(
        (1.0 - reservoir) * normalized_masses,
        features,
        reservoir,
        lineages=lineages,
        dtype=torch.float32,
    )
    return HierarchicalExample(state, target, tuple(genealogy))


def generate_dataset(size: int, seed: int, config: SyntheticConfig) -> list[HierarchicalExample]:
    rng = np.random.default_rng(seed)
    return [generate_hierarchical_example(rng, config) for _ in range(size)]


def make_histories(
    examples: Sequence[HierarchicalExample],
    corruption: gm.CorruptionConfig,
    seed: int,
) -> list[gm.ReverseHistory]:
    rng = random.Random(seed)
    return [gm.simulate_forward_history(example.state, corruption, rng) for example in examples]


def batches(values: Sequence, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.no_grad()
def evaluate_path_loss(
    model: gm.CoagulationModel,
    histories: Sequence[gm.ReverseHistory],
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    rng = random.Random(seed)
    for batch in batches(histories, batch_size):
        terms = model.path_loss(batch, rng)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * len(batch)
        count += len(batch)
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def merge_pair_metrics(
    model: gm.CoagulationModel,
    histories: Sequence[gm.ReverseHistory],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    probability = 0.0
    uniform_probability = 0.0
    candidate_recall = 0
    events = 0
    rows = [
        (event, history.leaf_budget, history.condition)
        for history in histories
        for event in history.merges
    ]
    for batch in batches(rows, batch_size):
        conditions = model._conditions([row[2] for row in batch], len(batch))
        outputs = model.merge_outputs(
            [row[0].state for row in batch],
            torch.tensor([row[0].time for row in batch], device=model.device),
            torch.tensor([row[1] for row in batch], device=model.device),
            conditions,
        )
        for output, row in zip(outputs, batch):
            target = torch.tensor(row[0].pair, device=model.device)
            matches = torch.all(output.pairs == target[None, :], dim=1).nonzero(as_tuple=False)
            events += 1
            if len(matches) == 1:
                candidate_recall += 1
                target_index = int(matches[0, 0])
                probability += float(output.pair_probabilities[target_index].cpu())
                uniform_probability += 1.0 / len(output.pairs)
                correct += int(int(output.pair_probabilities.argmax()) == target_index)
    denominator = max(events, 1)
    return {
        "merge_events": events,
        "merge_pair_accuracy": correct / denominator,
        "merge_target_probability": probability / denominator,
        "uniform_target_probability": uniform_probability / denominator,
        "candidate_recall": candidate_recall / denominator,
    }


@torch.no_grad()
def latent_genealogy_metrics(
    model: gm.CoagulationModel,
    examples: Sequence[HierarchicalExample],
    batch_size: int,
) -> dict[str, float]:
    """Rank true terminal sibling pairs without exposing genealogy to training."""
    model.eval()
    correct = 0
    target_probability = 0.0
    uniform_probability = 0.0
    target_candidate_recall = 0.0
    evaluated = 0
    for batch in batches(examples, batch_size):
        outputs = model.merge_outputs(
            [example.state for example in batch],
            torch.full((len(batch),), model.config.horizon, device=model.device),
            torch.tensor([example.state.n for example in batch], device=model.device),
        )
        for example, output in zip(batch, outputs):
            lineage_to_index = {lineage: index for index, lineage in enumerate(example.state.lineages)}
            true_pairs = {
                tuple(sorted((lineage_to_index[first], lineage_to_index[second])))
                for _, first, second in example.genealogy
                if first in lineage_to_index and second in lineage_to_index
            }
            if not true_pairs or len(output.pairs) == 0:
                continue
            pair_rows = [tuple(pair) for pair in output.pairs.cpu().tolist()]
            true_indices = [index for index, pair in enumerate(pair_rows) if pair in true_pairs]
            evaluated += 1
            target_candidate_recall += len(true_indices) / len(true_pairs)
            if true_indices:
                target_probability += float(output.pair_probabilities[true_indices].sum().cpu())
                uniform_probability += len(true_indices) / len(output.pairs)
                correct += int(int(output.pair_probabilities.argmax()) in true_indices)
    denominator = max(evaluated, 1)
    return {
        "sets_evaluated": evaluated,
        "terminal_sibling_top_pair_accuracy": correct / denominator,
        "terminal_sibling_probability": target_probability / denominator,
        "terminal_sibling_uniform_probability": uniform_probability / denominator,
        "terminal_sibling_candidate_recall": target_candidate_recall / denominator,
    }


def decoded_components(state: gm.WeightedSetState) -> tuple[np.ndarray, np.ndarray]:
    masses = state.masses.detach().float().cpu().numpy()
    features = state.features.detach().float().cpu().numpy()
    if masses.size:
        masses = masses / masses.sum()
    return masses, features


def invariant_summary(state: gm.WeightedSetState, max_cardinality: int) -> np.ndarray:
    masses, features = decoded_components(state)
    if len(masses) == 0:
        return np.zeros(7, dtype=np.float64)
    centroid = np.sum(masses[:, None] * features, axis=0)
    radius = np.sqrt(np.sum(masses * np.sum((features - centroid) ** 2, axis=1)))
    entropy = -np.sum(masses * np.log(np.maximum(masses, 1e-12)))
    if len(masses) >= 2:
        delta = features[:, None, :] - features[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=-1))
        pair_distances = distances[np.triu_indices(len(masses), 1)]
        pair_mean, pair_std = float(pair_distances.mean()), float(pair_distances.std())
    else:
        pair_mean = pair_std = 0.0
    return np.asarray(
        [
            len(masses) / max_cardinality,
            masses.max(),
            entropy,
            centroid[0],
            centroid[1],
            radius,
            pair_mean + pair_std,
        ],
        dtype=np.float64,
    )


def empirical_wasserstein(first: np.ndarray, second: np.ndarray, quantiles: int = 2000) -> float:
    if len(first) == 0 or len(second) == 0:
        return float("nan")
    probabilities = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.mean(np.abs(np.quantile(first, probabilities) - np.quantile(second, probabilities))))


def pair_distances(states: Sequence[gm.WeightedSetState]) -> np.ndarray:
    values = []
    for state in states:
        _, features = decoded_components(state)
        if len(features) >= 2:
            delta = features[:, None, :] - features[None, :, :]
            distances = np.sqrt(np.sum(delta * delta, axis=-1))
            values.extend(distances[np.triu_indices(len(features), 1)].tolist())
    return np.asarray(values)


def cardinality_tv(
    reference: Sequence[gm.WeightedSetState], generated: Sequence[gm.WeightedSetState], maximum: int
) -> tuple[float, list[dict[str, float | int | str]]]:
    # The final bin is overflow. Dropping out-of-support generated counts
    # before normalization would make a poor count model look artificially
    # better.
    overflow = maximum + 1
    reference_counts = np.bincount(
        [min(state.n, overflow) for state in reference], minlength=overflow + 1
    )
    generated_counts = np.bincount(
        [min(state.n, overflow) for state in generated], minlength=overflow + 1
    )
    reference_probability = reference_counts / reference_counts.sum()
    generated_probability = generated_counts / generated_counts.sum()
    rows = [
        {
            "cardinality": cardinality,
            "cardinality_label": f">{maximum}" if cardinality == overflow else str(cardinality),
            "reference_probability": float(reference_probability[cardinality]),
            "generated_probability": float(generated_probability[cardinality]),
        }
        for cardinality in range(overflow + 1)
    ]
    return float(0.5 * np.abs(reference_probability - generated_probability).sum()), rows


def mmd_rbf(first: np.ndarray, second: np.ndarray) -> float:
    combined = np.concatenate((first, second), axis=0)
    # Standardize invariant summaries so cardinality and spatial coordinates
    # contribute on comparable scales.
    scale = combined.std(axis=0)
    scale[scale < 1e-8] = 1.0
    first = (first - combined.mean(axis=0)) / scale
    second = (second - combined.mean(axis=0)) / scale
    probe = combined[: min(len(combined), 500)]
    probe = (probe - combined.mean(axis=0)) / scale
    distances = np.sum((probe[:, None, :] - probe[None, :, :]) ** 2, axis=-1)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0

    def kernel(left, right):
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
        return np.exp(-squared / max(2.0 * bandwidth, 1e-8))

    return float(kernel(first, first).mean() + kernel(second, second).mean() - 2.0 * kernel(first, second).mean())


def energy_distance(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, pairs: int = 100_000) -> float:
    def mean_distance(left, right, count):
        left_indices = rng.integers(0, len(left), size=count)
        right_indices = rng.integers(0, len(right), size=count)
        return float(np.linalg.norm(left[left_indices] - right[right_indices], axis=1).mean())

    count = min(pairs, max(10_000, len(first) * len(second)))
    return 2.0 * mean_distance(first, second, count) - mean_distance(first, first, count) - mean_distance(
        second, second, count
    )


def classifier_two_sample(first: np.ndarray, second: np.ndarray, seed: int) -> float:
    """Linear two-sample classifier accuracy on a held-out 40% split."""
    rng = np.random.default_rng(seed)
    features = np.concatenate((first, second), axis=0)
    labels = np.concatenate((np.zeros(len(first)), np.ones(len(second))))
    order = rng.permutation(len(features))
    features, labels = features[order], labels[order]
    split = int(0.6 * len(features))
    mean = features[:split].mean(axis=0)
    scale = features[:split].std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = np.concatenate(((features[:split] - mean) / scale, np.ones((split, 1))), axis=1)
    test = np.concatenate(((features[split:] - mean) / scale, np.ones((len(features) - split, 1))), axis=1)
    weights = np.zeros(train.shape[1])
    for _ in range(600):
        logits = np.clip(train @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = train.T @ (probabilities - labels[:split]) / split + 1e-3 * weights
        weights -= 0.1 * gradient
    predictions = test @ weights >= 0.0
    return float(np.mean(predictions == labels[split:]))


@torch.no_grad()
def sample_model(
    model: gm.CoagulationModel,
    count: int,
    corruption: gm.CorruptionConfig,
    seed: int,
    *,
    leaf_temperature: float = 1.0,
    merge_rate_scale: float = 1.0,
) -> tuple[list[gm.WeightedSetState], float, float]:
    model.eval()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    samples = [
        model.sample(
            1.0,
            corruption,
            rng=rng,
            leaf_temperature=leaf_temperature,
            merge_rate_scale=merge_rate_scale,
        )
        for _ in range(count)
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        peak_memory = torch.cuda.max_memory_allocated(model.device) / (1024.0**2)
    else:
        peak_memory = 0.0
    elapsed = time.perf_counter() - start
    return samples, 1000.0 * elapsed / count, peak_memory


def evaluate_samples(
    reference: Sequence[gm.WeightedSetState],
    generated: Sequence[gm.WeightedSetState],
    maximum: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    tv, cardinality_rows = cardinality_tv(reference, generated, maximum)
    reference_masses = np.concatenate([decoded_components(state)[0] for state in reference if state.n])
    generated_masses = np.concatenate([decoded_components(state)[0] for state in generated if state.n])
    reference_pairs = pair_distances(reference)
    generated_pairs = pair_distances(generated)
    reference_summary = np.stack([invariant_summary(state, maximum) for state in reference])
    generated_summary = np.stack([invariant_summary(state, maximum) for state in generated])
    residuals = np.asarray([state.conservation_residual() for state in generated])
    metrics = {
        "cardinality_tv": tv,
        "component_mass_w1": empirical_wasserstein(reference_masses, generated_masses),
        "pair_distance_w1": empirical_wasserstein(reference_pairs, generated_pairs),
        "summary_mmd": mmd_rbf(reference_summary, generated_summary),
        "summary_energy_distance": energy_distance(
            reference_summary, generated_summary, np.random.default_rng(seed)
        ),
        "classifier_two_sample_accuracy": classifier_two_sample(reference_summary, generated_summary, seed),
        "mass_residual_mean": float(residuals.mean()),
        "mass_residual_max": float(residuals.max()),
        "empty_sample_rate": float(np.mean([state.n == 0 for state in generated])),
    }
    return metrics, cardinality_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(path: Path, rows: Sequence[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(epochs, [row["train_loss"] for row in rows], label="train path NLL")
    axis.plot(epochs, [row["validation_loss"] for row in rows], label="validation path NLL")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("NLL per set")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cardinality(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray([row["cardinality"] for row in rows])
    width = 0.38
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar(values - width / 2, [row["reference_probability"] for row in rows], width, label="data")
    axis.bar(values + width / 2, [row["generated_probability"] for row in rows], width, label="model")
    axis.set_xticks(values, [str(row["cardinality_label"]) for row in rows])
    axis.set_xlabel("Final cardinality")
    axis.set_ylabel("Probability")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_model(config: gm.NeuralConfig, model_type: str) -> gm.CoagulationModel:
    if model_type == "coagulation":
        return gm.CoagulationModel(config)
    if model_type == "birth_only":
        return gm.BirthOnlyModel(config)
    raise ValueError(f"unknown model_type: {model_type}")


def train_and_evaluate(
    output: Path,
    synthetic: SyntheticConfig,
    corruption: gm.CorruptionConfig,
    run: RunConfig,
    device: torch.device,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    random.seed(run.seed)
    np.random.seed(run.seed)
    torch.manual_seed(run.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run.seed)

    train_examples = generate_dataset(run.train_size, run.seed + 1, synthetic)
    validation_examples = generate_dataset(run.validation_size, run.seed + 2, synthetic)
    test_examples = generate_dataset(run.test_size, run.seed + 3, synthetic)
    validation_histories = make_histories(validation_examples, corruption, run.seed + 4)
    test_histories = make_histories(test_examples, corruption, run.seed + 5)

    neural_config = gm.NeuralConfig(
        feature_dim=synthetic.feature_dim,
        max_components=corruption.max_components,
        hidden_dim=run.hidden_dim,
        embedding_dim=run.hidden_dim,
        transformer_layers=run.transformer_layers,
        attention_heads=4,
        mixture_components=run.mixture_components,
        horizon=corruption.horizon,
    )
    model = build_model(neural_config, run.model_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay
    )
    training_rows: list[dict[str, float]] = []
    best_validation = float("inf")
    best_state = None
    training_start = time.perf_counter()
    order_rng = random.Random(run.seed + 6)

    for epoch in range(1, run.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        order_rng.shuffle(order)
        train_total = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for batch_number, batch_indices in enumerate(batches(order, run.batch_size)):
            examples = [train_examples[index] for index in batch_indices]
            history_seed = run.seed * 1_000_003 + epoch * 10_007 + batch_number
            histories = make_histories(examples, corruption, history_seed)
            optimizer.zero_grad(set_to_none=True)
            terms = model.path_loss(histories, random.Random(history_seed + 1))
            if not bool(torch.isfinite(terms["loss"])):
                raise FloatingPointError("non-finite path loss")
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.gradient_clip)
            optimizer.step()
            train_total += float(terms["loss"].detach().cpu()) * len(examples)
            seen += len(examples)

        if epoch % run.validation_every == 0 or epoch == run.epochs:
            validation = evaluate_path_loss(
                model, validation_histories, run.batch_size, run.seed + 10_000 + epoch
            )
        else:
            validation = {"loss": float("nan")}
        train_loss = train_total / seen
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        training_rows.append(row)
        print(
            f"epoch={epoch:03d} train_nll={train_loss:.4f} "
            f"val_nll={validation['loss']:.4f} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if math.isfinite(validation["loss"]) and validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())

    training_seconds = time.perf_counter() - training_start
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "neural_config": asdict(neural_config),
            "synthetic_config": asdict(synthetic),
            "corruption_config": asdict(corruption),
            "run_config": asdict(run),
            "model_type": run.model_type,
        },
        output / "model.pt",
    )

    validation_metrics = evaluate_path_loss(
        model, validation_histories, run.batch_size, run.seed + 20_000
    )
    test_path_metrics = evaluate_path_loss(model, test_histories, run.batch_size, run.seed + 30_000)
    if run.model_type == "coagulation":
        pair_metrics = merge_pair_metrics(model, test_histories, run.batch_size)
        genealogy_metrics = latent_genealogy_metrics(model, test_examples, run.batch_size)
    else:
        pair_metrics = {"not_applicable": True}
        genealogy_metrics = {"not_applicable": True}
    generated, milliseconds_per_sample, peak_memory_mb = sample_model(
        model, run.samples, corruption, run.seed + 40_000
    )
    reference_states = [example.state for example in test_examples[: run.samples]]
    sample_metrics, cardinality_rows = evaluate_samples(
        reference_states, generated, synthetic.max_cardinality, run.seed + 50_000
    )
    sample_metrics["milliseconds_per_sample"] = milliseconds_per_sample
    sample_metrics["peak_gpu_memory_mb"] = peak_memory_mb

    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "synthetic_config": asdict(synthetic),
        "corruption_config": asdict(corruption),
        "run_config": asdict(run),
        "model_type": run.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "best_validation_loss": best_validation,
        "validation_path": validation_metrics,
        "test_path": test_path_metrics,
        "merge_metrics": pair_metrics,
        "latent_genealogy_metrics": genealogy_metrics,
        "sample_metrics": sample_metrics,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "training_history.csv", training_rows)
    write_csv(output / "cardinality.csv", cardinality_rows)
    plot_training(output / "training_curve.png", training_rows)
    plot_cardinality(output / "cardinality.png", cardinality_rows)
    return result


def reevaluate_checkpoint(output: Path, device: torch.device) -> dict:
    """Recreate deterministic test data/samples and refresh saved metrics."""
    summary_path = output / "summary.json"
    if not summary_path.exists() or not (output / "model.pt").exists():
        raise FileNotFoundError("evaluate-only requires summary.json and model.pt in the output directory")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    synthetic = SyntheticConfig(**previous["synthetic_config"])
    corruption = gm.CorruptionConfig(**previous["corruption_config"])
    run = RunConfig(**previous["run_config"])
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=False)
    neural_config = gm.NeuralConfig(**checkpoint["neural_config"])
    model = build_model(neural_config, run.model_type).to(device)
    model.load_state_dict(checkpoint["model_state"])

    test_examples = generate_dataset(run.test_size, run.seed + 3, synthetic)
    test_histories = make_histories(test_examples, corruption, run.seed + 5)
    test_path_metrics = evaluate_path_loss(model, test_histories, run.batch_size, run.seed + 30_000)
    if run.model_type == "coagulation":
        pair_metrics = merge_pair_metrics(model, test_histories, run.batch_size)
        genealogy_metrics = latent_genealogy_metrics(model, test_examples, run.batch_size)
    else:
        pair_metrics = {"not_applicable": True}
        genealogy_metrics = {"not_applicable": True}
    generated, milliseconds_per_sample, peak_memory_mb = sample_model(
        model, run.samples, corruption, run.seed + 40_000
    )
    reference_states = [example.state for example in test_examples[: run.samples]]
    sample_metrics, cardinality_rows = evaluate_samples(
        reference_states, generated, synthetic.max_cardinality, run.seed + 50_000
    )
    sample_metrics["milliseconds_per_sample"] = milliseconds_per_sample
    sample_metrics["peak_gpu_memory_mb"] = peak_memory_mb
    previous["test_path"] = test_path_metrics
    previous["merge_metrics"] = pair_metrics
    previous["latent_genealogy_metrics"] = genealogy_metrics
    previous["sample_metrics"] = sample_metrics
    previous["evaluation_revision"] = "cardinality-overflow-bin-v1"
    summary_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "cardinality.csv", cardinality_rows)
    plot_cardinality(output / "cardinality.png", cardinality_rows)
    return previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gpu_synthetic"))
    parser.add_argument("--model-type", choices=("coagulation", "birth_only"), default="coagulation")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-cardinality", type=int, default=2)
    parser.add_argument("--max-cardinality", type=int, default=8)
    parser.add_argument("--cardinality-mean", type=float, default=4.5)
    parser.add_argument("--split-alpha", type=float, default=2.0)
    parser.add_argument("--interaction-strength", type=float, default=1.0)
    parser.add_argument("--separation-scale", type=float, default=0.65)
    parser.add_argument("--forward-split-rate", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--max-components", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="reload output-dir/model.pt and refresh deterministic test/sample metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    synthetic = SyntheticConfig(
        min_cardinality=args.min_cardinality,
        max_cardinality=args.max_cardinality,
        cardinality_mean=args.cardinality_mean,
        split_alpha=args.split_alpha,
        interaction_strength=args.interaction_strength,
        separation_scale=args.separation_scale,
    )
    corruption = gm.CorruptionConfig(
        split_rate=0.0 if args.model_type == "birth_only" else args.forward_split_rate,
        gamma=args.gamma,
        max_components=args.max_components,
    )
    run = RunConfig(
        model_type=args.model_type,
        seed=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        samples=args.samples,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.evaluate_only:
        result = reevaluate_checkpoint(args.output_dir, device)
    else:
        result = train_and_evaluate(args.output_dir, synthetic, corruption, run, device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
