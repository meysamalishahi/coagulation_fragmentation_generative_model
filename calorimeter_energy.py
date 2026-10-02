#!/usr/bin/env python3
"""Independent calorimeter response and layer-energy generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import calorimeter_data as calo


LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class EnergyTransform:
    """Train-only standardization for response and stick-breaking variables."""

    incident_log_mean: float
    incident_log_std: float
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    layer_ids: tuple[int, ...]
    epsilon: float = 1e-6
    max_response: float = 4.0

    def __post_init__(self) -> None:
        if len(self.layer_ids) < 2:
            raise ValueError("at least two calorimeter layers are required")
        if len(self.target_mean) != len(self.layer_ids) or len(self.target_std) != len(self.layer_ids):
            raise ValueError("one target statistic is required per energy variable")
        if self.incident_log_std <= 0.0 or any(value <= 0.0 for value in self.target_std):
            raise ValueError("standard deviations must be positive")
        if self.epsilon <= 0.0 or self.max_response <= 0.0:
            raise ValueError("epsilon and max_response must be positive")

    @property
    def dimension(self) -> int:
        return len(self.layer_ids)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "EnergyTransform":
        converted = dict(values)
        for key in ("target_mean", "target_std", "layer_ids"):
            converted[key] = tuple(converted[key])
        return cls(**converted)

    def condition(self, incident_energies_mev: Tensor) -> Tensor:
        incident = incident_energies_mev.reshape(-1).clamp_min(self.epsilon)
        return ((incident.log() - self.incident_log_mean) / self.incident_log_std)[:, None]

    def _physical_targets(self, incident_energies_mev: Tensor, layer_energies_mev: Tensor) -> tuple[Tensor, Tensor]:
        incident = incident_energies_mev.reshape(-1)
        layers = layer_energies_mev.reshape(-1, self.dimension).clamp_min(0.0)
        total = layers.sum(dim=-1)
        nonempty = total > self.epsilon
        response = (total / incident.clamp_min(self.epsilon)).clamp_min(self.epsilon)
        variables = [response.log()]
        remaining = total
        for index in range(self.dimension - 1):
            ratio = torch.where(
                remaining > self.epsilon,
                layers[:, index] / remaining.clamp_min(self.epsilon),
                torch.full_like(remaining, 0.5),
            ).clamp(self.epsilon, 1.0 - self.epsilon)
            variables.append(torch.logit(ratio))
            remaining = (remaining - layers[:, index]).clamp_min(0.0)
        return torch.stack(variables, dim=-1), nonempty

    def encode_targets(
        self, incident_energies_mev: Tensor, layer_energies_mev: Tensor
    ) -> tuple[Tensor, Tensor]:
        values, nonempty = self._physical_targets(incident_energies_mev, layer_energies_mev)
        mean = values.new_tensor(self.target_mean)
        std = values.new_tensor(self.target_std)
        return (values - mean) / std, nonempty

    def decode_targets(
        self, incident_energies_mev: Tensor, standardized_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = standardized_targets.reshape(-1, self.dimension)
        mean = values.new_tensor(self.target_mean)
        std = values.new_tensor(self.target_std)
        physical = values * std + mean
        response = physical[:, 0].exp().clamp(max=self.max_response)
        total = response * incident_energies_mev.reshape(-1).to(values)
        ratios = torch.sigmoid(physical[:, 1:])
        remaining = total
        layers = []
        for index in range(self.dimension - 1):
            energy = ratios[:, index] * remaining
            layers.append(energy)
            remaining = (remaining - energy).clamp_min(0.0)
        layers.append(remaining)
        return total, torch.stack(layers, dim=-1)


@dataclass(frozen=True)
class EnergyModelConfig:
    dimension: int = 5
    hidden_dim: int = 676
    hidden_layers: int = 5
    mixture_components: int = 16
    min_scale: float = 1e-3

    def __post_init__(self) -> None:
        if min(self.dimension, self.hidden_dim, self.hidden_layers, self.mixture_components) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive")


@dataclass
class EnergyMixtureParameters:
    nonempty_logit: Tensor
    mixture_logits: Tensor
    location: Tensor
    scale: Tensor


class LayerEnergyModel(nn.Module):
    """Conditional Gaussian mixture for total response and layer fractions."""

    def __init__(self, config: EnergyModelConfig):
        super().__init__()
        self.config = config
        output_dim = 1 + config.mixture_components * (1 + 2 * config.dimension)
        modules: list[nn.Module] = [nn.Linear(1, config.hidden_dim), nn.SiLU()]
        for _ in range(config.hidden_layers - 1):
            modules.extend((nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU()))
        modules.append(nn.Linear(config.hidden_dim, output_dim))
        self.network = nn.Sequential(*modules)

    def parameters_for(self, condition: Tensor) -> EnergyMixtureParameters:
        condition = condition.reshape(-1, 1)
        raw = self.network(condition)
        nonempty_logit = raw[:, 0]
        mixture = raw[:, 1:].reshape(
            len(condition), self.config.mixture_components, 1 + 2 * self.config.dimension
        )
        start = 1
        stop = start + self.config.dimension
        return EnergyMixtureParameters(
            nonempty_logit,
            mixture[..., 0],
            mixture[..., start:stop],
            F.softplus(mixture[..., stop:]) + self.config.min_scale,
        )

    def nll(self, condition: Tensor, targets: Tensor, nonempty: Tensor) -> dict[str, Tensor]:
        parameters = self.parameters_for(condition)
        targets = targets.to(parameters.location)
        nonempty = nonempty.to(device=targets.device, dtype=torch.bool)
        standardized = (targets[:, None, :] - parameters.location) / parameters.scale
        component_log_prob = -0.5 * (standardized.square() + LOG_2PI).sum(dim=-1)
        component_log_prob -= parameters.scale.log().sum(dim=-1)
        continuous_nll = -torch.logsumexp(
            F.log_softmax(parameters.mixture_logits, dim=-1) + component_log_prob,
            dim=-1,
        )
        empty_nll = F.binary_cross_entropy_with_logits(
            parameters.nonempty_logit, nonempty.to(targets.dtype), reduction="none"
        )
        masked_continuous = torch.where(nonempty, continuous_nll, torch.zeros_like(continuous_nll))
        loss = empty_nll + masked_continuous
        denominator = nonempty.sum().clamp_min(1)
        return {
            "loss": loss.mean(),
            "empty_nll": empty_nll.mean(),
            "continuous_nll": masked_continuous.sum() / denominator,
        }

    @torch.no_grad()
    def sample_standardized(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        parameters = self.parameters_for(condition)
        nonempty = torch.bernoulli(torch.sigmoid(parameters.nonempty_logit)).to(torch.bool)
        mixtures = torch.distributions.Categorical(logits=parameters.mixture_logits).sample()
        rows = torch.arange(len(condition), device=condition.device)
        location = parameters.location[rows, mixtures]
        scale = parameters.scale[rows, mixtures]
        values = location + scale * torch.randn_like(location)
        return values, nonempty

    @torch.no_grad()
    def sample(
        self, incident_energies_mev: Tensor, transform: EnergyTransform
    ) -> tuple[Tensor, Tensor, Tensor]:
        condition = transform.condition(incident_energies_mev).to(next(self.parameters()))
        values, nonempty = self.sample_standardized(condition)
        total, layers = transform.decode_targets(incident_energies_mev.to(values), values)
        total = torch.where(nonempty, total, torch.zeros_like(total))
        layers = torch.where(nonempty[:, None], layers, torch.zeros_like(layers))
        return total, layers, nonempty


def layer_energies(showers_mev: np.ndarray, geometry: calo.CalorimeterGeometry) -> np.ndarray:
    """Sum dense voxel energies into the geometry's ordered physical layers."""
    showers = np.asarray(showers_mev)
    if showers.ndim != 2 or showers.shape[1] != geometry.voxel_count:
        raise ValueError("showers do not match the supplied calorimeter geometry")
    layer_ids = tuple(int(value) for value in np.unique(geometry.layer_ids))
    return np.stack(
        [showers[:, geometry.layer_ids == layer_id].sum(axis=1) for layer_id in layer_ids],
        axis=1,
    )


def load_layer_energy_arrays(
    path: Path,
    geometry: calo.CalorimeterGeometry,
    *,
    start: int = 0,
    count: int | None = None,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only the compact incident/layer-energy training representation."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    metadata = calo.inspect_hdf5(path)
    stop = metadata["events"] if count is None else min(metadata["events"], start + count)
    if start < 0 or start >= stop:
        raise ValueError("requested event slice is empty")
    incidents: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        for first in range(start, stop, chunk_size):
            last = min(stop, first + chunk_size)
            showers = np.asarray(handle["showers"][first:last], dtype=np.float32)
            incidents.append(
                np.asarray(handle["incident_energies"][first:last], dtype=np.float32).reshape(-1)
            )
            layers.append(layer_energies(np.maximum(showers, 0.0), geometry))
    return np.concatenate(incidents), np.concatenate(layers)


def fit_energy_transform(
    incident_energies_mev: np.ndarray,
    layer_energies_mev: np.ndarray,
    layer_ids: Sequence[int],
    *,
    epsilon: float = 1e-6,
) -> EnergyTransform:
    """Fit all preprocessing statistics using training data only."""
    incident = torch.as_tensor(incident_energies_mev, dtype=torch.float64).reshape(-1)
    layers = torch.as_tensor(layer_energies_mev, dtype=torch.float64)
    if layers.ndim != 2 or len(layers) != len(incident) or layers.shape[1] != len(layer_ids):
        raise ValueError("incident and layer-energy arrays have incompatible shapes")
    if bool(torch.any(incident <= 0.0)):
        raise ValueError("incident energies must be positive")
    provisional = EnergyTransform(
        float(incident.log().mean()),
        float(incident.log().std(unbiased=False).clamp_min(epsilon)),
        tuple(0.0 for _ in layer_ids),
        tuple(1.0 for _ in layer_ids),
        tuple(int(value) for value in layer_ids),
        epsilon,
    )
    values, nonempty = provisional._physical_targets(incident, layers)
    selected = values[nonempty]
    if not len(selected):
        raise ValueError("training data contains no nonempty calorimeter showers")
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(epsilon)
    return EnergyTransform(
        provisional.incident_log_mean,
        provisional.incident_log_std,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
        provisional.layer_ids,
        epsilon,
    )
