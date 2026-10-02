#!/usr/bin/env python3
"""Neural mass-conserving coagulation--fragmentation model.

This module implements the trainable reverse model and the data corruption
path described in the paper.  It deliberately contains no benchmark-specific
code; synthetic and real datasets can construct ``WeightedSetState`` objects
and reuse the same likelihood and sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class WeightedSetState:
    """One unordered weighted set and its reservoir.

    Lineage identifiers are bookkeeping only.  They are never passed to the
    neural network.
    """

    masses: Tensor
    features: Tensor
    reservoir: float
    total_mass: float
    lineages: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.masses.ndim != 1:
            raise ValueError("masses must have shape [N]")
        if self.features.ndim != 2 or self.features.shape[0] != self.masses.shape[0]:
            raise ValueError("features must have shape [N, feature_dim]")
        if self.lineages and len(self.lineages) != len(self.masses):
            raise ValueError("one lineage identifier is required per component")
        if self.reservoir < -1e-8 or self.total_mass <= 0:
            raise ValueError("reservoir must be nonnegative and total_mass positive")

    @property
    def n(self) -> int:
        return int(self.masses.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "WeightedSetState":
        return WeightedSetState(
            self.masses.to(device=device, dtype=dtype),
            self.features.to(device=device, dtype=dtype),
            self.reservoir,
            self.total_mass,
            self.lineages,
        )

    def permute(self, order: Tensor) -> "WeightedSetState":
        indices = order.detach().cpu().tolist()
        return WeightedSetState(
            self.masses[order],
            self.features[order],
            self.reservoir,
            self.total_mass,
            tuple(self.lineages[i] for i in indices) if self.lineages else (),
        )

    def conservation_residual(self) -> float:
        represented = self.reservoir + float(self.masses.detach().double().sum().cpu())
        return abs(represented - self.total_mass) / max(self.total_mass, 1e-12)


def make_state(
    masses: Tensor | Sequence[float],
    features: Tensor | Sequence[Sequence[float]],
    reservoir: float,
    *,
    lineages: Sequence[int] | None = None,
    dtype: torch.dtype = torch.float64,
) -> WeightedSetState:
    masses_tensor = torch.as_tensor(masses, dtype=dtype)
    features_tensor = torch.as_tensor(features, dtype=dtype)
    if features_tensor.ndim == 1:
        features_tensor = features_tensor[:, None]
    ids = tuple(range(len(masses_tensor))) if lineages is None else tuple(lineages)
    total = float(masses_tensor.double().sum()) + float(reservoir)
    return WeightedSetState(masses_tensor, features_tensor, float(reservoir), total, ids)


def empty_state(total_mass: float, feature_dim: int, *, device: torch.device, dtype: torch.dtype) -> WeightedSetState:
    return WeightedSetState(
        torch.empty(0, device=device, dtype=dtype),
        torch.empty((0, feature_dim), device=device, dtype=dtype),
        float(total_mass),
        float(total_mass),
        (),
    )


def _replace_reservoir(state: WeightedSetState, masses: Tensor) -> WeightedSetState:
    component_mass = float(masses.detach().double().sum().cpu())
    reservoir = state.total_mass - component_mass
    rounding_tolerance = max(
        1e-12,
        16.0 * torch.finfo(masses.dtype).eps * state.total_mass,
    )
    if reservoir < -rounding_tolerance:
        raise ValueError(
            f"flow created component mass {component_mass} above total mass {state.total_mass}"
        )
    if reservoir < 0.0:
        # Float32 multiplication can overshoot the closed conservative
        # boundary by a few ULPs. Project that boundary case back onto the
        # simplex; larger conservation failures are rejected above.
        masses = masses * (state.total_mass / component_mass)
        reservoir = 0.0
    if abs(reservoir) < 1e-12 * state.total_mass:
        reservoir = 0.0
    return WeightedSetState(masses, state.features, reservoir, state.total_mass, state.lineages)


def flow_state(state: WeightedSetState, integrated_gamma: float, *, reverse: bool) -> WeightedSetState:
    """Apply the closed-form conservative flow over one time interval."""
    if state.n == 0 or state.reservoir <= 0.0 or integrated_gamma == 0.0:
        return state
    component_mass = float(state.masses.detach().double().sum().cpu())
    if component_mass <= 0.0:
        return state
    ratio = component_mass / state.reservoir
    exponent = integrated_gamma if reverse else -integrated_gamma
    new_ratio = ratio * math.exp(exponent)
    new_component_mass = state.total_mass * new_ratio / (1.0 + new_ratio)
    masses = state.masses * (new_component_mass / component_mass)
    return _replace_reservoir(state, masses)


def split_component(
    state: WeightedSetState,
    index: int,
    rho: float,
    separation: Tensor,
    child_lineages: tuple[int, int],
) -> WeightedSetState:
    mass = state.masses[index]
    feature = state.features[index]
    rho_tensor = mass.new_tensor(rho)
    first_mass = rho_tensor * mass
    second_mass = (1.0 - rho_tensor) * mass
    first_feature = feature - (1.0 - rho_tensor) * separation
    second_feature = feature + rho_tensor * separation
    keep = torch.ones(state.n, dtype=torch.bool, device=state.masses.device)
    keep[index] = False
    masses = torch.cat((state.masses[keep], first_mass[None], second_mass[None]))
    features = torch.cat((state.features[keep], first_feature[None], second_feature[None]), dim=0)
    old_ids = state.lineages or tuple(range(state.n))
    ids = tuple(identifier for j, identifier in enumerate(old_ids) if j != index) + child_lineages
    return WeightedSetState(masses, features, state.reservoir, state.total_mass, ids)


def delete_component(state: WeightedSetState, index: int) -> tuple[WeightedSetState, float, Tensor, int]:
    mass = float(state.masses[index].detach().cpu())
    feature = state.features[index].clone()
    lineage = state.lineages[index]
    keep = torch.ones(state.n, dtype=torch.bool, device=state.masses.device)
    keep[index] = False
    return (
        WeightedSetState(
            state.masses[keep],
            state.features[keep],
            state.reservoir + mass,
            state.total_mass,
            tuple(identifier for j, identifier in enumerate(state.lineages) if j != index),
        ),
        mass,
        feature,
        lineage,
    )


def birth_component(state: WeightedSetState, mass: float | Tensor, feature: Tensor, lineage: int) -> WeightedSetState:
    mass_tensor = torch.as_tensor(mass, device=state.masses.device, dtype=state.masses.dtype).reshape(())
    mass_value = float(mass_tensor.detach().cpu())
    rounding_tolerance = max(
        1e-7,
        16.0 * torch.finfo(state.masses.dtype).eps * state.total_mass,
    )
    if not 0.0 < mass_value <= state.reservoir + rounding_tolerance:
        raise ValueError(f"birth mass {mass_value} is outside (0, reservoir={state.reservoir}]")
    # A forward deletion stores a tensor value while the reservoir is updated
    # through host floats. Clamp only sub-ULP reconstruction drift at the
    # boundary; materially invalid births still raise above.
    if mass_value > state.reservoir:
        mass_tensor = mass_tensor.new_tensor(state.reservoir)
        mass_value = state.reservoir
    return WeightedSetState(
        torch.cat((state.masses, mass_tensor[None])),
        torch.cat((state.features, feature.to(state.features)[None]), dim=0),
        max(0.0, state.reservoir - mass_value),
        state.total_mass,
        state.lineages + (lineage,),
    )


def merge_components(
    state: WeightedSetState,
    first: int,
    second: int,
    parent_lineage: int | None = None,
) -> WeightedSetState:
    if first == second:
        raise ValueError("a merge requires two distinct components")
    first, second = sorted((first, second))
    mass = state.masses[first] + state.masses[second]
    feature = (
        state.masses[first] * state.features[first] + state.masses[second] * state.features[second]
    ) / mass
    keep = torch.ones(state.n, dtype=torch.bool, device=state.masses.device)
    keep[first] = False
    keep[second] = False
    parent = max(state.lineages, default=-1) + 1 if parent_lineage is None else parent_lineage
    ids = tuple(identifier for j, identifier in enumerate(state.lineages) if bool(keep[j])) + (parent,)
    return WeightedSetState(
        torch.cat((state.masses[keep], mass[None])),
        torch.cat((state.features[keep], feature[None]), dim=0),
        state.reservoir,
        state.total_mass,
        ids,
    )


@dataclass(frozen=True)
class CorruptionConfig:
    horizon: float = 1.0
    switch_time: float = 0.5
    split_rate: float = 1.5
    gamma: float = 0.8
    max_components: int = 32
    split_beta: float = 2.0
    separation_scale: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < self.switch_time < self.horizon:
            raise ValueError("switch_time must lie strictly inside the horizon")
        if self.split_rate < 0.0 or self.gamma < 0.0 or self.max_components < 1:
            raise ValueError("rates must be nonnegative and max_components positive")

    @property
    def nucleation_duration(self) -> float:
        return self.horizon - self.switch_time


@dataclass(frozen=True)
class ForwardSplit:
    time: float
    parent_lineage: int
    child_lineages: tuple[int, int]


@dataclass(frozen=True)
class ForwardDeletion:
    time: float
    mass: float
    feature: Tensor
    lineage: int


@dataclass(frozen=True)
class BirthObservation:
    state: WeightedSetState
    time: float
    leaf_budget: int
    mass: float
    feature: Tensor


@dataclass(frozen=True)
class MergeObservation:
    state: WeightedSetState
    time: float
    pair: tuple[int, int]


@dataclass(frozen=True)
class MergeInterval:
    state_at_start: WeightedSetState
    start: float
    end: float


@dataclass(frozen=True)
class ReverseHistory:
    total_mass: float
    leaf_budget: int
    births: tuple[BirthObservation, ...]
    merges: tuple[MergeObservation, ...]
    merge_intervals: tuple[MergeInterval, ...]
    endpoint: WeightedSetState
    gamma: float
    condition: Tensor | None = None


def _canonical_separation(rng: random.Random, feature_dim: int, scale: float, reference: Tensor) -> Tensor:
    values = [rng.gauss(0.0, scale) for _ in range(feature_dim)]
    if all(abs(value) < 1e-12 for value in values):
        values[0] = scale
    for value in values:
        if abs(value) > 1e-12:
            if value < 0.0:
                values = [-entry for entry in values]
            break
    return reference.new_tensor(values)


def simulate_forward_history(
    initial: WeightedSetState,
    config: CorruptionConfig,
    rng: random.Random,
    condition: Tensor | None = None,
) -> ReverseHistory:
    """Simulate Algorithm 1 and return its reversed sufficient record."""
    if initial.n > config.max_components:
        raise ValueError("initial cardinality exceeds max_components")
    if not initial.lineages:
        initial = WeightedSetState(
            initial.masses,
            initial.features,
            initial.reservoir,
            initial.total_mass,
            tuple(range(initial.n)),
        )
    state = initial
    next_lineage = max(state.lineages, default=-1) + 1
    split_events: list[ForwardSplit] = []
    time = 0.0

    while time < config.switch_time and 0 < state.n < config.max_components and config.split_rate > 0.0:
        wait = rng.expovariate(state.n * config.split_rate)
        event_time = time + wait
        if event_time >= config.switch_time:
            break
        state = flow_state(state, config.gamma * (event_time - time), reverse=False)
        time = event_time
        parent_index = rng.randrange(state.n)
        parent_lineage = state.lineages[parent_index]
        rho = rng.betavariate(config.split_beta, config.split_beta)
        separation = _canonical_separation(rng, state.feature_dim, config.separation_scale, state.features)
        children = (next_lineage, next_lineage + 1)
        next_lineage += 2
        state = split_component(state, parent_index, rho, separation, children)
        split_events.append(ForwardSplit(time, parent_lineage, children))

    state = flow_state(state, config.gamma * (config.switch_time - time), reverse=False)
    time = config.switch_time
    leaf_budget = state.n
    deletion_schedule = sorted(
        (rng.uniform(config.switch_time, config.horizon), lineage) for lineage in state.lineages
    )
    deletion_events: list[ForwardDeletion] = []
    for deletion_time, lineage in deletion_schedule:
        state = flow_state(state, config.gamma * (deletion_time - time), reverse=False)
        time = deletion_time
        index = state.lineages.index(lineage)
        state, mass, feature, deleted_lineage = delete_component(state, index)
        deletion_events.append(ForwardDeletion(time, mass, feature, deleted_lineage))
    state = flow_state(state, config.gamma * (config.horizon - time), reverse=False)
    if state.n != 0 or state.conservation_residual() > 1e-9:
        raise RuntimeError("forward corruption did not reach the conservative empty state")

    reverse = empty_state(
        initial.total_mass,
        initial.feature_dim,
        device=initial.masses.device,
        dtype=initial.masses.dtype,
    )
    births: list[BirthObservation] = []
    reverse_time = 0.0
    reverse_deletions = sorted(deletion_events, key=lambda event: config.horizon - event.time)
    for event in reverse_deletions:
        event_time = config.horizon - event.time
        reverse = flow_state(reverse, config.gamma * (event_time - reverse_time), reverse=True)
        births.append(BirthObservation(reverse, event_time, leaf_budget, event.mass, event.feature))
        reverse = birth_component(reverse, event.mass, event.feature, event.lineage)
        reverse_time = event_time
    reverse = flow_state(
        reverse,
        config.gamma * (config.nucleation_duration - reverse_time),
        reverse=True,
    )
    reverse_time = config.nucleation_duration

    merges: list[MergeObservation] = []
    intervals: list[MergeInterval] = []
    reverse_splits = sorted(split_events, key=lambda event: config.horizon - event.time)
    for event in reverse_splits:
        event_time = config.horizon - event.time
        intervals.append(MergeInterval(reverse, reverse_time, event_time))
        reverse = flow_state(reverse, config.gamma * (event_time - reverse_time), reverse=True)
        first = reverse.lineages.index(event.child_lineages[0])
        second = reverse.lineages.index(event.child_lineages[1])
        merges.append(MergeObservation(reverse, event_time, tuple(sorted((first, second)))))
        reverse = merge_components(reverse, first, second, event.parent_lineage)
        reverse_time = event_time
    intervals.append(MergeInterval(reverse, reverse_time, config.horizon))
    reverse = flow_state(reverse, config.gamma * (config.horizon - reverse_time), reverse=True)

    return ReverseHistory(
        initial.total_mass,
        leaf_budget,
        tuple(births),
        tuple(merges),
        tuple(intervals),
        reverse,
        config.gamma,
        condition,
    )


@dataclass(frozen=True)
class NeuralConfig:
    feature_dim: int
    max_components: int = 32
    hidden_dim: int = 128
    embedding_dim: int = 128
    transformer_layers: int = 2
    attention_heads: int = 4
    mixture_components: int = 6
    condition_dim: int = 0
    horizon: float = 1.0
    k_max: float = 4.0
    sparse_k: int | None = None
    min_scale: float = 1e-3

    def __post_init__(self) -> None:
        if self.feature_dim < 1 or self.max_components < 1:
            raise ValueError("feature_dim and max_components must be positive")
        if self.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if self.mixture_components < 1 or self.k_max <= 0.0:
            raise ValueError("mixture_components and k_max must be positive")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3):
        super().__init__()
        modules: list[nn.Module] = []
        width = input_dim
        for _ in range(layers - 1):
            modules.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        modules.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


@dataclass
class EncodedBatch:
    component_embeddings: Tensor
    pooled: Tensor
    padding_mask: Tensor


class SetEncoder(nn.Module):
    """Permutation-equivariant set encoder with a learned global token."""

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        context_dim = 4 + config.condition_dim
        element_dim = config.feature_dim + 1 + context_dim
        self.element_mlp = MLP(element_dim, config.hidden_dim, config.embedding_dim)
        self.context_mlp = MLP(context_dim, config.hidden_dim, config.embedding_dim, layers=2)
        self.null_token = nn.Parameter(torch.zeros(config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.embedding_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.embedding_dim),
        )

    def forward(
        self,
        states: Sequence[WeightedSetState],
        times: Tensor,
        leaf_budgets: Tensor,
        conditions: Tensor,
    ) -> EncodedBatch:
        if not states:
            raise ValueError("cannot encode an empty batch")
        device = times.device
        dtype = times.dtype
        batch_size = len(states)
        max_n = max((state.n for state in states), default=0)
        features = torch.zeros((batch_size, max_n, self.config.feature_dim), device=device, dtype=dtype)
        mass_fraction = torch.zeros((batch_size, max_n, 1), device=device, dtype=dtype)
        component_padding = torch.ones((batch_size, max_n), device=device, dtype=torch.bool)
        reservoir_fraction = torch.empty((batch_size, 1), device=device, dtype=dtype)
        log_total_mass = torch.empty((batch_size, 1), device=device, dtype=dtype)
        for row, state in enumerate(states):
            if state.feature_dim != self.config.feature_dim:
                raise ValueError("state feature dimension does not match model")
            if state.n:
                features[row, : state.n] = state.features.to(device=device, dtype=dtype)
                mass_fraction[row, : state.n, 0] = state.masses.to(device=device, dtype=dtype) / state.total_mass
                component_padding[row, : state.n] = False
            reservoir_fraction[row, 0] = state.reservoir / state.total_mass
            log_total_mass[row, 0] = math.log(state.total_mass)
        time_column = (times / self.config.horizon).reshape(-1, 1)
        leaf_fraction = (leaf_budgets / self.config.max_components).reshape(-1, 1)
        context = torch.cat((reservoir_fraction, time_column, log_total_mass, leaf_fraction, conditions), dim=-1)
        if max_n:
            repeated_context = context[:, None, :].expand(-1, max_n, -1)
            elements = self.element_mlp(torch.cat((features, mass_fraction, repeated_context), dim=-1))
        else:
            elements = torch.empty((batch_size, 0, self.config.embedding_dim), device=device, dtype=dtype)
        global_token = self.null_token.to(dtype=dtype)[None, None, :] + self.context_mlp(context)[:, None, :]
        tokens = torch.cat((global_token, elements), dim=1)
        padding = torch.cat(
            (torch.zeros((batch_size, 1), device=device, dtype=torch.bool), component_padding), dim=1
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding)
        return EncodedBatch(encoded[:, 1:], encoded[:, 0], component_padding)


@dataclass
class BirthParameters:
    mixture_logits: Tensor
    alpha: Tensor
    beta: Tensor
    location: Tensor
    scale: Tensor


@dataclass
class MergeOutput:
    pairs: Tensor
    pair_logits: Tensor
    pair_probabilities: Tensor
    total_rate: Tensor
    intensities: Tensor


class CoagulationModel(nn.Module):
    """Complete trainable reverse event model from Section 8 of the paper."""

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        self.encoder = SetEncoder(config)
        self.leaf_head = MLP(1 + config.condition_dim, config.hidden_dim, config.max_components + 1)
        mixture_width = 3 + 2 * config.feature_dim
        self.birth_head = MLP(
            config.embedding_dim,
            config.hidden_dim,
            config.mixture_components * mixture_width,
        )
        pair_input_dim = 3 * config.embedding_dim
        self.pair_head = MLP(pair_input_dim, config.hidden_dim, 1)
        self.rate_head = MLP(config.embedding_dim, config.hidden_dim, 1, layers=2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _conditions(self, conditions: Sequence[Tensor | None], count: int) -> Tensor:
        if self.config.condition_dim == 0:
            return torch.empty((count, 0), device=self.device, dtype=self.dtype)
        rows = []
        for condition in conditions:
            if condition is None:
                raise ValueError("this model requires a conditioning vector")
            rows.append(condition.to(device=self.device, dtype=self.dtype).reshape(self.config.condition_dim))
        return torch.stack(rows)

    def leaf_logits(self, total_masses: Tensor, conditions: Tensor | None = None) -> Tensor:
        total_masses = total_masses.to(device=self.device, dtype=self.dtype).reshape(-1, 1)
        if conditions is None:
            conditions = torch.empty((len(total_masses), 0), device=self.device, dtype=self.dtype)
        else:
            conditions = conditions.to(device=self.device, dtype=self.dtype)
        return self.leaf_head(torch.cat((total_masses.log(), conditions), dim=-1))

    def encode(
        self,
        states: Sequence[WeightedSetState],
        times: Tensor,
        leaf_budgets: Tensor,
        conditions: Tensor | None = None,
    ) -> EncodedBatch:
        count = len(states)
        if conditions is None:
            conditions = torch.empty((count, 0), device=self.device, dtype=self.dtype)
        return self.encoder(
            states,
            times.to(device=self.device, dtype=self.dtype),
            leaf_budgets.to(device=self.device, dtype=self.dtype),
            conditions.to(device=self.device, dtype=self.dtype),
        )

    def birth_parameters(self, pooled: Tensor) -> BirthParameters:
        batch = pooled.shape[0]
        components = self.config.mixture_components
        feature_dim = self.config.feature_dim
        raw = self.birth_head(pooled).reshape(batch, components, 3 + 2 * feature_dim)
        return BirthParameters(
            raw[..., 0],
            F.softplus(raw[..., 1]) + 1e-3,
            F.softplus(raw[..., 2]) + 1e-3,
            raw[..., 3 : 3 + feature_dim],
            F.softplus(raw[..., 3 + feature_dim :]) + self.config.min_scale,
        )

    def birth_log_prob(
        self,
        states: Sequence[WeightedSetState],
        times: Tensor,
        leaf_budgets: Tensor,
        masses: Tensor,
        features: Tensor,
        conditions: Tensor | None = None,
    ) -> Tensor:
        encoded = self.encode(states, times, leaf_budgets, conditions)
        parameters = self.birth_parameters(encoded.pooled)
        reservoir = masses.new_tensor([state.reservoir for state in states]).to(self.device, self.dtype)
        masses = masses.to(self.device, self.dtype)
        features = features.to(self.device, self.dtype)
        omega = (masses / reservoir).clamp(1e-6, 1.0 - 1e-6)
        omega_expanded = omega[:, None]
        beta_log_prob = (
            (parameters.alpha - 1.0) * omega_expanded.log()
            + (parameters.beta - 1.0) * torch.log1p(-omega_expanded)
            - torch.lgamma(parameters.alpha)
            - torch.lgamma(parameters.beta)
            + torch.lgamma(parameters.alpha + parameters.beta)
        )
        standardized = (features[:, None, :] - parameters.location) / parameters.scale
        normal_log_prob = -0.5 * (standardized.square() + LOG_2PI).sum(-1) - parameters.scale.log().sum(-1)
        mixture_log_prob = F.log_softmax(parameters.mixture_logits, dim=-1)
        return torch.logsumexp(mixture_log_prob + beta_log_prob + normal_log_prob, dim=-1) - reservoir.log()

    @torch.no_grad()
    def sample_birth(
        self,
        state: WeightedSetState,
        time: float,
        leaf_budget: int,
        condition: Tensor | None = None,
    ) -> tuple[float, Tensor]:
        conditions = self._conditions([condition], 1)
        encoded = self.encode(
            [state],
            torch.tensor([time], device=self.device),
            torch.tensor([leaf_budget], device=self.device),
            conditions,
        )
        parameters = self.birth_parameters(encoded.pooled)
        mixture = torch.distributions.Categorical(logits=parameters.mixture_logits[0]).sample()
        index = int(mixture)
        omega = torch.distributions.Beta(parameters.alpha[0, index], parameters.beta[0, index]).sample()
        feature = parameters.location[0, index] + parameters.scale[0, index] * torch.randn_like(
            parameters.location[0, index]
        )
        mass = float((omega * state.reservoir).cpu())
        return mass, feature

    @torch.no_grad()
    def sample_birth_batch(
        self,
        states: Sequence[WeightedSetState],
        times: Sequence[float],
        leaf_budgets: Sequence[int],
        conditions: Sequence[Tensor | None],
    ) -> list[tuple[float, Tensor]]:
        """Vectorized birth-mark sampling for independent set trajectories."""
        if not states:
            return []
        condition_batch = self._conditions(conditions, len(states))
        encoded = self.encode(
            states,
            torch.tensor(times, device=self.device, dtype=self.dtype),
            torch.tensor(leaf_budgets, device=self.device),
            condition_batch,
        )
        parameters = self.birth_parameters(encoded.pooled)
        mixtures = torch.distributions.Categorical(logits=parameters.mixture_logits).sample()
        rows = torch.arange(len(states), device=self.device)
        alpha = parameters.alpha[rows, mixtures]
        beta = parameters.beta[rows, mixtures]
        omega = torch.distributions.Beta(alpha, beta).sample()
        locations = parameters.location[rows, mixtures]
        scales = parameters.scale[rows, mixtures]
        features = locations + scales * torch.randn_like(locations)
        reservoirs = torch.tensor(
            [state.reservoir for state in states], device=self.device, dtype=self.dtype
        )
        masses = omega * reservoirs
        return [(float(mass.cpu()), feature) for mass, feature in zip(masses, features)]

    @torch.no_grad()
    def _sample_birth_phase_batch(
        self,
        total_masses: Sequence[float],
        corruption: CorruptionConfig,
        conditions: Sequence[Tensor | None],
        rng: random.Random,
        leaf_temperature: float,
    ) -> tuple[list[WeightedSetState], list[int], list[int]]:
        count = len(total_masses)
        condition_batch = self._conditions(conditions, count)
        totals = torch.tensor(total_masses, device=self.device, dtype=self.dtype)
        logits = self.leaf_logits(totals, condition_batch)
        sampled = torch.distributions.Categorical(logits=logits / leaf_temperature).sample()
        leaf_budgets = [int(value) for value in sampled.cpu().tolist()]
        states = [
            empty_state(total_mass, self.config.feature_dim, device=self.device, dtype=self.dtype)
            for total_mass in total_masses
        ]
        birth_times = [
            sorted(rng.uniform(0.0, corruption.nucleation_duration) for _ in range(budget))
            for budget in leaf_budgets
        ]
        times = [0.0] * count
        next_lineages = [0] * count
        for rank in range(max(leaf_budgets, default=0)):
            active = [index for index, budget in enumerate(leaf_budgets) if rank < budget]
            active_states = []
            active_times = []
            for index in active:
                birth_time = birth_times[index][rank]
                states[index] = flow_state(
                    states[index], corruption.gamma * (birth_time - times[index]), reverse=True
                )
                times[index] = birth_time
                active_states.append(states[index])
                active_times.append(birth_time)
            marks = self.sample_birth_batch(
                active_states,
                active_times,
                [leaf_budgets[index] for index in active],
                [conditions[index] for index in active],
            )
            for index, (mass, feature) in zip(active, marks):
                states[index] = birth_component(
                    states[index], mass, feature, next_lineages[index]
                )
                next_lineages[index] += 1
        for index in range(count):
            states[index] = flow_state(
                states[index],
                corruption.gamma * (corruption.nucleation_duration - times[index]),
                reverse=True,
            )
        return states, leaf_budgets, next_lineages

    def candidate_pairs(self, state: WeightedSetState) -> Tensor:
        n = state.n
        if n < 2:
            return torch.empty((0, 2), device=self.device, dtype=torch.long)
        if self.config.sparse_k is None or self.config.sparse_k >= n - 1:
            return torch.triu_indices(n, n, offset=1, device=self.device).T
        features = state.features.to(self.device, self.dtype)
        distances = torch.cdist(features, features)
        distances.fill_diagonal_(float("inf"))
        neighbors = distances.topk(k=min(self.config.sparse_k, n - 1), largest=False).indices
        first = torch.arange(n, device=self.device)[:, None].expand_as(neighbors).reshape(-1)
        second = neighbors.reshape(-1)
        pairs = torch.stack((torch.minimum(first, second), torch.maximum(first, second)), dim=-1)
        return torch.unique(pairs, dim=0, sorted=True)

    def merge_outputs(
        self,
        states: Sequence[WeightedSetState],
        times: Tensor,
        leaf_budgets: Tensor,
        conditions: Tensor | None = None,
    ) -> list[MergeOutput]:
        encoded = self.encode(states, times, leaf_budgets, conditions)
        outputs: list[MergeOutput] = []
        for row, state in enumerate(states):
            pairs = self.candidate_pairs(state)
            if len(pairs) == 0:
                zero = encoded.pooled[row, 0] * 0.0
                outputs.append(MergeOutput(pairs, zero[None][:0], zero[None][:0], zero, zero[None][:0]))
                continue
            first = encoded.component_embeddings[row, pairs[:, 0]]
            second = encoded.component_embeddings[row, pairs[:, 1]]
            pooled = encoded.pooled[row][None, :].expand(len(pairs), -1)
            pair_input = torch.cat((first + second, (first - second).abs(), pooled), dim=-1)
            logits = self.pair_head(pair_input).squeeze(-1)
            probabilities = F.softmax(logits, dim=0)
            total_rate = (
                self.config.k_max
                * len(pairs)
                * torch.sigmoid(self.rate_head(encoded.pooled[row]).squeeze(-1))
            )
            outputs.append(MergeOutput(pairs, logits, probabilities, total_rate, total_rate * probabilities))
        return outputs

    def path_loss(self, histories: Sequence[ReverseHistory], rng: random.Random | None = None) -> dict[str, Tensor]:
        """Evaluate the trainable path NLL with a stratified survival estimate."""
        if not histories:
            raise ValueError("at least one history is required")
        rng = random.Random() if rng is None else rng
        batch_size = len(histories)
        conditions = self._conditions([history.condition for history in histories], batch_size)
        totals = torch.tensor([history.total_mass for history in histories], device=self.device, dtype=self.dtype)
        targets = torch.tensor([history.leaf_budget for history in histories], device=self.device, dtype=torch.long)
        count_terms = F.cross_entropy(self.leaf_logits(totals, conditions), targets, reduction="none")
        birth_terms = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        merge_terms = torch.zeros_like(birth_terms)
        survival_terms = torch.zeros_like(birth_terms)

        birth_rows: list[tuple[int, BirthObservation, Tensor | None]] = []
        merge_rows: list[tuple[int, MergeObservation, int, Tensor | None]] = []
        interval_rows: list[tuple[int, WeightedSetState, float, float, int, Tensor | None]] = []
        for history_index, history in enumerate(histories):
            birth_rows.extend((history_index, event, history.condition) for event in history.births)
            merge_rows.extend(
                (history_index, event, history.leaf_budget, history.condition) for event in history.merges
            )
            for interval in history.merge_intervals:
                if interval.end > interval.start:
                    sample_time = rng.uniform(interval.start, interval.end)
                    state = flow_state(
                        interval.state_at_start,
                        history.gamma * (sample_time - interval.start),
                        reverse=True,
                    )
                    interval_rows.append(
                        (history_index, state, sample_time, interval.end - interval.start, history.leaf_budget, history.condition)
                    )

        if birth_rows:
            birth_conditions = self._conditions([row[2] for row in birth_rows], len(birth_rows))
            log_prob = self.birth_log_prob(
                [row[1].state for row in birth_rows],
                torch.tensor([row[1].time for row in birth_rows], device=self.device),
                torch.tensor([row[1].leaf_budget for row in birth_rows], device=self.device),
                torch.tensor([row[1].mass for row in birth_rows], device=self.device),
                torch.stack([row[1].feature for row in birth_rows]),
                birth_conditions,
            )
            indices = torch.tensor([row[0] for row in birth_rows], device=self.device)
            birth_terms.index_add_(0, indices, -log_prob)

        if merge_rows:
            merge_conditions = self._conditions([row[3] for row in merge_rows], len(merge_rows))
            outputs = self.merge_outputs(
                [row[1].state for row in merge_rows],
                torch.tensor([row[1].time for row in merge_rows], device=self.device),
                torch.tensor([row[2] for row in merge_rows], device=self.device),
                merge_conditions,
            )
            values = []
            for output, row in zip(outputs, merge_rows):
                target = torch.tensor(row[1].pair, device=self.device)
                matches = torch.all(output.pairs == target[None, :], dim=1).nonzero(as_tuple=False)
                if len(matches) != 1:
                    raise ValueError("observed merge pair is absent from the candidate graph")
                values.append(-torch.log(output.intensities[matches[0, 0]].clamp_min(1e-12)))
            indices = torch.tensor([row[0] for row in merge_rows], device=self.device)
            merge_terms.index_add_(0, indices, torch.stack(values))

        if interval_rows:
            interval_conditions = self._conditions([row[5] for row in interval_rows], len(interval_rows))
            outputs = self.merge_outputs(
                [row[1] for row in interval_rows],
                torch.tensor([row[2] for row in interval_rows], device=self.device),
                torch.tensor([row[4] for row in interval_rows], device=self.device),
                interval_conditions,
            )
            values = torch.stack(
                [output.total_rate * row[3] for output, row in zip(outputs, interval_rows)]
            )
            indices = torch.tensor([row[0] for row in interval_rows], device=self.device)
            survival_terms.index_add_(0, indices, values)

        per_history = count_terms + birth_terms + merge_terms + survival_terms
        return {
            "loss": per_history.mean(),
            "count_nll": count_terms.mean(),
            "birth_nll": birth_terms.mean(),
            "merge_nll": merge_terms.mean(),
            "survival": survival_terms.mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        total_mass: float,
        corruption: CorruptionConfig,
        condition: Tensor | None = None,
        rng: random.Random | None = None,
        leaf_temperature: float = 1.0,
        merge_rate_scale: float = 1.0,
    ) -> WeightedSetState:
        """Run exact birth timing and thinning-based coagulation sampling."""
        if leaf_temperature <= 0.0 or merge_rate_scale < 0.0:
            raise ValueError("leaf_temperature must be positive and merge_rate_scale nonnegative")
        rng = random.Random() if rng is None else rng
        condition_batch = self._conditions([condition], 1)
        logits = self.leaf_logits(
            torch.tensor([total_mass], device=self.device, dtype=self.dtype), condition_batch
        )
        leaf_budget = int(torch.distributions.Categorical(logits=logits[0] / leaf_temperature).sample())
        state = empty_state(total_mass, self.config.feature_dim, device=self.device, dtype=self.dtype)
        birth_times = sorted(rng.uniform(0.0, corruption.nucleation_duration) for _ in range(leaf_budget))
        time = 0.0
        next_lineage = 0
        for birth_time in birth_times:
            state = flow_state(state, corruption.gamma * (birth_time - time), reverse=True)
            mass, feature = self.sample_birth(state, birth_time, leaf_budget, condition)
            state = birth_component(state, mass, feature, next_lineage)
            next_lineage += 1
            time = birth_time
        state = flow_state(
            state, corruption.gamma * (corruption.nucleation_duration - time), reverse=True
        )
        time = corruption.nucleation_duration

        while time < corruption.horizon and state.n >= 2:
            candidates = self.candidate_pairs(state)
            envelope = self.config.k_max * len(candidates) * max(merge_rate_scale, 1.0)
            if envelope <= 0.0:
                break
            wait = rng.expovariate(envelope)
            if time + wait >= corruption.horizon:
                break
            state = flow_state(state, corruption.gamma * wait, reverse=True)
            time += wait
            output = self.merge_outputs(
                [state],
                torch.tensor([time], device=self.device),
                torch.tensor([leaf_budget], device=self.device),
                condition_batch,
            )[0]
            acceptance = float((merge_rate_scale * output.total_rate / envelope).clamp(0.0, 1.0).cpu())
            if rng.random() < acceptance:
                pair_index = int(torch.distributions.Categorical(probs=output.pair_probabilities).sample())
                first, second = output.pairs[pair_index].tolist()
                state = merge_components(state, first, second, next_lineage)
                next_lineage += 1
        state = flow_state(state, corruption.gamma * (corruption.horizon - time), reverse=True)
        if state.conservation_residual() > 1e-6:
            raise RuntimeError("neural sampler violated mass conservation")
        return state

    @torch.no_grad()
    def sample_batch(
        self,
        total_masses: Sequence[float],
        corruption: CorruptionConfig,
        conditions: Sequence[Tensor | None] | None = None,
        rng: random.Random | None = None,
        leaf_temperature: float = 1.0,
        merge_rate_scale: float = 1.0,
    ) -> list[WeightedSetState]:
        """Generate independent trajectories while batching neural evaluations.

        Event clocks and accept/reject decisions remain trajectory-specific;
        only birth-mark and merge-network evaluations are padded and batched.
        """
        if leaf_temperature <= 0.0 or merge_rate_scale < 0.0:
            raise ValueError("leaf_temperature must be positive and merge_rate_scale nonnegative")
        if not total_masses:
            return []
        rng = random.Random() if rng is None else rng
        conditions = list(conditions) if conditions is not None else [None] * len(total_masses)
        if len(conditions) != len(total_masses):
            raise ValueError("one condition is required per total mass")
        states, leaf_budgets, next_lineages = self._sample_birth_phase_batch(
            total_masses, corruption, conditions, rng, leaf_temperature
        )
        times = [corruption.nucleation_duration] * len(states)
        finished = [False] * len(states)
        while True:
            proposed_indices = []
            proposed_states = []
            proposed_times = []
            envelopes = []
            for index, state in enumerate(states):
                if finished[index] or state.n < 2:
                    continue
                candidates = self.candidate_pairs(state)
                envelope = self.config.k_max * len(candidates) * max(merge_rate_scale, 1.0)
                if envelope <= 0.0:
                    finished[index] = True
                    continue
                wait = rng.expovariate(envelope)
                if times[index] + wait >= corruption.horizon:
                    finished[index] = True
                    continue
                states[index] = flow_state(state, corruption.gamma * wait, reverse=True)
                times[index] += wait
                proposed_indices.append(index)
                proposed_states.append(states[index])
                proposed_times.append(times[index])
                envelopes.append(envelope)
            if not proposed_indices:
                break
            proposed_conditions = self._conditions(
                [conditions[index] for index in proposed_indices], len(proposed_indices)
            )
            outputs = self.merge_outputs(
                proposed_states,
                torch.tensor(proposed_times, device=self.device, dtype=self.dtype),
                torch.tensor(
                    [leaf_budgets[index] for index in proposed_indices], device=self.device
                ),
                proposed_conditions,
            )
            for index, envelope, output in zip(proposed_indices, envelopes, outputs):
                acceptance = float(
                    (merge_rate_scale * output.total_rate / envelope).clamp(0.0, 1.0).cpu()
                )
                if rng.random() < acceptance:
                    pair_index = int(
                        torch.distributions.Categorical(probs=output.pair_probabilities).sample()
                    )
                    first, second = output.pairs[pair_index].tolist()
                    states[index] = merge_components(
                        states[index], first, second, next_lineages[index]
                    )
                    next_lineages[index] += 1
        for index, state in enumerate(states):
            states[index] = flow_state(
                state, corruption.gamma * (corruption.horizon - times[index]), reverse=True
            )
            if states[index].conservation_residual() > 1e-6:
                raise RuntimeError("batched neural sampler violated mass conservation")
        return states


class BirthOnlyModel(CoagulationModel):
    """Capacity-matched single-component insertion ablation.

    The class deliberately inherits the same parameter tensors as the
    coagulation model.  Pair/rate networks are reused as a state-dependent
    adapter for the coupled birth density, so the comparison has the same
    nominal parameter count without carrying dormant merge parameters.
    """

    def birth_parameters(self, pooled: Tensor) -> BirthParameters:
        batch = pooled.shape[0]
        components = self.config.mixture_components
        feature_dim = self.config.feature_dim
        width = 3 + 2 * feature_dim
        raw = self.birth_head(pooled).reshape(batch, components, width)
        repeated = torch.cat((pooled, pooled, pooled), dim=-1)
        pair_adapter = self.pair_head(repeated).reshape(batch, 1, 1)
        rate_adapter = self.rate_head(pooled).reshape(batch, 1, 1)
        basis = torch.linspace(-1.0, 1.0, components * width, device=raw.device, dtype=raw.dtype)
        basis = basis.reshape(1, components, width)
        raw = raw + 0.1 * (pair_adapter + rate_adapter) * basis
        return BirthParameters(
            raw[..., 0],
            F.softplus(raw[..., 1]) + 1e-3,
            F.softplus(raw[..., 2]) + 1e-3,
            raw[..., 3 : 3 + feature_dim],
            F.softplus(raw[..., 3 + feature_dim :]) + self.config.min_scale,
        )

    def path_loss(self, histories: Sequence[ReverseHistory], rng: random.Random | None = None) -> dict[str, Tensor]:
        """Birth/count path NLL for deletion-only corruption histories."""
        del rng
        if not histories:
            raise ValueError("at least one history is required")
        if any(history.merges for history in histories):
            raise ValueError("birth-only training requires split_rate=0 histories")
        batch_size = len(histories)
        conditions = self._conditions([history.condition for history in histories], batch_size)
        totals = torch.tensor([history.total_mass for history in histories], device=self.device, dtype=self.dtype)
        targets = torch.tensor([history.leaf_budget for history in histories], device=self.device, dtype=torch.long)
        count_terms = F.cross_entropy(self.leaf_logits(totals, conditions), targets, reduction="none")
        birth_terms = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        rows = [
            (history_index, event, history.condition)
            for history_index, history in enumerate(histories)
            for event in history.births
        ]
        if rows:
            birth_conditions = self._conditions([row[2] for row in rows], len(rows))
            log_prob = self.birth_log_prob(
                [row[1].state for row in rows],
                torch.tensor([row[1].time for row in rows], device=self.device),
                torch.tensor([row[1].leaf_budget for row in rows], device=self.device),
                torch.tensor([row[1].mass for row in rows], device=self.device),
                torch.stack([row[1].feature for row in rows]),
                birth_conditions,
            )
            indices = torch.tensor([row[0] for row in rows], device=self.device)
            birth_terms.index_add_(0, indices, -log_prob)
        zero = birth_terms.sum() * 0.0
        per_history = count_terms + birth_terms
        return {
            "loss": per_history.mean(),
            "count_nll": count_terms.mean(),
            "birth_nll": birth_terms.mean(),
            "merge_nll": zero,
            "survival": zero,
        }

    @torch.no_grad()
    def sample(
        self,
        total_mass: float,
        corruption: CorruptionConfig,
        condition: Tensor | None = None,
        rng: random.Random | None = None,
        leaf_temperature: float = 1.0,
        merge_rate_scale: float = 1.0,
    ) -> WeightedSetState:
        """Generate solely through sequential births and conservative flow."""
        del merge_rate_scale
        if leaf_temperature <= 0.0:
            raise ValueError("leaf_temperature must be positive")
        rng = random.Random() if rng is None else rng
        condition_batch = self._conditions([condition], 1)
        logits = self.leaf_logits(
            torch.tensor([total_mass], device=self.device, dtype=self.dtype), condition_batch
        )
        leaf_budget = int(torch.distributions.Categorical(logits=logits[0] / leaf_temperature).sample())
        state = empty_state(total_mass, self.config.feature_dim, device=self.device, dtype=self.dtype)
        birth_times = sorted(rng.uniform(0.0, corruption.nucleation_duration) for _ in range(leaf_budget))
        time = 0.0
        for lineage, birth_time in enumerate(birth_times):
            state = flow_state(state, corruption.gamma * (birth_time - time), reverse=True)
            mass, feature = self.sample_birth(state, birth_time, leaf_budget, condition)
            state = birth_component(state, mass, feature, lineage)
            time = birth_time
        state = flow_state(state, corruption.gamma * (corruption.horizon - time), reverse=True)
        if state.conservation_residual() > 1e-6:
            raise RuntimeError("birth-only sampler violated mass conservation")
        return state

    @torch.no_grad()
    def sample_batch(
        self,
        total_masses: Sequence[float],
        corruption: CorruptionConfig,
        conditions: Sequence[Tensor | None] | None = None,
        rng: random.Random | None = None,
        leaf_temperature: float = 1.0,
        merge_rate_scale: float = 1.0,
    ) -> list[WeightedSetState]:
        del merge_rate_scale
        if leaf_temperature <= 0.0:
            raise ValueError("leaf_temperature must be positive")
        if not total_masses:
            return []
        rng = random.Random() if rng is None else rng
        conditions = list(conditions) if conditions is not None else [None] * len(total_masses)
        if len(conditions) != len(total_masses):
            raise ValueError("one condition is required per total mass")
        states, _, _ = self._sample_birth_phase_batch(
            total_masses, corruption, conditions, rng, leaf_temperature
        )
        for index, state in enumerate(states):
            states[index] = flow_state(
                state, corruption.gamma * (corruption.horizon - corruption.nucleation_duration), reverse=True
            )
            if states[index].conservation_residual() > 1e-6:
                raise RuntimeError("batched birth-only sampler violated mass conservation")
        return states


class VoxelBirthOnlyModel(BirthOnlyModel):
    """Birth-only model whose marks are categorical detector cells.

    The continuous Gaussian-mixture head is interpreted as a distribution on
    the finite set of supplied cell centers. This preserves the existing
    parameter budget while guaranteeing valid geometry and masking cells that
    are already present in the generated set.
    """

    def __init__(
        self,
        config: NeuralConfig,
        voxel_features: Tensor | Sequence[Sequence[float]],
        *,
        coordinate_tolerance: float = 1e-5,
    ):
        super().__init__(config)
        features = torch.as_tensor(voxel_features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != config.feature_dim:
            raise ValueError("voxel_features must have shape [V, feature_dim]")
        if len(features) < config.max_components:
            raise ValueError("max_components cannot exceed the number of detector cells")
        if coordinate_tolerance < 0.0:
            raise ValueError("coordinate_tolerance must be nonnegative")
        if len(torch.unique(features, dim=0)) != len(features):
            raise ValueError("voxel_features must contain unique cell centers")
        self.register_buffer("voxel_features", features)
        self.coordinate_tolerance = float(coordinate_tolerance)

    def _cell_indices(self, features: Tensor) -> Tensor:
        features = features.to(device=self.device, dtype=self.dtype)
        cells = self.voxel_features.to(dtype=self.dtype)
        # ``torch.cdist`` may return a small nonzero value for two identical
        # float32 rows because its matrix-multiplication implementation forms
        # ``x^2 + y^2 - 2xy``.  Direct subtraction is exact for coordinates
        # copied from the registered official geometry and avoids rejecting
        # valid cells under a tight tolerance.
        squared_distances = (features[:, None, :] - cells[None, :, :]).square().sum(dim=-1)
        minimum_squared, indices = squared_distances.min(dim=-1)
        minimum = minimum_squared.sqrt()
        if bool(torch.any(minimum > self.coordinate_tolerance)):
            maximum = float(minimum.max().detach().cpu())
            raise ValueError(f"feature is not an official detector cell (distance={maximum:g})")
        return indices

    def _masked_cell_logits(
        self,
        parameters: BirthParameters,
        states: Sequence[WeightedSetState],
        mixtures: Tensor | None = None,
    ) -> Tensor:
        cells = self.voxel_features.to(device=self.device, dtype=self.dtype)
        if mixtures is None:
            difference = cells[None, None, :, :] - parameters.location[:, :, None, :]
            scaled = difference / parameters.scale[:, :, None, :]
            logits = -0.5 * scaled.square().sum(dim=-1)
        else:
            rows = torch.arange(len(states), device=self.device)
            locations = parameters.location[rows, mixtures]
            scales = parameters.scale[rows, mixtures]
            difference = cells[None, :, :] - locations[:, None, :]
            logits = -0.5 * (difference / scales[:, None, :]).square().sum(dim=-1)
        for row, state in enumerate(states):
            if not state.n:
                continue
            occupied = self._cell_indices(state.features)
            if mixtures is None:
                logits[row, :, occupied] = -torch.inf
            else:
                logits[row, occupied] = -torch.inf
        if bool(torch.any(torch.isneginf(logits).all(dim=-1))):
            raise ValueError("no unoccupied detector cell remains for a birth")
        return logits

    def birth_log_prob(
        self,
        states: Sequence[WeightedSetState],
        times: Tensor,
        leaf_budgets: Tensor,
        masses: Tensor,
        features: Tensor,
        conditions: Tensor | None = None,
    ) -> Tensor:
        encoded = self.encode(states, times, leaf_budgets, conditions)
        parameters = self.birth_parameters(encoded.pooled)
        reservoir = masses.new_tensor([state.reservoir for state in states]).to(self.device, self.dtype)
        masses = masses.to(self.device, self.dtype)
        target_cells = self._cell_indices(features)
        omega = (masses / reservoir).clamp(1e-6, 1.0 - 1e-6)
        beta_log_prob = (
            (parameters.alpha - 1.0) * omega[:, None].log()
            + (parameters.beta - 1.0) * torch.log1p(-omega[:, None])
            - torch.lgamma(parameters.alpha)
            - torch.lgamma(parameters.beta)
            + torch.lgamma(parameters.alpha + parameters.beta)
        )
        cell_logits = self._masked_cell_logits(parameters, states)
        cell_log_prob = F.log_softmax(cell_logits, dim=-1).gather(
            -1,
            target_cells[:, None, None].expand(-1, self.config.mixture_components, 1),
        ).squeeze(-1)
        mixture_log_prob = F.log_softmax(parameters.mixture_logits, dim=-1)
        return torch.logsumexp(
            mixture_log_prob + beta_log_prob + cell_log_prob, dim=-1
        ) - reservoir.log()

    @torch.no_grad()
    def sample_birth(
        self,
        state: WeightedSetState,
        time: float,
        leaf_budget: int,
        condition: Tensor | None = None,
    ) -> tuple[float, Tensor]:
        return self.sample_birth_batch([state], [time], [leaf_budget], [condition])[0]

    @torch.no_grad()
    def sample_birth_batch(
        self,
        states: Sequence[WeightedSetState],
        times: Sequence[float],
        leaf_budgets: Sequence[int],
        conditions: Sequence[Tensor | None],
    ) -> list[tuple[float, Tensor]]:
        if not states:
            return []
        condition_batch = self._conditions(conditions, len(states))
        encoded = self.encode(
            states,
            torch.tensor(times, device=self.device, dtype=self.dtype),
            torch.tensor(leaf_budgets, device=self.device),
            condition_batch,
        )
        parameters = self.birth_parameters(encoded.pooled)
        mixtures = torch.distributions.Categorical(logits=parameters.mixture_logits).sample()
        rows = torch.arange(len(states), device=self.device)
        omega = torch.distributions.Beta(
            parameters.alpha[rows, mixtures], parameters.beta[rows, mixtures]
        ).sample()
        cell_logits = self._masked_cell_logits(parameters, states, mixtures)
        cells = torch.distributions.Categorical(logits=cell_logits).sample()
        features = self.voxel_features[cells].to(dtype=self.dtype)
        reservoirs = torch.tensor(
            [state.reservoir for state in states], device=self.device, dtype=self.dtype
        )
        masses = omega * reservoirs
        return [(float(mass.cpu()), feature) for mass, feature in zip(masses, features)]
