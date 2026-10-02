#!/usr/bin/env python3
"""CaloChallenge HDF5 to conservative weighted-set conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

import gpu_model as gm


@dataclass(frozen=True)
class CalorimeterGeometry:
    features: np.ndarray
    layer_ids: np.ndarray
    particle: str
    source: str

    @property
    def voxel_count(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class CalorimeterRecord:
    state: gm.WeightedSetState
    incident_energy_mev: float
    deposited_energy_mev: float
    retained_energy_mev: float
    condition: torch.Tensor


@dataclass(frozen=True)
class DenseCalorimeterBatch:
    """A contiguous slice in the official dense CaloChallenge representation."""

    showers_mev: np.ndarray
    incident_energies_mev: np.ndarray

    def __post_init__(self) -> None:
        if self.showers_mev.ndim != 2:
            raise ValueError("showers_mev must have shape [events, voxels]")
        if self.incident_energies_mev.ndim != 1:
            raise ValueError("incident_energies_mev must have shape [events]")
        if len(self.showers_mev) != len(self.incident_energies_mev):
            raise ValueError("showers and incident energies must contain the same number of events")


def load_geometry(xml_path: Path, particle: str = "photon") -> CalorimeterGeometry:
    """Read the official polar binning and return normalized (layer, x, y) cell centers."""
    root = ET.parse(xml_path).getroot()
    particle_node = next((node for node in root if node.attrib.get("name") == particle), None)
    if particle_node is None:
        raise ValueError(f"particle {particle!r} is absent from {xml_path}")
    cells: list[tuple[int, float, float]] = []
    relevant_layers = []
    max_radius = 0.0
    raw_layers = []
    for layer in particle_node:
        layer_id = int(layer.attrib["id"])
        edges = [float(value) for value in layer.attrib["r_edges"].split(",")]
        alpha_bins = int(layer.attrib["n_bin_alpha"])
        if len(edges) < 2:
            continue
        relevant_layers.append(layer_id)
        max_radius = max(max_radius, edges[-1])
        radial = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
        alphas = np.linspace(-math.pi, math.pi, alpha_bins + 1)
        alpha_centers = (alphas[:-1] + alphas[1:]) / 2.0
        raw_layers.append((layer_id, radial, alpha_centers))
    if not raw_layers or max_radius <= 0.0:
        raise ValueError(f"no active calorimeter cells found in {xml_path}")
    first_layer, last_layer = min(relevant_layers), max(relevant_layers)
    layer_span = max(last_layer - first_layer, 1)
    layer_ids = []
    for layer_id, radial, alpha_centers in raw_layers:
        z = 2.0 * (layer_id - first_layer) / layer_span - 1.0
        # Official flattening: radial bins vary fastest inside each angular slice.
        for alpha in alpha_centers:
            for radius in radial:
                cells.append((z, radius * math.cos(alpha) / max_radius, radius * math.sin(alpha) / max_radius))
                layer_ids.append(layer_id)
    return CalorimeterGeometry(
        np.asarray(cells, dtype=np.float32),
        np.asarray(layer_ids, dtype=np.int64),
        particle,
        str(xml_path),
    )


def inspect_hdf5(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        if "showers" not in handle or "incident_energies" not in handle:
            raise ValueError(f"{path} lacks CaloChallenge showers/incident_energies datasets")
        return {
            "events": int(handle["showers"].shape[0]),
            "voxels": int(handle["showers"].shape[1]),
            "showers_dtype": str(handle["showers"].dtype),
            "incident_shape": tuple(int(value) for value in handle["incident_energies"].shape),
        }


def incident_log_stats(path: Path, limit: int | None = None) -> tuple[float, float]:
    with h5py.File(path, "r") as handle:
        count = len(handle["incident_energies"]) if limit is None else min(limit, len(handle["incident_energies"]))
        energies = np.asarray(handle["incident_energies"][:count], dtype=np.float64).reshape(-1)
    logs = np.log(energies.clip(min=1e-12))
    return float(logs.mean()), float(max(logs.std(), 1e-6))


def load_dense_batch(
    path: Path,
    *,
    start: int = 0,
    count: int | None = None,
    min_energy_mev: float | None = None,
) -> DenseCalorimeterBatch:
    """Load an official dense slice without applying a top-k approximation.

    ``min_energy_mev`` reproduces the evaluator's cell threshold when supplied.
    The returned arrays always own their memory, so the HDF5 file can be closed
    immediately and callers may safely modify the arrays.
    """
    if start < 0 or (count is not None and count < 1):
        raise ValueError("start must be nonnegative and count must be positive")
    if min_energy_mev is not None and min_energy_mev < 0.0:
        raise ValueError("min_energy_mev must be nonnegative")
    metadata = inspect_hdf5(path)
    stop = metadata["events"] if count is None else min(metadata["events"], start + count)
    if start >= stop:
        raise ValueError("requested event slice is empty")
    with h5py.File(path, "r") as handle:
        showers = np.array(handle["showers"][start:stop], dtype=np.float32, copy=True)
        incidents = np.array(
            handle["incident_energies"][start:stop], dtype=np.float32, copy=True
        ).reshape(-1)
    showers = np.nan_to_num(showers, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(showers, 0.0, out=showers)
    if min_energy_mev is not None:
        showers[showers < min_energy_mev] = 0.0
    if not np.isfinite(incidents).all() or np.any(incidents <= 0.0):
        raise ValueError(f"{path} contains a nonpositive or nonfinite incident energy")
    return DenseCalorimeterBatch(showers, incidents)


def dataset1_incident_energies(*, repeats: int = 1000, seed: int | None = None) -> np.ndarray:
    """Return the canonical Dataset-1 incident-energy schedule in MeV.

    One repeat contains 121 events: ten copies of energies 2^8 through
    2^18 MeV, followed by 5/3/2/1 copies at 2^19/2^20/2^21/2^22 MeV.
    This is the schedule used by the public ViT-CFM Dataset-1 sampler.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    energies = np.tile(np.logspace(8, 18, 11, base=2.0, dtype=np.float64), 10)
    high = np.concatenate(
        (
            np.full(5, 2.0**19),
            np.full(3, 2.0**20),
            np.full(2, 2.0**21),
            np.full(1, 2.0**22),
        )
    )
    result = np.tile(np.concatenate((energies, high)), repeats)
    if seed is not None:
        np.random.default_rng(seed).shuffle(result)
    return result.reshape(-1, 1)


def states_to_dense_showers(
    states: list[gm.WeightedSetState],
    geometry: CalorimeterGeometry,
    *,
    mass_scales_mev: np.ndarray | None = None,
    require_exact_cells: bool = True,
    coordinate_tolerance: float = 1e-5,
) -> np.ndarray:
    """Rasterize weighted-set states into the official ``[N, V]`` layout.

    Duplicate generated cells are summed. For a categorical-cell model,
    ``require_exact_cells`` ensures every generated feature is an official
    geometry center rather than silently snapping arbitrary coordinates.
    """
    if coordinate_tolerance < 0.0:
        raise ValueError("coordinate_tolerance must be nonnegative")
    if mass_scales_mev is None:
        scales = np.ones(len(states), dtype=np.float64)
    else:
        scales = np.asarray(mass_scales_mev, dtype=np.float64).reshape(-1)
        if len(scales) != len(states):
            raise ValueError("one mass scale is required per state")
        if not np.isfinite(scales).all() or np.any(scales < 0.0):
            raise ValueError("mass scales must be finite and nonnegative")
    result = np.zeros((len(states), geometry.voxel_count), dtype=np.float32)
    centers = np.asarray(geometry.features, dtype=np.float64)
    for row, (state, scale) in enumerate(zip(states, scales)):
        if state.n == 0:
            continue
        features = state.features.detach().cpu().numpy().astype(np.float64, copy=False)
        masses = state.masses.detach().cpu().numpy().astype(np.float64, copy=False)
        if features.shape[1] != centers.shape[1]:
            raise ValueError("state and geometry feature dimensions differ")
        squared = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        indices = squared.argmin(axis=1)
        distances = np.sqrt(squared[np.arange(len(indices)), indices])
        if require_exact_cells and np.any(distances > coordinate_tolerance):
            maximum = float(distances.max())
            raise ValueError(f"generated feature is not an official cell center (distance={maximum:g})")
        np.add.at(result[row], indices, masses * scale)
    return result


def write_official_hdf5(
    path: Path,
    showers_mev: np.ndarray,
    incident_energies_mev: np.ndarray,
) -> None:
    """Write a sample accepted by the official CaloChallenge evaluator."""
    showers = np.asarray(showers_mev)
    incidents = np.asarray(incident_energies_mev).reshape(-1, 1)
    if showers.ndim != 2 or len(showers) != len(incidents):
        raise ValueError("expected showers [N, V] and one incident energy per shower")
    if not np.isfinite(showers).all() or np.any(showers < 0.0):
        raise ValueError("showers must contain finite nonnegative energies")
    if not np.isfinite(incidents).all() or np.any(incidents <= 0.0):
        raise ValueError("incident energies must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("showers", data=showers, dtype=np.float32)
        handle.create_dataset("incident_energies", data=incidents, dtype=np.float32)
    temporary.replace(path)


def load_records(
    path: Path,
    geometry: CalorimeterGeometry,
    *,
    start: int = 0,
    count: int | None = None,
    max_components: int = 48,
    relative_threshold: float = 1e-2,
    log_energy_mean: float,
    log_energy_std: float,
) -> list[CalorimeterRecord]:
    """Load events, retaining the most energetic cells and putting omitted energy in the reservoir."""
    if max_components < 1 or relative_threshold < 0.0:
        raise ValueError("max_components must be positive and threshold nonnegative")
    metadata = inspect_hdf5(path)
    if metadata["voxels"] != geometry.voxel_count:
        raise ValueError(
            f"geometry has {geometry.voxel_count} cells but {path} has {metadata['voxels']} voxels"
        )
    stop = metadata["events"] if count is None else min(metadata["events"], start + count)
    if start < 0 or start >= stop:
        raise ValueError("requested event slice is empty")
    records = []
    with h5py.File(path, "r") as handle:
        showers = np.asarray(handle["showers"][start:stop], dtype=np.float64)
        incidents = np.asarray(handle["incident_energies"][start:stop], dtype=np.float64).reshape(-1)
    for event_index, (shower, incident) in enumerate(zip(showers, incidents), start=start):
        if not np.isfinite(incident) or incident <= 0.0:
            raise ValueError(f"nonpositive incident energy in event {event_index}")
        shower = np.nan_to_num(shower, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
        deposited = float(shower.sum())
        active = np.flatnonzero(shower > relative_threshold * incident)
        if not len(active) and deposited > 0.0:
            active = np.asarray([int(shower.argmax())])
        if len(active) > max_components:
            chosen = np.argpartition(shower[active], -max_components)[-max_components:]
            active = active[chosen]
        active = active[np.argsort(-shower[active], kind="stable")]
        retained = float(shower[active].sum())
        # Rare over-response events are scaled only enough to satisfy the conservative state domain.
        scale = max(incident, retained)
        masses = torch.as_tensor(shower[active] / scale, dtype=torch.float32)
        reservoir = max(0.0, 1.0 - float(masses.double().sum()))
        state = gm.WeightedSetState(
            masses,
            torch.as_tensor(geometry.features[active], dtype=torch.float32),
            reservoir,
            1.0,
            tuple(range(len(active))),
        )
        condition = torch.tensor([(math.log(incident) - log_energy_mean) / log_energy_std], dtype=torch.float32)
        records.append(CalorimeterRecord(state, float(incident), deposited, retained, condition))
    return records
