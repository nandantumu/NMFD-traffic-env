from __future__ import annotations

import collections
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class NMFDParameters:
    """Static network data and physical parameters for an NMFD environment."""

    num_regions: int = struct.field(pytree_node=False)
    dt: float = struct.field(pytree_node=False)
    n_substeps: int = struct.field(pytree_node=False)
    adjacency: jnp.ndarray
    theta: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    u_low: float
    u_high: float

    @property
    def state_dim(self) -> int:
        return self.num_regions**2

    @property
    def control_dim(self) -> int:
        return self.num_regions**2


@dataclass(frozen=True)
class ObjectiveConfig:
    horizon: int
    state_weight: float
    control_weight: float
    control_rate_weight: float


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dim: int
    num_hidden_layers: int
    activation: str


@dataclass(frozen=True)
class TrainingConfig:
    sample_pool_size: int
    epochs: int
    steps_per_epoch: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float


@dataclass(frozen=True)
class CellOverride:
    row: int
    column: int
    mean: float
    std: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class InitialStateScenario:
    name: str
    distribution: str
    mean: float
    std: float
    min_value: float
    max_value: float
    overrides: tuple[CellOverride, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    environment: NMFDParameters
    objective: ObjectiveConfig
    policy: PolicyConfig
    training: TrainingConfig
    scenarios: dict[str, InitialStateScenario]


def build_shortest_path_routing(adjacency: np.ndarray) -> np.ndarray:
    """Construct theta[i, h, j] from equal-weight shortest-path next hops."""

    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("shortest-path routing currently requires symmetric adjacency")

    regions = adjacency.shape[0]
    connected = adjacency > 0
    np.fill_diagonal(connected, False)
    neighbors = {i: np.flatnonzero(connected[i]).tolist() for i in range(regions)}
    theta = np.zeros((regions, regions, regions), dtype=np.float32)

    for destination in range(regions):
        distance = {destination: 0}
        queue: collections.deque[int] = collections.deque([destination])
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)

        if len(distance) != regions:
            raise ValueError("adjacency graph must be connected")

        for source in range(regions):
            if source == destination:
                continue
            next_hops = [
                neighbor
                for neighbor in neighbors[source]
                if distance[neighbor] == distance[source] - 1
            ]
            weight = 1.0 / len(next_hops)
            theta[source, next_hops, destination] = weight

    return theta


def _scenario_from_mapping(name: str, raw: dict[str, Any], regions: int) -> InitialStateScenario:
    distribution = str(raw.get("distribution", "normal")).lower()
    if distribution != "normal":
        raise ValueError(f"scenario {name!r}: only normal sampling is currently supported")

    lower = float(raw.get("min_value", 0.0))
    upper = float(raw["max_value"])
    overrides = []
    for item in raw.get("overrides", []):
        row = int(item["row"])
        column = int(item["column"])
        if not (0 <= row < regions and 0 <= column < regions):
            raise ValueError(f"scenario {name!r}: override ({row}, {column}) is out of range")
        overrides.append(
            CellOverride(
                row=row,
                column=column,
                mean=float(item.get("mean", raw["mean"])),
                std=float(item.get("std", raw["std"])),
                min_value=float(item.get("min_value", lower)),
                max_value=float(item.get("max_value", upper)),
            )
        )
    return InitialStateScenario(
        name=name,
        distribution=distribution,
        mean=float(raw["mean"]),
        std=float(raw["std"]),
        min_value=lower,
        max_value=upper,
        overrides=tuple(overrides),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a standalone NMFD experiment TOML file."""

    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    env_raw = raw["environment"]
    regions = int(env_raw["num_regions"])
    adjacency = np.asarray(env_raw["adjacency"], dtype=np.float32)
    if adjacency.shape != (regions, regions):
        raise ValueError(
            f"adjacency has shape {adjacency.shape}; expected ({regions}, {regions})"
        )
    theta = build_shortest_path_routing(adjacency)
    environment = NMFDParameters(
        num_regions=regions,
        dt=float(env_raw["dt"]),
        n_substeps=int(env_raw.get("n_substeps", 1)),
        adjacency=jnp.asarray(adjacency),
        theta=jnp.asarray(theta),
        a=jnp.full((regions,), float(env_raw["a"]), dtype=jnp.float32),
        b=jnp.full((regions,), float(env_raw["b"]), dtype=jnp.float32),
        c=jnp.full((regions,), float(env_raw["c"]), dtype=jnp.float32),
        u_low=float(env_raw.get("u_low", 0.2)),
        u_high=float(env_raw.get("u_high", 0.8)),
    )
    if environment.n_substeps < 1:
        raise ValueError("n_substeps must be positive")
    if environment.u_low >= environment.u_high:
        raise ValueError("u_low must be smaller than u_high")

    objective_raw = raw["objective"]
    objective = ObjectiveConfig(
        horizon=int(objective_raw["horizon"]),
        state_weight=float(objective_raw.get("state_weight", 1.0)),
        control_weight=float(objective_raw.get("control_weight", 0.0)),
        control_rate_weight=float(objective_raw.get("control_rate_weight", 0.0)),
    )
    if objective.horizon < 1:
        raise ValueError("objective horizon must be positive")
    policy_raw = raw["policy"]
    policy = PolicyConfig(
        hidden_dim=int(policy_raw.get("hidden_dim", 256)),
        num_hidden_layers=int(policy_raw.get("num_hidden_layers", 3)),
        activation=str(policy_raw.get("activation", "gelu")),
    )
    if policy.hidden_dim < 1 or policy.num_hidden_layers < 1:
        raise ValueError("policy dimensions must be positive")
    training_raw = raw["training"]
    training = TrainingConfig(
        sample_pool_size=int(training_raw.get("sample_pool_size", 50_000)),
        epochs=int(training_raw.get("epochs", 100)),
        steps_per_epoch=int(training_raw.get("steps_per_epoch", 200)),
        batch_size=int(training_raw.get("batch_size", 512)),
        learning_rate=float(training_raw.get("learning_rate", 2e-3)),
        weight_decay=float(training_raw.get("weight_decay", 1e-5)),
        gradient_clip=float(training_raw.get("gradient_clip", 1.0)),
    )
    scenarios = {
        name: _scenario_from_mapping(name, scenario_raw, regions)
        for name, scenario_raw in raw["scenarios"].items()
    }
    return ExperimentConfig(
        seed=int(raw.get("seed", 0)),
        environment=environment,
        objective=objective,
        policy=policy,
        training=training,
        scenarios=scenarios,
    )
