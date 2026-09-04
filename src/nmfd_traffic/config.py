from __future__ import annotations

import collections
import itertools
import math
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
    """Horizon for the shared total-vehicle-time controller objective."""

    horizon: int


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture of the perimeter-control DPC multilayer perceptron."""

    hidden_dim: int
    num_hidden_layers: int
    activation: str


@dataclass(frozen=True)
class MPPIConfig:
    """Sampling settings for 2017 information-theoretic MPPI.

    temperature is the paper's inverse-temperature parameter lambda.
    noise_std defines the isotropic covariance Sigma = noise_std**2 I.
    alpha is the fraction of zero-centered recovery samples and also gives
    the importance-correction coefficient gamma = lambda * (1 - alpha).
    The final two fields define the JAX-native Savitzky--Golay update filter.
    """

    planning_horizon: int
    samples: int
    iterations: int
    temperature: float
    noise_std: float
    alpha: float
    smoothing_window: int
    smoothing_polynomial: int


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings for offline DPC policy training."""

    epochs: int
    steps_per_epoch: int
    batch_size: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SimulationConfig:
    """Stochastic disturbance settings shared by training and evaluation."""

    state_noise_std: float


@dataclass(frozen=True)
class DemandProfile:
    """Piecewise-linear arrival-rate profile for one origin-destination pair."""

    name: str
    origin: int
    destination: int
    times: tuple[float, ...]
    rates: tuple[float, ...]


@dataclass(frozen=True)
class TrafficScenario:
    """Initial condition and demand perturbation for a traffic experiment."""

    name: str
    initial_state: float
    demand_scale: float
    demand_noise_std: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete environment, controller, training, and scenario configuration."""

    seed: int
    environment: NMFDParameters
    objective: ObjectiveConfig
    policy: PolicyConfig
    mppi: MPPIConfig
    training: TrainingConfig
    simulation: SimulationConfig
    demand_profiles: tuple[DemandProfile, ...]
    scenarios: dict[str, TrafficScenario]

    @property
    def naive_mppi(self) -> MPPIConfig:
        """Return MPPI settings under the configuration's former field name."""

        return self.mppi


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


def _demand_profile_from_mapping(raw: dict[str, Any], regions: int) -> DemandProfile:
    name = str(raw["name"])
    origin = int(raw["origin"])
    destination = int(raw["destination"])
    times = tuple(float(value) for value in raw["times"])
    rates = tuple(float(value) for value in raw["rates"])

    if not (0 <= origin < regions and 0 <= destination < regions):
        raise ValueError(f"demand profile {name!r}: origin or destination is invalid")
    if origin == destination:
        raise ValueError(f"demand profile {name!r}: origin and destination must differ")
    if len(times) != len(rates) or len(times) < 2:
        raise ValueError(
            f"demand profile {name!r}: times and rates need equal lengths of at "
            "least two"
        )
    if not all(math.isfinite(value) for value in (*times, *rates)):
        raise ValueError(f"demand profile {name!r}: values must be finite")
    if any(later <= earlier for earlier, later in itertools.pairwise(times)):
        raise ValueError(f"demand profile {name!r}: times must be strictly increasing")
    if any(rate < 0.0 for rate in rates):
        raise ValueError(f"demand profile {name!r}: rates must be nonnegative")
    return DemandProfile(name, origin, destination, times, rates)


def _scenario_from_mapping(name: str, raw: dict[str, Any]) -> TrafficScenario:
    scenario = TrafficScenario(
        name=name,
        initial_state=float(raw.get("initial_state", 0.0)),
        demand_scale=float(raw.get("demand_scale", 1.0)),
        demand_noise_std=float(raw.get("demand_noise_std", 0.0)),
    )
    values = (
        scenario.initial_state,
        scenario.demand_scale,
        scenario.demand_noise_std,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"scenario {name!r}: values must be finite")
    if min(values) < 0.0:
        raise ValueError(f"scenario {name!r}: values must be nonnegative")
    return scenario


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a standalone NMFD experiment TOML file."""

    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    env_raw = raw["environment"]
    regions = int(env_raw["num_regions"])
    if regions < 1:
        raise ValueError("num_regions must be positive")
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
    physical_values = (
        environment.dt,
        environment.u_low,
        environment.u_high,
        *np.asarray(environment.a),
        *np.asarray(environment.b),
        *np.asarray(environment.c),
    )
    if not all(math.isfinite(float(value)) for value in physical_values):
        raise ValueError("environment parameters must be finite")
    if environment.dt <= 0.0:
        raise ValueError("dt must be positive")
    if environment.n_substeps < 1:
        raise ValueError("n_substeps must be positive")
    if environment.u_low >= environment.u_high:
        raise ValueError("u_low must be smaller than u_high")

    objective_raw = raw["objective"]
    objective = ObjectiveConfig(horizon=int(objective_raw["horizon"]))
    if objective.horizon < 2:
        raise ValueError("objective horizon must be at least two")
    policy_raw = raw["policy"]
    policy = PolicyConfig(
        hidden_dim=int(policy_raw.get("hidden_dim", 128)),
        num_hidden_layers=int(policy_raw.get("num_hidden_layers", 3)),
        activation=str(policy_raw.get("activation", "tanh")),
    )
    if policy.hidden_dim < 1 or policy.num_hidden_layers < 1:
        raise ValueError("policy dimensions must be positive")
    if policy.activation.lower() not in {"gelu", "relu", "silu", "tanh"}:
        raise ValueError(f"unsupported policy activation {policy.activation!r}")
    mppi_raw = raw.get("mppi", raw.get("naive_mppi", {}))
    mppi = MPPIConfig(
        planning_horizon=int(mppi_raw.get("planning_horizon", 8)),
        samples=int(mppi_raw.get("samples", 128)),
        iterations=int(mppi_raw.get("iterations", 1)),
        temperature=float(mppi_raw.get("temperature", 1.0)),
        noise_std=float(mppi_raw.get("noise_std", 1.0)),
        alpha=float(mppi_raw.get("alpha", 0.01)),
        smoothing_window=int(mppi_raw.get("smoothing_window", 5)),
        smoothing_polynomial=int(mppi_raw.get("smoothing_polynomial", 2)),
    )
    if mppi.planning_horizon < 2:
        raise ValueError("MPPI planning_horizon must be at least two")
    if mppi.samples < 1 or mppi.iterations < 1:
        raise ValueError("MPPI samples and iterations must be positive")
    mppi_floats = (
        mppi.temperature,
        mppi.noise_std,
        mppi.alpha,
    )
    if not all(math.isfinite(value) for value in mppi_floats):
        raise ValueError("MPPI floating-point settings must be finite")
    if mppi.temperature <= 0.0:
        raise ValueError("MPPI temperature must be positive")
    if mppi.noise_std <= 0.0:
        raise ValueError("MPPI noise_std must be positive")
    if not 0.0 <= mppi.alpha <= 1.0:
        raise ValueError("MPPI alpha must be between zero and one")
    if mppi.smoothing_window < 1 or mppi.smoothing_window % 2 == 0:
        raise ValueError("MPPI smoothing_window must be a positive odd integer")
    if not 0 <= mppi.smoothing_polynomial < mppi.smoothing_window:
        raise ValueError(
            "MPPI smoothing_polynomial must be nonnegative and smaller than "
            "smoothing_window"
        )
    training_raw = raw["training"]
    training = TrainingConfig(
        epochs=int(training_raw.get("epochs", 1_000)),
        steps_per_epoch=int(training_raw.get("steps_per_epoch", 1)),
        batch_size=int(training_raw.get("batch_size", 256)),
        learning_rate=float(training_raw.get("learning_rate", 1e-4)),
        weight_decay=float(training_raw.get("weight_decay", 1e-6)),
    )
    if min(training.epochs, training.steps_per_epoch, training.batch_size) < 1:
        raise ValueError("training sizes and iteration counts must be positive")
    training_floats = (
        training.learning_rate,
        training.weight_decay,
    )
    if not all(math.isfinite(value) for value in training_floats):
        raise ValueError("training floating-point settings must be finite")
    if training.learning_rate <= 0.0:
        raise ValueError("training learning_rate must be positive")
    if training.weight_decay < 0.0:
        raise ValueError("training weight_decay must be nonnegative")
    simulation_raw = raw["simulation"]
    simulation = SimulationConfig(
        state_noise_std=float(simulation_raw.get("state_noise_std", 0.25))
    )
    if not math.isfinite(simulation.state_noise_std):
        raise ValueError("simulation state_noise_std must be finite")
    if simulation.state_noise_std < 0.0:
        raise ValueError("simulation state_noise_std must be nonnegative")
    demand_profiles = tuple(
        _demand_profile_from_mapping(item, regions)
        for item in raw.get("demand_profiles", [])
    )
    if not demand_profiles:
        raise ValueError("at least one demand profile is required")
    profile_names = [profile.name for profile in demand_profiles]
    if len(set(profile_names)) != len(profile_names):
        raise ValueError("demand profile names must be unique")
    od_pairs = [(profile.origin, profile.destination) for profile in demand_profiles]
    if len(set(od_pairs)) != len(od_pairs):
        raise ValueError("demand profile origin-destination pairs must be unique")
    scenarios = {
        name: _scenario_from_mapping(name, scenario_raw)
        for name, scenario_raw in raw["scenarios"].items()
    }
    if not scenarios:
        raise ValueError("at least one traffic scenario is required")
    return ExperimentConfig(
        seed=int(raw.get("seed", 0)),
        environment=environment,
        objective=objective,
        policy=policy,
        mppi=mppi,
        training=training,
        simulation=simulation,
        demand_profiles=demand_profiles,
        scenarios=scenarios,
    )
