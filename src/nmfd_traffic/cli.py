from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoints import load_parameters, save_parameters
from .config import ExperimentConfig, load_config
from .dynamics import rollout_controls
from .mppi import rollout_mppi
from .objective import total_vehicle_time
from .policy import create_policy
from .scenarios import sample_scenario
from .training import create_train_state, make_train_step, rollout_policy
from .visualization import (
    plot_mean_controls,
    plot_network,
    plot_per_region_accumulation,
    plot_total_accumulation,
)

DEFAULT_CONFIG_RESOURCE = "seven_region.toml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _resolve_asset(
    stack: ExitStack,
    supplied_path: Path | None,
    resource_name: str,
) -> tuple[Path, str]:
    """Resolve an explicit path or materialize a packaged default asset."""

    if supplied_path is not None:
        return supplied_path, str(supplied_path.resolve())
    resource = resources.files("nmfd_traffic").joinpath("data", resource_name)
    materialized = Path(stack.enter_context(resources.as_file(resource)))
    return materialized, f"package:nmfd_traffic/data/{resource_name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return metadata.version("nmfd-traffic-env")
    except metadata.PackageNotFoundError:
        return "unknown"


def _repository_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _repository_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _reproducibility_metadata(
    config_path: Path,
    checkpoint_path: Path,
    config_reference: str,
    checkpoint_reference: str,
    seed: int,
) -> dict[str, object]:
    checkpoint = {
        "source": checkpoint_reference,
        "sha256": _sha256(checkpoint_path),
    }
    sidecar = checkpoint_path.with_suffix(".json")
    if sidecar.is_file():
        checkpoint["provenance"] = {
            "source": str(sidecar.resolve()),
            "sha256": _sha256(sidecar),
        }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "config": {
            "source": config_reference,
            "sha256": _sha256(config_path),
        },
        "checkpoint": checkpoint,
        "software": {
            "nmfd_traffic_env": _package_version(),
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
        "repository_commit": _repository_commit(config_path),
        "repository_dirty": _repository_dirty(config_path),
    }


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DPC, MPPI, and open gates on an NMFD scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="experiment TOML; defaults to the packaged seven-region configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DPC parameters produced by nmfd-train",
    )
    parser.add_argument(
        "--scenario",
        default="nominal",
        help="name of any scenario defined by the selected configuration",
    )
    parser.add_argument("--rollouts", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mppi-samples", type=_positive_int)
    parser.add_argument("--mppi-iterations", type=_positive_int)
    parser.add_argument("--mppi-temperature", type=_positive_float)
    parser.add_argument("--mppi-noise-std", type=_positive_float)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _block_until_ready(tree):
    return jax.tree_util.tree_map(lambda value: value.block_until_ready(), tree)


def _timed_rollout(function, *arguments, steps: int, rollouts: int):
    _block_until_ready(function(*arguments))
    start = time.perf_counter()
    result = _block_until_ready(function(*arguments))
    runtime_ms = (time.perf_counter() - start) * 1000.0 / (steps * rollouts)
    return result, runtime_ms


def _trajectory_metrics(trajectory: np.ndarray, dt: float) -> dict[str, float]:
    totals = trajectory[:, 1:, :].sum(axis=-1)
    terminal = totals[:, -1]
    vehicle_time = np.asarray(total_vehicle_time(jnp.asarray(trajectory), dt))
    return {
        "terminal_vehicles_mean": float(terminal.mean()),
        "terminal_vehicles_std": float(terminal.std()),
        "total_vehicle_time_seconds_mean": float(vehicle_time.mean()),
        "total_vehicle_time_seconds_std": float(vehicle_time.std()),
    }


def evaluate_main(argv: Sequence[str] | None = None) -> None:
    """Run the evaluation CLI with explicit or packaged experiment assets."""

    parser = _evaluation_parser()
    args = parser.parse_args(argv)
    with ExitStack() as stack:
        config_path, config_reference = _resolve_asset(
            stack,
            args.config,
            DEFAULT_CONFIG_RESOURCE,
        )
        checkpoint_path = args.checkpoint
        checkpoint_reference = str(checkpoint_path.resolve())
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint_path}")
        config = load_config(config_path)
        if args.scenario not in config.scenarios:
            available = ", ".join(sorted(config.scenarios))
            parser.error(
                f"unknown scenario {args.scenario!r}; available scenarios: {available}"
            )
        _run_evaluation(
            args,
            config,
            config_path,
            checkpoint_path,
            config_reference,
            checkpoint_reference,
        )


def _run_evaluation(
    args: argparse.Namespace,
    config: ExperimentConfig,
    config_path: Path,
    checkpoint_path: Path,
    config_reference: str,
    checkpoint_reference: str,
) -> None:
    seed = config.seed if args.seed is None else args.seed
    mppi = replace(
        config.mppi,
        samples=(
            config.mppi.samples if args.mppi_samples is None else args.mppi_samples
        ),
        iterations=(
            config.mppi.iterations
            if args.mppi_iterations is None
            else args.mppi_iterations
        ),
        temperature=(
            config.mppi.temperature
            if args.mppi_temperature is None
            else args.mppi_temperature
        ),
        noise_std=(
            config.mppi.noise_std
            if args.mppi_noise_std is None
            else args.mppi_noise_std
        ),
    )
    policy = create_policy(config.policy, config.environment)
    template = policy.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, config.environment.state_dim), dtype=jnp.float32),
        train=False,
    )["params"]
    parameters = load_parameters(checkpoint_path, template)
    initial, demand, state_noise = sample_scenario(
        jax.random.PRNGKey(seed),
        args.rollouts,
        config.objective.horizon,
        config.scenarios[args.scenario],
        config.demand_profiles,
        config.environment,
        config.simulation.state_noise_std,
    )

    horizon = config.objective.horizon
    dpc_rollout = jax.jit(
        lambda x, d, noise: rollout_policy(
            parameters,
            policy.apply,
            x,
            config.environment,
            horizon,
            d,
            noise,
        )
    )
    (dpc_states, dpc_controls), dpc_runtime = _timed_rollout(
        dpc_rollout,
        initial,
        demand,
        state_noise,
        steps=horizon,
        rollouts=args.rollouts,
    )

    mppi_key = jax.random.PRNGKey(seed + 10_000)
    mppi_rollout = jax.jit(
        lambda key, x, d, noise: rollout_mppi(
            key,
            x,
            horizon,
            config.environment,
            config.objective,
            mppi,
            d,
            noise,
        )
    )
    (mppi_states, mppi_controls), mppi_runtime = _timed_rollout(
        mppi_rollout,
        mppi_key,
        initial,
        demand,
        state_noise,
        steps=horizon,
        rollouts=args.rollouts,
    )

    open_control = jnp.full(
        (args.rollouts, horizon, config.environment.control_dim),
        config.environment.u_high,
    )
    baseline_rollout = jax.jit(
        lambda x, u, d, noise: rollout_controls(
            x,
            u,
            config.environment,
            d,
            noise,
        )
    )
    baseline_states, baseline_runtime = _timed_rollout(
        baseline_rollout,
        initial,
        open_control,
        demand,
        state_noise,
        steps=horizon,
        rollouts=args.rollouts,
    )

    trajectories = {
        "DPC": np.asarray(dpc_states),
        "MPPI": np.asarray(mppi_states),
        "Open gates": np.asarray(baseline_states),
    }
    controls = {
        "DPC": np.asarray(dpc_controls),
        "MPPI": np.asarray(mppi_controls),
    }

    report = {
        "reproducibility": _reproducibility_metadata(
            config_path,
            checkpoint_path,
            config_reference,
            checkpoint_reference,
            seed,
        ),
        "scenario": args.scenario,
        "rollouts": args.rollouts,
        "horizon": horizon,
        "objective": {
            "name": "L1 total vehicle time",
            "formula": "dt * sum(k=1..N-1, ||x[k]||_1)",
            "units": "vehicle-seconds",
            "shared_by": ["dpc", "mppi"],
            "reference": "Tumu et al. (2024), Equation (16a)",
        },
        "experiment": {
            "environment_model": "NMFD",
            "acyclic_plant_model_used": False,
            "dt_seconds": config.environment.dt,
            "state_noise_std": config.simulation.state_noise_std,
            "control_bounds": [
                config.environment.u_low,
                config.environment.u_high,
            ],
            "fixed_routing": "equal split over shortest-path next hops",
            "scenario": asdict(config.scenarios[args.scenario]),
            "demand_profiles": [asdict(profile) for profile in config.demand_profiles],
        },
        "dpc_settings": {
            "formulation": "perimeter control only",
            "paper": (
                "Differentiable Predictive Control for Large-Scale Urban Road Networks"
            ),
            "arxiv": "2406.10433",
            "policy": asdict(config.policy),
            "output_map": "u_low + (u_high - u_low) * sigmoid(raw_control)",
        },
        "mppi_settings": {
            "formulation": "Williams et al. 2017 information-theoretic MPC",
            "doi": "10.1109/TRO.2018.2865891",
            "planning_horizon": mppi.planning_horizon,
            "samples": mppi.samples,
            "iterations": mppi.iterations,
            "temperature": mppi.temperature,
            "noise_std": mppi.noise_std,
            "covariance": "noise_std^2 * identity",
            "alpha": mppi.alpha,
            "gamma": mppi.temperature * (1.0 - mppi.alpha),
            "smoothing_window": mppi.smoothing_window,
            "smoothing_polynomial": mppi.smoothing_polynomial,
            "traffic_objective": "shared L1 total vehicle time",
        },
        "controllers": {
            "dpc": {
                **_trajectory_metrics(trajectories["DPC"], config.environment.dt),
                "runtime_ms_per_step": dpc_runtime,
            },
            "mppi": {
                **_trajectory_metrics(
                    trajectories["MPPI"],
                    config.environment.dt,
                ),
                "runtime_ms_per_step": mppi_runtime,
            },
            "open_gates": {
                **_trajectory_metrics(
                    trajectories["Open gates"], config.environment.dt
                ),
                "runtime_ms_per_step": baseline_runtime,
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.scenario
    metrics_path = args.output_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not args.no_plots:
        display_name = args.scenario.replace("_", " ").title()
        plot_total_accumulation(
            trajectories,
            config.environment.dt,
            args.output_dir / f"{prefix}_total_accumulation.png",
            title=f"Total network accumulation — {display_name}",
        )
        plot_per_region_accumulation(
            trajectories,
            config.environment,
            args.output_dir / f"{prefix}_per_region.png",
            title=f"Per-region accumulation — {display_name}",
        )
        plot_mean_controls(
            controls,
            config.environment,
            args.output_dir / f"{prefix}_mean_controls.png",
            title=f"Mean perimeter controls — {display_name}",
        )
        plot_network(
            config.environment,
            args.output_dir / "network_topology.png",
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote evaluation artifacts to {args.output_dir}")


def _training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the deterministic NMFD DPC policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="experiment TOML; defaults to the packaged seven-region configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/dpc_policy.msgpack"),
    )
    parser.add_argument(
        "--scenario",
        default="nominal",
        help="traffic scenario used for training",
    )
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--steps-per-epoch", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--seed", type=int)
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    """Run the DPC training CLI with an explicit or packaged configuration."""

    parser = _training_parser()
    args = parser.parse_args(argv)
    with ExitStack() as stack:
        config_path, config_reference = _resolve_asset(
            stack,
            args.config,
            DEFAULT_CONFIG_RESOURCE,
        )
        config = load_config(config_path)
        if args.scenario not in config.scenarios:
            available = ", ".join(sorted(config.scenarios))
            parser.error(
                f"unknown scenario {args.scenario!r}; available scenarios: {available}"
            )
        _run_training(args, config, config_path, config_reference)


def _run_training(
    args: argparse.Namespace,
    config: ExperimentConfig,
    config_path: Path,
    config_reference: str,
) -> None:
    seed = config.seed if args.seed is None else args.seed
    training = replace(
        config.training,
        epochs=config.training.epochs if args.epochs is None else args.epochs,
        steps_per_epoch=(
            config.training.steps_per_epoch
            if args.steps_per_epoch is None
            else args.steps_per_epoch
        ),
        batch_size=(
            config.training.batch_size if args.batch_size is None else args.batch_size
        ),
    )
    policy = create_policy(config.policy, config.environment)
    key = jax.random.PRNGKey(seed)
    key, initialization_key = jax.random.split(key)
    state = create_train_state(initialization_key, policy, training)
    train_step = make_train_step(config.environment, config.objective)

    for epoch in range(1, training.epochs + 1):
        losses = []
        for _ in range(training.steps_per_epoch):
            key, scenario_key = jax.random.split(key)
            initial, demand, state_noise = sample_scenario(
                scenario_key,
                training.batch_size,
                config.objective.horizon,
                config.scenarios[args.scenario],
                config.demand_profiles,
                config.environment,
                config.simulation.state_noise_std,
            )
            state, metrics = train_step(state, initial, demand, state_noise)
            losses.append(float(metrics["loss"]))
        print(f"epoch {epoch:03d}: loss={np.mean(losses):.6e}")

    save_parameters(
        args.output,
        state.params,
        metadata={
            "format_version": 1,
            "seed": seed,
            "paper": {
                "title": (
                    "Differentiable Predictive Control for Large-Scale Urban "
                    "Road Networks"
                ),
                "arxiv": "2406.10433",
            },
            "policy": {
                **asdict(config.policy),
                "controller": "perimeter control only",
                "output_map": "sigmoid affine to control bounds",
            },
            "objective": {
                "name": "L1 total vehicle time",
                "formula": "dt * sum(k=1..N-1, ||x[k]||_1)",
                "horizon": config.objective.horizon,
            },
            "environment": {
                "model": "NMFD",
                "acyclic_plant_model_used": False,
                "dt_seconds": config.environment.dt,
                "n_substeps": config.environment.n_substeps,
                "control_bounds": [
                    config.environment.u_low,
                    config.environment.u_high,
                ],
                "routing": "fixed shortest paths",
            },
            "simulation": asdict(config.simulation),
            "scenario": asdict(config.scenarios[args.scenario]),
            "demand_profiles": [asdict(profile) for profile in config.demand_profiles],
            "training": {
                **asdict(training),
                "completed_updates": training.epochs * training.steps_per_epoch,
                "optimizer": "AdamW",
            },
            "source_config": config_reference,
            "source_config_sha256": _sha256(config_path),
            "software": {
                "nmfd_traffic_env": _package_version(),
                "python": platform.python_version(),
                "jax": jax.__version__,
            },
            "repository_commit": _repository_commit(config_path),
            "repository_dirty": _repository_dirty(config_path),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    print(f"saved {args.output}")
