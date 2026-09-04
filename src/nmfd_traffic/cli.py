from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoints import load_parameters, save_parameters
from .config import load_config
from .dynamics import rollout_controls
from .mppi import rollout_naive_mppi
from .policy import create_policy
from .scenarios import sample_initial_states
from .training import create_train_state, make_train_step, rollout_policy
from .visualization import (
    plot_mean_controls,
    plot_network,
    plot_per_region_accumulation,
    plot_total_accumulation,
)


DEFAULT_CONFIG = Path("configs/seven_region.toml")
DEFAULT_CHECKPOINT = Path("checkpoints/dpc_policy.msgpack")


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DPC, naive MPPI, and open gates on an NMFD scenario"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--scenario",
        choices=("in_distribution", "out_of_distribution"),
        default="in_distribution",
    )
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mppi-samples", type=int)
    parser.add_argument("--mppi-iterations", type=int)
    parser.add_argument("--mppi-noise-std", type=float)
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
    vehicle_hours = totals.sum(axis=-1) * dt / 3600.0
    return {
        "terminal_vehicles_mean": float(terminal.mean()),
        "terminal_vehicles_std": float(terminal.std()),
        "vehicle_hours_mean": float(vehicle_hours.mean()),
        "vehicle_hours_std": float(vehicle_hours.std()),
    }


def evaluate_main() -> None:
    args = _evaluation_parser().parse_args()
    config = load_config(args.config)
    if args.rollouts < 1:
        raise ValueError("rollouts must be positive")
    seed = config.seed if args.seed is None else args.seed
    mppi = replace(
        config.naive_mppi,
        samples=args.mppi_samples or config.naive_mppi.samples,
        iterations=args.mppi_iterations or config.naive_mppi.iterations,
        noise_std=args.mppi_noise_std or config.naive_mppi.noise_std,
    )
    policy = create_policy(config.policy, config.environment)
    template = policy.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, config.environment.state_dim), dtype=jnp.float32),
        train=False,
    )["params"]
    parameters = load_parameters(args.checkpoint, template)
    initial = sample_initial_states(
        jax.random.PRNGKey(seed),
        args.rollouts,
        config.scenarios[args.scenario],
        config.environment,
    )

    horizon = config.objective.horizon
    dpc_rollout = jax.jit(
        lambda x: rollout_policy(
            parameters,
            policy.apply,
            x,
            config.environment,
            horizon,
        )
    )
    (dpc_states, dpc_controls), dpc_runtime = _timed_rollout(
        dpc_rollout,
        initial,
        steps=horizon,
        rollouts=args.rollouts,
    )

    mppi_key = jax.random.PRNGKey(seed + 10_000)
    mppi_rollout = jax.jit(
        lambda key, x: rollout_naive_mppi(
            key,
            x,
            horizon,
            config.environment,
            config.objective,
            mppi,
        )
    )
    (mppi_states, mppi_controls), mppi_runtime = _timed_rollout(
        mppi_rollout,
        mppi_key,
        initial,
        steps=horizon,
        rollouts=args.rollouts,
    )

    open_control = jnp.full(
        (args.rollouts, horizon, config.environment.control_dim),
        config.environment.u_high,
    )
    baseline_rollout = jax.jit(
        lambda x, u: rollout_controls(x, u, config.environment)
    )
    baseline_states, baseline_runtime = _timed_rollout(
        baseline_rollout,
        initial,
        open_control,
        steps=horizon,
        rollouts=args.rollouts,
    )

    trajectories = {
        "DPC": np.asarray(dpc_states),
        "Naive MPPI": np.asarray(mppi_states),
        "Open gates": np.asarray(baseline_states),
    }
    controls = {
        "DPC": np.asarray(dpc_controls),
        "Naive MPPI": np.asarray(mppi_controls),
    }

    report = {
        "scenario": args.scenario,
        "rollouts": args.rollouts,
        "horizon": horizon,
        "naive_mppi_settings": {
            "samples": mppi.samples,
            "iterations": mppi.iterations,
            "temperature": mppi.temperature,
            "noise_std": mppi.noise_std,
        },
        "controllers": {
            "dpc": {
                **_trajectory_metrics(trajectories["DPC"], config.environment.dt),
                "runtime_ms_per_step": dpc_runtime,
            },
            "naive_mppi": {
                **_trajectory_metrics(
                    trajectories["Naive MPPI"],
                    config.environment.dt,
                ),
                "runtime_ms_per_step": mppi_runtime,
            },
            "open_gates": {
                **_trajectory_metrics(trajectories["Open gates"], config.environment.dt),
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
    parser = argparse.ArgumentParser(description="Train the deterministic NMFD DPC policy")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/dpc_policy_trained.msgpack"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--sample-pool-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def train_main() -> None:
    args = _training_parser().parse_args()
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    training = replace(
        config.training,
        epochs=args.epochs or config.training.epochs,
        steps_per_epoch=args.steps_per_epoch or config.training.steps_per_epoch,
        sample_pool_size=args.sample_pool_size or config.training.sample_pool_size,
        batch_size=args.batch_size or config.training.batch_size,
    )
    policy = create_policy(config.policy, config.environment)
    key = jax.random.PRNGKey(seed)
    key, initialization_key, pool_key = jax.random.split(key, 3)
    state = create_train_state(initialization_key, policy, training)
    pool = sample_initial_states(
        pool_key,
        training.sample_pool_size,
        config.scenarios["in_distribution"],
        config.environment,
    )
    train_step = make_train_step(config.environment, config.objective)

    for epoch in range(1, training.epochs + 1):
        losses = []
        for _ in range(training.steps_per_epoch):
            key, batch_key = jax.random.split(key)
            indices = jax.random.randint(
                batch_key,
                (training.batch_size,),
                minval=0,
                maxval=training.sample_pool_size,
            )
            state, metrics = train_step(state, pool[indices])
            losses.append(float(metrics["loss"]))
        print(f"epoch {epoch:03d}: loss={np.mean(losses):.6e}")

    save_parameters(
        args.output,
        state.params,
        metadata={
            "epochs": training.epochs,
            "steps_per_epoch": training.steps_per_epoch,
            "seed": seed,
            "source_config": str(args.config),
        },
    )
    print(f"saved {args.output}")
