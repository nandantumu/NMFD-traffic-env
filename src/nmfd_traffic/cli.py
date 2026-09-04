from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoints import load_parameters, save_parameters
from .config import load_config
from .dynamics import rollout_controls
from .policy import create_policy
from .scenarios import sample_initial_states
from .training import create_train_state, make_train_step, rollout_policy


DEFAULT_CONFIG = Path("configs/seven_region.toml")
DEFAULT_CHECKPOINT = Path("checkpoints/dpc_policy.msgpack")


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DPC on an NMFD scenario")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--scenario",
        choices=("in_distribution", "out_of_distribution"),
        default="in_distribution",
    )
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int)
    return parser


def evaluate_main() -> None:
    args = _evaluation_parser().parse_args()
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
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

    dpc_rollout = jax.jit(
        lambda x: rollout_policy(
            parameters,
            policy.apply,
            x,
            config.environment,
            config.objective.horizon,
        )
    )
    states, _ = dpc_rollout(initial)
    states.block_until_ready()

    open_control = jnp.full(
        (args.rollouts, config.objective.horizon, config.environment.control_dim),
        config.environment.u_high,
    )
    baseline = jax.jit(
        lambda x, u: rollout_controls(x, u, config.environment)
    )(initial, open_control)
    baseline.block_until_ready()

    def metrics(trajectory: jax.Array) -> dict[str, float]:
        totals = np.asarray(trajectory[:, 1:, :].sum(axis=-1))
        terminal = totals[:, -1]
        vehicle_hours = totals.sum(axis=-1) * config.environment.dt / 3600.0
        return {
            "terminal_vehicles_mean": float(terminal.mean()),
            "terminal_vehicles_std": float(terminal.std()),
            "vehicle_hours_mean": float(vehicle_hours.mean()),
            "vehicle_hours_std": float(vehicle_hours.std()),
        }

    report = {
        "scenario": args.scenario,
        "rollouts": args.rollouts,
        "dpc": metrics(states),
        "open_gates_baseline": metrics(baseline),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


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
