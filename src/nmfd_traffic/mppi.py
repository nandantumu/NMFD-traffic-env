from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import MPPIConfig, NMFDParameters, ObjectiveConfig
from .dynamics import step


def initial_control_plan(
    batch_size: int,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
) -> jax.Array:
    """Create the fully-open warm start used by the naive MPPI controller."""

    return jnp.full(
        (batch_size, objective.horizon, environment.control_dim),
        environment.u_high,
        dtype=jnp.float32,
    )


def _sample_returns(
    initial_state: jax.Array,
    sampled_controls: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
) -> jax.Array:
    """Return reward-to-go for each sampled sequence and horizon index."""

    batch_size, samples, horizon, _ = sampled_controls.shape
    state = jnp.broadcast_to(
        initial_state[:, None, :],
        (batch_size, samples, environment.state_dim),
    ).reshape(batch_size * samples, environment.state_dim)
    controls = sampled_controls.reshape(
        batch_size * samples,
        horizon,
        environment.control_dim,
    )

    def rollout_step(current: jax.Array, control: jax.Array):
        next_state = step(current, control, environment)
        cost = objective.state_weight * jnp.sum(next_state**2, axis=-1)
        cost += objective.control_weight * jnp.sum(control**2, axis=-1)
        return next_state, -cost

    _, rewards = jax.lax.scan(rollout_step, state, jnp.swapaxes(controls, 0, 1))
    rewards = jnp.swapaxes(rewards, 0, 1).reshape(batch_size, samples, horizon)
    return jnp.flip(jnp.cumsum(jnp.flip(rewards, axis=-1), axis=-1), axis=-1)


def _normalized_weights(returns: jax.Array, config: MPPIConfig) -> jax.Array:
    best = jnp.max(returns, axis=1, keepdims=True)
    worst = jnp.min(returns, axis=1, keepdims=True)
    scaled = (returns - best) / (best - worst + config.damping)
    weights = jnp.exp(scaled / config.temperature)
    return weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-12)


def update_control_plan(
    key: jax.Array,
    state: jax.Array,
    control_plan: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
) -> jax.Array:
    """Perform one sampled MPPI update of a batched control plan."""

    perturbation = config.noise_std * jax.random.normal(
        key,
        (
            state.shape[0],
            config.samples,
            objective.horizon,
            environment.control_dim,
        ),
        dtype=state.dtype,
    )
    lower = environment.u_low - control_plan[:, None, :, :]
    upper = environment.u_high - control_plan[:, None, :, :]
    perturbation = jnp.clip(perturbation, lower, upper)
    sampled_controls = control_plan[:, None, :, :] + perturbation

    returns = _sample_returns(state, sampled_controls, environment, objective)
    weights = _normalized_weights(returns, config)
    update = jnp.einsum("bsh,bshn->bhn", weights, perturbation)
    return jnp.clip(control_plan + update, environment.u_low, environment.u_high)


def select_action(
    key: jax.Array,
    state: jax.Array,
    control_plan: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
) -> tuple[jax.Array, jax.Array]:
    """Refine a warm-start plan and return its first control action."""

    iteration_keys = jax.random.split(key, config.iterations)

    def body(plan: jax.Array, iteration_key: jax.Array):
        return update_control_plan(
            iteration_key,
            state,
            plan,
            environment,
            objective,
            config,
        ), None

    refined_plan, _ = jax.lax.scan(body, control_plan, iteration_keys)
    return refined_plan[:, 0, :], refined_plan


def rollout_naive_mppi(
    key: jax.Array,
    initial_state: jax.Array,
    steps: int,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
) -> tuple[jax.Array, jax.Array]:
    """Run receding-horizon naive MPPI and include the initial state."""

    step_keys = jax.random.split(key, steps)
    plan = initial_control_plan(initial_state.shape[0], environment, objective)

    def body(carry, step_key: jax.Array):
        state, current_plan = carry
        control, refined_plan = select_action(
            step_key,
            state,
            current_plan,
            environment,
            objective,
            config,
        )
        next_state = step(state, control, environment)
        terminal_control = jnp.full_like(refined_plan[:, :1, :], environment.u_high)
        warm_start = jnp.concatenate(
            [refined_plan[:, 1:, :], terminal_control],
            axis=1,
        )
        return (next_state, warm_start), (next_state, control)

    _, (states, controls) = jax.lax.scan(
        body,
        (initial_state, plan),
        step_keys,
    )
    states = jnp.swapaxes(states, 0, 1)
    controls = jnp.swapaxes(controls, 0, 1)
    states = jnp.concatenate([initial_state[:, None, :], states], axis=1)
    return states, controls
