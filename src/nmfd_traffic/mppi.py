"""JAX implementation of Williams et al.'s 2017 information-theoretic MPPI.

The implementation follows Algorithms 1 and 2 from the paper selected in
docs/mppi.md. The configured scalar noise_std represents the isotropic
covariance Sigma = noise_std**2 * I.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import MPPIConfig, NMFDParameters, ObjectiveConfig
from .dynamics import step, step_with_noise
from .objective import total_vehicle_time


def initial_control_plan(
    batch_size: int,
    environment: NMFDParameters,
    config: MPPIConfig,
) -> jax.Array:
    """Create the fully-open warm start used by the MPPI controller."""

    return jnp.full(
        (batch_size, config.planning_horizon, environment.control_dim),
        environment.u_high,
        dtype=jnp.float32,
    )


def _sample_control_sequences(
    control_plan: jax.Array,
    perturbations: jax.Array,
    alpha: float,
) -> jax.Array:
    """Build the nominal- and zero-centered samples from Algorithm 1.

    The first (1 - alpha) fraction is sampled around the nominal plan. The
    remaining recovery samples are centered at zero. The perturbations stay
    unmodified because the paper's mean update uses the raw Gaussian samples.
    """

    sample_count = perturbations.shape[1]
    nominal_sample = jnp.arange(sample_count) < (1.0 - alpha) * sample_count
    nominal_sample = nominal_sample[None, :, None, None]
    return jnp.where(
        nominal_sample,
        control_plan[:, None, :, :] + perturbations,
        perturbations,
    )


def _rollout_costs(
    initial_state: jax.Array,
    sampled_inputs: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Evaluate S(V; x0) for every complete sampled trajectory.

    This calls the same L1 total-vehicle-time objective as DPC. The separate
    importance-sampling correction is added later by ``_trajectory_scores``.
    """

    batch_size, samples, horizon, _ = sampled_inputs.shape
    state = jnp.broadcast_to(
        initial_state[:, None, :],
        (batch_size, samples, environment.state_dim),
    ).reshape(batch_size * samples, environment.state_dim)

    applied_controls = jnp.clip(
        sampled_inputs,
        environment.u_low,
        environment.u_high,
    )
    flat_controls = applied_controls.reshape(
        batch_size * samples,
        horizon,
        environment.control_dim,
    )
    if demand is None:
        demand = jnp.zeros(
            (batch_size, horizon, environment.state_dim),
            dtype=initial_state.dtype,
        )
    expected_demand_shape = (batch_size, horizon, environment.state_dim)
    if demand.shape != expected_demand_shape:
        raise ValueError(f"demand must have shape {expected_demand_shape}")
    flat_demand = jnp.broadcast_to(
        demand[:, None, :, :],
        (batch_size, samples, horizon, environment.state_dim),
    ).reshape(batch_size * samples, horizon, environment.state_dim)

    def rollout_step(current_state: jax.Array, inputs: tuple[jax.Array, ...]):
        control, current_demand = inputs
        next_state = step(current_state, control, environment, current_demand)
        return next_state, next_state

    _, states = jax.lax.scan(
        rollout_step,
        state,
        (
            jnp.swapaxes(flat_controls, 0, 1),
            jnp.swapaxes(flat_demand, 0, 1),
        ),
    )
    states = jnp.swapaxes(states, 0, 1)
    trajectory = jnp.concatenate([state[:, None, :], states], axis=1)
    costs = total_vehicle_time(trajectory, environment.dt)
    return costs.reshape(batch_size, samples)


def _trajectory_scores(
    initial_state: jax.Array,
    control_plan: jax.Array,
    sampled_inputs: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Add the 2017 paper's importance-sampling correction to rollout costs."""

    rollout_cost = _rollout_costs(
        initial_state,
        sampled_inputs,
        environment,
        objective,
        demand,
    )
    inverse_variance = 1.0 / config.noise_std**2
    gamma = config.temperature * (1.0 - config.alpha)
    importance_correction = (
        gamma
        * inverse_variance
        * jnp.sum(
            control_plan[:, None, :, :] * sampled_inputs,
            axis=(-2, -1),
        )
    )
    return rollout_cost + importance_correction


def _normalized_weights(scores: jax.Array, config: MPPIConfig) -> jax.Array:
    """Convert complete-trajectory scores to Algorithm 2 weights."""

    minimum = jnp.min(scores, axis=1, keepdims=True)
    unnormalized = jnp.exp(-(scores - minimum) / config.temperature)
    return unnormalized / jnp.sum(unnormalized, axis=1, keepdims=True)


def _savgol_coefficients(config: MPPIConfig, dtype: jnp.dtype) -> jax.Array:
    """Return center-point Savitzky--Golay coefficients using JAX only."""

    half_window = config.smoothing_window // 2
    positions = jnp.arange(-half_window, half_window + 1, dtype=dtype)
    powers = jnp.arange(config.smoothing_polynomial + 1)
    design = positions[:, None] ** powers[None, :]
    return jnp.linalg.pinv(design)[0]


def _smooth_control_update(update: jax.Array, config: MPPIConfig) -> jax.Array:
    """Apply the paper's Savitzky--Golay filter along the plan horizon."""

    if config.smoothing_window == 1:
        return update

    coefficients = _savgol_coefficients(config, update.dtype)
    half_window = config.smoothing_window // 2
    padded = jnp.pad(
        update,
        ((0, 0), (half_window, half_window), (0, 0)),
        mode="edge",
    )
    horizon = update.shape[1]
    return sum(
        coefficients[index] * padded[:, index : index + horizon, :]
        for index in range(config.smoothing_window)
    )


def _weighted_perturbation_update(
    weights: jax.Array,
    perturbations: jax.Array,
) -> jax.Array:
    """Average every perturbation sequence with one weight per trajectory."""

    return jnp.einsum("bs,bshn->bhn", weights, perturbations)


def _update_control_plan_from_perturbations(
    state: jax.Array,
    control_plan: jax.Array,
    perturbations: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Perform one deterministic MPPI update from supplied Gaussian samples."""

    sampled_inputs = _sample_control_sequences(
        control_plan,
        perturbations,
        config.alpha,
    )
    scores = _trajectory_scores(
        state,
        control_plan,
        sampled_inputs,
        environment,
        objective,
        config,
        demand,
    )
    weights = _normalized_weights(scores, config)
    raw_update = _weighted_perturbation_update(weights, perturbations)
    return control_plan + _smooth_control_update(raw_update, config)


def update_control_plan(
    key: jax.Array,
    state: jax.Array,
    control_plan: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Perform one paper-aligned sampled update of a batched control plan."""

    perturbations = config.noise_std * jax.random.normal(
        key,
        (
            state.shape[0],
            config.samples,
            config.planning_horizon,
            environment.control_dim,
        ),
        dtype=state.dtype,
    )
    return _update_control_plan_from_perturbations(
        state,
        control_plan,
        perturbations,
        environment,
        objective,
        config,
        demand,
    )


def select_action(
    key: jax.Array,
    state: jax.Array,
    control_plan: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Refine a warm-start plan and return its admissible first action."""

    iteration_keys = jax.random.split(key, config.iterations)

    def body(plan: jax.Array, iteration_key: jax.Array):
        return update_control_plan(
            iteration_key,
            state,
            plan,
            environment,
            objective,
            config,
            demand,
        ), None

    refined_plan, _ = jax.lax.scan(body, control_plan, iteration_keys)
    action = jnp.clip(
        refined_plan[:, 0, :],
        environment.u_low,
        environment.u_high,
    )
    return action, refined_plan


def rollout_mppi(
    key: jax.Array,
    initial_state: jax.Array,
    steps: int,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run receding-horizon MPPI with known demand and noisy plant states."""

    step_keys = jax.random.split(key, steps)
    plan = initial_control_plan(initial_state.shape[0], environment, config)
    batch_size = initial_state.shape[0]
    expected_shape = (batch_size, steps, environment.state_dim)
    if demand is None:
        demand = jnp.zeros(expected_shape, dtype=initial_state.dtype)
    if state_noise is None:
        state_noise = jnp.zeros(expected_shape, dtype=initial_state.dtype)
    if demand.shape != expected_shape:
        raise ValueError(f"demand must have shape {expected_shape}")
    if state_noise.shape != expected_shape:
        raise ValueError(f"state_noise must have shape {expected_shape}")
    demand_padding = jnp.zeros(
        (batch_size, config.planning_horizon - 1, environment.state_dim),
        dtype=demand.dtype,
    )
    padded_demand = jnp.concatenate([demand, demand_padding], axis=1)

    def body(carry, inputs: tuple[jax.Array, ...]):
        state, current_plan = carry
        time_index, step_key, current_demand, current_noise = inputs
        forecast_demand = jax.lax.dynamic_slice_in_dim(
            padded_demand,
            time_index,
            config.planning_horizon,
            axis=1,
        )
        control, refined_plan = select_action(
            step_key,
            state,
            current_plan,
            environment,
            objective,
            config,
            forecast_demand,
        )
        next_state = step_with_noise(
            state,
            control,
            environment,
            current_demand,
            current_noise,
        )
        terminal_control = jnp.full_like(refined_plan[:, :1, :], environment.u_high)
        warm_start = jnp.concatenate(
            [refined_plan[:, 1:, :], terminal_control],
            axis=1,
        )
        return (next_state, warm_start), (next_state, control)

    _, (states, controls) = jax.lax.scan(
        body,
        (initial_state, plan),
        (
            jnp.arange(steps),
            step_keys,
            jnp.swapaxes(demand, 0, 1),
            jnp.swapaxes(state_noise, 0, 1),
        ),
    )
    states = jnp.swapaxes(states, 0, 1)
    controls = jnp.swapaxes(controls, 0, 1)
    states = jnp.concatenate([initial_state[:, None, :], states], axis=1)
    return states, controls


def rollout_naive_mppi(
    key: jax.Array,
    initial_state: jax.Array,
    steps: int,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
    config: MPPIConfig,
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Backward-compatible name for rollout_mppi."""

    return rollout_mppi(
        key,
        initial_state,
        steps,
        environment,
        objective,
        config,
        demand,
        state_noise,
    )
