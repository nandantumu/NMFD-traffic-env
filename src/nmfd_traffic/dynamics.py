from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import NMFDParameters


def dynamics(
    state: jax.Array,
    control: jax.Array,
    params: NMFDParameters,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the continuous-time NMFD derivative.

    The last dimension of ``state`` and ``demand`` is a flattened row-major
    ``(current region, final destination)`` matrix. The last dimension of
    ``control`` is a flattened row-major ``(sending region, receiving region)``
    matrix. Leading batch dimensions are supported. Demand is an optional
    exogenous arrival rate for each state cell.
    """

    regions = params.num_regions
    expected = params.state_dim
    if state.shape[-1] != expected or control.shape[-1] != expected:
        raise ValueError(f"state and control must have last dimension {expected}")

    original_shape = state.shape
    x = jnp.reshape(state, (-1, regions, regions))
    u = jnp.reshape(control, (-1, regions, regions))
    if demand is None:
        d = jnp.zeros_like(x)
    else:
        if demand.shape[-1] != expected:
            raise ValueError(f"demand must have last dimension {expected}")
        d = jnp.reshape(demand, (-1, regions, regions))
        d = jnp.broadcast_to(d, x.shape)

    accumulation = jnp.sum(x, axis=-1)
    production = (
        params.a[None, :] * accumulation**3
        + params.b[None, :] * accumulation**2
        + params.c[None, :] * accumulation
    )

    safe_accumulation = jnp.clip(jnp.sum(x, axis=-1), min=1e-8)
    destination_fraction = x / safe_accumulation[..., None]
    transfer = (
        params.theta[None, :, :, :]
        * destination_fraction[:, :, None, :]
        * production[:, :, None, None]
    )

    diagonal = jnp.eye(regions, dtype=bool)
    transfer = jnp.where(diagonal[None, :, None, :], 0.0, transfer)
    completion = jnp.diagonal(destination_fraction, axis1=-2, axis2=-1) * production

    off_diagonal = 1.0 - jnp.eye(regions, dtype=x.dtype)
    effective_control = params.adjacency[None, :, :] * u * off_diagonal[None, :, :]
    outflow = jnp.einsum("bih,bihj->bij", effective_control, transfer)
    inflow = jnp.einsum("bhi,bhij->bij", effective_control, transfer)

    derivative = d - outflow + inflow
    diagonal_derivative = jnp.diagonal(d, axis1=-2, axis2=-1) - completion
    diagonal_derivative += jnp.diagonal(inflow, axis1=-2, axis2=-1)
    derivative = derivative * off_diagonal[None, :, :]
    derivative += diagonal[None, :, :] * diagonal_derivative[:, :, None]
    return jnp.reshape(derivative, original_shape)


def rk4_step(
    state: jax.Array,
    control: jax.Array,
    params: NMFDParameters,
    dt: float,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Take one fourth-order Runge--Kutta step."""

    k1 = dynamics(state, control, params, demand)
    k2 = dynamics(state + 0.5 * dt * k1, control, params, demand)
    k3 = dynamics(state + 0.5 * dt * k2, control, params, demand)
    k4 = dynamics(state + dt * k3, control, params, demand)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def step(
    state: jax.Array,
    control: jax.Array,
    params: NMFDParameters,
    demand: jax.Array | None = None,
) -> jax.Array:
    """Advance one control interval and enforce nonnegative accumulation."""

    control = jnp.clip(control, params.u_low, params.u_high)
    substep_dt = params.dt / params.n_substeps

    def body(_: int, current: jax.Array) -> jax.Array:
        next_state = rk4_step(current, control, params, substep_dt, demand)
        return jnp.clip(next_state, min=0.0)

    return jax.lax.fori_loop(0, params.n_substeps, body, state)


def step_with_noise(
    state: jax.Array,
    control: jax.Array,
    params: NMFDParameters,
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> jax.Array:
    """Advance the NMFD model, add state noise, and enforce nonnegativity."""

    next_state = step(state, control, params, demand)
    if state_noise is not None:
        if state_noise.shape[-1] != params.state_dim:
            raise ValueError(f"state_noise must have last dimension {params.state_dim}")
        next_state = next_state + state_noise
    return jnp.clip(next_state, min=0.0)


def rollout_controls(
    initial_state: jax.Array,
    controls: jax.Array,
    params: NMFDParameters,
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> jax.Array:
    """Roll out controls under optional demand/noise and include the initial state."""

    batch_size, horizon, _ = controls.shape
    if demand is None:
        demand = jnp.zeros(
            (batch_size, horizon, params.state_dim), dtype=initial_state.dtype
        )
    if state_noise is None:
        state_noise = jnp.zeros_like(demand)
    if demand.shape != (batch_size, horizon, params.state_dim):
        raise ValueError("demand must match the control batch and horizon")
    if state_noise.shape != demand.shape:
        raise ValueError("state_noise must have the same shape as demand")

    def body(current: jax.Array, inputs: tuple[jax.Array, ...]):
        control, current_demand, current_noise = inputs
        next_state = step_with_noise(
            current,
            control,
            params,
            current_demand,
            current_noise,
        )
        return next_state, next_state

    scan_inputs = tuple(
        jnp.swapaxes(value, 0, 1) for value in (controls, demand, state_noise)
    )
    _, states = jax.lax.scan(body, initial_state, scan_inputs)
    states = jnp.swapaxes(states, 0, 1)
    return jnp.concatenate([initial_state[:, None, :], states], axis=1)
