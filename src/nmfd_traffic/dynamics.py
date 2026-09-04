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

    The last dimension of ``state``, ``control``, and ``demand`` is flattened
    row-major ``(origin region, destination region)``. Leading batch dimensions
    are supported. Demand is an optional exogenous OD arrival rate.
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

    accumulation = jnp.clip(jnp.sum(x, axis=-1), 0.0, 10_000.0)
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


def rollout_controls(
    initial_state: jax.Array,
    controls: jax.Array,
    params: NMFDParameters,
) -> jax.Array:
    """Roll out a supplied control sequence and include the initial state."""

    def body(current: jax.Array, control: jax.Array):
        next_state = step(current, control, params)
        return next_state, next_state

    _, states = jax.lax.scan(body, initial_state, jnp.swapaxes(controls, 0, 1))
    states = jnp.swapaxes(states, 0, 1)
    return jnp.concatenate([initial_state[:, None, :], states], axis=1)
