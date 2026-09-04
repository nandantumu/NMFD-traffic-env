"""Controller-independent traffic objectives.

Both DPC and MPPI call the functions in this module. Keeping the objective in
one place prevents the two controllers from silently optimizing different
traffic metrics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def total_vehicle_time(trajectory: jax.Array, dt: float) -> jax.Array:
    """Return the L1 total vehicle time for each complete trajectory.

    ``trajectory`` has shape ``(..., horizon + 1, state_dim)`` and includes the
    initial and terminal states. Following Equation (16a) of Tumu et al.
    (2024), the objective sums states ``x[1]`` through ``x[N - 1]``. The result
    retains all leading dimensions and has units of vehicle-seconds.
    """

    if trajectory.ndim < 2:
        raise ValueError("trajectory must have time and state dimensions")
    if trajectory.shape[-2] < 3:
        raise ValueError("trajectory must contain at least three states")
    interior_states = trajectory[..., 1:-1, :]
    return dt * jnp.sum(jnp.abs(interior_states), axis=(-2, -1))


def mean_total_vehicle_time(trajectory: jax.Array, dt: float) -> jax.Array:
    """Average :func:`total_vehicle_time` over all leading dimensions."""

    return jnp.mean(total_vehicle_time(trajectory, dt))
