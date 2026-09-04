"""Construction of reproducible traffic demand and noise trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import DemandProfile, NMFDParameters, TrafficScenario


def demand_trajectory(
    profiles: tuple[DemandProfile, ...],
    params: NMFDParameters,
    steps: int,
) -> jax.Array:
    """Interpolate configured OD demand profiles at the control interval."""

    if steps < 1:
        raise ValueError("steps must be positive")
    times = params.dt * jnp.arange(steps, dtype=jnp.float32)
    demand = jnp.zeros((steps, params.state_dim), dtype=jnp.float32)
    for profile in profiles:
        rates = jnp.interp(
            times,
            jnp.asarray(profile.times, dtype=jnp.float32),
            jnp.asarray(profile.rates, dtype=jnp.float32),
            left=0.0,
            right=0.0,
        )
        index = profile.origin * params.num_regions + profile.destination
        demand = demand.at[:, index].set(rates)
    return demand


def sample_scenario(
    key: jax.Array,
    count: int,
    steps: int,
    scenario: TrafficScenario,
    profiles: tuple[DemandProfile, ...],
    params: NMFDParameters,
    state_noise_std: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample matched initial states, demand, and additive state noise.

    Demand noise is applied only to the configured OD pairs and clipped at zero,
    as in the paper's scenario-shift experiment. State noise is left Gaussian;
    nonnegativity is enforced after it is added to a simulated state.
    """

    if count < 1:
        raise ValueError("count must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")

    demand_key, state_noise_key = jax.random.split(key)
    initial_state = jnp.full(
        (count, params.state_dim),
        scenario.initial_state,
        dtype=jnp.float32,
    )
    base_demand = scenario.demand_scale * demand_trajectory(profiles, params, steps)
    demand = jnp.broadcast_to(base_demand, (count, steps, params.state_dim))

    active_od = jnp.zeros((params.state_dim,), dtype=jnp.float32)
    for profile in profiles:
        index = profile.origin * params.num_regions + profile.destination
        active_od = active_od.at[index].set(1.0)
    demand_noise = scenario.demand_noise_std * jax.random.normal(
        demand_key,
        demand.shape,
        dtype=demand.dtype,
    )
    demand = jnp.clip(demand + demand_noise * active_od, min=0.0)

    state_noise = state_noise_std * jax.random.normal(
        state_noise_key,
        (count, steps, params.state_dim),
        dtype=jnp.float32,
    )
    return initial_state, demand, state_noise
