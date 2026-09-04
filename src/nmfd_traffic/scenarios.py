from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import InitialStateScenario, NMFDParameters


def sample_initial_states(
    key: jax.Array,
    count: int,
    scenario: InitialStateScenario,
    params: NMFDParameters,
) -> jax.Array:
    """Draw a reproducible batch of flattened OD accumulation matrices."""

    if count < 1:
        raise ValueError("count must be positive")

    keys = jax.random.split(key, len(scenario.overrides) + 1)
    states = scenario.mean + scenario.std * jax.random.normal(
        keys[0], (count, params.state_dim), dtype=jnp.float32
    )
    states = jnp.clip(states, scenario.min_value, scenario.max_value)

    for override, override_key in zip(scenario.overrides, keys[1:], strict=True):
        values = override.mean + override.std * jax.random.normal(
            override_key, (count,), dtype=jnp.float32
        )
        values = jnp.clip(values, override.min_value, override.max_value)
        flat_index = override.row * params.num_regions + override.column
        states = states.at[:, flat_index].set(values)
    return states
