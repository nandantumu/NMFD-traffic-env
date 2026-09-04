from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nmfd_traffic import dynamics, load_config, step


CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_zero_state_is_equilibrium_without_demand():
    params = load_config(CONFIG).environment
    state = jnp.zeros((2, params.state_dim))
    control = jnp.full_like(state, params.u_high)
    np.testing.assert_array_equal(np.asarray(dynamics(state, control, params)), 0.0)


def test_demand_enters_at_zero_state():
    params = load_config(CONFIG).environment
    state = jnp.zeros((1, params.state_dim))
    control = jnp.full_like(state, params.u_high)
    demand = jnp.linspace(0.0, 0.1, params.state_dim)[None, :]
    np.testing.assert_allclose(dynamics(state, control, params, demand), demand)


def test_step_is_jittable_and_nonnegative():
    params = load_config(CONFIG).environment
    state = jnp.full((3, params.state_dim), 100.0)
    control = jnp.full_like(state, params.u_high)
    next_state = jax.jit(lambda x, u: step(x, u, params))(state, control)
    assert next_state.shape == state.shape
    assert np.all(np.asarray(next_state) >= 0.0)
