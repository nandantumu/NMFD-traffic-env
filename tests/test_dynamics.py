from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_interregional_flow_conserves_vehicles_before_trip_completion():
    params = load_config(CONFIG).environment
    state = jnp.zeros((1, params.state_dim)).at[0, 1].set(500.0)
    control = jnp.full_like(state, params.u_high)

    derivative = np.asarray(dynamics(state, control, params))[0]

    assert derivative[1] < 0.0  # Vehicles leave current region 0.
    assert derivative[1 * params.num_regions + 1] > 0.0  # They enter region 1.
    assert derivative.sum() == pytest.approx(0.0, abs=1e-7)


def test_vehicles_already_at_their_destination_complete_trips():
    params = load_config(CONFIG).environment
    accumulation = 1_200.0
    state = jnp.zeros((1, params.state_dim)).at[0, 0].set(accumulation)
    control = jnp.full_like(state, params.u_high)

    derivative = np.asarray(dynamics(state, control, params))[0]
    production = (
        float(params.a[0]) * accumulation**3
        + float(params.b[0]) * accumulation**2
        + float(params.c[0]) * accumulation
    )

    assert derivative[0] == pytest.approx(-production, rel=1e-5)
    assert derivative.sum() == pytest.approx(-production, rel=1e-5)


def test_production_uses_accumulation_above_ten_thousand_without_clipping():
    params = load_config(CONFIG).environment
    accumulation = 12_000.0
    state = jnp.zeros((1, params.state_dim)).at[0, 0].set(accumulation)
    control = jnp.full_like(state, params.u_high)

    derivative = np.asarray(dynamics(state, control, params))[0]
    production = (
        float(params.a[0]) * accumulation**3
        + float(params.b[0]) * accumulation**2
        + float(params.c[0]) * accumulation
    )

    assert derivative[0] == pytest.approx(-production, rel=1e-5)


def test_diagonal_and_nonadjacent_controls_have_no_effect():
    params = load_config(CONFIG).environment
    state = jnp.arange(1, params.state_dim + 1, dtype=jnp.float32)[None, :]
    baseline_control = jnp.full_like(state, params.u_low)
    changed_control = baseline_control.at[0, 0].set(params.u_high)
    changed_control = changed_control.at[0, 2].set(params.u_high)

    baseline = dynamics(state, baseline_control, params)
    changed = dynamics(state, changed_control, params)

    np.testing.assert_allclose(changed, baseline)


def test_step_matches_vmap_and_propagates_demand():
    params = load_config(CONFIG).environment
    states = jnp.full((3, params.state_dim), 100.0)
    controls = jnp.full_like(states, params.u_high)

    batched = step(states, controls, params)
    mapped = jax.vmap(lambda state, control: step(state, control, params))(
        states,
        controls,
    )
    np.testing.assert_allclose(mapped, batched, rtol=1e-5, atol=1e-5)

    zero_state = jnp.zeros((1, params.state_dim))
    demand = jnp.zeros_like(zero_state).at[0, 1].set(0.01)
    without_demand = step(zero_state, controls[:1], params)
    with_demand = jax.jit(lambda x: step(x, controls[:1], params, demand))(zero_state)
    assert float(with_demand.sum()) > float(without_demand.sum())
