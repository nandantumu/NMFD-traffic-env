from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nmfd_traffic import (
    create_policy,
    create_train_state,
    load_config,
    make_train_step,
    mean_total_vehicle_time,
    sample_scenario,
    total_vehicle_time,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_policy_has_three_128_unit_tanh_layers_and_bounded_output():
    config = load_config(CONFIG)
    policy = create_policy(config.policy, config.environment)
    state = jnp.ones((3, config.environment.state_dim))
    variables = policy.init(jax.random.PRNGKey(0), state, train=False)
    control = policy.apply(variables, state, train=False)
    parameters = variables["params"]

    assert [parameters[f"Dense_{index}"]["kernel"].shape for index in range(4)] == [
        (49, 128),
        (128, 128),
        (128, 128),
        (128, 49),
    ]
    assert control.shape == (3, config.environment.control_dim)
    assert np.all(np.asarray(control) >= config.environment.u_low)
    assert np.all(np.asarray(control) <= config.environment.u_high)


def test_zero_parameters_map_to_the_midpoint_through_the_sigmoid_output():
    config = load_config(CONFIG)
    policy = create_policy(config.policy, config.environment)
    variables = policy.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, config.environment.state_dim)),
        train=False,
    )
    zero_variables = jax.tree_util.tree_map(jnp.zeros_like, variables)

    control = policy.apply(
        zero_variables,
        jnp.ones((1, config.environment.state_dim)),
        train=False,
    )

    np.testing.assert_allclose(control, 0.5)


def test_total_vehicle_time_uses_l1_interior_states_and_seconds():
    trajectory = jnp.array(
        [
            [[100.0, 100.0], [1.0, -2.0], [3.0, 4.0], [200.0, 200.0]],
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
        ]
    )

    costs = total_vehicle_time(trajectory, dt=30.0)

    np.testing.assert_allclose(costs, [300.0, 240.0])
    np.testing.assert_allclose(mean_total_vehicle_time(trajectory, 30.0), 270.0)


def test_one_training_update_with_demand_and_noise_is_finite():
    config = load_config(CONFIG)
    small_policy_config = replace(config.policy, hidden_dim=16)
    small_objective = replace(config.objective, horizon=3)
    training = replace(config.training, batch_size=4)
    policy = create_policy(small_policy_config, config.environment)
    state = create_train_state(jax.random.PRNGKey(0), policy, training)
    initial, demand, state_noise = sample_scenario(
        jax.random.PRNGKey(1),
        4,
        small_objective.horizon,
        config.scenarios["nominal"],
        config.demand_profiles,
        config.environment,
        config.simulation.state_noise_std,
    )

    new_state, metrics = make_train_step(config.environment, small_objective)(
        state,
        initial,
        demand,
        state_noise,
    )

    assert int(new_state.step) == 1
    assert np.isfinite(float(metrics["loss"]))
    assert metrics["loss"] == metrics["total_vehicle_time"]
