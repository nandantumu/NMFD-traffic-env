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
    sample_initial_states,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_policy_shape_and_bounds():
    config = load_config(CONFIG)
    policy = create_policy(config.policy, config.environment)
    state = jnp.ones((3, config.environment.state_dim))
    variables = policy.init(jax.random.PRNGKey(0), state, train=False)
    control = policy.apply(variables, state, train=False)
    assert control.shape == (3, config.environment.control_dim)
    assert np.all(np.asarray(control) >= config.environment.u_low)
    assert np.all(np.asarray(control) <= config.environment.u_high)


def test_one_training_update_is_finite():
    config = load_config(CONFIG)
    small_policy_config = replace(config.policy, hidden_dim=16, num_hidden_layers=2)
    small_objective = replace(config.objective, horizon=2)
    training = replace(config.training, batch_size=4)
    policy = create_policy(small_policy_config, config.environment)
    state = create_train_state(jax.random.PRNGKey(0), policy, training)
    initial = sample_initial_states(
        jax.random.PRNGKey(1),
        4,
        config.scenarios["in_distribution"],
        config.environment,
    )
    new_state, metrics = make_train_step(config.environment, small_objective)(
        state, initial
    )
    assert int(new_state.step) == 1
    assert np.isfinite(float(metrics["loss"]))
