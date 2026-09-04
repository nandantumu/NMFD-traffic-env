from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from nmfd_traffic import load_config, rollout_naive_mppi, sample_initial_states


CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_naive_mppi_rollout_shape_bounds_and_reproducibility():
    config = load_config(CONFIG)
    objective = replace(config.objective, horizon=3)
    mppi = replace(config.naive_mppi, samples=4, iterations=1)
    initial = sample_initial_states(
        jax.random.PRNGKey(0),
        2,
        config.scenarios["in_distribution"],
        config.environment,
    )
    key = jax.random.PRNGKey(1)
    first_states, first_controls = jax.jit(
        lambda: rollout_naive_mppi(
            key,
            initial,
            2,
            config.environment,
            objective,
            mppi,
        )
    )()
    second_states, second_controls = rollout_naive_mppi(
        key,
        initial,
        2,
        config.environment,
        objective,
        mppi,
    )

    assert first_states.shape == (2, 3, config.environment.state_dim)
    assert first_controls.shape == (2, 2, config.environment.control_dim)
    assert np.all(np.asarray(first_controls) >= config.environment.u_low)
    assert np.all(np.asarray(first_controls) <= config.environment.u_high)
    np.testing.assert_allclose(first_states, second_states, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(first_controls, second_controls, rtol=1e-5, atol=1e-5)
