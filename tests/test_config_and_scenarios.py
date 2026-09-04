from pathlib import Path

import jax
import numpy as np
import pytest

from nmfd_traffic import load_config, sample_initial_states


CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_seven_region_config_and_routing():
    config = load_config(CONFIG)
    params = config.environment
    assert params.num_regions == 7
    assert params.state_dim == params.control_dim == 49
    theta = np.asarray(params.theta)
    assert theta.shape == (7, 7, 7)
    for source in range(7):
        for destination in range(7):
            expected = 0.0 if source == destination else 1.0
            assert theta[source, :, destination].sum() == pytest.approx(expected)


def test_scenario_hotspots_override_global_prior():
    config = load_config(CONFIG)
    states = np.asarray(
        sample_initial_states(
            jax.random.PRNGKey(0),
            256,
            config.scenarios["in_distribution"],
            config.environment,
        )
    )
    hotspot = states[:, 1 * 7 + 5]
    ordinary = states[:, 0]
    assert hotspot.mean() > 3500.0
    assert ordinary.mean() < 250.0
