from pathlib import Path

import jax
import numpy as np
import pytest

from nmfd_traffic import (
    build_shortest_path_routing,
    demand_trajectory,
    load_config,
    sample_scenario,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_seven_region_config_matches_the_paper_target():
    config = load_config(CONFIG)
    params = config.environment

    assert params.num_regions == 7
    assert params.state_dim == params.control_dim == 49
    assert params.dt == pytest.approx(30.0)
    assert params.n_substeps == 1
    assert params.u_low == pytest.approx(0.1)
    assert params.u_high == pytest.approx(0.9)
    assert config.objective.horizon == 240
    assert config.mppi.planning_horizon == 8
    assert config.policy.hidden_dim == 128
    assert config.policy.num_hidden_layers == 3
    assert config.policy.activation == "tanh"
    assert config.training.batch_size == 256
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.training.weight_decay == pytest.approx(1e-6)
    assert config.simulation.state_noise_std == pytest.approx(0.25)
    assert len(config.demand_profiles) == 5
    assert sorted(config.scenarios) == ["noisy_demand", "nominal"]


def test_fixed_shortest_path_routing_is_valid():
    config = load_config(CONFIG)
    theta = np.asarray(config.environment.theta)

    assert theta.shape == (7, 7, 7)
    for source in range(7):
        for destination in range(7):
            expected = 0.0 if source == destination else 1.0
            assert theta[source, :, destination].sum() == pytest.approx(expected)


def test_demand_trajectory_populates_only_the_five_paper_od_pairs():
    config = load_config(CONFIG)
    demand = np.asarray(
        demand_trajectory(
            config.demand_profiles,
            config.environment,
            config.objective.horizon,
        )
    )
    active = set(np.flatnonzero(np.any(demand > 0.0, axis=0)))

    assert demand.shape == (240, 49)
    assert active == {0 * 7 + 6, 6 * 7 + 0, 4 * 7 + 2, 5 * 7 + 1, 1 * 7 + 5}
    assert demand.max() == pytest.approx(15.5)
    assert np.all(demand >= 0.0)


def test_sampled_scenario_is_reproducible_and_keeps_gaussian_state_noise():
    config = load_config(CONFIG)
    arguments = (
        3,
        8,
        config.scenarios["nominal"],
        config.demand_profiles,
        config.environment,
        config.simulation.state_noise_std,
    )

    first = sample_scenario(jax.random.PRNGKey(2), *arguments)
    second = sample_scenario(jax.random.PRNGKey(2), *arguments)

    for first_value, second_value in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_value, second_value)
    initial, demand, state_noise = map(np.asarray, first)
    assert initial.shape == (3, 49)
    assert demand.shape == state_noise.shape == (3, 8, 49)
    np.testing.assert_array_equal(initial, 0.0)
    np.testing.assert_array_equal(demand[0], demand[1])
    assert np.any(state_noise < 0.0)
    assert np.any(state_noise > 0.0)


def test_demand_shift_noise_is_nonnegative_and_only_changes_active_od_pairs():
    config = load_config(CONFIG)
    _, nominal, _ = sample_scenario(
        jax.random.PRNGKey(0),
        2,
        20,
        config.scenarios["nominal"],
        config.demand_profiles,
        config.environment,
        0.0,
    )
    _, shifted, _ = sample_scenario(
        jax.random.PRNGKey(0),
        2,
        20,
        config.scenarios["noisy_demand"],
        config.demand_profiles,
        config.environment,
        0.0,
    )
    active = {
        profile.origin * 7 + profile.destination for profile in config.demand_profiles
    }
    inactive = sorted(set(range(49)) - active)

    assert np.all(np.asarray(shifted) >= 0.0)
    np.testing.assert_array_equal(shifted[..., inactive], nominal[..., inactive])
    assert not np.array_equal(
        shifted[..., sorted(active)],
        nominal[..., sorted(active)],
    )


def test_shortest_path_routing_splits_equally_across_known_next_hops():
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )

    theta = build_shortest_path_routing(adjacency)

    assert theta[0, 1, 3] == pytest.approx(0.5)
    assert theta[0, 2, 3] == pytest.approx(0.5)
    assert theta[0, 3, 3] == pytest.approx(0.0)
    assert theta[1, 0, 2] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("dt = 30.0", "dt = 0.0", "dt must be positive"),
        ("horizon = 240", "horizon = 1", "horizon must be at least two"),
        ("alpha = 0.01", "alpha = 1.1", "alpha must be between"),
        (
            "planning_horizon = 8",
            "planning_horizon = 1",
            "planning_horizon must be at least two",
        ),
        (
            "smoothing_window = 5",
            "smoothing_window = 4",
            "smoothing_window must be a positive odd",
        ),
        (
            "smoothing_polynomial = 2",
            "smoothing_polynomial = 5",
            "smoothing_polynomial must be nonnegative",
        ),
        ("batch_size = 256", "batch_size = 0", "training sizes"),
        ("state_noise_std = 0.25", "state_noise_std = -1", "must be nonnegative"),
        ("times = [0, 210,", "times = [210, 0,", "strictly increasing"),
        ("rates = [0, 0, 8.0,", "rates = [0, 0, -8.0,", "rates must be nonnegative"),
    ],
)
def test_invalid_configuration_values_are_rejected(
    tmp_path,
    original,
    replacement,
    message,
):
    config_text = CONFIG.read_text(encoding="utf-8").replace(
        original,
        replacement,
        1,
    )
    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(invalid_config)
