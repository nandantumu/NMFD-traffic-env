from pathlib import Path

import jax
import numpy as np
import pytest

from nmfd_traffic import (
    build_shortest_path_routing,
    load_config,
    sample_initial_states,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def test_seven_region_config_and_routing():
    config = load_config(CONFIG)
    params = config.environment
    assert params.num_regions == 7
    assert params.state_dim == params.control_dim == 49
    assert config.mppi.samples == 128
    assert config.mppi.iterations == 1
    assert config.mppi.alpha == pytest.approx(0.01)
    assert config.mppi.smoothing_window == 5
    assert config.mppi.smoothing_polynomial == 2
    assert config.naive_mppi is config.mppi
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


def test_legacy_naive_mppi_configuration_table_remains_loadable(tmp_path):
    legacy_config = tmp_path / "legacy.toml"
    legacy_config.write_text(
        CONFIG.read_text(encoding="utf-8").replace("[mppi]", "[naive_mppi]"),
        encoding="utf-8",
    )

    config = load_config(legacy_config)

    assert config.mppi.samples == 128
    assert config.naive_mppi is config.mppi


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
        ("dt = 120.0", "dt = 0.0", "dt must be positive"),
        ("state_weight = 1.0", "state_weight = -1.0", "objective weights"),
        ("alpha = 0.01", "alpha = 1.1", "alpha must be between"),
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
        ("batch_size = 512", "batch_size = 0", "training sizes"),
        ("std = 20.0", "std = -1.0", "std must be nonnegative"),
        ("min_value = 0.0", "min_value = 2000.0", "min_value must not exceed"),
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
