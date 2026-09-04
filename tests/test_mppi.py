from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nmfd_traffic import (
    MPPIConfig,
    NMFDParameters,
    ObjectiveConfig,
    initial_control_plan,
    load_config,
    rollout_mppi,
    rollout_naive_mppi,
    sample_initial_states,
)
from nmfd_traffic.mppi import (
    _normalized_weights,
    _rollout_costs,
    _sample_control_sequences,
    _savgol_coefficients,
    _smooth_control_update,
    _trajectory_scores,
    _update_control_plan_from_perturbations,
    _weighted_perturbation_update,
)

CONFIG = Path(__file__).parents[1] / "configs" / "seven_region.toml"


def _mppi_config(**changes) -> MPPIConfig:
    config = MPPIConfig(
        samples=4,
        iterations=1,
        temperature=2.0,
        noise_std=0.5,
        alpha=0.25,
        smoothing_window=1,
        smoothing_polynomial=0,
    )
    return replace(config, **changes)


def _stationary_environment() -> NMFDParameters:
    """Return a one-region environment whose state does not change."""

    return NMFDParameters(
        num_regions=1,
        dt=1.0,
        n_substeps=1,
        adjacency=jnp.zeros((1, 1)),
        theta=jnp.zeros((1, 1, 1)),
        a=jnp.zeros((1,)),
        b=jnp.zeros((1,)),
        c=jnp.zeros((1,)),
        u_low=0.2,
        u_high=0.8,
    )


def test_algorithm_2_weights_use_fixed_temperature_and_one_score_per_sample():
    config = _mppi_config(temperature=2.0)
    scores = jnp.array([[1.0, 3.0, 5.0]])

    weights = _normalized_weights(scores, config)
    expected = np.exp(np.array([0.0, -1.0, -2.0]))
    expected /= expected.sum()

    assert weights.shape == (1, 3)
    np.testing.assert_allclose(weights[0], expected, rtol=1e-6)
    np.testing.assert_allclose(
        _normalized_weights(scores + 100.0, config),
        weights,
        rtol=1e-6,
    )
    assert not np.allclose(
        _normalized_weights(2.0 * scores, config),
        weights,
    )


def test_one_trajectory_weight_updates_every_horizon_position():
    weights = jnp.array([[0.25, 0.75]])
    perturbations = jnp.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[5.0], [6.0], [7.0]],
            ]
        ]
    )

    update = _weighted_perturbation_update(weights, perturbations)

    np.testing.assert_allclose(update, [[[4.0], [5.0], [6.0]]])


def test_plan_update_uses_raw_perturbations_after_bounded_rollouts():
    environment = _stationary_environment()
    objective = ObjectiveConfig(
        horizon=2,
        state_weight=0.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    config = _mppi_config(
        samples=2,
        alpha=1.0,
        smoothing_window=1,
        smoothing_polynomial=0,
    )
    state = jnp.array([[0.0]])
    plan = jnp.full((1, 2, 1), environment.u_high)
    perturbations = jnp.ones((1, 2, 2, 1))

    updated = _update_control_plan_from_perturbations(
        state,
        plan,
        perturbations,
        environment,
        objective,
        config,
    )

    # Rollouts clamp sampled inputs to 0.8, but Algorithm 1 updates the mean
    # with the original Gaussian perturbations rather than truncated noise.
    np.testing.assert_allclose(updated, plan + 1.0)


def test_algorithm_1_includes_nominal_and_zero_centered_samples():
    plan = jnp.full((1, 2, 1), 10.0)
    perturbations = jnp.arange(8.0).reshape(1, 4, 2, 1)

    sampled = _sample_control_sequences(plan, perturbations, alpha=0.5)

    np.testing.assert_allclose(sampled[:, :2], plan[:, None] + perturbations[:, :2])
    np.testing.assert_allclose(sampled[:, 2:], perturbations[:, 2:])


def test_rollout_cost_has_explicit_terminal_cost_and_nmfd_extensions():
    environment = _stationary_environment()
    objective = ObjectiveConfig(
        horizon=2,
        state_weight=1.0,
        control_weight=2.0,
        control_rate_weight=3.0,
    )
    initial_state = jnp.array([[2.0]])
    sampled_inputs = jnp.array([[[[1.2], [-0.2]], [[0.3], [0.6]]]])

    costs = _rollout_costs(
        initial_state,
        sampled_inputs,
        environment,
        objective,
    )

    # Each trajectory pays c(x0) + c(x1) + phi(x2) = 4 + 4 + 4.
    # Inputs are clamped to [0.2, 0.8] before optional NMFD costs.
    expected = np.array(
        [
            [
                12.0 + 2.0 * (0.8**2 + 0.2**2) + 3.0 * (0.2 - 0.8) ** 2,
                12.0 + 2.0 * (0.3**2 + 0.6**2) + 3.0 * (0.6 - 0.3) ** 2,
            ]
        ]
    )
    np.testing.assert_allclose(costs, expected, rtol=1e-6)


def test_trajectory_score_contains_covariance_aware_importance_correction():
    environment = _stationary_environment()
    objective = ObjectiveConfig(
        horizon=2,
        state_weight=0.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    config = _mppi_config(
        temperature=2.0,
        noise_std=0.5,
        alpha=0.25,
    )
    initial_state = jnp.array([[0.0]])
    plan = jnp.array([[[0.5], [0.25]]])
    sampled_inputs = jnp.array([[[[0.6], [0.4]], [[0.2], [0.8]]]])

    scores = _trajectory_scores(
        initial_state,
        plan,
        sampled_inputs,
        environment,
        objective,
        config,
    )

    # gamma / variance = (2 * (1 - 0.25)) / 0.5**2 = 6.
    expected = 6.0 * np.array([[0.5 * 0.6 + 0.25 * 0.4, 0.5 * 0.2 + 0.25 * 0.8]])
    np.testing.assert_allclose(scores, expected, rtol=1e-6)


def test_savgol_filter_uses_the_quadratic_five_point_coefficients():
    config = _mppi_config(smoothing_window=5, smoothing_polynomial=2)

    coefficients = _savgol_coefficients(config, jnp.float32)
    expected = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0

    np.testing.assert_allclose(coefficients, expected, rtol=1e-5, atol=1e-6)
    constant = jnp.ones((1, 7, 1))
    np.testing.assert_allclose(
        _smooth_control_update(constant, config),
        constant,
        rtol=1e-5,
        atol=1e-6,
    )
    identity_config = _mppi_config(
        smoothing_window=1,
        smoothing_polynomial=0,
    )
    np.testing.assert_array_equal(
        _smooth_control_update(constant, identity_config),
        constant,
    )


def test_mppi_rollout_shape_bounds_reproducibility_and_legacy_alias():
    config = load_config(CONFIG)
    objective = replace(config.objective, horizon=3)
    mppi = replace(
        config.mppi,
        samples=4,
        iterations=1,
        alpha=0.25,
        smoothing_window=3,
        smoothing_polynomial=2,
    )
    initial = sample_initial_states(
        jax.random.PRNGKey(0),
        2,
        config.scenarios["in_distribution"],
        config.environment,
    )
    key = jax.random.PRNGKey(1)
    first_states, first_controls = jax.jit(
        lambda: rollout_mppi(
            key,
            initial,
            2,
            config.environment,
            objective,
            mppi,
        )
    )()
    second_states, second_controls = rollout_mppi(
        key,
        initial,
        2,
        config.environment,
        objective,
        mppi,
    )
    alias_states, alias_controls = rollout_naive_mppi(
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
    np.testing.assert_allclose(alias_states, second_states, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(alias_controls, second_controls, rtol=1e-5, atol=1e-5)


def test_initial_plan_is_fully_open():
    config = load_config(CONFIG)

    plan = initial_control_plan(2, config.environment, config.objective)

    assert plan.shape == (
        2,
        config.objective.horizon,
        config.environment.control_dim,
    )
    np.testing.assert_allclose(plan, config.environment.u_high)


def test_mppi_rollout_can_be_transformed_with_vmap():
    environment = _stationary_environment()
    objective = ObjectiveConfig(
        horizon=2,
        state_weight=1.0,
        control_weight=0.0,
        control_rate_weight=0.0,
    )
    config = _mppi_config(
        samples=2,
        alpha=0.5,
        smoothing_window=1,
        smoothing_polynomial=0,
    )
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    initial_states = jnp.array([[1.0], [2.0]])

    vmapped_rollout = jax.jit(
        jax.vmap(
            lambda key, state: rollout_mppi(
                key,
                state[None, :],
                1,
                environment,
                objective,
                config,
            )
        )
    )
    states, controls = vmapped_rollout(keys, initial_states)

    assert states.shape == (2, 1, 2, 1)
    assert controls.shape == (2, 1, 1, 1)
