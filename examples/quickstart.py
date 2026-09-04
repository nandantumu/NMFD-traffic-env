"""Roll out the paper-target demand scenario with fully open gates."""

from pathlib import Path

import jax
import jax.numpy as jnp

from nmfd_traffic import load_config, rollout_controls, sample_scenario

config = load_config(Path("configs/seven_region.toml"))
horizon = config.objective.horizon
initial, demand, state_noise = sample_scenario(
    jax.random.PRNGKey(config.seed),
    count=1,
    steps=horizon,
    scenario=config.scenarios["nominal"],
    profiles=config.demand_profiles,
    params=config.environment,
    state_noise_std=config.simulation.state_noise_std,
)
controls = jnp.full(
    (1, horizon, config.environment.control_dim),
    config.environment.u_high,
)
states = jax.jit(
    lambda x, u, d, noise: rollout_controls(
        x,
        u,
        config.environment,
        d,
        noise,
    )
)(initial, controls, demand, state_noise)

print("state trajectory:", states.shape)
print("control trajectory:", controls.shape)
print("initial vehicles:", float(states[0, 0].sum()))
print("terminal vehicles:", float(states[0, -1].sum()))
