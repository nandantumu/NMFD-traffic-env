from pathlib import Path

import jax
import jax.numpy as jnp

from nmfd_traffic import (
    create_policy,
    load_config,
    load_parameters,
    rollout_policy,
    sample_initial_states,
)


config = load_config(Path("configs/seven_region.toml"))
policy = create_policy(config.policy, config.environment)
template = policy.init(
    jax.random.PRNGKey(config.seed),
    jnp.zeros((1, config.environment.state_dim)),
    train=False,
)["params"]
parameters = load_parameters("checkpoints/dpc_policy.msgpack", template)
initial_state = sample_initial_states(
    jax.random.PRNGKey(1),
    count=1,
    scenario=config.scenarios["in_distribution"],
    params=config.environment,
)
states, controls = jax.jit(
    lambda x: rollout_policy(
        parameters,
        policy.apply,
        x,
        config.environment,
        config.objective.horizon,
    )
)(initial_state)

print("state trajectory:", states.shape)
print("control trajectory:", controls.shape)
print("initial vehicles:", float(states[0, 0].sum()))
print("terminal vehicles:", float(states[0, -1].sum()))
