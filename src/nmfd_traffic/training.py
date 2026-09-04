from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import NMFDParameters, ObjectiveConfig, TrainingConfig
from .dynamics import step


class DPCTrainState(train_state.TrainState):
    """Flax training state for the deterministic DPC policy."""


def create_train_state(
    key: jax.Array, policy, training: TrainingConfig
) -> DPCTrainState:
    sample = jnp.zeros((1, policy.input_dim), dtype=jnp.float32)
    variables = policy.init(key, sample, train=False)
    optimizer = optax.chain(
        optax.clip_by_global_norm(training.gradient_clip),
        optax.adamw(training.learning_rate, weight_decay=training.weight_decay),
    )
    return DPCTrainState.create(
        apply_fn=policy.apply,
        params=variables["params"],
        tx=optimizer,
    )


def rollout_policy(
    policy_params: Any,
    apply_fn,
    initial_state: jax.Array,
    environment: NMFDParameters,
    horizon: int,
) -> tuple[jax.Array, jax.Array]:
    """Run a deterministic closed-loop rollout and include the initial state."""

    def body(current: jax.Array, _: None):
        control = apply_fn({"params": policy_params}, current, train=False)
        next_state = step(current, control, environment)
        return next_state, (next_state, control)

    _, (states, controls) = jax.lax.scan(body, initial_state, None, length=horizon)
    states = jnp.swapaxes(states, 0, 1)
    controls = jnp.swapaxes(controls, 0, 1)
    states = jnp.concatenate([initial_state[:, None, :], states], axis=1)
    return states, controls


def dpc_loss(
    policy_params: Any,
    apply_fn,
    initial_state: jax.Array,
    environment: NMFDParameters,
    objective: ObjectiveConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    states, controls = rollout_policy(
        policy_params,
        apply_fn,
        initial_state,
        environment,
        objective.horizon,
    )
    state_cost = objective.state_weight * jnp.mean(jnp.sum(states**2, axis=-1))
    control_cost = objective.control_weight * jnp.mean(jnp.sum(controls**2, axis=-1))
    if objective.horizon > 1:
        differences = controls[:, 1:, :] - controls[:, :-1, :]
        rate_cost = objective.control_rate_weight * jnp.mean(
            jnp.sum(differences**2, axis=-1)
        )
    else:
        rate_cost = jnp.zeros((), dtype=states.dtype)
    loss = state_cost + control_cost + rate_cost
    return loss, {
        "state_cost": state_cost,
        "control_cost": control_cost,
        "rate_cost": rate_cost,
    }


def make_train_step(environment: NMFDParameters, objective: ObjectiveConfig):
    """Create a JIT-compiled DPC update specialized to an environment."""

    @jax.jit
    def train_step(state: DPCTrainState, initial_state: jax.Array):
        def loss_function(policy_params):
            return dpc_loss(
                policy_params,
                state.apply_fn,
                initial_state,
                environment,
                objective,
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=gradients)
        return state, {"loss": loss, **metrics}

    return train_step
