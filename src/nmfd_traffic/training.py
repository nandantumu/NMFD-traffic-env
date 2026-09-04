from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import NMFDParameters, ObjectiveConfig, TrainingConfig
from .dynamics import step_with_noise
from .objective import mean_total_vehicle_time


class DPCTrainState(train_state.TrainState):
    """Flax training state for the deterministic DPC policy."""


def create_train_state(
    key: jax.Array, policy, training: TrainingConfig
) -> DPCTrainState:
    sample = jnp.zeros((1, policy.input_dim), dtype=jnp.float32)
    variables = policy.init(key, sample, train=False)
    optimizer = optax.adamw(
        training.learning_rate,
        weight_decay=training.weight_decay,
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
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run the PC policy under demand/noise and include the initial state."""

    batch_size = initial_state.shape[0]
    expected_shape = (batch_size, horizon, environment.state_dim)
    if demand is None:
        demand = jnp.zeros(expected_shape, dtype=initial_state.dtype)
    if state_noise is None:
        state_noise = jnp.zeros(expected_shape, dtype=initial_state.dtype)
    if demand.shape != expected_shape:
        raise ValueError(f"demand must have shape {expected_shape}")
    if state_noise.shape != expected_shape:
        raise ValueError(f"state_noise must have shape {expected_shape}")

    def body(current: jax.Array, inputs: tuple[jax.Array, jax.Array]):
        current_demand, current_noise = inputs
        control = apply_fn({"params": policy_params}, current, train=False)
        next_state = step_with_noise(
            current,
            control,
            environment,
            current_demand,
            current_noise,
        )
        return next_state, (next_state, control)

    scan_inputs = (jnp.swapaxes(demand, 0, 1), jnp.swapaxes(state_noise, 0, 1))
    _, (states, controls) = jax.lax.scan(body, initial_state, scan_inputs)
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
    demand: jax.Array | None = None,
    state_noise: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate the paper's L1 total-vehicle-time loss for a policy rollout."""

    states, _ = rollout_policy(
        policy_params,
        apply_fn,
        initial_state,
        environment,
        objective.horizon,
        demand,
        state_noise,
    )
    loss = mean_total_vehicle_time(states, environment.dt)
    return loss, {"total_vehicle_time": loss}


def make_train_step(environment: NMFDParameters, objective: ObjectiveConfig):
    """Create a JIT-compiled DPC update specialized to an environment."""

    @jax.jit
    def train_step(
        state: DPCTrainState,
        initial_state: jax.Array,
        demand: jax.Array,
        state_noise: jax.Array,
    ):
        def loss_function(policy_params):
            return dpc_loss(
                policy_params,
                state.apply_fn,
                initial_state,
                environment,
                objective,
                demand,
                state_noise,
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=gradients)
        return state, {"loss": loss, **metrics}

    return train_step
