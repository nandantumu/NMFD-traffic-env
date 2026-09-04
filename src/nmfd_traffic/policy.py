from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import NMFDParameters, PolicyConfig


def _activation(name: str):
    functions = {
        "relu": nn.relu,
        "tanh": jnp.tanh,
        "gelu": nn.gelu,
        "silu": nn.silu,
    }
    try:
        return functions[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation {name!r}") from error


def _linear_uniform():
    def initialize(key, shape, dtype=jnp.float32):
        bound = 1.0 / jnp.sqrt(jnp.asarray(shape[0], dtype=dtype))
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return initialize


class DPCPolicy(nn.Module):
    """Deterministic bounded MLP policy used for differentiable predictive control."""

    input_dim: int
    control_dim: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    activation: str = "gelu"
    bounds_low: Sequence[float] | None = None
    bounds_high: Sequence[float] | None = None

    @nn.compact
    def __call__(self, state: jax.Array, train: bool = False) -> jax.Array:
        activation = _activation(self.activation)
        initializer = _linear_uniform()
        hidden = state
        for _ in range(self.num_hidden_layers):
            hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=initializer,
                bias_init=initializer,
            )(hidden)
            hidden = activation(hidden)

        raw_control = nn.Dense(
            self.control_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        if self.bounds_low is None or self.bounds_high is None:
            return raw_control
        lower = jnp.asarray(self.bounds_low, dtype=raw_control.dtype)
        upper = jnp.asarray(self.bounds_high, dtype=raw_control.dtype)
        return 0.5 * (upper + lower) + 0.5 * (upper - lower) * jnp.tanh(raw_control)


def create_policy(config: PolicyConfig, params: NMFDParameters) -> DPCPolicy:
    """Build the configured DPC policy with bounds matching the environment."""

    return DPCPolicy(
        input_dim=params.state_dim,
        control_dim=params.control_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        bounds_low=(params.u_low,) * params.control_dim,
        bounds_high=(params.u_high,) * params.control_dim,
    )
