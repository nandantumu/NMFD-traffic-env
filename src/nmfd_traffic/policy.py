from __future__ import annotations

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


class DPCPolicy(nn.Module):
    """Deterministic perimeter-control policy used by DPC.

    Hidden activations are configured independently from the final sigmoid. The
    sigmoid-affine output map is Equation (17) of Tumu et al. (2024) and makes
    every control feasible by construction.
    """

    input_dim: int
    control_dim: int
    hidden_dim: int = 128
    num_hidden_layers: int = 3
    activation: str = "tanh"
    bounds_low: float = 0.1
    bounds_high: float = 0.9

    @nn.compact
    def __call__(self, state: jax.Array, train: bool = False) -> jax.Array:
        activation = _activation(self.activation)
        hidden = state
        for _ in range(self.num_hidden_layers):
            hidden = nn.Dense(self.hidden_dim)(hidden)
            hidden = activation(hidden)

        raw_control = nn.Dense(self.control_dim)(hidden)
        lower = jnp.asarray(self.bounds_low, dtype=raw_control.dtype)
        upper = jnp.asarray(self.bounds_high, dtype=raw_control.dtype)
        return lower + (upper - lower) * nn.sigmoid(raw_control)


def create_policy(config: PolicyConfig, params: NMFDParameters) -> DPCPolicy:
    """Build the configured DPC policy with bounds matching the environment."""

    return DPCPolicy(
        input_dim=params.state_dim,
        control_dim=params.control_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        activation=config.activation,
        bounds_low=params.u_low,
        bounds_high=params.u_high,
    )
