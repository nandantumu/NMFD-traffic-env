# NMFD Traffic Environment

A compact JAX environment for regional traffic control with a Networked
Macroscopic Fundamental Diagram (NMFD), with a deterministic Differentiable
Predictive Control (DPC) policy and a horizon-based naive MPPI controller.

This repository was extracted from the urban-traffic example in `L2O_MPPI`.
It intentionally excludes the vehicle and quadruped examples, PyTorch training
paths, learned MPPI variants, experiment tracking, and F1TENTH dependencies.

## Included

- Batched, differentiable NMFD dynamics in JAX.
- Optional exogenous origin-destination demand.
- Fourth-order Runge--Kutta integration with configurable substeps.
- Shortest-path routing generated from an adjacency matrix.
- Reproducible in-distribution and out-of-distribution initial-state scenarios.
- A bounded deterministic Flax MLP trained end-to-end through the dynamics.
- A receding-horizon naive MPPI controller with configurable samples,
  iterations, temperature, and exploration scale.
- An inference-only checkpoint for the original seven-region DPC policy.
- Reusable total-accumulation, per-region, control-heatmap, and topology plots.
- Focused tests and small command-line entry points.

The direct runtime dependencies are JAX, Flax, NumPy, Optax, and Matplotlib.

## Setup with uv

```bash
git clone <repository-url> NMFD-traffic-env
cd NMFD-traffic-env
uv sync
uv run pytest
```

The locked environment uses CPU-compatible JAX by default. Install the JAX
accelerator build appropriate for your platform separately if GPU execution is
needed.

## Compare DPC and naive MPPI

Evaluate 100 matched rollouts from the training distribution:

```bash
uv run nmfd-evaluate --scenario in_distribution
```

Evaluate the shifted congestion distribution:

```bash
uv run nmfd-evaluate --scenario out_of_distribution
```

Each command evaluates DPC, naive MPPI, and a constant open-gates reference on
the same initial states. It writes a metrics JSON file and four plots under
`results/`:

- total network accumulation with mean and one-standard-deviation bands;
- per-region accumulation;
- mean DPC and MPPI control heatmaps;
- the region-network topology.

For a faster exploratory run, reduce the number of rollouts and MPPI samples:

```bash
uv run nmfd-evaluate \
  --scenario in_distribution \
  --rollouts 10 \
  --mppi-samples 32 \
  --output-dir results/quick
```

Use `--no-plots` when only the JSON metrics are needed. Example plots generated
with the checked-in configuration are available under `figures/`.

Or run the minimal Python example:

```bash
uv run python examples/quickstart.py
```

## Train a policy

The checked-in configuration retains the original training settings. A short
smoke run can be launched with:

```bash
uv run nmfd-train \
  --epochs 1 \
  --steps-per-epoch 5 \
  --sample-pool-size 2048 \
  --batch-size 32
```

This writes `checkpoints/dpc_policy_trained.msgpack`. A full reproduction uses:

```bash
uv run nmfd-train
```

## Python API

```python
from pathlib import Path

import jax
import jax.numpy as jnp

from nmfd_traffic import load_config, step

config = load_config(Path("configs/seven_region.toml"))
params = config.environment

state = jnp.full((1, params.state_dim), 100.0)
control = jnp.full((1, params.control_dim), params.u_high)
next_state = jax.jit(lambda x, u: step(x, u, params))(state, control)
```

State and control tensors use flattened row-major `(origin, destination)`
matrices. For `R` regions, both have final dimension `R**2`. Only off-diagonal
controls on adjacency edges affect the dynamics; the complete square control
shape is retained for compatibility and straightforward reshaping.

## Model

For accumulation `x_ij` in origin region `i` with destination `j`, total
regional accumulation and production are

```text
x_i      = sum_j x_ij
g_i(x_i) = a_i x_i^3 + b_i x_i^2 + c_i x_i.
```

Shortest-path routing fractions `theta_ihj` define transfer flow

```text
m_ihj = theta_ihj (x_ij / x_i) g_i(x_i),
```

while `m_ii = (x_ii / x_i) g_i(x_i)` completes trips already in their
destination region. Perimeter controls multiply transfer flows between adjacent
regions. See `src/nmfd_traffic/dynamics.py` for the vectorized equations.

The default configuration has seven regions, 49 state components, a 120-second
control interval, four RK4 substeps, and a 40-step horizon. The DPC policy is a
five-layer, 512-unit GELU MLP with bounded outputs in `[0.2, 0.8]`.

At every control step, naive MPPI samples full 40-step control sequences around
a warm-start plan, rolls each sequence through the same NMFD dynamics, and uses
normalized reward-to-go weights to update every point in the plan. It executes
the first action, shifts the optimized plan, and repeats. The default uses 128
samples and one update iteration.

The controller implementation is in `src/nmfd_traffic/mppi.py`; plotting is in
`src/nmfd_traffic/visualization.py`. Both are functional and compatible with
`jax.jit` where applicable.

## Scenarios

`configs/seven_region.toml` defines:

- `in_distribution`: ordinary OD cells use `Normal(150, 20)`; cells `(1,5)`,
  `(0,6)`, `(5,0)`, and `(6,1)` use `Normal(4000, 100)`.
- `out_of_distribution`: ordinary cells shift to `Normal(200, 40)` and the
  same four hotspots shift to `Normal(5000, 200)`.

Training and evaluation are zero-demand stabilization tasks. The public
`dynamics` and `step` functions nevertheless accept an optional demand tensor
so the environment can also represent continuing arrivals.

## Checkpoint format

`checkpoints/dpc_policy.msgpack` contains only Flax inference parameters—no
optimizer state and no executable pickle payload. Initialize the configured
policy to obtain a matching parameter template, then call `load_parameters`.

## Licensing

The source repository did not contain a project-level license. Add an explicit
license before distributing this repository outside your organization.
