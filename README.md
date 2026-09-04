# NMFD traffic control environment

A compact, educational JAX implementation of a seven-region Networked
Macroscopic Fundamental Diagram (NMFD), a perimeter-control-only
Differentiable Predictive Control (DPC) policy, and information-theoretic MPPI.

The DPC experiment targets [Tumu et al. (2024)][dpc-paper]. The MPPI controller
targets Algorithms 1 and 2 of [Williams et al. (2017)][mppi-paper]. Both
controllers optimize one shared L1 total-vehicle-time objective.

## What is included

- Batched NMFD dynamics with optional time-varying OD demand.
- Fourth-order Runge--Kutta integration in a functional JAX API.
- Fixed routing generated from equal-weight shortest-path next hops.
- A perimeter-control policy with three 128-neuron `tanh` hidden layers.
- The paper's sigmoid-affine output map, which enforces controls in
  `[0.1, 0.9]` by construction.
- Offline DPC training through the closed-loop NMFD rollout.
- Receding-horizon 2017 information-theoretic MPPI.
- Reproducible nominal and noisy-demand scenarios.
- Hash-bound checkpoint provenance and detailed evaluation reports.
- Ruff, branch-coverage tests, wheel checks, and GitHub Actions CI.

The runtime remains JAX-only. PyTorch and Gym are not dependencies.

## Setup

```bash
git clone <repository-url> NMFD-traffic-env
cd NMFD-traffic-env
uv sync
uv run pytest
```

The lock file selects CPU-compatible JAX. Install the JAX accelerator build
appropriate for your platform separately when GPU execution is needed.

## Train the PC-only DPC policy

No pretrained checkpoint is included: the previous checkpoint used a different
policy, objective, scenario, and numerical configuration and was intentionally
removed. A smoke run is:

```bash
uv run nmfd-train \
  --epochs 1 \
  --steps-per-epoch 1 \
  --batch-size 4
```

The default command runs the configuration's 1,000 optimizer updates:

```bash
uv run nmfd-train
```

Both commands write `checkpoints/dpc_policy.msgpack` and a
`checkpoints/dpc_policy.json` provenance sidecar. The sidecar records the
artifact hash and size, seed, policy architecture, objective, scenario, demand
profiles, optimizer settings, software versions, source configuration hash,
and repository state. Generated checkpoints are ignored by Git.

The paper reports Adam with learning rate `1e-4`, weight decay `1e-6`, and
batch size 256, but does not report a training-update count. The checked-in
1,000-update default is therefore a repository choice, not a claimed paper
hyperparameter.

## Compare DPC and MPPI

Pass the checkpoint explicitly so an incompatible or undocumented policy cannot
be selected silently:

```bash
uv run nmfd-evaluate \
  --checkpoint checkpoints/dpc_policy.msgpack \
  --scenario nominal
```

For the representative demand-shift case:

```bash
uv run nmfd-evaluate \
  --checkpoint checkpoints/dpc_policy.msgpack \
  --scenario noisy_demand
```

Evaluation matches the initial states, demand trajectories, and state-noise
realizations across DPC, MPPI, and the open-gates baseline. It writes a metrics
JSON file and, unless `--no-plots` is set, plots under `results/`. Reports
contain the random seed, software/Git provenance, asset hashes, policy and MPPI
settings, demand/noise configuration, shared objective, performance metrics,
and runtime.

For a quick comparison:

```bash
uv run nmfd-evaluate \
  --checkpoint checkpoints/dpc_policy.msgpack \
  --rollouts 2 \
  --mppi-samples 8 \
  --no-plots
```

## Paper-target experiment

The default configuration uses the paper's stated seven-region settings:

| Setting | Value |
| --- | --- |
| NMFD coefficients | `a=4.133e-11`, `b=-8.282e-7`, `c=0.0042` |
| Control interval | 30 s |
| Experiment/training horizon | 240 steps |
| MPPI receding planning horizon | 8 steps |
| Control bounds | `[0.1, 0.9]` |
| State/measurement noise standard deviation | 0.25 |
| Initial state | zero vehicles |
| Routing | fixed shortest paths |
| DPC hidden width | 128 |
| DPC architecture selected here | three hidden layers, `tanh` |
| Batch size | 256 |
| Learning rate / weight decay | `1e-4` / `1e-6` |

The demand scenario contains the five paper OD flows: both directions between
regions 0 and 6, both directions between regions 5 and 1, and region 4 to
region 2. Figure 4 of the paper plots these values but does not publish a
numeric table. The checked-in piecewise-linear profiles are explicitly
documented approximations of that figure; they should not be described as the
authors' exact samples.

As requested, training and evaluation both use this NMFD model. The paper's
separate acyclic evaluation plant is not implemented. There is also no routing
guidance decoder: the implementation is perimeter control only.

See [the DPC implementation note](docs/dpc.md) for the full mapping and
intentional deviations, and [the MPPI implementation note](docs/mppi.md) for
the sampling-controller equations.

## Shared controller objective

For a complete trajectory `[x[0], ..., x[N]]`, both controllers use

```text
J = dt * sum(k=1..N-1, ||x[k]||_1).
```

This is Equation (16a) of the traffic DPC paper, expressed in
vehicle-seconds. The shared implementation is
`src/nmfd_traffic/objective.py`. MPPI still adds the 2017
importance-sampling correction to its sampled-trajectory score; that correction
is part of the MPPI estimator, not a different traffic objective.

## Python API

```python
from pathlib import Path

import jax
import jax.numpy as jnp

from nmfd_traffic import load_config, step

config = load_config(Path("configs/seven_region.toml"))
params = config.environment
state = jnp.zeros((1, params.state_dim))
control = jnp.full((1, params.control_dim), params.u_high)
demand = jnp.zeros_like(state).at[0, 6].set(5.0)

next_state = jax.jit(lambda x, u, d: step(x, u, params, d))(
    state,
    control,
    demand,
)
```

State and demand tensors flatten a row-major
`(current region, final destination)` matrix. Control tensors flatten a
`(sending region, receiving region)` matrix. All have final dimension
`num_regions**2`; leading batch dimensions are supported.

For accumulation `x_ij`, regional production is

```text
x_i      = sum_j x_ij
g_i(x_i) = a_i x_i^3 + b_i x_i^2 + c_i x_i.
```

Shortest-path routing fractions `theta_ihj` determine transfers, while
`u_ih` gates flow across adjacent region boundaries. Production is evaluated
at the actual regional accumulation and is not clipped or saturated. Integrated
states are clipped only at zero to prevent negative vehicle counts.

## Development

Run the same checks as CI:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=nmfd_traffic --cov-report=term-missing
uv build --wheel
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for readability, testing, documentation,
and JAX transformation expectations.

## License

MIT. See [LICENSE](LICENSE).

[dpc-paper]: https://arxiv.org/abs/2406.10433
[mppi-paper]: https://arxiv.org/abs/1707.02342
