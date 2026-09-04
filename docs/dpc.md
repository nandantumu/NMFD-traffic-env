# DPC implementation

## Target

The implementation targets the perimeter-control (PC) case in:

> R. Tumu, W. Shaw Cortez, J. Drgoňa, D. L. Vrabie, and S. Glavaski,
> “Differentiable Predictive Control for Large-Scale Urban Road Networks,”
> 2024. [arXiv:2406.10433][traffic-paper].

The broader DPC method is described in:

> J. Drgoňa, A. Tuor, and D. Vrabie, “Learning Constrained Parametric
> Differentiable Predictive Control Policies With Guarantees,” *IEEE
> Transactions on Systems, Man, and Cybernetics: Systems*, 2024.
> [DOI:10.1109/TSMC.2024.3368026][dpc-paper].

DPC trains a policy offline by differentiating a finite-horizon control loss
through a closed-loop dynamics rollout. Online evaluation is then one neural
policy call rather than an optimization solve.

## Implemented PC policy

This repository implements perimeter control only. Routing is fixed before
training by computing graph shortest paths. When several adjacent next hops
have the same shortest distance, their routing fractions are split equally.
There is no learned routing-guidance decoder.

The policy is

```text
b[0] = x
b[l+1] = tanh(W[l] b[l] + c[l]),  l = 0, 1, 2
raw_u = W[out] b[3] + c[out]
u = u_low + (u_high - u_low) * sigmoid(raw_u).
```

Thus there are three hidden layers, each with 128 neurons and a `tanh`
activation. The last sigmoid-affine map is Equation (17) of the traffic paper;
it guarantees `u_low <= u <= u_high` by construction. The three-layer depth
and `tanh` activation are project selections requested for this implementation
pass. The paper specifies hidden/feature width 128 but does not tabulate the
network depth or hidden activation.

## Closed-loop training

For every optimizer update:

1. Build a batch of zero initial states.
2. Interpolate the five configured OD demand profiles for all 240 steps.
3. Sample any configured demand shift and the additive state noise.
4. Evaluate the same policy and NMFD model sequentially for the full horizon.
5. Compute the shared total-vehicle-time objective.
6. Backpropagate through the entire JAX rollout and update the policy with
   AdamW.

The paper reports learning rate `1e-4`, weight decay `1e-6`, and batch size
256. Those values are used by the default configuration. It does not report
the number of optimizer updates. The configuration's 1,000-update default is a
documented repository choice.

## Objective

For each complete rollout `[x[0], ..., x[N]]`, the implementation uses

```text
J = dt * sum(k=1..N-1, ||x[k]||_1).
```

This is Equation (16a) with the time step included so reports have units of
vehicle-seconds. It is implemented once in
`src/nmfd_traffic/objective.py` and called by both the DPC loss and the MPPI
sample scorer. No quadratic state loss, control-magnitude penalty, or
control-rate penalty is added.

MPPI separately uses the covariance-aware importance-sampling correction
required by its 2017 formulation. That correction belongs to the sampling
estimator; it does not change the traffic objective.

## Dynamics, demand, and noise

The default experiment adopts the paper's stated values:

- seven regions and its illustrated adjacency;
- `dt=30` seconds and `N=T=240`;
- `a=4.133e-11`, `b=-8.282e-7`, and `c=0.0042`;
- controls in `[0.1, 0.9]`;
- zero initial vehicle accumulation;
- additive state/measurement noise with standard deviation 0.25;
- five OD demand flows: 0 to 6, 6 to 0, 4 to 2, 5 to 1, and 1 to 5.

The paper only plots the demand samples in Figure 4. It does not provide a
numeric table or public machine-readable artifact. The TOML profiles are
therefore piecewise-linear approximations read from that figure. This
limitation is included in configuration comments, checkpoint provenance, and
evaluation reports.

The configuration uses one RK4 integration step per 30-second control interval.
The paper states its control interval but does not specify an internal ODE
substep count, so this is another explicit repository discretization choice.

The paper's scenario-shift experiment adds normal noise to the demand vectors
and clips the resulting spawn rates at zero. The `noisy_demand` scenario
implements that behavior with standard deviation 1.0; the value is
configurable.

## Deliberate model choice

The paper trains with an NMFD model and evaluates model mismatch with a
separate acyclic NMFD plant. At the project owner's direction, this repository
does **not** implement that acyclic plant. DPC training, MPPI prediction, and
evaluation all use the same NMFD dynamics. Reports state
`acyclic_plant_model_used: false` so results cannot be confused with the
paper's model-mismatch experiment.

## Checkpoints and reproducibility

The former checked-in policy was incompatible with this architecture and
experiment and has been removed. Run `nmfd-train` to create a new checkpoint.
Every generated msgpack file has a JSON sidecar that records:

- an SHA-256 digest and byte size bound to the checkpoint payload;
- random seed and generation time;
- paper and objective identity;
- policy architecture and constraint map;
- NMFD, routing, demand, noise, and scenario settings;
- optimizer hyperparameters and completed update count;
- source configuration path/hash, software versions, and Git state.

Evaluation requires an explicit checkpoint path and includes both checkpoint
and sidecar hashes in its report.

## Verification

Focused tests check the exact hidden-layer shapes, the sigmoid midpoint and
bounds, L1 objective indexing/units, shared use by MPPI, demand interpolation,
demand-shift clipping, Gaussian state noise, full stochastic rollouts,
checkpoint provenance, seeded reproducibility, and JAX `jit`/`vmap`
compatibility.

[traffic-paper]: https://arxiv.org/abs/2406.10433
[dpc-paper]: https://doi.org/10.1109/TSMC.2024.3368026
