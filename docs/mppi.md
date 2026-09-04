# MPPI implementation

## Status and decision

The controller in `src/nmfd_traffic/mppi.py` implements the selected 2017
information-theoretic MPPI formulation. The former reward-to-go baseline remains
available only through repository history; `rollout_naive_mppi` is retained as
a backward-compatible name for `rollout_mppi`.

The implementation targets the formulation in:

> G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou,
> “Information-Theoretic Model Predictive Control: Theory and Applications to
> Autonomous Driving,” *IEEE Transactions on Robotics*, vol. 34, no. 6,
> pp. 1603–1622, 2018. DOI: [10.1109/TRO.2018.2865891][2017-doi].
> [Author manuscript][2017-paper].

Algorithms 1 and 2 in that paper are the normative implementation reference.
The shorter [ACDSLab MPPI overview][acds-overview] is useful introductory
material, but it omits some importance-sampling details from the paper.

This decision supersedes using the 2015 covariance-variable derivation as the
implementation specification. That earlier paper is still important historical
context: its Equation 56 uses a time-indexed cost-to-go, so reverse cumulative
returns are not inherently incompatible with every formulation called MPPI.
The 2017 target instead assigns one cost and one weight to each complete sampled
control trajectory.

## Implemented formulation

The paper considers a general discrete-time system

```text
x[t + 1] = F(x[t], v[t])
v[t] ~ Normal(u[t], Sigma),
```

where `u[t]` is the nominal command and `v[t]` is a sampled input. This general
discrete formulation is a better fit for the NMFD environment than the 2015
paper's initial continuous-time, control-affine stochastic-diffusion model.

For each controller update, the implementation performs the following
operations.

1. Sample complete perturbation sequences

   ```text
   epsilon[k, t] ~ Normal(0, Sigma).
   ```

   Following Algorithm 1, most trajectories are sampled around the current
   nominal sequence, `v[k, t] = u[t] + epsilon[k, t]`. A configurable `alpha`
   fraction is sampled without the nominal sequence,
   `v[k, t] = epsilon[k, t]`, to provide recovery/exploration trajectories.

2. Apply constraints inside the rollout dynamics

   ```text
   x[k, t + 1] = F(x[k, t], g(v[k, t])),
   ```

   where `g` clamps an input to the admissible NMFD control interval. The raw
   Gaussian `epsilon[k, t]` is retained for the later update. Clipping the
   perturbation itself would truncate the sampling distribution and bias the
   update at a control bound.

3. Compute one corrected score for each complete trajectory

   ```text
   J[k] = S(V[k]; x[0])
          + gamma * sum_t u[t]^T inverse(Sigma) v[k, t],
   S(V; x[0]) = phi(x[T]) + sum_t c(x[t]).
   ```

   The second term is the importance-sampling/control correction from the
   paper. A plain quadratic penalty on the sampled input is not a substitute
   for this term. For the paper's parameterization,
   `gamma = lambda * (1 - alpha)`.

4. Convert the trajectory scores to stable importance weights

   ```text
   rho = min_k J[k]
   w[k] = exp(-(J[k] - rho) / lambda)
          / sum_j exp(-(J[j] - rho) / lambda).
   ```

   Subtracting `rho` is a numerical-stability shift and does not change the
   normalized weights. The scores are **not** divided by their sampled
   minimum-to-maximum range: doing so changes the effective inverse temperature
   from one controller update to the next.

5. Update the entire nominal sequence using the same trajectory weights

   ```text
   delta_u[t] = sum_k w[k] * epsilon[k, t]
   U = U + smooth(delta_u).
   ```

   Algorithm 1 uses a Savitzky–Golay filter for `smooth`. The JAX
   implementation provides the paper's smoothing operation without adding
   SciPy as a runtime dependency. An identity filter may be useful as an
   explicitly selected ablation, but it must not silently replace the
   paper-faithful default.

6. Execute the first nominal action, shift the unused sequence one step, append
   a documented initialization value, and repeat from the measured state.

All operations remain JAX-only and functional so the controller can be
transformed with `jax.jit` and `jax.vmap`, in addition to its native batch
dimension.

## NMFD objective mapping

The paper distinguishes the state-dependent trajectory cost from the
importance-sampling control correction. For direct comparison with DPC, this
environment uses the traffic objective from Equation (16a) of Tumu et al.:

```text
S(V; x[0]) = dt * sum(t=1..T-1, ||x[t]||_1).
```

The shared function in `src/nmfd_traffic/objective.py` computes this value in
vehicle-seconds for both controllers. There are no additional state, control,
or control-rate weights, and the generic MPPI terminal term `phi(x[T])` is
zero. MPPI still adds the 2017 covariance-aware
importance-sampling term when calculating sample weights; that term is part of
the MPPI estimator rather than an NMFD performance objective.

The prediction rollout receives the known time-varying demand forecast. The
executed plant receives the same demand and state-noise realization as DPC and
the open-gates baseline. MPPI plans against the expected state dynamics rather
than future noise samples.

The default MPPI planning horizon is eight control intervals, matching the
online MPC horizon reported in the traffic paper while the full evaluation and
DPC training rollout remain 240 steps. Horizon length is application-specific
and is not prescribed by the 2017 MPPI algorithm.

## Legacy-to-current comparison

The implementation pass replaced the following legacy behaviors.

| Area | Former baseline | Implemented 2017 formulation |
| --- | --- | --- |
| Rollout score | Reverse cumulative return at every horizon index | One corrected score per complete trajectory |
| Weight shape | One weight per sample and horizon index | One weight per sample |
| Temperature | Cost gap divided by sampled cost range and `temperature` | Corrected cost divided by fixed `lambda` |
| Control term | Quadratic cost of the clipped sampled control | Covariance-aware importance-sampling correction |
| Traffic objective | Quadratic accumulation cost | Shared L1 total vehicle time from the traffic DPC paper |
| Covariance | Scalar independent standard deviation | Positive-definite `Sigma`; scalar diagonal is a supported special case |
| Constraints | Perturbation clipped before both rollout and update | Sampled input is clamped for rollout evaluation; raw perturbation updates the mean |
| Recovery samples | Every sample is centered on the nominal sequence | Configurable `alpha` fraction is zero-centered |
| Smoothing | None | Savitzky–Golay update smoothing |
| Receding horizon | Execute, shift, append, repeat | Same structure |

The most consequential former error was the sampled-range normalization. If
`Delta J` is the sampled cost range, that exponent behaved as though `lambda`
were `temperature * (Delta J + damping)`. The meaning of temperature therefore
changed with the random samples, batch member, and horizon position.

## Verification criteria

The focused MPPI tests demonstrate the following:

- a hand-computed set of corrected trajectory scores produces the expected
  fixed-temperature weights;
- adding a common constant to all trajectory scores leaves weights unchanged,
  while multiplying their range changes concentration as fixed `lambda`
  requires;
- each sampled trajectory has one weight shared across every horizon index;
- the score includes the shared L1 traffic objective and covariance-aware
  importance-sampling correction;
- raw perturbations, rather than clipped perturbations, drive the nominal
  sequence update;
- control clamping affects simulated dynamics and keeps executed controls
  admissible;
- nominal-centered and zero-centered sample branches follow `alpha`;
- smoothing, execution, sequence shifting, and terminal initialization match
  Algorithm 1;
- known demand is propagated through every sampled trajectory;
- seeded execution is reproducible, and the rollout remains compatible with
  `jax.jit`, `jax.vmap`, and native batched execution.

Evaluation output records the paper title, DOI, covariance convention,
`temperature` (the paper's `lambda`), `alpha`, derived `gamma`, smoothing
parameters, and the shared traffic objective.

## References

- G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou,
  “Information-Theoretic Model Predictive Control: Theory and Applications to
  Autonomous Driving,” *IEEE Transactions on Robotics*, 2018.
  [DOI][2017-doi] · [author manuscript][2017-paper]
- G. Williams, A. Aldrich, and E. A. Theodorou, “Model Predictive Path Integral
  Control using Covariance Variable Importance Sampling,” 2015.
  [DOI][2015-doi] · [author manuscript][2015-paper]
- R. Tumu, W. Shaw Cortez, J. Drgoňa, D. L. Vrabie, and S. Glavaski,
  “Differentiable Predictive Control for Large-Scale Urban Road Networks,”
  2024. [author manuscript][traffic-paper]

[2017-doi]: https://doi.org/10.1109/TRO.2018.2865891
[2017-paper]: https://arxiv.org/abs/1707.02342
[2015-doi]: https://doi.org/10.48550/arXiv.1509.01149
[2015-paper]: https://arxiv.org/abs/1509.01149
[acds-overview]: https://acdslab.github.io/mppi-generic-website/docs/mppi.html
[traffic-paper]: https://arxiv.org/abs/2406.10433
