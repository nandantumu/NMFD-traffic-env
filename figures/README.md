# Reference figures

These artifacts were generated from commit-ready source with fixed seed `0`,
the checked-in seven-region configuration, 10 matched rollouts per scenario,
and the full configured naive MPPI setting of 128 samples and one update
iteration.

They are intended as quick visual examples, not high-confidence benchmark
estimates. Regenerate more stable statistics with the CLI default of 100
rollouts:

```bash
uv run nmfd-evaluate --scenario in_distribution
uv run nmfd-evaluate --scenario out_of_distribution
```

Runtime values in the JSON files are machine-specific steady-state JAX timings
after compilation. The trajectory metrics and uncertainty bands use the same
10 sampled initial conditions for all controllers within each scenario.
