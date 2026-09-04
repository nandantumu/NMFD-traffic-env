# Reference topology

`network_topology.png` shows the seven-region adjacency used by the
paper-target configuration. The previous controller-comparison plots were
removed because they were produced with an incompatible legacy policy,
objective, and scenario.

After training a new checkpoint, generate current comparison artifacts with:

```bash
uv run nmfd-evaluate \
  --checkpoint checkpoints/dpc_policy.msgpack \
  --scenario nominal \
  --output-dir results
```

Generated reports contain their seeds, software/Git state, configuration and
checkpoint hashes, objective, controller settings, demand/noise scenario, and
runtime measurements.
