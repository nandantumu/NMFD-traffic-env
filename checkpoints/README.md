# Generated checkpoints

No pretrained checkpoint is committed. The previous artifact was trained for a
different policy, objective, scenario, and numerical configuration, so it is
not compatible with the paper-target perimeter-control implementation.

Create a checkpoint with:

```bash
uv run nmfd-train
```

This writes `dpc_policy.msgpack` and `dpc_policy.json` in this directory.
Both generated files are ignored by Git.

The msgpack file contains only the Flax inference parameter tree. Its JSON
sidecar records a SHA-256 digest and byte size for the payload, plus the seed,
paper, objective, architecture, environment, demand/noise scenario, optimizer,
source configuration, software versions, generation time, and Git state.

Evaluation requires an explicit checkpoint:

```bash
uv run nmfd-evaluate --checkpoint checkpoints/dpc_policy.msgpack
```

The evaluator hashes both the checkpoint and, when present, its provenance
sidecar into the metrics report.
