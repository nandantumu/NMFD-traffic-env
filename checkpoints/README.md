# Checkpoint

`dpc_policy.msgpack` contains only the Flax parameter tree for the deterministic
seven-region DPC policy. It was converted from
`L2O_MPPI/ss_learning/nmfd_policy_jax.pkl`; optimizer state and the original
pickle wrapper were intentionally discarded.

The matching architecture and physical parameters are defined in
`configs/seven_region.toml`.

## Provenance

- Source repository: `https://github.com/vietanhle0101/L2O_MPPI`
- Source commit: `9fdd6de61225c37968363d5a7586649e2ec02652`
- Source path: `ss_learning/nmfd_policy_jax.pkl`
- Source SHA-256:
  `86e067a589c8fd19d60160e4aca4813bb16ead8cab211c0c33360f62245b68fe`
- Source training state: epoch 100, global step 20,000
- Training entry point: `ss_learning/train_nmfd_jax.py`
- Imported into this repository in commit:
  `9de791a0f02ac4e7ab71404890ac71c933d8c17f`
- Artifact SHA-256:
  `8b0737b167edcb582784de894c3be5856a2826a0012637266264fb879cdfe47c`

The conversion extracts `payload["params"]` from the trusted source pickle and
writes `flax.serialization.to_bytes(payload["params"])`. Reproduce it with:

```bash
uv run python scripts/convert_l2o_checkpoint.py \
  /path/to/L2O_MPPI/ss_learning/nmfd_policy_jax.pkl \
  checkpoints/dpc_policy.msgpack
```

Never run the converter on an untrusted pickle: Python pickle loading can
execute arbitrary code. The machine-readable record is
`dpc_policy.provenance.json`.
