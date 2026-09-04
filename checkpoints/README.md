# Checkpoint

`dpc_policy.msgpack` contains only the Flax parameter tree for the deterministic
seven-region DPC policy. It was converted from
`L2O_MPPI/ss_learning/nmfd_policy_jax.pkl`; optimizer state and the original
pickle wrapper were intentionally discarded.

The matching architecture and physical parameters are defined in
`configs/seven_region.toml`.
