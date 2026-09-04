# Project instructions

- Use `uv` for environment and dependency management.
- Keep the runtime JAX-only; do not add PyTorch or Gym as a dependency.
- Preserve the functional environment API so dynamics can be transformed with `jax.jit` and `jax.vmap`.
