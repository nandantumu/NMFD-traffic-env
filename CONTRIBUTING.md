# Contributing

This project favors explicit numerical code, documented tensor shapes, and
small pure functions that remain compatible with `jax.jit` and `jax.vmap`.
Runtime code must remain JAX-only; do not add PyTorch or Gym dependencies.

## Development environment

Install every locked dependency group with uv:

```bash
uv sync --locked --all-groups
```

Run the same checks used by continuous integration:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=nmfd_traffic --cov-report=term-missing
uv build --wheel
```

To format a change locally, run `uv run ruff format .`. Optional Git hooks can
be installed with `uvx pre-commit install`; they use the Ruff version locked by
this project.

## Tests and documentation

Tests should state the physical or API invariant they protect, not only confirm
that code executes. Document array axes and expected shapes whenever a public
function accepts or returns a JAX array. If behavior is based on a paper, cite
the exact equation or algorithm section in the implementation and its tests.
