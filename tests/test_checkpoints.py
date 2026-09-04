import hashlib
import json
from importlib import resources
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nmfd_traffic import load_parameters, save_parameters

ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_round_trip_and_metadata(tmp_path):
    parameters = {
        "dense": {
            "kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            "bias": jnp.ones((3,), dtype=jnp.float32),
        }
    }
    output = tmp_path / "parameters.msgpack"

    save_parameters(output, parameters, metadata={"seed": 4, "source": "test"})
    restored = load_parameters(output, parameters)

    np.testing.assert_array_equal(
        restored["dense"]["kernel"],
        parameters["dense"]["kernel"],
    )
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["seed"] == 4
    assert metadata["source"] == "test"
    assert metadata["artifact"] == {
        "format": "Flax serialization.to_bytes parameter tree",
        "path": "parameters.msgpack",
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
    }


def test_packaged_configuration_matches_the_repository_copy():
    package = resources.files("nmfd_traffic").joinpath("data")
    packaged = package.joinpath("seven_region.toml")
    repository = ROOT / "configs/seven_region.toml"

    with resources.as_file(packaged) as packaged_path:
        assert _sha256(packaged_path) == _sha256(repository)
