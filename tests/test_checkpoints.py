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
    assert json.loads(output.with_suffix(".json").read_text()) == {
        "seed": 4,
        "source": "test",
    }


def test_packaged_assets_match_the_repository_copies():
    package = resources.files("nmfd_traffic").joinpath("data")
    pairs = [
        (package.joinpath("seven_region.toml"), ROOT / "configs/seven_region.toml"),
        (
            package.joinpath("dpc_policy.msgpack"),
            ROOT / "checkpoints/dpc_policy.msgpack",
        ),
        (
            package.joinpath("dpc_policy.provenance.json"),
            ROOT / "checkpoints/dpc_policy.provenance.json",
        ),
    ]

    for packaged, repository in pairs:
        with resources.as_file(packaged) as packaged_path:
            assert _sha256(packaged_path) == _sha256(repository)


def test_checked_in_checkpoint_matches_its_provenance_record():
    provenance_path = ROOT / "checkpoints/dpc_policy.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / provenance["artifact"]["path"]

    assert checkpoint.stat().st_size == provenance["artifact"]["size_bytes"]
    assert _sha256(checkpoint) == provenance["artifact"]["sha256"]
