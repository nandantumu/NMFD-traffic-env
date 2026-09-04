"""Convert a trusted L2O_MPPI pickle checkpoint to inference-only msgpack."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

from flax import serialization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted nmfd_policy_jax.pkl")
    parser.add_argument("output", type=Path, help="destination msgpack path")
    args = parser.parse_args()

    # Pickle can execute code while loading. This converter is intentionally
    # restricted to a checkpoint obtained from the trusted provenance source.
    with args.source.open("rb") as file:
        payload = pickle.load(file)
    if "params" not in payload:
        raise ValueError("source checkpoint does not contain a 'params' tree")

    converted = serialization.to_bytes(payload["params"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converted)
    digest = hashlib.sha256(converted).hexdigest()
    print(f"wrote {args.output} ({len(converted)} bytes, sha256={digest})")


if __name__ == "__main__":
    main()
