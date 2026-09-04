from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flax import serialization


def save_parameters(
    path: str | Path, parameters: Any, metadata: dict | None = None
) -> None:
    """Save inference parameters as Flax msgpack, without optimizer state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialization.to_bytes(parameters))
    if metadata is not None:
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_parameters(path: str | Path, template: Any) -> Any:
    """Load parameters into a structure initialized from the same policy."""

    return serialization.from_bytes(template, Path(path).read_bytes())
