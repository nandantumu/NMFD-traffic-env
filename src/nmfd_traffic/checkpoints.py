from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flax import serialization


def save_parameters(
    path: str | Path, parameters: Any, metadata: dict | None = None
) -> None:
    """Save inference parameters and a hash-bound provenance sidecar."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes(parameters)
    output.write_bytes(payload)
    if metadata is not None:
        provenance = dict(metadata)
        provenance["artifact"] = {
            "format": "Flax serialization.to_bytes parameter tree",
            "path": output.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        output.with_suffix(".json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_parameters(path: str | Path, template: Any) -> Any:
    """Load parameters into a structure initialized from the same policy."""

    return serialization.from_bytes(template, Path(path).read_bytes())
