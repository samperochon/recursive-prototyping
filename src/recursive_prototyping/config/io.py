"""JSON snapshots of config dataclasses, for run provenance."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def save_config_snapshot(config: Any, path: str | Path) -> None:
    """Dump a config dataclass instance to JSON (tuple fields become lists)."""
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"config must be a dataclass instance, got {config!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, sort_keys=True)
        f.write("\n")
