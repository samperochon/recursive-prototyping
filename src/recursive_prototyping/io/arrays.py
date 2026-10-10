"""Save/load numpy arrays as .npy (single array) or .npz (multiple named arrays)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_array(path: str | Path, array: np.ndarray) -> None:
    """Save a single array to a .npy file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_array(path: str | Path) -> np.ndarray:
    """Load a single array from a .npy file."""
    return np.load(Path(path))


def save_arrays(path: str | Path, **arrays: np.ndarray) -> None:
    """Save multiple named arrays to a single .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .npz file into a dict mapping array name to array."""
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}
