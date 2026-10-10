"""Save/load pandas DataFrames as parquet tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(path: str | Path, df: pd.DataFrame) -> None:
    """Save a DataFrame to a parquet file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a parquet file."""
    return pd.read_parquet(Path(path), engine="pyarrow")
