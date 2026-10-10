"""Read, write and merge the qualification mapping: one annotator's label per cluster per
round, stored as parquet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from recursive_prototyping.io.tables import load_table, save_table

_COLUMNS = ["cluster_id", "round", "cluster_type", "annotator_id"]
_INT_COLUMNS = ["cluster_id", "round"]


@dataclass
class QualificationRecord:
    """One annotation: a cluster of a given round, labelled by one annotator.

    ``cluster_type`` is one of ``PrototypingConfig.annotation_categories`` (not checked here).
    """

    cluster_id: int
    round: int
    cluster_type: str
    annotator_id: str


def save_qualification_mapping(records: list[QualificationRecord], path: str | Path) -> None:
    """Write ``records`` as parquet."""
    df = pd.DataFrame([asdict(record) for record in records], columns=_COLUMNS)
    for col in _INT_COLUMNS:
        df[col] = df[col].astype("int64")
    save_table(path, df)


def load_qualification_mapping(path: str | Path) -> pd.DataFrame:
    """Load a saved mapping: columns [cluster_id, round, cluster_type, annotator_id]."""
    return load_table(path)


def merge_qualification_mapping(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert ``new`` into ``existing``, keyed on (cluster_id, round, annotator_id).

    Returns a new DataFrame; neither input is mutated.
    """
    key_cols = ["cluster_id", "round", "annotator_id"]

    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        new_keys = set(new[key_cols].itertuples(index=False, name=None))
        existing_keys = existing[key_cols].apply(tuple, axis=1)
        merged = pd.concat(
            [existing.loc[~existing_keys.isin(new_keys)], new], ignore_index=True
        )

    merged = merged.reset_index(drop=True)
    # concat can upcast the int columns to float; cast them back.
    for col in _INT_COLUMNS:
        merged[col] = merged[col].astype("int64")
    return merged
