"""Unit tests for recursive_prototyping.prototyping.annotation_io."""

import numpy as np
import pandas as pd

from recursive_prototyping.prototyping.annotation_io import (
    QualificationRecord,
    load_qualification_mapping,
    merge_qualification_mapping,
    save_qualification_mapping,
)


def test_save_load_roundtrip_preserves_values_and_dtypes(output_dir):
    records = [
        QualificationRecord(cluster_id=0, round=1, cluster_type="task-definitive", annotator_id="alice"),
        QualificationRecord(cluster_id=1, round=1, cluster_type="noise", annotator_id="alice"),
        QualificationRecord(cluster_id=2, round=2, cluster_type="exo-ambiguous", annotator_id="bob"),
    ]
    path = output_dir / "qualification.parquet"

    save_qualification_mapping(records, path)
    loaded = load_qualification_mapping(path)

    expected = pd.DataFrame(
        {
            "cluster_id": pd.array([0, 1, 2], dtype="int64"),
            "round": pd.array([1, 1, 2], dtype="int64"),
            "cluster_type": pd.array(
                ["task-definitive", "noise", "exo-ambiguous"], dtype="object"
            ),
            "annotator_id": pd.array(["alice", "alice", "bob"], dtype="object"),
        }
    )

    assert loaded["cluster_id"].dtype == np.dtype("int64")
    assert loaded["round"].dtype == np.dtype("int64")
    pd.testing.assert_frame_equal(loaded, expected)


def test_merge_qualification_mapping_upsert_semantics():
    existing = pd.DataFrame(
        {
            "cluster_id": pd.array([0, 1, 2], dtype="int64"),
            "round": pd.array([1, 1, 2], dtype="int64"),
            "cluster_type": ["task-definitive", "noise", "exo-ambiguous"],
            "annotator_id": ["alice", "alice", "alice"],
        }
    )
    # (1, 1, "alice") overwrites the existing row's cluster_type.
    # (5, 2, "bob") is a brand-new key.
    new = pd.DataFrame(
        {
            "cluster_id": pd.array([1, 5], dtype="int64"),
            "round": pd.array([1, 2], dtype="int64"),
            "cluster_type": ["task-ambiguous", "exo-definitive"],
            "annotator_id": ["alice", "bob"],
        }
    )

    merged = merge_qualification_mapping(existing, new)

    assert len(merged) == 4

    overwritten = merged[
        (merged.cluster_id == 1) & (merged["round"] == 1) & (merged.annotator_id == "alice")
    ]
    assert len(overwritten) == 1
    assert overwritten.iloc[0]["cluster_type"] == "task-ambiguous"

    untouched_0 = merged[
        (merged.cluster_id == 0) & (merged["round"] == 1) & (merged.annotator_id == "alice")
    ]
    assert len(untouched_0) == 1
    assert untouched_0.iloc[0]["cluster_type"] == "task-definitive"

    untouched_2 = merged[
        (merged.cluster_id == 2) & (merged["round"] == 2) & (merged.annotator_id == "alice")
    ]
    assert len(untouched_2) == 1
    assert untouched_2.iloc[0]["cluster_type"] == "exo-ambiguous"

    new_row = merged[
        (merged.cluster_id == 5) & (merged["round"] == 2) & (merged.annotator_id == "bob")
    ]
    assert len(new_row) == 1
    assert new_row.iloc[0]["cluster_type"] == "exo-definitive"

    assert merged["cluster_id"].dtype == np.dtype("int64")
    assert merged["round"].dtype == np.dtype("int64")

    # inputs must not be mutated
    assert len(existing) == 3
    assert existing[existing.cluster_id == 1].iloc[0]["cluster_type"] == "noise"
    assert len(new) == 2


def test_full_roundtrip_save_load_merge_save(output_dir):
    path = output_dir / "qualification.parquet"
    initial = [
        QualificationRecord(cluster_id=0, round=1, cluster_type="task-definitive", annotator_id="alice"),
        QualificationRecord(cluster_id=1, round=1, cluster_type="noise", annotator_id="alice"),
    ]
    save_qualification_mapping(initial, path)
    loaded = load_qualification_mapping(path)

    updates = [
        QualificationRecord(cluster_id=1, round=1, cluster_type="task-ambiguous", annotator_id="alice"),
        QualificationRecord(cluster_id=2, round=2, cluster_type="exo-definitive", annotator_id="bob"),
    ]
    updates_path = output_dir / "updates.parquet"
    save_qualification_mapping(updates, updates_path)
    new_df = load_qualification_mapping(updates_path)

    merged = merge_qualification_mapping(loaded, new_df)

    merged_records = [QualificationRecord(**row) for row in merged.to_dict(orient="records")]
    save_qualification_mapping(merged_records, path)
    final = load_qualification_mapping(path)

    assert len(final) == 3
    assert final["cluster_id"].dtype == np.dtype("int64")
    assert final["round"].dtype == np.dtype("int64")

    overwritten = final[(final.cluster_id == 1) & (final["round"] == 1)]
    assert overwritten.iloc[0]["cluster_type"] == "task-ambiguous"

    new_row = final[(final.cluster_id == 2) & (final["round"] == 2)]
    assert new_row.iloc[0]["cluster_type"] == "exo-definitive"
    assert new_row.iloc[0]["annotator_id"] == "bob"
