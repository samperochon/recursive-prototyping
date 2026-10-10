"""Unit tests for recursive_prototyping.io: array/table round-trips."""

import numpy as np
import pandas as pd
import pytest

from recursive_prototyping.io.arrays import (
    load_array,
    load_arrays,
    save_array,
    save_arrays,
)
from recursive_prototyping.io.tables import load_table, save_table

DTYPES_AND_SHAPES = [
    (np.float32, (10,)),
    (np.float64, (10,)),
    (np.int64, (10,)),
    (np.float32, (4, 3)),
    (np.int64, (5, 2)),
]


@pytest.mark.parametrize("dtype,shape", DTYPES_AND_SHAPES)
def test_save_load_array_roundtrip(output_dir, dtype, shape):
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    path = output_dir / "nested" / "array.npy"

    save_array(path, array)
    loaded = load_array(path)

    assert loaded.dtype == array.dtype
    assert loaded.shape == array.shape
    np.testing.assert_array_equal(loaded, array)


def test_save_load_array_accepts_str_path(output_dir):
    array = np.array([1, 2, 3], dtype=np.int64)
    path = str(output_dir / "array_str.npy")

    save_array(path, array)
    loaded = load_array(path)

    np.testing.assert_array_equal(loaded, array)


def test_save_load_arrays_roundtrip(output_dir):
    arrays = {
        "embeddings": np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32),
        "labels": np.array([0, 1, 2, 1, 0], dtype=np.int64),
        "scores": np.array([0.1, 0.2, 0.3], dtype=np.float64),
    }
    path = output_dir / "nested" / "bundle.npz"

    save_arrays(path, **arrays)
    loaded = load_arrays(path)

    assert set(loaded.keys()) == set(arrays.keys())
    for name, original in arrays.items():
        assert loaded[name].dtype == original.dtype
        assert loaded[name].shape == original.shape
        np.testing.assert_array_equal(loaded[name], original)


def test_save_load_table_roundtrip(output_dir):
    df = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype="int64"),
            "score": pd.array([0.5, 1.5, -2.25], dtype="float64"),
            "label": pd.array(["task-definitive", "exo-ambiguous", "noise"], dtype="object"),
            "accepted": pd.array([True, False, True], dtype="bool"),
        }
    )
    path = output_dir / "nested" / "table.parquet"

    save_table(path, df)
    loaded = load_table(path)

    pd.testing.assert_frame_equal(loaded, df)
