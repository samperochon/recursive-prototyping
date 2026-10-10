"""Unit tests for config/io.py: JSON snapshotting of config dataclasses."""

import json

import pytest

from recursive_prototyping.config.io import save_config_snapshot
from recursive_prototyping.config.schema import PrototypingConfig, SegmentationConfig


def test_json_snapshot_tuple_field_becomes_list(tmp_path):
    config = PrototypingConfig()
    snapshot_path = tmp_path / "snapshot.json"
    save_config_snapshot(config, snapshot_path)
    with snapshot_path.open() as f:
        snapshot = json.load(f)
    assert snapshot["annotation_categories"] == list(config.annotation_categories)


def test_save_config_snapshot_creates_parent_dirs(tmp_path):
    config = SegmentationConfig()
    snapshot_path = tmp_path / "nested" / "dir" / "snapshot.json"
    save_config_snapshot(config, snapshot_path)
    assert snapshot_path.exists()


def test_save_config_snapshot_rejects_a_non_dataclass(tmp_path):
    with pytest.raises(TypeError, match="dataclass"):
        save_config_snapshot({"a": 1}, tmp_path / "snapshot.json")
