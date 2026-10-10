"""Shared pytest fixtures for recursive_prototyping tests."""

import pytest


@pytest.fixture
def output_dir(tmp_path):
    """A scratch directory for a test to write outputs into."""
    path = tmp_path / "outputs"
    path.mkdir()
    return path
