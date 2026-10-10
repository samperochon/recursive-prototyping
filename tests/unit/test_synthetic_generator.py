"""Unit tests for recursive_prototyping.synthetic.generator."""

import numpy as np

from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
    write_synthetic_dataset,
)


def _small_config(**overrides) -> SyntheticDatasetConfig:
    defaults = dict(
        n_participants=3,
        n_action_clusters=4,
        embedding_dim=32,
        min_windows=50,
        max_windows=80,
        cluster_std=0.05,
        background_noise_std=0.3,
        seed=47,
    )
    defaults.update(overrides)
    return SyntheticDatasetConfig(**defaults)


def _independent_cpts(oracle_cluster_ids: np.ndarray) -> np.ndarray:
    """Recompute change-point boundaries independently of the generator's logic."""
    n = len(oracle_cluster_ids)
    boundaries = [0]
    for i in range(1, n):
        if oracle_cluster_ids[i] != oracle_cluster_ids[i - 1]:
            boundaries.append(i)
    boundaries.append(n)
    return np.array(boundaries, dtype=np.int64)


def test_determinism_same_seed_bit_identical():
    config = _small_config(seed=47)
    result_a = generate_synthetic_embeddings(config)
    result_b = generate_synthetic_embeddings(config)

    assert result_a.keys() == result_b.keys()
    for participant_id in result_a:
        np.testing.assert_array_equal(
            result_a[participant_id].embeddings, result_b[participant_id].embeddings
        )
        np.testing.assert_array_equal(
            result_a[participant_id].oracle_cluster_ids,
            result_b[participant_id].oracle_cluster_ids,
        )


def test_determinism_different_seed_differs():
    result_a = generate_synthetic_embeddings(_small_config(seed=47))
    result_b = generate_synthetic_embeddings(_small_config(seed=48))

    any_participant = next(iter(result_a))
    # Different seeds should not produce identical embeddings.
    assert not np.array_equal(
        result_a[any_participant].embeddings, result_b[any_participant].embeddings
    )


def test_embeddings_are_unit_norm():
    config = _small_config()
    participants = generate_synthetic_embeddings(config)

    for participant in participants.values():
        norms = np.linalg.norm(participant.embeddings, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)


def test_oracle_cluster_ids_in_valid_range():
    config = _small_config()
    participants = generate_synthetic_embeddings(config)

    for participant in participants.values():
        ids = participant.oracle_cluster_ids
        assert np.all((ids == -1) | ((ids >= 0) & (ids < config.n_action_clusters)))


def test_oracle_cpts_match_independent_recomputation():
    config = _small_config()
    participants = generate_synthetic_embeddings(config)

    for participant in participants.values():
        n_windows = len(participant.oracle_cluster_ids)
        cpts = participant.oracle_cpts

        # Strictly increasing, starts at 0, ends at n_windows.
        assert cpts[0] == 0
        assert cpts[-1] == n_windows
        assert np.all(np.diff(cpts) > 0)

        expected = _independent_cpts(participant.oracle_cluster_ids)
        np.testing.assert_array_equal(cpts, expected)


def test_shapes_consistent_with_config():
    config = _small_config()
    participants = generate_synthetic_embeddings(config)

    assert len(participants) == config.n_participants
    for i in range(config.n_participants):
        assert f"synth_{i:03d}" in participants

    for participant in participants.values():
        n_windows = len(participant.embeddings)
        assert config.min_windows <= n_windows <= config.max_windows
        assert participant.embeddings.shape == (n_windows, config.embedding_dim)
        assert participant.oracle_cluster_ids.shape == (n_windows,)


def test_write_synthetic_dataset(output_dir):
    config = _small_config()
    metadata = write_synthetic_dataset(config, output_dir)

    assert len(metadata) == config.n_participants
    assert set(metadata.columns) >= {
        "participant_id",
        "embedding_path",
        "n_windows",
        "oracle_cluster_ids",
        "oracle_cpts",
    }

    for _, row in metadata.iterrows():
        embedding_path = row["embedding_path"]
        loaded = np.load(embedding_path)
        assert loaded.shape == (row["n_windows"], config.embedding_dim)
        assert len(row["oracle_cluster_ids"]) == row["n_windows"]
        assert row["oracle_cpts"][0] == 0
        assert row["oracle_cpts"][-1] == row["n_windows"]

    metadata_files = list(output_dir.glob("*.parquet"))
    assert len(metadata_files) == 1
