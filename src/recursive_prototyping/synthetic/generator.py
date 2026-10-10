"""Synthetic embedding sequences for demos and tests.

Every participant revisits a shared bank of action-cluster directions in contiguous runs,
interleaved with runs of background noise, so the data has change points and a shared
vocabulary to recover. Oracle labels and boundaries are returned alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recursive_prototyping.io.arrays import save_array
from recursive_prototyping.io.tables import save_table

# Run-length bounds (in windows) for action and background runs.
_MIN_RUN_LENGTH = 5
_MAX_RUN_LENGTH = 40


@dataclass
class SyntheticDatasetConfig:
    """Parameters of the synthetic dataset."""

    n_participants: int = 8
    n_action_clusters: int = 6
    embedding_dim: int = 1408
    min_windows: int = 200
    max_windows: int = 400
    cluster_std: float = 0.05
    background_noise_std: float = 0.3
    seed: int = 47


@dataclass
class ParticipantSynth:
    """One synthetic participant's embedding sequence with oracle ground truth."""

    participant_id: str
    embeddings: np.ndarray  # [n_windows, embedding_dim], unit-norm rows
    oracle_cluster_ids: np.ndarray  # [n_windows] int, -1 for background windows
    oracle_cpts: np.ndarray  # change-point boundaries [0, t1, ..., n_windows]


def _unit_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows of a 2D array to unit L2 norm."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms


def _random_unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Sample `n` vectors uniformly on the unit sphere in R^dim."""
    vectors = rng.normal(size=(n, dim))
    return _unit_normalize(vectors)


def _cpts_from_cluster_ids(oracle_cluster_ids: np.ndarray) -> np.ndarray:
    """Derive change-point boundaries from contiguous runs in `oracle_cluster_ids`."""
    n_windows = len(oracle_cluster_ids)
    change_indices = np.flatnonzero(
        oracle_cluster_ids[1:] != oracle_cluster_ids[:-1]
    ) + 1
    return np.concatenate(([0], change_indices, [n_windows])).astype(np.int64)


def _generate_participant_sequence(
    rng: np.random.Generator,
    cluster_bank: np.ndarray,
    n_windows: int,
    cluster_std: float,
    background_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one participant's embeddings [n_windows, dim] and cluster ids from
    alternating action and background runs."""
    n_action_clusters, embedding_dim = cluster_bank.shape
    embeddings = np.empty((n_windows, embedding_dim), dtype=np.float64)
    oracle_cluster_ids = np.empty(n_windows, dtype=np.int64)

    position = 0
    # Start with either an action run or a background run.
    is_background_turn = bool(rng.integers(0, 2))
    while position < n_windows:
        run_length = min(
            int(rng.integers(_MIN_RUN_LENGTH, _MAX_RUN_LENGTH + 1)),
            n_windows - position,
        )

        if is_background_turn:
            # A fresh random direction per run, not in the shared bank.
            base_direction = _random_unit_vectors(rng, 1, embedding_dim)
            noise = rng.normal(
                scale=background_noise_std, size=(run_length, embedding_dim)
            )
            run_embeddings = _unit_normalize(base_direction + noise)
            run_cluster_ids = np.full(run_length, -1, dtype=np.int64)
        else:
            cluster_idx = int(rng.integers(0, n_action_clusters))
            direction = cluster_bank[cluster_idx]
            noise = rng.normal(scale=cluster_std, size=(run_length, embedding_dim))
            run_embeddings = _unit_normalize(direction + noise)
            run_cluster_ids = np.full(run_length, cluster_idx, dtype=np.int64)

        embeddings[position : position + run_length] = run_embeddings
        oracle_cluster_ids[position : position + run_length] = run_cluster_ids
        position += run_length
        is_background_turn = not is_background_turn

    return embeddings, oracle_cluster_ids


def generate_synthetic_embeddings(
    config: SyntheticDatasetConfig,
) -> dict[str, ParticipantSynth]:
    """Generate every participant's sequence, keyed by participant id.

    The cluster bank is drawn once and shared; output is deterministic for `config.seed`.
    """
    rng = np.random.default_rng(config.seed)

    cluster_bank = _random_unit_vectors(
        rng, config.n_action_clusters, config.embedding_dim
    )

    participants: dict[str, ParticipantSynth] = {}
    for i in range(config.n_participants):
        participant_id = f"synth_{i:03d}"
        n_windows = int(rng.integers(config.min_windows, config.max_windows + 1))

        embeddings, oracle_cluster_ids = _generate_participant_sequence(
            rng,
            cluster_bank,
            n_windows,
            config.cluster_std,
            config.background_noise_std,
        )
        oracle_cpts = _cpts_from_cluster_ids(oracle_cluster_ids)

        participants[participant_id] = ParticipantSynth(
            participant_id=participant_id,
            embeddings=embeddings,
            oracle_cluster_ids=oracle_cluster_ids,
            oracle_cpts=oracle_cpts,
        )

    return participants


def write_synthetic_dataset(
    config: SyntheticDatasetConfig, output_dir
) -> pd.DataFrame:
    """Generate a synthetic dataset, write it to `output_dir`, and return its metadata.

    Writes one .npy of embeddings per participant and `synthetic_metadata.parquet`
    (participant_id, embedding_path, n_windows, oracle_cluster_ids, oracle_cpts).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = generate_synthetic_embeddings(config)

    rows = []
    for participant_id, participant in participants.items():
        embedding_path = output_dir / f"{participant_id}_embeddings.npy"
        save_array(embedding_path, participant.embeddings)

        rows.append(
            {
                "participant_id": participant_id,
                "embedding_path": str(embedding_path),
                "n_windows": len(participant.embeddings),
                "oracle_cluster_ids": participant.oracle_cluster_ids.tolist(),
                "oracle_cpts": participant.oracle_cpts.tolist(),
            }
        )

    metadata = pd.DataFrame(rows)
    save_table(output_dir / "synthetic_metadata.parquet", metadata)

    return metadata
