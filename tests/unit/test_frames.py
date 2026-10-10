"""Unit tests for recursive_prototyping.synthetic.frames."""

import numpy as np

from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    ParticipantSynth,
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

_IMAGE_SIZE = 32


def _hand_built_participants() -> dict[str, ParticipantSynth]:
    """Two small fake participants with known oracle_cluster_ids, including background (-1)."""
    p0 = ParticipantSynth(
        participant_id="p0",
        embeddings=np.zeros((5, 4)),
        oracle_cluster_ids=np.array([0, 0, 1, -1, 2]),
        oracle_cpts=np.array([0, 2, 3, 4, 5]),
    )
    p1 = ParticipantSynth(
        participant_id="p1",
        embeddings=np.zeros((3, 4)),
        oracle_cluster_ids=np.array([-1, 1, 3]),
        oracle_cpts=np.array([0, 1, 3]),
    )
    return {"p0": p0, "p1": p1}


def test_central_frame_shape_and_dtype():
    participants = _hand_built_participants()
    source = SyntheticFrameSource(participants, image_size=_IMAGE_SIZE)

    frame = source.central_frame("p0", 0)

    assert frame.shape == (_IMAGE_SIZE, _IMAGE_SIZE, 3)
    assert frame.dtype == np.uint8


def test_determinism_same_call_bit_identical():
    participants = _hand_built_participants()
    source = SyntheticFrameSource(participants, image_size=_IMAGE_SIZE)

    frame_a = source.central_frame("p0", 3)  # background window
    frame_b = source.central_frame("p0", 3)

    np.testing.assert_array_equal(frame_a, frame_b)


def test_different_oracle_clusters_produce_different_frames():
    participants = _hand_built_participants()
    source = SyntheticFrameSource(participants, image_size=_IMAGE_SIZE)

    frame_cluster_0 = source.central_frame("p0", 0)  # oracle_cluster_id == 0
    frame_cluster_1 = source.central_frame("p0", 2)  # oracle_cluster_id == 1

    assert not np.array_equal(frame_cluster_0, frame_cluster_1)


def test_background_frame_distinguishable_from_every_cluster_color():
    participants = _hand_built_participants()
    source = SyntheticFrameSource(participants, image_size=_IMAGE_SIZE)

    background_frame = source.central_frame("p0", 3)  # oracle_cluster_id == -1

    for participant_id, participant in participants.items():
        for embedding_index, cluster_id in enumerate(participant.oracle_cluster_ids):
            if cluster_id == -1:
                continue
            cluster_frame = source.central_frame(participant_id, embedding_index)
            assert not np.array_equal(background_frame, cluster_frame)


def test_integration_with_generated_synthetic_embeddings():
    """Extra confidence: exercise central_frame against the real generator output."""
    config = SyntheticDatasetConfig(
        n_participants=2,
        n_action_clusters=3,
        embedding_dim=8,
        min_windows=20,
        max_windows=30,
        cluster_std=0.05,
        background_noise_std=0.3,
        seed=47,
    )
    participants = generate_synthetic_embeddings(config)
    source = SyntheticFrameSource(participants, image_size=16)

    any_participant_id = next(iter(participants))
    frame = source.central_frame(any_participant_id, 0)

    assert frame.shape == (16, 16, 3)
    assert frame.dtype == np.uint8
